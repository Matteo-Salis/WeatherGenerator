# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.datasets.utils import (
    healpix_verts_rots,
    r3tos2,
)


class Tokenizer:
    """
    Base class for tokenizers.
    """

    def __init__(self, hl_source: int, hl_target: int = None, grid_source=None, grid_target=None):
        ref = torch.tensor([1.0, 0.0, 0.0])

        # Sources are tokenized on the encoder's own grid (hl_source / grid_source); targets on the
        # forecast grid (hl_target / grid_target). When the target level/grid is omitted it mirrors
        # the source, recovering the single-grid behaviour. Per-cell geometry is built full at each
        # level then restricted to its active region.
        self.hl_source = hl_source
        self.hl_target = hl_target if hl_target is not None else hl_source
        self.grid_source = grid_source
        self.grid_target = grid_target if grid_target is not None else grid_source
        # source level/grid drive source tokenization in get_tokens_windows
        self.healpix_level = self.hl_source
        self.grid = self.grid_source

        self.num_healpix_cells_source = (
            self.grid_source.num_cells if self.grid_source is not None else 12 * 4**self.hl_source
        )
        self.num_healpix_cells_target = (
            self.grid_target.num_cells if self.grid_target is not None else 12 * 4**self.hl_target
        )

        self.size_time_embedding = 6

        verts00_s, verts00_rots_s = healpix_verts_rots(self.hl_source, 0.0, 0.0)
        verts10_s, verts10_rots_s = healpix_verts_rots(self.hl_source, 1.0, 0.0)
        verts11_s, verts11_rots_s = healpix_verts_rots(self.hl_source, 1.0, 1.0)
        verts01_s, verts01_rots_s = healpix_verts_rots(self.hl_source, 0.0, 1.0)
        vertsmm_s, vertsmm_rots_s = healpix_verts_rots(self.hl_source, 0.5, 0.5)
        self.hpy_verts = [
            verts00_s.to(torch.float32),
            verts10_s.to(torch.float32),
            verts11_s.to(torch.float32),
            verts01_s.to(torch.float32),
            vertsmm_s.to(torch.float32),
        ]
        self.hpy_verts_rots_source = [
            verts00_rots_s.to(torch.float32),
            verts10_rots_s.to(torch.float32),
            verts11_rots_s.to(torch.float32),
            verts01_rots_s.to(torch.float32),
            vertsmm_rots_s.to(torch.float32),
        ]

        verts00, verts00_rots = healpix_verts_rots(self.hl_target, 0.0, 0.0)
        verts10, verts10_rots = healpix_verts_rots(self.hl_target, 1.0, 0.0)
        verts11, verts11_rots = healpix_verts_rots(self.hl_target, 1.0, 1.0)
        verts01, verts01_rots = healpix_verts_rots(self.hl_target, 0.0, 1.0)
        vertsmm, vertsmm_rots = healpix_verts_rots(self.hl_target, 0.5, 0.5)
        self.hpy_verts_rots_target = [
            verts00_rots.to(torch.float32),
            verts10_rots.to(torch.float32),
            verts11_rots.to(torch.float32),
            verts01_rots.to(torch.float32),
            vertsmm_rots.to(torch.float32),
        ]

        transforms = [
            ([verts10, verts11, verts01, vertsmm], verts00_rots),
            ([verts00, verts11, verts01, vertsmm], verts10_rots),
            ([verts00, verts10, verts01, vertsmm], verts11_rots),
            ([verts00, verts11, verts10, vertsmm], verts01_rots),
            ([verts00, verts10, verts11, verts01], vertsmm_rots),
        ]

        self.verts_local = []
        for _verts, rot in transforms:
            # Compute local coordinates
            verts = torch.stack(_verts)
            # shape: <healpix, 4, 3>
            verts = verts.transpose(0, 1)
            # Batch multiplication by the 3x3 rotation matrices.
            # shape: <healpix, 3, 3> @ <healpix, 4, 3> -> <healpix, 4, 3>
            # Needs to transpose first to <healpix, 3, 4> then transpose back.
            t1 = torch.bmm(rot, verts.transpose(-1, -2)).transpose(-2, -1)
            t2 = ref - t1
            self.verts_local.append(t2.flatten(1, 2))

        self.hpy_verts_local_target = torch.stack(self.verts_local).transpose(0, 1)

        # add local coords wrt to center of neighboring cells
        # (since the neighbors are used in the prediction)
        num_healpix_cells = 12 * 4**self.hl_target
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(num_healpix_cells), 2**self.hl_target, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hpy_nctrs_target = (
            vertsmm[temp.flatten()]
            .reshape((num_healpix_cells, 8, 3))
            .transpose(1, 0)
            .to(torch.float32)
        )

        if self.grid_source is not None and not self.grid_source.is_full:
            a2g = torch.from_numpy(self.grid_source.active_to_global).to(torch.long)
            self.hpy_verts = [v[a2g] for v in self.hpy_verts]
            self.hpy_verts_rots_source = [v[a2g] for v in self.hpy_verts_rots_source]
        if self.grid_target is not None and not self.grid_target.is_full:
            a2g = torch.from_numpy(self.grid_target.active_to_global).to(torch.long)
            self.hpy_verts_rots_target = [v[a2g] for v in self.hpy_verts_rots_target]
            self.hpy_verts_local_target = self.hpy_verts_local_target[a2g]
            self.hpy_nctrs_target = self.hpy_nctrs_target[:, a2g, :]

    def compute_source_centroids(self, source_tokens_cells: list[torch.Tensor]) -> torch.Tensor:
        source_means = [
            (
                self.hpy_verts[-1][i].unsqueeze(0).repeat(len(s), 1)
                if len(s) > 0
                else torch.tensor([])
            )
            for i, s in enumerate(source_tokens_cells)
        ]
        source_means_lens = [len(s) for s in source_means]
        # merge and split to vectorize computations
        source_means = torch.cat(source_means)
        # TODO: precompute also source_means_r3 and then just cat
        source_centroids = torch.cat(
            [source_means.to(torch.float32), r3tos2(source_means).to(torch.float32)], -1
        )
        source_centroids = torch.split(source_centroids, source_means_lens)

        return source_centroids

    def get_size_time_embedding(self) -> int:
        """
        Get size of time embedding
        """
        return self.size_time_embedding
