# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import SampleMetaData
from weathergen.datasets.masking import Masker
from weathergen.datasets.stream_data import spoof
from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    restrict_to_active,
    tokenize_apply_mask_source,
    tokenize_apply_mask_target,
    tokenize_space,
    tokenize_spacetime,
)


def readerdata_to_torch(rdata: IOReaderData) -> IOReaderData:
    """
    Convert data, coords, and geoinfos to torch tensor
    """
    if type(rdata.coords) is not torch.Tensor:
        rdata.coords = torch.tensor(rdata.coords)
    if type(rdata.geoinfos) is not torch.Tensor:
        rdata.geoinfos = torch.tensor(rdata.geoinfos)
    if type(rdata.data) is not torch.Tensor:
        rdata.data = torch.tensor(rdata.data)

    return rdata


def num_tokens(idxs_cells_lens) -> int:
    """
    Total number of tokens across cells of a (possibly active-restricted) tokenization.
    """
    return sum(len(lens) for lens in idxs_cells_lens)


class TokenizerMasking(Tokenizer):
    def __init__(
        self,
        masker: Masker,
        hl_source: int,
        hl_target: int = None,
        grid_source=None,
        grid_target=None,
        decoder_absolute_coords: bool = False,
    ):
        super().__init__(hl_source, hl_target, grid_source, grid_target)
        self.decoder_absolute_coords = decoder_absolute_coords
        self.masker = masker
        self.rng = None
        self.token_size = None

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.masker.reset_rng(rng)
        self.rng = rng

    def get_tokens_windows(self, stream_info, data, pad_tokens):
        """
        Tokenize data (to amortize over the different views that are generated)

        """

        tok_spacetime = stream_info.get("tokenize_spacetime", False)
        tok = tokenize_spacetime if tok_spacetime else tokenize_space
        # sources are tokenized on this encoder's grid (hl_source / grid_source); targets on the
        # forecast grid (hl_target / grid_target)
        hl, grid = (
            (self.hl_source, self.grid_source)
            if pad_tokens
            else (self.hl_target, self.grid_target)
        )
        token_size = stream_info["token_size"]

        # active cell ids; for a full-globe grid this is arange(12 * 4**hl), so restricting
        # below reproduces the un-restricted per-cell sequence exactly
        active_to_global = grid.active_to_global if grid is not None else np.arange(12 * 4**hl)
        # full globe cannot restrict to nothing, so it never needs the spoof below
        passthrough = grid is None or grid.is_full

        tokens = []
        for i, rdata in enumerate(data):
            # skip empty data
            if rdata.is_empty():
                tokens += [(None, None)]
                continue
            # tokenize data at the grid's level
            idxs_cells, idxs_cells_lens = tok(
                readerdata_to_torch(rdata), token_size, hl, pad_tokens
            )
            # the tokenizers return sparse per-cell maps; this is where they become the
            # per-cell sequence (in active order) that the masking below indexes positionally
            idxs_cells, idxs_cells_lens = restrict_to_active(
                idxs_cells, idxs_cells_lens, active_to_global
            )
            if not passthrough:
                if num_tokens(idxs_cells_lens) == 0 and not rdata.is_spoof:
                    rdata = spoof(
                        hl,
                        rdata.datetimes[0],
                        rdata.geoinfos.shape[1],
                        rdata.data.shape[1],
                        cells=active_to_global,
                    )
                    rdata.is_spoof = True
                    data[i] = rdata
                    idxs_cells, idxs_cells_lens = tok(
                        readerdata_to_torch(rdata), token_size, hl, pad_tokens
                    )
                    idxs_cells, idxs_cells_lens = restrict_to_active(
                        idxs_cells, idxs_cells_lens, active_to_global
                    )
            tokens += [(idxs_cells, idxs_cells_lens)]

        return tokens

    def build_samples_for_stream(
        self,
        training_mode: str,
        stream_info: dict,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Create masks for samples (source masks on this tokenizer's source grid, target masks on
        its target/forecast grid).
        """
        return self.masker.build_samples_for_stream(
            training_mode, self.grid_source, self.grid_target, stream_info
        )

    def cell_to_token_mask(self, idxs_cells, idxs_cells_lens, mask):
        """ """

        mask_tokens, mask_channels = None, None
        num_tokens = torch.tensor([len(t) for t in idxs_cells_lens]).sum().item()

        # If there are no tokens, return empty lists.
        if num_tokens == 0:
            return (mask_tokens, mask_channels)

        # TODO, TODO, TODO: use np.repeat
        # https://stackoverflow.com/questions/26038778/repeat-each-values-of-an-array-different-times
        # build token level mask: for each cell replicate the keep flag across its tokens
        token_level_flags: list[np.typing.NDArray] = []
        for km, lens_cell in zip(mask, idxs_cells_lens, strict=True):
            num_tokens_cell = len(lens_cell)
            if num_tokens_cell == 0:
                continue
            token_level_flags.append(
                np.ones(num_tokens_cell, dtype=bool)
                if km
                else np.zeros(num_tokens_cell, dtype=bool)
            )
        if token_level_flags:
            mask_tokens = np.concatenate(token_level_flags)
        else:
            mask_tokens = np.array([], dtype=bool)

        return (mask_tokens, mask_channels)

    def get_source(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        idxs_cells_data,
        time_win: tuple,
        cell_mask: torch.Tensor,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = idxs_cells_data

        # select strategy from XXX depending on stream and if student or teacher

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        source_tokens_cells, source_tokens_lens = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            stream_info["stream_id"],
            rdata,
            time_win,
            self.hpy_verts_rots_source[-1],
            encode_times_source,
        )

        return (source_tokens_cells, source_tokens_lens)

    def get_target_coords(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        # TODO: split up
        _, _, _, coords_local, coords_per_cell, idxs_data = tokenize_apply_mask_target(
            stream_info["stream_id"],
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
            self.decoder_absolute_coords,
        )

        return (coords_local, coords_per_cell, idxs_data)

    def get_target_values(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        data, datetimes, coords, _, _, idxs_data = tokenize_apply_mask_target(
            stream_info["stream_id"],
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
        )

        idxs_ord_inv = None
        if data.numel() > 0:
            # flatten per-token indices into one flat list
            idxs_flat = torch.cat([idxs for idxs_cell in idxs_cells for idxs in idxs_cell])
            # compute indices for inversion
            _, idxs_ord_inv = torch.sort(idxs_flat)

        return (data, datetimes, coords, idxs_ord_inv, idxs_data)
