# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regrid a per-cell latent between two HEALPix levels (an encoder's grid -> the forecast grid).

- ``L == F`` with the same active region: identity (a differing region at the same level falls
  back to a shift-0 active->active gather).
- ``L  > F`` (encoder finer than forecast): **reduce** -- pool the ``4**(L-F)`` child cells that
  fall under each forecast cell (nested ids: parent = child >> 2*(L-F)). The pooling op is
  configurable (``avg`` default, ``max``, ``sum``).
- ``L  < F`` (encoder coarser than forecast): **broadcast** -- every forecast cell copies the value
  of its ancestor cell (ancestor = cell >> 2*(F-L)); the same value lands on all the ancestor's
  children.
"""

import numpy as np
import torch
from numpy.typing import NDArray


class LatentRegridder(torch.nn.Module):
    """Map ``(rs, num_cells_src, Q, dim)`` latents from a source grid onto a destination grid."""

    def __init__(self, grid_src, grid_dst, reduce_op: str = "avg") -> None:
        super().__init__()
        assert reduce_op in ("avg", "max", "sum"), f"unsupported reduce_op {reduce_op}"
        self.reduce_op = reduce_op
        self.level_src = grid_src.level
        self.level_dst = grid_dst.level
        self.num_dst = grid_dst.num_cells
        if self.level_src == self.level_dst:
            same_grid = grid_src.num_cells == grid_dst.num_cells and np.array_equal(
                grid_src.active_to_global, grid_dst.active_to_global
            )
            self.mode = "identity" if same_grid else "broadcast"
        elif self.level_src > self.level_dst:
            self.mode = "reduce"
        else:
            self.mode = "broadcast"

        if self.mode == "reduce":
            shift = 2 * (self.level_src - self.level_dst)
            parent_global = grid_src.active_to_global >> shift
            src_to_dst = grid_dst.to_active(parent_global)
            counts = np.bincount(
                src_to_dst[src_to_dst >= 0], minlength=self.num_dst
            ).astype(np.float32)
            # store numpy for reset_parameters; register zero-filled buffers so the model
            # can be constructed on the meta device (torch.from_numpy bypasses meta context)
            self._src_to_dst_np = src_to_dst
            self._dst_counts_np = counts
            self.register_buffer("src_to_dst", torch.zeros(len(src_to_dst), dtype=torch.long))
            self.register_buffer("dst_counts", torch.zeros(self.num_dst))
        elif self.mode == "broadcast":
            # each active destination cell -> its ancestor source active index, or -1
            shift = 2 * (self.level_dst - self.level_src)
            ancestor_global = grid_dst.active_to_global >> shift
            # (num_cells_dst,), -1 where the ancestor cell is out of region
            dst_to_src = grid_src.to_active(ancestor_global)
            self._dst_to_src_np = dst_to_src
            self.register_buffer("dst_to_src", torch.zeros(len(dst_to_src), dtype=torch.long))

    def dst_coverage(self) -> NDArray:
        """Boolean mask over destination cells receiving at least one source contribution."""
        if self.mode == "identity":
            return np.ones(self.num_dst, dtype=bool)
        if self.mode == "broadcast":
            return self._dst_to_src_np >= 0
        return self._dst_counts_np > 0

    def reset_parameters(self) -> None:
        if self.mode == "reduce":
            self.src_to_dst.data.copy_(torch.from_numpy(self._src_to_dst_np))
            self.dst_counts.data.copy_(torch.from_numpy(self._dst_counts_np))
        elif self.mode == "broadcast":
            self.dst_to_src.data.copy_(torch.from_numpy(self._dst_to_src_np))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (rs, num_cells_src, Q, dim)
        if self.mode == "identity":
            return latent

        rs, _, q, dim = latent.shape
        if self.mode == "broadcast":
            out = latent.new_zeros((rs, self.num_dst, q, dim))
            valid = self.dst_to_src >= 0
            out[:, valid] = latent[:, self.dst_to_src[valid]]
            return out

        # reduce
        valid = self.src_to_dst >= 0
        idx = self.src_to_dst[valid]
        src = latent[:, valid]
        if self.reduce_op == "max":
            out = latent.new_full((rs, self.num_dst, q, dim), float("-inf"))
            out.index_reduce_(1, idx, src, "amax", include_self=True)
            out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
            return out

        out = latent.new_zeros((rs, self.num_dst, q, dim))
        out.index_add_(1, idx, src)
        if self.reduce_op == "avg":
            denom = self.dst_counts.clamp(min=1.0).view(1, self.num_dst, 1, 1)
            out = out / denom
        return out
