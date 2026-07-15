# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fusion of the per-level encoder latents after regridding onto the forecast grid."""

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.common.config import Config
from weathergen.model.attention import MultiCrossAttentionHeadVarlen
from weathergen.model.layers import MLP
from weathergen.utils.utils import get_dtype


def _sum_tensors(xs: list[torch.Tensor]) -> torch.Tensor:
    # sequential pairwise sum: keeps the float op order of the original running sum
    out = xs[0]
    for x in xs[1:]:
        out = out + x
    return out


class SumFusion(torch.nn.Module):
    """Additive fusion: elementwise sum of the per-encoder regridded latents."""

    def __init__(self, cf: Config, num_encoders: int) -> None:
        super().__init__()
        self.num_encoders = num_encoders

    def reset_parameters(self) -> None:
        pass

    def forward(
        self, cells: list[torch.Tensor], auxs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _sum_tensors(cells), _sum_tensors(auxs)


class ConcatMLPFusion(torch.nn.Module):
    """Concatenate the encoder latents along the hidden dim and project back with an MLP.

    With ``encoder_fusion_mlp_residual`` (default) the MLP output is added to the additive
    baseline and the final MLP layer is zero-initialized, so training starts exactly at the
    summation baseline. Cells not covered by a regional encoder contribute zero blocks.
    """

    def __init__(self, cf: Config, num_encoders: int) -> None:
        super().__init__()
        dim = cf.ae_global_dim_embed
        self.with_residual = cf.get("encoder_fusion_mlp_residual", True)
        self.mlp = MLP(
            dim_in=num_encoders * dim,
            dim_out=dim,
            num_layers=cf.get("encoder_fusion_mlp_num_layers", 2),
            hidden_factor=cf.get("encoder_fusion_mlp_hidden_factor", 1.0),
            pre_layer_norm=True,
            dropout_rate=cf.get("encoder_fusion_dropout_rate", 0.0),
            norm_type=cf.norm_type,
            norm_eps=cf.mlp_norm_eps,
        )

    def reset_parameters(self) -> None:
        if self.with_residual:
            final = self.mlp.layers[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(
        self, cells: list[torch.Tensor], auxs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(torch.cat(cells, dim=-1))
        fused_cells = _sum_tensors(cells) + x if self.with_residual else x
        return fused_cells, _sum_tensors(auxs)


class PerceiverFusion(torch.nn.Module):
    """Per-cell learnable queries cross-attend into the covering encoders' latents.

    For each forecast-grid cell the KV set is the Q tokens of every encoder covering that cell;
    varlen attention masks out non-covering encoders, so their (zero) latents never receive
    attention mass.
    """

    def __init__(self, cf: Config, coverage_masks: NDArray) -> None:
        super().__init__()
        dim = cf.ae_global_dim_embed
        self.num_queries = cf.ae_local_num_queries
        self.num_encoders, self.num_cells = coverage_masks.shape

        self.q_fusion = torch.nn.Parameter(
            torch.zeros(self.num_cells, self.num_queries, dim), requires_grad=True
        )

        cell_idx, enc_idx = np.nonzero(coverage_masks.T)
        self._gather_np = (cell_idx * self.num_encoders + enc_idx).astype(np.int64)
        counts = coverage_masks.sum(axis=0).astype(np.int32)
        assert counts.min() >= 1, "every forecast-grid cell needs at least one covering encoder"
        self._kv_lens_np = np.concatenate([[0], counts * self.num_queries]).astype(np.int32)
        self.register_buffer("kv_gather_idx", torch.zeros(len(self._gather_np), dtype=torch.long))
        self.register_buffer("kv_lens", torch.zeros(self.num_cells + 1, dtype=torch.int32))
        self.register_buffer("q_lens", torch.zeros(self.num_cells + 1, dtype=torch.int32))

        self.cross_attn = MultiCrossAttentionHeadVarlen(
            dim,
            dim,
            num_heads=cf.get("encoder_fusion_num_heads", 16),
            with_residual=True,
            with_qk_lnorm=cf.get("encoder_fusion_with_qk_lnorm", True),
            dropout_rate=cf.get("encoder_fusion_dropout_rate", 0.0),
            with_flash=cf.with_flash_attention,
            norm_type=cf.norm_type,
            qk_norm_type=cf.get("qk_norm_type", cf.norm_type),
            norm_eps=cf.norm_eps,
            attention_dtype=get_dtype(cf.attention_dtype),
        )
        self.mlp = (
            MLP(
                dim,
                dim,
                with_residual=True,
                dropout_rate=cf.get("encoder_fusion_dropout_rate", 0.0),
                norm_type=cf.norm_type,
                norm_eps=cf.mlp_norm_eps,
            )
            if cf.get("encoder_fusion_with_mlp", True)
            else None
        )

    def reset_parameters(self) -> None:
        self.kv_gather_idx.data.copy_(torch.from_numpy(self._gather_np))
        self.kv_lens.data.copy_(torch.from_numpy(self._kv_lens_np))
        self.q_lens.data.fill_(self.num_queries)
        self.q_lens.data[0] = 0

        # small per-cell random init; the random draw already breaks symmetry across cells
        torch.nn.init.uniform_(self.q_fusion, 0.0, 1.0 / self.q_fusion.shape[-1])

    def forward(
        self, cells: list[torch.Tensor], auxs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rs, num_cells, nq, dim = cells[0].shape
        # (rs, num_cells, n_enc, Q, dim) -> gather the covered (cell, encoder) pairs, cell-major
        x = torch.stack(cells, dim=2).flatten(1, 2)
        kv = x.index_select(1, self.kv_gather_idx).reshape(-1, dim)

        q = self.q_fusion.unsqueeze(0).expand(rs, -1, -1, -1).reshape(-1, dim)
        if rs == 1:
            q_lens, kv_lens = self.q_lens, self.kv_lens
        else:
            q_lens = torch.cat([self.q_lens[:1], self.q_lens[1:].repeat(rs)])
            kv_lens = torch.cat([self.kv_lens[:1], self.kv_lens[1:].repeat(rs)])

        out = self.cross_attn(q, kv, q_lens, kv_lens)
        if self.mlp is not None:
            out = self.mlp(out)
        fused_cells = out.reshape(rs, num_cells, nq, dim)
        return fused_cells, _sum_tensors(auxs)


def create_fusion_module(cf: Config, coverage_masks: NDArray) -> torch.nn.Module:
    """Build the fusion module selected by ``encoder_fusion_mode``.

    ``coverage_masks`` is ``(num_encoders, num_cells_F)`` bool, ordered by encoder level.
    """
    mode = cf.get("encoder_fusion_mode", "sum")
    num_encoders = coverage_masks.shape[0]
    if mode == "sum":
        return SumFusion(cf, num_encoders)
    elif mode == "concat_mlp":
        return ConcatMLPFusion(cf, num_encoders)
    elif mode == "perceiver":
        return PerceiverFusion(cf, coverage_masks)
    assert False, f"unknown encoder_fusion_mode '{mode}' (sum, concat_mlp, perceiver)"
