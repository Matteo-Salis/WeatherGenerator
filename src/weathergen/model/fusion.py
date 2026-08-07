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
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiSelfAttentionHead,
)
from weathergen.model.layers import MLP
from weathergen.model.positional_encoding import get_rope_mode
from weathergen.utils.utils import get_dtype


def _sum_tensors(xs: list[torch.Tensor]) -> torch.Tensor:
    # sequential pairwise sum: keeps the float op order of the original running sum
    out = xs[0]
    for x in xs[1:]:
        out = out + x
    return out


class SumFusion(torch.nn.Module):
    """Additive fusion: elementwise sum of the per-encoder regridded latents."""

    name: "SumFusion"

    def __init__(self, cf: Config, num_encoders: int) -> None:
        """
        Initialize the SumFusion with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_encoders: Number of per-level encoders being fused.
        """
        super(SumFusion, self).__init__()
        self.cf = cf
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

    name: "ConcatMLPFusion"

    def __init__(self, cf: Config, num_encoders: int) -> None:
        """
        Initialize the ConcatMLPFusion with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_encoders: Number of per-level encoders being fused.
        """
        super(ConcatMLPFusion, self).__init__()
        self.cf = cf
        self.with_residual = cf.get("encoder_fusion_mlp_residual", True)

        self.mlp = MLP(
            dim_in=num_encoders * self.cf.ae_global_dim_embed,
            dim_out=self.cf.ae_global_dim_embed,
            num_layers=cf.get("encoder_fusion_mlp_num_layers", 2),
            hidden_factor=cf.get("encoder_fusion_mlp_hidden_factor", 1.0),
            pre_layer_norm=True,
            dropout_rate=cf.get("encoder_fusion_dropout_rate", 0.0),
            norm_type=self.cf.norm_type,
            norm_eps=self.cf.mlp_norm_eps,
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
    """Per-cell learnable queries cross-attend into the covering encoders' latents."""

    name: "PerceiverFusion"

    def __init__(self, cf: Config, coverage_masks: NDArray) -> None:
        """
        Initialize the PerceiverFusion with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param coverage_masks: (num_encoders, num_cells) bool, ordered by encoder level.
        """
        super(PerceiverFusion, self).__init__()
        self.cf = cf
        self.num_queries = self.cf.ae_local_num_queries
        self.num_encoders, self.num_cells = coverage_masks.shape

        self.q_fusion = torch.nn.Parameter(
            torch.zeros(self.num_cells, self.num_queries, self.cf.ae_global_dim_embed),
            requires_grad=True,
        )

        cell_idx, enc_idx = np.nonzero(coverage_masks.T)
        self._gather_np = (cell_idx * self.num_encoders + enc_idx).astype(np.int64)
        counts = coverage_masks.sum(axis=0).astype(np.int32)
        assert counts.min() >= 1, "every forecast-grid cell needs at least one covering encoder"
        self._kv_lens_np = np.concatenate([[0], counts * self.num_queries]).astype(np.int32)
        self.register_buffer("kv_gather_idx", torch.zeros(len(self._gather_np), dtype=torch.long))
        self.register_buffer("kv_lens", torch.zeros(self.num_cells + 1, dtype=torch.int32))
        self.register_buffer("q_lens", torch.zeros(self.num_cells + 1, dtype=torch.int32))

        self.fusion_blocks = torch.nn.ModuleList()

        for _ in range(cf.get("encoder_fusion_num_blocks", 2)):
            self.fusion_blocks.append(
                MultiCrossAttentionHeadVarlen(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    num_heads=cf.get("encoder_fusion_num_heads", 16),
                    with_residual=True,
                    with_qk_lnorm=cf.get("encoder_fusion_with_qk_lnorm", True),
                    dropout_rate=cf.get("encoder_fusion_dropout_rate", 0.0),
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    qk_norm_type=cf.get("qk_norm_type", self.cf.norm_type),
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )
            # MLP block
            self.fusion_blocks.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=cf.get("encoder_fusion_dropout_rate", 0.0),
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
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

        for block in self.fusion_blocks:
            q = checkpoint(block, q, kv, q_lens, kv_lens, use_reentrant=False)

        return q.reshape(rs, num_cells, nq, dim), _sum_tensors(auxs)


class FusionGlobalEngine(torch.nn.Module):
    """Run a fusion strategy, then mix the fused latents across forecast-grid cells."""

    name: "FusionGlobalEngine"

    def __init__(self, cf: Config, fusion: torch.nn.Module) -> None:
        """
        Initialize the FusionGlobalEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param fusion: The per-cell fusion strategy to run before the global blocks.
        """
        super(FusionGlobalEngine, self).__init__()
        self.cf = cf
        self.fusion = fusion
        rope_mode = get_rope_mode(self.cf)

        self.global_blocks = torch.nn.ModuleList()

        for _ in range(cf.get("encoder_fusion_global_num_blocks", 2)):
            self.global_blocks.append(
                MultiSelfAttentionHead(
                    self.cf.ae_global_dim_embed,
                    num_heads=self.cf.ae_global_num_heads,
                    dropout_rate=self.cf.ae_global_dropout_rate,
                    with_qk_lnorm=self.cf.ae_global_with_qk_lnorm,
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    qk_norm_type=cf.get("qk_norm_type", self.cf.norm_type),
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                    rope_mode=rope_mode,
                )
            )
            # MLP block
            self.global_blocks.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_global_dropout_rate,
                    hidden_factor=self.cf.ae_global_mlp_hidden_factor,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def reset_parameters(self) -> None:
        self.fusion.reset_parameters()

    def forward(
        self, cells: list[torch.Tensor], auxs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_cells, fused_aux = self.fusion(cells, auxs)
        tokens = fused_cells.flatten(1, 2)
        aux_info = None
        for block in self.global_blocks:
            tokens = checkpoint(block, tokens, None, aux_info, use_reentrant=False)

        return tokens.unflatten(1, fused_cells.shape[1:3]), fused_aux


def create_fusion_module(cf: Config, coverage_masks: NDArray) -> torch.nn.Module:
    """Build the fusion strategy selected by ``encoder_fusion_mode``, plus the global stage.

    ``coverage_masks`` is ``(num_encoders, num_cells_F)`` bool, ordered by encoder level.
    Set ``encoder_fusion_global_num_blocks: 0`` to skip the global stage.
    """
    mode = cf.get("encoder_fusion_mode", "sum")
    num_encoders = coverage_masks.shape[0]
    if mode == "sum":
        fusion = SumFusion(cf, num_encoders)
    elif mode == "concat_mlp":
        fusion = ConcatMLPFusion(cf, num_encoders)
    elif mode == "perceiver":
        fusion = PerceiverFusion(cf, coverage_masks)
    else:
        assert False, f"unknown encoder_fusion_mode '{mode}' (sum, concat_mlp, perceiver)"

    if cf.get("encoder_fusion_global_num_blocks", 2) > 0:
        fusion = FusionGlobalEngine(cf, fusion)
    return fusion
