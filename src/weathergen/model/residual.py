# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Refinements applied on top of a residual prediction.

"""

import torch
from monai.networks.nets import UNet
from torch import nn
from torch.utils.checkpoint import checkpoint

from weathergen.model.attention import MultiSelfAttentionHeadVarlen

RESIDUAL_MODES = ("none", "add", "channel_mlp", "window_attention", "grid_conv")


class ChannelMLPRefine(nn.Module):
    """
    Correct each point from its own channels alone.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()

        # no norm in front: the channels arrive z-scored, and normalizing across them would hide
        # a point's overall anomaly level
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        for m in linears:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        # zeroing the last layer makes the module the identity at init
        nn.init.zeros_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)

    def forward(self, pred: torch.Tensor, cell_lens: torch.Tensor | None = None) -> torch.Tensor:
        return pred + self.mlp(pred)


class WindowAttentionRefine(nn.Module):
    """
    Predict each point from the other points of its healpix cell.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()

        dim_embed, num_heads, norm_eps = 64, 4, 1e-5

        self.norm = nn.LayerNorm(num_channels, eps=norm_eps)
        self.proj_in = nn.Linear(num_channels, dim_embed)
        self.block = MultiSelfAttentionHeadVarlen(dim_embed, num_heads, norm_eps=norm_eps)
        self.proj_out = nn.Linear(dim_embed, num_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # the output is the prediction itself, not a correction to add, so no zeroed layer here
        for m in (self.proj_in, self.proj_out):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, pred: torch.Tensor, cell_lens: torch.Tensor) -> torch.Tensor:
        """``cell_lens`` holds the points per healpix cell, concatenated over the batch."""

        ens, num_points, num_channels = pred.shape

        lens = torch.cat([cell_lens.new_zeros(1), cell_lens.repeat(ens)])

        z = self.proj_in(self.norm(pred)).flatten(0, 1)
        z = self.block(z, lens)

        return self.proj_out(z).reshape(ens, num_points, num_channels).to(pred.dtype)


class UNetRefine(nn.Module):
    """
    Predict each point from its geographic neighbourhood, on the stream's own raster.
    """

    def __init__(self, num_channels: int, grid_shape: tuple[int, int]) -> None:
        super().__init__()

        num_scales = 3

        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.cells = self.grid_shape[0] * self.grid_shape[1]
        self.step = 2**num_scales

        self.unet = UNet(
            spatial_dims=2,
            in_channels=num_channels,
            out_channels=num_channels,
            channels=(num_channels, num_channels, num_channels * 2, num_channels * 2),
            strides=(2,) * num_scales,
            num_res_units=2,
        )

    def reset_parameters(self) -> None:
        for m in self.unet.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _unet(self, image: torch.Tensor) -> torch.Tensor:
        """Run the U-Net over a raster of any size, by padding it up and cropping back."""

        height, width = image.shape[-2:]
        image = nn.functional.pad(image, (0, -width % self.step, 0, -height % self.step))

        return self.unet(image)[..., :height, :width]

    def forward(self, pred: torch.Tensor, grid_idx: torch.Tensor, num_images: int) -> torch.Tensor:
        """``grid_idx`` is each point's pixel, already offset by the sample it belongs to."""

        ens, num_points, num_channels = pred.shape
        slots = self.cells + 1

        member = torch.arange(ens, device=pred.device) * (num_images * slots)
        idx = (grid_idx.unsqueeze(0) + member.unsqueeze(1)).reshape(-1)

        values = pred.reshape(-1, num_channels).float()
        binned = values.new_zeros((ens * num_images * slots, num_channels))
        count = values.new_zeros((ens * num_images * slots, 1))
        binned.index_add_(0, idx, values)
        count.index_add_(0, idx, values.new_ones((idx.shape[0], 1)))

        image = binned / count.clamp(min=1.0)
        image = image.reshape(ens * num_images, slots, num_channels)[:, :-1]
        image = image.reshape(-1, *self.grid_shape, num_channels).permute(0, 3, 1, 2)

        out = checkpoint(self._unet, image, use_reentrant=False)
        out = out.permute(0, 2, 3, 1).reshape(ens * num_images, self.cells, num_channels)

        padded = out.new_zeros((ens * num_images, slots, num_channels))
        padded[:, :-1] = out
        padded = padded.reshape(-1, num_channels).index_select(0, idx)
        padded = padded.reshape(ens, num_points, num_channels).to(pred.dtype)

        spare = ((idx % slots) == self.cells).reshape(ens, num_points, 1)
        return torch.where(spare, pred, padded)


def residual_prediction_mode(config) -> str:
    """
    The residual prediction variant, which applies to every stream of the run.

    Rejects an unknown name, so that a typo fails the run rather than silently disabling it.
    """

    mode = config.get("residual_prediction") or "none"
    if mode not in RESIDUAL_MODES:
        raise ValueError(f"unknown residual_prediction {mode!r}, expected one of {RESIDUAL_MODES}")

    return mode


def build_residual_refine(
    mode: str, num_channels: int, grid_shape: tuple[int, int] | None = None
) -> nn.Module | None:
    """
    The module a variant puts on top of the residual sum, or None where it adds none.

    ``grid_shape`` is the stream's native raster. It is None for a stream that has no regular grid
    to scatter onto, which leaves that stream a plain addition even under grid_conv.
    """

    match mode:
        case "none" | "add":
            return None
        case "channel_mlp":
            return ChannelMLPRefine(num_channels)
        case "window_attention":
            return WindowAttentionRefine(num_channels)
        case "grid_conv":
            return UNetRefine(num_channels, grid_shape) if grid_shape else None
        case _:
            raise ValueError(
                f"unknown residual_prediction {mode!r}, expected one of {RESIDUAL_MODES}"
            )


def residual_grid_idx(
    streams_data: list, step: int, lens: list[int], cells: int, device: torch.device
) -> torch.Tensor:
    """
    The pixel of every point predicted for a step, offset so each sample gets its own image.
    """

    out = []
    for i, (sd, num_points) in enumerate(zip(streams_data, lens, strict=True)):
        rows, base = sd.target_row_idxs[step], i * (cells + 1)
        if sd.source_grid_idx is None or rows is None:
            out.append(torch.full((num_points,), base + cells, dtype=torch.int64, device=device))
        else:
            out.append(sd.source_grid_idx.index_select(0, rows) + base)

    return torch.cat(out)
