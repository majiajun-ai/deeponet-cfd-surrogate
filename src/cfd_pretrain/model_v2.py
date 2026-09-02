"""DeepONet-v2 with physical branch conditioning and optional Fourier coordinates."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .model import _mlp


class FourierCoordinates(nn.Module):
    def __init__(self, input_dim: int, num_frequencies: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        if self.num_frequencies > 0:
            frequencies = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * math.pi
            self.register_buffer("frequencies", frequencies, persistent=False)
        else:
            self.register_buffer("frequencies", torch.empty(0), persistent=False)

    @property
    def output_dim(self) -> int:
        if self.num_frequencies <= 0:
            return self.input_dim
        return self.input_dim * (1 + 2 * self.num_frequencies)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies <= 0:
            return coordinates
        phase = coordinates.unsqueeze(-1) * self.frequencies
        return torch.cat([coordinates, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)], dim=-1)


class DeepONetV2(nn.Module):
    """Low-rank multi-output DeepONet with a replaceable output head.

    The shared branch and trunk produce a latent product per query. A linear
    output head maps that latent product to u/v (and can later be replaced by
    a 4-channel u/v/w/p head). This keeps the transfer interface explicit.
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        output_channels: int,
        width: int = 160,
        depth: int = 4,
        latent_dim: int = 80,
        activation: str = "gelu",
        coordinate_encoding: str = "none",
        num_frequencies: int = 0,
    ) -> None:
        super().__init__()
        self.branch_dim = int(branch_dim)
        self.trunk_dim = int(trunk_dim)
        self.output_channels = int(output_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.latent_dim = int(latent_dim)
        self.activation = str(activation)
        self.coordinate_encoding = str(coordinate_encoding).lower()
        self.num_frequencies = int(num_frequencies)
        if self.coordinate_encoding not in {"none", "fourier"}:
            raise ValueError(f"Unsupported coordinate encoding: {coordinate_encoding}")
        if self.coordinate_encoding == "none":
            self.coordinate_encoder = FourierCoordinates(self.trunk_dim, 0)
        else:
            self.coordinate_encoder = FourierCoordinates(self.trunk_dim, self.num_frequencies)
        self.branch_net = _mlp(self.branch_dim, self.width, self.depth, self.latent_dim, self.activation)
        self.trunk_net = _mlp(self.coordinate_encoder.output_dim, self.width, self.depth, self.latent_dim, self.activation)
        self.output_head = nn.Linear(self.latent_dim, self.output_channels)

    def forward(self, branch: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        if branch.ndim != 2:
            raise ValueError(f"branch must be [B,branch_dim], got {tuple(branch.shape)}")
        if trunk.ndim == 2:
            trunk_encoded = self.coordinate_encoder(trunk)
            trunk_latent_single = self.trunk_net(trunk_encoded)
            trunk_latent = trunk_latent_single.unsqueeze(0).expand(branch.shape[0], -1, -1)
        else:
            if trunk.ndim != 3:
                raise ValueError(f"trunk must be [B,Q,trunk_dim] or [Q,trunk_dim], got {tuple(trunk.shape)}")
            if trunk.shape[0] != branch.shape[0]:
                raise ValueError("branch and trunk batch dimensions do not match")
            # The packaged CFD grid is shared by every sample in a batch. Reusing
            # its trunk embedding is exactly equivalent and avoids a large CPU
            # cost at the 48x32 query resolution. The fallback keeps the model
            # valid for future per-sample query grids.
            if trunk.shape[0] > 1 and torch.equal(trunk, trunk[:1].expand_as(trunk)):
                trunk_encoded = self.coordinate_encoder(trunk[0])
                trunk_latent_single = self.trunk_net(trunk_encoded)
                trunk_latent = trunk_latent_single.unsqueeze(0).expand(trunk.shape[0], -1, -1)
            else:
                trunk_encoded = self.coordinate_encoder(trunk)
                trunk_latent = self.trunk_net(trunk_encoded.reshape(-1, trunk_encoded.shape[-1])).reshape(
                    trunk.shape[0], trunk.shape[1], self.latent_dim
                )
        branch_latent = self.branch_net(branch)
        latent_product = trunk_latent * branch_latent[:, None, :]
        return self.output_head(latent_product)

    def config(self) -> dict[str, Any]:
        return {
            "branch_dim": self.branch_dim,
            "trunk_dim": self.trunk_dim,
            "output_channels": self.output_channels,
            "width": self.width,
            "depth": self.depth,
            "latent_dim": self.latent_dim,
            "activation": self.activation,
            "coordinate_encoding": self.coordinate_encoding,
            "num_frequencies": self.num_frequencies,
        }

    def replace_output_head(self, output_channels: int) -> None:
        """Replace only the channel head for a future target variable layout."""

        self.output_channels = int(output_channels)
        self.output_head = nn.Linear(self.latent_dim, self.output_channels)


def build_model(model_config: dict[str, Any]) -> DeepONetV2:
    allowed = {
        "branch_dim",
        "trunk_dim",
        "output_channels",
        "width",
        "depth",
        "latent_dim",
        "activation",
        "coordinate_encoding",
        "num_frequencies",
    }
    return DeepONetV2(**{key: value for key, value in model_config.items() if key in allowed})
