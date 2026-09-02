"""A compact multi-output MLP DeepONet."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _mlp(in_dim: int, width: int, depth: int, out_dim: int, activation: str = "tanh") -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    layers: list[nn.Module] = []
    current = in_dim
    act: type[nn.Module] = nn.Tanh if activation.lower() == "tanh" else nn.GELU
    for _ in range(depth):
        layers.append(nn.Linear(current, width))
        layers.append(act())
        current = width
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Branch/trunk product with a shared branch and multi-output trunk heads.

    For a branch vector ``b`` and query ``y``, the trunk produces one latent
    vector per output channel.  The standard DeepONet contraction is

        output_k(y) = sum_l branch_l(b) * trunk_{k,l}(y).

    ``trunk`` accepts either ``[Q, trunk_dim]`` (shared query set) or
    ``[B, Q, trunk_dim]`` (per-sample query sets).
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        output_channels: int,
        width: int = 128,
        depth: int = 3,
        latent_dim: int = 64,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.branch_dim = int(branch_dim)
        self.trunk_dim = int(trunk_dim)
        self.output_channels = int(output_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.latent_dim = int(latent_dim)
        self.activation = activation
        self.branch_net = _mlp(self.branch_dim, self.width, self.depth, self.latent_dim, activation)
        self.trunk_net = _mlp(
            self.trunk_dim,
            self.width,
            self.depth,
            self.output_channels * self.latent_dim,
            activation,
        )

    def forward(self, branch: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        if branch.ndim != 2:
            raise ValueError(f"branch must be [B, Db], got {tuple(branch.shape)}")
        if trunk.ndim == 2:
            trunk = trunk.unsqueeze(0).expand(branch.shape[0], -1, -1)
        if trunk.ndim != 3:
            raise ValueError(f"trunk must be [Q, Dt] or [B, Q, Dt], got {tuple(trunk.shape)}")
        if trunk.shape[0] != branch.shape[0]:
            raise ValueError("branch and trunk batch dimensions do not match")
        branch_latent = self.branch_net(branch)  # [B, L]
        trunk_latent = self.trunk_net(trunk)  # [B, Q, K*L]
        trunk_latent = trunk_latent.view(
            trunk.shape[0], trunk.shape[1], self.output_channels, self.latent_dim
        )
        return torch.einsum("bl,bqkl->bqk", branch_latent, trunk_latent)

    def config(self) -> dict[str, int | str]:
        return {
            "branch_dim": self.branch_dim,
            "trunk_dim": self.trunk_dim,
            "output_channels": self.output_channels,
            "width": self.width,
            "depth": self.depth,
            "latent_dim": self.latent_dim,
            "activation": self.activation,
        }


def copy_state_overlap(source: nn.Module, target: nn.Module) -> list[str]:
    """Copy matching tensors and overlapping slices when dimensions change."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    copied: list[str] = []
    for key, target_tensor in target_state.items():
        if key not in source_state:
            continue
        source_tensor = source_state[key]
        if source_tensor.ndim != target_tensor.ndim:
            continue
        slices = tuple(slice(0, min(a, b)) for a, b in zip(source_tensor.shape, target_tensor.shape))
        target_tensor[slices] = source_tensor[slices].to(dtype=target_tensor.dtype)
        copied.append(key)
    target.load_state_dict(target_state)
    return copied


def freeze_for_strategy(model: DeepONet, strategy: str) -> list[str]:
    """Apply the future fine-tuning strategies requested by the project."""

    strategy = strategy.lower()
    if strategy not in {"all", "freeze_trunk", "low_lr_all"}:
        raise ValueError("strategy must be all, freeze_trunk, or low_lr_all")
    for parameter in model.parameters():
        parameter.requires_grad = True
    if strategy == "freeze_trunk":
        for parameter in model.trunk_net.parameters():
            parameter.requires_grad = False
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
