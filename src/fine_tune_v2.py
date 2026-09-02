"""Smoke-test transfer from 2D velocity DeepONet-v2 to a future 3D+p head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import torch

from cfd_pretrain.common import count_parameters, resolve_path, write_json
from cfd_pretrain.model_v2 import build_model


def copy_overlapping_state(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> list[str]:
    state = model.state_dict()
    copied: list[str] = []
    for name, old_value in checkpoint_state.items():
        if name not in state:
            continue
        new_value = state[name]
        if old_value.ndim != new_value.ndim:
            continue
        slices = tuple(slice(0, min(old_size, new_size)) for old_size, new_size in zip(old_value.shape, new_value.shape))
        new_value[slices].copy_(old_value[slices])
        copied.append(name)
    model.load_state_dict(state)
    return copied


def configure(model: torch.nn.Module, strategy: str, learning_rate: float) -> dict[str, Any]:
    if strategy not in {"all", "freeze_trunk", "low_lr_all"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if strategy == "freeze_trunk":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not name.startswith("trunk_net.")
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return {
        "strategy": strategy,
        "learning_rate": float(learning_rate if strategy != "low_lr_all" else min(learning_rate, 1e-5)),
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "trainable_parameter_count": count_parameters(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strategy", choices=("all", "freeze_trunk", "low_lr_all"), default="freeze_trunk")
    parser.add_argument("--output-channels", type=int, default=4)
    parser.add_argument("--branch-dim", type=int, default=7)
    parser.add_argument("--trunk-dim", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_config = checkpoint["model_config"]
    new_config = dict(old_config)
    new_config.update({"branch_dim": args.branch_dim, "trunk_dim": args.trunk_dim, "output_channels": args.output_channels})
    model = build_model(new_config)
    copied = copy_overlapping_state(model, checkpoint["model_state"])
    strategy = configure(model, args.strategy, args.learning_rate)
    result = {
        "checkpoint": str(checkpoint_path),
        "old_model_config": old_config,
        "future_model_config": model.config(),
        "copied_tensors": copied,
        "shared_latent_preserved": all(name.startswith(("branch_net.", "trunk_net.")) for name in copied if name not in {"output_head.weight", "output_head.bias"}),
        "replaceable_output_head": {"old_channels": int(old_config["output_channels"]), "new_channels": args.output_channels, "new_head_initialized_for_unseen_channels": args.output_channels > int(old_config["output_channels"])},
        "strategy": strategy,
        "dry_run": bool(args.dry_run),
        "status": "READY",
        "note": "2D u/v weights are not claimed as 3D velocity/pressure pretraining; w/p rows require monopile/OpenFOAM data fine-tuning.",
    }
    write_json(PROJECT_ROOT / "reports_v2" / "transfer_interface_smoke_v2.json", result)
    print(__import__("json").dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
