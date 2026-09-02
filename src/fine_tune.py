"""Future OpenFOAM adapter and pretrained DeepONet fine-tuning interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import h5py
import numpy as np
import torch

from cfd_pretrain.common import resolve_path, write_json
from cfd_pretrain.model import DeepONet, copy_state_overlap, freeze_for_strategy


@dataclass
class MonopileCase:
    """Canonical future CFD case representation.

    Coordinates and fields are point-wise arrays.  ``time`` can be scalar or
    a length-N array matching the point samples.  Force coefficients remain
    optional metadata and are deliberately not required by the field operator.
    """

    case_id: str
    U: float
    D: float
    H: float
    Re: float
    turbulence_intensity: float
    coordinates: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    time: np.ndarray | float
    Cd: float | None = None
    Cl: float | None = None
    Cl_RMS: float | None = None
    St: float | None = None


class OpenFOAMOperatorAdapter:
    """Validate canonical OpenFOAM exports and map them to dimensionless fields."""

    required_fields = ("coordinates", "velocity", "pressure")

    @staticmethod
    def validate(case: MonopileCase) -> None:
        if case.coordinates.ndim != 2 or case.coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [N, 3]")
        if case.velocity.shape != case.coordinates.shape:
            raise ValueError("velocity must have shape [N, 3]")
        if case.pressure.ndim not in {1, 2} or case.pressure.shape[0] != case.coordinates.shape[0]:
            raise ValueError("pressure must have shape [N] or [N, 1]")
        if case.U <= 0 or case.D <= 0 or case.H <= 0:
            raise ValueError("U, D, H must be positive")
        for name, array in (("coordinates", case.coordinates), ("velocity", case.velocity), ("pressure", case.pressure)):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN/Inf")

    @staticmethod
    def normalize(case: MonopileCase, rho: float = 1000.0, p_ref: float = 0.0) -> dict[str, np.ndarray]:
        OpenFOAMOperatorAdapter.validate(case)
        coordinates_star = case.coordinates / case.D
        velocity_star = case.velocity / case.U
        pressure_star = (case.pressure.reshape(-1) - p_ref) / (0.5 * rho * case.U**2)
        time = np.asarray(case.time if np.ndim(case.time) else np.full(len(case.coordinates), case.time), dtype=np.float32)
        time_star = time * case.U / case.D
        return {
            "coordinates_star": coordinates_star.astype(np.float32),
            "velocity_star": velocity_star.astype(np.float32),
            "pressure_star": pressure_star.astype(np.float32),
            "time_star": time_star.astype(np.float32),
        }


def load_pretrained(
    checkpoint_path: str | Path,
    branch_dim: int | None = None,
    trunk_dim: int | None = None,
    output_channels: int | None = None,
    device: str = "cpu",
) -> tuple[DeepONet, dict[str, Any]]:
    """Load a checkpoint and optionally widen dimensions for monopile fields."""

    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device, weights_only=False)
    old_config = dict(checkpoint["model_config"])
    new_config = dict(old_config)
    if branch_dim is not None:
        new_config["branch_dim"] = int(branch_dim)
    if trunk_dim is not None:
        new_config["trunk_dim"] = int(trunk_dim)
    if output_channels is not None:
        new_config["output_channels"] = int(output_channels)
    model = DeepONet(**new_config).to(device)
    source = DeepONet(**old_config).to(device)
    source.load_state_dict(checkpoint["model_state"])
    copied = copy_state_overlap(source, model)
    return model, {"old_config": old_config, "new_config": new_config, "copied_tensors": copied}


def configure_finetune(model: DeepONet, strategy: str, learning_rate: float = 1e-4):
    trainable = freeze_for_strategy(model, strategy)
    if strategy == "low_lr_all":
        learning_rate = min(learning_rate, 1e-5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    return optimizer, {"strategy": strategy, "trainable_parameter_names": trainable, "learning_rate": learning_rate}


def write_schema() -> None:
    schema = {
        "case_type": "MonopileCase",
        "required": {
            "case_id": "string",
            "U": "m/s",
            "D": "m",
            "H": "m",
            "Re": "dimensionless",
            "turbulence_intensity": "fraction",
            "coordinates": "[N,3] meters",
            "velocity": "[N,3] m/s",
            "pressure": "[N] Pa gauge or absolute with p_ref recorded",
            "time": "scalar or [N] seconds",
        },
        "optional": ["Cd", "Cl", "Cl_RMS", "St"],
        "operator_mapping": {
            "branch": "[U, D, H, Re, turbulence_intensity, boundary/geometry sensors] with training normalization",
            "trunk": "[x/D, y/D, z/D, t*]",
            "target": "[u/U, v/U, w/U, (p-p_ref)/(0.5*rho*U^2)]",
        },
        "strategies": {
            "A_all": "fine-tune all branch/trunk parameters",
            "B_freeze_trunk": "freeze trunk; train branch and output-compatible layers",
            "C_low_lr_all": "fine-tune all parameters with a lower learning rate",
        },
    }
    write_json(PROJECT_ROOT / "metadata" / "future_openfoam_schema.json", schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/public_pretrain/best.pt")
    parser.add_argument("--strategy", choices=["all", "freeze_trunk", "low_lr_all"], default="all")
    parser.add_argument("--branch-dim", type=int, default=7)
    parser.add_argument("--trunk-dim", type=int, default=3)
    parser.add_argument("--output-channels", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    write_schema()
    model, load_info = load_pretrained(args.checkpoint, args.branch_dim, args.trunk_dim, args.output_channels)
    optimizer, strategy_info = configure_finetune(model, args.strategy)
    result = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "load": load_info,
        "strategy": strategy_info,
        "dry_run": args.dry_run,
        "status": "READY",
        "note": "For 3D velocity+pressure use branch/trunk/output dimensions 7+/4/4 and supply a separately adapted branch sensor layout.",
    }
    write_json(PROJECT_ROOT / "reports" / "finetune_interface_smoke.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
