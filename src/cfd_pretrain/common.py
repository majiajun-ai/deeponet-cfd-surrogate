"""Small, dependency-light utilities shared by the pipeline scripts."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = ensure_parent(Path(path))
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_json_attr(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_json_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return json.loads(str(value))


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Data preparation must not require PyTorch just to seed NumPy/Python.
        pass


def select_device(requested: str = "auto"):
    """Return a torch device and a short audit dictionary.

    The function intentionally probes NPU support without assuming that
    ``torch_npu`` is installed.  This keeps the same code usable on a PC,
    Kunpeng CPU, CUDA GPU, or Ascend node.
    """

    import torch

    requested = requested.lower()
    audit: dict[str, Any] = {
        "requested": requested,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "npu_available": False,
        "resolved": "cpu",
        "fallback_reason": None,
    }

    npu_available = False
    if requested in {"auto", "npu"}:
        try:
            import torch_npu  # type: ignore  # noqa: F401

            npu_available = bool(hasattr(torch, "npu") and torch.npu.is_available())
        except Exception as exc:
            audit["fallback_reason"] = f"NPU probe unavailable: {type(exc).__name__}: {exc}"
    audit["npu_available"] = npu_available

    if requested == "npu" and npu_available:
        audit["resolved"] = "npu"
        return torch.device("npu"), audit
    if requested == "cuda" and torch.cuda.is_available():
        audit["resolved"] = "cuda"
        return torch.device("cuda"), audit
    if requested == "auto":
        if npu_available:
            audit["resolved"] = "npu"
            return torch.device("npu"), audit
        if torch.cuda.is_available():
            audit["resolved"] = "cuda"
            return torch.device("cuda"), audit

    if requested not in {"auto", "cpu", "cuda", "npu"}:
        audit["fallback_reason"] = f"Unknown device request: {requested}"
    elif requested == "cuda" and not torch.cuda.is_available():
        audit["fallback_reason"] = "CUDA requested but torch.cuda.is_available() is false"
    elif requested == "npu" and not npu_available:
        audit["fallback_reason"] = "NPU requested but torch_npu/NPU runtime is unavailable"
    return torch.device("cpu"), audit


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
