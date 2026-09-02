"""Lazy HDF5 dataset for standardized DeepONet samples."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class H5OperatorDataset(Dataset):
    def __init__(self, path: str | Path, split: str) -> None:
        self.path = str(path)
        self.split = split
        self._handle = None
        import h5py

        with h5py.File(self.path, "r") as handle:
            group = handle[self.split]
            self.length = int(group["branch"].shape[0])
            self.branch_dim = int(group["branch"].shape[1])
            self.query_count = int(group["trunk"].shape[1])
            self.trunk_dim = int(group["trunk"].shape[2])
            self.output_channels = int(group["target"].shape[2])

    def _open(self):
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r")
        return self._handle[self.split]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        group = self._open()
        return {
            "branch": torch.from_numpy(group["branch"][index]).float(),
            "trunk": torch.from_numpy(group["trunk"][index]).float(),
            "target": torch.from_numpy(group["target"][index]).float(),
            "mask": torch.from_numpy(group["mask"][index]).float(),
            "case_id": int(group["case_id"][index]),
            "frame_index": int(group["frame_index"][index]),
        }

    def __del__(self) -> None:
        try:
            if self._handle is not None:
                self._handle.close()
        except Exception:
            pass
