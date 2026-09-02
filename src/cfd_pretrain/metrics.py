"""Masked metrics in the physically nondimensionalized field space."""

from __future__ import annotations

from typing import Any

import numpy as np


def _masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = np.broadcast_to(mask[..., None].astype(bool), array.shape)
    return array[expanded]


def field_metrics(
    prediction_model_norm: np.ndarray,
    target_model_norm: np.ndarray,
    mask: np.ndarray,
    field_mean: np.ndarray,
    field_std: np.ndarray,
) -> dict[str, float]:
    """Compute channel and aggregate metrics after inverse model scaling."""

    pred = prediction_model_norm * field_std + field_mean
    target = target_model_norm * field_std + field_mean
    valid = mask.astype(bool)
    delta = pred - target
    metrics: dict[str, float] = {}
    names = ["u", "v", "w", "p"][: pred.shape[-1]]
    for channel, name in enumerate(names):
        d = delta[..., channel][valid]
        truth = target[..., channel][valid]
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(d * d)))
        metrics[f"{name}_mae"] = float(np.mean(np.abs(d)))
        metrics[f"{name}_relative_l2"] = float(np.linalg.norm(d) / max(np.linalg.norm(truth), 1e-12))
    metrics["field_rmse"] = float(np.sqrt(np.mean(delta[valid] ** 2)))
    metrics["field_mae"] = float(np.mean(np.abs(delta[valid])))
    metrics["valid_fraction"] = float(np.mean(valid))
    if pred.shape[-1] >= 2:
        velocity_error = np.linalg.norm(delta[..., :2], axis=-1)[valid]
        velocity_truth = np.linalg.norm(target[..., :2], axis=-1)[valid]
        metrics["velocity_relative_l2"] = float(
            np.linalg.norm(velocity_error) / max(np.linalg.norm(velocity_truth), 1e-12)
        )
    if pred.shape[-1] >= 4:
        pressure_delta = delta[..., 3][valid]
        pressure_truth = target[..., 3][valid]
        metrics["pressure_rmse"] = float(np.sqrt(np.mean(pressure_delta**2)))
        metrics["pressure_relative_l2"] = float(
            np.linalg.norm(pressure_delta) / max(np.linalg.norm(pressure_truth), 1e-12)
        )
    return metrics


def mean_field_baseline(train_target: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    valid = train_mask[..., None].astype(np.float64)
    numerator = np.sum(train_target * valid, axis=0)
    denominator = np.sum(valid, axis=0)
    return (numerator / np.maximum(denominator, 1e-12)).astype(np.float32)
