"""Full-field, wake, near-cylinder, time and probe metrics for v2."""

from __future__ import annotations

from typing import Any

import numpy as np


def _metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask.astype(bool)
    delta = pred - truth
    result: dict[str, float] = {}
    for index, name in enumerate(("u", "v")[: pred.shape[-1]]):
        error = delta[..., index][valid]
        target = truth[..., index][valid]
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[f"{name}_mae"] = float(np.mean(np.abs(error)))
        result[f"{name}_relative_l2"] = float(np.linalg.norm(error) / max(np.linalg.norm(target), 1e-12))
    result["field_rmse"] = float(np.sqrt(np.mean(delta[valid] ** 2)))
    result["field_mae"] = float(np.mean(np.abs(delta[valid])))
    result["relative_l2"] = float(np.linalg.norm(delta[valid]) / max(np.linalg.norm(truth[valid]), 1e-12))
    result["valid_fraction"] = float(np.mean(valid))
    if pred.shape[-1] >= 2:
        velocity_error = np.linalg.norm(delta[..., :2], axis=-1)[valid]
        velocity_target = np.linalg.norm(truth[..., :2], axis=-1)[valid]
        result["velocity_relative_l2"] = float(np.linalg.norm(velocity_error) / max(np.linalg.norm(velocity_target), 1e-12))
    return result


def region_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    obstacle_mask: np.ndarray,
    trunk: np.ndarray,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> dict[str, float]:
    region = obstacle_mask.astype(bool)
    if x_limits is not None:
        region &= (trunk[..., 0] > x_limits[0]) & (trunk[..., 0] <= x_limits[1])
    if y_limits is not None:
        region &= (trunk[..., 1] >= y_limits[0]) & (trunk[..., 1] <= y_limits[1])
    if not region.any():
        return {"field_rmse": float("nan"), "relative_l2": float("nan"), "valid_fraction": 0.0}
    return _metrics(pred, truth, region)


def full_and_regions(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, trunk: np.ndarray) -> dict[str, Any]:
    # The released domain reaches x/D=8, so use the largest physically available downstream interval.
    result: dict[str, Any] = {"full": _metrics(pred, truth, mask)}
    result["wake"] = region_metrics(pred, truth, mask, trunk, x_limits=(0.0, 8.0), y_limits=(-3.0, 3.0))
    result["near_cylinder"] = region_metrics(pred, truth, mask, trunk, x_limits=(-1.0, 3.0), y_limits=(-2.0, 2.0))
    return result


def by_time(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, time_norm: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, lower, upper in (("early", 0.0, 1.0 / 3.0), ("mid", 1.0 / 3.0, 2.0 / 3.0), ("late", 2.0 / 3.0, 1.000001)):
        selected = (time_norm >= lower) & (time_norm < upper)
        if selected.any():
            result[name] = _metrics(pred[selected], truth[selected], mask[selected])
    return result


def by_case(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    trunk: np.ndarray,
    time_norm: np.ndarray,
    case_ids: np.ndarray,
    re_by_case: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(int(value) for value in case_ids.tolist())):
        selected = case_ids == case_id
        metrics = full_and_regions(pred[selected], truth[selected], mask[selected], trunk[selected])
        rows.append({"case_id": case_id, "Re": float(re_by_case[case_id]), **metrics})
    return rows


def _cross_correlation_lag(truth: np.ndarray, pred: np.ndarray) -> int:
    truth_centered = truth - np.mean(truth)
    pred_centered = pred - np.mean(pred)
    max_lag = max(1, min(len(truth) // 3, 20))
    best_lag = 0
    best_score = -float("inf")
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = truth_centered[-lag:], pred_centered[:lag]
        elif lag > 0:
            left, right = truth_centered[:-lag], pred_centered[lag:]
        else:
            left, right = truth_centered, pred_centered
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        score = float(np.dot(left, right) / max(denom, 1e-12))
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def probe_diagnostics(
    pred: np.ndarray,
    truth: np.ndarray,
    trunk: np.ndarray,
    time_norm: np.ndarray,
    case_ids: np.ndarray,
    probe: tuple[float, float] = (4.0, 0.0),
) -> list[dict[str, Any]]:
    distances = np.sum((trunk[0] - np.asarray(probe, dtype=np.float32)) ** 2, axis=-1)
    query_index = int(np.argmin(distances))
    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(int(value) for value in case_ids.tolist())):
        selected = case_ids == case_id
        order = np.argsort(time_norm[selected])
        truth_series = truth[selected, query_index, 0][order]
        pred_series = pred[selected, query_index, 0][order]
        error = pred_series - truth_series
        truth_std = float(np.std(truth_series))
        pred_std = float(np.std(pred_series))
        correlation = (
            float(np.corrcoef(truth_series, pred_series)[0, 1])
            if len(truth_series) > 1 and truth_std > 1e-12 and pred_std > 1e-12
            else float("nan")
        )
        amplitude_ratio = pred_std / max(truth_std, 1e-12)
        lag = _cross_correlation_lag(truth_series, pred_series) if len(truth_series) > 3 else 0
        amplitude_error = abs(amplitude_ratio - 1.0)
        if np.isfinite(correlation) and correlation < 0.8 and abs(lag) > 0:
            diagnosis = "primarily phase/temporal alignment error"
        elif amplitude_error > 0.25:
            diagnosis = "primarily amplitude error"
        else:
            diagnosis = "mixed or small probe error"
        rows.append(
            {
                "case_id": case_id,
                "query_index": query_index,
                "probe_x_star": float(trunk[0, query_index, 0]),
                "probe_y_star": float(trunk[0, query_index, 1]),
                "u_probe_rmse": float(np.sqrt(np.mean(error**2))),
                "u_truth_std": truth_std,
                "u_prediction_std": pred_std,
                "amplitude_ratio": float(amplitude_ratio),
                "correlation": correlation,
                "best_lag_frames": int(lag),
                "diagnosis": diagnosis,
            }
        )
    return rows
