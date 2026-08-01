"""CPU-only, leave-one-frame-out expected-height surface utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

MODELS = (
    "H0_VERTICAL_BAND_MEDIAN",
    "H1_PERSPECTIVE_GRID_MEDIAN",
    "H2_LOCAL_2D_WEIGHTED_MEDIAN",
    "H3_LOCAL_2D_UPPER_MEDIAN",
    "H4_ROBUST_QUADRATIC_SURFACE",
)
_H4_CACHE = {}


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), quantile * weights.sum(), side="left")])


def predict(model, target, references: Sequence[dict]):
    """Predict log height without using target frame references."""
    refs = [r for r in references if r["frame_id"] != target["frame_id"]]
    if len(refs) < 24:
        return {"prediction_available": False, "fallback_level": 3, "support_count": len(refs)}
    x = np.asarray([r["x_norm"] for r in refs])
    y = np.asarray([r["y_norm"] for r in refs])
    z = np.log(np.asarray([r["height"] for r in refs]))
    tx, ty = target["x_norm"], target["y_norm"]
    fallback = 0
    if model == MODELS[0]:
        band = min(11, int(ty * 12))
        keep = np.asarray([min(11, int(v * 12)) == band for v in y])
        if keep.sum() < 24:
            distances = np.abs(np.minimum(11, (y * 12).astype(int)) - band)
            keep = distances == distances.min()
            fallback = 1
        values = z[keep]
        weights = np.ones(len(values))
    elif model == MODELS[1]:
        cell = (min(5, int(tx * 6)), min(11, int(ty * 12)))
        d = np.abs(np.minimum(5, (x * 6).astype(int)) - cell[0]) + np.abs(
            np.minimum(11, (y * 12).astype(int)) - cell[1]
        )
        keep = d == d.min()
        values = z[keep]
        weights = np.ones(len(values))
        fallback = int(d[keep][0])
    elif model in MODELS[2:4]:
        d = np.sqrt((x - tx) ** 2 + 4 * (y - ty) ** 2)
        idx = np.argsort(d)[:64]
        values = z[idx]
        weights = 1 / (d[idx] + 1e-6)
        q = 0.5 if model == MODELS[2] else 0.65
        pred = weighted_quantile(values, weights, q)
        return _result(pred, values, len(values), 0)
    else:
        key = (id(references), target["frame_id"])
        cached = _H4_CACHE.get(key)
        if cached is None:
            design = np.column_stack([np.ones(len(x)), x, y, x * x, x * y, y * y])
            weights = np.ones(len(x))
            beta = np.zeros(6)
            for _ in range(20):
                beta = np.linalg.lstsq(design * weights[:, None], z * weights, rcond=None)[0]
                resid = z - design @ beta
                scale = max(np.median(np.abs(resid - np.median(resid))) * 1.4826, 1e-3)
                weights = np.minimum(1, (1.5 * scale) / np.maximum(np.abs(resid), 1e-9))
            cached = (beta, z)
            _H4_CACHE[key] = cached
        beta, z = cached
        pred = float(np.asarray([1, tx, ty, tx * tx, tx * ty, ty * ty]) @ beta)
        return _result(pred, z, len(z), 0)
    return _result(float(np.median(values)), values, len(values), fallback)


def _result(log_height, values, support, fallback):
    mad = float(np.median(np.abs(values - np.median(values))))
    return {
        "prediction_available": support >= 24,
        "expected_height_px": float(math.exp(log_height)),
        "support_count": support,
        "local_log_height_mad": mad,
        "local_log_sigma": mad * 1.4826,
        "local_p10": float(np.quantile(values, 0.1)),
        "local_p90": float(np.quantile(values, 0.9)),
        "fallback_level": fallback,
    }


def scale_features(candidate, prediction):
    if not prediction.get("prediction_available"):
        return {**prediction, "scale_status": "SCALE_UNCERTAIN"}
    relative = candidate["height"] / prediction["expected_height_px"]
    residual = math.log(relative)
    denom = max(prediction["local_log_sigma"], 0.05)
    status = (
        "VALID"
        if prediction["support_count"] >= 24
        and prediction["fallback_level"] <= 2
        and prediction["local_log_sigma"] <= 0.45
        else "SCALE_UNCERTAIN"
    )
    return {
        **prediction,
        "relative_height": relative,
        "log_scale_residual": residual,
        "robust_scale_z": residual / denom,
        "scale_status": status,
    }
