from __future__ import annotations

import math
from typing import Any

import numpy as np


def rdp_gaussian_rho(alpha: float, sensitivity: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray:
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    sensitivity_arr = np.asarray(sensitivity, dtype=np.float64)
    safe_sigma = np.maximum(sigma_arr, 1e-8)
    return alpha * np.square(sensitivity_arr) / (2.0 * np.square(safe_sigma))


def rdp_to_dp_epsilon(total_rho: np.ndarray | float, alpha: float, delta: float) -> np.ndarray:
    rho_arr = np.asarray(total_rho, dtype=np.float64)
    return rho_arr + math.log(1.0 / delta) / max(alpha - 1.0, 1e-8)


def rho_to_sigma(
    rho: np.ndarray | float,
    *,
    alpha: float,
    clip_radius: np.ndarray | float,
    sigma_min: float,
    sigma_max: float,
) -> np.ndarray:
    rho_arr = np.asarray(rho, dtype=np.float64)
    clip_arr = np.asarray(clip_radius, dtype=np.float64)
    sigma = np.full_like(rho_arr, sigma_max, dtype=np.float64)
    active = rho_arr > 1e-8
    if np.any(active):
        sigma_active = np.sqrt(
            2.0 * alpha * np.square(np.maximum(clip_arr[active], 0.0)) / np.maximum(rho_arr[active], 1e-8)
        )
        sigma[active] = np.clip(sigma_active, sigma_min, sigma_max)
    return sigma


def empirical_l2_sensitivity(samples: np.ndarray) -> np.ndarray:
    sample_arr = np.asarray(samples, dtype=np.float64)
    if sample_arr.ndim < 3 or sample_arr.shape[0] <= 1:
        return np.zeros(sample_arr.shape[1:-1], dtype=np.float64)
    diff = sample_arr[:, None, ...] - sample_arr[None, :, ...]
    pairwise_norm = np.linalg.norm(diff, axis=-1)
    return pairwise_norm.max(axis=(0, 1))


def gaussian_channel_kl(base_std: np.ndarray, privacy_sigma: np.ndarray) -> np.ndarray:
    base_std_arr = np.asarray(base_std, dtype=np.float64)
    privacy_sigma_arr = np.asarray(privacy_sigma, dtype=np.float64)
    base_var = np.maximum(np.square(base_std_arr), 1e-8)
    private_var = base_var + np.square(privacy_sigma_arr)
    ratio = private_var / base_var
    return 0.5 * np.sum(ratio - 1.0 - np.log(ratio), axis=-1)


def summarize_privacy(
    *,
    sigmas: np.ndarray,
    total_rho_spent: np.ndarray,
    claimed_sensitivity: np.ndarray | float | None = None,
    actual_sensitivity: np.ndarray | float | None = None,
    alpha: float,
    delta: float,
    actual_total_rho_spent: np.ndarray | float | None = None,
) -> dict[str, Any]:
    epsilon = rdp_to_dp_epsilon(total_rho_spent, alpha, delta)
    payload = {
        "sigma": np.asarray(sigmas, dtype=np.float64).tolist(),
        "total_rho_spent": np.asarray(total_rho_spent, dtype=np.float64).tolist(),
        "epsilon": np.asarray(epsilon, dtype=np.float64).tolist(),
    }
    if claimed_sensitivity is not None:
        payload["claimed_sensitivity"] = np.asarray(claimed_sensitivity, dtype=np.float64).tolist()
    if actual_total_rho_spent is not None:
        actual_total_rho = np.asarray(actual_total_rho_spent, dtype=np.float64)
        payload["actual_rho_spent"] = actual_total_rho.tolist()
        payload["actual_epsilon"] = np.asarray(rdp_to_dp_epsilon(actual_total_rho, alpha, delta), dtype=np.float64).tolist()
        claimed = np.maximum(np.asarray(total_rho_spent, dtype=np.float64), 1e-8)
        payload["overspend_ratio"] = np.asarray(actual_total_rho / claimed, dtype=np.float64).tolist()
    if actual_sensitivity is not None:
        actual_rho = rdp_gaussian_rho(alpha, actual_sensitivity, sigmas)
        payload["actual_rho_spent"] = np.asarray(actual_rho, dtype=np.float64).tolist()
        payload["actual_epsilon"] = np.asarray(rdp_to_dp_epsilon(actual_rho, alpha, delta), dtype=np.float64).tolist()
        claimed = np.maximum(np.asarray(total_rho_spent, dtype=np.float64), 1e-8)
        payload["overspend_ratio"] = np.asarray(actual_rho / claimed, dtype=np.float64).tolist()
    return payload
