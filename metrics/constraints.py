from __future__ import annotations

from typing import Any

import numpy as np


def welfare_regret(oracle_welfare: float, achieved_welfare: float) -> float:
    return float(max(oracle_welfare - achieved_welfare, 0.0))


def approximate_ic(truthful_utilities: np.ndarray, deviating_utilities: np.ndarray) -> np.ndarray:
    truthful = np.asarray(truthful_utilities, dtype=np.float64)
    deviating = np.asarray(deviating_utilities, dtype=np.float64)
    return np.maximum(deviating - truthful, 0.0)


def approximate_ir(utilities: np.ndarray) -> np.ndarray:
    utility_arr = np.asarray(utilities, dtype=np.float64)
    return np.maximum(-utility_arr, 0.0)


def summarize_constraints(
    *,
    truthful_utilities: np.ndarray,
    deviating_utilities: np.ndarray,
    oracle_welfare: float,
    achieved_welfare: float,
    kl_distortion: np.ndarray | None = None,
    gamma: float = 1.0,
    welfare_lipschitz: float = 1.0,
    utility_lipschitz: np.ndarray | float = 1.0,
    baseline_epsilon_ic: np.ndarray | None = None,
    baseline_epsilon_ir: np.ndarray | None = None,
) -> dict[str, Any]:
    ic = approximate_ic(truthful_utilities, deviating_utilities)
    ir = approximate_ir(truthful_utilities)
    payload = {
        "epsilon_ic": ic.tolist(),
        "epsilon_ir": ir.tolist(),
        "welfare_regret": welfare_regret(oracle_welfare, achieved_welfare),
    }
    if kl_distortion is not None:
        kl_arr = np.asarray(kl_distortion, dtype=np.float64)
        gamma_weights = np.power(float(gamma), np.arange(1, kl_arr.shape[0] + 1, dtype=np.float64))
        step_distortion = np.sqrt(np.maximum(0.5 * kl_arr.sum(axis=1), 0.0))
        welfare_bound = float(welfare_lipschitz * np.sum(gamma_weights * step_distortion))
        utility_lip = np.asarray(utility_lipschitz, dtype=np.float64)
        if utility_lip.ndim == 0:
            utility_lip = np.full_like(ic, float(utility_lip), dtype=np.float64)
        c_i = utility_lip * np.sqrt(2.0)
        perturbation_bound = c_i * np.sum(gamma_weights * np.sqrt(np.maximum(kl_arr.sum(axis=1), 0.0)))
        baseline_ic = np.zeros_like(ic) if baseline_epsilon_ic is None else np.asarray(baseline_epsilon_ic, dtype=np.float64)
        baseline_ir = np.zeros_like(ir) if baseline_epsilon_ir is None else np.asarray(baseline_epsilon_ir, dtype=np.float64)
        payload["welfare_regret_bound"] = welfare_bound
        payload["epsilon_ic_bound"] = (baseline_ic + perturbation_bound).tolist()
        payload["epsilon_ir_bound"] = (baseline_ir + perturbation_bound).tolist()
        payload["posterior_perturbation_bound"] = perturbation_bound.tolist()
        payload["kl_step_distortion"] = step_distortion.tolist()
    return payload
