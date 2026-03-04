from __future__ import annotations

import math

import torch


def rdp_per_step(alpha: float, sensitivity: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """rho_t = alpha * Delta_t^2 / (2 * sigma_t^2)."""
    return alpha * (sensitivity**2) / (2.0 * sigma2.clamp_min(1e-8))


def epsilon_from_rdp(rho_total: torch.Tensor, alpha: float, delta: float) -> torch.Tensor:
    """Convert RDP to (epsilon, delta)-DP."""
    return rho_total + math.log(1.0 / delta) / (alpha - 1.0)


def gaussian_kl_diag(
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
    mu_q: torch.Tensor,
    var_q: torch.Tensor,
) -> torch.Tensor:
    """
    KL( N(mu_p, diag(var_p)) || N(mu_q, diag(var_q)) ).
    Returns per-sample KL.
    """
    var_p = var_p.clamp_min(1e-8)
    var_q = var_q.clamp_min(1e-8)
    term1 = torch.log(var_q).sum(dim=-1) - torch.log(var_p).sum(dim=-1)
    term2 = (var_p / var_q).sum(dim=-1)
    term3 = (((mu_q - mu_p) ** 2) / var_q).sum(dim=-1)
    k = mu_p.shape[-1]
    return 0.5 * (term1 - k + term2 + term3)
