from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from PIL.core.models import PrivateEncoder, TypeEstimator
from PIL.metrics.privacy import gaussian_kl_diag, rdp_per_step


@dataclass
class DPMACConfig:
    n_agents: int = 3
    horizon: int = 20
    batch_size: int = 64
    state_dim: int = 8
    obs_dim: int = 6
    type_dim: int = 4
    latent_dim: int = 6
    gamma: float = 0.98

    # DPMAC theorem parameters
    epsilon_i: float = 1.0
    delta_dp: float = 1e-5
    gamma1: float = 0.9
    gamma2: float = 0.9
    beta: float = 0.5
    message_norm_bound_c: float = 1.0

    # Training knobs
    primal_lr: float = 3e-4
    coord_bonus_weight: float = 0.5
    alpha_rdp_proxy: float = 10.0
    sensitivity_scale: float = 1.0
    device: str = "cpu"
    seed: int = 42


class DPMACTrainer:
    """
    Baseline for Differentially Private Multi-Agent Communication (DPMAC).

    Key differences vs PIL-APS:
    - Privacy budget epsilon_i is fixed (exogenous), not scheduled.
    - Noise variance is fixed by DPMAC theorem.
    - No IC/IR constraints or dual variables.
    """

    def __init__(self, cfg: DPMACConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.encoder = PrivateEncoder(cfg.obs_dim, cfg.type_dim, cfg.latent_dim).to(self.device)
        self.estimator = TypeEstimator(cfg.n_agents, cfg.latent_dim, cfg.type_dim).to(self.device)

        # Lightweight shared policy head for action logits from [obs_i, theta_hat].
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.obs_dim + cfg.type_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.primal_lr)

        self.alpha = self._compute_alpha()
        self.sigma2 = self._compute_sigma2_from_theorem()
        self.dp_ratio = self.sigma2 / (4.0 * cfg.message_norm_bound_c**2)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.estimator.parameters()) + list(self.policy_head.parameters())

    def _compute_alpha(self) -> float:
        cfg = self.cfg
        return math.log(1.0 / cfg.delta_dp) / (cfg.epsilon_i * (1.0 - cfg.beta)) + 1.0

    def _compute_sigma2_from_theorem(self) -> float:
        cfg = self.cfg
        sigma2 = (
            14.0
            * (cfg.gamma2**2)
            * (cfg.gamma1**2)
            * cfg.n_agents
            * (cfg.message_norm_bound_c**2)
            * self.alpha
            / (cfg.beta * cfg.epsilon_i)
        )
        # Additional sufficient condition in DPMAC write-up: sigma^2/(4C^2) >= 0.7
        sigma2_min = 2.8 * (cfg.message_norm_bound_c**2)
        return max(sigma2, sigma2_min)

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        b, n, t = self.cfg.batch_size, self.cfg.n_agents, self.cfg.horizon
        d_s, d_o = self.cfg.state_dim, self.cfg.obs_dim

        theta = torch.randn(b, n, self.cfg.type_dim, device=self.device)
        states = torch.randn(b, t, d_s, device=self.device)

        obs_state_proj = states[..., :d_o].unsqueeze(2).expand(-1, -1, n, -1)
        if self.cfg.type_dim >= d_o:
            theta_obs = theta[..., :d_o]
        else:
            pad = torch.zeros(b, n, d_o - self.cfg.type_dim, device=self.device)
            theta_obs = torch.cat([theta, pad], dim=-1)
        obs_type_proj = theta_obs[:, None, :, :].expand(-1, t, -1, -1)
        obs = obs_state_proj + 0.5 * obs_type_proj + 0.1 * torch.randn(b, t, n, d_o, device=self.device)
        return {"theta": theta, "states": states, "obs": obs}

    def _base_rewards(self, state_t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        target = theta.mean(dim=1)[:, : self.cfg.type_dim]
        state_proj = state_t[:, : self.cfg.type_dim]
        mismatch = (state_proj - target).pow(2).mean(dim=-1, keepdim=True)
        return (-mismatch).expand(-1, self.cfg.n_agents)

    def _bound_message_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Project each message vector onto l2-ball radius C."""
        c = self.cfg.message_norm_bound_c
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(c / norm, max=1.0)
        return x * scale

    def train_step(self) -> Dict[str, float]:
        cfg = self.cfg
        batch = self._sample_batch()
        theta, states, obs = batch["theta"], batch["states"], batch["obs"]

        b, t = cfg.batch_size, cfg.horizon
        discounted_welfare = torch.zeros(b, device=self.device)
        total_kl = torch.zeros(b, device=self.device)
        total_rho_proxy = torch.zeros(b, device=self.device)

        gamma_t = 1.0
        sigma2_t = torch.full((b,), self.sigma2, device=self.device)

        for step in range(t):
            s_t = states[:, step, :]
            o_t = obs[:, step, :, :]

            x_t = self.encoder(o_t, theta)
            x_t = self._bound_message_norm(x_t)

            sigma_t = sigma2_t.sqrt().view(b, 1, 1)
            noise = torch.randn_like(x_t) * sigma_t
            m_t = x_t + noise

            theta_hat_t = self.estimator(m_t)
            theta_bar = theta.mean(dim=1)

            # Simple shared policy behavior on noisy communication.
            theta_hat_expand = theta_hat_t.unsqueeze(1).expand(-1, cfg.n_agents, -1)
            policy_in = torch.cat([o_t, theta_hat_expand], dim=-1)
            action_signal = self.policy_head(policy_in).squeeze(-1)

            r_base = self._base_rewards(s_t, theta)
            coord_bonus = -((theta_hat_t - theta_bar) ** 2).mean(dim=-1, keepdim=True)
            r_tilde = r_base + cfg.coord_bonus_weight * coord_bonus.expand(-1, cfg.n_agents)

            # Mild regularization on action scale for stability.
            r_tilde = r_tilde - 0.01 * (action_signal**2)

            delta_t = cfg.sensitivity_scale * x_t.norm(dim=-1).mean(dim=1)
            rho_t = rdp_per_step(cfg.alpha_rdp_proxy, delta_t, sigma2_t)
            total_rho_proxy = total_rho_proxy + rho_t

            mu_private = x_t.mean(dim=1)
            var_private = x_t.var(dim=1, unbiased=False) + sigma2_t.view(b, 1)
            mu_star = x_t.detach().mean(dim=1)
            var_star = x_t.detach().var(dim=1, unbiased=False) + 1e-4
            kl_t = gaussian_kl_diag(mu_private, var_private, mu_star, var_star)
            total_kl = total_kl + kl_t

            discounted_welfare = discounted_welfare + gamma_t * r_tilde.sum(dim=-1)
            gamma_t *= cfg.gamma

        welfare_obj = discounted_welfare.mean()
        loss = -welfare_obj

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optim.step()

        return {
            "welfare": float(welfare_obj.item()),
            "kl": float(total_kl.mean().item()),
            "rho_total_proxy": float(total_rho_proxy.mean().item()),
            "epsilon_total": float(cfg.epsilon_i),
            "sigma2": float(self.sigma2),
            "alpha": float(self.alpha),
            "sigma2_over_4c2": float(self.dp_ratio),
            "dp_condition_satisfied": float(self.dp_ratio >= 0.7),
            "loss": float(loss.item()),
        }

    def train(self, num_iterations: int = 200, log_every: int = 20) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for it in tqdm(range(1, num_iterations + 1), desc="Training DPMAC"):
            metrics = self.train_step()
            metrics["iteration"] = it
            history.append(metrics)
            if it % log_every == 0:
                print(
                    (
                        f"iter={it:04d} welfare={metrics['welfare']:.3f} "
                        f"eps={metrics['epsilon_total']:.3f} "
                        f"sigma2={metrics['sigma2']:.3f} "
                        f"KL={metrics['kl']:.3f}"
                    )
                )
        return history
