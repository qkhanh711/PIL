from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from PIL.core.models import ModelDims, PrivateEncoder, PrivacyScheduler, TransferMechanism, TypeEstimator
from PIL.metrics.constraints import ic_violation, ir_violation
from PIL.metrics.privacy import epsilon_from_rdp, gaussian_kl_diag, rdp_per_step


@dataclass
class PILAPSConfig:
    n_agents: int = 3
    horizon: int = 20
    batch_size: int = 64
    state_dim: int = 8
    obs_dim: int = 6
    type_dim: int = 4
    latent_dim: int = 6
    gamma: float = 0.98
    alpha_rdp: float = 10.0
    delta_dp: float = 1e-5
    rho_budget: float = 5.0
    sigma_min: float = 0.05
    lambda_kl: float = 0.2
    dual_lr: float = 5e-2
    primal_lr: float = 3e-4
    sensitivity_scale: float = 1.0
    device: str = "cpu"
    seed: int = 42


class PILAPSTrainer:
    def __init__(self, cfg: PILAPSConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        dims = ModelDims(
            n_agents=cfg.n_agents,
            state_dim=cfg.state_dim,
            obs_dim=cfg.obs_dim,
            type_dim=cfg.type_dim,
            latent_dim=cfg.latent_dim,
        )

        self.encoder = PrivateEncoder(dims.obs_dim, dims.type_dim, dims.latent_dim).to(self.device)
        self.scheduler = PrivacyScheduler(dims.state_dim, dims.type_dim, cfg.sigma_min).to(self.device)
        self.estimator = TypeEstimator(dims.n_agents, dims.latent_dim, dims.type_dim).to(self.device)
        self.transfer = TransferMechanism(dims.state_dim, dims.type_dim, dims.n_agents).to(self.device)

        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.primal_lr)

        # Dual variables (shadow prices): lambda_rho, lambda_IC, lambda_IR.
        self.lambda_rho = torch.tensor(0.0, device=self.device)
        self.lambda_ic = torch.tensor(0.0, device=self.device)
        self.lambda_ir = torch.tensor(0.0, device=self.device)

    def parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.scheduler.parameters())
            + list(self.estimator.parameters())
            + list(self.transfer.parameters())
        )

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        """
        Synthetic batch generator for a Bayesian decentralized Markov game.
        Types are fixed over horizon within each sample.
        """
        b, n, t = self.cfg.batch_size, self.cfg.n_agents, self.cfg.horizon
        d_s, d_o = self.cfg.state_dim, self.cfg.obs_dim

        theta = torch.randn(b, n, self.cfg.type_dim, device=self.device)
        states = torch.randn(b, t, d_s, device=self.device)

        # Observations depend on state and type through linear projections + noise.
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
        """Simple cooperative reward surrogate per agent at time t."""
        target = theta.mean(dim=1)[:, : self.cfg.type_dim]
        state_proj = state_t[:, : self.cfg.type_dim]
        mismatch = (state_proj - target).pow(2).mean(dim=-1, keepdim=True)
        return (-mismatch).expand(-1, self.cfg.n_agents)

    def train_step(self) -> Dict[str, float]:
        cfg = self.cfg
        batch = self._sample_batch()
        theta, states, obs = batch["theta"], batch["states"], batch["obs"]

        b, t = cfg.batch_size, cfg.horizon
        theta_hat_prev = torch.zeros(b, cfg.type_dim, device=self.device)

        discounted_welfare = torch.zeros(b, device=self.device)
        discounted_truth_utility = torch.zeros(b, cfg.n_agents, device=self.device)
        discounted_mis_utility = torch.zeros(b, cfg.n_agents, device=self.device)
        total_rho = torch.zeros(b, device=self.device)
        total_kl = torch.zeros(b, device=self.device)

        gamma_t = 1.0
        for step in range(t):
            s_t = states[:, step, :]
            o_t = obs[:, step, :, :]

            x_t = self.encoder(o_t, theta)
            sigma2_t = self.scheduler(s_t, theta_hat_prev)
            sigma_t = sigma2_t.sqrt().view(b, 1, 1)

            noise = torch.randn_like(x_t) * sigma_t
            m_t = x_t + noise

            theta_hat_t = self.estimator(m_t)
            c_t = self.transfer(s_t, theta_hat_t)
            r_base = self._base_rewards(s_t, theta)
            r_tilde = r_base + c_t

            theta_mis = theta.roll(shifts=1, dims=1)
            x_mis_t = self.encoder(o_t, theta_mis)
            m_mis_t = x_mis_t + noise.detach()
            theta_hat_mis_t = self.estimator(m_mis_t)
            c_mis_t = self.transfer(s_t, theta_hat_mis_t)
            r_mis_t = r_base + c_mis_t

            delta_t = cfg.sensitivity_scale * x_t.norm(dim=-1).mean(dim=1)
            rho_t = rdp_per_step(cfg.alpha_rdp, delta_t, sigma2_t)
            total_rho = total_rho + rho_t

            mu_private = x_t.mean(dim=1)
            var_private = x_t.var(dim=1, unbiased=False) + sigma2_t.view(b, 1)
            mu_star = x_t.detach().mean(dim=1)
            var_star = x_t.detach().var(dim=1, unbiased=False) + 1e-4
            kl_t = gaussian_kl_diag(mu_private, var_private, mu_star, var_star)
            total_kl = total_kl + kl_t

            discounted_welfare = discounted_welfare + gamma_t * r_tilde.sum(dim=-1)
            discounted_truth_utility = discounted_truth_utility + gamma_t * r_tilde
            discounted_mis_utility = discounted_mis_utility + gamma_t * r_mis_t

            theta_hat_prev = theta_hat_t
            gamma_t *= cfg.gamma

        welfare_obj = discounted_welfare.mean()
        kl_penalty = total_kl.mean()
        ic_v = ic_violation(discounted_truth_utility, discounted_mis_utility).mean()
        ir_v = ir_violation(discounted_truth_utility).mean()
        rho_violation = torch.relu(total_rho.mean() - cfg.rho_budget)

        lagrangian = (
            -(welfare_obj - cfg.lambda_kl * kl_penalty)
            + self.lambda_rho * rho_violation
            + self.lambda_ic * ic_v
            + self.lambda_ir * ir_v
        )

        self.optim.zero_grad(set_to_none=True)
        lagrangian.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optim.step()

        with torch.no_grad():
            self.lambda_rho = torch.clamp(self.lambda_rho + cfg.dual_lr * rho_violation.detach(), min=0.0)
            self.lambda_ic = torch.clamp(self.lambda_ic + cfg.dual_lr * ic_v.detach(), min=0.0)
            self.lambda_ir = torch.clamp(self.lambda_ir + cfg.dual_lr * ir_v.detach(), min=0.0)
            eps_total = epsilon_from_rdp(total_rho.mean(), cfg.alpha_rdp, cfg.delta_dp)

        return {
            "welfare": float(welfare_obj.item()),
            "kl": float(kl_penalty.item()),
            "rho_total": float(total_rho.mean().item()),
            "epsilon_total": float(eps_total.item()),
            "ic_violation": float(ic_v.item()),
            "ir_violation": float(ir_v.item()),
            "lambda_rho": float(self.lambda_rho.item()),
            "lambda_ic": float(self.lambda_ic.item()),
            "lambda_ir": float(self.lambda_ir.item()),
            "lagrangian": float(lagrangian.item()),
        }

    def train(self, num_iterations: int = 200, log_every: int = 20) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for it in tqdm(range(1, num_iterations + 1), desc="Training PIL-APS"):
            metrics = self.train_step()
            metrics["iteration"] = it
            history.append(metrics)
            if it % log_every == 0:
                print(
                    (
                        f"iter={it:04d} welfare={metrics['welfare']:.3f} "
                        f"rho={metrics['rho_total']:.3f} eps={metrics['epsilon_total']:.3f} "
                        f"IC={metrics['ic_violation']:.3f} IR={metrics['ir_violation']:.3f}"
                    )
                )
        return history
