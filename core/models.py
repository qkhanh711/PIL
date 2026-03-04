from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelDims:
    n_agents: int
    state_dim: int
    obs_dim: int
    type_dim: int
    latent_dim: int


class PrivateEncoder(nn.Module):
    """x_{i,t} = f_phi(theta_i, o_{i,t})."""

    def __init__(self, obs_dim: int, type_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + type_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, theta], dim=-1)
        return self.net(x)


class PrivacyScheduler(nn.Module):
    """sigma_t^2 = softplus(g_eta(s_t, theta_hat_{t-1})) + sigma_min^2."""

    def __init__(self, state_dim: int, type_dim: int, sigma_min: float) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.net = nn.Sequential(
            nn.Linear(state_dim + type_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, prev_theta_hat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, prev_theta_hat], dim=-1)
        raw = self.net(x).squeeze(-1)
        return F.softplus(raw) + self.sigma_min**2


class TypeEstimator(nn.Module):
    """theta_hat_t = h_psi(m_{1,t}, ..., m_{N,t})."""

    def __init__(self, n_agents: int, latent_dim: int, type_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, type_dim),
        )

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        # messages: [batch, n_agents, latent_dim]
        flat = messages.reshape(messages.shape[0], -1)
        return self.net(flat)


class TransferMechanism(nn.Module):
    """c_t = C_omega(theta_hat_t, s_t), returns per-agent transfer vector."""

    def __init__(self, state_dim: int, type_dim: int, n_agents: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + type_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_agents),
        )

    def forward(self, state: torch.Tensor, theta_hat: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([state, theta_hat], dim=-1)
        return self.net(inp)
