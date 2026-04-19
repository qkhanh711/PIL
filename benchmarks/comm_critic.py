from __future__ import annotations

import torch
from torch import nn

from core.models import build_mlp


def clip_by_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return x
    norms = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    scales = torch.clamp(max_norm / norms, max=1.0)
    return x * scales


class DirectedMessageSender(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, message_dim: int, stochastic: bool) -> None:
        super().__init__()
        self.stochastic = stochastic
        self.backbone = build_mlp(input_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, message_dim)
        self.log_std_head = nn.Linear(hidden_dim, message_dim) if stochastic else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mean = torch.tanh(self.mean_head(hidden))
        if self.stochastic:
            assert self.log_std_head is not None
            log_std = torch.clamp(self.log_std_head(hidden), min=-3.0, max=0.25)
            std = torch.exp(log_std)
        else:
            std = torch.full_like(mean, 1e-4)
        return mean, std


class MessageReceiver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, message_dim: int) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dim, message_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class CentralizedCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = build_mlp(input_dim, hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        return self.value_head(hidden)


class JointPosterior(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.backbone = build_mlp(input_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_var_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mean = torch.sigmoid(self.mean_head(hidden))
        log_var = torch.clamp(self.log_var_head(hidden), min=-4.0, max=1.5)
        return mean, log_var
