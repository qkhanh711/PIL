from __future__ import annotations

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class SenderNet(nn.Module):
    """Gaussian sender used by each agent."""

    def __init__(self, input_dim: int, hidden_dim: int, message_dim: int) -> None:
        super().__init__()
        self.message_dim = message_dim
        self.context_backbone = build_mlp(input_dim, hidden_dim, hidden_dim)
        self.context_mean_head = nn.Linear(hidden_dim, message_dim)
        self.log_std_head = nn.Linear(hidden_dim, message_dim)
        self.type_scale = nn.Parameter(torch.ones(message_dim))
        self.type_bias = nn.Parameter(torch.zeros(message_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.context_backbone(x)
        theta = x[:, :1]
        context_mean = self.context_mean_head(hidden)
        type_component = theta * self.type_scale.view(1, self.message_dim) + self.type_bias.view(1, self.message_dim)
        mean = type_component + 0.25 * context_mean
        log_std = torch.clamp(self.log_std_head(hidden), min=-2.5, max=1.0)
        std = torch.exp(log_std)
        return mean, std


class PosteriorNet(nn.Module):
    """Privacy-aware posterior estimator over persistent agent types."""

    def __init__(self, input_dim: int, hidden_dim: int, num_agents: int) -> None:
        super().__init__()
        self.backbone = build_mlp(input_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, num_agents)
        self.log_var_head = nn.Linear(hidden_dim, num_agents)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mean = torch.sigmoid(self.mean_head(hidden))
        log_var = torch.clamp(self.log_var_head(hidden), min=-4.0, max=1.5)
        return mean, log_var


class ActorNet(nn.Module):
    """Decentralized continuous actor for each agent."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = build_mlp(input_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))
