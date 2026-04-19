from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from benchmarks.comm_critic import (
    CentralizedCritic,
    DirectedMessageSender,
    JointPosterior,
    MessageReceiver,
    clip_by_norm,
)
from core.models import build_mlp
from core.trainer import AdaptivePrivacyScheduler, PrivacySchedule
from metrics.privacy import summarize_privacy, rho_to_sigma


@dataclass
class MatrixGameConfig:
    seed: int = 7
    device: str = "cpu"
    game: str = "binary_sum"
    algorithm: str = "pil"
    num_agents: int = 3
    episode_length: int = 4
    num_blocks: int = 25
    inner_updates: int = 20
    train_batch_size: int = 128
    eval_batch_size: int = 512
    hidden_dim: int = 64
    critic_hidden_dim: int = 96
    message_dim: int = 4
    lr: float = 3e-3
    critic_lr: float = 4e-3
    max_grad_norm: float = 5.0
    signal_prob: float = 0.5
    gamma: float = 0.95
    posterior_loss_coef: float = 0.12
    planner_bonus_coef: float = 0.06
    privacy_penalty_coef: float = 0.02
    message_std_coef: float = 0.01
    message_clip: float = 1.0
    privacy_alpha: float = 4.0
    delta: float = 1e-4
    sensitivity: float = 1.0
    total_rho_budget: float = 12.0
    rho_min: float = 0.05
    rho_max: float = 1.0
    lambda_init: float = 1.0
    lambda_max: float = 48.0
    benefit_init: float = 1.0
    uncertainty_weight: float = 0.8
    error_weight: float = 1.1
    reward_weight: float = 0.25
    frontload_factor: float = 0.35
    type_cost_base: float = 1.0
    type_cost_scale: float = 0.5
    noise_std_min: float = 0.05
    noise_std_max: float = 2.5
    fixed_baseline_fraction: float = 1.0
    privacy_block_length: int | None = None

    @classmethod
    def from_namespace(cls, args: Any) -> "MatrixGameConfig":
        valid_keys = {field.name for field in fields(cls)}
        payload = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MatrixSender(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, message_dim: int, stochastic: bool) -> None:
        super().__init__()
        self.stochastic = stochastic
        self.backbone = build_mlp(obs_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, message_dim)
        self.log_std_head = nn.Linear(hidden_dim, message_dim) if stochastic else None

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = torch.tanh(self.mean_head(hidden))
        if self.stochastic:
            assert self.log_std_head is not None
            log_std = torch.clamp(self.log_std_head(hidden), min=-3.0, max=0.25)
            std = torch.exp(log_std)
        else:
            std = torch.full_like(mean, 1e-4)
        return mean, std


class TarMACSender(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, message_dim: int) -> None:
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_dim, hidden_dim)
        self.query_head = nn.Linear(hidden_dim, message_dim)
        self.key_head = nn.Linear(hidden_dim, message_dim)
        self.value_head = nn.Linear(hidden_dim, message_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        query = self.query_head(hidden)
        key = self.key_head(hidden)
        value = torch.tanh(self.value_head(hidden))
        return query, key, value


class ScalarActor(nn.Module):
    def __init__(self, obs_dim: int, message_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + message_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, received_message: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([obs, received_message], dim=-1)))


class MatrixGameRunner:
    """Centralized-critic communication benchmark for fast matrix-game ablations."""

    def __init__(self, config: MatrixGameConfig) -> None:
        self.config = config
        self.algorithm = config.algorithm.lower()
        self.game = config.game.lower()
        if self.config.privacy_block_length is None:
            self.config.privacy_block_length = 1 if self.game == "binary_sum" else self.config.episode_length
        self.device = torch.device(config.device if config.device else "cpu")
        self.num_steps = 1 if self.game == "binary_sum" else config.episode_length
        self._set_seed(config.seed)
        self._init_modules()

        self.scheduler = AdaptivePrivacyScheduler(config) if self.algorithm == "pil" else None
        self.fixed_schedule = self._build_fixed_schedule()
        self.total_rho_spent = np.zeros(config.num_agents, dtype=np.float64)
        self.history: list[dict[str, Any]] = []

    def _block_length(self) -> int:
        return self.num_steps

    def _block_rho(self, schedule: PrivacySchedule) -> np.ndarray:
        return schedule.block_rho(self._block_length())

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _policy_params(self) -> list[nn.Parameter]:
        params = list(self.actors.parameters())
        params.extend(list(self.mean_senders.parameters()))
        params.extend(list(self.attn_senders.parameters()))
        params.extend(list(self.directed_senders.parameters()))
        params.extend(list(self.receivers.parameters()))
        if self.posterior is not None:
            params.extend(list(self.posterior.parameters()))
        return params

    def _init_modules(self) -> None:
        obs_dim = 3
        self.actors = nn.ModuleList(
            [ScalarActor(obs_dim, self.config.message_dim, self.config.hidden_dim) for _ in range(self.config.num_agents)]
        ).to(self.device)
        self.mean_senders = nn.ModuleList()
        self.attn_senders = nn.ModuleList()
        self.directed_senders = nn.ModuleList()
        self.receivers = nn.ModuleList()
        self.posterior = None

        use_messages = self.algorithm != "maddpg"
        use_attention = self.algorithm == "tarmac"
        directed_messages = self.algorithm in {"dpmac", "pil"}

        if use_messages and use_attention:
            self.attn_senders = nn.ModuleList(
                [TarMACSender(obs_dim, self.config.hidden_dim, self.config.message_dim) for _ in range(self.config.num_agents)]
            ).to(self.device)
        elif use_messages and directed_messages:
            self.directed_senders = nn.ModuleList(
                [
                    DirectedMessageSender(
                        obs_dim + 2,
                        self.config.hidden_dim,
                        self.config.message_dim,
                        stochastic=True,
                    )
                    for _ in range(self.config.num_agents)
                ]
            ).to(self.device)
            receiver_input_dim = max(self.config.num_agents - 1, 1) * (self.config.message_dim + 1)
            self.receivers = nn.ModuleList(
                [MessageReceiver(receiver_input_dim, self.config.hidden_dim, self.config.message_dim) for _ in range(self.config.num_agents)]
            ).to(self.device)
        elif use_messages:
            self.mean_senders = nn.ModuleList(
                [MatrixSender(obs_dim, self.config.hidden_dim, self.config.message_dim, stochastic=False) for _ in range(self.config.num_agents)]
            ).to(self.device)

        joint_obs_dim = self.config.num_agents * obs_dim
        joint_msg_dim = self.config.num_agents * self.config.message_dim
        joint_act_dim = self.config.num_agents
        context_dim = self.config.num_agents + 2
        self.critic = CentralizedCritic(
            joint_obs_dim + joint_msg_dim + joint_act_dim + context_dim,
            self.config.critic_hidden_dim,
        ).to(self.device)

        if self.algorithm == "pil" and use_messages:
            self.posterior = JointPosterior(
                joint_msg_dim + 1 + self.config.num_agents,
                self.config.hidden_dim,
                1,
            ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(self._policy_params(), lr=self.config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    def _build_fixed_schedule(self) -> PrivacySchedule:
        if self.algorithm not in {"pil", "dpmac"}:
            rho = np.zeros(self.config.num_agents, dtype=np.float64)
            sigma = np.zeros(self.config.num_agents, dtype=np.float64)
            clip = np.zeros(self.config.num_agents, dtype=np.float64)
        else:
            block_length = max(int(self.config.privacy_block_length or self._block_length()), 1)
            base_rho = (
                self.config.total_rho_budget / max(self.config.num_blocks * block_length, 1)
            ) * self.config.fixed_baseline_fraction
            rho = np.full(self.config.num_agents, base_rho, dtype=np.float64)
            rho = np.clip(rho, self.config.rho_min, self.config.rho_max)
            clip = np.full(self.config.num_agents, self.config.clip_multiplier, dtype=np.float64)
            sigma = rho_to_sigma(
                rho,
                alpha=self.config.privacy_alpha,
                clip_radius=clip,
                sigma_min=self.config.noise_std_min,
                sigma_max=self.config.noise_std_max,
            )
        return PrivacySchedule(price=0.0, rho=rho, sigma=sigma, clip=clip)

    def select_schedule(self, block_index: int, last_summary: dict[str, Any] | None) -> PrivacySchedule:
        if self.scheduler is not None:
            return self.scheduler.next_schedule(block_index, last_summary)
        return self.fixed_schedule

    def _post_block_update(self, schedule: PrivacySchedule) -> None:
        block_rho = self._block_rho(schedule)
        self.total_rho_spent += block_rho
        if self.scheduler is not None:
            self.scheduler.consume_budget(schedule.rho, block_length=self._block_length())

    def _sample_context(self, batch_size: int) -> dict[str, torch.Tensor]:
        probs = torch.full(
            (batch_size, self.num_steps, self.config.num_agents),
            self.config.signal_prob,
            dtype=torch.float32,
            device=self.device,
        )
        signals = torch.bernoulli(probs)
        return {"signals": signals}

    def _observations_for_step(self, signals: torch.Tensor, step_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        current = signals[:, step_index, :]
        cumulative = signals[:, : step_index + 1, :].mean(dim=1)
        progress = torch.full(
            (signals.shape[0], self.config.num_agents),
            float(step_index + 1) / float(self.num_steps),
            dtype=torch.float32,
            device=self.device,
        )
        obs_tensor = torch.stack([current, cumulative, progress], dim=-1)
        target = cumulative.mean(dim=1, keepdim=True)
        return obs_tensor, target

    def _zero_message_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "received": torch.zeros(batch_size, self.config.num_agents, self.config.message_dim, device=self.device),
            "sent": torch.zeros(batch_size, self.config.num_agents, self.config.message_dim, device=self.device),
            "base_std": torch.full(
                (batch_size, self.config.num_agents, self.config.message_dim),
                1e-4,
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def _message_state(
        self,
        obs_tensor: torch.Tensor,
        action_tensor: torch.Tensor | None,
        schedule: PrivacySchedule,
        *,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        batch_size = obs_tensor.shape[0]
        if self.algorithm == "maddpg":
            return self._zero_message_state(batch_size)

        if self.algorithm == "tarmac":
            queries = []
            keys = []
            values = []
            for agent_idx, sender in enumerate(self.attn_senders):
                query, key, value = sender(obs_tensor[:, agent_idx, :])
                queries.append(query)
                keys.append(key)
                values.append(value)
            sent_tensor = torch.stack(values, dim=1)
            received = []
            scale = math.sqrt(float(self.config.message_dim))
            for agent_idx in range(self.config.num_agents):
                other_indices = [other for other in range(self.config.num_agents) if other != agent_idx]
                if not other_indices:
                    received.append(torch.zeros(batch_size, self.config.message_dim, device=self.device))
                    continue
                query = queries[agent_idx]
                key_stack = torch.stack([keys[other] for other in other_indices], dim=1)
                value_stack = torch.stack([values[other] for other in other_indices], dim=1)
                logits = torch.sum(key_stack * query.unsqueeze(1), dim=-1) / max(scale, 1e-6)
                attn = torch.softmax(logits, dim=1)
                received.append(torch.sum(attn.unsqueeze(-1) * value_stack, dim=1))
            received_tensor = torch.stack(received, dim=1)
            tiny_std = torch.full_like(sent_tensor, 1e-4)
            return {"received": received_tensor, "sent": sent_tensor, "base_std": tiny_std}

        if self.algorithm == "i2c":
            messages = []
            stds = []
            for agent_idx, sender in enumerate(self.mean_senders):
                mean, std = sender(obs_tensor[:, agent_idx, :])
                messages.append(mean)
                stds.append(std)
            sent_tensor = torch.stack(messages, dim=1)
            std_tensor = torch.stack(stds, dim=1)
            received = []
            for agent_idx in range(self.config.num_agents):
                other_indices = [other for other in range(self.config.num_agents) if other != agent_idx]
                if not other_indices:
                    received.append(torch.zeros(batch_size, self.config.message_dim, device=self.device))
                    continue
                received.append(sent_tensor[:, other_indices, :].mean(dim=1))
            received_tensor = torch.stack(received, dim=1)
            return {"received": received_tensor, "sent": sent_tensor, "base_std": std_tensor}

        if action_tensor is None:
            action_tensor = torch.zeros(batch_size, self.config.num_agents, 1, device=self.device)
        sigma = torch.tensor(schedule.sigma, dtype=torch.float32, device=self.device).view(1, self.config.num_agents, 1)
        edge_messages: dict[tuple[int, int], torch.Tensor] = {}
        sent_messages = []
        stds = []
        denom = max(self.config.num_agents - 1, 1)
        for agent_idx, sender in enumerate(self.directed_senders):
            outgoing_messages = []
            outgoing_stds = []
            obs_slice = obs_tensor[:, agent_idx, :]
            act_slice = action_tensor[:, agent_idx, :]
            for receiver_idx in range(self.config.num_agents):
                if receiver_idx == agent_idx:
                    continue
                receiver_id = torch.full(
                    (batch_size, 1),
                    float(receiver_idx) / float(denom),
                    dtype=torch.float32,
                    device=self.device,
                )
                sender_input = torch.cat([obs_slice, act_slice, receiver_id], dim=-1)
                mean, std = sender(sender_input)
                latent = mean + std * torch.randn_like(std) if training else mean
                latent = clip_by_norm(latent, self.config.message_clip)
                if np.any(schedule.sigma > 0.0):
                    latent = latent + sigma[:, agent_idx, :] * torch.randn_like(latent)
                edge_messages[(agent_idx, receiver_idx)] = latent
                outgoing_messages.append(latent)
                outgoing_stds.append(std)
            sent_messages.append(torch.mean(torch.stack(outgoing_messages, dim=1), dim=1))
            stds.append(torch.mean(torch.stack(outgoing_stds, dim=1), dim=1))

        sent_tensor = torch.stack(sent_messages, dim=1)
        std_tensor = torch.stack(stds, dim=1)
        received = []
        for receiver_idx, receiver in enumerate(self.receivers):
            inputs = []
            for sender_idx in range(self.config.num_agents):
                if sender_idx == receiver_idx:
                    continue
                sender_id = torch.full(
                    (batch_size, 1),
                    float(sender_idx) / float(denom),
                    dtype=torch.float32,
                    device=self.device,
                )
                inputs.append(torch.cat([edge_messages[(sender_idx, receiver_idx)], sender_id], dim=-1))
            receiver_input = torch.cat(inputs, dim=-1)
            received.append(receiver(receiver_input))
        received_tensor = torch.stack(received, dim=1)
        return {"received": received_tensor, "sent": sent_tensor, "base_std": std_tensor}

    def _posterior_stats(
        self,
        received_tensor: torch.Tensor,
        target: torch.Tensor,
        progress: torch.Tensor,
        schedule: PrivacySchedule,
    ) -> dict[str, torch.Tensor] | None:
        if self.posterior is None:
            return None
        sigma_tensor = torch.tensor(schedule.sigma, dtype=torch.float32, device=self.device).view(1, -1)
        sigma_features = sigma_tensor.expand(received_tensor.shape[0], -1)
        posterior_input = torch.cat([received_tensor.reshape(received_tensor.shape[0], -1), progress, sigma_features], dim=-1)
        mean, log_var = self.posterior(posterior_input)
        var = log_var.exp().clamp(min=1e-4, max=5.0)
        nll = 0.5 * (torch.log(2.0 * torch.pi * var) + (target - mean).pow(2) / var)
        quality = torch.exp(-((mean - target).pow(2) / (var + 1e-4))).mean(dim=-1)
        return {
            "mean": mean,
            "var": var,
            "nll": nll.mean(),
            "quality": quality.mean(),
            "uncertainty": var.mean(),
            "error": torch.abs(mean - target).mean(),
        }

    def _critic_context(
        self,
        posterior_stats: dict[str, torch.Tensor] | None,
        schedule: PrivacySchedule,
        batch_size: int,
    ) -> torch.Tensor:
        sigma_tensor = torch.tensor(schedule.sigma, dtype=torch.float32, device=self.device).view(1, -1).expand(batch_size, -1)
        if posterior_stats is None:
            zeros = torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)
            return torch.cat([sigma_tensor, zeros], dim=-1)
        return torch.cat(
            [
                sigma_tensor,
                posterior_stats["mean"],
                posterior_stats["var"],
            ],
            dim=-1,
        )

    def _discounted_returns(self, reward_terms: list[torch.Tensor]) -> torch.Tensor:
        returns = []
        running = torch.zeros_like(reward_terms[-1])
        for reward in reversed(reward_terms):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        return torch.stack(returns, dim=0)

    def _simulate_batch(
        self,
        schedule: PrivacySchedule,
        *,
        batch_size: int,
        context: dict[str, torch.Tensor] | None = None,
        training: bool,
    ) -> dict[str, Any]:
        context = self._sample_context(batch_size) if context is None else context
        signals = context["signals"]
        received_state = self._zero_message_state(batch_size)["received"]

        trajectory: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        reward_terms: list[torch.Tensor] = []
        posterior_nll_terms: list[torch.Tensor] = []
        quality_terms: list[torch.Tensor] = []
        sender_std_terms: list[torch.Tensor] = []

        message_norm_sum = torch.zeros(self.config.num_agents, device=self.device)
        distortion_sum = torch.zeros(self.config.num_agents, device=self.device)
        allocation_error_sum = torch.zeros(self.config.num_agents, device=self.device)
        uncertainty_sum = torch.zeros(self.config.num_agents, device=self.device)
        posterior_error_sum = torch.zeros(self.config.num_agents, device=self.device)
        team_reward_sum = torch.zeros(1, device=self.device)
        oracle_reward_sum = torch.zeros(1, device=self.device)
        prediction_error_sum = torch.zeros(1, device=self.device)
        posterior_nll_sum = torch.zeros(1, device=self.device)
        posterior_quality_sum = torch.zeros(1, device=self.device)

        for step_index in range(self.num_steps):
            obs_tensor, target = self._observations_for_step(signals, step_index)
            predictions = []
            for agent_idx, actor in enumerate(self.actors):
                pred = actor(obs_tensor[:, agent_idx, :], received_state[:, agent_idx, :])
                predictions.append(pred)
            action_tensor = torch.stack(predictions, dim=1)

            progress = obs_tensor[:, 0, 2:3]
            posterior_stats = self._posterior_stats(received_state, target, progress, schedule)
            critic_context = self._critic_context(posterior_stats, schedule, batch_size)
            trajectory.append(
                (
                    obs_tensor.reshape(batch_size, -1),
                    received_state.reshape(batch_size, -1),
                    action_tensor.reshape(batch_size, -1),
                    critic_context,
                )
            )

            prediction_error = (action_tensor.squeeze(-1) - target).pow(2)
            team_error = prediction_error.mean(dim=1)
            team_reward = 1.0 - team_error
            oracle_reward = torch.ones_like(team_reward)
            modified_reward = team_reward

            if posterior_stats is not None:
                posterior_nll_terms.append(posterior_stats["nll"])
                quality_terms.append(posterior_stats["quality"])
                posterior_nll_sum += posterior_stats["nll"]
                posterior_quality_sum += posterior_stats["quality"]
                uncertainty_sum += posterior_stats["var"].mean().expand(self.config.num_agents)
                posterior_error_sum += posterior_stats["error"].expand(self.config.num_agents)
                if self.algorithm == "pil":
                    sigma_penalty = self.config.privacy_penalty_coef * float(np.mean(schedule.sigma))
                    modified_reward = modified_reward + self.config.planner_bonus_coef * posterior_stats["quality"] - sigma_penalty

            next_bundle = self._message_state(obs_tensor, action_tensor, schedule, training=training)
            received_state = next_bundle["received"]

            if self.algorithm in {"dpmac", "pil"}:
                sender_std_terms.append(next_bundle["base_std"].mean())

            base_var = next_bundle["base_std"].pow(2).clamp(min=1e-8)
            sigma_tensor = torch.tensor(schedule.sigma, dtype=torch.float32, device=self.device).view(1, self.config.num_agents, 1)
            private_var = base_var + sigma_tensor.expand_as(next_bundle["base_std"]).pow(2)
            kl_distortion = 0.5 * torch.sum(private_var / base_var - 1.0 - torch.log(private_var / base_var), dim=-1)

            reward_terms.append(modified_reward)
            team_reward_sum += team_reward.mean()
            oracle_reward_sum += oracle_reward.mean()
            prediction_error_sum += team_error.mean()
            message_norm_sum += next_bundle["sent"].norm(dim=-1).mean(dim=0)
            distortion_sum += kl_distortion.mean(dim=0)
            allocation_error_sum += torch.abs(action_tensor.squeeze(-1) - target).mean(dim=0)

        returns = self._discounted_returns(reward_terms)
        divisor = float(self.num_steps)
        return {
            "trajectory": trajectory,
            "returns": returns,
            "posterior_nll_terms": posterior_nll_terms,
            "quality_terms": quality_terms,
            "sender_std_terms": sender_std_terms,
            "team_reward_tensor": team_reward_sum / divisor,
            "oracle_team_reward_tensor": oracle_reward_sum / divisor,
            "prediction_error_tensor": prediction_error_sum / divisor,
            "posterior_nll_tensor": posterior_nll_sum / max(divisor, 1.0),
            "posterior_quality_tensor": posterior_quality_sum / max(divisor, 1.0),
            "posterior_uncertainty_tensor": uncertainty_sum / divisor,
            "allocation_error_tensor": allocation_error_sum / divisor,
            "message_norm_tensor": message_norm_sum / divisor,
            "distortion_tensor": distortion_sum / divisor,
            "posterior_error_tensor": posterior_error_sum / divisor,
        }

    def _critic_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        critic_inputs = []
        for obs_tensor, msg_tensor, action_tensor, context_tensor in batch["trajectory"]:
            critic_inputs.append(
                torch.cat(
                    [obs_tensor.detach(), msg_tensor.detach(), action_tensor.detach(), context_tensor.detach()],
                    dim=-1,
                )
            )
        critic_input_tensor = torch.cat(critic_inputs, dim=0)
        critic_target = batch["returns"].reshape(-1).detach()
        critic_prediction = self.critic(critic_input_tensor).squeeze(-1)
        return F.mse_loss(critic_prediction, critic_target)

    def _policy_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        actor_inputs = [torch.cat(parts, dim=-1) for parts in batch["trajectory"]]
        actor_input_tensor = torch.cat(actor_inputs, dim=0)
        actor_value = self.critic(actor_input_tensor).squeeze(-1)
        actor_loss = -actor_value.mean()
        if batch["posterior_nll_terms"]:
            actor_loss = actor_loss + self.config.posterior_loss_coef * torch.stack(batch["posterior_nll_terms"]).mean()
        if batch["sender_std_terms"]:
            actor_loss = actor_loss + self.config.message_std_coef * torch.stack(batch["sender_std_terms"]).mean()
        return actor_loss

    def _detach_metrics(self, raw_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "team_reward": float(raw_metrics["team_reward_tensor"].detach().cpu().item()),
            "oracle_team_reward": float(raw_metrics["oracle_team_reward_tensor"].detach().cpu().item()),
            "prediction_error": float(raw_metrics["prediction_error_tensor"].detach().cpu().item()),
            "posterior_nll": float(raw_metrics["posterior_nll_tensor"].detach().cpu().item()),
            "posterior_quality": float(raw_metrics["posterior_quality_tensor"].detach().cpu().item()),
            "posterior_uncertainty": raw_metrics["posterior_uncertainty_tensor"].detach().cpu().numpy().tolist(),
            "allocation_error": raw_metrics["allocation_error_tensor"].detach().cpu().numpy().tolist(),
            "message_norm": raw_metrics["message_norm_tensor"].detach().cpu().numpy().tolist(),
            "kl_distortion": raw_metrics["distortion_tensor"].detach().cpu().numpy().tolist(),
            "posterior_error": raw_metrics["posterior_error_tensor"].detach().cpu().numpy().tolist(),
        }

    @torch.no_grad()
    def evaluate_block(self, schedule: PrivacySchedule, block_index: int) -> dict[str, Any]:
        context = self._sample_context(self.config.eval_batch_size)
        truthful = self._detach_metrics(
            self._simulate_batch(schedule, batch_size=self.config.eval_batch_size, context=context, training=False)
        )
        welfare_regret = max(float(truthful["oracle_team_reward"]) - float(truthful["team_reward"]), 0.0)

        if self.algorithm in {"pil", "dpmac"}:
            privacy_metrics = summarize_privacy(
                sigmas=np.asarray(schedule.sigma, dtype=np.float64),
                total_rho_spent=self.total_rho_spent + self._block_rho(schedule),
                alpha=self.config.privacy_alpha,
                delta=self.config.delta,
            )
        else:
            zeros = np.zeros(self.config.num_agents, dtype=np.float64)
            privacy_metrics = {
                "sigma": zeros.tolist(),
                "total_rho_spent": zeros.tolist(),
                "epsilon": zeros.tolist(),
            }

        return {
            "block": block_index,
            "price": float(schedule.price),
            "rho": np.asarray(schedule.rho, dtype=np.float64).tolist(),
            "sigma": np.asarray(schedule.sigma, dtype=np.float64).tolist(),
            "average_episode_reward": truthful["team_reward"],
            "oracle_team_reward": truthful["oracle_team_reward"],
            "prediction_error": truthful["prediction_error"],
            "posterior_nll": truthful["posterior_nll"],
            "posterior_quality": truthful["posterior_quality"],
            "posterior_uncertainty": truthful["posterior_uncertainty"],
            "allocation_error": truthful["allocation_error"],
            "message_norm": truthful["message_norm"],
            "kl_distortion": truthful["kl_distortion"],
            "posterior_error": truthful["posterior_error"],
            "welfare_regret": welfare_regret,
            "privacy": privacy_metrics,
        }

    def _progress_desc(self, run_label: str | None = None) -> str:
        if run_label is not None:
            return run_label
        return f"{self.game.upper()}-{self.algorithm.upper()}"

    def _progress_colour(self) -> str:
        return {
            "pil": "#2E86AB",
            "dpmac": "#F18F01",
            "i2c": "#4F772D",
            "tarmac": "#7B2CBF",
            "maddpg": "#D1495B",
        }.get(self.algorithm, "#5C7AEA")

    def _progress_postfix(
        self,
        schedule: PrivacySchedule,
        block_index: int,
        *,
        loss: float | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        postfix = {
            "blk": f"{block_index + 1}/{self.config.num_blocks}",
            "lam": f"{schedule.price:.2f}",
            "rho": f"{np.mean(schedule.rho):.2f}",
            "sig": f"{np.mean(schedule.sigma):.2f}",
        }
        if loss is not None:
            postfix["loss"] = f"{loss:.3f}"
        if summary is not None:
            postfix["rew"] = f"{summary['average_episode_reward']:.3f}"
            postfix["err"] = f"{summary['prediction_error']:.3f}"
        return postfix

    def run(
        self,
        *,
        show_progress: bool = True,
        run_label: str | None = None,
        position: int = 0,
        leave_progress: bool = True,
    ) -> dict[str, Any]:
        last_summary = None
        total_updates = self.config.num_blocks * self.config.inner_updates
        with tqdm(
            total=total_updates,
            desc=self._progress_desc(run_label),
            unit="upd",
            dynamic_ncols=True,
            mininterval=0.1,
            smoothing=0.08,
            colour=self._progress_colour(),
            position=position,
            leave=leave_progress,
            disable=not show_progress,
        ) as progress:
            for block_index in range(self.config.num_blocks):
                schedule = self.select_schedule(block_index, last_summary)
                progress.set_postfix(self._progress_postfix(schedule, block_index), refresh=False)
                last_loss = None
                for _ in range(self.config.inner_updates):
                    batch = self._simulate_batch(schedule, batch_size=self.config.train_batch_size, training=True)

                    critic_loss = self._critic_loss(batch)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                    self.critic_optimizer.step()

                    for param in self.critic.parameters():
                        param.requires_grad_(False)
                    actor_loss = self._policy_loss(batch)
                    self.policy_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self._policy_params(), self.config.max_grad_norm)
                    self.policy_optimizer.step()
                    for param in self.critic.parameters():
                        param.requires_grad_(True)

                    last_loss = float((actor_loss + critic_loss).detach().cpu().item())
                    progress.update(1)

                summary = self.evaluate_block(schedule, block_index)
                self._post_block_update(schedule)
                self.history.append(summary)
                last_summary = {
                    "posterior_uncertainty": summary["posterior_uncertainty"],
                    "allocation_error": summary["allocation_error"],
                    "oracle_team_reward": summary["oracle_team_reward"],
                    "team_reward": summary["average_episode_reward"],
                }
                progress.set_postfix(
                    self._progress_postfix(schedule, block_index, loss=last_loss, summary=summary),
                    refresh=False,
                )
        return {
            "config": self.config.to_dict(),
            "game": self.game,
            "algorithm": self.algorithm,
            "history": self.history,
            "final": self.history[-1] if self.history else {},
        }

    @staticmethod
    def save_results(results: dict[str, Any], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
