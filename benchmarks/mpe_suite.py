from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
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
from metrics.privacy import rho_to_sigma, summarize_privacy


@dataclass
class MPEBenchmarkConfig:
    seed: int = 7
    device: str = "cpu"
    scenario: str = "cn"
    algorithm: str = "dpmac"
    episodes: int = 100
    eval_episodes: int = 8
    eval_interval: int = 10
    max_cycles: int = 25
    hidden_dim: int = 64
    critic_hidden_dim: int = 96
    message_dim: int = 8
    lr: float = 3e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    action_std: float = 0.15
    privacy_sigma: float = 0.20
    privacy_sigma_min: float = 0.05
    privacy_sigma_max: float = 0.35
    privacy_alpha: float = 4.0
    delta: float = 1e-4
    sensitivity: float = 1.0
    entropy_coef: float = 1e-3
    posterior_loss_coef: float = 0.15
    planner_bonus_coef: float = 0.12
    privacy_penalty_coef: float = 0.05
    message_std_coef: float = 0.01
    message_clip: float = 1.0
    grad_clip: float = 5.0
    total_rho_budget: float = 9.0
    rho_min: float = 0.02
    rho_max: float = 0.45
    lambda_init: float = 1.0
    lambda_max: float = 32.0
    benefit_init: float = 1.0
    uncertainty_weight: float = 1.0
    error_weight: float = 1.2
    reward_weight: float = 0.5
    frontload_factor: float = 0.2
    type_cost_base: float = 1.0
    type_cost_scale: float = 0.35
    noise_std_min: float = 0.08
    noise_std_max: float = 1.2
    fixed_baseline_fraction: float = 1.0
    privacy_block_length: int | None = None

    @classmethod
    def from_namespace(cls, args: Any) -> "MPEBenchmarkConfig":
        valid_keys = {field.name for field in fields(cls)}
        payload = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


class MeanMessageSender(nn.Module):
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


class ContinuousActor(nn.Module):
    def __init__(self, obs_dim: int, message_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + message_dim, hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, received_message: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([obs, received_message], dim=-1)))


def scenario_spec(config: MPEBenchmarkConfig) -> dict[str, Any]:
    scenario = config.scenario.lower()
    if scenario == "cn":
        from pettingzoo.mpe import simple_spread_v3 as env_mod

        return {
            "name": "Cooperative Navigation",
            "key": "cn",
            "env_mod": env_mod,
            "env_kwargs": {"continuous_actions": True, "max_cycles": config.max_cycles},
            "controlled_selector": lambda agents: list(agents),
        }
    if scenario == "ccn":
        from pettingzoo.mpe import simple_speaker_listener_v4 as env_mod

        return {
            "name": "Cooperative Communication & Navigation",
            "key": "ccn",
            "env_mod": env_mod,
            "env_kwargs": {"continuous_actions": True, "max_cycles": config.max_cycles},
            "controlled_selector": lambda agents: list(agents),
        }
    if scenario == "pp":
        from pettingzoo.mpe import simple_tag_v3 as env_mod

        return {
            "name": "Predator-Prey",
            "key": "pp",
            "env_mod": env_mod,
            "env_kwargs": {"continuous_actions": True, "max_cycles": config.max_cycles},
            "controlled_selector": lambda agents: [agent for agent in agents if agent.startswith("adversary_")],
        }
    raise ValueError(f"Unsupported scenario: {config.scenario}")


class MPEBenchmarkRunner:
    """Centralized-critic benchmark runner for PettingZoo MPE baselines."""

    def __init__(self, config: MPEBenchmarkConfig) -> None:
        self.config = config
        if self.config.privacy_block_length is None:
            self.config.privacy_block_length = max(int(self.config.eval_interval), 1)
        self.spec = scenario_spec(config)
        self.device = torch.device(config.device if config.device else "cpu")
        self.algorithm = config.algorithm.lower()
        self._set_seed(config.seed)
        self._init_env()
        self.total_rho_spent = np.zeros(len(self.controlled_agents), dtype=np.float64)
        self.current_schedule = self._initial_schedule()
        self._init_modules()

    def _block_length(self) -> int:
        return max(int(self.config.eval_interval), 1)

    def _block_rho(self, schedule: PrivacySchedule) -> np.ndarray:
        return schedule.block_rho(self._block_length())

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _make_env(self):
        return self.spec["env_mod"].parallel_env(**self.spec["env_kwargs"])

    def _init_env(self) -> None:
        env = self._make_env()
        obs, _ = env.reset(seed=self.config.seed)
        del obs
        self.all_agents = list(env.agents)
        self.controlled_agents = self.spec["controlled_selector"](self.all_agents)
        self.other_agents = [agent for agent in self.all_agents if agent not in self.controlled_agents]
        self.controlled_index = {agent: idx for idx, agent in enumerate(self.controlled_agents)}
        self.obs_dims = {agent: int(np.prod(env.observation_space(agent).shape)) for agent in self.all_agents}
        self.action_dims = {agent: int(np.prod(env.action_space(agent).shape)) for agent in self.all_agents}
        self.action_lows = {
            agent: torch.tensor(env.action_space(agent).low, dtype=torch.float32, device=self.device)
            for agent in self.all_agents
        }
        self.action_highs = {
            agent: torch.tensor(env.action_space(agent).high, dtype=torch.float32, device=self.device)
            for agent in self.all_agents
        }
        env.close()

    def _privacy_scheduler_config(self) -> SimpleNamespace:
        num_blocks = max(1, math.ceil(self.config.episodes / max(self.config.eval_interval, 1)))
        return SimpleNamespace(
            num_agents=len(self.controlled_agents),
            num_blocks=num_blocks,
            total_rho_budget=self.config.total_rho_budget,
            rho_min=self.config.rho_min,
            rho_max=self.config.rho_max,
            lambda_init=self.config.lambda_init,
            lambda_max=self.config.lambda_max,
            benefit_init=self.config.benefit_init,
            uncertainty_weight=self.config.uncertainty_weight,
            error_weight=self.config.error_weight,
            reward_weight=self.config.reward_weight,
            frontload_factor=self.config.frontload_factor,
            type_cost_base=self.config.type_cost_base,
            type_cost_scale=self.config.type_cost_scale,
            privacy_alpha=self.config.privacy_alpha,
            sensitivity=self.config.sensitivity,
            noise_std_min=self.config.noise_std_min,
            noise_std_max=self.config.noise_std_max,
            privacy_block_length=max(int(self.config.privacy_block_length or self.config.eval_interval), 1),
        )

    def _build_fixed_schedule(self) -> PrivacySchedule:
        rho = np.zeros(len(self.controlled_agents), dtype=np.float64)
        sigma = np.zeros(len(self.controlled_agents), dtype=np.float64)
        clip = np.zeros(len(self.controlled_agents), dtype=np.float64)
        if self.algorithm == "dpmac":
            scheduler_config = self._privacy_scheduler_config()
            base_rho = (
                scheduler_config.total_rho_budget
                / max(scheduler_config.num_blocks * scheduler_config.privacy_block_length, 1)
            ) * self.config.fixed_baseline_fraction
            rho = np.full(len(self.controlled_agents), base_rho, dtype=np.float64)
            rho = np.clip(rho, self.config.rho_min, self.config.rho_max)
            clip = np.full(len(self.controlled_agents), self.config.clip_multiplier, dtype=np.float64)
            sigma = rho_to_sigma(
                rho,
                alpha=self.config.privacy_alpha,
                clip_radius=clip,
                sigma_min=self.config.noise_std_min,
                sigma_max=self.config.noise_std_max,
            )
        return PrivacySchedule(price=0.0, rho=rho, sigma=sigma, clip=clip)

    def _initial_schedule(self) -> PrivacySchedule:
        if self.algorithm == "pil":
            scheduler = AdaptivePrivacyScheduler(self._privacy_scheduler_config())
            self.privacy_scheduler = scheduler
            return scheduler.next_schedule(0, None)
        self.privacy_scheduler = None
        return self._build_fixed_schedule()

    def _init_modules(self) -> None:
        self.actors = nn.ModuleDict()
        self.mean_senders = nn.ModuleDict()
        self.attn_senders = nn.ModuleDict()
        self.directed_senders = nn.ModuleDict()
        self.receivers = nn.ModuleDict()
        self.posterior = None

        num_controlled = len(self.controlled_agents)
        use_messages = self.algorithm != "maddpg"
        use_attention = self.algorithm == "tarmac"
        stochastic_sender = self.algorithm in {"dpmac", "pil"}
        directed_messages = self.algorithm in {"dpmac", "pil"}

        joint_obs_dim = sum(self.obs_dims[agent] for agent in self.controlled_agents)
        joint_act_dim = sum(self.action_dims[agent] for agent in self.controlled_agents)
        joint_msg_dim = num_controlled * self.config.message_dim
        context_dim = 3 * num_controlled

        for agent in self.controlled_agents:
            agent_key = self._agent_key(agent)
            self.actors[agent_key] = ContinuousActor(
                self.obs_dims[agent],
                self.config.message_dim,
                self.config.hidden_dim,
                self.action_dims[agent],
            ).to(self.device)

            if use_messages and use_attention:
                self.attn_senders[agent_key] = TarMACSender(
                    self.obs_dims[agent],
                    self.config.hidden_dim,
                    self.config.message_dim,
                ).to(self.device)
            elif use_messages and directed_messages:
                self.directed_senders[agent_key] = DirectedMessageSender(
                    self.obs_dims[agent] + self.action_dims[agent] + 1,
                    self.config.hidden_dim,
                    self.config.message_dim,
                    stochastic=stochastic_sender,
                ).to(self.device)
                receiver_input_dim = max(num_controlled - 1, 1) * (self.config.message_dim + 1)
                self.receivers[agent_key] = MessageReceiver(
                    receiver_input_dim,
                    self.config.hidden_dim,
                    self.config.message_dim,
                ).to(self.device)
            elif use_messages:
                self.mean_senders[agent_key] = MeanMessageSender(
                    self.obs_dims[agent],
                    self.config.hidden_dim,
                    self.config.message_dim,
                    stochastic=False,
                ).to(self.device)

        self.critic = CentralizedCritic(
            joint_obs_dim + joint_act_dim + joint_msg_dim + context_dim,
            self.config.critic_hidden_dim,
        ).to(self.device)

        if self.algorithm == "pil" and use_messages:
            self.posterior = JointPosterior(
                joint_msg_dim + num_controlled,
                self.config.hidden_dim,
                num_controlled,
            ).to(self.device)

        policy_params = list(self.actors.parameters())
        policy_params.extend(list(self.mean_senders.parameters()))
        policy_params.extend(list(self.attn_senders.parameters()))
        policy_params.extend(list(self.directed_senders.parameters()))
        policy_params.extend(list(self.receivers.parameters()))
        if self.posterior is not None:
            policy_params.extend(list(self.posterior.parameters()))

        self.policy_optimizer = torch.optim.Adam(policy_params, lr=self.config.lr) if policy_params else None
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.history: list[dict[str, Any]] = []

    def _agent_key(self, agent: str) -> str:
        return _safe_name(agent)

    def _obs_tensor(self, obs_value: np.ndarray, obs_dim: int) -> torch.Tensor:
        return torch.tensor(obs_value, dtype=torch.float32, device=self.device).view(1, obs_dim)

    def _collect_obs_tensors(self, obs: dict[str, np.ndarray]) -> tuple[dict[str, torch.Tensor], list[str]]:
        obs_tensors = {}
        present = []
        for agent in self.controlled_agents:
            if agent in obs:
                obs_tensors[agent] = self._obs_tensor(obs[agent], self.obs_dims[agent])
                present.append(agent)
            else:
                obs_tensors[agent] = torch.zeros(1, self.obs_dims[agent], dtype=torch.float32, device=self.device)
        return obs_tensors, present

    def _scale_action(self, action: torch.Tensor, agent: str) -> torch.Tensor:
        low = self.action_lows[agent].view(1, -1)
        high = self.action_highs[agent].view(1, -1)
        center = 0.5 * (high + low)
        half_span = 0.5 * (high - low)
        scaled = center + half_span * action
        return torch.max(torch.min(scaled, high), low)

    def _random_action(self, env, agent: str) -> np.ndarray:
        return env.action_space(agent).sample()

    def _private_target(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(obs_tensor.mean(dim=-1, keepdim=True))

    def _zero_message_state(self) -> dict[str, torch.Tensor]:
        return {
            agent: torch.zeros(1, self.config.message_dim, dtype=torch.float32, device=self.device)
            for agent in self.controlled_agents
        }

    def _sigma_tensor(self) -> torch.Tensor:
        return torch.tensor(
            np.asarray(self.current_schedule.sigma, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
            device=self.device,
        )

    def _sigma_vector(self) -> np.ndarray:
        return np.asarray(self.current_schedule.sigma, dtype=np.float64)

    def _message_bundle(
        self,
        obs_tensors: dict[str, torch.Tensor],
        action_tensors: dict[str, torch.Tensor] | None,
        *,
        training: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if self.algorithm == "maddpg":
            zeros = self._zero_message_state()
            tiny_std = {agent: torch.full_like(message, 1e-4) for agent, message in zeros.items()}
            return {"received": zeros, "sent": zeros, "base_std": tiny_std}

        if self.algorithm == "tarmac":
            queries = {}
            keys = {}
            values = {}
            for agent in self.controlled_agents:
                q, k, v = self.attn_senders[self._agent_key(agent)](obs_tensors[agent])
                queries[agent] = q
                keys[agent] = k
                values[agent] = v
            received = {}
            scale = math.sqrt(float(self.config.message_dim))
            for agent in self.controlled_agents:
                other_agents = [other for other in self.controlled_agents if other != agent]
                if not other_agents:
                    received[agent] = torch.zeros(1, self.config.message_dim, device=self.device)
                    continue
                logits = torch.stack(
                    [torch.sum(queries[agent] * keys[other], dim=-1) / max(scale, 1e-6) for other in other_agents],
                    dim=0,
                ).squeeze(-1)
                attn = torch.softmax(logits, dim=0)
                value_stack = torch.cat([values[other] for other in other_agents], dim=0)
                received[agent] = torch.sum(attn.unsqueeze(-1) * value_stack, dim=0, keepdim=True)
            tiny_std = {agent: torch.full_like(value, 1e-4) for agent, value in values.items()}
            return {"received": received, "sent": values, "base_std": tiny_std}

        if self.algorithm == "i2c":
            sent = {}
            base_std = {}
            for agent in self.controlled_agents:
                mean, std = self.mean_senders[self._agent_key(agent)](obs_tensors[agent])
                sent[agent] = mean
                base_std[agent] = std
            received = {}
            for agent in self.controlled_agents:
                other_agents = [other for other in self.controlled_agents if other != agent]
                if not other_agents:
                    received[agent] = torch.zeros(1, self.config.message_dim, device=self.device)
                else:
                    received[agent] = torch.mean(torch.cat([sent[other] for other in other_agents], dim=0), dim=0, keepdim=True)
            return {"received": received, "sent": sent, "base_std": base_std}

        edge_messages: dict[tuple[str, str], torch.Tensor] = {}
        sent = {}
        base_std = {}
        if action_tensors is None:
            action_tensors = {
                agent: torch.zeros(1, self.action_dims[agent], dtype=torch.float32, device=self.device)
                for agent in self.controlled_agents
            }
        denom = max(len(self.controlled_agents) - 1, 1)
        for sender in self.controlled_agents:
            sender_messages = []
            sender_stds = []
            obs_tensor = obs_tensors[sender]
            act_tensor = action_tensors[sender]
            sender_module = self.directed_senders[self._agent_key(sender)]
            for receiver in self.controlled_agents:
                if receiver == sender:
                    continue
                receiver_id = torch.full(
                    (1, 1),
                    float(self.controlled_index[receiver]) / float(denom),
                    dtype=torch.float32,
                    device=self.device,
                )
                sender_input = torch.cat([obs_tensor, act_tensor, receiver_id], dim=-1)
                mean, std = sender_module(sender_input)
                latent = mean + std * torch.randn_like(std) if training else mean
                latent = clip_by_norm(latent, self.config.message_clip)
                sigma_value = float(self.current_schedule.sigma[self.controlled_index[sender]])
                if sigma_value > 0.0:
                    latent = latent + sigma_value * torch.randn_like(latent)
                edge_messages[(sender, receiver)] = latent
                sender_messages.append(latent)
                sender_stds.append(std)
            sent[sender] = torch.mean(torch.stack(sender_messages, dim=0), dim=0)
            base_std[sender] = torch.mean(torch.stack(sender_stds, dim=0), dim=0)

        received = {}
        for receiver in self.controlled_agents:
            receiver_inputs = []
            for sender in self.controlled_agents:
                if sender == receiver:
                    continue
                sender_id = torch.full(
                    (1, 1),
                    float(self.controlled_index[sender]) / float(denom),
                    dtype=torch.float32,
                    device=self.device,
                )
                receiver_inputs.append(torch.cat([edge_messages[(sender, receiver)], sender_id], dim=-1))
            receiver_input = torch.cat(receiver_inputs, dim=-1)
            received[receiver] = self.receivers[self._agent_key(receiver)](receiver_input)

        return {"received": received, "sent": sent, "base_std": base_std}

    def _joint_obs(self, obs_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs_tensors[agent] for agent in self.controlled_agents], dim=-1)

    def _joint_messages(self, received: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([received[agent] for agent in self.controlled_agents], dim=-1)

    def _joint_actions(self, action_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([action_tensors[agent] for agent in self.controlled_agents], dim=-1)

    def _posterior_stats(
        self,
        obs_tensors: dict[str, torch.Tensor],
        received: dict[str, torch.Tensor],
        sigma_tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if self.posterior is None:
            return None
        posterior_input = torch.cat([self._joint_messages(received), sigma_tensor], dim=-1)
        mean, log_var = self.posterior(posterior_input)
        var = log_var.exp().clamp(min=1e-4, max=5.0)
        target = torch.cat([self._private_target(obs_tensors[agent]) for agent in self.controlled_agents], dim=-1)
        nll = 0.5 * (torch.log(2.0 * torch.pi * var) + (target - mean).pow(2) / var)
        quality = torch.exp(-((mean - target).pow(2) / (var + 1e-4))).mean(dim=-1)
        error = torch.abs(mean - target).mean(dim=-1)
        return {
            "mean": mean,
            "var": var,
            "target": target,
            "nll": nll.mean(),
            "quality": quality.mean(),
            "uncertainty": var.mean(),
            "error": error.mean(),
        }

    def _critic_context(
        self,
        sigma_tensor: torch.Tensor,
        posterior_stats: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        num_controlled = len(self.controlled_agents)
        if posterior_stats is None:
            zeros = torch.zeros(1, num_controlled, dtype=torch.float32, device=self.device)
            return torch.cat([sigma_tensor, zeros, zeros], dim=-1)
        return torch.cat([sigma_tensor, posterior_stats["mean"], posterior_stats["var"]], dim=-1)

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def _rollout_episode(self, *, training: bool, seed_offset: int = 0) -> dict[str, Any]:
        env = self._make_env()
        obs, _ = env.reset(seed=self.config.seed + seed_offset)
        received_messages = self._zero_message_state()

        env_rewards: list[float] = []
        train_rewards: list[float] = []
        entropies: list[torch.Tensor] = []
        trajectory: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        posterior_nlls: list[torch.Tensor] = []
        posterior_qualities: list[torch.Tensor] = []
        posterior_uncertainties: list[torch.Tensor] = []
        posterior_errors: list[torch.Tensor] = []
        sender_std_terms: list[torch.Tensor] = []

        message_norm_sum = np.zeros(len(self.controlled_agents), dtype=np.float64)
        kl_distortion_sum = np.zeros(len(self.controlled_agents), dtype=np.float64)
        message_counts = np.zeros(len(self.controlled_agents), dtype=np.float64)
        probe_messages = {agent: [] for agent in self.controlled_agents}
        probe_targets = {agent: [] for agent in self.controlled_agents}

        while env.agents:
            obs_tensors, controlled_present = self._collect_obs_tensors(obs)
            sigma_tensor = self._sigma_tensor()
            action_tensors = {
                agent: torch.zeros(1, self.action_dims[agent], dtype=torch.float32, device=self.device)
                for agent in self.controlled_agents
            }
            action_dict: dict[str, np.ndarray] = {}
            total_entropy = torch.zeros(1, dtype=torch.float32, device=self.device)

            for agent in controlled_present:
                actor = self.actors[self._agent_key(agent)]
                mean = actor(obs_tensors[agent], received_messages[agent])
                dist = Normal(mean, torch.full_like(mean, self.config.action_std))
                raw_action = dist.rsample() if training else mean
                scaled_action = self._scale_action(raw_action, agent)
                action_tensors[agent] = scaled_action
                action_dict[agent] = scaled_action.squeeze(0).detach().cpu().numpy()
                total_entropy = total_entropy + dist.entropy().sum(dim=-1)

            for agent in env.agents:
                if agent in action_dict:
                    continue
                action_dict[agent] = self._random_action(env, agent)

            posterior_stats = self._posterior_stats(obs_tensors, received_messages, sigma_tensor)
            critic_context = self._critic_context(sigma_tensor, posterior_stats)
            trajectory.append(
                (
                    self._joint_obs(obs_tensors),
                    self._joint_messages(received_messages),
                    self._joint_actions(action_tensors),
                    critic_context,
                )
            )
            entropies.append(total_entropy.squeeze(0))

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            del terminations, truncations, infos

            reward_values = [rewards[agent] for agent in controlled_present] if controlled_present else [0.0]
            env_reward = float(np.mean(reward_values))
            train_reward = env_reward
            if posterior_stats is not None:
                posterior_nlls.append(posterior_stats["nll"])
                posterior_qualities.append(posterior_stats["quality"])
                posterior_uncertainties.append(posterior_stats["uncertainty"])
                posterior_errors.append(posterior_stats["error"])
                if self.algorithm == "pil":
                    train_reward += self.config.planner_bonus_coef * float(posterior_stats["quality"].detach().cpu().item())
                    train_reward -= self.config.privacy_penalty_coef * float(np.mean(self.current_schedule.sigma))

            env_rewards.append(env_reward)
            train_rewards.append(train_reward)

            next_bundle = self._message_bundle(obs_tensors, action_tensors, training=training)
            if self.algorithm in {"dpmac", "pil"}:
                sender_std_terms.append(
                    torch.mean(
                        torch.stack([next_bundle["base_std"][agent].mean() for agent in self.controlled_agents], dim=0)
                    )
                )

            for agent in self.controlled_agents:
                agent_index = self.controlled_index[agent]
                sent_message = next_bundle["sent"][agent]
                base_std = next_bundle["base_std"][agent]
                private_target = self._private_target(obs_tensors[agent])

                message_norm_sum[agent_index] += float(sent_message.norm(dim=-1).mean().detach().cpu().item())
                message_counts[agent_index] += 1.0
                probe_messages[agent].append(sent_message.detach().cpu().numpy())
                probe_targets[agent].append(private_target.detach().cpu().numpy())

                sigma_value = float(self.current_schedule.sigma[agent_index])
                if sigma_value > 0.0:
                    base_var = base_std.pow(2).clamp(min=1e-8)
                    private_var = base_var + sigma_value ** 2
                    kl_value = 0.5 * torch.sum(private_var / base_var - 1.0 - torch.log(private_var / base_var), dim=-1)
                    kl_distortion_sum[agent_index] += float(kl_value.mean().detach().cpu().item())

            received_messages = next_bundle["received"]
            obs = next_obs

        env.close()
        safe_counts = np.maximum(message_counts, 1.0)
        return {
            "episode_reward": float(np.sum(env_rewards)),
            "step_rewards": env_rewards,
            "train_step_rewards": train_rewards,
            "trajectory": trajectory,
            "entropies": entropies,
            "posterior_nlls": posterior_nlls,
            "posterior_qualities": posterior_qualities,
            "posterior_uncertainties": posterior_uncertainties,
            "posterior_errors": posterior_errors,
            "sender_std_terms": sender_std_terms,
            "message_norm": (message_norm_sum / safe_counts).tolist(),
            "kl_distortion": (kl_distortion_sum / safe_counts).tolist(),
            "probe_messages": probe_messages,
            "probe_targets": probe_targets,
        }

    def _policy_loss(self, rollout: dict[str, Any]) -> torch.Tensor:
        assert self.policy_optimizer is not None
        actor_values = []
        for obs_tensor, msg_tensor, act_tensor, context_tensor in rollout["trajectory"]:
            critic_input = torch.cat([obs_tensor, msg_tensor, act_tensor, context_tensor], dim=-1)
            actor_values.append(self.critic(critic_input).squeeze(-1))
        actor_loss = -torch.stack(actor_values).mean()
        if rollout["entropies"]:
            actor_loss = actor_loss - self.config.entropy_coef * torch.stack(rollout["entropies"]).mean()
        if rollout["posterior_nlls"]:
            actor_loss = actor_loss + self.config.posterior_loss_coef * torch.stack(rollout["posterior_nlls"]).mean()
        if rollout["sender_std_terms"]:
            actor_loss = actor_loss + self.config.message_std_coef * torch.stack(rollout["sender_std_terms"]).mean()
        return actor_loss

    def _critic_loss(self, rollout: dict[str, Any], returns: torch.Tensor) -> torch.Tensor:
        critic_values = []
        for obs_tensor, msg_tensor, act_tensor, context_tensor in rollout["trajectory"]:
            critic_input = torch.cat(
                [obs_tensor.detach(), msg_tensor.detach(), act_tensor.detach(), context_tensor.detach()],
                dim=-1,
            )
            critic_values.append(self.critic(critic_input).squeeze(-1))
        critic_tensor = torch.stack(critic_values).squeeze(-1)
        return F.mse_loss(critic_tensor, returns)

    def _train_episode(self, episode_index: int) -> dict[str, Any]:
        assert self.policy_optimizer is not None
        rollout = self._rollout_episode(training=True, seed_offset=episode_index)
        returns = self._discounted_returns(rollout["train_step_rewards"])

        critic_loss = self._critic_loss(rollout, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad_(False)
        actor_loss = self._policy_loss(rollout)
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        policy_params = list(self.actors.parameters())
        policy_params.extend(list(self.mean_senders.parameters()))
        policy_params.extend(list(self.attn_senders.parameters()))
        policy_params.extend(list(self.directed_senders.parameters()))
        policy_params.extend(list(self.receivers.parameters()))
        if self.posterior is not None:
            policy_params.extend(list(self.posterior.parameters()))
        nn.utils.clip_grad_norm_(policy_params, self.config.grad_clip)
        self.policy_optimizer.step()
        for param in self.critic.parameters():
            param.requires_grad_(True)

        return {
            "episode_reward": rollout["episode_reward"],
            "loss": float((actor_loss + critic_loss).detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
        }

    def _fit_linear_probe(
        self,
        probe_messages: dict[str, list[np.ndarray]],
        probe_targets: dict[str, list[np.ndarray]],
    ) -> tuple[list[float], list[float]]:
        empirical_leakage = []
        posterior_error = []
        for agent in self.controlled_agents:
            message_batches = probe_messages.get(agent, [])
            target_batches = probe_targets.get(agent, [])
            if not message_batches or not target_batches:
                empirical_leakage.append(0.0)
                posterior_error.append(0.0)
                continue
            x_matrix = np.concatenate(message_batches, axis=0).reshape(len(message_batches), -1)
            y_vector = np.concatenate(target_batches, axis=0).reshape(len(target_batches), 1)
            x_aug = np.concatenate([x_matrix, np.ones((x_matrix.shape[0], 1), dtype=np.float32)], axis=1)
            coeffs, _, _, _ = np.linalg.lstsq(x_aug, y_vector, rcond=None)
            predictions = x_aug @ coeffs
            mae = float(np.mean(np.abs(predictions - y_vector)))
            mse = float(np.mean(np.square(predictions - y_vector)))
            baseline = float(np.mean(np.square(y_vector - y_vector.mean())))
            r2 = 0.0 if baseline <= 1e-8 else max(0.0, 1.0 - mse / baseline)
            empirical_leakage.append(r2)
            posterior_error.append(mae)
        return empirical_leakage, posterior_error

    @torch.no_grad()
    def evaluate(self, episode_index: int) -> dict[str, Any]:
        rewards = []
        message_norms = []
        kl_distortions = []
        uncertainty_values = []
        quality_values = []
        probe_messages = {agent: [] for agent in self.controlled_agents}
        probe_targets = {agent: [] for agent in self.controlled_agents}

        for offset in range(self.config.eval_episodes):
            rollout = self._rollout_episode(training=False, seed_offset=episode_index * 1000 + offset)
            rewards.append(rollout["episode_reward"])
            message_norms.append(np.asarray(rollout["message_norm"], dtype=np.float64))
            kl_distortions.append(np.asarray(rollout["kl_distortion"], dtype=np.float64))
            if rollout["posterior_uncertainties"]:
                uncertainty_values.append(float(torch.stack(rollout["posterior_uncertainties"]).mean().cpu().item()))
            if rollout["posterior_qualities"]:
                quality_values.append(float(torch.stack(rollout["posterior_qualities"]).mean().cpu().item()))
            for agent in self.controlled_agents:
                probe_messages[agent].extend(rollout["probe_messages"][agent])
                probe_targets[agent].extend(rollout["probe_targets"][agent])

        empirical_leakage, posterior_error = self._fit_linear_probe(probe_messages, probe_targets)
        current_sigma = self._sigma_vector()
        if np.any(current_sigma > 0.0):
            privacy_metrics = summarize_privacy(
                sigmas=current_sigma,
                total_rho_spent=self.total_rho_spent + self._block_rho(self.current_schedule),
                alpha=self.config.privacy_alpha,
                delta=self.config.delta,
            )
        else:
            zeros = np.zeros(len(self.controlled_agents), dtype=np.float64)
            privacy_metrics = {
                "sigma": zeros.tolist(),
                "total_rho_spent": zeros.tolist(),
                "epsilon": zeros.tolist(),
            }
        return {
            "average_episode_reward": float(np.mean(rewards)),
            "std_episode_reward": float(np.std(rewards)),
            "message_norm": np.mean(message_norms, axis=0).tolist(),
            "kl_distortion": np.mean(kl_distortions, axis=0).tolist(),
            "empirical_leakage": empirical_leakage,
            "posterior_error": posterior_error,
            "posterior_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
            "posterior_quality": float(np.mean(quality_values)) if quality_values else 0.0,
            "privacy": privacy_metrics,
        }

    def _advance_privacy_schedule(self, block_index: int, eval_info: dict[str, Any]) -> None:
        block_rho = self._block_rho(self.current_schedule)
        self.total_rho_spent += block_rho
        if self.privacy_scheduler is None:
            return
        self.privacy_scheduler.consume_budget(self.current_schedule.rho, block_length=self._block_length())
        if block_index + 1 >= self.privacy_scheduler.config.num_blocks:
            return
        team_reward = float(eval_info["average_episode_reward"])
        posterior_error = np.asarray(eval_info.get("posterior_error", [0.0]), dtype=np.float64)
        reward_gap = max(abs(team_reward) * 0.05, float(np.mean(posterior_error)), 1e-3)
        last_summary = {
            "posterior_uncertainty": np.full(
                len(self.controlled_agents),
                float(eval_info.get("posterior_uncertainty", 0.0)),
                dtype=np.float64,
            ),
            "allocation_error": posterior_error,
            "oracle_team_reward": team_reward + reward_gap,
            "team_reward": team_reward,
        }
        self.current_schedule = self.privacy_scheduler.next_schedule(block_index + 1, last_summary)

    def run(self, *, show_progress: bool = True, position: int = 0, leave_progress: bool = True) -> dict[str, Any]:
        desc = f"{self.spec['key'].upper()}-{self.algorithm.upper()}"
        progress = tqdm(
            range(self.config.episodes),
            desc=desc,
            unit="ep",
            dynamic_ncols=True,
            mininterval=0.1,
            smoothing=0.08,
            colour="#5C7AEA",
            position=position,
            leave=leave_progress,
            disable=not show_progress,
        )
        running_reward = None
        block_index = 0
        for episode_index in progress:
            train_info = self._train_episode(episode_index)
            running_reward = (
                train_info["episode_reward"]
                if running_reward is None
                else 0.9 * running_reward + 0.1 * train_info["episode_reward"]
            )
            if ((episode_index + 1) % self.config.eval_interval) == 0 or episode_index == self.config.episodes - 1:
                eval_info = self.evaluate(episode_index)
                entry = {
                    "episode": episode_index + 1,
                    "train_episode_reward": float(train_info["episode_reward"]),
                    "running_train_reward": float(running_reward),
                    "loss": float(train_info["loss"]),
                    "actor_loss": float(train_info["actor_loss"]),
                    "critic_loss": float(train_info["critic_loss"]),
                    "average_episode_reward": float(eval_info["average_episode_reward"]),
                    "std_episode_reward": float(eval_info["std_episode_reward"]),
                    "message_norm": eval_info["message_norm"],
                    "kl_distortion": eval_info["kl_distortion"],
                    "empirical_leakage": eval_info["empirical_leakage"],
                    "posterior_error": eval_info["posterior_error"],
                    "posterior_uncertainty": float(eval_info["posterior_uncertainty"]),
                    "posterior_quality": float(eval_info["posterior_quality"]),
                    "privacy": eval_info["privacy"],
                    "privacy_price": float(self.current_schedule.price),
                    "privacy_rho": np.asarray(self.current_schedule.rho, dtype=np.float64).tolist(),
                    "privacy_sigma": np.asarray(self.current_schedule.sigma, dtype=np.float64).tolist(),
                }
                self.history.append(entry)
                epsilon_mean = float(np.mean(entry["privacy"]["epsilon"])) if entry["privacy"]["epsilon"] else 0.0
                progress.set_postfix(
                    {
                        "ep": episode_index + 1,
                        "train": f"{train_info['episode_reward']:.2f}",
                        "eval": f"{eval_info['average_episode_reward']:.2f}",
                        "sig": f"{float(np.mean(entry['privacy_sigma'])):.2f}",
                        "eps": f"{epsilon_mean:.2f}",
                        "loss": f"{train_info['loss']:.3f}",
                    },
                    refresh=False,
                )
                self._advance_privacy_schedule(block_index, eval_info)
                block_index += 1
        if hasattr(progress, "close"):
            progress.close()
        return {
            "config": self.config.to_dict(),
            "scenario": self.spec["name"],
            "algorithm": self.algorithm,
            "adaptive_privacy": self.algorithm == "pil",
            "controlled_agents": self.controlled_agents,
            "history": self.history,
            "best": max(
                self.history,
                key=lambda entry: (
                    float(entry.get("average_episode_reward", -float("inf"))),
                    -float(np.mean(entry.get("privacy", {}).get("epsilon", [0.0]))),
                ),
            ) if self.history else {},
            "last": self.history[-1] if self.history else {},
            "checkpoint_selection": "last",
            "final": self.history[-1] if self.history else {},
        }

    @staticmethod
    def save_results(results: dict[str, Any], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
