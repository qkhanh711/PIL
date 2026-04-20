from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.models import ActorNet, PosteriorNet, SenderNet
from metrics.constraints import summarize_constraints
from metrics.privacy import rdp_gaussian_rho, summarize_privacy, rho_to_sigma

from tqdm.auto import tqdm



@dataclass
class PILConfig:
    seed: int = 7
    device: str = "cpu"
    num_agents: int = 3
    message_dim: int = 2
    hidden_dim: int = 64
    episode_length: int = 8
    num_blocks: int = 30
    inner_updates: int = 25
    train_batch_size: int = 64
    eval_batch_size: int = 256
    lr: float = 3e-3
    max_grad_norm: float = 5.0
    theta_low: float = 0.2
    theta_high: float = 1.0
    demand_low: float = 0.35
    demand_high: float = 0.95
    service_reward_weight: float = 2.5
    demand_weight: float = 3.0
    allocation_weight: float = 4.0
    effort_weight: float = 0.15
    decoder_loss_coef: float = 0.35
    sender_std_coef: float = 0.02
    kl_loss_coef: float = 0.015
    contract_temperature: float = 0.5
    contract_uncertainty_coef: float = 0.42
    contract_sigma_coef: float = 0.2
    contract_price_coef: float = 0.04
    theorem_price_coef: float = 0.1
    outer_layer_mode: str = "surrogate"
    transfer_scale: float = 1.0
    kalman_gain_floor: float = 1e-4
    kalman_process_noise: float = 1e-4
    clip_correction_coef: float = 0.75
    welfare_lipschitz: float = 1.0
    utility_lipschitz: float = 1.0
    posterior_mode: str = "kalman"
    privacy_alpha: float = 4.0
    delta: float = 1e-4
    sensitivity: float = 1.0
    total_rho_budget: float = 18.0
    rho_min: float = 0.05
    rho_max: float = 1.2
    lambda_init: float = 1.0
    lambda_max: float = 64.0
    benefit_init: float = 1.0
    uncertainty_weight: float = 1.0
    error_weight: float = 1.1
    reward_weight: float = 0.55
    frontload_factor: float = 0.3
    discount_gamma: float = 0.92
    type_cost_base: float = 1.0
    type_cost_scale: float = 0.5
    noise_std_min: float = 0.05
    noise_std_max: float = 3.0
    scheduler_obs_noise: float = 0.15
    scheduler_var_power: float = 1.5
    scheduler_refine_steps: int = 2
    fixed_baseline_fraction: float = 1.0
    ema_decay: float = 0.985
    privacy_block_length: int | None = None
    clip_multiplier: float = 1.2
    clip_floor: float = 0.15
    clip_ceiling: float = 3.0
    clip_margin: float = 0.1
    clip_margin_tail_coef: float = 1.5
    enforce_clip_margin_condition: bool = True
    scheduler_variant: str = "heuristic"
    scheduler_mode: str = "clip_la"

    @classmethod
    def from_namespace(cls, args: Any) -> "PILConfig":
        valid_keys = {field.name for field in fields(cls)}
        payload = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacySchedule:
    price: float
    rho: np.ndarray
    sigma: np.ndarray
    clip: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": float(self.price),
            "rho": np.asarray(self.rho, dtype=np.float64).tolist(),
            "sigma": np.asarray(self.sigma, dtype=np.float64).tolist(),
            "clip": np.asarray(self.clip, dtype=np.float64).tolist(),
        }

    def block_rho(self, block_length: int) -> np.ndarray:
        rho = np.asarray(self.rho, dtype=np.float64)
        if rho.ndim == 2:
            return rho.sum(axis=0)
        return rho * float(max(block_length, 0))

    def step_params(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.asarray(self.rho, dtype=np.float64)
        sigma = np.asarray(self.sigma, dtype=np.float64)
        clip = np.asarray(self.clip, dtype=np.float64)
        if rho.ndim == 2:
            return rho[step], sigma[step], clip[step]
        return rho, sigma, clip


class AdaptivePrivacyScheduler:
    """Outer privacy-pricing game for PIL."""

    def __init__(self, config: PILConfig) -> None:
        self.config = config
        self.num_agents = config.num_agents
        self.block_length = max(int(getattr(config, "privacy_block_length", 1) or 1), 1)
        type_span = np.linspace(0.0, 1.0, self.num_agents, dtype=np.float64)
        self.type_costs = config.type_cost_base * (1.0 + config.type_cost_scale * type_span)
        self.remaining_budget = np.full(self.num_agents, config.total_rho_budget, dtype=np.float64)
        self.current_benefits = np.full(self.num_agents, config.benefit_init, dtype=np.float64)
        self.current_lambda = float(config.lambda_init)
        self.nominal_rho = min(
            config.total_rho_budget / max(config.num_blocks * self.block_length, 1),
            config.rho_max,
        )
        self.nominal_total_budget = float(self.nominal_rho * self.num_agents)
        self.previous_rho = np.full(self.num_agents, self.nominal_rho, dtype=np.float64)
        self.previous_clip = np.full(self.num_agents, config.clip_multiplier, dtype=np.float64)
        self.posterior_var_state = np.ones(self.num_agents, dtype=np.float64)

    def _benefit_signal(self) -> float:
        return float(np.mean(self.current_benefits) / (1.0 + np.mean(self.current_benefits)))

    def _outer_layer_mode(self) -> str:
        return str(getattr(self.config, "outer_layer_mode", "surrogate"))

    def _nominal_price_anchor(self) -> float:
        target_rho = np.minimum(np.full(self.num_agents, self.nominal_rho, dtype=np.float64), self.remaining_budget)
        price_terms = (
            np.maximum(self.type_costs, 1e-8)
            * target_rho
            * (target_rho + 1.0)
            / np.maximum(self.current_benefits, 1e-8)
        )
        return float(np.mean(price_terms))

    def _best_response(self, price: float) -> np.ndarray:
        price = max(float(price), 0.0)
        scaled = price * np.maximum(self.current_benefits, 1e-8) / np.maximum(self.type_costs, 1e-8)
        rho = 0.5 * (np.sqrt(1.0 + 4.0 * scaled) - 1.0)
        rho = np.clip(rho, self.config.rho_min, self.config.rho_max)
        rho = np.minimum(rho, self.remaining_budget)
        rho[self.remaining_budget <= 1e-8] = 0.0
        return rho

    def _target_total_spend(self, block_index: int) -> float:
        remaining_total = float(np.sum(self.remaining_budget))
        if remaining_total <= 1e-8:
            return 0.0
        blocks_left = max(1, self.config.num_blocks - block_index)
        base_target = remaining_total / max(blocks_left * self.block_length, 1)
        signal = self._benefit_signal()
        frontload = 1.0 + 0.35 * self.config.frontload_factor * signal
        target = base_target * frontload
        max_total = float(np.sum(np.minimum(self.remaining_budget, self.config.rho_max)))
        stability_cap = self.nominal_total_budget * (1.0 + 0.20 * signal)
        target = min(target, max_total)
        target = min(target, stability_cap)
        target = max(target, 0.0)
        return float(target)

    def _clip_sequence(self, base_clip: np.ndarray) -> np.ndarray:
        horizon = self.block_length
        gamma = float(np.clip(self.config.discount_gamma, 1e-4, 0.9999))
        steps = np.arange(horizon, dtype=np.float64)
        scale = np.power(gamma, 0.5 * steps)[:, None]
        clip_seq = scale * base_clip[None, :]
        clip_seq = np.clip(clip_seq, self.config.clip_floor, self.config.clip_ceiling)
        return clip_seq

    def _margin_clip_floor(self, last_summary: dict[str, Any] | None) -> np.ndarray:
        if not self.config.enforce_clip_margin_condition:
            return np.full(self.num_agents, self.config.clip_floor, dtype=np.float64)
        message_bound = (
            np.asarray(last_summary.get("message_norm", self.previous_clip), dtype=np.float64)
            if last_summary
            else self.previous_clip
        )
        tail_margin = self.config.clip_margin_tail_coef * np.sqrt(self.config.message_dim * self.config.scheduler_obs_noise)
        safe_floor = np.maximum(message_bound, self.previous_clip) + tail_margin
        return np.clip(safe_floor, self.config.clip_floor, self.config.clip_ceiling)

    def _variance_path(self, prior_var: float, rho_seq: np.ndarray, clip_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        horizon = rho_seq.shape[0]
        vars_out = np.zeros(horizon, dtype=np.float64)
        a_out = np.zeros(horizon, dtype=np.float64)
        v = max(float(prior_var), 1e-6)
        for t in range(horizon):
            sigma_sq = 2.0 * self.config.privacy_alpha * max(float(clip_seq[t]) ** 2, 1e-8) / max(float(rho_seq[t]), 1e-8)
            a = self.config.scheduler_obs_noise + sigma_sq
            vars_out[t] = v
            a_out[t] = a
            v = (v * a) / max(v + a, 1e-8)
        return vars_out, a_out

    def _phi_from_variance(self, var_path: np.ndarray) -> np.ndarray:
        gamma = float(np.clip(self.config.discount_gamma, 1e-4, 0.9999))
        horizon = var_path.shape[0]
        phi = np.zeros(horizon, dtype=np.float64)
        running = 0.0
        for t in range(horizon - 1, -1, -1):
            running = (gamma * running) + max(float(var_path[t]), 1e-8) ** self.config.scheduler_var_power
            phi[t] = running
        return phi

    def _kkt_waterfill_agent(
        self,
        prior_var: float,
        clip_seq: np.ndarray,
        block_budget: float,
    ) -> np.ndarray:
        horizon = clip_seq.shape[0]
        if block_budget <= 1e-8:
            return np.zeros(horizon, dtype=np.float64)
        rho = np.full(horizon, max(block_budget / max(horizon, 1), self.config.rho_min), dtype=np.float64)
        rho = np.clip(rho, self.config.rho_min, self.config.rho_max)
        total = float(rho.sum())
        if total > block_budget and total > 1e-8:
            rho *= block_budget / total
        for _ in range(max(int(self.config.scheduler_refine_steps * 8), 8)):
            var_path, a_path = self._variance_path(prior_var, rho, clip_seq)
            phi = self._phi_from_variance(var_path)
            weights = 2.0 * np.maximum(clip_seq, 1e-8) * np.sqrt(np.maximum(phi, 1e-12)) / np.maximum(a_path, 1e-8)
            active = np.ones(horizon, dtype=bool)
            rho_new = np.zeros(horizon, dtype=np.float64)
            remaining_budget = block_budget
            while np.any(active):
                active_weights = weights[active]
                if float(active_weights.sum()) <= 1e-12:
                    rho_new[active] = remaining_budget / max(np.sum(active), 1)
                    break
                proposal = remaining_budget * active_weights / float(active_weights.sum())
                active_indices = np.where(active)[0]
                clipped_any = False
                for idx, value in zip(active_indices, proposal):
                    if value < self.config.rho_min:
                        rho_new[idx] = self.config.rho_min
                        remaining_budget -= self.config.rho_min
                        active[idx] = False
                        clipped_any = True
                    elif value > self.config.rho_max:
                        rho_new[idx] = self.config.rho_max
                        remaining_budget -= self.config.rho_max
                        active[idx] = False
                        clipped_any = True
                if not clipped_any:
                    rho_new[active] = proposal
                    break
                if remaining_budget <= 1e-8:
                    break
            rho_new = np.clip(rho_new, 0.0, self.config.rho_max)
            total_new = float(rho_new.sum())
            if total_new > block_budget and total_new > 1e-8:
                rho_new *= block_budget / total_new
            if np.max(np.abs(rho_new - rho)) < 1e-5:
                rho = rho_new
                break
            rho = rho_new
        return rho

    def _stabilize_rho(self, rho: np.ndarray, block_index: int) -> np.ndarray:
        signal = self._benefit_signal()
        blocks_left = max(1, self.config.num_blocks - block_index)
        fair_share_cap = (self.remaining_budget / max(blocks_left * self.block_length, 1)) * (1.0 + 0.25 * signal)
        nominal_cap = np.full(self.num_agents, self.nominal_rho * (1.0 + 0.20 * signal), dtype=np.float64)
        upward_cap = self.previous_rho * (1.05 + 0.15 * signal)
        cap = np.minimum.reduce(
            [
                self.remaining_budget,
                np.maximum(fair_share_cap, 0.0),
                np.maximum(nominal_cap, 0.0),
                np.maximum(upward_cap, 0.0),
            ]
        )
        stabilized = np.minimum(rho, cap)
        stabilized = np.minimum(stabilized, self.remaining_budget)
        return np.clip(stabilized, 0.0, self.config.rho_max)

    def _solve_price(self, target_total: float) -> tuple[float, np.ndarray]:
        if target_total <= 1e-8:
            return 0.0, np.zeros(self.num_agents, dtype=np.float64)
        max_total = float(np.sum(np.minimum(self.remaining_budget, self.config.rho_max)))
        target_total = min(target_total, max_total)
        low = 0.0
        high = max(self.current_lambda, self.config.lambda_init, 1.0)
        rho_high = self._best_response(high)
        while float(np.sum(rho_high)) < target_total and high < self.config.lambda_max:
            high = min(high * 2.0, self.config.lambda_max)
            rho_high = self._best_response(high)
        for _ in range(40):
            mid = 0.5 * (low + high)
            rho_mid = self._best_response(mid)
            if float(np.sum(rho_mid)) >= target_total:
                high = mid
            else:
                low = mid
        rho = self._best_response(high)
        total = float(np.sum(rho))
        if total > target_total and total > 1e-8:
            rho = rho * (target_total / total)
            rho = np.minimum(rho, self.remaining_budget)
            rho = np.clip(rho, 0.0, self.config.rho_max)
        return float(high), rho

    def _update_benefits(self, last_summary: dict[str, Any] | None) -> None:
        if not last_summary:
            return
        uncertainty = np.asarray(last_summary["posterior_uncertainty"], dtype=np.float64)
        allocation_error = np.asarray(last_summary["allocation_error"], dtype=np.float64)
        reward_gap = max(float(last_summary["oracle_team_reward"]) - float(last_summary["team_reward"]), 0.0)
        self.current_benefits = (
            self.config.benefit_init
            + self.config.uncertainty_weight * uncertainty
            + self.config.error_weight * allocation_error
            + self.config.reward_weight * reward_gap
        )

    def next_schedule(self, block_index: int, last_summary: dict[str, Any] | None) -> PrivacySchedule:
        self._update_benefits(last_summary)
        uncertainty = (
            np.asarray(last_summary["posterior_uncertainty"], dtype=np.float64)
            if last_summary
            else np.ones(self.num_agents, dtype=np.float64)
        )
        self.posterior_var_state = np.maximum(uncertainty, 1e-6)
        proxy_var = np.maximum(uncertainty + self.config.clip_margin, 1e-6)
        base_clip = self.config.clip_multiplier * np.sqrt(self.config.message_dim * proxy_var)
        base_clip = np.maximum(base_clip, self._margin_clip_floor(last_summary))
        base_clip = np.clip(base_clip, self.config.clip_floor, self.config.clip_ceiling)
        base_clip = np.minimum(base_clip, self.previous_clip * 1.25 + self.config.clip_floor)
        clip = self._clip_sequence(base_clip)
        if self.config.scheduler_variant == "exact_wf":
            blocks_left = max(1, self.config.num_blocks - block_index)
            block_budget_per_agent = self.remaining_budget / float(blocks_left * self.block_length)
            block_budget_per_agent = np.clip(block_budget_per_agent, 0.0, self.config.rho_max)
            raw_price = 0.0
            guide_price = 0.0
        else:
            target_total = self._target_total_spend(block_index)
            raw_price, block_budget_per_agent = self._solve_price(target_total)
            block_budget_per_agent = np.asarray(block_budget_per_agent, dtype=np.float64).reshape(-1)
            signal = self._benefit_signal()
            price_cap = max(
                self.config.lambda_init,
                self._nominal_price_anchor(),
                self.current_lambda * (1.10 + 0.20 * signal),
            )
            guide_price = min(raw_price, price_cap, self.config.lambda_max)
        rho = np.zeros((self.block_length, self.num_agents), dtype=np.float64)
        block_budgets = block_budget_per_agent * float(self.block_length)
        for agent_idx in range(self.num_agents):
            budget_cap = min(float(block_budgets[agent_idx]), float(self.remaining_budget[agent_idx]))
            rho[:, agent_idx] = self._kkt_waterfill_agent(
                float(self.posterior_var_state[agent_idx]),
                clip[:, agent_idx],
                budget_cap,
            )
        if float(np.sum(rho)) > 1e-8:
            mean_rho = rho.mean(axis=0)
            implementing_price = self.type_costs * mean_rho * (mean_rho + 1.0) / np.maximum(self.current_benefits, 1e-8)
            if self._outer_layer_mode() == "theorem":
                base_price = raw_price if self.config.scheduler_variant != "exact_wf" else float(np.mean(implementing_price))
                price = float(np.clip(base_price, 0.0, self.config.lambda_max))
            else:
                price = float(np.clip(np.mean(implementing_price), self.config.lambda_init, self.config.lambda_max))
        else:
            price = 0.0
        if (
            self._outer_layer_mode() != "theorem"
            and self.config.scheduler_variant != "exact_wf"
            and guide_price > 0.0
            and price > 0.0
        ):
            price = float(np.clip(0.5 * price + 0.5 * guide_price, self.config.lambda_init, self.config.lambda_max))
        if self.config.scheduler_variant != "exact_wf":
            stabilized = self._stabilize_rho(rho.mean(axis=0), block_index)
            original_mean = np.maximum(rho.mean(axis=0), 1e-8)
            rho *= (stabilized / original_mean)[None, :]
        sigma = rho_to_sigma(
            rho,
            alpha=self.config.privacy_alpha,
            clip_radius=clip,
            sigma_min=self.config.noise_std_min,
            sigma_max=self.config.noise_std_max,
        )
        self.current_lambda = price
        self.previous_rho = np.asarray(rho.mean(axis=0), dtype=np.float64)
        self.previous_clip = np.asarray(base_clip, dtype=np.float64)
        return PrivacySchedule(price=price, rho=rho, sigma=sigma, clip=clip)

    def consume_budget(self, rho: np.ndarray, *, block_length: int = 1) -> None:
        rho_arr = np.asarray(rho, dtype=np.float64)
        if rho_arr.ndim == 2:
            spent = rho_arr.sum(axis=0)
        else:
            spent = rho_arr * float(max(block_length, 0))
        self.remaining_budget = np.maximum(self.remaining_budget - spent, 0.0)


class BaseCommunicationTrainer:
    """Differentiable cooperative communication trainer used for PIL and all baselines."""

    def __init__(
        self,
        config: PILConfig,
        *,
        adaptive_privacy: bool,
        use_transfers: bool,
        use_messages: bool = True,
        deterministic_messages: bool = False,
        use_contract: bool = True,
        privacy_mode: str | None = None,
        baseline_name: str | None = None,
    ) -> None:
        self.config = config
        self.adaptive_privacy = adaptive_privacy
        self.use_transfers = use_transfers
        self.use_messages = use_messages
        self.deterministic_messages = deterministic_messages
        self.use_contract = use_contract
        self.privacy_mode = "adaptive" if adaptive_privacy else (privacy_mode or "fixed")
        self.baseline_name = baseline_name
        self.use_privacy_planner = adaptive_privacy and use_messages and use_contract
        if self.config.privacy_block_length is None:
            self.config.privacy_block_length = self.config.episode_length
        self._set_seed(config.seed)
        self.device = torch.device(config.device if config.device else "cpu")

        self.senders = nn.ModuleList(
            [SenderNet(2, config.hidden_dim, config.message_dim) for _ in range(config.num_agents)]
        ).to(self.device)
        posterior_input_dim = 1 + config.num_agents * config.message_dim + config.num_agents
        self.posterior = PosteriorNet(posterior_input_dim, config.hidden_dim, config.num_agents).to(self.device)
        actor_input_dim = 7 if self.use_privacy_planner else 3
        self.actors = nn.ModuleList(
            [ActorNet(actor_input_dim, config.hidden_dim) for _ in range(config.num_agents)]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        self.ema_decay = float(config.ema_decay)
        self.ema_shadow: dict[str, torch.Tensor] | None = None
        if 0.0 < self.ema_decay < 1.0:
            self.ema_shadow = self._capture_state()

        type_span = torch.linspace(0.0, 1.0, config.num_agents, device=self.device)
        self.type_costs = config.type_cost_base * (1.0 + config.type_cost_scale * type_span)
        self.scheduler = AdaptivePrivacyScheduler(config) if self.privacy_mode == "adaptive" else None
        self.fixed_schedule = None if self.privacy_mode == "adaptive" else self._build_fixed_schedule()
        self.total_rho_spent = np.zeros(config.num_agents, dtype=np.float64)
        self.total_claimed_runtime_rho_spent = np.zeros(config.num_agents, dtype=np.float64)
        self.total_unclipped_runtime_rho_spent = np.zeros(config.num_agents, dtype=np.float64)
        self.history: list[dict[str, Any]] = []

    def parameters(self) -> list[nn.Parameter]:
        params = list(self.senders.parameters()) + list(self.actors.parameters())
        if self.config.posterior_mode == "network":
            params += list(self.posterior.parameters())
        return params

    def _named_module_groups(self) -> dict[str, nn.Module]:
        return {
            "senders": self.senders,
            "posterior": self.posterior,
            "actors": self.actors,
        }

    def _capture_state(self) -> dict[str, torch.Tensor]:
        snapshot: dict[str, torch.Tensor] = {}
        for group_name, module in self._named_module_groups().items():
            for key, value in module.state_dict().items():
                snapshot[f"{group_name}.{key}"] = value.detach().clone()
        return snapshot

    def _load_state(self, state: dict[str, torch.Tensor]) -> None:
        for group_name, module in self._named_module_groups().items():
            prefix = f"{group_name}."
            group_state = {
                key[len(prefix) :]: value.clone()
                for key, value in state.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(group_state, strict=True)

    def _update_ema(self) -> None:
        if self.ema_shadow is None:
            return
        current_state = self._capture_state()
        for key, value in current_state.items():
            shadow = self.ema_shadow[key]
            shadow.mul_(self.ema_decay).add_(value, alpha=1.0 - self.ema_decay)

    @contextmanager
    def _ema_scope(self):
        if self.ema_shadow is None:
            yield
            return
        current_state = self._capture_state()
        self._load_state(self.ema_shadow)
        try:
            yield
        finally:
            self._load_state(current_state)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_fixed_schedule(self) -> PrivacySchedule:
        if self.privacy_mode == "none":
            rho = np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64)
            sigma = np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64)
            clip = np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64)
        else:
            block_length = max(int(self.config.privacy_block_length or self.config.episode_length), 1)
            base_rho = (
                self.config.total_rho_budget / max(self.config.num_blocks * block_length, 1)
            ) * self.config.fixed_baseline_fraction
            rho = np.full((self.config.episode_length, self.config.num_agents), base_rho, dtype=np.float64)
            rho = np.clip(rho, self.config.rho_min, self.config.rho_max)
            clip = np.full((self.config.episode_length, self.config.num_agents), self.config.clip_multiplier, dtype=np.float64)
            sigma = rho_to_sigma(
                rho,
                alpha=self.config.privacy_alpha,
                clip_radius=clip,
                sigma_min=self.config.noise_std_min,
                sigma_max=self.config.noise_std_max,
            )
        return PrivacySchedule(price=0.0, rho=rho, sigma=sigma, clip=clip)

    def _block_rho(self, schedule: PrivacySchedule) -> np.ndarray:
        return schedule.block_rho(self.config.episode_length)

    def _sample_context(self, batch_size: int) -> dict[str, torch.Tensor]:
        types = torch.rand(batch_size, self.config.num_agents, device=self.device)
        types = self.config.theta_low + (self.config.theta_high - self.config.theta_low) * types
        demands = torch.rand(batch_size, self.config.episode_length, 1, device=self.device)
        demands = self.config.demand_low + (self.config.demand_high - self.config.demand_low) * demands
        return {"types": types, "demands": demands}

    def _decoder_input(self, messages: torch.Tensor, demand: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_features = sigma.squeeze(-1).expand(demand.shape[0], -1)
        return torch.cat([demand, messages.reshape(demand.shape[0], -1), sigma_features], dim=1)

    def _kalman_posterior_step(
        self,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
        message_tensor: torch.Tensor,
        sender_mean_tensor: torch.Tensor,
        sender_std_tensor: torch.Tensor,
        privacy_sigma: torch.Tensor,
        clip_radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        message_scalar = message_tensor.mean(dim=-1)
        sender_mean_scalar = sender_mean_tensor.mean(dim=-1)
        sender_var_scalar = sender_std_tensor.pow(2).mean(dim=-1)
        privacy_var = privacy_sigma.squeeze(-1).pow(2)
        obs_var = (sender_var_scalar + privacy_var).clamp_min(self.config.kalman_gain_floor)
        mean_norm = sender_mean_tensor.norm(dim=-1)
        raw_excess = torch.clamp(mean_norm - clip_radius.squeeze(-1), min=0.0)
        subgaussian_scale = sender_std_tensor.norm(dim=-1).clamp_min(self.config.kalman_gain_floor)
        tail_factor = torch.exp(-0.5 * torch.square(raw_excess / subgaussian_scale))
        bias_bound = self.config.clip_correction_coef * raw_excess * tail_factor
        bias_direction = torch.sign(sender_mean_scalar - prior_mean)
        clipped_bias = bias_direction * bias_bound
        innovation = message_scalar - sender_mean_scalar
        centered_obs = prior_mean + innovation - clipped_bias
        gain = prior_var / (prior_var + obs_var).clamp_min(self.config.kalman_gain_floor)
        posterior_mean = prior_mean + gain * (centered_obs - prior_mean)
        posterior_var = (1.0 - gain) * prior_var + self.config.kalman_process_noise
        return posterior_mean.clamp(self.config.theta_low, self.config.theta_high), posterior_var.clamp(min=1e-4, max=5.0)

    def _posterior_step(
        self,
        demand: torch.Tensor,
        planner_messages: torch.Tensor,
        sender_means: torch.Tensor,
        sender_stds: torch.Tensor,
        sigma: torch.Tensor,
        clip: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.posterior_mode == "network":
            posterior_mean, posterior_log_var = self.posterior(
                self._decoder_input(planner_messages, demand, sigma)
            )
            posterior_mean = self.config.theta_low + (
                self.config.theta_high - self.config.theta_low
            ) * posterior_mean
            posterior_var = posterior_log_var.exp().clamp(min=1e-4, max=5.0)
            return posterior_mean, posterior_var
        return self._kalman_posterior_step(
            prior_mean,
            prior_var,
            planner_messages,
            sender_means,
            sender_stds,
            sigma,
            clip,
        )

    def _contract_from_posterior(
        self,
        posterior_mean: torch.Tensor,
        demand: torch.Tensor,
        posterior_var: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        *,
        price: float = 0.0,
    ) -> torch.Tensor:
        outer_layer_mode = str(getattr(self.config, "outer_layer_mode", "surrogate"))
        if outer_layer_mode == "theorem":
            score = posterior_mean.clamp_min(0.0)
            if posterior_var is not None:
                score = score - self.config.contract_uncertainty_coef * posterior_var
            if sigma is not None:
                sigma_batch = sigma.expand(posterior_mean.shape[0], -1, -1)
                score = score - self.config.contract_sigma_coef * sigma_batch.squeeze(-1)
            score = score - self.config.theorem_price_coef * float(price)
            score = score.clamp_min(0.0)
            shares = score / score.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return demand * shares
        centered = posterior_mean
        if posterior_var is not None:
            centered = centered - self.config.contract_uncertainty_coef * posterior_var
        if sigma is not None:
            sigma_batch = sigma.expand(posterior_mean.shape[0], -1, -1)
            centered = centered - self.config.contract_sigma_coef * sigma_batch.squeeze(-1)
        centered = centered - self.config.contract_price_coef * float(price)
        centered = centered / max(self.config.contract_temperature, 1e-6)
        shares = F.softmax(centered, dim=1)
        return demand * shares

    def _actor_input(
        self,
        agent_idx: int,
        types: torch.Tensor,
        demand: torch.Tensor,
        contract: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_var: torch.Tensor,
        planner_messages: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_privacy_planner:
            return torch.cat(
                [types[:, agent_idx : agent_idx + 1], demand, contract[:, agent_idx : agent_idx + 1]],
                dim=1,
            )
        message_energy = planner_messages[:, agent_idx, :].norm(dim=-1, keepdim=True)
        sigma_batch = sigma.expand(types.shape[0], -1, -1)
        return torch.cat(
            [
                types[:, agent_idx : agent_idx + 1],
                demand,
                contract[:, agent_idx : agent_idx + 1],
                posterior_mean[:, agent_idx : agent_idx + 1],
                posterior_var[:, agent_idx : agent_idx + 1],
                message_energy,
                sigma_batch[:, agent_idx, :],
            ],
            dim=1,
        )

    def _planner_action(
        self,
        agent_idx: int,
        actor_output: torch.Tensor,
        contract: torch.Tensor,
        posterior_var: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_privacy_planner:
            return actor_output
        if str(getattr(self.config, "outer_layer_mode", "surrogate")) == "theorem":
            return contract[:, agent_idx : agent_idx + 1]
        sigma_batch = sigma.expand(actor_output.shape[0], -1, -1)
        uncertainty = posterior_var[:, agent_idx : agent_idx + 1]
        noise_level = sigma_batch[:, agent_idx, :]
        actor_weight = 0.95 - 0.60 * torch.sigmoid(1.5 * (uncertainty + noise_level - 1.0))
        actor_weight = actor_weight.clamp(0.25, 0.95)
        return actor_weight * actor_output + (1.0 - actor_weight) * contract[:, agent_idx : agent_idx + 1]

    def _simulate_batch(
        self,
        schedule: PrivacySchedule,
        *,
        batch_size: int,
        context: dict[str, torch.Tensor] | None = None,
        use_transfers: bool | None = None,
        deviating_agent: int | None = None,
        oracle_contract: bool = False,
    ) -> dict[str, Any]:
        use_transfers = self.use_transfers if use_transfers is None else use_transfers
        context = self._sample_context(batch_size) if context is None else context
        types = context["types"]
        demands = context["demands"]
        prior_mean = torch.full_like(types, 0.5 * (self.config.theta_low + self.config.theta_high))
        prior_var = torch.full_like(
            types,
            ((self.config.theta_high - self.config.theta_low) ** 2) / 12.0,
        )

        team_reward_sum = torch.zeros(1, device=self.device)
        oracle_team_reward_sum = torch.zeros(1, device=self.device)
        modified_reward_sum = torch.zeros(1, device=self.device)
        posterior_nll_sum = torch.zeros(1, device=self.device)
        sender_std_sum = torch.zeros(1, device=self.device)
        agent_utility_sum = torch.zeros(self.config.num_agents, device=self.device)
        quality_sum = torch.zeros(self.config.num_agents, device=self.device)
        uncertainty_sum = torch.zeros(self.config.num_agents, device=self.device)
        allocation_error_sum = torch.zeros(self.config.num_agents, device=self.device)
        message_norm_sum = torch.zeros(self.config.num_agents, device=self.device)
        distortion_sum = torch.zeros(self.config.num_agents, device=self.device)
        kl_step_records: list[torch.Tensor] = []
        posterior_error_sum = torch.zeros(self.config.num_agents, device=self.device)
        clip_rate_sum = torch.zeros(self.config.num_agents, device=self.device)
        latent_sensitivity_sum = torch.zeros(self.config.num_agents, device=self.device)
        clipped_runtime_rho_sum = torch.zeros(self.config.num_agents, device=self.device)
        unclipped_runtime_rho_sum = torch.zeros(self.config.num_agents, device=self.device)

        for step in range(self.config.episode_length):
            demand = demands[:, step]
            step_rho_np, step_sigma_np, step_clip_np = schedule.step_params(step)
            sigma = torch.tensor(step_sigma_np, dtype=torch.float32, device=self.device).view(1, self.config.num_agents, 1)
            rho = torch.tensor(step_rho_np, dtype=torch.float32, device=self.device).view(1, self.config.num_agents)
            clip = torch.tensor(step_clip_np, dtype=torch.float32, device=self.device).view(1, self.config.num_agents, 1)
            if self.use_messages:
                messages = []
                sender_means = []
                intrinsic_stds = []
                for agent_idx, sender in enumerate(self.senders):
                    sender_input = torch.cat([types[:, agent_idx : agent_idx + 1], demand], dim=1)
                    mean, std = sender(sender_input)
                    sender_means.append(mean)
                    if self.deterministic_messages:
                        latent = mean
                        intrinsic_std = torch.full_like(std, 1e-4)
                    else:
                        latent = mean + std * torch.randn_like(std)
                        intrinsic_std = std
                    latent_norm = latent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    clip_scale = torch.clamp(clip[:, agent_idx, :] / latent_norm, max=1.0)
                    clipped_latent = latent * clip_scale
                    private_message = clipped_latent if self.config.scheduler_mode != "naive_la" else latent
                    if np.any(schedule.sigma > 0.0):
                        private_message = private_message + sigma[:, agent_idx, :] * torch.randn_like(latent)
                    if deviating_agent == agent_idx:
                        private_message = torch.zeros_like(private_message)
                    messages.append(private_message)
                    intrinsic_stds.append(intrinsic_std)
                    clip_rate_sum[agent_idx] += (latent_norm.squeeze(-1) > clip[:, agent_idx, :].squeeze(-1)).float().mean()
                    clipped_step_sensitivity = (2.0 * clip[:, agent_idx, :].squeeze(-1)).mean()
                    unclipped_step_sensitivity = 2.0 * latent_norm.max()
                    latent_sensitivity_sum[agent_idx] += unclipped_step_sensitivity
                    sigma_sq = torch.square(sigma[:, agent_idx, :].mean().clamp_min(1e-8))
                    clipped_runtime_rho_sum[agent_idx] += self.config.privacy_alpha * torch.square(
                        clipped_step_sensitivity
                    ) / (2.0 * sigma_sq)
                    unclipped_runtime_rho_sum[agent_idx] += self.config.privacy_alpha * torch.square(
                        unclipped_step_sensitivity
                    ) / (2.0 * sigma_sq)

                message_tensor = torch.stack(messages, dim=1)
                sender_mean_tensor = torch.stack(sender_means, dim=1)
                std_tensor = torch.stack(intrinsic_stds, dim=1)
            else:
                message_tensor = torch.zeros(
                    batch_size,
                    self.config.num_agents,
                    self.config.message_dim,
                    device=self.device,
                )
                sender_mean_tensor = torch.zeros_like(message_tensor)
                std_tensor = torch.full_like(message_tensor, 1e-4)

            planner_message_tensor = message_tensor

            if oracle_contract:
                posterior_mean = types
                posterior_var = torch.full_like(types, 1e-4)
            elif self.use_contract and self.use_messages:
                posterior_mean, posterior_var = self._posterior_step(
                    demand,
                    planner_message_tensor,
                    sender_mean_tensor,
                    std_tensor,
                    sigma,
                    clip,
                    prior_mean,
                    prior_var,
                )
            else:
                posterior_mean = prior_mean
                posterior_var = prior_var

            true_shares = types / types.sum(dim=1, keepdim=True).clamp_min(1e-8)
            oracle_contract_tensor = demand * true_shares
            if oracle_contract:
                contract = oracle_contract_tensor
            elif self.use_contract:
                contract = self._contract_from_posterior(
                    posterior_mean,
                    demand,
                    posterior_var,
                    sigma,
                    price=schedule.price,
                )
            else:
                contract = torch.zeros_like(oracle_contract_tensor)

            if oracle_contract:
                action_tensor = oracle_contract_tensor.clone()
            else:
                actions = []
                for agent_idx, actor in enumerate(self.actors):
                    actor_input = self._actor_input(
                        agent_idx,
                        types,
                        demand,
                        contract,
                        posterior_mean,
                        posterior_var,
                        planner_message_tensor,
                        sigma,
                    )
                    actor_output = actor(actor_input)
                    actions.append(
                        self._planner_action(agent_idx, actor_output, contract, posterior_var, sigma)
                    )
                action_tensor = torch.cat(actions, dim=1)

            demand_gap = (action_tensor.sum(dim=1, keepdim=True) - demand).pow(2)
            allocation_gap = (action_tensor - oracle_contract_tensor).pow(2).sum(dim=1, keepdim=True)
            team_reward = (
                self.config.service_reward_weight * demand.squeeze(1)
                - self.config.demand_weight * demand_gap.squeeze(1)
                - self.config.allocation_weight * allocation_gap.squeeze(1)
            )
            oracle_team_reward = self.config.service_reward_weight * demand.squeeze(1)

            effort_cost = self.config.effort_weight * action_tensor.pow(2) / (types + 0.1)
            quality = torch.exp(-((posterior_mean - types).pow(2) / (posterior_var + 1e-4))).clamp(min=0.0, max=1.0)
            privacy_charge = 0.5 * self.type_costs.view(1, -1) * rho.pow(2) / max(self.config.episode_length, 1)
            transfers = self.config.transfer_scale * (schedule.price * quality * contract - privacy_charge)
            if not use_transfers:
                transfers = torch.zeros_like(transfers)

            utilities = team_reward.unsqueeze(1) + transfers - effort_cost
            posterior_nll = 0.5 * (
                torch.log(2.0 * torch.pi * posterior_var)
                + (types - posterior_mean).pow(2) / posterior_var
            )

            sigma_squared = sigma.expand(batch_size, -1, self.config.message_dim).pow(2)
            base_var = std_tensor.pow(2).clamp(min=1e-8)
            private_var = base_var + sigma_squared
            kl_distortion = 0.5 * torch.sum(private_var / base_var - 1.0 - torch.log(private_var / base_var), dim=-1)
            kl_step_records.append(kl_distortion.mean(dim=0))

            team_reward_sum += team_reward.mean()
            oracle_team_reward_sum += oracle_team_reward.mean()
            modified_reward_sum += utilities.mean()
            posterior_nll_sum += posterior_nll.mean()
            sender_std_sum += std_tensor.mean()
            agent_utility_sum += utilities.mean(dim=0)
            quality_sum += quality.mean(dim=0)
            uncertainty_sum += posterior_var.mean(dim=0)
            allocation_error_sum += torch.abs(contract - oracle_contract_tensor).mean(dim=0)
            message_norm_sum += message_tensor.norm(dim=-1).mean(dim=0)
            distortion_sum += kl_distortion.mean(dim=0)
            posterior_error_sum += torch.abs(posterior_mean - types).mean(dim=0)
            prior_mean = posterior_mean.detach()
            prior_var = posterior_var.detach()

        divisor = float(self.config.episode_length)
        modified_reward = modified_reward_sum / divisor
        posterior_nll = posterior_nll_sum / divisor
        sender_std = sender_std_sum / divisor
        mean_kl = distortion_sum.mean() / divisor
        objective = (
            modified_reward
            - self.config.decoder_loss_coef * posterior_nll
            - self.config.sender_std_coef * sender_std
            - self.config.kl_loss_coef * mean_kl
        )

        return {
            "objective": objective.squeeze(0),
            "team_reward_tensor": team_reward_sum / divisor,
            "oracle_team_reward_tensor": oracle_team_reward_sum / divisor,
            "modified_reward_tensor": modified_reward,
            "posterior_nll_tensor": posterior_nll,
            "agent_utility_tensor": agent_utility_sum / divisor,
            "quality_tensor": quality_sum / divisor,
            "posterior_uncertainty_tensor": uncertainty_sum / divisor,
            "allocation_error_tensor": allocation_error_sum / divisor,
            "message_norm_tensor": message_norm_sum / divisor,
            "distortion_tensor": distortion_sum / divisor,
            "kl_step_tensor": torch.stack(kl_step_records, dim=0),
            "posterior_error_tensor": posterior_error_sum / divisor,
            "clip_rate_tensor": clip_rate_sum / divisor,
            "latent_sensitivity_tensor": latent_sensitivity_sum,
            "claimed_runtime_rho_tensor": clipped_runtime_rho_sum,
            "unclipped_runtime_rho_tensor": unclipped_runtime_rho_sum,
        }

    def _detach_metrics(self, raw_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "team_reward": float(raw_metrics["team_reward_tensor"].detach().cpu().item()),
            "oracle_team_reward": float(raw_metrics["oracle_team_reward_tensor"].detach().cpu().item()),
            "modified_reward": float(raw_metrics["modified_reward_tensor"].detach().cpu().item()),
            "posterior_nll": float(raw_metrics["posterior_nll_tensor"].detach().cpu().item()),
            "avg_agent_utility": raw_metrics["agent_utility_tensor"].detach().cpu().numpy().tolist(),
            "quality": raw_metrics["quality_tensor"].detach().cpu().numpy().tolist(),
            "posterior_uncertainty": raw_metrics["posterior_uncertainty_tensor"].detach().cpu().numpy().tolist(),
            "allocation_error": raw_metrics["allocation_error_tensor"].detach().cpu().numpy().tolist(),
            "message_norm": raw_metrics["message_norm_tensor"].detach().cpu().numpy().tolist(),
            "kl_distortion": raw_metrics["distortion_tensor"].detach().cpu().numpy().tolist(),
            "kl_step_distortion": raw_metrics["kl_step_tensor"].detach().cpu().numpy().tolist(),
            "posterior_error": raw_metrics["posterior_error_tensor"].detach().cpu().numpy().tolist(),
            "clip_rate": raw_metrics["clip_rate_tensor"].detach().cpu().numpy().tolist(),
            "latent_sensitivity": raw_metrics["latent_sensitivity_tensor"].detach().cpu().numpy().tolist(),
            "claimed_runtime_rho": raw_metrics["claimed_runtime_rho_tensor"].detach().cpu().numpy().tolist(),
            "unclipped_runtime_rho": raw_metrics["unclipped_runtime_rho_tensor"].detach().cpu().numpy().tolist(),
        }

    def select_schedule(self, block_index: int, last_summary: dict[str, Any] | None) -> PrivacySchedule:
        if self.adaptive_privacy:
            assert self.scheduler is not None
            return self.scheduler.next_schedule(block_index, last_summary)
        assert self.fixed_schedule is not None
        return self.fixed_schedule

    def _post_block_update(self, schedule: PrivacySchedule) -> None:
        block_rho = self._block_rho(schedule)
        self.total_rho_spent += block_rho
        if self.scheduler is not None:
            self.scheduler.consume_budget(schedule.rho, block_length=self.config.episode_length)

    def _progress_desc(self, run_label: str | None = None) -> str:
        if run_label is not None:
            return run_label
        if self.baseline_name is not None:
            return self.baseline_name
        return "PIL-APS" if self.adaptive_privacy else "DPMAC"

    def _progress_colour(self) -> str:
        if self.baseline_name == "MADDPG":
            return "#D1495B"
        if self.baseline_name == "I2C":
            return "#4F772D"
        return "#2E86AB" if self.adaptive_privacy else "#F18F01"

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
            "rho": f"{np.mean(np.asarray(schedule.rho)):.2f}",
            "sig": f"{np.mean(np.asarray(schedule.sigma)):.2f}",
        }
        if loss is not None:
            postfix["loss"] = f"{loss:.3f}"
        if summary is not None:
            postfix["rew"] = f"{summary['team_reward']:.3f}"
            postfix["reg"] = f"{summary['welfare_regret']:.3f}"
        return postfix

    @staticmethod
    def _epsilon_mean(entry: dict[str, Any]) -> float:
        epsilons = entry.get("privacy", {}).get("epsilon", [])
        return float(np.mean(epsilons)) if epsilons else 0.0

    def _select_final_entry(self) -> dict[str, Any]:
        if not self.history:
            return {}
        if not self.adaptive_privacy:
            return self.history[-1]
        return max(
            self.history,
            key=lambda entry: (
                float(entry["team_reward"]),
                -float(entry["welfare_regret"]),
                -self._epsilon_mean(entry),
            ),
        )

    @torch.no_grad()
    def evaluate_block(self, schedule: PrivacySchedule, block_index: int) -> dict[str, Any]:
        context = self._sample_context(self.config.eval_batch_size)
        truthful = self._detach_metrics(self._simulate_batch(schedule, batch_size=self.config.eval_batch_size, context=context))
        deviating_utilities = np.asarray(truthful["avg_agent_utility"], dtype=np.float64)
        for agent_idx in range(self.config.num_agents):
            deviating = self._detach_metrics(
                self._simulate_batch(
                    schedule,
                    batch_size=self.config.eval_batch_size,
                    context=context,
                    deviating_agent=agent_idx,
                )
            )
            deviating_utilities[agent_idx] = float(deviating["avg_agent_utility"][agent_idx])

        non_private_schedule = PrivacySchedule(
            price=0.0,
            rho=np.zeros(self.config.num_agents, dtype=np.float64),
            sigma=np.zeros(self.config.num_agents, dtype=np.float64),
            clip=np.zeros(self.config.num_agents, dtype=np.float64),
        )
        if self.privacy_mode == "none" or not self.use_messages:
            non_private_schedule = PrivacySchedule(
                price=0.0,
                rho=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
                sigma=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
                clip=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
            )
        else:
            non_private_schedule = PrivacySchedule(
                price=float(schedule.price),
                rho=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
                sigma=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
                clip=np.asarray(schedule.clip, dtype=np.float64),
            )
        reference_truthful = self._detach_metrics(
            self._simulate_batch(
                non_private_schedule,
                batch_size=self.config.eval_batch_size,
                context=context,
            )
        )
        reference_deviating_utilities = np.asarray(reference_truthful["avg_agent_utility"], dtype=np.float64)
        for agent_idx in range(self.config.num_agents):
            reference_deviating = self._detach_metrics(
                self._simulate_batch(
                    non_private_schedule,
                    batch_size=self.config.eval_batch_size,
                    context=context,
                    deviating_agent=agent_idx,
                )
            )
            reference_deviating_utilities[agent_idx] = float(reference_deviating["avg_agent_utility"][agent_idx])

        oracle_schedule = PrivacySchedule(
            price=0.0,
            rho=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
            sigma=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
            clip=np.zeros((self.config.episode_length, self.config.num_agents), dtype=np.float64),
        )
        oracle = self._detach_metrics(
            self._simulate_batch(
                oracle_schedule,
                batch_size=self.config.eval_batch_size,
                context=context,
                use_transfers=False,
                oracle_contract=True,
            )
        )

        constraint_metrics = summarize_constraints(
            truthful_utilities=np.asarray(truthful["avg_agent_utility"], dtype=np.float64),
            deviating_utilities=deviating_utilities,
            oracle_welfare=float(oracle["team_reward"]),
            achieved_welfare=float(truthful["team_reward"]),
            kl_distortion=np.asarray(truthful["kl_step_distortion"], dtype=np.float64),
            gamma=self.config.discount_gamma,
            welfare_lipschitz=self.config.welfare_lipschitz,
            utility_lipschitz=np.full(self.config.num_agents, self.config.utility_lipschitz, dtype=np.float64),
            baseline_epsilon_ic=np.maximum(
                reference_deviating_utilities - np.asarray(reference_truthful["avg_agent_utility"], dtype=np.float64),
                0.0,
            ),
            baseline_epsilon_ir=np.maximum(
                -np.asarray(reference_truthful["avg_agent_utility"], dtype=np.float64),
                0.0,
            ),
        )
        if self.privacy_mode == "none" or not self.use_messages:
            zeros = np.zeros(self.config.num_agents, dtype=np.float64)
            privacy_metrics = {
                "sigma": zeros.tolist(),
                "total_rho_spent": zeros.tolist(),
                "epsilon": zeros.tolist(),
                "clip": zeros.tolist(),
            }
        else:
            claimed_rho = self.total_rho_spent + self._block_rho(schedule)
            claimed_block_rho = np.asarray(truthful["claimed_runtime_rho"], dtype=np.float64)
            claimed_total_rho = self.total_claimed_runtime_rho_spent + claimed_block_rho
            privacy_metrics = summarize_privacy(
                sigmas=np.asarray(schedule.sigma, dtype=np.float64).mean(axis=0),
                total_rho_spent=claimed_rho,
                claimed_sensitivity=2.0 * np.asarray(schedule.clip, dtype=np.float64).max(axis=0),
                alpha=self.config.privacy_alpha,
                delta=self.config.delta,
                actual_total_rho_spent=claimed_total_rho,
                accounting_mode="runtime_clipped_transcript",
                actual_claim_label="runtime_clipped",
            )
            privacy_metrics["clip"] = np.asarray(schedule.clip, dtype=np.float64).tolist()
            privacy_metrics["runtime_clipped_rho"] = claimed_block_rho.tolist()
            privacy_metrics["runtime_clipped_rho_from_schedule"] = rdp_gaussian_rho(
                self.config.privacy_alpha,
                2.0 * np.asarray(schedule.clip, dtype=np.float64),
                np.asarray(schedule.sigma, dtype=np.float64),
            ).sum(axis=0).tolist()
            if self.config.scheduler_mode == "naive_la":
                naive_block_rho = np.asarray(truthful["unclipped_runtime_rho"], dtype=np.float64)
                naive_total_rho = self.total_unclipped_runtime_rho_spent + naive_block_rho
                naive_metrics = summarize_privacy(
                    sigmas=np.asarray(schedule.sigma, dtype=np.float64).mean(axis=0),
                    total_rho_spent=claimed_rho,
                    alpha=self.config.privacy_alpha,
                    delta=self.config.delta,
                    actual_total_rho_spent=naive_total_rho,
                    accounting_mode="naive_unclipped_counterexample",
                    actual_claim_label="naive_unclipped",
                )
                privacy_metrics["naive_unclipped_counterexample"] = {
                    "runtime_rho": naive_block_rho.tolist(),
                    "total_rho_spent": naive_total_rho.tolist(),
                    "epsilon": naive_metrics["actual_epsilon"],
                    "overspend_ratio": naive_metrics["overspend_ratio"],
                }

        return {
            "block": block_index,
            "price": float(schedule.price),
            "rho": np.asarray(schedule.rho, dtype=np.float64).tolist(),
            "sigma": np.asarray(schedule.sigma, dtype=np.float64).tolist(),
            "clip": np.asarray(schedule.clip, dtype=np.float64).tolist(),
            "team_reward": truthful["team_reward"],
            "oracle_team_reward": oracle["team_reward"],
            "reference_team_reward": reference_truthful["team_reward"],
            "modified_reward": truthful["modified_reward"],
            "posterior_nll": truthful["posterior_nll"],
            "avg_agent_utility": truthful["avg_agent_utility"],
            "reference_avg_agent_utility": reference_truthful["avg_agent_utility"],
            "quality": truthful["quality"],
            "posterior_uncertainty": truthful["posterior_uncertainty"],
            "allocation_error": truthful["allocation_error"],
            "message_norm": truthful["message_norm"],
            "kl_distortion": truthful["kl_distortion"],
            "posterior_error": truthful["posterior_error"],
            "clip_rate": truthful["clip_rate"],
            "latent_sensitivity": truthful["latent_sensitivity"],
            "claimed_runtime_rho": truthful["claimed_runtime_rho"],
            "unclipped_runtime_rho": truthful["unclipped_runtime_rho"],
            "epsilon_ic": constraint_metrics["epsilon_ic"],
            "epsilon_ir": constraint_metrics["epsilon_ir"],
            "baseline_epsilon_ic": np.maximum(
                reference_deviating_utilities - np.asarray(reference_truthful["avg_agent_utility"], dtype=np.float64),
                0.0,
            ).tolist(),
            "baseline_epsilon_ir": np.maximum(
                -np.asarray(reference_truthful["avg_agent_utility"], dtype=np.float64),
                0.0,
            ).tolist(),
            "epsilon_ic_bound": constraint_metrics.get("epsilon_ic_bound", []),
            "epsilon_ir_bound": constraint_metrics.get("epsilon_ir_bound", []),
            "welfare_regret": constraint_metrics["welfare_regret"],
            "welfare_regret_bound": constraint_metrics.get("welfare_regret_bound", 0.0),
            "posterior_perturbation_bound": constraint_metrics.get("posterior_perturbation_bound", []),
            "privacy": privacy_metrics,
        }

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
            smoothing=0.08,
            mininterval=0.1,
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
                    batch = self._simulate_batch(schedule, batch_size=self.config.train_batch_size)
                    loss = -batch["objective"]
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self._update_ema()
                    last_loss = float(loss.detach().cpu().item())
                    progress.update(1)
                summary = self.evaluate_block(schedule, block_index)
                self._post_block_update(schedule)
                self.total_claimed_runtime_rho_spent += np.asarray(summary["claimed_runtime_rho"], dtype=np.float64)
                self.total_unclipped_runtime_rho_spent += np.asarray(summary["unclipped_runtime_rho"], dtype=np.float64)
                self.history.append(summary)
                last_summary = summary
                progress.set_postfix(
                    self._progress_postfix(schedule, block_index, loss=last_loss, summary=summary),
                    refresh=False,
                )
        last_entry = self.history[-1] if self.history else {}
        best_entry = self._select_final_entry()
        ema_last_entry: dict[str, Any] = {}
        if self.history and self.ema_shadow is not None:
            final_schedule = PrivacySchedule(
                price=float(last_entry["price"]),
                rho=np.asarray(last_entry["rho"], dtype=np.float64),
                sigma=np.asarray(last_entry["sigma"], dtype=np.float64),
                clip=np.asarray(
                    last_entry.get("clip", np.zeros((self.config.episode_length, self.config.num_agents))),
                    dtype=np.float64,
                ),
            )
            with self._ema_scope():
                ema_last_entry = self.evaluate_block(final_schedule, int(last_entry["block"]))
        return {
            "config": self.config.to_dict(),
            "adaptive_privacy": self.adaptive_privacy,
            "use_transfers": self.use_transfers,
            "history": self.history,
            "last": last_entry,
            "best": best_entry,
            "ema_last": ema_last_entry,
            "checkpoint_selection": "last",
            "final": last_entry,
        }

    @staticmethod
    def save_results(results: dict[str, Any], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))


class PILTrainer(BaseCommunicationTrainer):
    def __init__(self, config: PILConfig) -> None:
        super().__init__(config, adaptive_privacy=True, use_transfers=True)
