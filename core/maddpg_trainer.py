from __future__ import annotations

from core.trainer import BaseCommunicationTrainer, PILConfig


class MADDPGTrainer(BaseCommunicationTrainer):
    """No-communication decentralized actor baseline inspired by MADDPG."""

    def __init__(self, config: PILConfig) -> None:
        super().__init__(
            config,
            adaptive_privacy=False,
            use_transfers=False,
            use_messages=False,
            deterministic_messages=True,
            use_contract=False,
            privacy_mode="none",
            baseline_name="MADDPG",
        )
