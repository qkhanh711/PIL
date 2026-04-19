from __future__ import annotations

from core.trainer import BaseCommunicationTrainer, PILConfig


class I2CTrainer(BaseCommunicationTrainer):
    """Deterministic communication baseline inspired by I2C-style message passing."""

    def __init__(self, config: PILConfig) -> None:
        super().__init__(
            config,
            adaptive_privacy=False,
            use_transfers=False,
            use_messages=True,
            deterministic_messages=True,
            use_contract=True,
            privacy_mode="none",
            baseline_name="I2C",
        )

