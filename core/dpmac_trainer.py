from __future__ import annotations

from core.trainer import BaseCommunicationTrainer, PILConfig


class DPMACTrainer(BaseCommunicationTrainer):
    """Fixed-privacy baseline inspired by DPMAC."""

    def __init__(self, config: PILConfig) -> None:
        super().__init__(
            config,
            adaptive_privacy=False,
            use_transfers=False,
            use_messages=True,
            deterministic_messages=False,
            use_contract=True,
            privacy_mode="fixed",
            baseline_name="DPMAC",
        )
