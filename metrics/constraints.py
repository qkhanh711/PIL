from __future__ import annotations

import torch


def ic_violation(utility_truth: torch.Tensor, utility_misreport: torch.Tensor) -> torch.Tensor:
    """Positive part of IC violation: max(0, U_misreport - U_truth)."""
    return torch.relu(utility_misreport - utility_truth)


def ir_violation(utility_truth: torch.Tensor) -> torch.Tensor:
    """Positive part of IR violation: max(0, -U_truth)."""
    return torch.relu(-utility_truth)
