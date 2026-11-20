"""
SecFedRL: Checkpoint-Based Catastrophic Policy Collapse Prevention
Implements Eq. (6) from the paper.

Eq. (6): θ_recover = λ · θ_checkpoint + (1 − λ) · θ_current

where λ = min(λ_max, Δr_t / r̄_{t-1})

A collapse is defined as: Δr_t > 20% of the prior 5-round moving average.
Checkpoints are saved every 5 rounds.
λ_max = 0.85 (prevents full replacement).
"""

import copy
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch


class CollapseMonitor:
    """
    Monitors episode rewards and detects catastrophic policy collapse.
    Manages checkpoints and executes recovery (Eq. 6).
    """

    COLLAPSE_THRESHOLD = 0.20   # 20% drop triggers recovery
    CHECKPOINT_INTERVAL = 5     # save checkpoint every 5 rounds
    LAMBDA_MAX = 0.85           # maximum recovery interpolation weight

    def __init__(self, moving_avg_window: int = 5):
        self.window = moving_avg_window
        self.reward_history: deque = deque(maxlen=moving_avg_window)
        self.checkpoint: Optional[dict] = None
        self.checkpoint_round: int = -1
        self.last_recovery_round: int = -1

    def update(self, round_num: int, mean_reward: float, model_params: dict):
        """
        Call at the end of each round with the aggregated global model.

        Args:
            round_num:    Current round index (0-based).
            mean_reward:  Mean reward across agents this round.
            model_params: Current global model state_dict.

        Returns:
            (collapse_detected, recovered_params_or_None)
        """
        self.reward_history.append(mean_reward)

        # Save checkpoint every CHECKPOINT_INTERVAL rounds
        if (round_num + 1) % self.CHECKPOINT_INTERVAL == 0:
            self.checkpoint = {k: v.clone() for k, v in model_params.items()}
            self.checkpoint_round = round_num

        # Collapse detection: need at least 2 rounds to compare
        if len(self.reward_history) < 2:
            return False, None

        moving_avg = np.mean(list(self.reward_history)[:-1])  # exclude current

        # Handle negative reward environments (Pendulum)
        if moving_avg < 0:
            # Improvement = reduction in |reward|; collapse = increase in |reward|
            delta_ratio = (abs(mean_reward) - abs(moving_avg)) / (abs(moving_avg) + 1e-8)
        else:
            delta_ratio = (moving_avg - mean_reward) / (abs(moving_avg) + 1e-8)

        collapse = delta_ratio > self.COLLAPSE_THRESHOLD

        if collapse and self.checkpoint is not None:
            recovered = self._recover(model_params, delta_ratio)
            self.last_recovery_round = round_num
            return True, recovered

        return False, None

    def _recover(self, current_params: dict, delta_ratio: float) -> dict:
        """
        Merge current policy with checkpoint (Eq. 6).

        λ = min(λ_max, Δr_t / r̄_{t-1})  — scales with collapse severity
        """
        lam = min(self.LAMBDA_MAX, float(np.clip(delta_ratio, 0, 1)))
        recovered = {}
        for key in current_params:
            recovered[key] = (
                lam * self.checkpoint[key].float()
                + (1 - lam) * current_params[key].float()
            )
        return recovered

    def get_moving_avg(self) -> float:
        if not self.reward_history:
            return 0.0
        return float(np.mean(list(self.reward_history)))
