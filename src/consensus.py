"""
SecFedRL: Hash-Chained Audit Ledger + 2/3 Consensus
Implements Eq. (7) from the paper.

Eq. (7):  D_t = 1 if Σ_i vote_i(t) ≥ ⌈2N/3⌉, else 0

vote_i(t) = 1 if the aggregated reward for round t improves by at least
            δ=5% over the 5-round moving average r̄_{t-5:t-1}.

The ledger records accepted rounds as SHA-256 hash entries chained to
the previous accepted round's hash, providing tamper-evident auditability.
"""

import hashlib
import json
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import math


class ConsensusVoter:
    """
    Per-agent consensus voter.

    Each agent independently decides whether to vote pass or fail
    based on a relative improvement threshold (environment-agnostic).
    """

    def __init__(self, delta: float = 0.05, window: int = 5):
        self.delta = delta      # 5% relative improvement threshold
        self.window = window    # 5-round moving average
        self.reward_history: deque = deque(maxlen=window)

    def vote(self, current_reward: float) -> int:
        """
        Returns 1 (pass) if current_reward improves by ≥ delta over moving avg, else 0.

        Environment-agnostic:
        - For CartPole (positive rewards): tracks absolute improvement.
        - For Pendulum (negative rewards): tracks reduction in |reward| magnitude.
        """
        if len(self.reward_history) == 0:
            self.reward_history.append(current_reward)
            return 1  # First round: always accept (no baseline)

        moving_avg = sum(self.reward_history) / len(self.reward_history)

        if moving_avg == 0:
            improvement = 1.0 if current_reward > 0 else 0.0
        elif moving_avg > 0:
            # Positive reward environment (CartPole, LunarLander)
            improvement = (current_reward - moving_avg) / abs(moving_avg)
        else:
            # Negative reward environment (Pendulum)
            # Improvement = reduction in magnitude
            improvement = (abs(moving_avg) - abs(current_reward)) / abs(moving_avg)

        self.reward_history.append(current_reward)
        return 1 if improvement >= self.delta else 0


class DecentralizedConsensus:
    """
    Runs the 2n/3 consensus vote across N agents.

    Each agent submits a vote; the round is accepted if ≥ ⌈2N/3⌉ agents vote pass.
    """

    def __init__(self, n_agents: int, delta: float = 0.05, window: int = 5):
        self.n_agents = n_agents
        self.threshold = math.ceil(2 * n_agents / 3)
        self.voters: List[ConsensusVoter] = [
            ConsensusVoter(delta=delta, window=window) for _ in range(n_agents)
        ]

    def run_vote(self, agent_rewards: List[float]) -> Tuple[int, int, int]:
        """
        Collect votes from all agents and determine consensus decision.

        Args:
            agent_rewards: Per-agent episodic reward for the current round.

        Returns:
            (decision, vote_count, threshold)
              decision=1 → accepted, decision=0 → rejected/rolled back.
        """
        assert len(agent_rewards) == self.n_agents, (
            f"Expected {self.n_agents} rewards, got {len(agent_rewards)}"
        )

        votes = [v.vote(r) for v, r in zip(self.voters, agent_rewards)]
        vote_count = sum(votes)
        decision = 1 if vote_count >= self.threshold else 0
        return decision, vote_count, self.threshold


class HashChainedLedger:
    """
    Lightweight SHA-256 hash-chained audit ledger.

    Each accepted round is recorded as a JSON entry containing:
      - round number
      - aggregated model hash (SHA-256 of serialized state dict)
      - reward statistics
      - vote count
      - timestamp
      - previous entry hash (chain link)

    <0.1% overhead as reported in the paper.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self):
        self.entries: List[dict] = []
        self._prev_hash: str = self.GENESIS_HASH

    def _hash_model(self, model_params: dict) -> str:
        """Compute SHA-256 hash of model parameters."""
        import io
        import pickle
        buf = io.BytesIO()
        # Serialize to bytes for hashing
        for key in sorted(model_params.keys()):
            param_bytes = model_params[key].cpu().numpy().tobytes()
            buf.write(key.encode())
            buf.write(param_bytes)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def record(
        self,
        round_num: int,
        model_params: dict,
        mean_reward: float,
        vote_count: int,
        n_agents: int,
    ) -> str:
        """
        Record an accepted round. Returns the hash of this entry.

        Args:
            round_num:    Current training round index.
            model_params: Global model state_dict (accepted aggregation).
            mean_reward:  Mean episodic reward across benign agents this round.
            vote_count:   Number of pass votes received.
            n_agents:     Total number of agents.

        Returns:
            entry_hash: SHA-256 of this ledger entry.
        """
        model_hash = self._hash_model(model_params)
        entry = {
            "round": round_num,
            "timestamp": time.time(),
            "model_hash": model_hash,
            "mean_reward": round(mean_reward, 4),
            "vote_count": vote_count,
            "n_agents": n_agents,
            "prev_hash": self._prev_hash,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["entry_hash"] = entry_hash

        self.entries.append(entry)
        self._prev_hash = entry_hash
        return entry_hash

    def verify_chain(self) -> bool:
        """Verify hash chain integrity."""
        prev = self.GENESIS_HASH
        for entry in self.entries:
            stored_hash = entry.pop("entry_hash")
            recomputed = hashlib.sha256(
                json.dumps({k: v for k, v in entry.items() if k != "entry_hash"},
                           sort_keys=True).encode()
            ).hexdigest()
            entry["entry_hash"] = stored_hash
            if recomputed != stored_hash:
                return False
            if entry["prev_hash"] != prev:
                return False
            prev = stored_hash
        return True

    def __len__(self):
        return len(self.entries)

    def summary(self) -> dict:
        if not self.entries:
            return {"total_accepted": 0, "latest_hash": self.GENESIS_HASH}
        return {
            "total_accepted": len(self.entries),
            "latest_hash": self._prev_hash,
            "first_round": self.entries[0]["round"],
            "last_round": self.entries[-1]["round"],
        }
