"""
SecFedRL: Reputation-Aware Model Aggregation
Implements Eq. (3), (4), (5) from the paper.

Eq. (3):  r_j(t) = α · r_j(t-1) + (1-α) · [β1·R̂_j + β2·C_j + β3·G_j]
Eq. (4):  w_j = r_j(t) · (1 − S_j) / Σ_k [r_k(t) · (1 − S_k)]
Eq. (5):  θ_g = Σ_{j ∈ B_t} w_j · θ_j
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


class ReputationTracker:
    """
    Maintains ground-truth-free reputation scores r_j(t) ∈ [0,1] for each peer.

    Initial reputation: r_j(0) = 0.5 for all peers.
    Update rule (Eq. 3): EMA over reward, behavioral consistency, gradient alignment.
    """

    def __init__(
        self,
        n_agents: int,
        alpha: float = 0.9,
        beta1: float = 1 / 3,
        beta2: float = 1 / 3,
        beta3: float = 1 / 3,
        init_rep: float = 0.5,
    ):
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        # r_j(0) = 0.5
        self.reputation: Dict[int, float] = {j: init_rep for j in range(n_agents)}

    def update(
        self,
        peer_id: int,
        reward_norm: float,         # R̂_j — normalized episodic reward ∈ [0,1]
        consistency: float,          # C_j = 1 − S_j / max_k S_k
        grad_alignment: float,       # G_j = cosine similarity of ∇θ_j and ∇θ_g
    ):
        """Update reputation for a single peer (Eq. 3)."""
        composite = self.beta1 * reward_norm + self.beta2 * consistency + self.beta3 * grad_alignment
        composite = float(np.clip(composite, 0.0, 1.0))
        r_prev = self.reputation[peer_id]
        self.reputation[peer_id] = self.alpha * r_prev + (1 - self.alpha) * composite

    def get(self, peer_id: int) -> float:
        return self.reputation[peer_id]

    def get_all(self) -> Dict[int, float]:
        return dict(self.reputation)

    def reset(self):
        self.reputation = {j: 0.5 for j in range(self.n_agents)}


def compute_reputation_inputs(
    peer_reward: float,
    all_peer_rewards: List[float],
    suspicion_score: float,
    max_suspicion: float,
    grad_norm_peer: Optional[float],
    grad_norm_global: Optional[float],
    global_grad_vec: Optional[torch.Tensor] = None,
    peer_grad_vec: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Compute (R̂_j, C_j, G_j) for reputation update.

    Args:
        peer_reward:         Episodic reward of peer j this round.
        all_peer_rewards:    Rewards of all trusted peers (for normalization).
        suspicion_score:     S_j (composite suspicion score).
        max_suspicion:       max_k S_k across trusted peers.
        grad_norm_peer:      ||∇θ_j||₂.
        grad_norm_global:    ||∇θ_g||₂ (gradient of global model).
        global_grad_vec:     Flattened gradient vector of the global model.
        peer_grad_vec:       Flattened gradient vector of peer j.

    Returns:
        (reward_norm, consistency, grad_alignment) all ∈ [0, 1].
    """
    # R̂_j: normalize to [0, 1] using min-max over trusted peers
    rewards_arr = np.array(all_peer_rewards)
    r_min, r_max = rewards_arr.min(), rewards_arr.max()
    if r_max > r_min:
        reward_norm = float(np.clip((peer_reward - r_min) / (r_max - r_min + EPS), 0, 1))
    else:
        reward_norm = 0.5

    # C_j: behavioral consistency = 1 − S_j / max_k S_k
    if max_suspicion > EPS:
        consistency = float(np.clip(1.0 - suspicion_score / max_suspicion, 0, 1))
    else:
        consistency = 1.0

    # G_j: gradient alignment via cosine similarity
    if global_grad_vec is not None and peer_grad_vec is not None:
        g_align = F.cosine_similarity(
            global_grad_vec.unsqueeze(0), peer_grad_vec.unsqueeze(0)
        ).item()
        grad_alignment = float(np.clip((g_align + 1) / 2, 0, 1))   # map [-1,1] → [0,1]
    elif grad_norm_peer is not None and grad_norm_global is not None:
        # Fallback: use magnitude proximity
        ratio = min(grad_norm_peer, grad_norm_global) / (max(grad_norm_peer, grad_norm_global) + EPS)
        grad_alignment = float(np.clip(ratio, 0, 1))
    else:
        grad_alignment = 0.5

    return reward_norm, consistency, grad_alignment


def reputation_weighted_aggregate(
    peer_state_dicts: Dict[int, dict],
    reputation: Dict[int, float],
    suspicion_scores: Dict[int, float],
) -> dict:
    """
    Compute reputation-weighted average of peer model parameters (Eq. 4–5).

    Aggregation weights: w_j = r_j(t) · (1 − S_j) / Σ_k [r_k(t) · (1 − S_k)]
    Global update:       θ_g = Σ_{j ∈ B_t} w_j · θ_j

    Args:
        peer_state_dicts:  {peer_id: state_dict} for peers in the trusted set B_t.
        reputation:        {peer_id: float} current reputation scores.
        suspicion_scores:  {peer_id: float} composite suspicion scores.

    Returns:
        Aggregated state_dict (weighted average).
    """
    if not peer_state_dicts:
        raise ValueError("peer_state_dicts is empty — no trusted peers available.")

    # Compute unnormalized weights
    raw_weights = {}
    for pid in peer_state_dicts:
        r = reputation.get(pid, 0.5)
        s = suspicion_scores.get(pid, 0.0)
        # Clamp suspicion to [0, 1] so weight is always non-negative
        s_clamped = float(np.clip(s, 0.0, 1.0))
        raw_weights[pid] = r * (1.0 - s_clamped)

    total = sum(raw_weights.values())
    if total < EPS:
        # Uniform fallback if all weights are zero
        total = len(peer_state_dicts)
        weights = {pid: 1.0 / total for pid in peer_state_dicts}
    else:
        weights = {pid: w / total for pid, w in raw_weights.items()}

    # Weighted sum of state dicts (Eq. 5)
    peer_ids = list(peer_state_dicts.keys())
    agg = {}
    first_sd = peer_state_dicts[peer_ids[0]]
    for key in first_sd:
        agg[key] = torch.zeros_like(first_sd[key], dtype=torch.float32)

    for pid, sd in peer_state_dicts.items():
        w = weights[pid]
        for key in sd:
            agg[key] += w * sd[key].float()

    return agg
