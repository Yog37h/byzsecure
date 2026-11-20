"""
SecFedRL: 7-Metric Behavioral Byzantine Detection
Implements Eq. (2): composite suspicion score via z-score normalization.

The 7 metrics are:
  1. KL divergence        (forward)
  2. Cosine similarity    (of feature vectors)
  3. Policy agreement     (argmax action match)
  4. Value alignment      (normalized critic difference)
  5. Entropy divergence   (absolute entropy gap)
  6. Reverse KL           (backward KL)
  7. Gradient magnitude consistency

Statistical stabilization: metrics averaged over a 10-round sliding window
(10 × k=4 peers = up to 40 observations per metric per agent).
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_METRIC_NAMES = [
    "kl_div", "cosine_sim", "policy_agreement",
    "value_alignment", "entropy_div", "reverse_kl", "grad_magnitude",
]
N_METRICS = len(_METRIC_NAMES)
EPS = 1e-8


def compute_metrics(
    net_i, net_j,
    virtual_states: np.ndarray,
    device: torch.device,
    grad_norm_i: Optional[float] = None,
    grad_norm_j: Optional[float] = None,
) -> np.ndarray:
    """
    Compute all 7 behavioral metrics between agent i's model and peer j's model.

    Args:
        net_i:           Agent i's ActorCritic network (the evaluating agent).
        net_j:           Peer j's ActorCritic network (being evaluated).
        virtual_states:  (N_vt, obs_dim) states sampled from agent i's replay buffer.
        device:          Torch device.
        grad_norm_i:     Gradient norm of agent i (for metric 7). Pass None to skip.
        grad_norm_j:     Gradient norm of agent j (for metric 7). Pass None to skip.

    Returns:
        metrics: np.ndarray of shape (7,) with one value per metric.
                 Higher values → more suspicious for all metrics except cosine_sim.
    """
    states_t = torch.FloatTensor(virtual_states).to(device)

    with torch.no_grad():
        logits_i, val_i = net_i(states_t)
        logits_j, val_j = net_j(states_t)

    # Determine if discrete or continuous action space
    # (Continuous: Normal dist; Discrete: Categorical)
    is_continuous = hasattr(net_i, "log_std")

    if is_continuous:
        std_i = net_i.log_std.exp().expand_as(logits_i)
        std_j = net_j.log_std.exp().expand_as(logits_j)
        pi_i = torch.distributions.Normal(logits_i, std_i)
        pi_j = torch.distributions.Normal(logits_j, std_j)
        log_pi_i = pi_i.log_prob(logits_i).sum(-1)   # use mean as reference action
        log_pi_j = pi_j.log_prob(logits_i).sum(-1)
        # KL divergence (forward): KL(pi_i || pi_j)
        kl_fwd = torch.distributions.kl_divergence(pi_i, pi_j).sum(-1).mean().item()
        # Reverse KL: KL(pi_j || pi_i)
        kl_rev = torch.distributions.kl_divergence(pi_j, pi_i).sum(-1).mean().item()
        # Entropy
        h_i = pi_i.entropy().sum(-1)
        h_j = pi_j.entropy().sum(-1)
        # Policy agreement: fraction of states where argmax logit matches
        agree = (logits_i.argmax(-1) == logits_j.argmax(-1)).float().mean().item()
        # Feature vectors for cosine (use logits as proxy)
        phi_i = logits_i.reshape(logits_i.size(0), -1)
        phi_j = logits_j.reshape(logits_j.size(0), -1)
    else:
        pi_i = torch.distributions.Categorical(logits=logits_i)
        pi_j = torch.distributions.Categorical(logits=logits_j)
        # KL divergence (forward)
        kl_fwd = torch.distributions.kl_divergence(pi_i, pi_j).mean().item()
        # Reverse KL
        kl_rev = torch.distributions.kl_divergence(pi_j, pi_i).mean().item()
        # Entropy
        h_i = pi_i.entropy()
        h_j = pi_j.entropy()
        # Policy agreement
        agree = (pi_i.probs.argmax(-1) == pi_j.probs.argmax(-1)).float().mean().item()
        # Feature vectors
        phi_i = logits_i
        phi_j = logits_j

    # 1. KL divergence (higher = more suspicious)
    m1_kl = float(np.clip(kl_fwd, 0, 100))

    # 2. Cosine similarity (convert to dissimilarity: 1 - cos, higher = more suspicious)
    cos_sim = F.cosine_similarity(phi_i, phi_j, dim=-1).mean().item()
    m2_cos = 1.0 - float(np.clip(cos_sim, -1, 1))

    # 3. Policy agreement (1 - agreement rate, higher = more suspicious)
    m3_agree = 1.0 - float(np.clip(agree, 0, 1))

    # 4. Value alignment (normalized absolute critic difference, higher = more suspicious)
    denom = torch.maximum(val_i.abs(), val_j.abs()) + EPS
    val_align = torch.clamp(1.0 - (val_i - val_j).abs() / denom, min=0.0)
    m4_val = float(1.0 - val_align.mean().item())   # invert: 0=aligned, 1=misaligned

    # 5. Entropy divergence (absolute difference in policy entropy, higher = more suspicious)
    entropy_div = (h_i - h_j).abs().mean().item()
    m5_entropy = float(np.clip(entropy_div, 0, 100))

    # 6. Reverse KL (higher = more suspicious)
    m6_rkl = float(np.clip(kl_rev, 0, 100))

    # 7. Gradient magnitude consistency (|||∇θ_i||₂ - ||∇θ_j||₂|, higher = more suspicious)
    if grad_norm_i is not None and grad_norm_j is not None:
        m7_grad = abs(grad_norm_i - grad_norm_j)
    else:
        m7_grad = 0.0

    return np.array([m1_kl, m2_cos, m3_agree, m4_val, m5_entropy, m6_rkl, m7_grad],
                    dtype=np.float64)


class ByzantineDetector:
    """
    Maintains the sliding-window metric history and computes composite
    suspicion scores (Eq. 2 of the paper).

    Eq. (2):  S_j = Σ_m  w_m · (M_m(j) − μ_m) / σ_m

    μ_m and σ_m are estimated over a sliding window of the last 10 rounds
    × k peers = up to 40 observations per metric.
    """

    def __init__(
        self,
        n_agents: int,
        k_peers: int = 4,
        window: int = 10,
        weights: Optional[np.ndarray] = None,
    ):
        self.n_agents = n_agents
        self.k_peers = k_peers
        self.window = window
        # Equal weights by default; shape (7,)
        self.weights = weights if weights is not None else np.ones(N_METRICS) / N_METRICS

        # Per-agent sliding window: deque of (N_metrics,) vectors
        self.history: Dict[int, deque] = {
            i: deque(maxlen=window * k_peers) for i in range(n_agents)
        }

    def update_and_score(
        self,
        agent_id: int,
        peer_metrics: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        """
        Update the sliding window with this round's peer metrics,
        then compute suspicion scores for all peers.

        Args:
            agent_id:    Index of the evaluating agent.
            peer_metrics: {peer_id: metrics_array(7,)} for each sampled peer.

        Returns:
            suspicion: {peer_id: float} composite suspicion score.
        """
        buf = self.history[agent_id]

        # Add this round's observations
        for metrics in peer_metrics.values():
            buf.append(metrics.copy())

        if len(buf) < 2:
            # Not enough data; return zero suspicion for everyone
            return {pid: 0.0 for pid in peer_metrics}

        all_obs = np.stack(list(buf))           # (N_obs, 7)
        mu = all_obs.mean(axis=0)               # (7,)
        sigma = all_obs.std(axis=0) + EPS       # (7,)

        suspicion = {}
        for peer_id, m in peer_metrics.items():
            z = (m - mu) / sigma                # (7,)
            score = float(np.dot(self.weights, z))
            suspicion[peer_id] = score

        return suspicion

    def reset(self, agent_id: int):
        """Clear history for an agent (e.g., when it reconnects)."""
        self.history[agent_id].clear()


def filter_byzantine(
    suspicion_scores: Dict[int, float],
    trusted_set: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Dynamic outlier filtering: exclude peer j if S_j > μ_S + 2σ_S.
    No prior knowledge of M (number of malicious agents) required.

    Args:
        suspicion_scores: {peer_id: float}
        trusted_set: Optional forced trusted set (for ablation).

    Returns:
        (benign_ids, byzantine_ids)
    """
    if trusted_set is not None:
        all_ids = list(suspicion_scores.keys())
        byzantine = [p for p in all_ids if p not in trusted_set]
        return trusted_set, byzantine

    scores = np.array(list(suspicion_scores.values()))
    peer_ids = list(suspicion_scores.keys())

    if len(scores) <= 1:
        return peer_ids, []

    mu_s = scores.mean()
    sigma_s = scores.std() + EPS
    threshold = mu_s + 2.0 * sigma_s

    benign = [pid for pid, s in zip(peer_ids, scores) if s <= threshold]
    byzantine = [pid for pid, s in zip(peer_ids, scores) if s > threshold]

    # Safety: always keep at least one peer
    if not benign:
        best = peer_ids[int(np.argmin(scores))]
        benign = [best]
        byzantine = [p for p in peer_ids if p != best]

    return benign, byzantine
