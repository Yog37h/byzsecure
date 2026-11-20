"""
SecFedRL: P2P Mesh Topology Model Exchange
Simulates decentralized peer-to-peer model sharing.

Each agent randomly selects k=4 peers per round.
Communication complexity: O(kN), independent of total N.
"""

import random
from typing import Dict, List, Set, Tuple
import numpy as np


class P2PMesh:
    """
    Simulates a P2P mesh topology where each agent communicates
    with k randomly selected peers per round.
    """

    def __init__(self, n_agents: int, k_peers: int = 4, byzantine_ids: Set[int] = None):
        self.n_agents = n_agents
        self.k_peers = k_peers
        self.byzantine_ids = byzantine_ids or set()

    def sample_peers(self, agent_id: int, seed: Optional[int] = None) -> List[int]:
        """
        Uniformly sample k distinct peers for agent_id (excluding itself).

        Args:
            agent_id: The requesting agent.
            seed:     Optional random seed for reproducibility.

        Returns:
            List of k peer indices.
        """
        rng = random.Random(seed)
        candidates = [i for i in range(self.n_agents) if i != agent_id]
        k = min(self.k_peers, len(candidates))
        return rng.sample(candidates, k)

    def exchange_models(
        self,
        agent_id: int,
        all_state_dicts: Dict[int, dict],
        round_num: int,
    ) -> Dict[int, dict]:
        """
        Perform model exchange for agent_id.

        Byzantine agents corrupt their model updates before sharing:
          - UMP: scale all parameters by -5× (untargeted disruption)
          - TMP: add targeted perturbation (implemented externally via attack flags)

        In a real P2P system this would be a network call. Here we simulate
        it by returning copies of peer state dicts.

        Args:
            agent_id:         The receiving agent.
            all_state_dicts:  Dict mapping agent_id → state_dict.
            round_num:        Current round (used as seed for peer sampling).

        Returns:
            {peer_id: state_dict} for the sampled peers.
        """
        import torch
        peer_ids = self.sample_peers(agent_id, seed=round_num * self.n_agents + agent_id)
        received = {}
        for pid in peer_ids:
            sd = {k: v.clone() for k, v in all_state_dicts[pid].items()}
            received[pid] = sd
        return received


from typing import Optional


def apply_ump_attack(state_dict: dict, scale: float = -5.0) -> dict:
    """
    Untargeted Model Poisoning (UMP): scale all parameters by `scale`.
    Disrupts convergence without targeting a specific policy.
    """
    import torch
    return {k: v.float() * scale for k, v in state_dict.items()}


def apply_tmp_attack(
    state_dict: dict,
    target_state_dict: dict,
    alpha: float = 2.0,
) -> dict:
    """
    Targeted Model Poisoning (TMP): nudge parameters toward a target
    (sub-optimal) policy with amplification factor alpha.

    θ_attack = θ_malicious + alpha · (θ_target − θ_malicious)
    """
    import torch
    attacked = {}
    for k in state_dict:
        attacked[k] = state_dict[k].float() + alpha * (
            target_state_dict[k].float() - state_dict[k].float()
        )
    return attacked
