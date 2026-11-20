# SecFedRL: Byzantine-Resilient Federated Reinforcement Learning

> **Paper:** *SecFedRL: A Byzantine-Resilient Framework for Decentralized Federated Reinforcement Learning*  
> **Status:** Under review

---

## Overview

SecFedRL is a fully decentralized Byzantine-resilient framework for Peer-to-Peer (P2P)
Federated Reinforcement Learning (FRL). Unlike prior work that defends in parameter space,
SecFedRL operates in **policy space** via virtual-trajectory behavioral analysis — requiring
no ground-truth labels, no central server, and no prior knowledge of the number of
Byzantine agents.

### Key Components

| Component | Description |
|-----------|-------------|
| **A2C + GAE** | Local reinforcement learning with Generalized Advantage Estimation (λ=0.95, γ=0.99) |
| **7-Metric Detection** | KL divergence, cosine similarity, policy agreement, value alignment, entropy divergence, reverse KL, gradient magnitude |
| **Composite Suspicion Score** | Z-score normalized across k=4 peers, 10-round sliding window (Eq. 2) |
| **Reputation-Aware Aggregation** | EMA-based reputation r_j(t) combining reward, behavioral consistency, gradient alignment (Eq. 3–5) |
| **Collapse Prevention** | Checkpoint recovery with λ-weighted interpolation (Eq. 6) |
| **Consensus + Ledger** | 2n/3 threshold on relative δ=5% reward improvement; SHA-256 hash-chained audit ledger (Eq. 7) |

### Results Summary

| Environment | Avg. Per-Round TPR (20% Byzantine) | Cumul. TPR | Reward vs FedAvg |
|-------------|-------------------------------------|------------|------------------|
| CartPole-v1 | 98% | 100% (25–35 rounds) | +521% |
| LunarLander-v2 | 98% | 100% (30–42 rounds) | +362% |
| Pendulum-v1 | 98% | 100% (22–35 rounds) | +78% (absolute) |

Computational overhead: **<8.3%** over undefended FedAvg.

---

## Installation

```bash
git clone https://github.com/Yog37h/byzsecure.git
cd byzsecure
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, Gymnasium 0.26+

---

## Quick Start

### Train on CartPole-v1 (20% Byzantine, UMP attack)

```bash
python main.py --env CartPole-v1 --n_agents 20 --byzantine_frac 0.2 \
               --attack ump --rounds 100 --seeds 5
```

### Train on all three environments (5 seeds each)

```bash
for env in CartPole-v1 LunarLander-v2 Pendulum-v1; do
  python main.py --env $env --n_agents 20 --byzantine_frac 0.2 \
                 --attack ump --rounds 100 --seeds 5
done
```

### Reproduce ablation study

```bash
python main.py --env CartPole-v1 --n_agents 20 --byzantine_frac 0.2 \
               --ablate detection      # disable detection
python main.py --env CartPole-v1 --n_agents 20 --byzantine_frac 0.2 \
               --ablate reputation     # disable reputation weighting
python main.py --env CartPole-v1 --n_agents 20 --byzantine_frac 0.2 \
               --ablate recovery       # disable collapse prevention
```

---

## Project Structure

```
byzsecure/
├── main.py               # SecFedRL training orchestrator (Algorithm 1)
├── evaluate.py           # Evaluation and metric reporting
├── requirements.txt
├── configs/
│   ├── cartpole.yaml     # CartPole-v1 hyperparameters
│   ├── lunarlander.yaml  # LunarLander-v2 hyperparameters
│   └── pendulum.yaml     # Pendulum-v1 hyperparameters
├── src/
│   ├── agent.py          # A2C + GAE local RL training
│   ├── detection.py      # 7-metric Byzantine behavioral detection
│   ├── aggregation.py    # Reputation-aware model aggregation
│   ├── consensus.py      # Hash-chained ledger + 2/3 consensus
│   ├── recovery.py       # Checkpoint-based collapse prevention
│   └── p2p.py            # P2P mesh topology model exchange
└── paper/
    └── secfedrl.tex      # LaTeX source of the paper
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 20 | Total agents |
| k | 4 | Peers per agent |
| γ | 0.99 | Discount factor |
| λ_GAE | 0.95 | GAE smoothing |
| c_v | 0.5 | Critic loss weight |
| c_e | 0.01 | Entropy coefficient |
| η | 3×10⁻⁴ | Learning rate (Adam) |
| N_vt | 100 | Virtual trajectory states |
| λ_max | 0.85 | Max checkpoint recovery weight |
| δ | 5% | Relative consensus margin |
| T_local | 128–512 steps | Local steps per round (env-dependent) |
| α (EMA) | 0.9 | Reputation decay factor |
| Sliding window | 10 rounds | For μ_m/σ_m estimation in z-score |

---

## Attack Models

| Attack | Description |
|--------|-------------|
| **UMP** | Untargeted Model Poisoning — scales gradients by −5× to disrupt convergence |
| **TMP** | Targeted Model Poisoning — crafts updates to steer the global policy toward a specific sub-optimal attractor |

---

## Citation

If you use SecFedRL, please cite:

```bibtex
@article{secfedrl2025,
  title   = {SecFedRL: A Byzantine-Resilient Framework for Decentralized
             Federated Reinforcement Learning},
  journal = {Under Review},
  year    = {2025},
  url     = {https://github.com/Yog37h/byzsecure}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
