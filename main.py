"""
SecFedRL: Full Training Orchestrator
Implements Algorithm 1 from the paper.

Usage:
  python main.py --env CartPole-v1 --n_agents 20 --byzantine_frac 0.2 \\
                 --attack ump --rounds 100 --seeds 5

For all environments:
  python main.py --env CartPole-v1    --n_agents 20 --byzantine_frac 0.2 --rounds 100
  python main.py --env LunarLander-v2 --n_agents 20 --byzantine_frac 0.2 --rounds 100
  python main.py --env Pendulum-v1    --n_agents 20 --byzantine_frac 0.2 --rounds 100
"""

import argparse
import copy
import json
import os
import random
import time
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import gymnasium as gym

from src.agent import A2CAgent, ActorCritic
from src.detection import ByzantineDetector, compute_metrics, filter_byzantine
from src.aggregation import ReputationTracker, compute_reputation_inputs, reputation_weighted_aggregate
from src.consensus import DecentralizedConsensus, HashChainedLedger
from src.recovery import CollapseMonitor
from src.p2p import P2PMesh, apply_ump_attack, apply_tmp_attack


# ── Hyperparameters (Table 2 of the paper) ─────────────────────────────────
DEFAULTS = {
    "n_agents":        20,
    "k_peers":         4,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "lr":              3e-4,
    "c_v":             0.5,
    "c_e":             0.01,
    "n_vt":            100,     # virtual trajectory states per evaluation
    "lambda_max":      0.85,
    "delta_consensus": 0.05,    # 5% relative improvement threshold
    "alpha_rep":       0.9,     # EMA decay for reputation
    "sliding_window":  10,      # rounds for μ_m/σ_m estimation
    # Environment-specific T_local (local steps per round)
    "t_local": {
        "CartPole-v1":    128,
        "LunarLander-v2": 256,
        "Pendulum-v1":    512,
    },
    "hidden": 128,
}


def get_env_spec(env_name: str):
    """Return (obs_dim, act_dim, continuous) for a given gym environment."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    continuous = hasattr(env.action_space, "shape")
    act_dim = (
        env.action_space.shape[0] if continuous else env.action_space.n
    )
    env.close()
    return obs_dim, act_dim, continuous


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_secfedrl(args, seed: int) -> dict:
    """
    Run one complete SecFedRL training session.

    Implements Algorithm 1:
      Phase 1: Local A2C training
      Phase 2: P2P model exchange
      Phase 3: Virtual trajectory behavioral analysis (7 metrics)
      Phase 4: Byzantine filtering
      Phase 5: Reputation update + reputation-aware aggregation
      Phase 6: Collapse prevention
      Phase 7: Consensus vote + ledger recording

    Returns: metrics dict with per-round statistics.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    obs_dim, act_dim, continuous = get_env_spec(args.env)
    t_local = DEFAULTS["t_local"].get(args.env, 256)

    n_agents = args.n_agents
    n_byzantine = int(round(n_agents * args.byzantine_frac))
    benign_ids = list(range(n_agents - n_byzantine))
    byzantine_ids = list(range(n_agents - n_byzantine, n_agents))

    print(f"\n[seed={seed}] env={args.env} | agents={n_agents} | "
          f"byzantine={n_byzantine} ({args.byzantine_frac*100:.0f}%) | attack={args.attack}")

    # ── Initialise all agents ───────────────────────────────────────────────
    envs = [gym.make(args.env) for _ in range(n_agents)]
    agents = [
        A2CAgent(
            obs_dim=obs_dim, act_dim=act_dim, continuous=continuous,
            gamma=DEFAULTS["gamma"], gae_lambda=DEFAULTS["gae_lambda"],
            lr=DEFAULTS["lr"], c_v=DEFAULTS["c_v"], c_e=DEFAULTS["c_e"],
            t_local=t_local, hidden=DEFAULTS["hidden"], device=device,
        )
        for _ in range(n_agents)
    ]

    # ── Initialise SecFedRL components ─────────────────────────────────────
    detector = ByzantineDetector(n_agents, k_peers=DEFAULTS["k_peers"],
                                  window=DEFAULTS["sliding_window"])
    reputation = ReputationTracker(n_agents, alpha=DEFAULTS["alpha_rep"])
    consensus = DecentralizedConsensus(n_agents, delta=DEFAULTS["delta_consensus"])
    ledger = HashChainedLedger()
    monitor = CollapseMonitor()
    mesh = P2PMesh(n_agents, k_peers=DEFAULTS["k_peers"], byzantine_ids=set(byzantine_ids))

    # Metrics collected per round
    history = {
        "round": [], "mean_reward_benign": [], "task_success": [],
        "tpr": [], "fpr": [], "consensus_decision": [],
        "n_detected_byzantine": [], "collapse_detected": [],
    }

    global_params = agents[0].get_state_dict()   # Initialise global model

    for t in range(args.rounds):
        t_start = time.time()

        # ── Phase 1: Local A2C training ────────────────────────────────────
        agent_rewards: List[float] = []
        all_state_dicts: Dict[int, dict] = {}
        grad_norms: Dict[int, float] = {}

        for i, (agent, env) in enumerate(zip(agents, envs)):
            # Load current global model
            agent.load_state_dict({k: v.clone() for k, v in global_params.items()})
            ep_reward, _, _ = agent.collect_trajectory(env)
            agent_rewards.append(ep_reward)
            all_state_dicts[i] = agent.get_state_dict()

            # Compute gradient norm (Phase 3 metric 7)
            total_norm = 0.0
            for p in agent.net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms[i] = total_norm ** 0.5

        # ── Apply Byzantine attacks ────────────────────────────────────────
        for bid in byzantine_ids:
            if args.attack == "ump":
                all_state_dicts[bid] = apply_ump_attack(all_state_dicts[bid])
            elif args.attack == "tmp":
                # Target: policy that maximally disagrees with global
                target = {k: -v for k, v in global_params.items()}
                all_state_dicts[bid] = apply_tmp_attack(all_state_dicts[bid], target)

        # ── Phase 2–3: P2P exchange + behavioral analysis ─────────────────
        all_suspicion: Dict[int, Dict[int, float]] = {}   # agent_i → {peer_j: score}

        for i in benign_ids:
            received = mesh.exchange_models(i, all_state_dicts, round_num=t)
            vt_states = agents[i].sample_replay_states(DEFAULTS["n_vt"])

            if len(vt_states) == 0 or vt_states.shape[0] == 0:
                all_suspicion[i] = {pid: 0.0 for pid in received}
                continue

            peer_metrics: Dict[int, np.ndarray] = {}
            for pid, peer_sd in received.items():
                # Load peer model temporarily
                tmp_net = copy.deepcopy(agents[i].net)
                try:
                    tmp_net.load_state_dict(peer_sd)
                except RuntimeError:
                    peer_metrics[pid] = np.ones(7) * 10.0   # malformed = max suspicion
                    continue

                m = compute_metrics(
                    agents[i].net, tmp_net, vt_states,
                    device=agents[i].device,
                    grad_norm_i=grad_norms[i],
                    grad_norm_j=grad_norms.get(pid),
                )
                peer_metrics[pid] = m

            suspicion = detector.update_and_score(i, peer_metrics)
            all_suspicion[i] = suspicion

        # ── Phase 4: Aggregate suspicion scores (mean over evaluating agents) ──
        agg_suspicion: Dict[int, List[float]] = {j: [] for j in range(n_agents)}
        for i, scores in all_suspicion.items():
            for j, s in scores.items():
                agg_suspicion[j].append(s)

        mean_suspicion: Dict[int, float] = {
            j: (float(np.mean(v)) if v else 0.0) for j, v in agg_suspicion.items()
        }

        # Filter Byzantine using dynamic threshold
        benign_filtered, byz_detected = filter_byzantine(mean_suspicion)

        # Detection metrics
        true_pos = len(set(byz_detected) & set(byzantine_ids))
        false_pos = len(set(byz_detected) & set(benign_ids))
        tpr = true_pos / max(len(byzantine_ids), 1)
        fpr = false_pos / max(len(benign_ids), 1)

        # ── Phase 5: Reputation update + aggregation ───────────────────────
        all_rewards = [agent_rewards[j] for j in benign_filtered]
        max_sus = max((mean_suspicion[j] for j in benign_filtered), default=1.0)

        for j in benign_filtered:
            r_norm, consistency, grad_align = compute_reputation_inputs(
                peer_reward=agent_rewards[j],
                all_peer_rewards=all_rewards,
                suspicion_score=mean_suspicion[j],
                max_suspicion=max_sus,
                grad_norm_peer=grad_norms.get(j),
                grad_norm_global=None,
            )
            reputation.update(j, r_norm, consistency, grad_align)

        trusted_state_dicts = {j: all_state_dicts[j] for j in benign_filtered}
        trusted_reputation = {j: reputation.get(j) for j in benign_filtered}
        trusted_suspicion = {j: mean_suspicion[j] for j in benign_filtered}

        new_global = reputation_weighted_aggregate(
            trusted_state_dicts, trusted_reputation, trusted_suspicion
        )

        # ── Phase 6: Collapse prevention ──────────────────────────────────
        mean_reward_benign = float(np.mean([agent_rewards[j] for j in benign_ids]))
        collapse_detected, recovered_params = monitor.update(t, mean_reward_benign, new_global)
        if collapse_detected and recovered_params is not None:
            new_global = recovered_params

        # ── Phase 7: Consensus + ledger ───────────────────────────────────
        decision, vote_count, threshold = consensus.run_vote(agent_rewards)
        if decision == 1:
            global_params = new_global
            ledger.record(t, global_params, mean_reward_benign, vote_count, n_agents)

        # Task success: fraction of benign agents with reward above 80th percentile baseline
        all_benign_rewards = np.array([agent_rewards[j] for j in benign_ids])
        env_reward_range = {
            "CartPole-v1": (0, 500), "LunarLander-v2": (-400, 300), "Pendulum-v1": (-1600, 0)
        }
        rmin, rmax = env_reward_range.get(args.env, (all_benign_rewards.min(), all_benign_rewards.max()))
        success_threshold = rmin + 0.6 * (rmax - rmin)
        task_success = float((all_benign_rewards >= success_threshold).mean())

        # Record
        history["round"].append(t)
        history["mean_reward_benign"].append(mean_reward_benign)
        history["task_success"].append(task_success)
        history["tpr"].append(tpr)
        history["fpr"].append(fpr)
        history["consensus_decision"].append(decision)
        history["n_detected_byzantine"].append(len(byz_detected))
        history["collapse_detected"].append(int(collapse_detected))

        elapsed = time.time() - t_start
        if (t + 1) % 10 == 0 or t == 0:
            print(f"  Round {t+1:3d}/{args.rounds} | "
                  f"reward={mean_reward_benign:7.1f} | "
                  f"TPR={tpr:.2f} FPR={fpr:.2f} | "
                  f"consensus={'PASS' if decision else 'FAIL'} ({vote_count}/{n_agents}) | "
                  f"collapse={'YES' if collapse_detected else 'no'} | "
                  f"{elapsed:.1f}s")

    # Cleanup
    for env in envs:
        env.close()

    # Summary statistics
    converged_tprs = history["tpr"][30:]   # post warm-up phase
    summary = {
        "env": args.env,
        "seed": seed,
        "n_agents": n_agents,
        "n_byzantine": n_byzantine,
        "attack": args.attack,
        "final_reward": float(np.mean(history["mean_reward_benign"][-10:])),
        "avg_tpr_converged": float(np.mean(converged_tprs)) if converged_tprs else 0.0,
        "avg_fpr": float(np.mean(history["fpr"])),
        "task_success_final": float(np.mean(history["task_success"][-10:])),
        "ledger_entries": len(ledger),
        "collapses_detected": sum(history["collapse_detected"]),
        "ledger_summary": ledger.summary(),
        "history": history,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="SecFedRL Training")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        choices=["CartPole-v1", "LunarLander-v2", "Pendulum-v1"])
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--byzantine_frac", type=float, default=0.2)
    parser.add_argument("--attack", type=str, default="ump", choices=["ump", "tmp", "none"])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--ablate", type=str, default=None,
                        choices=["detection", "reputation", "recovery", "consensus"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for seed in range(args.seeds):
        result = run_secfedrl(args, seed)
        all_results.append(result)
        print(f"\n  [seed={seed}] final_reward={result['final_reward']:.1f} | "
              f"avg_TPR={result['avg_tpr_converged']:.3f} | "
              f"task_success={result['task_success_final']:.3f}")

    # Aggregate across seeds
    final_rewards = [r["final_reward"] for r in all_results]
    avg_tprs = [r["avg_tpr_converged"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.env} | {args.n_agents} agents | {args.byzantine_frac*100:.0f}% Byzantine ({args.attack})")
    print(f"  Final Reward:   {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Avg TPR (conv): {np.mean(avg_tprs):.3f} ± {np.std(avg_tprs):.3f}")
    print(f"{'='*60}\n")

    # Save results
    out_path = os.path.join(
        args.output_dir,
        f"{args.env}_{args.attack}_{int(args.byzantine_frac*100)}byz_{args.seeds}seeds.json"
    )
    # Remove non-serialisable numpy arrays from history
    for r in all_results:
        for k, v in r["history"].items():
            r["history"][k] = [float(x) for x in v]
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
