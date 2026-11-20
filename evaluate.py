"""
SecFedRL Evaluation Script
Loads saved results and reports publication-ready metrics.
"""

import argparse
import json
import os
import numpy as np


def report(result_file: str):
    with open(result_file) as f:
        results = json.load(f)

    env = results[0]["env"]
    attack = results[0]["attack"]
    n_byz = results[0]["n_byzantine"]
    n_agents = results[0]["n_agents"]
    seeds = len(results)

    final_rewards = [r["final_reward"] for r in results]
    tprs = [r["avg_tpr_converged"] for r in results]
    fprs = [r["avg_fpr"] for r in results]
    success = [r["task_success_final"] for r in results]

    print(f"\n{'='*65}")
    print(f"SecFedRL Evaluation: {env}")
    print(f"  Agents: {n_agents} | Byzantine: {n_byz} | Attack: {attack} | Seeds: {seeds}")
    print(f"{'='*65}")
    print(f"  Final Episodic Reward:      {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Avg Per-Round TPR (conv):   {np.mean(tprs)*100:.1f}% ± {np.std(tprs)*100:.1f}%")
    print(f"  Avg FPR:                    {np.mean(fprs)*100:.2f}%")
    print(f"  Task Success Rate:          {np.mean(success)*100:.1f}% ± {np.std(success)*100:.1f}%")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", help="Path to results JSON from main.py")
    args = parser.parse_args()
    report(args.result_file)


if __name__ == "__main__":
    main()
