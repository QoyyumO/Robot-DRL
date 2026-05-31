"""
Structured evaluation of the trained DQN agent and random-policy baseline.

Runs N_RUNS independent evaluation batches of EPISODES_PER_RUN episodes each.
Records per-episode metrics and computes per-run and overall averages for:
  - Cumulative reward           (RL metric, Section 2.11.1)
  - Episode length              (RL metric, Section 2.11.1)
  - Success rate %              (Robotics metric, Section 2.11.2)
  - Average final distance      (End-effector error, Section 2.11.2)
  - Average reward per step     (RL metric, Section 2.11.1)
  - Trajectory smoothness       (Robotics metric, Section 2.11.2)
    Measured as mean absolute step-to-step distance change (lower = smoother).

Usage:
    python evaluate.py                          # greedy DQN (default)
    python evaluate.py --policy random          # random baseline only
    python evaluate.py --policy both            # DQN + random comparison
    python evaluate.py --runs 30 --episodes 100
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Callable, Optional

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from environment import VisionControlEnv

N_RUNS = 30
EPISODES_PER_RUN = 100
MODEL_PATH_KERAS = "models/dqn_model.keras"
MODEL_PATH_H5 = "models/dqn_model.h5"
LOGS_DIR = "logs"

# Two-tailed t critical values at 95% confidence (df = n - 1).
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    return 1.96 if df >= 30 else _T_CRIT_95[30]


def _mean_std_ci(values: list[float]) -> dict:
    """Return mean, sample std (ddof=1), and 95% CI for a list of per-run values."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(np.std(arr, ddof=1))
    margin = _t_crit_95(n - 1) * std / np.sqrt(n)
    return {"mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin}


def _resolve_model_path() -> str:
    if os.path.isfile(MODEL_PATH_KERAS):
        return MODEL_PATH_KERAS
    if os.path.isfile(MODEL_PATH_H5):
        return MODEL_PATH_H5
    print(f"ERROR: No model found at {MODEL_PATH_KERAS} or {MODEL_PATH_H5}")
    sys.exit(1)


def _make_greedy_action_fn(model_path: str, n_actions: int, obs_shape):
    from agent import DQNAgent

    agent = DQNAgent(n_actions=n_actions, obs_shape=obs_shape, buffer_size=100)
    agent.load(model_path)
    agent.epsilon = 0.0

    def select_action(obs):
        return agent.select_action(obs, training=False)

    return select_action


def _make_random_action_fn(n_actions: int, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)

    def select_action(_obs):
        return int(rng.integers(0, n_actions))

    return select_action


def run_single_episode(env: VisionControlEnv, select_action: Callable):
    """Run one episode. Returns dict of episode metrics."""
    obs, info = env.reset()
    cumulative_reward = 0.0
    steps = 0
    distances = []
    success = False

    initial_dist = info.get("distance", env._distance_ee_to_target())
    distances.append(initial_dist)

    while True:
        action = select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward
        steps += 1
        dist = info.get("distance", 0.0)
        distances.append(dist)
        if info.get("success"):
            success = True
        if done or truncated:
            break

    final_distance = distances[-1]

    step_changes = [abs(distances[i + 1] - distances[i]) for i in range(len(distances) - 1)]
    trajectory_smoothness = float(np.mean(step_changes)) if step_changes else 0.0
    avg_reward_per_step = cumulative_reward / max(steps, 1)

    return {
        "cumulative_reward": cumulative_reward,
        "steps": steps,
        "final_distance": final_distance,
        "initial_distance": initial_dist,
        "success": success,
        "avg_reward_per_step": avg_reward_per_step,
        "trajectory_smoothness": trajectory_smoothness,
    }


def _summarize_run(run_episodes: list[dict], run_idx: int, episodes_per_run: int) -> dict:
    rewards = [e["cumulative_reward"] for e in run_episodes]
    steps_list = [e["steps"] for e in run_episodes]
    final_dists = [e["final_distance"] for e in run_episodes]
    successes = sum(1 for e in run_episodes if e["success"])
    rps = [e["avg_reward_per_step"] for e in run_episodes]
    smoothness = [e["trajectory_smoothness"] for e in run_episodes]

    return {
        "run": run_idx + 1,
        "episodes": episodes_per_run,
        "successes": successes,
        "success_rate_pct": (successes / episodes_per_run) * 100,
        "avg_cumulative_reward": float(np.mean(rewards)),
        "std_cumulative_reward": float(np.std(rewards)),
        "avg_episode_length": float(np.mean(steps_list)),
        "avg_final_distance_m": float(np.mean(final_dists)),
        "std_final_distance_m": float(np.std(final_dists)),
        "avg_reward_per_step": float(np.mean(rps)),
        "avg_trajectory_smoothness": float(np.mean(smoothness)),
    }


def _compute_stats(run_summaries: list[dict], n_runs: int, total_episodes: int) -> dict:
    all_sr = [s["success_rate_pct"] for s in run_summaries]
    all_ar = [s["avg_cumulative_reward"] for s in run_summaries]
    all_el = [s["avg_episode_length"] for s in run_summaries]
    all_fd = [s["avg_final_distance_m"] for s in run_summaries]
    all_rps = [s["avg_reward_per_step"] for s in run_summaries]
    all_sm = [s["avg_trajectory_smoothness"] for s in run_summaries]
    total_successes = sum(s["successes"] for s in run_summaries)

    return {
        "success_rate_pct": _mean_std_ci(all_sr),
        "avg_cumulative_reward": _mean_std_ci(all_ar),
        "avg_episode_length": _mean_std_ci(all_el),
        "avg_final_distance_m": _mean_std_ci(all_fd),
        "avg_reward_per_step": _mean_std_ci(all_rps),
        "avg_trajectory_smoothness": _mean_std_ci(all_sm),
        "total_successes": total_successes,
        "episode_success_rate_pct": (total_successes / total_episodes) * 100,
    }


def evaluate_policy(
    env: VisionControlEnv,
    select_action: Callable,
    policy_label: str,
    n_runs: int,
    episodes_per_run: int,
    stamp: str,
    seed: Optional[int] = None,
) -> tuple[list[dict], list[dict], dict]:
    """Run full evaluation for one policy. Returns episodes, run summaries, and stats."""
    total_episodes = n_runs * episodes_per_run
    slug = policy_label.lower().replace(" ", "_")
    detail_csv = os.path.join(LOGS_DIR, f"eval_{slug}_detail_{stamp}.csv")
    summary_csv = os.path.join(LOGS_DIR, f"eval_{slug}_summary_{stamp}.csv")
    stats_csv = os.path.join(LOGS_DIR, f"eval_{slug}_stats_{stamp}.csv")

    print(f"\n{'=' * 70}")
    print(f"POLICY: {policy_label}")
    print(f"{'=' * 70}")
    print(f"Evaluation: {n_runs} runs x {episodes_per_run} episodes = {total_episodes} total episodes")
    if policy_label.lower() == "random policy":
        print("Action selection: uniform random over all discrete actions (epsilon = 1.0 equivalent)")
    else:
        print("Action selection: greedy (epsilon = 0)")
    print()

    all_episodes = []
    run_summaries = []

    for run_idx in range(n_runs):
        if seed is not None:
            np.random.seed(seed + run_idx)

        run_episodes = []
        for ep_idx in range(episodes_per_run):
            result = run_single_episode(env, select_action)
            result["run"] = run_idx + 1
            result["episode"] = ep_idx + 1
            run_episodes.append(result)
            total_done = run_idx * episodes_per_run + ep_idx + 1
            if total_done % 50 == 0 or total_done == 1:
                print(f"  Progress: {total_done}/{total_episodes} episodes completed")

        summary = _summarize_run(run_episodes, run_idx, episodes_per_run)
        run_summaries.append(summary)
        all_episodes.extend(run_episodes)

        print(f"  Run {run_idx + 1:2d}: "
              f"Success={summary['successes']}/{episodes_per_run} ({summary['success_rate_pct']:.1f}%) | "
              f"AvgReward={summary['avg_cumulative_reward']:.1f} | "
              f"AvgDist={summary['avg_final_distance_m']:.4f}m | "
              f"AvgSteps={summary['avg_episode_length']:.1f} | "
              f"Smoothness={summary['avg_trajectory_smoothness']:.4f}")

    stats = _compute_stats(run_summaries, n_runs, total_episodes)
    _write_csvs(all_episodes, run_summaries, stats, n_runs, episodes_per_run,
                detail_csv, summary_csv, stats_csv)
    _print_summary_table(run_summaries, stats, episodes_per_run, n_runs, policy_label)

    print(f"\nDetail CSV : {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Stats CSV  : {stats_csv}")

    return all_episodes, run_summaries, stats


def _write_csvs(all_episodes, run_summaries, stats, n_runs, episodes_per_run,
                detail_csv, summary_csv, stats_csv):
    with open(detail_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "episode", "cumulative_reward", "steps",
                         "final_distance", "initial_distance", "success",
                         "avg_reward_per_step", "trajectory_smoothness"])
        for e in all_episodes:
            writer.writerow([
                e["run"], e["episode"],
                f"{e['cumulative_reward']:.4f}", e["steps"],
                f"{e['final_distance']:.6f}", f"{e['initial_distance']:.6f}",
                1 if e["success"] else 0,
                f"{e['avg_reward_per_step']:.4f}",
                f"{e['trajectory_smoothness']:.6f}",
            ])

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "episodes", "successes", "success_rate_pct",
                         "avg_cumulative_reward", "std_cumulative_reward",
                         "avg_episode_length", "avg_final_distance_m",
                         "std_final_distance_m", "avg_reward_per_step",
                         "avg_trajectory_smoothness"])
        for s in run_summaries:
            writer.writerow([
                s["run"], s["episodes"], s["successes"],
                f"{s['success_rate_pct']:.1f}",
                f"{s['avg_cumulative_reward']:.2f}",
                f"{s['std_cumulative_reward']:.2f}",
                f"{s['avg_episode_length']:.1f}",
                f"{s['avg_final_distance_m']:.4f}",
                f"{s['std_final_distance_m']:.4f}",
                f"{s['avg_reward_per_step']:.4f}",
                f"{s['avg_trajectory_smoothness']:.6f}",
            ])

        writer.writerow([
            "MEAN", episodes_per_run,
            f"{stats['total_successes'] / n_runs:.1f}",
            f"{stats['success_rate_pct']['mean']:.1f}",
            f"{stats['avg_cumulative_reward']['mean']:.2f}",
            f"{stats['avg_cumulative_reward']['std']:.2f}",
            f"{stats['avg_episode_length']['mean']:.1f}",
            f"{stats['avg_final_distance_m']['mean']:.4f}",
            f"{stats['avg_final_distance_m']['std']:.4f}",
            f"{stats['avg_reward_per_step']['mean']:.4f}",
            f"{stats['avg_trajectory_smoothness']['mean']:.6f}",
        ])

    with open(stats_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "ci_95_low", "ci_95_high", "n_runs"])
        for name, s in stats.items():
            if name in ("total_successes", "episode_success_rate_pct"):
                continue
            writer.writerow([
                name,
                f"{s['mean']:.4f}",
                f"{s['std']:.4f}",
                f"{s['ci_low']:.4f}",
                f"{s['ci_high']:.4f}",
                n_runs,
            ])
        writer.writerow([
            "episode_success_rate_pct",
            f"{stats['episode_success_rate_pct']:.4f}",
            "",
            "",
            "",
            n_runs * episodes_per_run,
        ])


def _print_summary_table(run_summaries, stats, episodes_per_run, n_runs, policy_label):
    print("\n" + "=" * 110)
    print(f"EVALUATION SUMMARY — {policy_label.upper()}")
    print("=" * 110)
    print(f"{'Run':<6} {'Episodes':<10} {'Successes':<11} {'Success%':<10} "
          f"{'AvgReward':<12} {'AvgSteps':<10} {'AvgDist(m)':<12} "
          f"{'Reward/Step':<13} {'Smoothness':<12}")
    print("-" * 110)
    for s in run_summaries:
        print(f"{s['run']:<6} {s['episodes']:<10} {s['successes']:<11} "
              f"{s['success_rate_pct']:<10.1f} "
              f"{s['avg_cumulative_reward']:<12.2f} "
              f"{s['avg_episode_length']:<10.1f} "
              f"{s['avg_final_distance_m']:<12.4f} "
              f"{s['avg_reward_per_step']:<13.4f} "
              f"{s['avg_trajectory_smoothness']:<12.6f}")
    print("-" * 110)
    print(f"{'MEAN':<6} {episodes_per_run:<10} {stats['total_successes'] / n_runs:<11.1f} "
          f"{stats['success_rate_pct']['mean']:<10.1f} "
          f"{stats['avg_cumulative_reward']['mean']:<12.2f} "
          f"{stats['avg_episode_length']['mean']:<10.1f} "
          f"{stats['avg_final_distance_m']['mean']:<12.4f} "
          f"{stats['avg_reward_per_step']['mean']:<13.4f} "
          f"{stats['avg_trajectory_smoothness']['mean']:<12.6f}")
    print("=" * 110)

    print("\nAGGREGATE STATISTICS (across independent runs, 95% CI)")
    print("-" * 70)
    sr = stats["success_rate_pct"]
    ar = stats["avg_cumulative_reward"]
    fd = stats["avg_final_distance_m"]
    el = stats["avg_episode_length"]
    print(f"  Success rate (per-run mean): "
          f"{sr['mean']:.2f}% ± {sr['std']:.2f}%  "
          f"[{sr['ci_low']:.2f}%, {sr['ci_high']:.2f}%]")
    print(f"  Episode success rate (pooled): {stats['episode_success_rate_pct']:.2f}% "
          f"({stats['total_successes']}/{n_runs * episodes_per_run})")
    print(f"  Avg cumulative reward: "
          f"{ar['mean']:.2f} ± {ar['std']:.2f}  "
          f"[{ar['ci_low']:.2f}, {ar['ci_high']:.2f}]")
    print(f"  Avg final distance (m): "
          f"{fd['mean']:.4f} ± {fd['std']:.4f}  "
          f"[{fd['ci_low']:.4f}, {fd['ci_high']:.4f}]")
    print(f"  Avg episode length: "
          f"{el['mean']:.1f} ± {el['std']:.1f}  "
          f"[{el['ci_low']:.1f}, {el['ci_high']:.1f}]")


def _write_comparison_csv(path: str, results: dict[str, dict]):
    """Write side-by-side policy comparison for thesis Table 4.13."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy",
            "success_rate_pct_mean",
            "success_rate_pct_std",
            "success_rate_pct_ci_95_low",
            "success_rate_pct_ci_95_high",
            "episode_success_rate_pct_pooled",
            "total_successes",
            "total_episodes",
            "avg_cumulative_reward_mean",
            "avg_cumulative_reward_std",
            "avg_final_distance_m_mean",
            "avg_final_distance_m_std",
            "avg_episode_length_mean",
        ])
        for policy_name, stats in results.items():
            sr = stats["success_rate_pct"]
            ar = stats["avg_cumulative_reward"]
            fd = stats["avg_final_distance_m"]
            el = stats["avg_episode_length"]
            total_episodes = stats.get("total_episodes", 0)
            writer.writerow([
                policy_name,
                f"{sr['mean']:.2f}",
                f"{sr['std']:.2f}",
                f"{sr['ci_low']:.2f}",
                f"{sr['ci_high']:.2f}",
                f"{stats['episode_success_rate_pct']:.2f}",
                stats["total_successes"],
                total_episodes,
                f"{ar['mean']:.2f}",
                f"{ar['std']:.2f}",
                f"{fd['mean']:.4f}",
                f"{fd['std']:.4f}",
                f"{el['mean']:.1f}",
            ])


def _print_comparison_table(results: dict[str, dict]):
    print("\n" + "=" * 90)
    print("POLICY COMPARISON (for thesis baseline table)")
    print("=" * 90)
    print(f"{'Policy':<18} {'Success%':<12} {'Pooled%':<10} {'Avg Reward':<14} "
          f"{'Avg Dist (m)':<14} {'Avg Steps':<10}")
    print("-" * 90)
    for policy_name, stats in results.items():
        sr = stats["success_rate_pct"]
        ar = stats["avg_cumulative_reward"]
        fd = stats["avg_final_distance_m"]
        el = stats["avg_episode_length"]
        print(f"{policy_name:<18} "
              f"{sr['mean']:.2f}±{sr['std']:.2f}  "
              f"{stats['episode_success_rate_pct']:<10.2f} "
              f"{ar['mean']:<14.2f} "
              f"{fd['mean']:<14.4f} "
              f"{el['mean']:<10.1f}")
    print("=" * 90)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN and/or random-policy baseline."
    )
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Number of independent evaluation runs (default: {N_RUNS})")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_RUN,
                        help=f"Episodes per run (default: {EPISODES_PER_RUN})")
    parser.add_argument(
        "--policy",
        choices=["greedy", "random", "both"],
        default="greedy",
        help="greedy = trained DQN; random = uniform random baseline; both = run and compare",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for random-policy runs (default: 42)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    n_runs = args.runs
    episodes_per_run = args.episodes
    policies = ["greedy", "random"] if args.policy == "both" else [args.policy]

    os.makedirs(LOGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = VisionControlEnv(headless=True)

    comparison_results = {}

    try:
        if "random" in policies:
            random_fn = _make_random_action_fn(env.n_actions, seed=args.seed)
            _, _, random_stats = evaluate_policy(
                env, random_fn, "Random Policy", n_runs, episodes_per_run, stamp, seed=args.seed,
            )
            random_stats["total_episodes"] = n_runs * episodes_per_run
            comparison_results["Random Policy"] = random_stats

        if "greedy" in policies:
            model_path = _resolve_model_path()
            print(f"\nLoading model from: {model_path}")
            greedy_fn = _make_greedy_action_fn(
                model_path, env.n_actions, env.observation_shape,
            )
            _, _, greedy_stats = evaluate_policy(
                env, greedy_fn, "Trained DQN", n_runs, episodes_per_run, stamp,
            )
            greedy_stats["total_episodes"] = n_runs * episodes_per_run
            comparison_results["Trained DQN"] = greedy_stats

        if len(comparison_results) == 2:
            comparison_csv = os.path.join(LOGS_DIR, f"eval_baseline_comparison_{stamp}.csv")
            _write_comparison_csv(comparison_csv, comparison_results)
            _print_comparison_table(comparison_results)
            print(f"\nComparison CSV: {comparison_csv}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
