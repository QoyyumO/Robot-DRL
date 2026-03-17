"""
Structured evaluation of the trained DQN agent.

Runs N_RUNS independent evaluation batches of EPISODES_PER_RUN episodes each,
using a purely greedy policy (epsilon = 0). Records per-episode metrics and
computes per-run and overall averages for:
  - Cumulative reward           (RL metric, Section 2.11.1)
  - Episode length              (RL metric, Section 2.11.1)
  - Success rate %              (Robotics metric, Section 2.11.2)
  - Average final distance      (End-effector error, Section 2.11.2)
  - Average reward per step     (RL metric, Section 2.11.1)
  - Trajectory smoothness       (Robotics metric, Section 2.11.2)
    Measured as mean absolute step-to-step distance change (lower = smoother).

Usage:
    python evaluate.py
"""

import os
import csv
import sys
import numpy as np
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from environment import VisionControlEnv
from agent import DQNAgent

N_RUNS = 10
EPISODES_PER_RUN = 100
MODEL_PATH_KERAS = "models/dqn_model.keras"
MODEL_PATH_H5 = "models/dqn_model.h5"
LOGS_DIR = "logs"


def _resolve_model_path() -> str:
    if os.path.isfile(MODEL_PATH_KERAS):
        return MODEL_PATH_KERAS
    if os.path.isfile(MODEL_PATH_H5):
        return MODEL_PATH_H5
    print(f"ERROR: No model found at {MODEL_PATH_KERAS} or {MODEL_PATH_H5}")
    sys.exit(1)


def run_single_episode(env: VisionControlEnv, agent: DQNAgent):
    """Run one greedy episode. Returns dict of episode metrics."""
    obs, info = env.reset()
    cumulative_reward = 0.0
    steps = 0
    distances = []
    success = False

    initial_dist = info.get("distance", env._distance_ee_to_target())
    distances.append(initial_dist)

    while True:
        action = agent.select_action(obs, training=False)
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


def main():
    model_path = _resolve_model_path()
    print(f"Loading model from: {model_path}")
    print(f"Evaluation: {N_RUNS} runs x {EPISODES_PER_RUN} episodes = {N_RUNS * EPISODES_PER_RUN} total episodes")
    print(f"Policy: greedy (epsilon = 0)\n")

    env = VisionControlEnv(headless=True)
    agent = DQNAgent(n_actions=env.n_actions, obs_shape=env.observation_shape,
                     buffer_size=100)
    agent.load(model_path)
    agent.epsilon = 0.0

    os.makedirs(LOGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_csv = os.path.join(LOGS_DIR, f"eval_detail_{stamp}.csv")
    summary_csv = os.path.join(LOGS_DIR, f"eval_summary_{stamp}.csv")

    all_episodes = []
    run_summaries = []

    for run_idx in range(N_RUNS):
        run_episodes = []
        for ep_idx in range(EPISODES_PER_RUN):
            result = run_single_episode(env, agent)
            result["run"] = run_idx + 1
            result["episode"] = ep_idx + 1
            run_episodes.append(result)
            total_done = run_idx * EPISODES_PER_RUN + ep_idx + 1
            if total_done % 25 == 0 or total_done == 1:
                print(f"  Progress: {total_done}/{N_RUNS * EPISODES_PER_RUN} episodes completed")

        rewards = [e["cumulative_reward"] for e in run_episodes]
        steps_list = [e["steps"] for e in run_episodes]
        final_dists = [e["final_distance"] for e in run_episodes]
        successes = sum(1 for e in run_episodes if e["success"])
        rps = [e["avg_reward_per_step"] for e in run_episodes]
        smoothness = [e["trajectory_smoothness"] for e in run_episodes]

        summary = {
            "run": run_idx + 1,
            "episodes": EPISODES_PER_RUN,
            "successes": successes,
            "success_rate_pct": (successes / EPISODES_PER_RUN) * 100,
            "avg_cumulative_reward": float(np.mean(rewards)),
            "std_cumulative_reward": float(np.std(rewards)),
            "avg_episode_length": float(np.mean(steps_list)),
            "avg_final_distance_m": float(np.mean(final_dists)),
            "std_final_distance_m": float(np.std(final_dists)),
            "avg_reward_per_step": float(np.mean(rps)),
            "avg_trajectory_smoothness": float(np.mean(smoothness)),
        }
        run_summaries.append(summary)
        all_episodes.extend(run_episodes)

        print(f"  Run {run_idx + 1:2d}: "
              f"Success={successes}/{EPISODES_PER_RUN} ({summary['success_rate_pct']:.1f}%) | "
              f"AvgReward={summary['avg_cumulative_reward']:.1f} | "
              f"AvgDist={summary['avg_final_distance_m']:.4f}m | "
              f"AvgSteps={summary['avg_episode_length']:.1f} | "
              f"Smoothness={summary['avg_trajectory_smoothness']:.4f}")

    # --- Per-episode detail CSV ---
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

    # --- Per-run summary CSV ---
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

        # Overall averages row
        all_sr = [s["success_rate_pct"] for s in run_summaries]
        all_ar = [s["avg_cumulative_reward"] for s in run_summaries]
        all_el = [s["avg_episode_length"] for s in run_summaries]
        all_fd = [s["avg_final_distance_m"] for s in run_summaries]
        all_rps = [s["avg_reward_per_step"] for s in run_summaries]
        all_sm = [s["avg_trajectory_smoothness"] for s in run_summaries]
        total_successes = sum(s["successes"] for s in run_summaries)
        writer.writerow([
            "AVERAGE", EPISODES_PER_RUN,
            f"{total_successes / N_RUNS:.1f}",
            f"{np.mean(all_sr):.1f}",
            f"{np.mean(all_ar):.2f}",
            f"{np.mean([s['std_cumulative_reward'] for s in run_summaries]):.2f}",
            f"{np.mean(all_el):.1f}",
            f"{np.mean(all_fd):.4f}",
            f"{np.mean([s['std_final_distance_m'] for s in run_summaries]):.4f}",
            f"{np.mean(all_rps):.4f}",
            f"{np.mean(all_sm):.6f}",
        ])

    env.close()

    # --- Print final summary table ---
    print("\n" + "=" * 110)
    print("EVALUATION SUMMARY")
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

    avg_sr = np.mean(all_sr)
    avg_ar = np.mean(all_ar)
    avg_el = np.mean(all_el)
    avg_fd = np.mean(all_fd)
    avg_rps_val = np.mean(all_rps)
    avg_sm = np.mean(all_sm)
    print(f"{'AVG':<6} {EPISODES_PER_RUN:<10} {total_successes / N_RUNS:<11.1f} "
          f"{avg_sr:<10.1f} {avg_ar:<12.2f} {avg_el:<10.1f} "
          f"{avg_fd:<12.4f} {avg_rps_val:<13.4f} {avg_sm:<12.6f}")
    print("=" * 110)
    print(f"\nDetail CSV : {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"\nTotal episodes: {N_RUNS * EPISODES_PER_RUN}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {avg_sr:.2f}%")
    print(f"Overall avg final distance: {avg_fd:.4f} m")
    print(f"Overall avg cumulative reward: {avg_ar:.2f}")
    print(f"Overall avg trajectory smoothness: {avg_sm:.6f}")


if __name__ == "__main__":
    main()
