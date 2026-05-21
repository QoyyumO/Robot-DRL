"""
Headless PPO training for vision-based robotic arm control.
Designed for Google Colab GPU (no display).

Usage:
    python train.py
    python train.py --resume
    python train.py --episodes 2000
"""

import os
import sys
import csv
import argparse
import signal
import time
import numpy as np
from datetime import datetime
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, _HERE)

from config_loader import cfg

_train_cfg = cfg["training"]
_ppo_cfg = cfg["ppo"]

LOGS_DIR = os.path.join(_ROOT, "logs")
MODELS_DIR = os.path.join(_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_model")

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    print("\n[train] Stop signal received — finishing rollout then saving.", flush=True)
    _stop_requested = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _save_training_csv(
    episodes: List[int],
    rewards: List[float],
    avg_losses: List[float],
    successes: List[int],
) -> Optional[str]:
    if not episodes:
        return None
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"ppo_training_metrics_{stamp}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "cumulative_reward", "avg_loss", "success"])
            for ep, rw, lo, su in zip(episodes, rewards, avg_losses, successes):
                writer.writerow([ep, f"{rw:.4f}", f"{lo:.6f}", su])
        return path
    except Exception as e:
        print(f"[train] Warning: could not save CSV: {e}", flush=True)
        return None


def _plot_training_metrics(
    episodes: List[int],
    rewards: List[float],
    avg_losses: List[float],
) -> Optional[str]:
    if not episodes:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(LOGS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"ppo_training_plot_{stamp}.png")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(episodes, rewards, linewidth=0.8, color="#2176AE")
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_title("PPO — Cumulative Reward per Episode")
        ax1.grid(True, alpha=0.3)
        ax2.plot(episodes, avg_losses, linewidth=0.8, color="#D7263D")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average PPO Loss")
        ax2.set_title("PPO — Training Loss per Episode")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as e:
        print(f"[train] Warning: could not save plot: {e}", flush=True)
        return None


def _setup_gpu() -> None:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[train] WARNING: No GPU detected. Running on CPU.", flush=True)
            print("        On Colab: Runtime → Change runtime type → T4 GPU", flush=True)
            return
        print(f"[train] TensorFlow detected {len(gpus)} GPU(s):", flush=True)
        for g in gpus:
            print(f"         {g}", flush=True)
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
        N = 2048
        with tf.device("/GPU:0"):
            a = tf.random.normal([N, N])
            b = tf.random.normal([N, N])
            _ = tf.matmul(a, b).numpy()
            t0 = time.monotonic()
            for _ in range(5):
                tf.matmul(a, b).numpy()
            gpu_ms = (time.monotonic() - t0) / 5 * 1000
        with tf.device("/CPU:0"):
            a_cpu = tf.random.normal([N, N])
            b_cpu = tf.random.normal([N, N])
            t0 = time.monotonic()
            tf.matmul(a_cpu, b_cpu).numpy()
            cpu_ms = (time.monotonic() - t0) * 1000
        print(
            f"[train] GPU compute check ({N}×{N} matmul): "
            f"GPU={gpu_ms:.1f} ms  CPU={cpu_ms:.1f} ms  "
            f"(speedup={cpu_ms / max(gpu_ms, 1e-6):.1f}×)",
            flush=True,
        )
    except ImportError:
        print("[train] TensorFlow not found.", flush=True)
    except Exception as e:
        print(f"[train] GPU setup warning: {e}", flush=True)


def _gpu_stats() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip()
        util, mem_used, mem_total = out.split(", ")
        return f"GPU {util}% util  {mem_used}/{mem_total} MB"
    except Exception:
        return ""


def _checkpoint_path() -> str:
    return f"{MODEL_PATH}.keras"


def train(args: argparse.Namespace) -> None:
    global _stop_requested

    n_envs = args.n_envs
    n_steps = _ppo_cfg["n_steps"]
    mixed_precision = _train_cfg["mixed_precision"]

    from vec_env import SubprocVecEnv
    print(f"[train] Starting {n_envs} environment worker(s)…", flush=True)
    vec_env = SubprocVecEnv(n_envs)

    _setup_gpu()

    from agent import PPOAgent

    print(
        f"[train] PPO: n_envs={n_envs}  n_steps={n_steps}  "
        f"batch={_ppo_cfg['batch_size']}  mixed_precision={mixed_precision}",
        flush=True,
    )

    try:
        agent = PPOAgent(
            n_actions=vec_env.n_actions,
            obs_shape=vec_env.observation_shape,
            mixed_precision=mixed_precision,
        )

        resume_path = (
            MODEL_PATH if os.path.isfile(MODEL_PATH)
            else f"{MODEL_PATH}.keras" if os.path.isfile(f"{MODEL_PATH}.keras")
            else None
        )
        if args.resume and resume_path:
            try:
                agent.load(resume_path)
                print(f"[train] Resumed from {resume_path}", flush=True)
            except Exception as e:
                print(f"[train] Could not load checkpoint: {e} — scratch", flush=True)
        else:
            print("[train] Training from scratch.", flush=True)

        obs_batch = vec_env.reset()
        ep_rewards = np.zeros(n_envs)
        episode = 0
        global_step = 0
        success_count = 0
        recent_losses: List[float] = []

        history_ep: List[int] = []
        history_reward: List[float] = []
        history_avg_loss: List[float] = []
        history_success: List[int] = []

        CHECKPOINT_EVERY = args.checkpoint_every
        LOG_EVERY = args.log_every
        max_episodes = args.episodes
        last_entropy = 0.0

        _perf_t0 = time.monotonic()
        _perf_steps = 0
        PERF_INTERVAL = 200

        print(
            f"[train] Starting. "
            f"max_episodes={'unlimited' if max_episodes == 0 else max_episodes} | "
            f"checkpoint_every={CHECKPOINT_EVERY} | log_every={LOG_EVERY}",
            flush=True,
        )

        while not _stop_requested:
            if max_episodes > 0 and episode >= max_episodes:
                print(f"[train] Reached {max_episodes} episodes.", flush=True)
                break

            # ── Collect rollout (n_steps × n_envs transitions) ───────────────
            rollout_obs = []
            rollout_actions = []
            rollout_rewards = []
            rollout_dones = []
            rollout_log_probs = []
            rollout_values = []

            for _ in range(n_steps):
                actions, log_probs, values = agent.act_batch(obs_batch, training=True)
                next_obs, rewards, dones, truncs, infos = vec_env.step(actions)

                rollout_obs.append(obs_batch.copy())
                rollout_actions.append(actions.copy())
                rollout_rewards.append(rewards.copy())
                rollout_dones.append((dones | truncs).astype(np.float32))
                rollout_log_probs.append(log_probs.copy())
                rollout_values.append(values.copy())

                ep_rewards += rewards
                obs_batch = next_obs
                global_step += n_envs
                _perf_steps += n_envs

                for i, info in enumerate(infos):
                    if info.get("episode_done"):
                        episode += 1
                        if info.get("success"):
                            success_count += 1
                        avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                        history_ep.append(episode)
                        history_reward.append(float(ep_rewards[i]))
                        history_avg_loss.append(avg_loss)
                        history_success.append(1 if info.get("success") else 0)

                        if episode % LOG_EVERY == 0:
                            print(
                                f"[train] ep={episode:>6}  reward={ep_rewards[i]:>8.2f}  "
                                f"avg_loss={avg_loss:.5f}  success={success_count}  "
                                f"H={last_entropy:.4f}  dist={info.get('distance', 0):.3f}",
                                flush=True,
                            )
                        ep_rewards[i] = 0.0

                        if CHECKPOINT_EVERY > 0 and episode % CHECKPOINT_EVERY == 0:
                            os.makedirs(MODELS_DIR, exist_ok=True)
                            agent.save(MODEL_PATH)
                            print(
                                f"[train] Checkpoint saved → {_checkpoint_path()}",
                                flush=True,
                            )

                if _perf_steps >= PERF_INTERVAL:
                    elapsed = time.monotonic() - _perf_t0
                    sps = _perf_steps / elapsed if elapsed > 0 else 0.0
                    stats = _gpu_stats()
                    print(
                        f"[train] {global_step:>8} steps | {sps:>6.1f} steps/sec"
                        + (f"  |  {stats}" if stats else ""),
                        flush=True,
                    )
                    _perf_t0 = time.monotonic()
                    _perf_steps = 0

            _, _, last_values = agent.act_batch(obs_batch, training=False)

            T, N = n_steps, n_envs
            obs_stacked = np.stack(rollout_obs, axis=0)       # (T, N, H, W)
            obs_flat = obs_stacked.reshape(T * N, *obs_stacked.shape[2:])
            actions_flat = np.stack(rollout_actions, axis=0).reshape(T * N)
            log_probs_flat = np.stack(rollout_log_probs, axis=0).reshape(T * N)

            all_adv = []
            all_ret = []
            for e in range(N):
                r_e = np.array([rollout_rewards[t][e] for t in range(T)], dtype=np.float32)
                v_e = np.array([rollout_values[t][e] for t in range(T)], dtype=np.float32)
                d_e = np.array([rollout_dones[t][e] for t in range(T)], dtype=np.float32)
                adv_e, ret_e = agent.compute_gae(
                    r_e, v_e, d_e, float(last_values[e]),
                    agent.gamma, agent.gae_lambda,
                )
                all_adv.append(adv_e)
                all_ret.append(ret_e)

            # Interleave env trajectories: (t0,e0), (t0,e1), … matches reshape order
            advantages = np.zeros(T * N, dtype=np.float32)
            returns = np.zeros(T * N, dtype=np.float32)
            for e in range(N):
                advantages[e::N] = all_adv[e]
                returns[e::N] = all_ret[e]

            obs_4d = obs_flat[:, :, :, np.newaxis].astype(np.float32)
            metrics = agent.train_on_rollout(
                obs_4d, actions_flat, log_probs_flat, advantages, returns
            )
            if metrics:
                recent_losses.append(float(metrics.get("loss", 0.0)))
                last_entropy = float(metrics.get("entropy", last_entropy))
                if len(recent_losses) > 50:
                    recent_losses.pop(0)

        os.makedirs(MODELS_DIR, exist_ok=True)
        agent.save(MODEL_PATH)
        print(f"[train] Model saved → {_checkpoint_path()}", flush=True)

        csv_path = _save_training_csv(
            history_ep, history_reward, history_avg_loss, history_success
        )
        plot_path = _plot_training_metrics(history_ep, history_reward, history_avg_loss)

        print(
            f"[train] Done. episodes={episode}  env_steps={global_step}  "
            f"successes={success_count}",
            flush=True,
        )
        if csv_path:
            print(f"[train] Metrics CSV → {csv_path}", flush=True)
        if plot_path:
            print(f"[train] Plot       → {plot_path}", flush=True)

    except Exception as e:
        print(f"[train] Fatal error: {e}", flush=True)
        raise
    finally:
        vec_env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless PPO training for Robot-DRL")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--episodes", type=int, default=0, metavar="N",
                        help="Stop after N episodes (0 = unlimited)")
    parser.add_argument("--n-envs", type=int, default=_train_cfg.get("n_envs", 4),
                        dest="n_envs", metavar="N")
    parser.add_argument("--checkpoint-every", type=int,
                        default=_train_cfg["checkpoint_every"], dest="checkpoint_every")
    parser.add_argument("--log-every", type=int,
                        default=_train_cfg["log_every"], dest="log_every")
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
