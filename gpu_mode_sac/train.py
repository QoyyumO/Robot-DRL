"""
Headless SAC training for vision-based robotic arm control.
Discrete-action Soft Actor-Critic (Haarnoja et al., 2018).

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
_sac_cfg = cfg["sac"]

LOGS_DIR = os.path.join(_ROOT, "logs")
MODELS_DIR = os.path.join(_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sac_model")

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    print("\n[train] Stop signal received — finishing current episode then saving.", flush=True)
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
        path = os.path.join(LOGS_DIR, f"sac_training_metrics_{stamp}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "cumulative_reward", "avg_critic_loss", "success"])
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
        path = os.path.join(LOGS_DIR, f"sac_training_plot_{stamp}.png")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(episodes, rewards, linewidth=0.8, color="#2176AE")
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_title("SAC — Cumulative Reward per Episode")
        ax1.grid(True, alpha=0.3)
        ax2.plot(episodes, avg_losses, linewidth=0.8, color="#D7263D")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Critic Loss")
        ax2.set_title("SAC — Critic Loss per Episode")
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
            return
        print(f"[train] TensorFlow detected {len(gpus)} GPU(s).", flush=True)
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
    except ImportError:
        print("[train] TensorFlow not found.", flush=True)


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


def train(args: argparse.Namespace) -> None:
    global _stop_requested

    n_envs = args.n_envs
    train_freq = _sac_cfg["train_freq"]
    gradient_steps = _sac_cfg["gradient_steps"]
    mixed_precision = _train_cfg["mixed_precision"]

    from vec_env import SubprocVecEnv
    print(f"[train] Starting {n_envs} environment worker(s)…", flush=True)
    vec_env = SubprocVecEnv(n_envs)

    _setup_gpu()

    from agent import SACAgent

    print(
        f"[train] SAC: n_envs={n_envs}  batch={_sac_cfg['batch_size']}  "
        f"train_freq={train_freq}  gradient_steps={gradient_steps}  "
        f"alpha={_sac_cfg['alpha']}  mixed_precision={mixed_precision}",
        flush=True,
    )

    try:
        agent = SACAgent(
            n_actions=vec_env.n_actions,
            obs_shape=vec_env.observation_shape,
            mixed_precision=mixed_precision,
        )

        q1_path = f"{MODEL_PATH}_q1.keras"
        if args.resume and (os.path.isfile(q1_path) or os.path.isfile(f"{MODEL_PATH}_q1")):
            try:
                agent.load(MODEL_PATH)
                print(f"[train] Resumed from {MODEL_PATH}", flush=True)
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

        _perf_t0 = time.monotonic()
        _perf_steps = 0
        PERF_INTERVAL = 200

        print(
            f"[train] Starting SAC. max_episodes="
            f"{'unlimited' if max_episodes == 0 else max_episodes}",
            flush=True,
        )

        while not _stop_requested:
            if max_episodes > 0 and episode >= max_episodes:
                print(f"[train] Reached {max_episodes} episodes.", flush=True)
                break

            actions = agent.select_actions_batch(obs_batch, training=True)
            next_obs, rewards, dones, truncs, infos = vec_env.step(actions)

            agent.store_batch(obs_batch, actions, rewards, next_obs, dones | truncs)

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
                    recent_losses.clear()

                    if episode % LOG_EVERY == 0:
                        print(
                            f"[train] ep={episode:>6}  reward={ep_rewards[i]:>8.2f}  "
                            f"avg_loss={avg_loss:.5f}  success={success_count}  "
                            f"dist={info.get('distance', 0):.3f}",
                            flush=True,
                        )
                    ep_rewards[i] = 0.0

                    if CHECKPOINT_EVERY > 0 and episode % CHECKPOINT_EVERY == 0:
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        agent.save(MODEL_PATH)
                        print(f"[train] Checkpoint → {MODEL_PATH}", flush=True)

            if global_step % train_freq == 0:
                for _ in range(gradient_steps):
                    loss = agent.train_step()
                    if loss is not None:
                        recent_losses.append(loss)

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

        os.makedirs(MODELS_DIR, exist_ok=True)
        agent.save(MODEL_PATH)
        print(f"[train] Model saved → {MODEL_PATH}", flush=True)

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
    parser = argparse.ArgumentParser(description="Headless SAC training for Robot-DRL")
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
