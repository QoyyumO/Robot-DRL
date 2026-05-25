"""
Headless training script for vision-based robotic arm DQN.
Designed for Google Colab GPU (no display, no tkinter, p.DIRECT rendering).

Usage:
    python train.py                          # train from scratch
    python train.py --resume                 # resume from saved checkpoint
    python train.py --episodes 2000          # run fixed number of episodes then stop
    python train.py --resume --episodes 500  # resume and run 500 more episodes

Outputs (written to ../logs/ relative to this file):
    models/dqn_model.keras        — saved model weights
    logs/training_metrics_*.csv   — per-episode metrics
    logs/training_plot_*.png      — reward + loss curves
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

# Resolve paths relative to this file so the script can be run from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # parent Robot-DRL directory

sys.path.insert(0, _HERE)  # ensure gpu_mode modules (including config_loader) are importable

from config_loader import cfg

_train_cfg = cfg["training"]

LOGS_DIR = os.path.join(_ROOT, "logs")
MODELS_DIR = os.path.join(_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "dqn_model.keras")
MODEL_PATH_H5 = os.path.join(MODELS_DIR, "dqn_model.h5")

# ── Graceful stop on Ctrl-C or SIGTERM ──────────────────────────────────────
_stop_requested = False

def _handle_signal(signum, frame):
    global _stop_requested
    print("\n[train] Stop signal received — finishing current episode then saving.", flush=True)
    _stop_requested = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Logging helpers ──────────────────────────────────────────────────────────

def _save_training_csv(
    episodes: List[int],
    rewards: List[float],
    avg_losses: List[float],
    successes: List[int],
    epsilons: List[float],
) -> Optional[str]:
    if not episodes:
        return None
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"training_metrics_{stamp}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "cumulative_reward", "avg_loss", "success", "epsilon"])
            for ep, rw, lo, su, eps in zip(episodes, rewards, avg_losses, successes, epsilons):
                writer.writerow([ep, f"{rw:.4f}", f"{lo:.6f}", su, f"{eps:.4f}"])
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
        matplotlib.use("Agg")  # non-interactive backend — safe on Colab and headless servers
        import matplotlib.pyplot as plt

        os.makedirs(LOGS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"training_plot_{stamp}.png")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(episodes, rewards, linewidth=0.8, color="#2176AE")
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_title("Cumulative Reward per Episode")
        ax1.grid(True, alpha=0.3)
        ax2.plot(episodes, avg_losses, linewidth=0.8, color="#D7263D")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Loss (MSE)")
        ax2.set_title("Average Training Loss per Episode")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as e:
        print(f"[train] Warning: could not save plot: {e}", flush=True)
        return None


# ── GPU setup ────────────────────────────────────────────────────────────────

def _setup_gpu() -> None:
    """
    Configure GPU memory growth, confirm the device TF will use,
    and run a timed compute check so you can see the GPU is actually active.
    Must be called before any tf.keras model is built.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[train] WARNING: No GPU detected by TensorFlow. Running on CPU.", flush=True)
            print("        On Colab: Runtime → Change runtime type → T4 GPU", flush=True)
            return

        print(f"[train] TensorFlow detected {len(gpus)} GPU(s):", flush=True)
        for g in gpus:
            print(f"         {g}", flush=True)

        # Memory growth: allocate VRAM on demand instead of claiming all 15 GB upfront.
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass  # already initialised — must be set before any GPU ops

        # ── Compute verification ─────────────────────────────────────────────
        # Runs a 2048×2048 matrix multiply on GPU and CPU and prints both times.
        # This proves compute is actually dispatched to the GPU, and also warms
        # up the CUDA context so the first real train_step isn't slow.
        #
        # NOTE on 0.4 GB memory usage: that is EXPECTED for this model.
        #   • Two DQN networks (~1.7 M params each) = ~14 MB of weights
        #   • Adam optimizer state (2× model params) = ~14 MB
        #   • TF/CUDA runtime = ~350-400 MB
        #   Total: ~0.4 GB — the model is simply small.
        #   GPU *compute* utilisation (%) is what matters, not memory.
        #   Use  !nvidia-smi -l 1  in a separate Colab cell to watch it live.
        N = 2048
        with tf.device("/GPU:0"):
            a = tf.random.normal([N, N])
            b = tf.random.normal([N, N])
            _ = tf.matmul(a, b).numpy()          # first call triggers CUDA JIT; discard
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
            f"(speedup={cpu_ms/gpu_ms:.1f}×)",
            flush=True,
        )
        print(
            "[train] TF ops confirmed on GPU. "
            "Low VRAM (0.4 GB) is normal — the DQN model is ~14 MB. "
            "Watch GPU *compute %* with:  !nvidia-smi -l 1",
            flush=True,
        )

    except ImportError:
        print("[train] TensorFlow not found.", flush=True)
    except Exception as e:
        print(f"[train] GPU setup warning: {e}", flush=True)


# ── GPU util helper ──────────────────────────────────────────────────────────

def _gpu_stats() -> str:
    """Return a short nvidia-smi string, or empty string if unavailable."""
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


# ── Main training loop ───────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    global _stop_requested

    n_envs          = args.n_envs
    train_freq      = _train_cfg["train_freq"]
    gradient_steps  = _train_cfg["gradient_steps"]
    mixed_precision = _train_cfg["mixed_precision"]

    # ── 1. Fork worker processes BEFORE initialising CUDA ───────────────────
    # Workers only use PyBullet + OpenCV (no TF/CUDA), so they must be created
    # before _setup_gpu() to avoid inheriting an active CUDA context.
    from vec_env import SubprocVecEnv
    print(f"[train] Starting {n_envs} environment worker(s)…", flush=True)
    vec_env = SubprocVecEnv(n_envs)

    # ── 2. Now initialise GPU / TF in the parent ─────────────────────────────
    _setup_gpu()

    from agent import DQNAgent

    print(
        f"[train] Config: n_envs={n_envs}  batch={cfg['dqn']['batch_size']}  "
        f"train_freq={train_freq}  gradient_steps={gradient_steps}  "
        f"mixed_precision={mixed_precision}",
        flush=True,
    )

    try:
        agent = DQNAgent(
            n_actions=vec_env.n_actions,
            obs_shape=vec_env.observation_shape,
            mixed_precision=mixed_precision,
        )
        print("[train] tf.function Bellman graph compiled on first train step.", flush=True)

        # ── Resume ───────────────────────────────────────────────────────────
        load_path = None
        if args.resume:
            load_path = MODEL_PATH if os.path.isfile(MODEL_PATH) else (
                MODEL_PATH_H5 if os.path.isfile(MODEL_PATH_H5) else None
            )
        if load_path:
            try:
                agent.load(load_path)
                print(f"[train] Resumed from {load_path}", flush=True)
            except Exception as e:
                print(f"[train] Could not load {load_path}: {e} — scratch", flush=True)
        else:
            print("[train] Training from scratch.", flush=True)

        # ── State ─────────────────────────────────────────────────────────────
        obs_batch   = vec_env.reset()          # (N, H, W)
        ep_rewards  = np.zeros(n_envs)         # cumulative reward per active episode
        episode     = 0
        global_step = 0
        success_count = 0
        recent_losses: List[float] = []        # gradient losses since last episode end

        history_ep:       List[int]   = []
        history_reward:   List[float] = []
        history_avg_loss: List[float] = []
        history_success:  List[int]   = []
        history_epsilon:  List[float] = []

        CHECKPOINT_EVERY = args.checkpoint_every
        LOG_EVERY        = args.log_every
        max_episodes     = args.episodes

        _perf_t0    = time.monotonic()
        _perf_steps = 0
        PERF_INTERVAL = 200

        print(
            f"[train] Starting. "
            f"max_episodes={'unlimited' if max_episodes == 0 else max_episodes} | "
            f"checkpoint_every={CHECKPOINT_EVERY} | log_every={LOG_EVERY}",
            flush=True,
        )

        # ── Main loop ─────────────────────────────────────────────────────────
        while not _stop_requested:
            if max_episodes > 0 and episode >= max_episodes:
                print(f"[train] Reached {max_episodes} episodes.", flush=True)
                break

            # ── N env steps in parallel (all N PyBullet processes run at once)
            actions = agent.select_actions_batch(obs_batch, training=True)
            next_obs, rewards, dones, truncs, infos = vec_env.step(actions)

            # Store all N transitions in one vectorised write
            agent.store_batch(obs_batch, actions, rewards, next_obs, dones | truncs)

            ep_rewards  += rewards
            obs_batch    = next_obs
            global_step += n_envs
            _perf_steps += n_envs

            # ── Episode boundaries ────────────────────────────────────────────
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
                    history_epsilon.append(agent.epsilon)
                    recent_losses.clear()

                    if episode % LOG_EVERY == 0:
                        print(
                            f"[train] ep={episode:>6}  reward={ep_rewards[i]:>8.2f}  "
                            f"avg_loss={avg_loss:.5f}  success={success_count}  "
                            f"ε={agent.epsilon:.4f}  dist={info.get('distance', 0):.3f}",
                            flush=True,
                        )
                    ep_rewards[i] = 0.0

                    if CHECKPOINT_EVERY > 0 and episode % CHECKPOINT_EVERY == 0:
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        agent.save(MODEL_PATH)
                        print(f"[train] Checkpoint saved → {MODEL_PATH}", flush=True)

            # ── GPU training round ────────────────────────────────────────────
            if global_step % train_freq == 0:
                for _ in range(gradient_steps):
                    loss = agent.train_step()
                    if loss is not None:
                        recent_losses.append(loss)

            # ── Throughput report ─────────────────────────────────────────────
            if _perf_steps >= PERF_INTERVAL:
                elapsed = time.monotonic() - _perf_t0
                sps     = _perf_steps / elapsed if elapsed > 0 else 0.0
                stats   = _gpu_stats()
                print(
                    f"[train] {global_step:>8} steps | {sps:>6.1f} steps/sec"
                    + (f"  |  {stats}" if stats else ""),
                    flush=True,
                )
                _perf_t0    = time.monotonic()
                _perf_steps = 0

        # ── Final save ────────────────────────────────────────────────────────
        os.makedirs(MODELS_DIR, exist_ok=True)
        agent.save(MODEL_PATH)
        print(f"[train] Model saved → {MODEL_PATH}", flush=True)

        csv_path  = _save_training_csv(
            history_ep, history_reward, history_avg_loss, history_success, history_epsilon
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


# ── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless DQN training for Robot-DRL (Colab GPU / parallel envs)"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved checkpoint if one exists")
    parser.add_argument("--episodes", type=int, default=0, metavar="N",
                        help="Stop after N episodes (default: 0 = unlimited)")
    parser.add_argument("--n-envs", type=int,
                        default=_train_cfg.get("n_envs", 1),
                        dest="n_envs", metavar="N",
                        help=f"Parallel environments "
                             f"(default: {_train_cfg.get('n_envs', 1)}, "
                             f"free Colab=2, Pro=4)")
    parser.add_argument("--checkpoint-every", type=int,
                        default=_train_cfg["checkpoint_every"],
                        dest="checkpoint_every", metavar="N",
                        help=f"Checkpoint every N episodes "
                             f"(default: {_train_cfg['checkpoint_every']})")
    parser.add_argument("--log-every", type=int,
                        default=_train_cfg["log_every"],
                        dest="log_every", metavar="N",
                        help=f"Log every N episodes "
                             f"(default: {_train_cfg['log_every']})")
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
