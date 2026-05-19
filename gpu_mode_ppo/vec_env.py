"""
SubprocVecEnv: run N VisionControlEnv instances in parallel subprocesses.

Each worker is a separate process running PyBullet on its own CPU core.
The main process collects all N observations, runs one GPU forward pass
for all N actions, then sends them back — one GPU call instead of N.

Usage:
    from vec_env import SubprocVecEnv
    vec = SubprocVecEnv(n_envs=4)
    obs_batch = vec.reset()                          # (4, 84, 84)
    actions   = agent.select_actions_batch(obs_batch)  # (4,)
    obs_batch, rewards, dones, truncs, infos = vec.step(actions)
    vec.close()

Episode boundaries:
    When done or truncated, the worker auto-resets and returns the NEW
    observation (not the terminal one).  info["episode_done"] == True
    signals the boundary so the caller can log episode stats.
"""

import multiprocessing as mp
import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Any

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker_fn(conn: "mp.connection.Connection", env_kwargs: dict) -> None:
    """
    Runs inside each subprocess.  Receives (cmd, payload) and sends results.
    Only uses PyBullet + OpenCV — never imports TensorFlow or CUDA.
    """
    sys.path.insert(0, _HERE)

    try:
        from environment import VisionControlEnv
    except Exception as exc:
        conn.send(RuntimeError(f"Worker import failed: {exc}"))
        conn.close()
        return

    env = VisionControlEnv(**env_kwargs)

    try:
        while True:
            cmd, payload = conn.recv()

            if cmd == "step":
                obs, reward, done, truncated, info = env.step(int(payload))
                if done or truncated:
                    # Auto-reset so the main loop never needs to call reset()
                    # individually per env.  Caller detects boundary via info.
                    obs, _ = env.reset()
                    info["episode_done"] = True
                else:
                    info["episode_done"] = False
                conn.send((obs, float(reward), bool(done), bool(truncated), info))

            elif cmd == "reset":
                obs, info = env.reset()
                conn.send((obs, info))

            elif cmd == "close":
                break

    except EOFError:
        pass  # parent closed pipe — normal shutdown
    finally:
        env.close()
        conn.close()


# ── SubprocVecEnv ─────────────────────────────────────────────────────────────

class SubprocVecEnv:
    """
    N independent VisionControlEnv instances in separate subprocesses.

    Construction forks worker processes.  Call this BEFORE initialising
    TensorFlow in the parent so workers start with a clean state.

    Recommended n_envs:
        Colab free  (2 vCPU)  →  n_envs = 2
        Colab Pro   (4 vCPU)  →  n_envs = 4
        Colab Pro+  (8 vCPU)  →  n_envs = 8
    """

    def __init__(self, n_envs: int, **env_kwargs):
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.n_envs = n_envs

        # "fork" on Linux (Colab): workers start instantly and inherit sys.path.
        # Workers never call any CUDA API so the inherited (unused) CUDA handle
        # in the child address space is harmless.
        ctx = mp.get_context("fork")

        self._pipes: List[mp.connection.Connection] = []
        self._procs: List[mp.Process] = []

        for _ in range(n_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_worker_fn,
                args=(child_conn, env_kwargs),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent keeps only its end
            self._pipes.append(parent_conn)
            self._procs.append(proc)

    # ── Properties matching VisionControlEnv ─────────────────────────────────

    @property
    def n_actions(self) -> int:
        return 5  # noop, extend, curl, left, right

    @property
    def observation_shape(self) -> tuple:
        from utils import PREPROCESS_HEIGHT, PREPROCESS_WIDTH
        return (PREPROCESS_HEIGHT, PREPROCESS_WIDTH)

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset all N envs. Returns obs of shape (N, H, W)."""
        for pipe in self._pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self._pipes]
        return np.stack([r[0] for r in results])  # (N, H, W)

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all N envs with actions (N,).
        Sends all actions first so workers execute in parallel,
        then collects results.

        Returns:
            obs       (N, H, W)  — new observation (post-autoreset if episode ended)
            rewards   (N,)       float32
            dones     (N,)       bool
            truncateds(N,)       bool
            infos     list of N dicts; info["episode_done"] marks episode ends
        """
        for pipe, action in zip(self._pipes, actions):
            pipe.send(("step", int(action)))

        results = [pipe.recv() for pipe in self._pipes]

        obs      = np.stack([r[0] for r in results])                    # (N, H, W)
        rewards  = np.array([r[1] for r in results], dtype=np.float32)  # (N,)
        dones    = np.array([r[2] for r in results], dtype=bool)        # (N,)
        truncs   = np.array([r[3] for r in results], dtype=bool)        # (N,)
        infos    = [r[4] for r in results]
        return obs, rewards, dones, truncs, infos

    def close(self) -> None:
        """Gracefully shut down all workers."""
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
