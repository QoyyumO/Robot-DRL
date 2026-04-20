"""
Tkinter dashboard for vision-based control: move extend, curl, left, right.
Shows simulation view (PyBullet camera) and Visual Feedback (84x84 processed).
Runs simulation and DQN in a background thread so the GUI stays responsive.
"""

import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
from typing import Optional, Callable, List, Tuple
import numpy as np
from datetime import datetime

# Optional imports: fail gracefully if not installed
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from utils import preprocess_frame_uint8, numpy_to_tk_photo, PREPROCESS_WIDTH, PREPROCESS_HEIGHT

# Optional: environment and agent (for real training/inference)
try:
    from environment import VisionControlEnv
    HAS_ENV = True
except ImportError:
    HAS_ENV = False
try:
    from agent import DQNAgent, OpenVINOInference
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False


LOGS_DIR = "logs"
DEMO_ATTEMPTS = 10
DEMO_SUCCESS_PAUSE_SECONDS = 0.5
DEMO_PRESET_TARGETS = [
    (0.55, 0.00, 0.08),
    (0.60, 0.12, 0.08),
    (0.62, -0.10, 0.08),
    (0.68, 0.05, 0.08),
    (0.70, -0.08, 0.08),
]


def _save_training_csv(
    episodes: List[int],
    rewards: List[float],
    avg_losses: List[float],
    successes: List[int],
    epsilons: List[float],
) -> Optional[str]:
    """Write per-episode training metrics to a timestamped CSV file."""
    if not episodes:
        return None
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"training_metrics_{stamp}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "cumulative_reward", "avg_loss", "success", "epsilon"])
            for ep, rw, lo, su, eps in zip(episodes, rewards, avg_losses, successes, epsilons):
                writer.writerow([ep, f"{rw:.4f}", f"{lo:.6f}", su, f"{eps:.4f}"])
        return path
    except Exception:
        return None


def _plot_training_metrics(
    episodes: List[int],
    rewards: List[float],
    avg_losses: List[float],
) -> Optional[str]:
    """Generate a two-panel PNG: cumulative reward and average loss vs. episode."""
    if not episodes:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(LOGS_DIR, exist_ok=True)
        from datetime import datetime
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
    except Exception:
        return None


def _save_openvino_demo_logs(
    attempts: List[dict],
    configured_attempts: int,
    success_pause_seconds: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Write demo-mode per-attempt CSV and summary TXT logs."""
    if not attempts:
        return None, None
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(LOGS_DIR, f"openvino_demo_attempts_{stamp}.csv")
        txt_path = os.path.join(LOGS_DIR, f"openvino_demo_summary_{stamp}.txt")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "attempt",
                    "target_x",
                    "target_y",
                    "target_z",
                    "start_ee_x",
                    "start_ee_y",
                    "start_ee_z",
                    "final_ee_x",
                    "final_ee_y",
                    "final_ee_z",
                    "success",
                    "steps",
                    "terminal_reason",
                    "final_distance",
                    "pause_seconds",
                ]
            )
            for row in attempts:
                tx, ty, tz = row.get("target", (None, None, None))
                sx, sy, sz = row.get("start_ee", (None, None, None))
                fx, fy, fz = row.get("final_ee", (None, None, None))
                writer.writerow(
                    [
                        row.get("attempt"),
                        tx,
                        ty,
                        tz,
                        sx,
                        sy,
                        sz,
                        fx,
                        fy,
                        fz,
                        1 if row.get("success") else 0,
                        row.get("steps"),
                        row.get("terminal_reason"),
                        f"{float(row.get('final_distance', 0.0)):.6f}",
                        row.get("pause_seconds", 0.0),
                    ]
                )

        successes = sum(1 for row in attempts if row.get("success"))
        completed_attempts = len(attempts)
        success_steps = [row.get("steps", 0) for row in attempts if row.get("success")]
        avg_success_steps = float(np.mean(success_steps)) if success_steps else 0.0
        avg_final_distance = float(np.mean([row.get("final_distance", 0.0) for row in attempts]))
        success_rate = (100.0 * successes / completed_attempts) if completed_attempts else 0.0
        failed_attempts = completed_attempts - successes

        with open(txt_path, "w", newline="") as f:
            f.write("OpenVINO Demo Mode Summary\n")
            f.write(f"Configured attempts: {configured_attempts}\n")
            f.write(f"Completed attempts: {completed_attempts}\n")
            f.write(f"Successes: {successes}\n")
            f.write(f"Failed attempts: {failed_attempts}\n")
            f.write(f"Success rate (%): {success_rate:.2f}\n")
            f.write(f"Average successful steps: {avg_success_steps:.2f}\n")
            f.write(f"Average final distance (m): {avg_final_distance:.4f}\n")
            f.write(f"Success pause seconds: {success_pause_seconds:.1f}\n")
            f.write(f"Attempts CSV: {csv_path}\n")

        return csv_path, txt_path
    except Exception:
        return None, None


class SimulationViewPanel(ttk.Frame):
    """Frame that displays the PyBullet camera feed (raw RGB)."""

    def __init__(self, parent, width: int = 320, height: int = 240, **kwargs):
        super().__init__(parent, **kwargs)
        self._width = width
        self._height = height
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._label = ttk.Label(self, text="Simulation view\n(no feed)", anchor=tk.CENTER)
        self._label.pack(fill=tk.BOTH, expand=True)

    def update_frame(self, rgb_array: Optional[np.ndarray]) -> None:
        """Update the panel with a new RGB frame (H, W, 3) from PyBullet."""
        if rgb_array is None or not HAS_PIL:
            return
        try:
            if len(rgb_array.shape) == 2:
                pil_img = Image.fromarray(rgb_array, mode="L")
            else:
                # PyBullet getCameraImage returns RGB
                pil_img = Image.fromarray(rgb_array, mode="RGB")
            try:
                pil_img = pil_img.resize((self._width, self._height), Image.Resampling.NEAREST)
            except AttributeError:
                pil_img = pil_img.resize((self._width, self._height), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(pil_img)
            self._label.configure(image=self._photo, text="")
        except Exception:
            self._label.configure(image="", text="Simulation view\n(no feed)")


class VisualFeedbackPanel(ttk.LabelFrame):
    """Displays what the DQN sees: full scene 84×84 (arm + target), image only."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="DQN input: full scene (arm + target) 84×84", **kwargs)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._label = ttk.Label(self, text="No DQN observation", anchor=tk.CENTER)
        self._label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def update_frame(self, rgb_frame: Optional[np.ndarray] = None, dqn_obs: Optional[np.ndarray] = None) -> None:
        """Show DQN observation (full scene 84×84) if provided; else preprocess rgb for fallback."""
        if not HAS_PIL:
            return
        try:
            if dqn_obs is not None and dqn_obs.size > 0:
                # DQN sees full scene (84x84 float [0,1])
                arr = np.ascontiguousarray((np.clip(dqn_obs.astype(np.float64), 0, 1) * 255).astype(np.uint8))
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                    self._photo = numpy_to_tk_photo(arr, is_grayscale=True)
                    if self._photo is not None:
                        self._label.configure(image=self._photo, text="")
                    return
            if rgb_frame is not None and rgb_frame.size > 0:
                processed = preprocess_frame_uint8(rgb_frame, PREPROCESS_WIDTH, PREPROCESS_HEIGHT)
                self._photo = numpy_to_tk_photo(processed, is_grayscale=True)
                if self._photo is not None:
                    self._label.configure(image=self._photo, text="")
                return
        except Exception:
            pass
        self._photo = None
        self._label.configure(image="", text="No DQN observation")


class MainDashboard:
    """Main Tkinter dashboard: simulation view, visual feedback, and control buttons."""

    def __init__(self):
        self._root = tk.Tk()
        self._root.title("Vision-Based Control — Two-Joint Arm")
        self._root.geometry("900x600")
        self._root.minsize(700, 450)

        # Thread-safe queue for GUI updates from worker thread
        self._update_queue: queue.Queue = queue.Queue()
        self._simulation_running = False
        self._inference_running = False
        self._worker_thread: Optional[threading.Thread] = None
        # Default paths: .keras is preferred (avoids HDF5 legacy warning)
        self._model_path = "models/dqn_model.keras"
        self._openvino_ir_dir = "openvino_ir"
        self._openvino_ir_xml = "openvino_ir/dqn_ir.xml"

        self._build_ui()
        self._schedule_queue_process()

    def _build_ui(self) -> None:
        main = ttk.Frame(self._root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Top: title
        ttk.Label(main, text="Vision-Based Target Reaching", font=("", 14, "bold")).pack(pady=(0, 10))

        # Center: two panels side by side
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Full scene (arm + target) for display
        left = ttk.LabelFrame(content, text="Simulation view (arm + target)")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._sim_view = SimulationViewPanel(left, width=320, height=240)
        self._sim_view.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Right: what the DQN sees (full scene, image only) — single LabelFrame from VisualFeedbackPanel
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._visual_feedback = VisualFeedbackPanel(right)
        self._visual_feedback.pack(fill=tk.BOTH, expand=True)

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)
        self._btn_train = ttk.Button(btn_frame, text="Start Training", command=self._on_start_training)
        self._btn_train.pack(side=tk.LEFT, padx=4)
        self._btn_inference = ttk.Button(btn_frame, text="Run OpenVINO Inference", command=self._on_run_inference)
        self._btn_inference.pack(side=tk.LEFT, padx=4)
        self._btn_demo = ttk.Button(btn_frame, text="Run OpenVINO Demo Mode", command=self._on_run_demo_mode)
        self._btn_demo.pack(side=tk.LEFT, padx=4)
        self._btn_convert = ttk.Button(btn_frame, text="Convert model → OpenVINO IR", command=self._on_convert_ir)
        self._btn_convert.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=4)

        # Status
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self._status_var).pack(anchor=tk.W)

    def _schedule_queue_process(self) -> None:
        """Process pending GUI updates from the worker (run on main thread)."""
        try:
            while True:
                msg = self._update_queue.get_nowait()
                self._process_update(msg)
        except queue.Empty:
            pass
        self._root.after(50, self._schedule_queue_process)

    def _process_update(self, msg: dict) -> None:
        """Handle a single update message from the worker."""
        kind = msg.get("kind")
        if kind == "frame":
            rgb = msg.get("rgb")
            dqn_obs = msg.get("dqn_obs")
            self._sim_view.update_frame(rgb)
            self._visual_feedback.update_frame(rgb_frame=rgb, dqn_obs=dqn_obs)
        elif kind == "status":
            self._status_var.set(msg.get("text", ""))

    def push_frame_update(self, rgb: Optional[np.ndarray] = None, dqn_obs: Optional[np.ndarray] = None) -> None:
        """Call from worker thread. Copies arrays so main thread has stable data (avoids race/crash)."""
        rgb_copy = np.copy(rgb) if rgb is not None and rgb.size > 0 else None
        dqn_copy = np.copy(dqn_obs) if dqn_obs is not None and dqn_obs.size > 0 else None
        self._update_queue.put({"kind": "frame", "rgb": rgb_copy, "dqn_obs": dqn_copy})

    def push_status(self, text: str) -> None:
        """Call from worker thread to update status text."""
        self._update_queue.put({"kind": "status", "text": text})

    def _on_start_training(self) -> None:
        """Start training in a background thread (env + DQN)."""
        if self._simulation_running:
            messagebox.showinfo("Info", "Training already running.")
            return
        if not HAS_ENV or not HAS_AGENT:
            messagebox.showwarning(
                "Missing deps",
                "Training requires pybullet and tensorflow.\nInstall: pip install pybullet tensorflow",
            )
            return
        self._simulation_running = True
        self._status_var.set("Training starting…")
        self._worker_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._worker_thread.start()

    def _get_model_path_for_load(self) -> str:
        """Path to load from: .keras if it exists, else .h5 for backward compatibility."""
        if os.path.isfile(self._model_path):
            return self._model_path
        alt = "models/dqn_model.h5"
        if os.path.isfile(alt):
            return alt
        return self._model_path

    def _training_loop(self) -> None:
        """Run env + DQN training; push frames and status to GUI. Loads from .keras or .h5 if it exists."""
        env = None
        try:
            env = VisionControlEnv(headless=True)
            agent = DQNAgent(n_actions=env.n_actions, obs_shape=env.observation_shape)
            load_path = self._get_model_path_for_load()
            if os.path.isfile(load_path):
                try:
                    agent.load(load_path)
                    self.push_status("Resuming from " + load_path + " — target reaching")
                except Exception as load_err:
                    self.push_status("Could not load " + load_path + ", training from scratch: " + str(load_err))
            else:
                self.push_status("Training from scratch — target reaching (DQN from image only)")
            obs, info = env.reset()
            rgb = info.get("rgb")
            if rgb is not None:
                self.push_frame_update(rgb, dqn_obs=info.get("dqn_obs"))
            episode = 0
            success_count = 0

            # Per-episode metric accumulators
            ep_reward = 0.0
            ep_losses: list = []
            # History across all episodes (for CSV + plotting)
            history_ep: list = []
            history_reward: list = []
            history_avg_loss: list = []
            history_success: list = []
            history_epsilon: list = []

            while self._simulation_running:
                action = agent.select_action(obs, training=True)
                next_obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                agent.store(obs, action, reward, next_obs, done or truncated)
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)
                rgb = info.get("rgb")
                if rgb is not None:
                    self.push_frame_update(rgb, dqn_obs=info.get("dqn_obs"))
                step = info.get("step", 0)
                if info.get("success"):
                    success_count += 1
                    self.push_status(
                        f"Target reached! | episode {episode} | successes {success_count} | step {step} | ε={agent.epsilon:.3f}"
                        + (f" | loss={loss:.4f}" if loss is not None else "")
                    )
                else:
                    self.push_status(
                        f"Training | episode {episode} | successes {success_count} | step {step} | ε={agent.epsilon:.3f}"
                        + (f" | loss={loss:.4f}" if loss is not None else "")
                    )
                obs = next_obs
                if done or truncated:
                    avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
                    history_ep.append(episode)
                    history_reward.append(ep_reward)
                    history_avg_loss.append(avg_loss)
                    history_success.append(1 if info.get("success") else 0)
                    history_epsilon.append(agent.epsilon)
                    ep_reward = 0.0
                    ep_losses = []
                    obs, info = env.reset()
                    episode += 1
                    if info.get("rgb") is not None:
                        self.push_frame_update(info["rgb"], dqn_obs=info.get("dqn_obs"))
            # Save on stop (always to .keras to avoid HDF5 legacy warning)
            os.makedirs(os.path.dirname(self._model_path) or ".", exist_ok=True)
            agent.save(self._model_path)

            # Save training metrics to CSV and generate plots
            csv_path = _save_training_csv(history_ep, history_reward, history_avg_loss,
                                          history_success, history_epsilon)
            plot_path = _plot_training_metrics(history_ep, history_reward, history_avg_loss)
            save_msg = f"Training stopped. Total successes: {success_count} | Model saved to {self._model_path}"
            if csv_path:
                save_msg += f" | Metrics: {csv_path}"
            if plot_path:
                save_msg += f" | Plot: {plot_path}"
            self.push_status(save_msg)
        except Exception as e:
            self.push_status(f"Training error: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self._simulation_running = False

    def _can_start_openvino(self) -> bool:
        """Validate preconditions before starting inference or demo mode."""
        if self._inference_running:
            messagebox.showinfo("Info", "Inference already running.")
            return False
        if not HAS_ENV or not HAS_AGENT:
            messagebox.showwarning(
                "Missing deps",
                "Inference requires pybullet and openvino.\nInstall: pip install pybullet openvino",
            )
            return False
        if not os.path.isfile(self._openvino_ir_xml):
            messagebox.showinfo(
                "No IR model",
                f"OpenVINO IR not found at {self._openvino_ir_xml}. Train and convert:\n"
                "1. Run Training, then use agent.convert_h5_to_openvino_ir() to create .xml/.bin.",
            )
            return False
        return True

    def _on_run_inference(self) -> None:
        """Run raw OpenVINO inference in a background thread."""
        if not self._can_start_openvino():
            return
        self._inference_running = True
        self._status_var.set("OpenVINO inference starting…")
        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker_thread.start()

    def _on_run_demo_mode(self) -> None:
        """Run OpenVINO Demo Mode with deterministic targets and fixed attempts."""
        if not self._can_start_openvino():
            return
        self._inference_running = True
        self._status_var.set("OpenVINO demo mode starting…")
        self._worker_thread = threading.Thread(target=self._inference_demo_loop, daemon=True)
        self._worker_thread.start()

    def _inference_loop(self) -> None:
        """Run OpenVINO inference (continuous raw mode)."""
        env = None
        try:
            env = VisionControlEnv(headless=True)
            ov_agent = OpenVINOInference(self._openvino_ir_xml, device="GPU")
            obs, info = env.reset()
            rgb = info.get("rgb")
            if rgb is not None:
                self.push_frame_update(rgb, dqn_obs=info.get("dqn_obs"))
            self.push_status("OpenVINO inference (CPU) — target reaching")
            while self._inference_running:
                action = ov_agent.select_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                rgb = info.get("rgb")
                if rgb is not None:
                    self.push_frame_update(rgb, dqn_obs=info.get("dqn_obs"))
                dist = info.get("distance", 0)
                self.push_status(f"OpenVINO | action={action} | reward={reward:.2f} | dist={dist:.3f}")
                if done or truncated:
                    obs, info = env.reset()
                    if info.get("rgb") is not None:
                        self.push_frame_update(info["rgb"], dqn_obs=info.get("dqn_obs"))
            self.push_status("Inference stopped.")
        except Exception as e:
            self.push_status(f"Inference error: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self._inference_running = False

    def _inference_demo_loop(self) -> None:
        """Run finite OpenVINO demo attempts with deterministic targets and logging."""
        env = None
        try:
            env = VisionControlEnv(
                headless=True,
                target_mode="preset_cycle",
                preset_targets=DEMO_PRESET_TARGETS,
                enable_early_truncation=False,
            )
            ov_agent = OpenVINOInference(self._openvino_ir_xml, device="GPU")
            

            attempt_rows: List[dict] = []
            for attempt_idx in range(1, DEMO_ATTEMPTS + 1):
                if not self._inference_running:
                    break

                obs, info = env.reset()
                rgb = info.get("rgb")
                if rgb is not None:
                    self.push_frame_update(rgb, dqn_obs=info.get("dqn_obs"))
                target = tuple(info.get("target_pos", (None, None, None)))
                start_ee = tuple(info.get("ee_pos", (None, None, None)))
                final_ee = start_ee

                steps = 0
                final_distance = float("inf")
                success = False
                terminal_reason = "stopped"
                while self._inference_running:
                    action = ov_agent.select_action(obs)
                    obs, _reward, done, truncated, step_info = env.step(action)
                    rgb = step_info.get("rgb")
                    if rgb is not None:
                        self.push_frame_update(rgb, dqn_obs=step_info.get("dqn_obs"))
                    steps = int(step_info.get("step", steps + 1))
                    final_distance = float(step_info.get("distance", final_distance))
                    success = bool(step_info.get("success", False))
                    final_ee = tuple(step_info.get("ee_pos", final_ee))
                    if success:
                        terminal_reason = "success"
                        break
                    if truncated:
                        terminal_reason = "truncated"
                        break
                    if done:
                        terminal_reason = "max_steps"
                        break

                pause_seconds = DEMO_SUCCESS_PAUSE_SECONDS if success else 0.0
                attempt_rows.append(
                    {
                        "attempt": attempt_idx,
                        "target": target,
                        "start_ee": start_ee,
                        "final_ee": final_ee,
                        "success": success,
                        "steps": steps,
                        "terminal_reason": terminal_reason,
                        "final_distance": final_distance,
                        "pause_seconds": pause_seconds,
                    }
                )
                self.push_status(f"Attempt {attempt_idx}/{DEMO_ATTEMPTS} completed")

                if success:
                    pause_deadline = time.time() + DEMO_SUCCESS_PAUSE_SECONDS
                    while self._inference_running and time.time() < pause_deadline:
                        time.sleep(0.1)

            _save_openvino_demo_logs(
                attempt_rows,
                configured_attempts=DEMO_ATTEMPTS,
                success_pause_seconds=DEMO_SUCCESS_PAUSE_SECONDS,
            )
            self.push_status("Demo finished.")
        except Exception as e:
            print(f"Demo mode error: {e}", flush=True)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self._inference_running = False

    def _on_convert_ir(self) -> None:
        """Convert saved .keras or .h5 model to OpenVINO IR (.xml/.bin) in a background thread."""
        if not HAS_AGENT:
            messagebox.showwarning("Missing deps", "OpenVINO and TensorFlow required for conversion.")
            return
        load_path = self._get_model_path_for_load()
        if not os.path.isfile(load_path):
            messagebox.showinfo("No model", f"No model at {self._model_path} or models/dqn_model.h5. Run Training first.")
            return
        self._status_var.set("Converting model → OpenVINO IR…")
        def do_convert():
            try:
                from agent import convert_h5_to_openvino_ir
                xml_path, bin_path = convert_h5_to_openvino_ir(
                    load_path,
                    self._openvino_ir_dir,
                    output_name="dqn_ir",
                )
                self.push_status(f"Converted: {xml_path}")
                self._root.after(0, lambda x=xml_path, b=bin_path: messagebox.showinfo("Done", f"Saved:\n{x}\n{b}"))
            except Exception as e:
                err_msg = str(e)
                self.push_status(f"Convert error: {e}")
                self._root.after(0, lambda msg=err_msg: messagebox.showerror("Convert failed", msg))
        threading.Thread(target=do_convert, daemon=True).start()

    def _on_stop(self) -> None:
        """Stop training or inference (stub: set flags for real loops)."""
        self._simulation_running = False
        self._inference_running = False
        self._status_var.set("Stopped.")

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._root.mainloop()


def main() -> None:
    if not HAS_PIL:
        print("Warning: PIL/Pillow not found. Install with: pip install Pillow")
    app = MainDashboard()
    app.run()


if __name__ == "__main__":
    main()
