"""
Tkinter dashboard for vision-based control demo.
Shows the PyBullet simulation and DQN input (84×84). Runs OpenVINO demo mode
in a background thread. Training is done separately via gpu_mode/train.py on Colab.
"""

import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
from typing import Optional, List, Tuple
import numpy as np
from datetime import datetime

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from utils import preprocess_frame_uint8, numpy_to_tk_photo, PREPROCESS_WIDTH, PREPROCESS_HEIGHT
from config_loader import cfg

try:
    from environment import VisionControlEnv
    HAS_ENV = True
except ImportError:
    HAS_ENV = False
try:
    from agent import OpenVINOInference, convert_h5_to_openvino_ir
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False

_demo = cfg.get("demo", {})
_paths = cfg.get("paths", {})
LOGS_DIR = _paths.get("logs_dir", "logs")
DEMO_ATTEMPTS = int(_demo.get("attempts", 10))
DEMO_SUCCESS_PAUSE_SECONDS = float(_demo.get("success_pause_seconds", 0.5))
DEMO_PRESET_TARGETS = [
    tuple(t) for t in _demo.get("preset_targets", [
        [0.55, 0.00, 0.08],
        [0.60, 0.12, 0.08],
        [0.62, -0.10, 0.08],
        [0.68, 0.05, 0.08],
        [0.70, -0.08, 0.08],
    ])
]


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
        if rgb_array is None or not HAS_PIL:
            return
        try:
            if len(rgb_array.shape) == 2:
                pil_img = Image.fromarray(rgb_array, mode="L")
            else:
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
    """Displays what the policy sees: full scene 84×84 (arm + target)."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Policy input: full scene (arm + target) 84×84", **kwargs)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._label = ttk.Label(self, text="No observation", anchor=tk.CENTER)
        self._label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def update_frame(self, rgb_frame: Optional[np.ndarray] = None, dqn_obs: Optional[np.ndarray] = None) -> None:
        if not HAS_PIL:
            return
        try:
            if dqn_obs is not None and dqn_obs.size > 0:
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
        self._label.configure(image="", text="No observation")


class MainDashboard:
    """Demo dashboard: simulation view, visual feedback, OpenVINO demo, and IR conversion."""

    def __init__(self):
        self._root = tk.Tk()
        self._root.title("Vision-Based Control — Demo")
        self._root.geometry("900x600")
        self._root.minsize(700, 450)

        self._update_queue: queue.Queue = queue.Queue()
        self._demo_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._model_path = _paths.get("model_keras", "models/dqn_model.keras")
        self._openvino_ir_dir = os.path.dirname(_paths.get("openvino_ir_xml", "openvino_ir/dqn_ir.xml")) or "openvino_ir"
        self._openvino_ir_xml = _paths.get("openvino_ir_xml", "openvino_ir/dqn_ir.xml")

        self._build_ui()
        self._schedule_queue_process()

    def _build_ui(self) -> None:
        main = ttk.Frame(self._root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Vision-Based Target Reaching — Demo", font=("", 14, "bold")).pack(pady=(0, 10))

        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(content, text="Simulation view (arm + target)")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._sim_view = SimulationViewPanel(left, width=320, height=240)
        self._sim_view.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._visual_feedback = VisualFeedbackPanel(right)
        self._visual_feedback.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(
            btn_frame,
            text="Run OpenVINO Demo Mode",
            command=self._on_run_demo_mode,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btn_frame,
            text="Convert model → OpenVINO IR",
            command=self._on_convert_ir,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=4)

        self._status_var = tk.StringVar(value="Ready — train on Colab (gpu_mode/), then convert and demo here.")
        ttk.Label(main, textvariable=self._status_var).pack(anchor=tk.W)

    def _schedule_queue_process(self) -> None:
        try:
            while True:
                msg = self._update_queue.get_nowait()
                self._process_update(msg)
        except queue.Empty:
            pass
        self._root.after(50, self._schedule_queue_process)

    def _process_update(self, msg: dict) -> None:
        kind = msg.get("kind")
        if kind == "frame":
            self._sim_view.update_frame(msg.get("rgb"))
            self._visual_feedback.update_frame(rgb_frame=msg.get("rgb"), dqn_obs=msg.get("dqn_obs"))
        elif kind == "status":
            self._status_var.set(msg.get("text", ""))

    def push_frame_update(self, rgb: Optional[np.ndarray] = None, dqn_obs: Optional[np.ndarray] = None) -> None:
        rgb_copy = np.copy(rgb) if rgb is not None and rgb.size > 0 else None
        dqn_copy = np.copy(dqn_obs) if dqn_obs is not None and dqn_obs.size > 0 else None
        self._update_queue.put({"kind": "frame", "rgb": rgb_copy, "dqn_obs": dqn_copy})

    def push_status(self, text: str) -> None:
        self._update_queue.put({"kind": "status", "text": text})

    def _get_model_path_for_load(self) -> str:
        if os.path.isfile(self._model_path):
            return self._model_path
        alt = _paths.get("model_h5", "models/dqn_model.h5")
        if os.path.isfile(alt):
            return alt
        return self._model_path

    def _can_start_demo(self) -> bool:
        if self._demo_running:
            messagebox.showinfo("Info", "Demo already running.")
            return False
        if not HAS_ENV or not HAS_AGENT:
            messagebox.showwarning(
                "Missing deps",
                "Demo requires pybullet and openvino.\nInstall: pip install pybullet openvino",
            )
            return False
        if not os.path.isfile(self._openvino_ir_xml):
            messagebox.showinfo(
                "No IR model",
                f"OpenVINO IR not found at {self._openvino_ir_xml}.\n"
                "Train on Colab (gpu_mode/train.py), copy the .keras model here, "
                "then use Convert model → OpenVINO IR.",
            )
            return False
        return True

    def _on_run_demo_mode(self) -> None:
        if not self._can_start_demo():
            return
        self._demo_running = True
        self._status_var.set("OpenVINO demo mode starting…")
        self._worker_thread = threading.Thread(target=self._demo_loop, daemon=True)
        self._worker_thread.start()

    def _demo_loop(self) -> None:
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
                if not self._demo_running:
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
                while self._demo_running:
                    action = ov_agent.select_action(obs)
                    obs, _reward, done, truncated, step_info = env.step(action)
                    rgb = step_info.get("rgb")
                    if rgb is not None:
                        self.push_frame_update(rgb, dqn_obs=step_info.get("dqn_obs"))
                    steps = int(step_info.get("step", steps + 1))
                    final_distance = float(step_info.get("distance", final_distance))
                    success = bool(step_info.get("success", False))
                    final_ee = tuple(step_info.get("ee_pos", final_ee))
                    status = (
                        f"Demo {attempt_idx}/{DEMO_ATTEMPTS} | step {steps} | "
                        f"dist={final_distance:.3f} | action={action}"
                    )
                    self.push_status(status)
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
                result = "SUCCESS" if success else terminal_reason
                self.push_status(f"Attempt {attempt_idx}/{DEMO_ATTEMPTS}: {result} ({steps} steps)")

                if success:
                    pause_deadline = time.time() + DEMO_SUCCESS_PAUSE_SECONDS
                    while self._demo_running and time.time() < pause_deadline:
                        time.sleep(0.1)

            csv_path, txt_path = _save_openvino_demo_logs(
                attempt_rows,
                configured_attempts=DEMO_ATTEMPTS,
                success_pause_seconds=DEMO_SUCCESS_PAUSE_SECONDS,
            )
            successes = sum(1 for row in attempt_rows if row.get("success"))
            msg = f"Demo finished. {successes}/{len(attempt_rows)} successes."
            if csv_path:
                msg += f" Log: {csv_path}"
            if txt_path:
                msg += f" Summary: {txt_path}"
            self.push_status(msg)
        except Exception as e:
            self.push_status(f"Demo error: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self._demo_running = False

    def _on_convert_ir(self) -> None:
        if not HAS_AGENT:
            messagebox.showwarning("Missing deps", "OpenVINO and TensorFlow required for conversion.")
            return
        load_path = self._get_model_path_for_load()
        if not os.path.isfile(load_path):
            messagebox.showinfo(
                "No model",
                f"No model at {self._model_path} or models/dqn_model.h5.\n"
                "Train with gpu_mode/train.py on Colab, then copy the checkpoint here.",
            )
            return
        self._status_var.set("Converting model → OpenVINO IR…")

        def do_convert():
            try:
                xml_path, bin_path = convert_h5_to_openvino_ir(
                    load_path,
                    self._openvino_ir_dir,
                    output_name="dqn_ir",
                )
                self.push_status(f"Converted: {xml_path}")
                self._root.after(
                    0,
                    lambda x=xml_path, b=bin_path: messagebox.showinfo("Done", f"Saved:\n{x}\n{b}"),
                )
            except Exception as e:
                err_msg = str(e)
                self.push_status(f"Convert error: {e}")
                self._root.after(0, lambda msg=err_msg: messagebox.showerror("Convert failed", msg))

        threading.Thread(target=do_convert, daemon=True).start()

    def _on_stop(self) -> None:
        self._demo_running = False
        self._status_var.set("Stopped.")

    def run(self) -> None:
        self._root.mainloop()


def main() -> None:
    if not HAS_PIL:
        print("Warning: PIL/Pillow not found. Install with: pip install Pillow")
    app = MainDashboard()
    app.run()


if __name__ == "__main__":
    main()
