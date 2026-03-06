# Vision-Based Robotic Arm Control

A reinforcement learning project for controlling a two-joint robotic arm using vision only. The DQN agent observes the simulation as an 84×84 image (arm + target) and learns to reach a target. Training uses TensorFlow; deployment can use OpenVINO for inference.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies (OpenCV, NumPy, Pillow, PyBullet, TensorFlow, OpenVINO, Matplotlib).

## Setup

1. **Clone the repository** (or extract the project folder).

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

   - **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   - **Windows (cmd):** `venv\Scripts\activate.bat`
   - **macOS/Linux:** `source venv/bin/activate`

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Launch the GUI dashboard:**

   ```bash
   python main_gui.py
   ```

2. **From the dashboard you can:**
   - **Start Training** – Run DQN training in the PyBullet simulation (saves model to `models/dqn_model.keras` and logs/plots to `logs/`).
   - **Convert model → OpenVINO IR** – Export the trained model to OpenVINO format (`.xml`/`.bin`) for faster inference.
   - **Run OpenVINO Inference** – Run the policy using the OpenVINO IR model (requires having run conversion first).
   - **Stop** – Stop the current training or inference run.

3. **Optional: run without GUI** – The core logic lives in `environment.py` (PyBullet env), `agent.py` (DQN + OpenVINO), and `utils.py` (image preprocessing). You can import these and run training or inference from your own script.

## Project Structure

- `main_gui.py` – Tkinter dashboard (training, inference, model conversion).
- `environment.py` – PyBullet environment for the two-joint arm and target.
- `agent.py` – DQN agent (TensorFlow) and OpenVINO inference.
- `utils.py` – Frame preprocessing (resize, grayscale, normalization).
- `robot_urdf/` – URDF and assets for the robotic arm.
- `requirements.txt` – Python dependencies.

Training outputs are saved under `models/` and `logs/` (create these if they don’t exist).
