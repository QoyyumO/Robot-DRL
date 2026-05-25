# Vision-Based Robotic Arm Control

A reinforcement learning project for controlling a two-joint robotic arm using vision only. The DQN agent observes the simulation as an 84×84 image (arm + target) and learns to reach a target. Training runs on Google Colab GPU; deployment uses OpenVINO via the local demo GUI.

## Requirements

- Python 3.8+
- **Training (Colab):** see `gpu_mode/requirements.txt` (TensorFlow, PyBullet, OpenCV, etc.)
- **Local demo:** see `requirements.txt` (adds Pillow, OpenVINO for GUI and IR conversion)

## Training on Google Colab (GPU)

1. In Colab, set **Runtime → Change runtime type → T4 GPU** (or better), then restart the runtime.

2. Run:

```python
!git clone -b gpu_v https://github.com/QoyyumO/Robot-DRL.git
!cd /content/Robot-DRL
!pip install -r /content/Robot-DRL/gpu_mode/requirements.txt
!python /content/Robot-DRL/gpu_mode/train.py --episodes 10000 --log-every 100
```

3. Outputs are written under `/content/Robot-DRL/`:
   - `models/dqn_model.keras` — checkpoint (download for local use)
   - `logs/training_metrics_*.csv` and `logs/training_plot_*.png` — metrics

Optional flags: `--resume` (continue from checkpoint), `--n-envs 4` (parallel envs, default in config).

**Other algorithms (same Colab pattern):**

```python
# PPO
!pip install -r /content/Robot-DRL/gpu_mode_ppo/requirements.txt
!python /content/Robot-DRL/gpu_mode_ppo/train.py --episodes 10000 --log-every 100

# SAC
!pip install -r /content/Robot-DRL/gpu_mode_sac/requirements.txt
!python /content/Robot-DRL/gpu_mode_sac/train.py --episodes 10000 --log-every 100
```

## Local setup (demo & conversion)

1. **Clone the repository** (or copy your trained `models/` and `logs/` from Colab):

   ```bash
   git clone -b gpu_v https://github.com/QoyyumO/Robot-DRL.git
   cd Robot-DRL
   ```

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

4. Place the trained checkpoint at `models/dqn_model.keras` (from Colab).

## Demo GUI (local)

```bash
python main_gui.py
```

From the dashboard:

- **Convert model → OpenVINO IR** — export `models/dqn_model.keras` to `openvino_ir/dqn_ir.xml` / `.bin`
- **Run OpenVINO Demo Mode** — run the policy on preset targets (logs to `logs/`)
- **Stop** — stop the current demo

## Project structure

| Path | Purpose |
|------|---------|
| `gpu_mode/` | Headless DQN training (Colab GPU) |
| `gpu_mode_ppo/` | PPO training (Colab) |
| `gpu_mode_sac/` | SAC training (Colab) |
| `main_gui.py` | Demo dashboard + OpenVINO conversion |
| `environment.py` | PyBullet environment |
| `agent.py` | DQN (training) and OpenVINO inference |
| `utils.py` | Image preprocessing (84×84) |
| `robot_urdf/` | Arm URDF and assets |
| `models/` | Saved checkpoints |
| `logs/` | Training and demo logs |
