"""
PyBullet wrapper for vision-based robotic arm control.
GPU/headless version: always uses p.DIRECT (no display) and p.ER_TINY_RENDERER.
URDF path resolves to the parent project directory (../robot_urdf/).
"""

import os
import numpy as np
from typing import Tuple, Any, Optional, Sequence

try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

from utils import preprocess_frame, PREPROCESS_WIDTH, PREPROCESS_HEIGHT
from config_loader import cfg

_env_cfg = cfg["environment"]
_rew_cfg = cfg["rewards"]

CAMERA_WIDTH: int = _env_cfg["camera_width"]
CAMERA_HEIGHT: int = _env_cfg["camera_height"]

ACTION_EXTEND = 1
ACTION_CURL = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

COMPLETION_RADIUS: float = _env_cfg["completion_radius"]


class VisionControlEnv:
    """
    Target reaching task: robotic arm must reach a target.
    DQN observation = full scene (arm + target) as 84x84 image — vision only.
    Always runs headless (p.DIRECT) — safe for Colab and any no-display environment.
    """

    # URDF lives one level up (in the parent Robot-DRL project directory)
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARM_URDF = os.path.join(_PROJECT_ROOT, "robot_urdf", "TwoJointRobot_short.urdf")
    ARM_END_EFFECTOR_LINK = 2

    def __init__(
        self,
        camera_width: int = CAMERA_WIDTH,
        camera_height: int = CAMERA_HEIGHT,
        time_step: float = _env_cfg["physics_time_step"],
        target_mode: str = "random",
        preset_targets: Optional[Sequence[Tuple[float, float, float]]] = None,
        enable_early_truncation: bool = True,
    ):
        if not HAS_PYBULLET:
            raise RuntimeError("PyBullet is required. Install with: pip install pybullet")
        self._cam_w = camera_width
        self._cam_h = camera_height
        self._time_step = time_step
        self._client_id: Optional[int] = None
        self._plane_id: Optional[int] = None
        self._arm_id: Optional[int] = None
        self._target_id: Optional[int] = None
        self._cam_distance: float = _env_cfg["camera_distance"]
        self._cam_yaw: float = _env_cfg["camera_yaw"]
        self._cam_pitch: float = _env_cfg["camera_pitch"]
        self._cam_target: list = list(_env_cfg["camera_target"])
        self._n_actions = 5
        self._step_count = 0
        self._max_steps: int = _env_cfg["max_steps_per_episode"]
        self._prev_distance: float = 0.0
        self._last_3_rewards: list = []
        self._target_pos_3d: Optional[Tuple[float, float, float]] = None
        self._target_mode = target_mode
        self._preset_targets = list(preset_targets) if preset_targets else []
        self._preset_target_idx = 0
        self._enable_early_truncation = enable_early_truncation
        self._early_truncation_threshold: int = _env_cfg["early_truncation_threshold"]
        if self._target_mode not in ("random", "preset_cycle"):
            raise ValueError("target_mode must be 'random' or 'preset_cycle'")
        if self._target_mode == "preset_cycle" and not self._preset_targets:
            raise ValueError("preset_targets must be provided when target_mode is 'preset_cycle'")

    def _sample_random_target(self) -> Tuple[float, float, float]:
        tx = float(np.random.uniform(_env_cfg["target_x_min"], _env_cfg["target_x_max"]))
        ty = float(np.random.uniform(_env_cfg["target_y_min"], _env_cfg["target_y_max"]))
        tz = float(_env_cfg["target_z_min"] + np.random.uniform(0, _env_cfg["target_z_range"]))
        return (tx, ty, tz)

    def _next_target_position(self) -> Tuple[float, float, float]:
        if self._target_mode == "preset_cycle":
            target = tuple(self._preset_targets[self._preset_target_idx % len(self._preset_targets)])
            self._preset_target_idx += 1
            return target
        return self._sample_random_target()

    def reset(self, target_position: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, dict]:
        """Reset simulation. Always uses p.DIRECT (headless)."""
        if self._client_id is not None:
            p.disconnect(self._client_id)

        # Always DIRECT — no display server needed
        self._client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, _env_cfg["gravity"])
        p.setTimeStep(self._time_step, physicsClientId=self._client_id)

        self._plane_id = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self._client_id)
        self._arm_id = p.loadURDF(
            self.ARM_URDF, [0, 0, 0], useFixedBase=True, physicsClientId=self._client_id
        )

        self._n_joints = p.getNumJoints(self._arm_id, physicsClientId=self._client_id)
        joint_1_body_idx = None
        joint_2_body_idx = None
        joint_limits = {}
        for j in range(self._n_joints):
            info = p.getJointInfo(self._arm_id, j, physicsClientId=self._client_id)
            if info[2] == p.JOINT_REVOLUTE:
                name = info[1].decode("utf-8")
                joint_limits[j] = (info[8], info[9])
                if name == "joint_1":
                    joint_1_body_idx = j
                elif name == "joint_2":
                    joint_2_body_idx = j

        BASE_CONTINUOUS_LIMIT = 4 * np.pi
        self._joint_indices = []
        self._joint_lower = []
        self._joint_upper = []
        for i, idx in enumerate((joint_1_body_idx, joint_2_body_idx)):
            if idx is not None:
                self._joint_indices.append(idx)
                lo, hi = joint_limits[idx]
                if i == 0 and joint_1_body_idx is not None and idx == joint_1_body_idx:
                    lo, hi = -BASE_CONTINUOUS_LIMIT, BASE_CONTINUOUS_LIMIT
                self._joint_lower.append(lo)
                self._joint_upper.append(hi)

        self._joint_positions = [0.0] * len(self._joint_indices)
        self._joint_delta: float = _env_cfg["joint_delta"]
        _joint_force: int = _env_cfg["joint_force"]
        for j in range(len(self._joint_indices)):
            p.setJointMotorControl2(
                self._arm_id, self._joint_indices[j], p.POSITION_CONTROL,
                targetPosition=self._joint_positions[j], force=_joint_force,
                physicsClientId=self._client_id,
            )

        self._target_pos_3d = (
            tuple(target_position) if target_position is not None else self._next_target_position()
        )

        try:
            self._target_id = p.loadURDF(
                "sphere_small.urdf",
                self._target_pos_3d,
                [0, 0, 0, 1],
                useFixedBase=True,
                physicsClientId=self._client_id,
            )
        except Exception:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03, physicsClientId=self._client_id)
            self._target_id = p.createMultiBody(
                0, col, -1, self._target_pos_3d, [0, 0, 0, 1], physicsClientId=self._client_id
            )

        self._step_count = 0
        self._last_3_rewards = []
        self._prev_distance = self._distance_ee_to_target()

        obs = self._get_dqn_obs()
        return obs, {
            "target_pos": self._target_pos_3d,
            "ee_pos": self._get_end_effector_pos(),
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step simulation. Reward: +1 closer, -1 farther."""
        if self._client_id is None or self._arm_id is None:
            raise RuntimeError("Call reset() first.")

        d = self._joint_delta
        n = len(self._joint_indices)
        if action == ACTION_EXTEND:
            if n > 1:
                self._joint_positions[1] = np.clip(
                    self._joint_positions[1] - d, self._joint_lower[1], self._joint_upper[1]
                )
        elif action == ACTION_CURL:
            if n > 1:
                self._joint_positions[1] = np.clip(
                    self._joint_positions[1] + d, self._joint_lower[1], self._joint_upper[1]
                )
        elif action == ACTION_LEFT:
            if n > 0:
                self._joint_positions[0] = np.clip(
                    self._joint_positions[0] + d, self._joint_lower[0], self._joint_upper[0]
                )
        elif action == ACTION_RIGHT:
            if n > 0:
                self._joint_positions[0] = np.clip(
                    self._joint_positions[0] - d, self._joint_lower[0], self._joint_upper[0]
                )

        _joint_force: int = _env_cfg["joint_force"]
        for j in range(len(self._joint_indices)):
            p.setJointMotorControl2(
                self._arm_id, self._joint_indices[j], p.POSITION_CONTROL,
                targetPosition=self._joint_positions[j], force=_joint_force,
                physicsClientId=self._client_id,
            )
        p.stepSimulation(physicsClientId=self._client_id)
        self._step_count += 1

        p.resetBasePositionAndOrientation(
            self._target_id, list(self._target_pos_3d), [0, 0, 0, 1],
            physicsClientId=self._client_id,
        )
        p.resetBaseVelocity(
            self._target_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client_id
        )

        dist = self._distance_ee_to_target()
        dis_change = dist - self._prev_distance
        if dis_change > 0:
            reward = _rew_cfg["farther"]
        elif dis_change < 0:
            reward = _rew_cfg["closer"]
        else:
            reward = _rew_cfg["same"]
        self._prev_distance = dist
        self._last_3_rewards.append(reward)
        if len(self._last_3_rewards) > 3:
            self._last_3_rewards.pop(0)

        sum_last_3 = sum(self._last_3_rewards) if len(self._last_3_rewards) >= 3 else 0
        done = self._step_count >= self._max_steps or dist < COMPLETION_RADIUS
        truncated = (sum_last_3 < self._early_truncation_threshold) if self._enable_early_truncation else False
        success = dist < COMPLETION_RADIUS
        if success:
            reward += _rew_cfg["success_bonus"]

        obs = self._get_dqn_obs()
        return obs, reward, done, truncated, {
            "step": self._step_count,
            "distance": dist,
            "success": success,
            "target_pos": self._target_pos_3d,
            "ee_pos": self._get_end_effector_pos(),
        }

    def _distance_ee_to_target(self) -> float:
        if self._target_pos_3d is None:
            return float("inf")
        ee = self._get_end_effector_pos()
        if ee is None:
            return float("inf")
        return float(np.linalg.norm(np.array(ee) - np.array(self._target_pos_3d)))

    def _get_end_effector_pos(self) -> Optional[Tuple[float, float, float]]:
        if self._arm_id is None or self._client_id is None:
            return None
        link_state = p.getLinkState(
            self._arm_id, self.ARM_END_EFFECTOR_LINK, physicsClientId=self._client_id
        )
        ee = link_state[0]
        return (float(ee[0]), float(ee[1]), float(ee[2]))

    def _get_dqn_obs(self) -> np.ndarray:
        """Full scene (arm + target) preprocessed to 84x84 — vision only."""
        rgb = self._get_camera_rgb()
        return preprocess_frame(rgb, width=PREPROCESS_WIDTH, height=PREPROCESS_HEIGHT)

    def _get_camera_rgb(self) -> np.ndarray:
        """Render frame using p.ER_TINY_RENDERER (CPU, no display required)."""
        view = p.getCameraImage(
            self._cam_w,
            self._cam_h,
            viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                self._cam_target, self._cam_distance,
                self._cam_yaw, self._cam_pitch, 0, upAxisIndex=2,
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                _env_cfg["camera_fov"], self._cam_w / self._cam_h, 0.1, 10
            ),
            renderer=p.ER_TINY_RENDERER,  # CPU renderer — no OpenGL/display needed
            physicsClientId=self._client_id,
        )
        rgb = np.reshape(view[2], (self._cam_h, self._cam_w, 4))[:, :, :3]
        return np.ascontiguousarray(rgb)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (PREPROCESS_HEIGHT, PREPROCESS_WIDTH)

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def close(self) -> None:
        if self._client_id is not None:
            p.disconnect(self._client_id)
            self._client_id = None
        self._plane_id = None
        self._arm_id = None
        self._target_id = None
