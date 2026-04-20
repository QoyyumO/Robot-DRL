"""
PyBullet wrapper for vision-based robotic arm control.
DQN observes the full scene (arm + target) as an 84x84 image — vision only, no position sensors.
The arm must reach a target object; reward is distance-based.
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


# Default camera image size (raw feed for GUI)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Actions: 0=noop, 1=extend, 2=curl, 3=left, 4=right (move end effector in plane)
ACTION_EXTEND = 1   
ACTION_CURL = 2    
ACTION_LEFT = 3
ACTION_RIGHT = 4

# Target reaching: reward = +1 closer, -1 farther, 0 same; terminal when sum(last 3) < -1
COMPLETION_RADIUS = 0.08  # success when end-effector within this (m) of target (8 cm demo threshold)


class VisionControlEnv:
    """
    Target reaching task: robotic arm must reach a target.
    DQN observation = full scene (arm + target) as 84x84 image — vision only, no position sensors.
    The network learns from pixels where the arm is relative to the target.
    """

    # Short arm: 0.5 m per segment (project-local URDF)
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    ARM_URDF = os.path.join(_PROJECT_ROOT, "robot_urdf", "TwoJointRobot_short.urdf")
    ARM_END_EFFECTOR_LINK = 2  # link_2 at tip of second arm segment

    def __init__(
        self,
        headless: bool = True,
        camera_width: int = CAMERA_WIDTH,
        camera_height: int = CAMERA_HEIGHT,
        time_step: float = 1.0 / 60.0,
        target_mode: str = "random",
        preset_targets: Optional[Sequence[Tuple[float, float, float]]] = None,
        enable_early_truncation: bool = True,
    ):
        if not HAS_PYBULLET:
            raise RuntimeError("PyBullet is required. Install with: pip install pybullet")
        self._headless = headless
        self._cam_w = camera_width
        self._cam_h = camera_height
        self._time_step = time_step
        self._client_id: Optional[int] = None
        self._plane_id: Optional[int] = None
        self._arm_id: Optional[int] = None
        self._target_id: Optional[int] = None
        self._cam_distance = 1.5
        self._cam_yaw = 60
        self._cam_pitch = -30
        self._cam_target = [0.5, 0, 0.08]  # look at workspace of short 2-link arm
        self._view_matrix: Optional[np.ndarray] = None
        self._proj_matrix: Optional[np.ndarray] = None
        self._n_actions = 5
        self._step_count = 0
        self._max_steps = 500
        self._prev_distance: float = 0.0
        self._last_3_rewards: list = []
        # Target position in 3D (set at reset)
        self._target_pos_3d: Optional[Tuple[float, float, float]] = None
        self._target_mode = target_mode
        self._preset_targets = list(preset_targets) if preset_targets else []
        self._preset_target_idx = 0
        self._enable_early_truncation = enable_early_truncation
        if self._target_mode not in ("random", "preset_cycle"):
            raise ValueError("target_mode must be 'random' or 'preset_cycle'")
        if self._target_mode == "preset_cycle" and not self._preset_targets:
            raise ValueError("preset_targets must be provided when target_mode is 'preset_cycle'")

    def _sample_random_target(self) -> Tuple[float, float, float]:
        """Sample random target in front of the short 2-link arm."""
        tx = float(np.random.uniform(0.25, 0.85))
        ty = float(np.random.uniform(-0.35, 0.35))
        tz = float(0.05 + np.random.uniform(0, 0.08))
        return (tx, ty, tz)

    def _next_target_position(self) -> Tuple[float, float, float]:
        """Return next target according to configured target mode."""
        if self._target_mode == "preset_cycle":
            target = tuple(self._preset_targets[self._preset_target_idx % len(self._preset_targets)])
            self._preset_target_idx += 1
            return target
        return self._sample_random_target()

    def reset(self, target_position: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, dict]:
        """Reset: new target position, arm at default. Obs = target-location image only (no arm)."""
        if self._client_id is not None:
            p.disconnect(self._client_id)
        if self._headless:
            self._client_id = p.connect(p.DIRECT)
        else:
            self._client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(self._time_step, physicsClientId=self._client_id)

        self._plane_id = p.loadURDF(
            "plane.urdf", [0, 0, 0], physicsClientId=self._client_id
        )
        self._arm_id = p.loadURDF(
            self.ARM_URDF, [0, 0, 0], useFixedBase=True, physicsClientId=self._client_id
        )
        # Joint control: get revolute joint indices and limits; map by name so base=left/right, elbow=extend/curl
        self._n_joints = p.getNumJoints(self._arm_id, physicsClientId=self._client_id)
        joint_1_body_idx = None  # base (left/right)
        joint_2_body_idx = None  # elbow (extend/curl)
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
        # Order: [base (joint_1), elbow (joint_2)]
        # Base (joint_1) is continuous in URDF: use large limits so np.clip doesn't lock it at 0
        BASE_CONTINUOUS_LIMIT = 4 * np.pi  # allow full 360° rotation
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
        self._joint_delta = 0.03  # rad per step
        for j in range(len(self._joint_indices)):
            p.setJointMotorControl2(
                self._arm_id, self._joint_indices[j], p.POSITION_CONTROL,
                targetPosition=self._joint_positions[j], force=100, physicsClientId=self._client_id
            )
        # Allow explicit target override; otherwise sample from configured strategy.
        self._target_pos_3d = tuple(target_position) if target_position is not None else self._next_target_position()
        # Target sphere at that position (fixed in place)
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
            self._target_id = p.createMultiBody(0, col, -1, self._target_pos_3d, [0, 0, 0, 1], physicsClientId=self._client_id)

        self._step_count = 0
        self._last_3_rewards = []
        self._prev_distance = self._distance_ee_to_target()

        # DQN observation = full scene (arm + target) 84x84 — image only, no position sensors
        obs = self._get_dqn_obs()
        rgb = self._get_camera_rgb()
        return obs, {
            "rgb": rgb,
            "dqn_obs": obs,
            "target_pos": self._target_pos_3d,
            "ee_pos": self._get_end_effector_pos(),
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step: move arm joints (position control). Reward: +1 closer, -1 farther. Terminal when sum(last 3) < -1 or success."""
        if self._client_id is None or self._arm_id is None:
            raise RuntimeError("Call reset() first.")

        # Update joint targets: 0=base (left/right), 1=elbow (extend/curl)
        d = self._joint_delta
        n = len(self._joint_indices)
        if action == ACTION_EXTEND:
            if n > 1:
                self._joint_positions[1] = np.clip(self._joint_positions[1] - d, self._joint_lower[1], self._joint_upper[1])
        elif action == ACTION_CURL:
            if n > 1:
                self._joint_positions[1] = np.clip(self._joint_positions[1] + d, self._joint_lower[1], self._joint_upper[1])
        elif action == ACTION_LEFT:
            if n > 0:
                self._joint_positions[0] = np.clip(self._joint_positions[0] + d, self._joint_lower[0], self._joint_upper[0])
        elif action == ACTION_RIGHT:
            if n > 0:
                self._joint_positions[0] = np.clip(self._joint_positions[0] - d, self._joint_lower[0], self._joint_upper[0])
        for j in range(len(self._joint_indices)):
            p.setJointMotorControl2(
                self._arm_id, self._joint_indices[j], p.POSITION_CONTROL,
                targetPosition=self._joint_positions[j], force=100, physicsClientId=self._client_id
            )
        p.stepSimulation(physicsClientId=self._client_id)
        self._step_count += 1
        # Keep target fixed: reset position and velocity so it never moves
        p.resetBasePositionAndOrientation(
            self._target_id, list(self._target_pos_3d), [0, 0, 0, 1], physicsClientId=self._client_id
        )
        p.resetBaseVelocity(self._target_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client_id)

        dist = self._distance_ee_to_target()
        # reward: +1 closer, -1 farther, 0 same
        dis_change = dist - self._prev_distance
        if dis_change > 0:
            reward = -1.0
        elif dis_change < 0:
            reward = 1.0
        else:
            reward = 0.0
        self._prev_distance = dist
        self._last_3_rewards.append(reward)
        if len(self._last_3_rewards) > 3:
            self._last_3_rewards.pop(0)

        # Terminal: sum of last 3 rewards < -1 (Algorithm 1) or success or max_steps
        sum_last_3 = sum(self._last_3_rewards) if len(self._last_3_rewards) >= 3 else 0
        done = self._step_count >= self._max_steps or dist < COMPLETION_RADIUS
        truncated = (sum_last_3 < -1) if self._enable_early_truncation else False
        success = dist < COMPLETION_RADIUS
        if success:
            reward += 5.0  # bonus for reaching target

        obs = self._get_dqn_obs()
        rgb = self._get_camera_rgb()
        return obs, reward, done, truncated, {
            "rgb": rgb,
            "dqn_obs": obs,
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
        ee_pos = np.array(ee)
        t = np.array(self._target_pos_3d)
        return float(np.linalg.norm(ee_pos - t))

    def _get_end_effector_pos(self) -> Optional[Tuple[float, float, float]]:
        """Current end-effector world position (x, y, z)."""
        if self._arm_id is None or self._client_id is None:
            return None
        link_state = p.getLinkState(
            self._arm_id, self.ARM_END_EFFECTOR_LINK, physicsClientId=self._client_id
        )
        ee = link_state[0]
        return (float(ee[0]), float(ee[1]), float(ee[2]))

    def _get_view_proj_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Camera view and projection as 4x4 (column-major from PyBullet)."""
        vm = p.computeViewMatrixFromYawPitchRoll(
            self._cam_target, self._cam_distance,
            self._cam_yaw, self._cam_pitch, 0, upAxisIndex=2,
        )
        pm = p.computeProjectionMatrixFOV(
            60, self._cam_w / self._cam_h, 0.1, 10
        )
        view = np.array(vm).reshape(4, 4).T
        proj = np.array(pm).reshape(4, 4).T
        return view, proj

    def _project_3d_to_84x84(self, pos_3d: Tuple[float, float, float]) -> Tuple[int, int]:
        """Project world position to pixel (i, j) in 84x84 image (row, col)."""
        view, proj = self._get_view_proj_matrices()
        p4 = np.array([pos_3d[0], pos_3d[1], pos_3d[2], 1.0], dtype=np.float64)
        p_cam = view @ p4
        p_clip = proj @ p_cam
        if abs(p_clip[3]) < 1e-6:
            return 42, 42
        ndc_x = p_clip[0] / p_clip[3]
        ndc_y = p_clip[1] / p_clip[3]
        # NDC to pixel: x right, y up in NDC; image row = down
        u = (ndc_x + 1.0) * 0.5 * self._cam_w
        v = (1.0 - ndc_y) * 0.5 * self._cam_h
        # Scale to 84x84
        i = int(v * PREPROCESS_HEIGHT / self._cam_h)
        j = int(u * PREPROCESS_WIDTH / self._cam_w)
        i = max(0, min(PREPROCESS_HEIGHT - 1, i))
        j = max(0, min(PREPROCESS_WIDTH - 1, j))
        return i, j

    def _get_target_location_obs(self) -> np.ndarray:
        """
        Legacy: target-only 84x84 (used for GUI display of 'target location' view if needed).
        """
        out = np.zeros((PREPROCESS_HEIGHT, PREPROCESS_WIDTH), dtype=np.float32)
        if self._target_pos_3d is None:
            return out
        i, j = self._project_3d_to_84x84(self._target_pos_3d)
        r = 5
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < PREPROCESS_HEIGHT and 0 <= nj < PREPROCESS_WIDTH:
                    d = np.sqrt(di * di + dj * dj)
                    out[ni, nj] = max(out[ni, nj], float(np.clip(1.0 - d / (r + 1), 0, 1)))
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _get_dqn_obs(self) -> np.ndarray:
        """
        DQN observation from image alone: full scene (arm + target) preprocessed to 84x84.
        No position sensors — the policy learns from pixels only (where the arm is vs where the target is).
        """
        rgb = self._get_camera_rgb()
        return preprocess_frame(rgb, width=PREPROCESS_WIDTH, height=PREPROCESS_HEIGHT)

    def get_camera_image_rgb(self) -> Optional[np.ndarray]:
        """Full scene (arm + target) for GUI."""
        if self._client_id is None:
            return None
        return self._get_camera_rgb()

    def _get_camera_rgb(self) -> np.ndarray:
        """Full scene RGB (arm + target) for Simulation View."""
        renderer = p.ER_TINY_RENDERER if self._headless else p.ER_BULLET_HARDWARE_OPENGL
        view = p.getCameraImage(
            self._cam_w,
            self._cam_h,
            viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                self._cam_target, self._cam_distance,
                self._cam_yaw, self._cam_pitch, 0, upAxisIndex=2,
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(60, self._cam_w / self._cam_h, 0.1, 10),
            renderer=renderer,
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
