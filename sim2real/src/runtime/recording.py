from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from paths import SIM2REAL_ROOT


def _sanitize_filename_part(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    clean = clean.strip("._-")
    return clean or "unknown"


def _ckpt_name_from_policy_path(policy_path: str | Path) -> str:
    path = Path(policy_path)
    if path.parent.name:
        return path.parent.name
    return path.stem


class PolicyRunRecorder:
    def __init__(
        self,
        *,
        robot: str,
        policy,
        joint_names: list[str],
        control_dt: float,
        output_root: Optional[str | Path] = None,
    ):
        self.robot = _sanitize_filename_part(robot)
        self.policy = policy
        self.joint_names = list(joint_names)
        self.control_dt = float(control_dt)
        if output_root is None:
            self.output_root = SIM2REAL_ROOT / "assets" / "policy_logs"
        else:
            root = Path(output_root)
            self.output_root = root if root.is_absolute() else SIM2REAL_ROOT / root

        self.ckpt_name = _sanitize_filename_part(_ckpt_name_from_policy_path(policy.policy_path))
        self.output_dir = self.output_root / self.robot
        self.output_path = self._next_output_path()

        self._saved = False
        self._frames: list[dict[str, np.ndarray | np.integer | np.floating | str]] = []
        self._start_wall_ns = time.time_ns()
        self._start_monotonic = time.monotonic()

    def _next_output_path(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{timestamp}_{self.ckpt_name}"
        path = self.output_dir / f"{base}.npz"
        if not path.exists():
            return path
        for idx in range(1, 1000):
            candidate = self.output_dir / f"{base}_{idx:03d}.npz"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Cannot find free recorder output path under {self.output_dir}")

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def record_step(self, controller, action_delta: np.ndarray) -> None:
        policy = self.policy
        self._frames.append(
            {
                "wall_time_ns": np.int64(time.time_ns()),
                "monotonic_time_s": np.float64(time.monotonic()),
                "policy_step": np.int64(controller.policy_step),
                "joint_pos": controller.qj.astype(np.float32, copy=True),
                "joint_vel": controller.dqj.astype(np.float32, copy=True),
                "action_delta": np.asarray(action_delta, dtype=np.float32).copy(),
                "cmd_q": controller.cmd_q.astype(np.float32, copy=True),
                "cmd_qd": controller.cmd_qd.astype(np.float32, copy=True),
                "cmd_kp": controller.cmd_kp.astype(np.float32, copy=True),
                "cmd_kd": controller.cmd_kd.astype(np.float32, copy=True),
                "cmd_enable": np.int8(controller.cmd_enable),
                "imu_angvel": controller.gyro.astype(np.float32, copy=True),
                "imu_linacc": controller.linacc.astype(np.float32, copy=True),
                "imu_quat_wxyz": controller.quat.astype(np.float32, copy=True),
                "policy_action_raw": policy.last_action.astype(np.float32, copy=True),
                "policy_action_scaled": policy.applied_action.astype(np.float32, copy=True),
                "ref_idx": np.int64(getattr(policy, "ref_idx", -1)),
                "ref_len": np.int64(getattr(policy, "ref_len", -1)),
                "motion_name": str(getattr(policy, "current_name", "")),
            }
        )

    def save(self) -> Optional[Path]:
        if self._saved:
            return self.output_path
        self._saved = True

        if len(self._frames) == 0:
            print("[PolicyRunRecorder] no policy frames captured; skip save")
            return None

        keys = self._frames[0].keys()
        payload = {}
        for key in keys:
            payload[key] = np.asarray([frame[key] for frame in self._frames])

        payload.update(
            {
                "robot": np.asarray(self.robot),
                "ckpt_name": np.asarray(self.ckpt_name),
                "policy_name": np.asarray(str(getattr(self.policy, "name", ""))),
                "policy_path": np.asarray(str(getattr(self.policy, "policy_path", ""))),
                "joint_names": np.asarray(self.joint_names),
                "action_joint_names": np.asarray(list(getattr(self.policy, "action_joint_names", []))),
                "control_dt": np.float32(self.control_dt),
                "control_freq": np.float32(1.0 / self.control_dt if self.control_dt > 0.0 else 0.0),
                "start_wall_time_ns": np.int64(self._start_wall_ns),
                "start_monotonic_time_s": np.float64(self._start_monotonic),
            }
        )

        np.savez_compressed(self.output_path, **payload)
        print(f"[PolicyRunRecorder] saved={self.output_path} frames={len(self._frames)}")
        return self.output_path
