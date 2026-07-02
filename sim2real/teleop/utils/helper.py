from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .math import quat_slerp_np


def default_controller_buttons() -> dict[str, Any]:
    return {
        "left_key_one": False,
        "left_key_two": False,
        "left_axis_click": False,
        "left_index_trig": False,
        "left_grip": False,
        "left_axis": [0.0, 0.0],
        "right_key_one": False,
        "right_key_two": False,
        "right_axis_click": False,
        "right_index_trig": False,
        "right_grip": False,
        "right_axis": [0.0, 0.0],
    }


@dataclass(frozen=True)
class ParsedXRobotMotionSnapshot:
    controller_buttons: dict[str, Any]
    motion_timestamp_ns: int
    poses: np.ndarray


def interpolate_qpos(prev_qpos: np.ndarray, next_qpos: np.ndarray, alpha: float) -> np.ndarray:
    t = float(np.clip(alpha, 0.0, 1.0))
    frame = prev_qpos * (1.0 - t) + next_qpos * t
    frame[3:7] = quat_slerp_np(prev_qpos[3:7], next_qpos[3:7], t)
    return frame.astype(np.float32)


def _parse_ns_timestamp(value: Any) -> int | None:
    try:
        timestamp_ns = int(value)
    except Exception:
        return None
    return timestamp_ns if timestamp_ns > 0 else None


def _parse_axis(values: Any) -> list[float] | None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < 2:
        return None
    try:
        return [float(arr[0]), float(arr[1])]
    except Exception:
        return None


def parse_xrobot_motion_snapshot(
    snapshot: Optional[dict],
    *,
    joint_count: int,
) -> ParsedXRobotMotionSnapshot | None:
    if not isinstance(snapshot, dict):
        return None

    top_timestamp_ns = _parse_ns_timestamp(snapshot.get("timestamp_ns", None))
    if top_timestamp_ns is None:
        return None

    controllers = snapshot.get("controllers", {})
    if not isinstance(controllers, dict):
        return None
    left = controllers.get("left", {})
    right = controllers.get("right", {})
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None

    left_axis = _parse_axis(left.get("axis", [0.0, 0.0]))
    right_axis = _parse_axis(right.get("axis", [0.0, 0.0]))
    if left_axis is None or right_axis is None:
        return None

    try:
        left_trigger = float(left.get("trigger", 0.0))
        left_grip = float(left.get("grip", 0.0))
        right_trigger = float(right.get("trigger", 0.0))
        right_grip = float(right.get("grip", 0.0))
    except Exception:
        return None

    body = snapshot.get("body", {})
    if not isinstance(body, dict) or not bool(body.get("available", False)):
        return None
    body_timestamp_ns = _parse_ns_timestamp(body.get("timestamp_ns", None))
    motion_timestamp_ns = body_timestamp_ns if body_timestamp_ns is not None else top_timestamp_ns

    poses = body.get("poses", None)
    try:
        pose_arr = np.asarray(poses, dtype=np.float32)
    except Exception:
        return None
    if pose_arr.ndim != 2 or pose_arr.shape[0] < int(joint_count) or pose_arr.shape[1] < 7:
        return None
    pose_arr = np.ascontiguousarray(pose_arr[: int(joint_count), :7], dtype=np.float32)

    return ParsedXRobotMotionSnapshot(
        controller_buttons={
            "left_key_one": bool(left.get("primary_button", False)),
            "left_key_two": bool(left.get("secondary_button", False)),
            "left_axis_click": bool(left.get("axis_click", False)),
            "left_index_trig": left_trigger > 1e-4,
            "left_grip": left_grip > 1e-4,
            "left_axis": left_axis,
            "right_key_one": bool(right.get("primary_button", False)),
            "right_key_two": bool(right.get("secondary_button", False)),
            "right_axis_click": bool(right.get("axis_click", False)),
            "right_index_trig": right_trigger > 1e-4,
            "right_grip": right_grip > 1e-4,
            "right_axis": right_axis,
        },
        motion_timestamp_ns=motion_timestamp_ns,
        poses=pose_arr,
    )
