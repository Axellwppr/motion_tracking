#!/usr/bin/env python3
"""
Record raw XRoboToolkit/XRobot body stream to NPZ without retargeting or ZMQ.

Behavior:
- Uses the XR callback API directly.
- Records the raw body poses in the original XRobot layout: [x, y, z, qx, qy, qz, qw].
- Uses controller buttons to gate recording by default:
  - right primary button (A) starts recording
  - left primary button (X) stops and saves
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from retarget.params import XR_BODY_JOINT_NAMES
from utils.helper import default_controller_buttons, parse_xrobot_motion_snapshot
from utils.xrobot_sdk import load_xrobotoolkit_sdk

TELEOP_ROOT = Path(__file__).resolve().parent
SIM2REAL_ROOT = TELEOP_ROOT.parent
DEFAULT_OUTPUT_DIR = SIM2REAL_ROOT / "assets" / "data" / "xrobot_raw"

START_BUTTON = "right_key_one"
STOP_BUTTON = "left_key_one"

xrt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record raw XRobot body stream to NPZ.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for recorded raw XRobot segments.",
    )
    parser.add_argument(
        "--prefix",
        default="xrobot_raw",
        help="Filename prefix for saved segments.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start recording immediately on launch; Ctrl+C saves and exits.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=50.0,
        help="Resample and save the recorded XRobot stream at this fixed FPS.",
    )
    return parser.parse_args()


def next_output_path(output_dir: Path, prefix: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    existing = sorted(output_dir.glob(f"{prefix}_{ts}_*.npz"))
    next_idx = len(existing)
    return output_dir / f"{prefix}_{ts}_{next_idx:08d}.npz"


def _compute_estimated_fps_ns(timestamps_ns: np.ndarray) -> np.float32:
    if timestamps_ns.size < 2:
        return np.float32(0.0)
    diffs = np.diff(timestamps_ns.astype(np.int64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return np.float32(0.0)
    median_dt_ns = float(np.median(diffs))
    return np.float32(1e9 / median_dt_ns)


class XRobotRawRecorder:
    def __init__(self, output_dir: Path, prefix: str, auto_start: bool, target_fps: float):
        self.output_dir = output_dir
        self.prefix = prefix
        self.auto_start = bool(auto_start)
        self.target_fps = float(target_fps)
        if self.target_fps <= 0.0:
            raise ValueError("target_fps must be > 0")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.recording = False
        self.prev_buttons = default_controller_buttons()
        self.completed_segments: deque[dict[str, Any]] = deque()
        self.callback_count = 0
        self.last_log_t = 0.0

        self._reset_active_buffers()
        if self.auto_start:
            self.recording = True
            print("[RawRecorder] auto-start enabled")

    def _reset_active_buffers(self) -> None:
        self.segment_recv_ns_start = 0
        self.segment_recv_ns_stop = 0
        self.last_recorded_motion_timestamp_ns: Optional[int] = None
        self.poses_list: list[np.ndarray] = []
        self.recv_ns_list: list[int] = []
        self.motion_timestamp_ns_list: list[int] = []
        self.left_key_one_list: list[bool] = []
        self.left_key_two_list: list[bool] = []
        self.left_axis_click_list: list[bool] = []
        self.left_index_trig_list: list[bool] = []
        self.left_grip_list: list[bool] = []
        self.left_axis_list: list[list[float]] = []
        self.right_key_one_list: list[bool] = []
        self.right_key_two_list: list[bool] = []
        self.right_axis_click_list: list[bool] = []
        self.right_index_trig_list: list[bool] = []
        self.right_grip_list: list[bool] = []
        self.right_axis_list: list[list[float]] = []

    def _start_recording_locked(self, recv_ns: int) -> None:
        self.recording = True
        self._reset_active_buffers()
        self.segment_recv_ns_start = int(recv_ns)
        print("[RawRecorder] start recording")

    def _finalize_active_segment_locked(self, recv_ns: int) -> Optional[dict[str, Any]]:
        self.recording = False
        self.segment_recv_ns_stop = int(recv_ns)
        if not self.poses_list:
            print("[RawRecorder] stop recording | no frames captured, skip save")
            self._reset_active_buffers()
            return None

        segment = {
            "segment_recv_ns_start": int(self.segment_recv_ns_start),
            "segment_recv_ns_stop": int(self.segment_recv_ns_stop),
            "poses": np.asarray(self.poses_list, dtype=np.float32).reshape(-1, len(XR_BODY_JOINT_NAMES), 7),
            "recv_ns": np.asarray(self.recv_ns_list, dtype=np.int64),
            "motion_timestamp_ns": np.asarray(self.motion_timestamp_ns_list, dtype=np.int64),
            "left_key_one": np.asarray(self.left_key_one_list, dtype=np.bool_),
            "left_key_two": np.asarray(self.left_key_two_list, dtype=np.bool_),
            "left_axis_click": np.asarray(self.left_axis_click_list, dtype=np.bool_),
            "left_index_trig": np.asarray(self.left_index_trig_list, dtype=np.bool_),
            "left_grip": np.asarray(self.left_grip_list, dtype=np.bool_),
            "left_axis": np.asarray(self.left_axis_list, dtype=np.float32).reshape(-1, 2),
            "right_key_one": np.asarray(self.right_key_one_list, dtype=np.bool_),
            "right_key_two": np.asarray(self.right_key_two_list, dtype=np.bool_),
            "right_axis_click": np.asarray(self.right_axis_click_list, dtype=np.bool_),
            "right_index_trig": np.asarray(self.right_index_trig_list, dtype=np.bool_),
            "right_grip": np.asarray(self.right_grip_list, dtype=np.bool_),
            "right_axis": np.asarray(self.right_axis_list, dtype=np.float32).reshape(-1, 2),
        }
        print(f"[RawRecorder] stop recording | frames={segment['poses'].shape[0]} | pending save")
        self._reset_active_buffers()
        return segment

    def _append_frame_locked(
        self,
        poses: np.ndarray,
        recv_ns: int,
        motion_timestamp_ns: int,
        buttons: dict[str, Any],
    ) -> None:
        self.poses_list.append(poses.copy())
        self.recv_ns_list.append(int(recv_ns))
        self.motion_timestamp_ns_list.append(int(motion_timestamp_ns))
        self.left_key_one_list.append(bool(buttons["left_key_one"]))
        self.left_key_two_list.append(bool(buttons["left_key_two"]))
        self.left_axis_click_list.append(bool(buttons["left_axis_click"]))
        self.left_index_trig_list.append(bool(buttons["left_index_trig"]))
        self.left_grip_list.append(bool(buttons["left_grip"]))
        self.left_axis_list.append([float(buttons["left_axis"][0]), float(buttons["left_axis"][1])])
        self.right_key_one_list.append(bool(buttons["right_key_one"]))
        self.right_key_two_list.append(bool(buttons["right_key_two"]))
        self.right_axis_click_list.append(bool(buttons["right_axis_click"]))
        self.right_index_trig_list.append(bool(buttons["right_index_trig"]))
        self.right_grip_list.append(bool(buttons["right_grip"]))
        self.right_axis_list.append([float(buttons["right_axis"][0]), float(buttons["right_axis"][1])])

    def on_frame(self, snapshot: dict) -> None:
        parsed_snapshot = parse_xrobot_motion_snapshot(snapshot, joint_count=len(XR_BODY_JOINT_NAMES))
        if parsed_snapshot is None:
            return

        recv_ns = time.monotonic_ns()
        buttons = parsed_snapshot.controller_buttons

        with self.lock:
            self.callback_count += 1

            start_pressed = bool(buttons.get(START_BUTTON, False))
            stop_pressed = bool(buttons.get(STOP_BUTTON, False))
            prev_start = bool(self.prev_buttons.get(START_BUTTON, False))
            prev_stop = bool(self.prev_buttons.get(STOP_BUTTON, False))

            if start_pressed and not prev_start and not self.recording:
                self._start_recording_locked(recv_ns)

            if (
                self.recording
                and parsed_snapshot.motion_timestamp_ns != self.last_recorded_motion_timestamp_ns
            ):
                self._append_frame_locked(
                    poses=parsed_snapshot.poses,
                    recv_ns=recv_ns,
                    motion_timestamp_ns=parsed_snapshot.motion_timestamp_ns,
                    buttons=buttons,
                )
                self.last_recorded_motion_timestamp_ns = int(parsed_snapshot.motion_timestamp_ns)

            if stop_pressed and not prev_stop and self.recording:
                segment = self._finalize_active_segment_locked(recv_ns)
                if segment is not None:
                    self.completed_segments.append(segment)

            self.prev_buttons = buttons

    def _save_segment(self, segment: dict[str, Any]) -> Path:
        output_path = next_output_path(self.output_dir, self.prefix)
        resampled = self._resample_segment(segment)
        raw_recv_ns = segment["recv_ns"]
        raw_motion_timestamp_ns = segment["motion_timestamp_ns"]
        fps_recv = _compute_estimated_fps_ns(raw_recv_ns)
        fps_motion = _compute_estimated_fps_ns(raw_motion_timestamp_ns)

        np.savez_compressed(
            output_path,
            fps=np.float32(self.target_fps),
            fps_recv=fps_recv,
            fps_motion=fps_motion,
            body_joint_names=np.asarray(XR_BODY_JOINT_NAMES),
            pose_layout=np.asarray(["x", "y", "z", "qx", "qy", "qz", "qw"]),
            poses=resampled["poses"],
            sample_ns=resampled["sample_ns"],
            recv_ns=resampled["source_recv_ns"],
            motion_timestamp_ns=resampled["source_motion_timestamp_ns"],
            segment_recv_ns_start=np.int64(segment["segment_recv_ns_start"]),
            segment_recv_ns_stop=np.int64(segment["segment_recv_ns_stop"]),
            left_key_one=resampled["left_key_one"],
            left_key_two=resampled["left_key_two"],
            left_axis_click=resampled["left_axis_click"],
            left_index_trig=resampled["left_index_trig"],
            left_grip=resampled["left_grip"],
            left_axis=resampled["left_axis"],
            right_key_one=resampled["right_key_one"],
            right_key_two=resampled["right_key_two"],
            right_axis_click=resampled["right_axis_click"],
            right_index_trig=resampled["right_index_trig"],
            right_grip=resampled["right_grip"],
            right_axis=resampled["right_axis"],
            raw_poses=segment["poses"],
            raw_recv_ns=raw_recv_ns,
            raw_motion_timestamp_ns=raw_motion_timestamp_ns,
            raw_left_key_one=segment["left_key_one"],
            raw_left_key_two=segment["left_key_two"],
            raw_left_axis_click=segment["left_axis_click"],
            raw_left_index_trig=segment["left_index_trig"],
            raw_left_grip=segment["left_grip"],
            raw_left_axis=segment["left_axis"],
            raw_right_key_one=segment["right_key_one"],
            raw_right_key_two=segment["right_key_two"],
            raw_right_axis_click=segment["right_axis_click"],
            raw_right_index_trig=segment["right_index_trig"],
            raw_right_grip=segment["right_grip"],
            raw_right_axis=segment["right_axis"],
        )
        print(
            f"[RawRecorder] saved={output_path} | raw_frames={segment['poses'].shape[0]} | "
            f"resampled_frames={resampled['poses'].shape[0]} | "
            f"fps_save={self.target_fps:.2f} | fps_motion={float(fps_motion):.2f}"
        )
        return output_path

    def _resample_segment(self, segment: dict[str, Any]) -> dict[str, np.ndarray]:
        raw_poses = np.asarray(segment["poses"], dtype=np.float32)
        raw_recv_ns = np.asarray(segment["recv_ns"], dtype=np.int64)
        raw_motion_timestamp_ns = np.asarray(segment["motion_timestamp_ns"], dtype=np.int64)

        time_ns = raw_motion_timestamp_ns.copy()
        invalid = time_ns <= 0
        time_ns[invalid] = raw_recv_ns[invalid]

        keep = np.ones(time_ns.shape[0], dtype=bool)
        if time_ns.shape[0] > 1:
            keep[1:] = time_ns[1:] > time_ns[:-1]

        src_time_ns = time_ns[keep]
        src_rel_s = (src_time_ns - src_time_ns[0]).astype(np.float64) / 1e9
        duration_s = float(src_rel_s[-1]) if src_rel_s.size > 0 else 0.0
        sample_count = max(1, int(np.floor(duration_s * self.target_fps + 1e-9)) + 1)
        target_rel_s = np.arange(sample_count, dtype=np.float64) / self.target_fps
        sample_ns = src_time_ns[0] + np.rint(target_rel_s * 1e9).astype(np.int64)

        poses_kept = raw_poses[keep]
        out_poses = np.empty((sample_count, poses_kept.shape[1], 7), dtype=np.float32)
        out_poses[..., :3] = self._resample_positions(poses_kept[..., :3], src_rel_s, target_rel_s)
        out_poses[..., 3:7] = self._resample_quaternions(poses_kept[..., 3:7], src_rel_s, target_rel_s)

        nearest_idx = self._nearest_indices(src_rel_s, target_rel_s)

        def _pick(name: str) -> np.ndarray:
            return np.asarray(segment[name])[keep][nearest_idx]

        return {
            "poses": out_poses,
            "sample_ns": sample_ns,
            "source_recv_ns": raw_recv_ns[keep][nearest_idx],
            "source_motion_timestamp_ns": raw_motion_timestamp_ns[keep][nearest_idx],
            "left_key_one": _pick("left_key_one"),
            "left_key_two": _pick("left_key_two"),
            "left_axis_click": _pick("left_axis_click"),
            "left_index_trig": _pick("left_index_trig"),
            "left_grip": _pick("left_grip"),
            "left_axis": _pick("left_axis"),
            "right_key_one": _pick("right_key_one"),
            "right_key_two": _pick("right_key_two"),
            "right_axis_click": _pick("right_axis_click"),
            "right_index_trig": _pick("right_index_trig"),
            "right_grip": _pick("right_grip"),
            "right_axis": _pick("right_axis"),
        }

    @staticmethod
    def _nearest_indices(src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
        if src_t.size == 1:
            return np.zeros(dst_t.shape[0], dtype=np.int64)
        right = np.searchsorted(src_t, dst_t, side="left")
        right = np.clip(right, 0, src_t.size - 1)
        left = np.clip(right - 1, 0, src_t.size - 1)
        choose_right = np.abs(src_t[right] - dst_t) < np.abs(dst_t - src_t[left])
        return np.where(choose_right, right, left).astype(np.int64)

    @staticmethod
    def _resample_positions(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
        if src_t.size == 1:
            return np.broadcast_to(values[:1], (dst_t.shape[0],) + values.shape[1:]).copy()
        out = np.empty((dst_t.shape[0],) + values.shape[1:], dtype=np.float32)
        joint_count = values.shape[1]
        for joint_idx in range(joint_count):
            for dim in range(3):
                out[:, joint_idx, dim] = np.interp(
                    dst_t, src_t, values[:, joint_idx, dim].astype(np.float64)
                ).astype(np.float32)
        return out

    @staticmethod
    def _resample_quaternions(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
        if src_t.size == 1:
            return np.broadcast_to(values[:1], (dst_t.shape[0],) + values.shape[1:]).copy()
        out = np.empty((dst_t.shape[0],) + values.shape[1:], dtype=np.float32)
        joint_count = values.shape[1]
        for joint_idx in range(joint_count):
            rots = R.from_quat(values[:, joint_idx, :].astype(np.float64))
            slerp = Slerp(src_t, rots)
            out[:, joint_idx, :] = slerp(dst_t).as_quat().astype(np.float32)
        return out

    def flush_completed_segments(self) -> int:
        saved = 0
        while True:
            with self.lock:
                if not self.completed_segments:
                    break
                segment = self.completed_segments.popleft()
            self._save_segment(segment)
            saved += 1
        return saved

    def flush_active_segment_on_exit(self) -> int:
        with self.lock:
            if not self.recording:
                return 0
            segment = self._finalize_active_segment_locked(time.monotonic_ns())
        if segment is None:
            return 0
        self._save_segment(segment)
        return 1

    def maybe_log_status(self) -> None:
        now = time.monotonic()
        if now - self.last_log_t < 1.0:
            return
        self.last_log_t = now
        with self.lock:
            recording = self.recording
            frames = len(self.poses_list)
            callbacks = self.callback_count
        print(f"[RawRecorder] callbacks={callbacks} recording={recording} buffered_frames={frames}")


def main() -> None:
    global xrt
    args = parse_args()
    xrt = load_xrobotoolkit_sdk()

    recorder = XRobotRawRecorder(
        output_dir=args.output_dir.expanduser().resolve(),
        prefix=str(args.prefix).strip(),
        auto_start=bool(args.auto_start),
        target_fps=float(args.target_fps),
    )

    xrt.init()
    xrt.register_frame_callback(recorder.on_frame)
    print("Raw XRobot recorder initialized")
    print(f"  output_dir: {recorder.output_dir}")
    print(f"  prefix: {recorder.prefix}")
    print(f"  body_joint_count: {len(XR_BODY_JOINT_NAMES)}")
    print(f"  target_fps: {recorder.target_fps:.2f}")
    print(f"  start_button: {START_BUTTON} (right primary/A)")
    print(f"  stop_button: {STOP_BUTTON} (left primary/X)")
    print(f"  auto_start: {recorder.auto_start}")
    print("Press A to start recording, X to stop and save. Ctrl+C exits.")

    try:
        while True:
            recorder.flush_completed_segments()
            recorder.maybe_log_status()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("[RawRecorder] KeyboardInterrupt, flushing pending data")
    finally:
        try:
            recorder.flush_completed_segments()
            recorder.flush_active_segment_on_exit()
        finally:
            try:
                xrt.clear_frame_callback()
            except Exception:
                pass
            if hasattr(xrt, "close"):
                try:
                    xrt.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
