#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the saved fixed-rate XRobot stream from raw_* arrays using a selected timestamp source."
    )
    parser.add_argument("input_path", type=Path, help="Input NPZ recorded by record_xrobot_motion.py")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ path")
    parser.add_argument(
        "--time-source",
        choices=("motion", "recv"),
        default="motion",
        help="Timestamp source used to rebuild the saved fixed-rate stream.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target output FPS. Defaults to the input file's saved fps.",
    )
    return parser.parse_args()


def _default_output_path(input_path: Path, time_source: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{time_source}_resampled.npz")


def _compute_estimated_fps_ns(timestamps_ns: np.ndarray) -> np.float32:
    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    if ts.size < 2:
        return np.float32(0.0)
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return np.float32(0.0)
    return np.float32(1e9 / float(np.median(diffs)))


def _load_raw_motion_timestamp_ns(npz: Any) -> np.ndarray:
    if "raw_motion_timestamp_ns" not in npz.files:
        raise KeyError("Input NPZ does not contain 'raw_motion_timestamp_ns'")
    time_ns = np.asarray(npz["raw_motion_timestamp_ns"], dtype=np.int64).copy()
    raw_recv_ns = np.asarray(npz["raw_recv_ns"], dtype=np.int64)
    invalid = time_ns <= 0
    time_ns[invalid] = raw_recv_ns[invalid]
    return time_ns


def _nearest_indices(src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if src_t.size == 1:
        return np.zeros(dst_t.shape[0], dtype=np.int64)
    right = np.searchsorted(src_t, dst_t, side="left")
    right = np.clip(right, 0, src_t.size - 1)
    left = np.clip(right - 1, 0, src_t.size - 1)
    choose_right = np.abs(src_t[right] - dst_t) < np.abs(dst_t - src_t[left])
    return np.where(choose_right, right, left).astype(np.int64)


def _resample_positions(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if src_t.size == 1:
        return np.broadcast_to(values[:1], (dst_t.shape[0],) + values.shape[1:]).copy()
    out = np.empty((dst_t.shape[0],) + values.shape[1:], dtype=np.float32)
    joint_count = values.shape[1]
    for joint_idx in range(joint_count):
        for dim in range(3):
            out[:, joint_idx, dim] = np.interp(
                dst_t,
                src_t,
                values[:, joint_idx, dim].astype(np.float64),
            ).astype(np.float32)
    return out


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


def _resample_segment(
    npz: Any,
    *,
    time_source: str,
    target_fps: float,
) -> dict[str, np.ndarray]:
    raw_poses = np.asarray(npz["raw_poses"], dtype=np.float32)
    raw_recv_ns = np.asarray(npz["raw_recv_ns"], dtype=np.int64)
    raw_motion_timestamp_ns = _load_raw_motion_timestamp_ns(npz)

    if time_source == "recv":
        time_ns = raw_recv_ns.copy()
    else:
        time_ns = raw_motion_timestamp_ns.copy()

    keep = np.ones(time_ns.shape[0], dtype=bool)
    if time_ns.shape[0] > 1:
        keep[1:] = time_ns[1:] > time_ns[:-1]

    src_time_ns = time_ns[keep]
    poses_kept = raw_poses[keep]
    src_rel_s = (src_time_ns - src_time_ns[0]).astype(np.float64) / 1e9
    duration_s = float(src_rel_s[-1]) if src_rel_s.size > 0 else 0.0
    sample_count = max(1, int(np.floor(duration_s * target_fps + 1e-9)) + 1)
    target_rel_s = np.arange(sample_count, dtype=np.float64) / target_fps
    sample_ns = src_time_ns[0] + np.rint(target_rel_s * 1e9).astype(np.int64)

    out_poses = np.empty((sample_count, poses_kept.shape[1], 7), dtype=np.float32)
    out_poses[..., :3] = _resample_positions(poses_kept[..., :3], src_rel_s, target_rel_s)
    out_poses[..., 3:7] = _resample_quaternions(poses_kept[..., 3:7], src_rel_s, target_rel_s)

    nearest_idx = _nearest_indices(src_rel_s, target_rel_s)

    def _pick(name: str) -> np.ndarray:
        return np.asarray(npz[name])[keep][nearest_idx]

    return {
        "poses": out_poses,
        "sample_ns": sample_ns,
        "recv_ns": raw_recv_ns[keep][nearest_idx],
        "motion_timestamp_ns": raw_motion_timestamp_ns[keep][nearest_idx],
        "left_key_one": _pick("raw_left_key_one"),
        "left_key_two": _pick("raw_left_key_two"),
        "left_axis_click": _pick("raw_left_axis_click"),
        "left_index_trig": _pick("raw_left_index_trig"),
        "left_grip": _pick("raw_left_grip"),
        "left_axis": _pick("raw_left_axis"),
        "right_key_one": _pick("raw_right_key_one"),
        "right_key_two": _pick("raw_right_key_two"),
        "right_axis_click": _pick("raw_right_axis_click"),
        "right_index_trig": _pick("raw_right_index_trig"),
        "right_grip": _pick("raw_right_grip"),
        "right_axis": _pick("raw_right_axis"),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path, args.time_source)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npz = np.load(input_path, allow_pickle=True)
    target_fps = float(args.target_fps) if args.target_fps is not None else float(npz["fps"])
    if target_fps <= 0.0:
        raise ValueError("target_fps must be > 0")

    drop_keys = {
        "fps_motion",
        "motion_timestamp_ns",
        "raw_motion_timestamp_ns",
    }
    payload = {key: npz[key] for key in npz.files if key not in drop_keys}
    resampled = _resample_segment(npz, time_source=args.time_source, target_fps=target_fps)
    raw_motion_timestamp_ns = _load_raw_motion_timestamp_ns(npz)
    payload.update(
        fps=np.float32(target_fps),
        fps_recv=_compute_estimated_fps_ns(np.asarray(npz["raw_recv_ns"], dtype=np.int64)),
        fps_motion=_compute_estimated_fps_ns(raw_motion_timestamp_ns),
        poses=resampled["poses"],
        sample_ns=resampled["sample_ns"],
        recv_ns=resampled["recv_ns"],
        motion_timestamp_ns=resampled["motion_timestamp_ns"],
        raw_motion_timestamp_ns=raw_motion_timestamp_ns,
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
    )

    np.savez_compressed(output_path, **payload)
    print(
        f"[ResampleRaw] saved={output_path} source={args.time_source} "
        f"frames={payload['poses'].shape[0]} fps={target_fps:.3f}"
    )


if __name__ == "__main__":
    main()
