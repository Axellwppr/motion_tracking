#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from retarget.fk import LocalKinematicsModel
from retarget.params import load_robot_axis_link_names
from retarget.viser_viewer import MJViserViewer
from retarget.xrobot_retarget import XRobotRetargetWorkerRuntime
from utils.robot_config import SUPPORTED_ROBOTS, load_teleop_robot_config


def _compute_fps_from_timestamps_ns(timestamps_ns: np.ndarray, fallback_fps: float) -> float:
    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    if ts.size < 2:
        return float(fallback_fps)
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float(fallback_fps)
    return float(1e9 / float(np.median(diffs)))


def _normalize_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _maybe_reindex_dof_pos(
    dof_pos: np.ndarray,
    *,
    names_in: tuple[str, ...],
    expected_joint_names: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, bool]:
    names_src = [_normalize_name(name) for name in names_in]
    expected_names = [_normalize_name(name) for name in expected_joint_names]
    if len(names_src) != len(expected_names):
        raise ValueError(
            f"joint_names length mismatch: got {len(names_src)}, expected {len(expected_names)}"
        )
    if names_src == expected_names:
        return dof_pos, False

    index_by_name: dict[str, int] = {}
    for idx, name in enumerate(names_src):
        if name in index_by_name:
            raise ValueError(f"duplicate joint name in source ordering: {name}")
        index_by_name[name] = idx

    expected_name_set = set(expected_names)
    missing = [name for name in expected_names if name not in index_by_name]
    extra = [name for name in names_src if name not in expected_name_set]
    if missing or extra:
        raise ValueError(
            f"joint_names do not match expected robot ordering. missing={missing}, extra={extra}"
        )

    reorder_index = [index_by_name[name] for name in expected_names]
    return dof_pos[:, reorder_index], True


def _rebuild_local_fk(
    *,
    target_robot: str,
    dof_pos: np.ndarray,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], bool]:
    kinematics_model = LocalKinematicsModel(target_robot)
    dof_pos_fk, reordered = _maybe_reindex_dof_pos(
        dof_pos,
        names_in=joint_names,
        expected_joint_names=kinematics_model.joint_names,
    )

    num_frames = int(dof_pos_fk.shape[0])
    fk_root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    fk_root_rot = np.zeros((num_frames, 4), dtype=np.float32)
    fk_root_rot[:, -1] = 1.0
    local_body_pos, local_body_rot = kinematics_model.forward_kinematics(
        fk_root_pos,
        fk_root_rot,
        np.asarray(dof_pos_fk, dtype=np.float32),
    )

    return (
        np.asarray(dof_pos_fk, dtype=np.float32),
        np.asarray(local_body_pos, dtype=np.float32),
        np.asarray(local_body_rot, dtype=np.float32),
        list(kinematics_model.joint_names),
        list(kinematics_model.body_names),
        reordered,
    )


def _load_input_stream(npz: Any) -> tuple[np.ndarray, np.ndarray, float]:
    poses_key = "poses"
    recv_key = "sample_ns" if "sample_ns" in npz.files else "recv_ns"
    fps_fallback = float(npz["fps"]) if "fps" in npz.files else 0.0
    if poses_key not in npz.files:
        raise KeyError(f"Input NPZ does not contain '{poses_key}'")
    if recv_key not in npz.files:
        raise KeyError(f"Input NPZ does not contain '{recv_key}'")

    poses = np.asarray(npz[poses_key], dtype=np.float32)
    recv_ns = np.asarray(npz[recv_key], dtype=np.int64)
    if poses.ndim != 3 or poses.shape[2] != 7:
        raise ValueError(f"{poses_key} must have shape (T, J, 7), got {poses.shape}")
    if recv_ns.ndim != 1 or recv_ns.shape[0] != poses.shape[0]:
        raise ValueError(f"{recv_key} must have shape ({poses.shape[0]},), got {recv_ns.shape}")

    fps = _compute_fps_from_timestamps_ns(recv_ns, fps_fallback if fps_fallback > 0 else 50.0)
    return poses, recv_ns, fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded XRobot raw NPZ, retarget it frame-by-frame at playback rate, "
            "visualize human link positions plus robot axes, and save the retargeted motion. "
            "With --zmq, run a request-driven replay server that advances one recorded frame per request."
        )
    )
    parser.add_argument("input_path", type=Path, help="Input NPZ recorded by record_xrobot_motion.py")
    parser.add_argument("--robot", choices=list(SUPPORTED_ROBOTS), default="l7")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=None,
        help="Override teleop config retarget.actual_human_height",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override teleop config retarget.max_iter",
    )
    parser.add_argument(
        "--disable-height-alignment",
        action="store_true",
        help="Disable foot-proxy root height alignment and keep the raw retarget root height.",
    )
    parser.add_argument(
        "--height-bootstrap-frames",
        type=int,
        default=None,
        help="Override teleop config height_alignment.bootstrap_frames.",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Replay as fast as possible instead of sleeping to input FPS.",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help=(
            "Playback speed multiplier for normal replay; 0.5 is half speed. "
            "Ignored with --no-rate-limit and --zmq."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of replayed frames, useful for quick testing.",
    )
    parser.add_argument(
        "--zmq",
        action="store_true",
        help="Serve the replay over the teleop ZMQ protocol and advance one frame per request.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="When used with --zmq, wrap to the first frame after the end instead of holding the last frame.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable the local viewer. Useful for headless smoke tests, especially with --zmq.",
    )
    return parser.parse_args()


def _default_output_path(input_path: Path, robot: str) -> Path:
    suffix = f"_replay_axes_{robot}.npz"
    return input_path.with_name(input_path.stem + suffix)


def _render_frame(
    viewer: MJViserViewer,
    qpos: np.ndarray,
    human_positions: np.ndarray | None,
    human_rotations_wxyz: np.ndarray | None,
    robot_axis_links: list[str],
) -> None:
    viewer.update_qpos(np.asarray(qpos, dtype=np.float32))
    viewer.draw_human_data("human", human_positions, human_rotations_wxyz)
    viewer.draw_bodies("robot", robot_axis_links)


def _save_output(
    *,
    output_path: Path,
    fps: float,
    qpos_arr: np.ndarray,
    config: Any,
) -> None:
    root_pos = qpos_arr[:, 0:3].astype(np.float32)
    root_quat_wxyz = qpos_arr[:, 3:7].astype(np.float32)
    root_rot_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos_arr[:, 7 : 7 + config.dof_count].astype(np.float32)
    dof_pos_out, local_body_pos, local_body_rot, joint_names_out, body_names_out, _ = _rebuild_local_fk(
        target_robot=str(config.robot_key),
        dof_pos=dof_pos,
        joint_names=config.dof_names,
    )

    np.savez_compressed(
        output_path,
        fps=np.float32(fps),
        root_pos=root_pos,
        root_rot=root_rot_xyzw,
        dof_pos=dof_pos_out,
        local_body_pos=local_body_pos.astype(np.float32),
        local_body_rot=local_body_rot.astype(np.float32),
        joint_names=np.asarray(joint_names_out),
        body_names=np.asarray(body_names_out),
    )


def _make_packet(
    *,
    seq: int,
    recv_ns: int,
    poses: np.ndarray,
    calibration_requested: bool = False,
) -> dict[str, Any]:
    return {
        "seq": int(seq),
        "recv_ns": int(recv_ns),
        "poses": np.asarray(poses, dtype=np.float32),
        "calibration_requested": bool(calibration_requested),
    }


def _calibration_request_from_buttons(
    *,
    button_name: str | None,
    buttons: dict[str, Any],
    prev_pressed: bool,
) -> tuple[bool, bool]:
    if button_name is None:
        return False, False
    pressed = bool(buttons.get(button_name, False))
    return bool(pressed and not prev_pressed), pressed


def _load_controller_stream(npz: Any, frame_count: int) -> dict[str, np.ndarray]:
    def _bool_array(name: str) -> np.ndarray:
        key = name
        if key in npz.files:
            arr = np.asarray(npz[key], dtype=np.bool_).reshape(-1)
            if arr.shape[0] >= frame_count:
                return arr[:frame_count]
        return np.zeros((frame_count,), dtype=np.bool_)

    def _axis_array(name: str) -> np.ndarray:
        key = name
        if key in npz.files:
            arr = np.asarray(npz[key], dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= frame_count:
                return arr[:frame_count]
        return np.zeros((frame_count, 2), dtype=np.float32)

    return {
        "left_key_one": _bool_array("left_key_one"),
        "left_key_two": _bool_array("left_key_two"),
        "left_axis_click": _bool_array("left_axis_click"),
        "left_index_trig": _bool_array("left_index_trig"),
        "left_grip": _bool_array("left_grip"),
        "left_axis": _axis_array("left_axis"),
        "right_key_one": _bool_array("right_key_one"),
        "right_key_two": _bool_array("right_key_two"),
        "right_axis_click": _bool_array("right_axis_click"),
        "right_index_trig": _bool_array("right_index_trig"),
        "right_grip": _bool_array("right_grip"),
        "right_axis": _axis_array("right_axis"),
    }


def _controller_buttons_at(controller_stream: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    return {
        "left_key_one": bool(controller_stream["left_key_one"][idx]),
        "left_key_two": bool(controller_stream["left_key_two"][idx]),
        "left_axis_click": bool(controller_stream["left_axis_click"][idx]),
        "left_index_trig": bool(controller_stream["left_index_trig"][idx]),
        "left_grip": bool(controller_stream["left_grip"][idx]),
        "left_axis": controller_stream["left_axis"][idx].astype(np.float32).tolist(),
        "right_key_one": bool(controller_stream["right_key_one"][idx]),
        "right_key_two": bool(controller_stream["right_key_two"][idx]),
        "right_axis_click": bool(controller_stream["right_axis_click"][idx]),
        "right_index_trig": bool(controller_stream["right_index_trig"][idx]),
        "right_grip": bool(controller_stream["right_grip"][idx]),
        "right_axis": controller_stream["right_axis"][idx].astype(np.float32).tolist(),
    }


class ReplayRetargetZMQServer:
    def __init__(
        self,
        *,
        config: Any,
        poses: np.ndarray,
        recv_ns: np.ndarray,
        fps: float,
        controller_stream: dict[str, np.ndarray],
        worker: XRobotRetargetWorkerRuntime,
        viewer: MJViserViewer | None,
        robot_axis_links: list[str],
        loop: bool,
        output_path: Path | None,
    ) -> None:
        self.config = config
        self.poses = np.asarray(poses, dtype=np.float32)
        self.recv_ns = np.asarray(recv_ns, dtype=np.int64)
        self.fps = float(fps)
        self.controller_stream = controller_stream
        self.worker = worker
        self.viewer = viewer
        self.robot_axis_links = robot_axis_links
        self.loop = bool(loop)
        self.output_path = output_path

        self.total_frames = int(self.poses.shape[0])
        self.next_frame_idx = 0
        self.served_count = 0
        self.req_count = 0
        self.reply_count = 0
        self.loop_count = 0
        self._warned_hold_last = False

        self._buttons_lock = threading.Lock()
        self._current_buttons = _controller_buttons_at(self.controller_stream, 0)
        self._prev_calibration_button_pressed = False
        self._served_qpos: list[np.ndarray] = []

        self.stop_event = threading.Event()
        self.control_thread: threading.Thread | None = None

        self.zmq_context = None
        self.req_sock = None
        self.rep_sock = None
        self.ctrl_sock = None

    def _current_control_payload(self) -> dict[str, Any]:
        with self._buttons_lock:
            buttons = dict(self._current_buttons)
        return {
            "t_ms": int(time.time() * 1000),
            "controller_buttons": buttons,
        }

    def _control_loop(self) -> None:
        import zmq

        period_s = 1.0 / float(self.config.ctrl_fps)
        while not self.stop_event.is_set():
            payload = self._current_control_payload()
            try:
                self.ctrl_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"[ReplayZMQ] control send failed: {exc}")
            self.stop_event.wait(timeout=period_s)

    def _select_frame_index(self) -> int:
        idx = int(self.next_frame_idx)
        if self.next_frame_idx + 1 < self.total_frames:
            self.next_frame_idx += 1
            return idx
        if self.loop:
            self.next_frame_idx = 0
            self.loop_count += 1
            print(f"[ReplayZMQ] loop back to frame 0 (loop_count={self.loop_count})")
            return idx
        if not self._warned_hold_last:
            print("[ReplayZMQ] reached the last frame; subsequent requests will keep serving the last frame")
            self._warned_hold_last = True
        self.next_frame_idx = max(self.total_frames - 1, 0)
        return idx

    def _drain_requests_blocking(self) -> dict[str, Any] | None:
        import zmq

        poller = zmq.Poller()
        poller.register(self.req_sock, zmq.POLLIN)
        while not self.stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if self.req_sock not in events:
                continue

            latest_req: dict[str, Any] | None = None
            any_start = False
            while True:
                try:
                    raw = self.req_sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                except Exception as exc:
                    print(f"[ReplayZMQ] request recv failed: {exc}")
                    break
                try:
                    req = json.loads(raw)
                except Exception:
                    print("[ReplayZMQ] bad request JSON")
                    continue
                if not isinstance(req, dict):
                    continue
                any_start = any_start or bool(req.get("start", False))
                latest_req = req

            if latest_req is None:
                continue
            latest_req["start"] = any_start
            return latest_req
        return None

    def _serve_one_reply(self, req: dict[str, Any]) -> None:
        import zmq

        frame_idx = self._select_frame_index()
        controller_buttons = _controller_buttons_at(self.controller_stream, frame_idx)
        calibration_requested, self._prev_calibration_button_pressed = _calibration_request_from_buttons(
            button_name=self.config.calibration_button,
            buttons=controller_buttons,
            prev_pressed=self._prev_calibration_button_pressed,
        )
        packet = _make_packet(
            seq=self.served_count + 1,
            recv_ns=int(self.recv_ns[frame_idx]),
            poses=self.poses[frame_idx],
            calibration_requested=calibration_requested,
        )
        result = self.worker.process_packet(packet)
        if result is None:
            raise RuntimeError(f"Retarget returned no output at replay frame {frame_idx}")

        qpos = np.asarray(result["qpos"], dtype=np.float32).reshape(-1)
        self._served_qpos.append(qpos[: self.config.qpos_size].copy())
        self.served_count += 1
        with self._buttons_lock:
            self._current_buttons = controller_buttons

        if self.viewer is not None:
            _render_frame(
                viewer=self.viewer,
                qpos=qpos,
                human_positions=result.get("human_positions"),
                human_rotations_wxyz=result.get("human_rotations_wxyz"),
                robot_axis_links=self.robot_axis_links,
            )

        qpos_frame = np.ascontiguousarray(qpos[: self.config.qpos_size].reshape(1, self.config.qpos_size))
        header = json.dumps(
            {
                "start": bool(req.get("start", False)),
                "num_frames": int(qpos_frame.shape[0]),
                "qpos_size": int(qpos_frame.shape[1]),
            }
        ).encode("utf-8")
        try:
            self.rep_sock.send_multipart([header, qpos_frame], flags=zmq.NOBLOCK, copy=False)
            self.reply_count += 1
        except zmq.Again:
            print("[ReplayZMQ] reply queue full, drop one reply")
        except Exception as exc:
            print(f"[ReplayZMQ] reply send failed: {exc}")

    def _save_if_requested(self) -> None:
        if self.output_path is None or len(self._served_qpos) == 0:
            return
        qpos_arr = np.asarray(self._served_qpos, dtype=np.float32).reshape(-1, self.config.qpos_size)
        _save_output(
            output_path=self.output_path,
            fps=self.fps,
            qpos_arr=qpos_arr,
            config=self.config,
        )
        print(f"[ReplayZMQ] saved={self.output_path} frames={qpos_arr.shape[0]} fps={self.fps:.3f}")

    def setup(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for --zmq replay mode.") from exc

        self.zmq_context = zmq.Context.instance()

        self.req_sock = self.zmq_context.socket(zmq.PULL)
        self.req_sock.setsockopt(zmq.LINGER, 0)
        self.req_sock.setsockopt(zmq.RCVHWM, 500)
        self.req_sock.bind(self.config.req_bind_addr)

        self.rep_sock = self.zmq_context.socket(zmq.PUSH)
        self.rep_sock.setsockopt(zmq.LINGER, 0)
        self.rep_sock.setsockopt(zmq.SNDHWM, 500)
        self.rep_sock.bind(self.config.rep_bind_addr)

        self.ctrl_sock = self.zmq_context.socket(zmq.PUSH)
        self.ctrl_sock.setsockopt(zmq.LINGER, 0)
        self.ctrl_sock.setsockopt(zmq.SNDHWM, 500)
        self.ctrl_sock.bind(self.config.ctrl_bind_addr)

        initial_start = bool(self._current_buttons.get("right_key_one", False))
        print("[ReplayZMQ] initialized")
        print(f"  input_frames: {self.total_frames}")
        print(f"  fps: {self.fps:.3f}")
        print(f"  robot_key: {self.config.robot_key}")
        print(f"  robot: {self.config.robot_key}")
        print(f"  req_bind_addr: {self.config.req_bind_addr}")
        print(f"  rep_bind_addr: {self.config.rep_bind_addr}")
        print(f"  ctrl_bind_addr: {self.config.ctrl_bind_addr}")
        print(f"  ctrl_fps: {self.config.ctrl_fps}")
        print(f"  loop: {self.loop}")
        print(f"  viewer: {self.viewer is not None}")
        print(f"  first_frame_start_button: {initial_start}")
        if not initial_start:
            print("[ReplayZMQ][Warning] first replay frame does not press right_key_one; VR client may not auto-start")

    def run(self) -> None:
        self.setup()
        self.control_thread = threading.Thread(
            target=self._control_loop,
            name="replay-zmq-control",
            daemon=True,
        )
        self.control_thread.start()

        try:
            while not self.stop_event.is_set():
                req = self._drain_requests_blocking()
                if req is None:
                    continue
                self.req_count += 1
                self._serve_one_reply(req)
                if self.req_count % 100 == 0:
                    print(
                        f"[ReplayZMQ] requests={self.req_count} replies={self.reply_count} "
                        f"served_frames={self.served_count} next_frame_idx={self.next_frame_idx}"
                    )
        except KeyboardInterrupt:
            print("[ReplayZMQ] KeyboardInterrupt, exiting.")
        finally:
            self.stop_event.set()
            if self.control_thread is not None:
                self.control_thread.join(timeout=1.0)
            for sock_name in ("req_sock", "rep_sock", "ctrl_sock"):
                sock = getattr(self, sock_name)
                if sock is not None:
                    try:
                        sock.close(0)
                    except Exception:
                        pass
            if self.viewer is not None:
                self.viewer.close()
            self._save_if_requested()


def main() -> None:
    args = parse_args()
    if float(args.playback_rate) <= 0.0:
        raise ValueError("--playback-rate must be > 0")
    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    config = load_teleop_robot_config(args.robot)
    actual_human_height = (
        float(args.actual_human_height)
        if args.actual_human_height is not None
        else float(config.actual_human_height)
    )
    max_iter = int(args.max_iter) if args.max_iter is not None else int(config.max_iter)
    height_bootstrap_frames = (
        int(args.height_bootstrap_frames)
        if args.height_bootstrap_frames is not None
        else int(config.height_alignment_bootstrap_frames)
    )

    npz = np.load(input_path, allow_pickle=True)
    poses, recv_ns, fps = _load_input_stream(npz)
    total_frames = int(poses.shape[0])
    if args.max_frames is not None:
        total_frames = min(total_frames, max(int(args.max_frames), 0))
    poses = np.asarray(poses[:total_frames], dtype=np.float32)
    recv_ns = np.asarray(recv_ns[:total_frames], dtype=np.int64)
    controller_stream = _load_controller_stream(npz, total_frames)

    use_viewer = not bool(args.no_viewer)
    worker = XRobotRetargetWorkerRuntime(
        {
            "qpos_size": int(config.qpos_size),
            "target_robot": str(config.robot_key),
            "actual_human_height": float(actual_human_height),
            "max_iter": int(max_iter),
            "send_human_motion": bool(use_viewer),
            "enable_height_alignment": bool(config.height_alignment_enabled) and not bool(args.disable_height_alignment),
            "height_alignment_xrobot_body_min_each_frame": bool(
                config.height_alignment_xrobot_body_min_each_frame
            ),
            "height_alignment_foot_body_names": list(config.height_alignment_foot_body_names),
            "height_alignment_target_z": float(config.height_alignment_target_z),
            "height_bootstrap_frames": int(height_bootstrap_frames),
        }
    )

    robot_axis_links = list(load_robot_axis_link_names(config.robot_key))
    viewer = None
    if use_viewer:
        viewer = MJViserViewer(str(config.robot_key))

    if args.zmq:
        output_path = args.output.expanduser().resolve() if args.output is not None else None
        server = ReplayRetargetZMQServer(
            config=config,
            poses=poses,
            recv_ns=recv_ns,
            fps=float(fps),
            controller_stream=controller_stream,
            worker=worker,
            viewer=viewer,
            robot_axis_links=robot_axis_links,
            loop=bool(args.loop),
            output_path=output_path,
        )
        server.run()
        return

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path=input_path, robot=args.robot)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qpos_list: list[np.ndarray] = []
    period_s = 1.0 / (float(fps) * float(args.playback_rate))
    height_alignment_mode = "off"
    if not args.disable_height_alignment and config.height_alignment_enabled:
        height_alignment_mode = (
            "xrobot_body_min_each_frame"
            if config.height_alignment_xrobot_body_min_each_frame
            else "startup_min"
        )

    print(
        f"[ReplayRetarget] input={input_path.name} frames={total_frames} "
        f"robot={config.robot_key} fps={fps:.3f} "
        f"playback_rate={args.playback_rate:.3f} max_iter={max_iter} "
        f"height_alignment={height_alignment_mode}"
    )

    try:
        next_tick = time.monotonic()
        prev_calibration_button_pressed = False
        for idx in range(total_frames):
            controller_buttons = _controller_buttons_at(controller_stream, idx)
            calibration_requested, prev_calibration_button_pressed = _calibration_request_from_buttons(
                button_name=config.calibration_button,
                buttons=controller_buttons,
                prev_pressed=prev_calibration_button_pressed,
            )
            packet = _make_packet(
                seq=idx + 1,
                recv_ns=int(recv_ns[idx]),
                poses=poses[idx],
                calibration_requested=calibration_requested,
            )
            result = worker.process_packet(packet)
            if result is None:
                raise RuntimeError(f"Retarget returned no output at frame {idx}")

            qpos = np.asarray(result["qpos"], dtype=np.float32).reshape(-1)
            qpos_list.append(qpos[: config.qpos_size].copy())

            if viewer is not None:
                _render_frame(
                    viewer=viewer,
                    qpos=qpos,
                    human_positions=result.get("human_positions"),
                    human_rotations_wxyz=result.get("human_rotations_wxyz"),
                    robot_axis_links=robot_axis_links,
                )

            if not args.no_rate_limit:
                next_tick += period_s
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()

            if (idx + 1) % 200 == 0 or idx + 1 == total_frames:
                print(f"[ReplayRetarget] processed {idx + 1}/{total_frames}")
    finally:
        if viewer is not None:
            viewer.close()

    qpos_arr = np.asarray(qpos_list, dtype=np.float32).reshape(-1, config.qpos_size)
    _save_output(
        output_path=output_path,
        fps=float(fps),
        qpos_arr=qpos_arr,
        config=config,
    )
    print(f"[ReplayRetarget] saved={output_path} frames={qpos_arr.shape[0]} fps={fps:.3f}")


if __name__ == "__main__":
    main()
