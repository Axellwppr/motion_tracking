#!/usr/bin/env python3
"""
Low-latency PICO/XRobot teleop bridge for sim2real.

Architecture:
1. XR callback thread stores the latest VR snapshot in a shared-memory latest-frame slot.
2. A retarget worker process waits for new VR data and only retargets the latest snapshot.
3. The retarget worker writes fixed-shape results into a shared-memory ring buffer.
4. A request thread serves the newest ZMQ request using time-based interpolation over that ring.
5. A control thread publishes controller buttons at a fixed rate.
"""

import argparse
import json
import multiprocessing as mp
import threading
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from retarget.params import XR_BODY_JOINT_NAMES, load_human_body_names, load_robot_axis_link_names
from retarget.viser_viewer import MJViserViewer
from retarget.xrobot_retarget import XRobotRetargetWorkerRuntime
from utils.buffer import SharedLatestVrFrame, SharedRetargetFrameRingBuffer
from utils.helper import (
    default_controller_buttons,
    parse_xrobot_motion_snapshot,
)
from utils.robot_config import SUPPORTED_ROBOTS, load_teleop_robot_config
from utils.xrobot_sdk import load_xrobotoolkit_sdk

xrt = None
XR_BODY_JOINT_COUNT = len(XR_BODY_JOINT_NAMES)
def _retarget_worker_main(
    shared_vr_frame: SharedLatestVrFrame,
    worker_stop_event: Any,
    worker_ready_event: Any,
    retarget_buffer: SharedRetargetFrameRingBuffer,
    worker_config: Dict[str, Any],
) -> None:
    try:
        runtime = XRobotRetargetWorkerRuntime(worker_config)
    except Exception:
        print("[Worker] retarget worker init failed")
        traceback.print_exc()
        return

    worker_ready_event.set()
    last_processed_seq = 0
    last_calibration_request_count = 0
    retarget_buffer_window_ns = int(worker_config["retarget_buffer_window_ns"])
    latest_pose_buffer = np.empty(shared_vr_frame.pose_shape, dtype=np.float32)
    while not worker_stop_event.is_set():
        if not shared_vr_frame.updated_event.wait(timeout=0.1):
            continue

        while not worker_stop_event.is_set():
            latest_packet = shared_vr_frame.read_latest_into(
                latest_pose_buffer,
                last_seq=last_processed_seq,
            )
            if latest_packet is None:
                break

            seq, recv_ns, calibration_request_count = latest_packet
            calibration_requested = int(calibration_request_count) != int(last_calibration_request_count)
            last_calibration_request_count = int(calibration_request_count)
            dropped_before_process = max(int(seq) - int(last_processed_seq) - 1, 0)
            packet = {
                "seq": int(seq),
                "recv_ns": int(recv_ns),
                "poses": latest_pose_buffer,
                "calibration_requested": bool(calibration_requested),
            }

            try:
                result = runtime.process_packet(packet)
                last_processed_seq = int(seq)
                if result is None:
                    continue
                retarget_buffer.append(
                    recv_ns=int(result["recv_ns"]),
                    seq=int(result["seq"]),
                    qpos=result["qpos"],
                    human_positions=result.get("human_positions"),
                    human_rotations_wxyz=result.get("human_rotations_wxyz"),
                    dropped_before_process=int(dropped_before_process),
                    window_ns=int(retarget_buffer_window_ns),
                )
            except Exception:
                print("[Worker] retarget worker runtime error")
                traceback.print_exc()
                return


class LowLatencyTeleopPoseZMQServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = load_teleop_robot_config(args.robot)
        self.robot = self.config.robot_key
        self.vis_fps = int(self.config.vis_fps)
        self.ctrl_fps = int(self.config.ctrl_fps)
        self.lookback_ns = int(float(self.config.lookback_ms) * 1e6)
        self.retarget_buffer_window_ns = int(float(self.config.retarget_buffer_window_s) * 1e9)
        self.log_interval_s = float(self.config.log_interval_s)
        self.robot_axis_links = list(load_robot_axis_link_names(self.robot))
        self.human_body_names = list(load_human_body_names(self.robot))

        self.viewer = None
        self.max_iter = int(self.config.max_iter)
        self.root_dof = 7
        self.qpos_size = self.config.qpos_size
        self.dof_count = self.config.dof_count

        self.zmq_context = None
        self.req_sock = None
        self.rep_sock = None
        self.ctrl_sock = None
        self.mp_ctx = mp.get_context("spawn")

        self.last_controller_buttons: Dict[str, Any] = default_controller_buttons()
        self.calibration_button_name = self.config.calibration_button
        self.prev_calibration_button_pressed = False

        self.latest_vr_lock = threading.Lock()
        self.latest_raw_recv_ns: int = 0
        self.latest_vr_motion_timestamp_ns: Optional[int] = None

        retarget_buffer_capacity = max(
            64,
            int(np.ceil((self.retarget_buffer_window_ns / 1e9) * 512.0)) + 8,
        )
        self.retarget_buffer = SharedRetargetFrameRingBuffer(
            self.mp_ctx,
            qpos_size=self.qpos_size,
            capacity=retarget_buffer_capacity,
            human_body_count=len(self.human_body_names),
            store_human=bool(self.config.visualize),
        )

        self.stop_event = threading.Event()
        self.req_count = 0
        self.reply_count = 0
        self.reply_drop_count = 0
        self.req_merged_total = 0
        self.fallback_count = 0
        self.req_interval_lock = threading.Lock()
        self._last_req_arrival_ns: Optional[int] = None
        self._req_interval_n = 0
        self._req_interval_sum_ms = 0.0
        self._req_interval_min_ms: Optional[float] = None
        self._req_interval_max_ms: Optional[float] = None

        self.callback_count = 0

        self.request_thread = None
        self.control_thread = None
        self.stats_thread = None
        self.visualization_thread = None

        self.shared_vr_frame = SharedLatestVrFrame(self.mp_ctx, joint_count=XR_BODY_JOINT_COUNT)
        self.worker_stop_event = self.mp_ctx.Event()
        self.worker_ready_event = self.mp_ctx.Event()
        self.retarget_process = None

    def _is_calibration_button_pressed(self, buttons: Dict[str, Any]) -> bool:
        if self.calibration_button_name is None:
            return False
        return bool(buttons.get(self.calibration_button_name, False))

    def _record_request_arrival(self, recv_ns: int) -> None:
        with self.req_interval_lock:
            if self._last_req_arrival_ns is not None:
                dt_ms = (int(recv_ns) - int(self._last_req_arrival_ns)) / 1e6
                if dt_ms >= 0.0:
                    self._req_interval_n += 1
                    self._req_interval_sum_ms += dt_ms
                    if self._req_interval_min_ms is None or dt_ms < self._req_interval_min_ms:
                        self._req_interval_min_ms = dt_ms
                    if self._req_interval_max_ms is None or dt_ms > self._req_interval_max_ms:
                        self._req_interval_max_ms = dt_ms
            self._last_req_arrival_ns = int(recv_ns)

    def _take_request_interval_stats(self) -> tuple[Optional[float], Optional[float], Optional[float], int]:
        with self.req_interval_lock:
            n = int(self._req_interval_n)
            if n > 0:
                mean_ms = self._req_interval_sum_ms / float(n)
                min_ms = self._req_interval_min_ms
                max_ms = self._req_interval_max_ms
            else:
                mean_ms = None
                min_ms = None
                max_ms = None
            self._req_interval_n = 0
            self._req_interval_sum_ms = 0.0
            self._req_interval_min_ms = None
            self._req_interval_max_ms = None
        return mean_ms, min_ms, max_ms, n

    def _on_vr_frame(self, snapshot: dict) -> None:
        parsed_snapshot = parse_xrobot_motion_snapshot(snapshot, joint_count=XR_BODY_JOINT_COUNT)
        if parsed_snapshot is None:
            return

        recv_ns = time.monotonic_ns()
        calibration_pressed = self._is_calibration_button_pressed(parsed_snapshot.controller_buttons)

        should_publish = False
        calibration_requested = False
        with self.latest_vr_lock:
            self.last_controller_buttons = parsed_snapshot.controller_buttons
            self.callback_count += 1
            calibration_requested = bool(calibration_pressed and not self.prev_calibration_button_pressed)
            self.prev_calibration_button_pressed = bool(calibration_pressed)
            if self.latest_vr_motion_timestamp_ns != parsed_snapshot.motion_timestamp_ns:
                self.latest_raw_recv_ns = recv_ns
                self.latest_vr_motion_timestamp_ns = parsed_snapshot.motion_timestamp_ns
                should_publish = True
            if calibration_requested:
                should_publish = True
        if not should_publish:
            return
        try:
            self.shared_vr_frame.publish(
                parsed_snapshot.poses,
                recv_ns=recv_ns,
                calibration_requested=calibration_requested,
            )
        except Exception:
            return

    def _get_latest_frame_ages_ms(self) -> tuple[Optional[float], Optional[float]]:
        now_ns = time.monotonic_ns()
        with self.latest_vr_lock:
            latest_raw_recv_ns = int(self.latest_raw_recv_ns) if self.latest_raw_recv_ns > 0 else None

        latest_retarget_recv_ns = self.retarget_buffer.latest_recv_ns()

        raw_motion_age_ms = None
        if latest_raw_recv_ns is not None:
            raw_motion_age_ms = round((now_ns - latest_raw_recv_ns) / 1e6, 3)

        retarget_age_ms = None
        if latest_retarget_recv_ns is not None:
            retarget_age_ms = round((now_ns - latest_retarget_recv_ns) / 1e6, 3)

        return retarget_age_ms, raw_motion_age_ms

    def _drain_requests_blocking(self, poller: Any) -> tuple[Optional[Dict[str, Any]], Optional[int], int]:
        import zmq

        while not self.stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if self.req_sock not in events:
                continue

            latest_req: Optional[Dict[str, Any]] = None
            req_recv_ns: Optional[int] = None
            merged_reqs = 0
            any_start = False

            while True:
                try:
                    raw = self.req_sock.recv_string(flags=zmq.NOBLOCK)
                    req_recv_ns = time.monotonic_ns()
                    self._record_request_arrival(req_recv_ns)
                    req = json.loads(raw)
                except zmq.Again:
                    break
                except zmq.ZMQError as exc:
                    print(f"[Warning] request recv failed: {exc}")
                    break
                except json.JSONDecodeError:
                    print("[Warning] bad request JSON")
                    continue
                if not isinstance(req, dict):
                    continue

                merged_reqs += 1
                any_start = any_start or bool(req.get("start", False))
                latest_req = req

            if latest_req is None:
                continue

            latest_req["start"] = any_start
            return latest_req, req_recv_ns, merged_reqs

        return None, None, 0

    def _request_loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self.req_sock, zmq.POLLIN)

        while not self.stop_event.is_set():
            req, req_recv_ns, merged_reqs = self._drain_requests_blocking(poller)
            if req is None or req_recv_ns is None:
                continue

            self.req_count += 1
            self.req_merged_total += int(merged_reqs)

            sample = self.retarget_buffer.sample(
                target_ns=req_recv_ns - self.lookback_ns,
            )
            if sample.qpos is None:
                continue
            if sample.used_fallback:
                self.fallback_count += 1
            qpos_frame = np.ascontiguousarray(np.asarray(sample.qpos, dtype=np.float32).reshape(1, -1))
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
                self.reply_drop_count += 1
                print("[Warning] reply queue full, drop one reply")
            except Exception as exc:
                print(f"[Warning] reply send failed: {exc}")

    def _stats_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.stop_event.wait(timeout=self.log_interval_s):
                break

            with self.latest_vr_lock:
                callback_count = int(self.callback_count)
            retarget_count, raw_motion_drop_count = self.retarget_buffer.counts_snapshot()
            req_count = int(self.req_count)
            reply_count = int(self.reply_count)
            reply_drop_count = int(self.reply_drop_count)
            req_merged_total = int(self.req_merged_total)
            fallback_count = int(self.fallback_count)
            retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms()
            req_dt_mean_ms, req_dt_min_ms, req_dt_max_ms, req_dt_n = self._take_request_interval_stats()
            fmt_ms = lambda v: "None" if v is None else f"{float(v):.3f}"
            print(
                "[Stats] "
                f"req={req_count}, rep={reply_count}, rep_drop={reply_drop_count}, "
                f"req_merged_total={req_merged_total}, "
                f"req_dt_ms_mean={fmt_ms(req_dt_mean_ms)}, req_dt_ms_min={fmt_ms(req_dt_min_ms)}, "
                f"req_dt_ms_max={fmt_ms(req_dt_max_ms)}, req_dt_n={req_dt_n}, "
                f"fallback={fallback_count}, raw_drop={raw_motion_drop_count}, "
                f"cb={callback_count}, retarget={retarget_count}, "
                f"retarget_age_ms={retarget_age_ms}, "
                f"raw_motion_age_ms={raw_motion_age_ms}"
            )

    def _control_loop(self) -> None:
        import zmq

        period_s = 1.0 / float(self.ctrl_fps)
        while not self.stop_event.is_set():
            with self.latest_vr_lock:
                buttons = dict(self.last_controller_buttons)

            payload = {
                "t_ms": int(time.time() * 1000),
                "controller_buttons": buttons,
            }
            try:
                self.ctrl_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"[Warning] control send failed: {exc}")

            self.stop_event.wait(timeout=period_s)

    def _visualization_loop(self) -> None:
        if self.viewer is None:
            return

        period_s = 1.0 / float(self.vis_fps)
        while not self.stop_event.is_set():
            latest_frame = self.retarget_buffer.latest_frame()
            qpos = latest_frame.qpos
            human_positions = latest_frame.human_positions
            human_rotations_wxyz = latest_frame.human_rotations_wxyz

            if qpos is not None:
                try:
                    self.viewer.update_qpos(qpos)
                    self.viewer.draw_human_data("human", human_positions, human_rotations_wxyz)
                    self.viewer.draw_bodies("robot", self.robot_axis_links)
                except Exception as exc:
                    print(f"[Warning] visualization failed, disabling viewer: {exc}")
                    self.viewer.close()
                    self.viewer = None
                    return

            self.stop_event.wait(timeout=period_s)

    def setup(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for the teleop ZMQ server.") from exc

        if self.config.visualize:
            self.viewer = MJViserViewer(str(self.robot))

        worker_config = {
            "actual_human_height": float(self.config.actual_human_height),
            "max_iter": int(self.max_iter),
            "target_robot": self.config.robot_key,
            "qpos_size": self.qpos_size,
            "send_human_motion": bool(self.viewer is not None),
            "enable_height_alignment": bool(self.config.height_alignment_enabled),
            "height_alignment_xrobot_body_min_each_frame": bool(
                self.config.height_alignment_xrobot_body_min_each_frame
            ),
            "height_alignment_foot_body_names": list(self.config.height_alignment_foot_body_names),
            "height_alignment_target_z": float(self.config.height_alignment_target_z),
            "height_bootstrap_frames": int(self.config.height_alignment_bootstrap_frames),
            "retarget_buffer_window_ns": int(self.retarget_buffer_window_ns),
        }
        self.retarget_process = self.mp_ctx.Process(
            target=_retarget_worker_main,
            args=(
                self.shared_vr_frame,
                self.worker_stop_event,
                self.worker_ready_event,
                self.retarget_buffer,
                worker_config,
            ),
            name="teleop-retarget-worker",
            daemon=True,
        )
        self.retarget_process.start()
        ready_deadline = time.monotonic() + 10.0
        while not self.worker_ready_event.wait(timeout=0.05):
            if self.retarget_process is not None and not self.retarget_process.is_alive():
                raise RuntimeError("Retarget worker exited before becoming ready.")
            if time.monotonic() >= ready_deadline:
                raise RuntimeError("Retarget worker did not become ready within 10 seconds.")

        xrt.init()
        xrt.register_frame_callback(self._on_vr_frame)

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

        print("Low-latency teleop ZMQ pose server initialized")
        print(f"  robot_key: {self.config.robot_key}")
        print(f"  robot: {self.config.robot_key}")
        print(f"  req_bind_addr: {self.config.req_bind_addr}")
        print(f"  rep_bind_addr: {self.config.rep_bind_addr}")
        print(f"  ctrl_bind_addr: {self.config.ctrl_bind_addr}")
        print(f"  ctrl_fps: {self.ctrl_fps}")
        print(f"  max_iter: {self.max_iter}")
        print(f"  qpos_size: {self.qpos_size}")
        print(f"  lookback_ms: {self.lookback_ns / 1e6:.3f}")
        print(f"  retarget_buffer_window_s: {self.retarget_buffer_window_ns / 1e9:.3f}")
        if self.config.height_alignment_enabled:
            mode = (
                "xrobot_body_min_each_frame"
                if self.config.height_alignment_xrobot_body_min_each_frame
                else "foot-proxy startup_min"
            )
            print(
                f"  height_alignment: {mode} "
                f"foot_bodies={self.config.height_alignment_foot_body_names} "
                f"bootstrap_frames={self.config.height_alignment_bootstrap_frames} "
                f"target_z={self.config.height_alignment_target_z:.3f}"
            )
        else:
            print("  height_alignment: disabled")
        print(f"  calibration_button: {self.config.calibration_button}")
        print(f"  log_interval_s: {self.log_interval_s:.3f}")
        print(f"  visualize: {self.config.visualize}")
        if self.viewer is not None:
            print("  viewer_url: http://localhost:8080")
        print(f"  retarget_worker_pid: {self.retarget_process.pid if self.retarget_process else None}")

    def run(self) -> None:
        self.setup()

        self.request_thread = threading.Thread(
            target=self._request_loop,
            name="teleop-request",
            daemon=True,
        )
        self.control_thread = threading.Thread(
            target=self._control_loop,
            name="teleop-control",
            daemon=True,
        )
        if self.viewer is not None:
            self.visualization_thread = threading.Thread(
                target=self._visualization_loop,
                name="teleop-visualization",
                daemon=True,
            )
        if self.log_interval_s > 0.0:
            self.stats_thread = threading.Thread(
                target=self._stats_loop,
                name="teleop-stats",
                daemon=True,
            )

        self.request_thread.start()
        self.control_thread.start()
        if self.visualization_thread is not None:
            self.visualization_thread.start()
        if self.stats_thread is not None:
            self.stats_thread.start()

        try:
            while not self.stop_event.is_set():
                if self.retarget_process is not None and not self.retarget_process.is_alive():
                    print("[Warning] retarget worker exited unexpectedly")
                    self.stop_event.set()
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting low-latency teleop ZMQ pose server.")
        finally:
            self.stop_event.set()
            self.worker_stop_event.set()
            self.shared_vr_frame.updated_event.set()
            try:
                xrt.clear_frame_callback()
            except Exception:
                pass

            for thread in (
                self.request_thread,
                self.control_thread,
                self.visualization_thread,
                self.stats_thread,
            ):
                if thread is not None:
                    thread.join(timeout=1.0)

            if self.retarget_process is not None:
                self.retarget_process.join(timeout=2.0)
                if self.retarget_process.is_alive():
                    self.retarget_process.terminate()
                    self.retarget_process.join(timeout=1.0)

            if self.viewer is not None:
                self.viewer.close()
            if self.req_sock is not None:
                self.req_sock.close(0)
            if self.rep_sock is not None:
                self.rep_sock.close(0)
            if self.ctrl_sock is not None:
                self.ctrl_sock.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency ZMQ teleop pose server")
    parser.add_argument(
        "--robot",
        choices=list(SUPPORTED_ROBOTS),
        default="g1",
        help="Robot key for config/<robot>/retarget/teleop.yaml",
    )
    return parser.parse_args()


def main() -> None:
    global xrt
    args = parse_args()
    xrt = load_xrobotoolkit_sdk()
    server = LowLatencyTeleopPoseZMQServer(args)
    server.run()


if __name__ == "__main__":
    main()
