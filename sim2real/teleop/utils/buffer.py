from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .helper import interpolate_qpos


@dataclass
class RetargetBufferSample:
    qpos: Optional[np.ndarray]
    used_fallback: bool
    info: Dict[str, Any]


@dataclass
class RetargetLatestFrame:
    recv_ns: Optional[int]
    seq: Optional[int]
    qpos: Optional[np.ndarray]
    human_positions: Optional[np.ndarray]
    human_rotations_wxyz: Optional[np.ndarray]


class SharedLatestVrFrame:
    def __init__(self, mp_ctx: Any, *, joint_count: int, pose_width: int = 7):
        self.pose_shape = (int(joint_count), int(pose_width))
        self.lock = mp_ctx.Lock()
        self.updated_event = mp_ctx.Event()
        self.seq = mp_ctx.Value("Q", 0, lock=False)
        self.recv_ns = mp_ctx.Value("Q", 0, lock=False)
        self.calibration_request_count = mp_ctx.Value("Q", 0, lock=False)
        self.pose_buffer = mp_ctx.Array("f", int(np.prod(self.pose_shape)), lock=False)
        self._pose_view: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pose_view"] = None
        return state

    def _poses_view(self) -> np.ndarray:
        if self._pose_view is None:
            self._pose_view = np.frombuffer(self.pose_buffer, dtype=np.float32).reshape(self.pose_shape)
        return self._pose_view

    def publish(self, poses: Any, *, recv_ns: int, calibration_requested: bool = False) -> int:
        pose_arr = np.asarray(poses, dtype=np.float32)
        pose_arr = pose_arr[: self.pose_shape[0], : self.pose_shape[1]]
        if pose_arr.shape != self.pose_shape:
            raise ValueError(f"XR poses must have shape {self.pose_shape}, got {pose_arr.shape}")

        with self.lock:
            np.copyto(self._poses_view(), pose_arr, casting="same_kind")
            next_seq = int(self.seq.value) + 1
            self.recv_ns.value = int(recv_ns)
            if calibration_requested:
                self.calibration_request_count.value = int(self.calibration_request_count.value) + 1
            self.seq.value = next_seq
            self.updated_event.set()
            return next_seq

    def read_latest_into(self, out: np.ndarray, *, last_seq: int) -> tuple[int, int, int] | None:
        out_arr = np.asarray(out, dtype=np.float32)
        if out_arr.shape != self.pose_shape:
            raise ValueError(f"Output pose array must have shape {self.pose_shape}, got {out_arr.shape}")

        with self.lock:
            current_seq = int(self.seq.value)
            if current_seq == int(last_seq):
                self.updated_event.clear()
                return None
            np.copyto(out_arr, self._poses_view(), casting="same_kind")
            return current_seq, int(self.recv_ns.value), int(self.calibration_request_count.value)


class SharedRetargetFrameRingBuffer:
    def __init__(
        self,
        mp_ctx: Any,
        *,
        qpos_size: int,
        capacity: int,
        human_body_count: int = 0,
        store_human: bool = False,
    ):
        self.qpos_size = int(qpos_size)
        self.capacity = max(int(capacity), 2)
        self.human_body_count = max(int(human_body_count), 0)
        self.store_human = bool(store_human and self.human_body_count > 0)
        self.lock = mp_ctx.Lock()
        self.head = mp_ctx.Value("I", 0, lock=False)
        self.size = mp_ctx.Value("I", 0, lock=False)
        self.retarget_count = mp_ctx.Value("Q", 0, lock=False)
        self.raw_drop_count = mp_ctx.Value("Q", 0, lock=False)
        self.recv_ns_buffer = mp_ctx.Array("q", self.capacity, lock=False)
        self.seq_buffer = mp_ctx.Array("Q", self.capacity, lock=False)
        self.qpos_buffer = mp_ctx.Array("f", self.capacity * self.qpos_size, lock=False)
        self._recv_ns_view: np.ndarray | None = None
        self._seq_view: np.ndarray | None = None
        self._qpos_view: np.ndarray | None = None

        self.human_positions_buffer = None
        self.human_rotations_buffer = None
        self._human_positions_view: np.ndarray | None = None
        self._human_rotations_view: np.ndarray | None = None
        if self.store_human:
            self.human_positions_buffer = mp_ctx.Array("f", self.capacity * self.human_body_count * 3, lock=False)
            self.human_rotations_buffer = mp_ctx.Array("f", self.capacity * self.human_body_count * 4, lock=False)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        for key in (
            "_recv_ns_view",
            "_seq_view",
            "_qpos_view",
            "_human_positions_view",
            "_human_rotations_view",
        ):
            state[key] = None
        return state

    def _recv_ns_array(self) -> np.ndarray:
        if self._recv_ns_view is None:
            self._recv_ns_view = np.frombuffer(self.recv_ns_buffer, dtype=np.int64)
        return self._recv_ns_view

    def _seq_array(self) -> np.ndarray:
        if self._seq_view is None:
            self._seq_view = np.frombuffer(self.seq_buffer, dtype=np.uint64)
        return self._seq_view

    def _qpos_array(self) -> np.ndarray:
        if self._qpos_view is None:
            self._qpos_view = np.frombuffer(self.qpos_buffer, dtype=np.float32).reshape(self.capacity, self.qpos_size)
        return self._qpos_view

    def _human_positions_array(self) -> np.ndarray | None:
        if not self.store_human or self.human_positions_buffer is None:
            return None
        if self._human_positions_view is None:
            self._human_positions_view = np.frombuffer(self.human_positions_buffer, dtype=np.float32).reshape(
                self.capacity,
                self.human_body_count,
                3,
            )
        return self._human_positions_view

    def _human_rotations_array(self) -> np.ndarray | None:
        if not self.store_human or self.human_rotations_buffer is None:
            return None
        if self._human_rotations_view is None:
            self._human_rotations_view = np.frombuffer(self.human_rotations_buffer, dtype=np.float32).reshape(
                self.capacity,
                self.human_body_count,
                4,
            )
        return self._human_rotations_view

    @staticmethod
    def _physical_index(head: int, logical_idx: int, capacity: int) -> int:
        return (int(head) + int(logical_idx)) % int(capacity)

    def latest_recv_ns(self) -> int | None:
        with self.lock:
            size = int(self.size.value)
            if size == 0:
                return None
            head = int(self.head.value)
            latest_idx = self._physical_index(head, size - 1, self.capacity)
            return int(self._recv_ns_array()[latest_idx])

    def counts_snapshot(self) -> tuple[int, int]:
        with self.lock:
            return int(self.retarget_count.value), int(self.raw_drop_count.value)

    def latest_frame(self) -> RetargetLatestFrame:
        with self.lock:
            size = int(self.size.value)
            if size == 0:
                return RetargetLatestFrame(
                    recv_ns=None,
                    seq=None,
                    qpos=None,
                    human_positions=None,
                    human_rotations_wxyz=None,
                )

            head = int(self.head.value)
            latest_idx = self._physical_index(head, size - 1, self.capacity)
            qpos = self._qpos_array()[latest_idx].copy()
            human_positions = None
            human_rotations_wxyz = None
            if self.store_human:
                human_positions = self._human_positions_array()[latest_idx].copy()
                human_rotations_wxyz = self._human_rotations_array()[latest_idx].copy()
            return RetargetLatestFrame(
                recv_ns=int(self._recv_ns_array()[latest_idx]),
                seq=int(self._seq_array()[latest_idx]),
                qpos=qpos,
                human_positions=human_positions,
                human_rotations_wxyz=human_rotations_wxyz,
            )

    def append(
        self,
        *,
        recv_ns: int,
        seq: int,
        qpos: np.ndarray,
        human_positions: np.ndarray | None,
        human_rotations_wxyz: np.ndarray | None,
        dropped_before_process: int,
        window_ns: int,
    ) -> None:
        qpos_arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos_arr.shape[0] != self.qpos_size:
            raise ValueError(f"qpos must have shape ({self.qpos_size},), got {qpos_arr.shape}")
        human_positions_arr = None
        human_rotations_arr = None
        if self.store_human:
            if human_positions is None or human_rotations_wxyz is None:
                raise ValueError("human motion arrays must be present when store_human=True")
            human_positions_arr = np.asarray(human_positions, dtype=np.float32)
            human_rotations_arr = np.asarray(human_rotations_wxyz, dtype=np.float32)
            expected_pos_shape = (self.human_body_count, 3)
            expected_rot_shape = (self.human_body_count, 4)
            if human_positions_arr.shape != expected_pos_shape:
                raise ValueError(f"human_positions must have shape {expected_pos_shape}, got {human_positions_arr.shape}")
            if human_rotations_arr.shape != expected_rot_shape:
                raise ValueError(
                    f"human_rotations_wxyz must have shape {expected_rot_shape}, got {human_rotations_arr.shape}"
                )

        with self.lock:
            head = int(self.head.value)
            size = int(self.size.value)
            recv_ns_arr = self._recv_ns_array()
            if size > 0:
                latest_idx = self._physical_index(head, size - 1, self.capacity)
                latest_recv_ns = int(recv_ns_arr[latest_idx])
                if int(recv_ns) <= latest_recv_ns:
                    raise ValueError(
                        f"recv_ns must be strictly increasing, got {int(recv_ns)} after {latest_recv_ns}"
                    )
            if size == self.capacity:
                head = self._physical_index(head, 1, self.capacity)
                size -= 1

            write_idx = self._physical_index(head, size, self.capacity)
            recv_ns_arr[write_idx] = int(recv_ns)
            self._seq_array()[write_idx] = int(seq)
            self._qpos_array()[write_idx] = qpos_arr
            if self.store_human:
                self._human_positions_array()[write_idx] = human_positions_arr
                self._human_rotations_array()[write_idx] = human_rotations_arr

            size += 1
            self.retarget_count.value = int(self.retarget_count.value) + 1
            if dropped_before_process > 0:
                self.raw_drop_count.value = int(self.raw_drop_count.value) + int(dropped_before_process)

            cutoff_ns = int(recv_ns) - int(window_ns)
            while size > 1 and int(recv_ns_arr[head]) < cutoff_ns:
                head = self._physical_index(head, 1, self.capacity)
                size -= 1

            self.head.value = head
            self.size.value = size

    def sample(self, *, target_ns: int) -> RetargetBufferSample:
        with self.lock:
            size = int(self.size.value)
            if size == 0:
                return RetargetBufferSample(
                    qpos=None,
                    used_fallback=True,
                    info={
                        "mode": "no_frame",
                        "buffer_len": 0,
                    },
                )

            head = int(self.head.value)
            recv_ns_arr = self._recv_ns_array()
            qpos_arr = self._qpos_array()

            latest_idx = self._physical_index(head, size - 1, self.capacity)
            latest_ns = int(recv_ns_arr[latest_idx])
            if target_ns >= latest_ns:
                return RetargetBufferSample(
                    qpos=qpos_arr[latest_idx].copy(),
                    used_fallback=True,
                    info={
                        "mode": "fallback_latest",
                        "buffer_len": size,
                    },
                )

            next_logical = size - 1
            next_ns = latest_ns
            next_idx = latest_idx
            scan_steps = 0
            while next_logical > 0:
                prev_logical = next_logical - 1
                prev_idx = self._physical_index(head, prev_logical, self.capacity)
                prev_ns = int(recv_ns_arr[prev_idx])
                scan_steps += 1
                if target_ns >= prev_ns:
                    dt = next_ns - prev_ns
                    assert dt > 0, f"retarget buffer timestamps are not strictly increasing: prev={prev_ns}, next={next_ns}"
                    prev_qpos = qpos_arr[prev_idx].copy()
                    next_qpos = qpos_arr[next_idx].copy()
                    alpha = float(target_ns - prev_ns) / float(dt)
                    return RetargetBufferSample(
                        qpos=interpolate_qpos(prev_qpos, next_qpos, alpha),
                        used_fallback=False,
                        info={
                            "mode": "interpolate",
                            "buffer_len": size,
                        },
                    )
                next_logical = prev_logical
                next_ns = prev_ns
                next_idx = prev_idx

            return RetargetBufferSample(
                qpos=qpos_arr[latest_idx].copy(),
                used_fallback=True,
                info={
                    "mode": "fallback_latest",
                    "buffer_len": size,
                },
            )
