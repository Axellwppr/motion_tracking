from __future__ import annotations

import copy
import dataclasses
import json
import select
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np


MAGIC = b"ULDP"
VERSION = 1
HEADER_STRUCT = struct.Struct("!4sBBHQQII")
CRC_STRUCT = struct.Struct("!I")
TYPE_KEY = "__udp_latest_type__"
DEFAULT_UDP_RECVBUF_BYTES = 1 << 20
DEFAULT_UDP_SNDBUF_BYTES = 1 << 20


@dataclass(frozen=True)
class LatestPacket:
    seq: int
    send_time_ns: int
    recv_time_ns: int
    addr: tuple[str, int] | None
    data: Any


@dataclass(frozen=True)
class ReceiverStats:
    packets_received: int
    packets_decoded: int
    crc_errors: int
    format_errors: int
    latest_seq: int


def _pack_value(value: Any, buffers: list[bytes]) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object dtype arrays are not supported")
        array = np.ascontiguousarray(value)
        offset = sum(len(buf) for buf in buffers)
        raw = array.view(np.uint8).tobytes()
        buffers.append(raw)
        return {
            TYPE_KEY: "ndarray",
            "dtype": array.dtype.str,
            "shape": tuple(array.shape),
            "offset": offset,
            "nbytes": len(raw),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        offset = sum(len(buf) for buf in buffers)
        buffers.append(value)
        return {
            TYPE_KEY: "bytes",
            "offset": offset,
            "nbytes": len(value),
        }
    if isinstance(value, tuple):
        return {TYPE_KEY: "tuple", "items": [_pack_value(v, buffers) for v in value]}
    if isinstance(value, list):
        return [_pack_value(v, buffers) for v in value]
    if isinstance(value, dict):
        return {str(k): _pack_value(v, buffers) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return {
            TYPE_KEY: "dataclass",
            "class_name": value.__class__.__qualname__,
            "fields": _pack_value(dataclasses.asdict(value), buffers),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported payload type: {type(value)!r}")


def _unpack_value(meta: Any, payload: memoryview) -> Any:
    if isinstance(meta, list):
        return [_unpack_value(v, payload) for v in meta]
    if not isinstance(meta, dict):
        return meta

    wire_type = meta.get(TYPE_KEY)
    if wire_type == "ndarray":
        offset = int(meta["offset"])
        nbytes = int(meta["nbytes"])
        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        array = np.frombuffer(payload[offset : offset + nbytes], dtype=dtype).reshape(shape)
        array.setflags(write=False)
        return array
    if wire_type == "bytes":
        offset = int(meta["offset"])
        nbytes = int(meta["nbytes"])
        return payload[offset : offset + nbytes].tobytes()
    if wire_type == "tuple":
        return tuple(_unpack_value(v, payload) for v in meta["items"])
    if wire_type == "dataclass":
        return _unpack_value(meta["fields"], payload)
    return {k: _unpack_value(v, payload) for k, v in meta.items()}


def encode_datagram(data: Any, *, seq: int, send_time_ns: int | None = None) -> bytes:
    buffers: list[bytes] = []
    meta_obj = _pack_value(data, buffers)
    meta_bytes = json.dumps(meta_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = b"".join(buffers)
    send_time_ns = time.perf_counter_ns() if send_time_ns is None else int(send_time_ns)
    header = HEADER_STRUCT.pack(MAGIC, VERSION, 0, 0, int(seq), send_time_ns, len(meta_bytes), len(payload))
    crc = zlib.crc32(header)
    crc = zlib.crc32(meta_bytes, crc)
    crc = zlib.crc32(payload, crc)
    return header + CRC_STRUCT.pack(crc & 0xFFFFFFFF) + meta_bytes + payload


def decode_datagram(packet: bytes) -> LatestPacket:
    if len(packet) < HEADER_STRUCT.size + CRC_STRUCT.size:
        raise ValueError("packet too short")
    magic, version, _, _, seq, send_time_ns, meta_len, payload_len = HEADER_STRUCT.unpack_from(packet, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError(f"unsupported version={version}")
    expected_len = HEADER_STRUCT.size + CRC_STRUCT.size + meta_len + payload_len
    if len(packet) != expected_len:
        raise ValueError("packet length mismatch")
    crc_recv = CRC_STRUCT.unpack_from(packet, HEADER_STRUCT.size)[0]
    crc_calc = zlib.crc32(packet[:HEADER_STRUCT.size])
    start = HEADER_STRUCT.size + CRC_STRUCT.size
    meta_bytes = packet[start : start + meta_len]
    payload = memoryview(packet)[start + meta_len :]
    crc_calc = zlib.crc32(meta_bytes, crc_calc)
    crc_calc = zlib.crc32(payload, crc_calc) & 0xFFFFFFFF
    if crc_calc != crc_recv:
        raise ValueError("crc mismatch")
    meta_obj = json.loads(meta_bytes.decode("utf-8"))
    data = _unpack_value(meta_obj, payload)
    return LatestPacket(
        seq=int(seq),
        send_time_ns=int(send_time_ns),
        recv_time_ns=0,
        addr=None,
        data=data,
    )


class UDPLatestSender:
    def __init__(self, host: str, port: int, *, bind_host: str = "", sndbuf_bytes: int = DEFAULT_UDP_SNDBUF_BYTES):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if bind_host:
            self._sock.bind((bind_host, 0))
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(sndbuf_bytes))
        self._target = (host, port)
        self._seq = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        self._sock.close()

    def send(self, data: Any) -> int:
        with self._lock:
            seq = self._seq
            self._seq += 1
        packet = encode_datagram(data, seq=seq)
        self._sock.sendto(packet, self._target)
        return seq


class UDPLatestReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        on_packet=None,
        recvbuf_bytes: int = DEFAULT_UDP_RECVBUF_BYTES,
        max_packet_bytes: int = 65535,
        poll_timeout_s: float = 0.05,
    ):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(recvbuf_bytes))
        self._sock.bind((host, port))
        self._sock.setblocking(False)

        self._max_packet_bytes = int(max_packet_bytes)
        self._poll_timeout_s = float(poll_timeout_s)
        self._on_packet = on_packet
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest: LatestPacket | None = None
        self._packets_received = 0
        self._packets_decoded = 0
        self._crc_errors = 0
        self._format_errors = 0

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=0.5)
        self._sock.close()

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self._sock], [], [], self._poll_timeout_s)
            if not readable:
                continue

            last_packet: bytes | None = None
            last_addr: tuple[str, int] | None = None
            recv_time_ns = 0
            while True:
                try:
                    packet, addr = self._sock.recvfrom(self._max_packet_bytes)
                except BlockingIOError:
                    break
                self._packets_received += 1
                last_packet = packet
                last_addr = addr
                recv_time_ns = time.perf_counter_ns()

            if last_packet is None:
                continue

            try:
                decoded = decode_datagram(last_packet)
            except ValueError as exc:
                if "crc" in str(exc):
                    self._crc_errors += 1
                else:
                    self._format_errors += 1
                continue

            decoded = LatestPacket(
                seq=decoded.seq,
                send_time_ns=decoded.send_time_ns,
                recv_time_ns=recv_time_ns,
                addr=last_addr,
                data=decoded.data,
            )
            with self._lock:
                self._latest = decoded
                self._packets_decoded += 1
                self._condition.notify_all()
            if self._on_packet is not None:
                self._on_packet(decoded)

    def read_latest_data(self, *, with_meta: bool = False, copy_data: bool = False) -> Any:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        if copy_data:
            data = copy.deepcopy(latest.data)
            latest = LatestPacket(
                seq=latest.seq,
                send_time_ns=latest.send_time_ns,
                recv_time_ns=latest.recv_time_ns,
                addr=latest.addr,
                data=data,
            )
        return latest if with_meta else latest.data

    def read_next_data(
        self,
        *,
        after_seq: int | None = None,
        timeout_s: float | None = None,
        with_meta: bool = False,
        copy_data: bool = False,
    ) -> Any:
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                latest = self._latest
                if latest is not None and (after_seq is None or int(latest.seq) > int(after_seq)):
                    break
                if self._stop.is_set():
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

        if copy_data:
            data = copy.deepcopy(latest.data)
            latest = LatestPacket(
                seq=latest.seq,
                send_time_ns=latest.send_time_ns,
                recv_time_ns=latest.recv_time_ns,
                addr=latest.addr,
                data=data,
            )
        return latest if with_meta else latest.data

    def get_stats(self) -> ReceiverStats:
        with self._lock:
            latest_seq = -1 if self._latest is None else self._latest.seq
        return ReceiverStats(
            packets_received=self._packets_received,
            packets_decoded=self._packets_decoded,
            crc_errors=self._crc_errors,
            format_errors=self._format_errors,
            latest_seq=latest_seq,
        )
