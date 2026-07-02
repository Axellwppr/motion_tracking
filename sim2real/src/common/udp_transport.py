from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.udp_latest import LatestPacket, UDPLatestReceiver, UDPLatestSender


def _cfg_get(data, key: str):
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


@dataclass(frozen=True)
class UDPRobotHighConfig:
    state_bind_host: str
    state_port: int
    cmd_host: str
    cmd_port: int

    @classmethod
    def from_data(cls, data) -> "UDPRobotHighConfig":
        return cls(
            state_bind_host=str(_cfg_get(data, "state_bind_host")),
            state_port=int(_cfg_get(data, "state_port")),
            cmd_host=str(_cfg_get(data, "cmd_host")),
            cmd_port=int(_cfg_get(data, "cmd_port")),
        )


@dataclass(frozen=True)
class UDPRobotLowConfig:
    state_host: str
    state_port: int
    cmd_bind_host: str
    cmd_port: int

    @classmethod
    def from_data(cls, data) -> "UDPRobotLowConfig":
        return cls(
            state_host=str(_cfg_get(data, "state_host")),
            state_port=int(_cfg_get(data, "state_port")),
            cmd_bind_host=str(_cfg_get(data, "cmd_bind_host")),
            cmd_port=int(_cfg_get(data, "cmd_port")),
        )


def _command_payload(
    *,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    enable: int,
    extra_command: dict[str, Any] | None = None,
    state_receive_time_ns: int | None = None,
) -> dict[str, Any]:
    payload = {
        "q_des": np.asarray(q_des, dtype=np.float32),
        "qd_des": np.asarray(qd_des, dtype=np.float32),
        "kp": np.asarray(kp, dtype=np.float32),
        "kd": np.asarray(kd, dtype=np.float32),
        "enable": int(enable),
    }
    if extra_command is not None:
        payload["extra_command"] = extra_command
    if state_receive_time_ns is not None:
        payload["state_receive_time_ns"] = int(state_receive_time_ns)
    return payload


def _state_payload(
    *,
    q: np.ndarray,
    dq: np.ndarray,
    quat_wxyz: np.ndarray,
    gyro: np.ndarray,
    linacc: np.ndarray,
    buttons: dict[str, bool],
    sticks: dict[str, float],
    state_receive_time_ns: int | None = None,
) -> dict[str, Any]:
    return {
        "q": np.asarray(q, dtype=np.float32),
        "dq": np.asarray(dq, dtype=np.float32),
        "quat_wxyz": np.asarray(quat_wxyz, dtype=np.float32),
        "gyro": np.asarray(gyro, dtype=np.float32),
        "linacc": np.asarray(linacc, dtype=np.float32),
        "buttons": buttons,
        "sticks": sticks,
        "state_receive_time_ns": int(time.perf_counter_ns() if state_receive_time_ns is None else state_receive_time_ns),
    }


class UDPRobotHigh:
    def __init__(self, cfg: UDPRobotHighConfig | dict | Any):
        self.cfg = cfg if isinstance(cfg, UDPRobotHighConfig) else UDPRobotHighConfig.from_data(cfg)
        self.state_receiver = UDPLatestReceiver(
            self.cfg.state_bind_host,
            self.cfg.state_port,
        )
        self.command_sender = UDPLatestSender(
            self.cfg.cmd_host,
            self.cfg.cmd_port,
        )
        self.state_receiver.start()

    def close(self) -> None:
        self.state_receiver.close()
        self.command_sender.close()

    def read_latest_state(self, *, with_meta: bool = False):
        return self.state_receiver.read_latest_data(with_meta=with_meta)

    def read_next_state(
        self,
        *,
        after_seq: int | None = None,
        timeout_s: float | None = None,
        with_meta: bool = False,
    ):
        return self.state_receiver.read_next_data(after_seq=after_seq, timeout_s=timeout_s, with_meta=with_meta)

    def send_command(
        self,
        *,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        enable: int,
        extra_command: dict[str, Any] | None = None,
        state_receive_time_ns: int | None = None,
    ) -> int:
        return self.command_sender.send(
            _command_payload(
                q_des=q_des,
                qd_des=qd_des,
                kp=kp,
                kd=kd,
                enable=enable,
                extra_command=extra_command,
                state_receive_time_ns=state_receive_time_ns,
            )
        )


class UDPRobotLow:
    def __init__(self, cfg: UDPRobotLowConfig | dict | Any, *, on_command_packet=None):
        self.cfg = cfg if isinstance(cfg, UDPRobotLowConfig) else UDPRobotLowConfig.from_data(cfg)
        self.state_sender = UDPLatestSender(
            self.cfg.state_host,
            self.cfg.state_port,
        )
        self.command_receiver = UDPLatestReceiver(
            self.cfg.cmd_bind_host,
            self.cfg.cmd_port,
            on_packet=on_command_packet,
        )
        self.command_receiver.start()

    def close(self) -> None:
        self.command_receiver.close()
        self.state_sender.close()

    def read_latest_command(self, *, with_meta: bool = False):
        return self.command_receiver.read_latest_data(with_meta=with_meta)

    def send_state(
        self,
        *,
        q: np.ndarray,
        dq: np.ndarray,
        quat_wxyz: np.ndarray,
        gyro: np.ndarray,
        linacc: np.ndarray,
        buttons: dict[str, bool],
        sticks: dict[str, float],
        state_receive_time_ns: int | None = None,
    ) -> int:
        return self.state_sender.send(
            _state_payload(
                q=q,
                dq=dq,
                quat_wxyz=quat_wxyz,
                gyro=gyro,
                linacc=linacc,
                buttons=buttons,
                sticks=sticks,
                state_receive_time_ns=state_receive_time_ns,
            )
        )
