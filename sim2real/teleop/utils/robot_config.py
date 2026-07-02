from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import yaml

TELEOP_ROOT = Path(__file__).resolve().parents[1]
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retarget.params import resolve_robot_xml_path
from paths import SUPPORTED_ROBOTS, retarget_teleop_config_path

SIM2REAL_ROOT = TELEOP_ROOT.parent
CONTROLLER_BUTTON_NAMES = frozenset(
    {
        "left_key_one",
        "left_key_two",
        "left_axis_click",
        "left_index_trig",
        "left_grip",
        "right_key_one",
        "right_key_two",
        "right_axis_click",
        "right_index_trig",
        "right_grip",
    }
)


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], prefix: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing required keys under {prefix}: {missing}")


def bind_addr_to_connect_addr(addr: str) -> str:
    if not addr.startswith("tcp://"):
        return addr
    host_port = addr[len("tcp://") :]
    host, sep, port = host_port.rpartition(":")
    if not sep:
        return addr
    if host in ("", "*", "0.0.0.0"):
        host = "127.0.0.1"
    return f"tcp://{host}:{port}"


def _load_robot_names_from_xml(target_robot: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        import mujoco as mj
    except ImportError as exc:
        raise ImportError(
            "Failed to import MuJoCo robot metadata dependencies. "
            "Please use the teleop runtime environment with mujoco installed."
        ) from exc

    model = mj.MjModel.from_xml_path(str(resolve_robot_xml_path(target_robot)))
    dof_names = []
    for i in range(model.njnt):
        joint_type = int(model.jnt_type[i])
        if joint_type == int(mj.mjtJoint.mjJNT_FREE):
            continue
        dof_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if dof_name is None:
            raise ValueError(f"Failed to resolve joint name at index {i} for target robot '{target_robot}'")
        dof_names.append(str(dof_name))

    body_names = []
    for i in range(model.nbody):
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if body_name is None:
            raise ValueError(f"Failed to resolve body name at index {i} for target robot '{target_robot}'")
        body_names.append(str(body_name))

    return tuple(dof_names), tuple(body_names)


@dataclass(frozen=True)
class TeleopRobotConfig:
    robot_key: str
    max_iter: int
    actual_human_height: float
    height_alignment_enabled: bool
    height_alignment_xrobot_body_min_each_frame: bool
    height_alignment_foot_body_names: tuple[str, ...]
    height_alignment_target_z: float
    height_alignment_bootstrap_frames: int
    calibration_button: str | None
    dof_names: tuple[str, ...]
    body_names: tuple[str, ...]
    req_bind_addr: str
    rep_bind_addr: str
    ctrl_bind_addr: str
    ctrl_fps: int
    vis_fps: int
    lookback_ms: float
    retarget_buffer_window_s: float
    log_interval_s: float
    visualize: bool
    recorder_period_ms: float
    recorder_output_dir: Path

    @property
    def dof_count(self) -> int:
        return len(self.dof_names)

    @property
    def qpos_size(self) -> int:
        return 7 + self.dof_count

    @property
    def req_connect_addr(self) -> str:
        return bind_addr_to_connect_addr(self.req_bind_addr)

    @property
    def rep_connect_addr(self) -> str:
        return bind_addr_to_connect_addr(self.rep_bind_addr)

    @property
    def ctrl_connect_addr(self) -> str:
        return bind_addr_to_connect_addr(self.ctrl_bind_addr)


def load_teleop_robot_config(robot: str) -> TeleopRobotConfig:
    robot_key = str(robot).strip().lower()
    if robot_key not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unsupported robot '{robot}'. Expected one of {SUPPORTED_ROBOTS}.")

    cfg_path = retarget_teleop_config_path(robot_key)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Teleop config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _require_keys(raw, ("robot_key", "retarget", "server", "recorder"), str(cfg_path))
    retarget = raw["retarget"]
    server = raw["server"]
    recorder = raw["recorder"]
    height_alignment = retarget.get("height_alignment", {})
    calibration = retarget.get("calibration", {})
    height_alignment_xrobot_body_min_each_frame = bool(
        height_alignment.get("xrobot_body_min_each_frame", False)
    )

    _require_keys(
        retarget,
        (
            "max_iter",
            "actual_human_height",
        ),
        "retarget",
    )
    _require_keys(
        server,
        (
            "req_bind_addr",
            "rep_bind_addr",
            "ctrl_bind_addr",
            "ctrl_fps",
            "vis_fps",
            "lookback_ms",
            "retarget_buffer_window_s",
            "log_interval_s",
            "visualize",
        ),
        "server",
    )
    _require_keys(recorder, ("period_ms", "output_dir"), "recorder")

    target_robot = robot_key
    dof_names, body_names = _load_robot_names_from_xml(target_robot)
    body_name_set = set(body_names)
    foot_body_names = tuple(
        str(name).strip() for name in height_alignment.get("foot_body_names", ())
    )
    if not height_alignment_xrobot_body_min_each_frame and len(foot_body_names) != 2:
        raise ValueError("retarget.height_alignment.foot_body_names must contain exactly two body names")
    if height_alignment_xrobot_body_min_each_frame and len(foot_body_names) not in (0, 2):
        raise ValueError(
            "retarget.height_alignment.foot_body_names must be omitted or contain exactly two body names "
            "when xrobot_body_min_each_frame is enabled"
        )
    missing_foot_body_names = [name for name in foot_body_names if name not in body_name_set]
    if missing_foot_body_names:
        raise ValueError(
            f"retarget.height_alignment.foot_body_names contains unknown body names for '{target_robot}': "
            f"{missing_foot_body_names}"
        )
    calibration_button_raw = calibration.get("button", None) if isinstance(calibration, dict) else None
    calibration_button = None
    if calibration_button_raw is not None:
        calibration_button = str(calibration_button_raw).strip()
        if not calibration_button:
            calibration_button = None
    if calibration_button is not None and calibration_button not in CONTROLLER_BUTTON_NAMES:
        raise ValueError(
            f"retarget.calibration.button must be one of {sorted(CONTROLLER_BUTTON_NAMES)}, "
            f"got {calibration_button!r}"
        )

    recorder_output_dir = Path(str(recorder["output_dir"]))
    if not recorder_output_dir.is_absolute():
        recorder_output_dir = SIM2REAL_ROOT / recorder_output_dir

    cfg = TeleopRobotConfig(
        robot_key=str(raw["robot_key"]).strip().lower(),
        max_iter=int(retarget["max_iter"]),
        actual_human_height=float(retarget["actual_human_height"]),
        height_alignment_enabled=bool(height_alignment.get("enabled", True)),
        height_alignment_xrobot_body_min_each_frame=height_alignment_xrobot_body_min_each_frame,
        height_alignment_foot_body_names=foot_body_names,
        height_alignment_target_z=float(height_alignment.get("target_z", 0.0)),
        height_alignment_bootstrap_frames=int(height_alignment.get("bootstrap_frames", 30)),
        calibration_button=calibration_button,
        dof_names=dof_names,
        body_names=body_names,
        req_bind_addr=str(server["req_bind_addr"]).strip(),
        rep_bind_addr=str(server["rep_bind_addr"]).strip(),
        ctrl_bind_addr=str(server["ctrl_bind_addr"]).strip(),
        ctrl_fps=int(server["ctrl_fps"]),
        vis_fps=int(server["vis_fps"]),
        lookback_ms=float(server["lookback_ms"]),
        retarget_buffer_window_s=float(server["retarget_buffer_window_s"]),
        log_interval_s=float(server["log_interval_s"]),
        visualize=bool(server["visualize"]),
        recorder_period_ms=float(recorder["period_ms"]),
        recorder_output_dir=recorder_output_dir,
    )

    if cfg.robot_key != robot_key:
        raise ValueError(f"Config robot_key='{cfg.robot_key}' does not match requested robot='{robot_key}'")
    if cfg.max_iter <= 0:
        raise ValueError("retarget.max_iter must be > 0")
    if cfg.ctrl_fps <= 0:
        raise ValueError("server.ctrl_fps must be > 0")
    if cfg.vis_fps <= 0:
        raise ValueError("server.vis_fps must be > 0")
    if cfg.lookback_ms < 0:
        raise ValueError("server.lookback_ms must be >= 0")
    if cfg.retarget_buffer_window_s <= 0:
        raise ValueError("server.retarget_buffer_window_s must be > 0")
    if cfg.log_interval_s < 0:
        raise ValueError("server.log_interval_s must be >= 0")
    if cfg.recorder_period_ms <= 0:
        raise ValueError("recorder.period_ms must be > 0")
    if cfg.height_alignment_bootstrap_frames <= 0:
        raise ValueError("retarget.height_alignment.bootstrap_frames must be > 0")

    return cfg
