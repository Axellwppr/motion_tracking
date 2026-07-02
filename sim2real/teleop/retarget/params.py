from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import SUPPORTED_ROBOTS, retarget_ik_config_path, robot_xml_path


HERE = Path(__file__).resolve().parent
TELEOP_ROOT = HERE.parent
SUPPORTED_TARGET_ROBOTS = SUPPORTED_ROBOTS
XR_BODY_JOINT_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)


def resolve_robot_xml_path(target_robot: str) -> Path:
    robot_key = str(target_robot).strip().lower()
    if robot_key not in SUPPORTED_ROBOTS:
        raise KeyError(f"Unsupported target robot '{target_robot}'. Expected one of {SUPPORTED_TARGET_ROBOTS}.")
    xml_path = robot_xml_path(robot_key)
    if not xml_path.exists():
        raise FileNotFoundError(f"Robot XML not found for '{robot_key}': {xml_path}")
    return xml_path


def load_xrobot_ik_config(target_robot: str) -> dict[str, Any]:
    robot_key = str(target_robot).strip().lower()
    if robot_key not in SUPPORTED_ROBOTS:
        raise KeyError(f"Unsupported xrobot IK target '{target_robot}'. Expected one of {SUPPORTED_TARGET_ROBOTS}.")

    cfg_path = retarget_ik_config_path(robot_key)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_robot_axis_link_names(target_robot: str) -> tuple[str, ...]:
    ik_config = load_xrobot_ik_config(target_robot)
    ik_match_table1 = ik_config.get("ik_match_table1", {})
    if not isinstance(ik_match_table1, dict):
        raise TypeError(f"ik_match_table1 must be a dict for target robot '{target_robot}'")
    return tuple(str(name) for name in ik_match_table1.keys())


def load_human_body_names(target_robot: str) -> tuple[str, ...]:
    ik_config = load_xrobot_ik_config(target_robot)
    human_scale_table = ik_config.get("human_scale_table", {})
    if not isinstance(human_scale_table, dict):
        raise TypeError(f"human_scale_table must be a dict for target robot '{target_robot}'")
    return tuple(str(name) for name in human_scale_table.keys())
