from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mink
import mujoco as mj
import numpy as np

from utils.math import (
    quat_apply_np,
    quat_mul_np,
    quat_normalize_np,
    quat_xyzw_to_wxyz_np,
    rotmat_to_quat_np,
)

from .params import XR_BODY_JOINT_NAMES, load_xrobot_ik_config, resolve_robot_xml_path

TELEOP_HEIGHT_ALIGNMENT_TARGET_Z = 0.0
TELEOP_HEIGHT_ALIGNMENT_BOOTSTRAP_FRAMES = 30
RETARGET_IK_TIMESTEP = 0.005
DEFAULT_IK_GROUP = "default"


@dataclass(frozen=True)
class XRobotRetargetConfig:
    target_robot: str
    actual_human_height: float | None = None
    solver: str = "daqp"
    damping: float = 5e-1
    max_iter: int = 10
    use_velocity_limit: bool = True


@dataclass(frozen=True)
class UpperBodyScaleSegment:
    name: str
    human_body_name: str
    human_parent_name: str
    robot_body_name: str
    robot_parent_name: str
    human_body_idx: int
    human_parent_idx: int
    robot_length: float
    robot_delta_parent_frame: np.ndarray
    scale_min: float
    scale_max: float
    calibrate_position_offset: bool
    position_offset_limit: float


class XRobotRetargeter:
    def __init__(self, config: XRobotRetargetConfig):
        self.config = config
        self.xml_file = str(resolve_robot_xml_path(config.target_robot))
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        self.model.opt.timestep = RETARGET_IK_TIMESTEP
        self.xr_body_name_to_index = {name: idx for idx, name in enumerate(XR_BODY_JOINT_NAMES)}

        self.robot_dof_names = {}
        for idx in range(self.model.nv):
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[idx])
            self.robot_dof_names[str(dof_name)] = idx

        self.robot_joint_names = []
        for idx in range(self.model.njnt):
            if int(self.model.jnt_type[idx]) == int(mj.mjtJoint.mjJNT_FREE):
                continue
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, idx)
            self.robot_joint_names.append(str(joint_name))

        self.robot_body_names = {}
        for idx in range(self.model.nbody):
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, idx)
            self.robot_body_names[str(body_name)] = idx

        ik_config = load_xrobot_ik_config(config.target_robot)
        self.human_height_assumption = float(ik_config["human_height_assumption"])
        self.base_human_scale_table = {
            key: float(value) for key, value in ik_config["human_scale_table"].items()
        }

        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.joint_groups = ik_config.get("joint_groups", {})
        self.task_groups = ik_config.get("task_groups", {})
        self.joint_limits = ik_config.get("joint_limits", {})
        self.human_root_name = str(ik_config["human_root_name"])
        self.robot_root_name = str(ik_config["robot_root_name"])
        self.use_ik_match_table1 = bool(ik_config["use_ik_match_table1"])
        self.use_ik_match_table2 = bool(ik_config["use_ik_match_table2"])
        self.ground = float(ik_config["ground_height"]) * np.array([0.0, 0.0, 1.0])
        self._apply_joint_limit_overrides()

        self.solver = str(config.solver)
        self.damping = float(config.damping)
        self.max_iter = int(config.max_iter)
        self.ground_offset = 0.0
        self._scaled_human_pos: np.ndarray | None = None
        self._scaled_human_rot: np.ndarray | None = None

        self._scale_body_names = tuple(self.base_human_scale_table.keys())
        self._scale_body_name_to_idx = {
            body_name: idx for idx, body_name in enumerate(self._scale_body_names)
        }
        self._scale_body_xr_indices = np.asarray(
            [self.xr_body_name_to_index[body_name] for body_name in self._scale_body_names],
            dtype=np.int64,
        )
        self._root_body_idx = self._scale_body_name_to_idx[self.human_root_name]
        self._non_root_body_indices = np.asarray(
            [idx for idx, body_name in enumerate(self._scale_body_names) if body_name != self.human_root_name],
            dtype=np.int64,
        )
        self._foot_body_mask = np.asarray(
            [("Foot" in body_name) or ("foot" in body_name) for body_name in self._scale_body_names],
            dtype=bool,
        )
        self._upper_body_scale_segments = self._parse_upper_body_scale_config(
            ik_config.get("upper_body_scale", None)
        )
        self._upper_body_segment_scales = np.ones((len(self._upper_body_scale_segments),), dtype=np.float64)
        self._upper_body_segment_position_offsets = np.zeros((len(self._upper_body_scale_segments), 3), dtype=np.float64)
        self._upper_body_scale_calibrated = False

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if config.use_velocity_limit:
            velocity_limits = {name: 3 * np.pi for name in self.robot_joint_names}
            self.ik_limits.append(mink.VelocityLimit(self.model, velocity_limits))

        self.set_human_height(config.actual_human_height)
        self.setup_retarget_configuration()

    def reset_configuration(self) -> None:
        self.configuration = mink.Configuration(self.model)
        self._scaled_human_pos = None
        self._scaled_human_rot = None

    def set_human_height(self, actual_human_height: float | None) -> None:
        ratio = 1.0 if actual_human_height is None else float(actual_human_height) / self.human_height_assumption
        self.human_scale_table = {
            key: self.base_human_scale_table[key] * ratio for key in self.base_human_scale_table
        }
        self._human_scale_factors = np.asarray(
            [self.human_scale_table[body_name] for body_name in self._scale_body_names],
            dtype=np.float64,
        )

    def _parse_joint_limit_override(self, joint_name: str, value: Any) -> tuple[float, float]:
        if isinstance(value, dict):
            lower_raw = value.get("lower", value.get("min"))
            upper_raw = value.get("upper", value.get("max"))
            if lower_raw is None or upper_raw is None:
                raise ValueError(f"joint_limits[{joint_name!r}] must define lower/min and upper/max")
            lower = float(lower_raw)
            upper = float(upper_raw)
        else:
            values = np.asarray(value, dtype=np.float64).reshape(-1)
            if values.shape[0] != 2:
                raise ValueError(f"joint_limits[{joint_name!r}] must contain exactly [lower, upper], got {value}")
            lower = float(values[0])
            upper = float(values[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError(f"Invalid joint limit for {joint_name!r}: lower={lower}, upper={upper}")
        return lower, upper

    def _apply_joint_limit_overrides(self) -> None:
        if self.joint_limits is None:
            return
        if not isinstance(self.joint_limits, dict):
            raise TypeError("retarget joint_limits must be a dict of joint_name -> [lower, upper]")

        for joint_name_raw, limit_value in self.joint_limits.items():
            joint_name = str(joint_name_raw)
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Unknown joint in retarget joint_limits: {joint_name}")
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"retarget joint_limits only supports hinge/slide joints, got {joint_name}")

            lower, upper = self._parse_joint_limit_override(joint_name, limit_value)
            self.model.jnt_limited[joint_id] = 1
            self.model.jnt_range[joint_id, 0] = lower
            self.model.jnt_range[joint_id, 1] = upper
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.model.qpos0[qpos_adr] = np.clip(float(self.model.qpos0[qpos_adr]), lower, upper)

    def _parse_scale_limits(self, value: Any, *, context: str) -> tuple[float, float]:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.shape[0] != 2:
            raise ValueError(f"{context} must contain exactly [min, max], got {value}")
        lower = float(values[0])
        upper = float(values[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or lower > upper:
            raise ValueError(f"Invalid {context}: min={lower}, max={upper}")
        return lower, upper

    def _parse_upper_body_scale_config(self, cfg: Any) -> tuple[UpperBodyScaleSegment, ...]:
        if cfg is None:
            return ()
        if not isinstance(cfg, dict):
            raise TypeError("retarget upper_body_scale must be a dict")
        if not bool(cfg.get("enabled", True)):
            return ()

        raw_segments = cfg.get("segments", ())
        if not raw_segments:
            return ()
        if not isinstance(raw_segments, list):
            raise TypeError("retarget upper_body_scale.segments must be a list")

        default_limits = self._parse_scale_limits(
            cfg.get("scale_limits", [0.6, 1.4]),
            context="retarget upper_body_scale.scale_limits",
        )
        default_position_offset_limit = float(cfg.get("position_offset_limit", np.inf))
        if not np.isfinite(default_position_offset_limit):
            default_position_offset_limit = np.inf
        elif default_position_offset_limit <= 0.0:
            raise ValueError("retarget upper_body_scale.position_offset_limit must be > 0")
        reference_data = mj.MjData(self.model)
        reference_data.qpos[:] = self.model.qpos0
        mj.mj_forward(self.model, reference_data)

        segments: list[UpperBodyScaleSegment] = []
        segment_body_owner: dict[str, str] = {}
        for idx, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                raise TypeError(f"retarget upper_body_scale.segments[{idx}] must be a dict")

            name = str(raw_segment.get("name", f"segment_{idx}"))
            human_body_name = str(raw_segment["human_body"])
            human_parent_name = str(raw_segment["human_parent"])
            robot_body_name = str(raw_segment["robot_body"])
            robot_parent_name = str(raw_segment["robot_parent"])
            if human_body_name in segment_body_owner:
                raise ValueError(
                    f"Human body '{human_body_name}' appears in multiple upper_body_scale segments: "
                    f"{segment_body_owner[human_body_name]} and {name}"
                )
            segment_body_owner[human_body_name] = name

            if human_body_name not in self._scale_body_name_to_idx:
                raise ValueError(f"Unknown upper_body_scale human_body: {human_body_name}")
            if human_parent_name not in self._scale_body_name_to_idx:
                raise ValueError(f"Unknown upper_body_scale human_parent: {human_parent_name}")

            robot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, robot_body_name)
            robot_parent_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, robot_parent_name)
            if robot_body_id == -1:
                raise ValueError(f"Unknown upper_body_scale robot_body: {robot_body_name}")
            if robot_parent_id == -1:
                raise ValueError(f"Unknown upper_body_scale robot_parent: {robot_parent_name}")

            robot_delta = reference_data.xpos[robot_body_id] - reference_data.xpos[robot_parent_id]
            robot_length = float(np.linalg.norm(robot_delta))
            if not np.isfinite(robot_length) or robot_length <= 1e-6:
                raise ValueError(
                    f"upper_body_scale segment '{name}' has invalid robot reference length: {robot_length}"
                )
            robot_parent_xmat = reference_data.xmat[robot_parent_id].reshape(3, 3)
            robot_delta_parent_frame = robot_parent_xmat.T @ robot_delta
            scale_limits = self._parse_scale_limits(
                raw_segment.get("scale_limits", default_limits),
                context=f"retarget upper_body_scale.segments[{idx}].scale_limits",
            )
            position_offset_limit = float(raw_segment.get("position_offset_limit", default_position_offset_limit))
            if not np.isfinite(position_offset_limit):
                position_offset_limit = np.inf
            elif position_offset_limit <= 0.0:
                raise ValueError(f"retarget upper_body_scale.segments[{idx}].position_offset_limit must be > 0")
            segments.append(
                UpperBodyScaleSegment(
                    name=name,
                    human_body_name=human_body_name,
                    human_parent_name=human_parent_name,
                    robot_body_name=robot_body_name,
                    robot_parent_name=robot_parent_name,
                    human_body_idx=self._scale_body_name_to_idx[human_body_name],
                    human_parent_idx=self._scale_body_name_to_idx[human_parent_name],
                    robot_length=robot_length,
                    robot_delta_parent_frame=np.asarray(robot_delta_parent_frame, dtype=np.float64),
                    scale_min=scale_limits[0],
                    scale_max=scale_limits[1],
                    calibrate_position_offset=bool(raw_segment.get("calibrate_position_offset", False)),
                    position_offset_limit=position_offset_limit,
                )
            )
        return tuple(segments)

    @property
    def upper_body_scale_enabled(self) -> bool:
        return len(self._upper_body_scale_segments) > 0

    @property
    def upper_body_scale_calibrated(self) -> bool:
        return bool(self._upper_body_scale_calibrated)

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        quat_arr = quat_normalize_np(quat)
        out = np.array(quat_arr, dtype=np.float64, copy=True)
        out[..., 1:] *= -1.0
        return out

    def _select_scale_body_arrays(
        self,
        body_pos: np.ndarray,
        body_rot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body_pos_arr = np.asarray(body_pos, dtype=np.float64)
        body_rot_arr = np.asarray(body_rot, dtype=np.float64)
        if body_pos_arr.ndim != 2 or body_pos_arr.shape[0] < len(XR_BODY_JOINT_NAMES) or body_pos_arr.shape[1] < 3:
            raise ValueError("body_pos for upper-body scale calibration must be full XR positions")
        if body_rot_arr.ndim != 2 or body_rot_arr.shape[0] < len(XR_BODY_JOINT_NAMES) or body_rot_arr.shape[1] < 4:
            raise ValueError("body_rot for upper-body scale calibration must be full XR rotations")
        selected_pos = body_pos_arr[self._scale_body_xr_indices, :3]
        selected_rot = quat_normalize_np(body_rot_arr[self._scale_body_xr_indices, :4])
        return selected_pos, selected_rot

    def calibrate_upper_body_scale(self, body_pos: np.ndarray, body_rot: np.ndarray) -> bool:
        if not self._upper_body_scale_segments:
            return False

        selected_pos, selected_rot = self._select_scale_body_arrays(body_pos, body_rot)
        aligned_rot = quat_normalize_np(quat_mul_np(selected_rot, self._scale_body_rot_offsets_quat))

        scales: list[float] = []
        raw_position_offsets = np.zeros((len(self._upper_body_scale_segments), 3), dtype=np.float64)
        position_offsets = np.zeros((len(self._upper_body_scale_segments), 3), dtype=np.float64)
        position_offset_applied = np.zeros((len(self._upper_body_scale_segments),), dtype=bool)
        for segment in self._upper_body_scale_segments:
            human_delta = selected_pos[segment.human_body_idx] - selected_pos[segment.human_parent_idx]
            human_length = float(np.linalg.norm(human_delta))
            if not np.isfinite(human_length) or human_length <= 1e-6:
                return False
            scale = segment.robot_length / human_length
            scales.append(float(np.clip(scale, segment.scale_min, segment.scale_max)))

        for idx, (segment, scale) in enumerate(zip(self._upper_body_scale_segments, scales, strict=True)):
            if not segment.calibrate_position_offset:
                continue
            human_delta = selected_pos[segment.human_body_idx] - selected_pos[segment.human_parent_idx]
            scaled_human_delta = human_delta * float(scale)
            parent_inv_rot = self._quat_conjugate(aligned_rot[segment.human_parent_idx])
            scaled_human_delta_parent_frame = quat_apply_np(
                parent_inv_rot,
                scaled_human_delta,
                normalize=False,
            )
            raw_offset = segment.robot_delta_parent_frame - scaled_human_delta_parent_frame
            raw_position_offsets[idx] = raw_offset
            if float(np.linalg.norm(raw_offset)) <= segment.position_offset_limit:
                position_offsets[idx] = raw_offset
                position_offset_applied[idx] = True

        self._upper_body_segment_position_offsets = position_offsets
        self._upper_body_segment_scales = np.asarray(scales, dtype=np.float64)
        self._upper_body_scale_calibrated = True
        self._print_upper_body_calibration_summary(
            selected_pos,
            raw_position_offsets,
            position_offsets,
            position_offset_applied,
        )
        return True

    def _print_upper_body_calibration_summary(
        self,
        selected_pos: np.ndarray,
        raw_position_offsets: np.ndarray,
        position_offsets: np.ndarray,
        position_offset_applied: np.ndarray,
    ) -> None:
        print("[Retarget] upper_body_scale calibration:")
        for idx, segment in enumerate(self._upper_body_scale_segments):
            human_delta = selected_pos[segment.human_body_idx] - selected_pos[segment.human_parent_idx]
            human_length = float(np.linalg.norm(human_delta))
            raw_offset = raw_position_offsets[idx]
            raw_offset_norm = float(np.linalg.norm(raw_offset))
            offset = position_offsets[idx]
            offset_norm = float(np.linalg.norm(offset))
            offset_state = "applied" if bool(position_offset_applied[idx]) else "skipped"
            if not segment.calibrate_position_offset:
                offset_state = "off"
            print(
                "  "
                f"{segment.name}: scale={self._upper_body_segment_scales[idx]:.4f} "
                f"human_len={human_length:.4f} robot_len={segment.robot_length:.4f} "
                f"raw_offset={np.array2string(raw_offset, precision=4, suppress_small=True)} "
                f"raw_offset_norm={raw_offset_norm:.4f} "
                f"offset={np.array2string(offset, precision=4, suppress_small=True)} "
                f"offset_norm={offset_norm:.4f} "
                f"limit={segment.position_offset_limit:.4f} state={offset_state}"
            )

    def setup_retarget_configuration(self) -> None:
        self.configuration = mink.Configuration(self.model)
        self.tasks1: list[mink.FrameTask] = []
        self.tasks2: list[mink.FrameTask] = []
        self.tasks1_by_group: dict[str, list[mink.FrameTask]] = {}
        self.tasks2_by_group: dict[str, list[mink.FrameTask]] = {}
        self.constraints_by_group: dict[str, list[mink.Task] | None] = {}
        self.ik_group_order: list[str] = []
        self.use_phased_ik = False
        task1_body_names: list[str] = []
        task2_body_names: list[str] = []
        task1_frame_names: list[str] = []
        task2_frame_names: list[str] = []
        pos_offsets_by_body: dict[str, np.ndarray] = {}
        rot_offsets_by_body: dict[str, np.ndarray] = {}

        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = self._parse_ik_entry(entry)
            if pos_weight == 0 and rot_weight == 0:
                continue
            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            body_name = str(body_name)
            task1_body_names.append(body_name)
            pos_offsets_by_body[body_name] = np.asarray(pos_offset, dtype=np.float64) - self.ground
            rot_offsets_by_body[body_name] = np.asarray(rot_offset, dtype=np.float64)
            self.tasks1.append(task)
            task1_frame_names.append(str(frame_name))

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = self._parse_ik_entry(entry)
            if pos_weight == 0 and rot_weight == 0:
                continue
            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            body_name = str(body_name)
            task2_body_names.append(body_name)
            self.tasks2.append(task)
            task2_frame_names.append(str(frame_name))
        missing_offsets = [body_name for body_name in self._scale_body_names if body_name not in pos_offsets_by_body]
        if missing_offsets:
            raise KeyError(f"Missing task-1 offsets for scale bodies: {missing_offsets}")

        self._scale_body_pos_offsets = np.asarray(
            [pos_offsets_by_body[body_name] for body_name in self._scale_body_names],
            dtype=np.float64,
        )
        self._scale_body_rot_offsets_quat = np.asarray(
            [rot_offsets_by_body[body_name] for body_name in self._scale_body_names],
            dtype=np.float64,
        )
        self._scale_body_rot_offsets_quat = quat_normalize_np(self._scale_body_rot_offsets_quat)
        self._task1_body_indices = np.asarray(
            [self._scale_body_name_to_idx[body_name] for body_name in task1_body_names],
            dtype=np.int64,
        )
        self._task2_body_indices = np.asarray(
            [self._scale_body_name_to_idx[body_name] for body_name in task2_body_names],
            dtype=np.int64,
        )
        self._setup_grouped_ik(task1_frame_names, task2_frame_names)

    @staticmethod
    def _parse_ik_entry(entry: Any) -> tuple[str, float, float, Any, Any]:
        body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
        return str(body_name), float(pos_weight), float(rot_weight), pos_offset, rot_offset

    def _joint_names_to_dof_indices(self, joint_names: tuple[str, ...]) -> list[int]:
        dof_indices: list[int] = []
        for joint_name in joint_names:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Unknown joint in retarget joint_groups: {joint_name}")
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == int(mj.mjtJoint.mjJNT_FREE):
                dof_width = 6
            elif joint_type == int(mj.mjtJoint.mjJNT_BALL):
                dof_width = 3
            else:
                dof_width = 1
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            dof_indices.extend(range(dof_adr, dof_adr + dof_width))
        return sorted(set(dof_indices))

    def _setup_grouped_ik(self, task1_frame_names: list[str], task2_frame_names: list[str]) -> None:
        joint_groups = self.joint_groups if isinstance(self.joint_groups, dict) else {}
        task_groups = self.task_groups if isinstance(self.task_groups, dict) else {}
        if DEFAULT_IK_GROUP in joint_groups or DEFAULT_IK_GROUP in task_groups:
            raise ValueError(f"'{DEFAULT_IK_GROUP}' is reserved; omit the group to use the default assignment")

        has_group_config = bool(joint_groups) or bool(task_groups)
        if has_group_config and set(joint_groups) != set(task_groups):
            raise ValueError(
                "retarget joint_groups and task_groups must define the same group names when grouping is enabled; "
                f"joint_groups={sorted(joint_groups)}, task_groups={sorted(task_groups)}"
            )

        configured_task_frames = {str(name) for name in set(self.ik_match_table1) | set(self.ik_match_table2)}
        active_task_frames = set(task1_frame_names) | set(task2_frame_names)
        frame_group: dict[str, str] = {}
        for group_name, frame_names in task_groups.items():
            group = str(group_name)
            for frame_name_raw in frame_names:
                frame_name = str(frame_name_raw)
                if frame_name in frame_group:
                    raise ValueError(
                        f"Frame '{frame_name}' appears in multiple retarget task_groups: "
                        f"{frame_group[frame_name]} and {group}"
                    )
                frame_group[frame_name] = group

        unknown_task_frames = sorted(set(frame_group) - configured_task_frames)
        if unknown_task_frames:
            raise ValueError(f"Unknown frame names in retarget task_groups: {unknown_task_frames}")
        if has_group_config:
            missing_task_frames = sorted(active_task_frames - set(frame_group))
            if missing_task_frames:
                raise ValueError(f"Missing retarget task_groups assignment for active frames: {missing_task_frames}")

        all_dof_indices = set(range(self.model.nv))
        joint_owner: dict[str, str] = {}
        for group_name, joint_names in joint_groups.items():
            group = str(group_name)
            for joint_name_raw in joint_names:
                joint_name = str(joint_name_raw)
                if joint_name in joint_owner:
                    raise ValueError(
                        f"Joint '{joint_name}' appears in multiple retarget joint_groups: "
                        f"{joint_owner[joint_name]} and {group}"
                    )
                joint_owner[joint_name] = group
        dof_indices_by_group = {
            str(group_name): set(self._joint_names_to_dof_indices(tuple(str(name) for name in joint_names)))
            for group_name, joint_names in joint_groups.items()
        }
        explicit_dofs = set().union(*dof_indices_by_group.values()) if dof_indices_by_group else set()
        if has_group_config:
            missing_dofs = sorted(all_dof_indices - explicit_dofs)
            if missing_dofs:
                raise ValueError(f"Missing retarget joint_groups coverage for DOF indices: {missing_dofs}")
        else:
            dof_indices_by_group[DEFAULT_IK_GROUP] = set(all_dof_indices)

        group_order: list[str] = [str(group_name) for group_name in task_groups] if has_group_config else [DEFAULT_IK_GROUP]

        self.tasks1_by_group = {group: [] for group in group_order}
        self.tasks2_by_group = {group: [] for group in group_order}
        for task, frame_name in zip(self.tasks1, task1_frame_names, strict=True):
            group = frame_group[frame_name] if has_group_config else DEFAULT_IK_GROUP
            self.tasks1_by_group[group].append(task)
        for task, frame_name in zip(self.tasks2, task2_frame_names, strict=True):
            group = frame_group[frame_name] if has_group_config else DEFAULT_IK_GROUP
            self.tasks2_by_group[group].append(task)

        self.constraints_by_group = {}
        for group in group_order:
            allowed_dofs = dof_indices_by_group[group]
            if not allowed_dofs and (self.tasks1_by_group.get(group) or self.tasks2_by_group.get(group)):
                raise ValueError(f"Retarget IK group '{group}' has tasks but no movable DOFs")
            frozen_dofs = sorted(all_dof_indices - allowed_dofs)
            self.constraints_by_group[group] = (
                [mink.DofFreezingTask(self.model, frozen_dofs)] if frozen_dofs else None
            )

        self.ik_group_order = [
            group for group in group_order
            if self.tasks1_by_group.get(group) or self.tasks2_by_group.get(group)
        ]
        self.use_phased_ik = len(self.ik_group_order) > 1 or any(
            self.constraints_by_group.get(group) for group in self.ik_group_order
        )

    @property
    def human_body_names(self) -> tuple[str, ...]:
        return self._scale_body_names

    @property
    def human_positions(self) -> np.ndarray | None:
        if self._scaled_human_pos is None:
            return None
        return self._scaled_human_pos.astype(np.float32, copy=True)

    @property
    def human_rotations_wxyz(self) -> np.ndarray | None:
        if self._scaled_human_rot is None:
            return None
        return self._scaled_human_rot.astype(np.float32, copy=True)

    def _update_targets_from_selected_arrays(
        self,
        body_pos: np.ndarray,
        body_rot: np.ndarray,
        offset_to_ground: bool = False,
        align_body_min_z: bool = False,
        body_min_target_z: float = TELEOP_HEIGHT_ALIGNMENT_TARGET_Z,
    ) -> None:
        body_pos_arr = np.asarray(body_pos, dtype=np.float64)
        body_rot_arr = np.asarray(body_rot, dtype=np.float64)
        if body_pos_arr.shape != (len(self._scale_body_names), 3):
            raise ValueError(f"body_pos must have shape {(len(self._scale_body_names), 3)}, got {body_pos_arr.shape}")
        if body_rot_arr.shape != (len(self._scale_body_names), 4):
            raise ValueError(f"body_rot must have shape {(len(self._scale_body_names), 4)}, got {body_rot_arr.shape}")

        root_pos = body_pos_arr[self._root_body_idx]
        scaled_pos = np.empty_like(body_pos_arr)
        scaled_root_pos = root_pos * self._human_scale_factors[self._root_body_idx]
        scaled_pos[self._root_body_idx] = scaled_root_pos
        if self._non_root_body_indices.size > 0:
            non_root_pos = body_pos_arr[self._non_root_body_indices]
            non_root_scale = self._human_scale_factors[self._non_root_body_indices, None]
            scaled_pos[self._non_root_body_indices] = (non_root_pos - root_pos) * non_root_scale + scaled_root_pos
        if self._upper_body_scale_calibrated:
            for segment, scale, position_offset in zip(
                self._upper_body_scale_segments,
                self._upper_body_segment_scales,
                self._upper_body_segment_position_offsets,
                strict=True,
            ):
                human_delta = body_pos_arr[segment.human_body_idx] - body_pos_arr[segment.human_parent_idx]
                scaled_delta = human_delta * float(scale)
                if segment.calibrate_position_offset:
                    aligned_parent_rot = quat_normalize_np(
                        quat_mul_np(
                            body_rot_arr[segment.human_parent_idx],
                            self._scale_body_rot_offsets_quat[segment.human_parent_idx],
                        )
                    )
                    scaled_delta = scaled_delta + quat_apply_np(
                        aligned_parent_rot,
                        position_offset,
                        normalize=True,
                    )
                scaled_pos[segment.human_body_idx] = scaled_pos[segment.human_parent_idx] + scaled_delta

        updated_rot = quat_normalize_np(quat_mul_np(body_rot_arr, self._scale_body_rot_offsets_quat))
        global_pos_offset = quat_apply_np(
            updated_rot,
            self._scale_body_pos_offsets,
            normalize=False,
        )
        offset_pos = scaled_pos + global_pos_offset

        if self.ground_offset != 0.0:
            offset_pos[:, 2] -= self.ground_offset

        if offset_to_ground and np.any(self._foot_body_mask):
            lowest_pos = float(np.min(offset_pos[self._foot_body_mask, 2]))
            offset_pos[:, 2] = offset_pos[:, 2] - lowest_pos + 0.1
        if align_body_min_z:
            lowest_pos = float(np.min(offset_pos[:, 2]))
            if np.isfinite(lowest_pos):
                offset_pos[:, 2] += float(body_min_target_z) - lowest_pos

        self._scaled_human_pos = offset_pos
        self._scaled_human_rot = updated_rot

        if self.use_ik_match_table1:
            task1_pos = offset_pos[self._task1_body_indices]
            task1_rot = updated_rot[self._task1_body_indices]
            for task, pos, rot in zip(self.tasks1, task1_pos, task1_rot, strict=True):
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            task2_pos = offset_pos[self._task2_body_indices]
            task2_rot = updated_rot[self._task2_body_indices]
            for task, pos, rot in zip(self.tasks2, task2_pos, task2_rot, strict=True):
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def retarget_from_full_arrays(
        self,
        body_pos: np.ndarray,
        body_rot: np.ndarray,
        offset_to_ground: bool = False,
        align_body_min_z: bool = False,
        body_min_target_z: float = TELEOP_HEIGHT_ALIGNMENT_TARGET_Z,
    ) -> np.ndarray:
        body_pos_arr = np.asarray(body_pos, dtype=np.float64)
        body_rot_arr = np.asarray(body_rot, dtype=np.float64)
        if body_pos_arr.shape[0] < len(XR_BODY_JOINT_NAMES) or body_rot_arr.shape[0] < len(XR_BODY_JOINT_NAMES):
            raise ValueError("Full XR body arrays must contain every XR_BODY_JOINT_NAMES entry")
        self._update_targets_from_selected_arrays(
            body_pos_arr[self._scale_body_xr_indices],
            body_rot_arr[self._scale_body_xr_indices],
            offset_to_ground=offset_to_ground,
            align_body_min_z=align_body_min_z,
            body_min_target_z=body_min_target_z,
        )
        return self._solve_ik()

    def retarget(
        self,
        human_data: dict[str, Any],
        offset_to_ground: bool = False,
        align_body_min_z: bool = False,
        body_min_target_z: float = TELEOP_HEIGHT_ALIGNMENT_TARGET_Z,
    ) -> np.ndarray:
        selected_pos = np.empty((len(self._scale_body_names), 3), dtype=np.float64)
        selected_rot = np.empty((len(self._scale_body_names), 4), dtype=np.float64)
        for idx, body_name in enumerate(self._scale_body_names):
            pos, rot = human_data[body_name]
            selected_pos[idx] = np.asarray(pos, dtype=np.float64)
            selected_rot[idx] = np.asarray(rot, dtype=np.float64)
        self._update_targets_from_selected_arrays(
            selected_pos,
            selected_rot,
            offset_to_ground=offset_to_ground,
            align_body_min_z=align_body_min_z,
            body_min_target_z=body_min_target_z,
        )
        return self._solve_ik()

    def _solve_ik(self) -> np.ndarray:
        configuration = self.configuration

        for group in self.ik_group_order:
            constraints = self.constraints_by_group.get(group)
            if self.use_ik_match_table1:
                self._solve_task_set(self.tasks1_by_group.get(group, []), constraints)
            if self.use_ik_match_table2:
                self._solve_task_set(self.tasks2_by_group.get(group, []), constraints)

        return configuration.data.qpos.copy()

    def _solve_task_set(
        self,
        tasks: list[mink.FrameTask],
        constraints: list[mink.Task] | None = None,
    ) -> None:
        if not tasks:
            return
        configuration = self.configuration
        dt = configuration.model.opt.timestep
        curr_error = self._task_error(tasks)
        vel = mink.solve_ik(
            configuration,
            tasks,
            dt,
            solver=self.solver,
            damping=self.damping,
            limits=self.ik_limits,
            constraints=constraints,
        )
        configuration.integrate_inplace(vel, dt)
        next_error = self._task_error(tasks)
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < self.max_iter:
            curr_error = next_error
            vel = mink.solve_ik(
                configuration,
                tasks,
                dt,
                solver=self.solver,
                damping=self.damping,
                limits=self.ik_limits,
                constraints=constraints,
            )
            configuration.integrate_inplace(vel, dt)
            next_error = self._task_error(tasks)
            num_iter += 1

    def _task_error(self, tasks: list[mink.FrameTask]) -> float:
        if not tasks:
            return 0.0
        return float(
            np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in tasks]))
        )

    def error1(self) -> float:
        return self._task_error(self.tasks1)

    def error2(self) -> float:
        return self._task_error(self.tasks2)


class XRobotRetargetWorkerRuntime:
    def __init__(self, worker_config: dict[str, Any]):
        self.qpos_size = int(worker_config["qpos_size"])
        self.retarget = XRobotRetargeter(
            XRobotRetargetConfig(
                target_robot=str(worker_config["target_robot"]),
                actual_human_height=float(worker_config["actual_human_height"]),
                max_iter=int(worker_config["max_iter"]),
            )
        )
        self.send_human_motion = bool(worker_config["send_human_motion"])
        self.enable_height_alignment = bool(worker_config.get("enable_height_alignment", True))
        self.height_alignment_xrobot_body_min_each_frame = bool(
            worker_config.get("height_alignment_xrobot_body_min_each_frame", False)
        )
        self.height_alignment_target_z = float(
            worker_config.get("height_alignment_target_z", TELEOP_HEIGHT_ALIGNMENT_TARGET_Z)
        )
        self.height_alignment_bootstrap_frames = int(
            worker_config.get("height_bootstrap_frames", TELEOP_HEIGHT_ALIGNMENT_BOOTSTRAP_FRAMES)
        )
        self.height_alignment_foot_body_names = tuple(
            str(name) for name in worker_config.get("height_alignment_foot_body_names", ())
        )
        self.height_alignment_baseline_z: float | None = None
        self.height_alignment_sample_count = 0
        self.height_alignment_body_ids = (
            self._resolve_height_alignment_body_ids()
            if self.enable_height_alignment and not self.height_alignment_xrobot_body_min_each_frame
            else np.empty((0,), dtype=np.int32)
        )
        self.rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        self.rotation_quat = rotmat_to_quat_np(self.rotation_matrix)
        self.rotation_quat_batch = np.broadcast_to(
            self.rotation_quat.reshape(1, 4),
            (len(XR_BODY_JOINT_NAMES), 4),
        )

    @property
    def human_body_names(self) -> tuple[str, ...]:
        return self.retarget.human_body_names

    @property
    def human_positions(self) -> np.ndarray | None:
        return self.retarget.human_positions

    @property
    def human_rotations_wxyz(self) -> np.ndarray | None:
        return self.retarget.human_rotations_wxyz

    def _body_poses_to_arrays(self, poses: Any) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            pose_arr = np.asarray(poses, dtype=np.float64)
            pose_arr = pose_arr[: len(XR_BODY_JOINT_NAMES), :7]
            if pose_arr.shape != (len(XR_BODY_JOINT_NAMES), 7):
                return None
        except Exception:
            return None

        body_pos = pose_arr[:, :3] @ self.rotation_matrix.T
        body_rot = quat_mul_np(self.rotation_quat_batch, quat_xyzw_to_wxyz_np(pose_arr[:, 3:7]))
        return body_pos, body_rot

    def _packet_requests_calibration(self, packet: dict[str, Any]) -> bool:
        return bool(packet.get("calibration_requested", False))

    def _reset_height_alignment(self) -> None:
        self.height_alignment_baseline_z = None
        self.height_alignment_sample_count = 0

    def _resolve_height_alignment_body_ids(self) -> np.ndarray:
        if len(self.height_alignment_foot_body_names) != 2:
            raise ValueError("height_alignment_foot_body_names must contain exactly two body names")
        model = self.retarget.configuration.model
        body_ids = []
        for body_name in self.height_alignment_foot_body_names:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"Unknown height alignment foot body: {body_name}")
            body_ids.append(int(body_id))
        return np.asarray(body_ids, dtype=np.int32)

    def _get_current_foot_proxy_min_z(self) -> float | None:
        if self.height_alignment_body_ids.size == 0:
            return None
        data = self.retarget.configuration.data
        frame_min = float(np.min(data.xpos[self.height_alignment_body_ids, 2]))
        if not np.isfinite(frame_min):
            return None
        return frame_min

    def _apply_height_alignment(self, qpos: np.ndarray) -> tuple[np.ndarray, float]:
        qpos_adj = np.asarray(qpos, dtype=np.float32).copy()
        if not self.enable_height_alignment:
            return qpos_adj, 0.0

        foot_proxy_min_z = self._get_current_foot_proxy_min_z()
        if foot_proxy_min_z is None:
            return qpos_adj, 0.0

        self.height_alignment_sample_count += 1
        if self.height_alignment_baseline_z is None:
            self.height_alignment_baseline_z = float(foot_proxy_min_z)
        elif self.height_alignment_sample_count <= self.height_alignment_bootstrap_frames:
            self.height_alignment_baseline_z = min(self.height_alignment_baseline_z, float(foot_proxy_min_z))

        baseline = self.height_alignment_baseline_z
        if baseline is None or not np.isfinite(baseline):
            return qpos_adj, 0.0

        z_shift = float(self.height_alignment_target_z - baseline)
        qpos_adj[2] += z_shift
        return qpos_adj, z_shift

    def _copy_human_arrays(
        self,
        *,
        z_shift: float = 0.0,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        human_positions = self.retarget.human_positions
        human_rotations_wxyz = self.retarget.human_rotations_wxyz
        if human_positions is None or human_rotations_wxyz is None:
            return None, None
        if z_shift != 0.0:
            human_positions = human_positions.copy()
            human_positions[:, 2] += float(z_shift)
        return human_positions, human_rotations_wxyz

    def process_packet(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        body_arrays = self._body_poses_to_arrays(packet.get("poses"))
        if body_arrays is None:
            return None
        body_pos, body_rot = body_arrays
        if self._packet_requests_calibration(packet):
            self._reset_height_alignment()
            if self.retarget.upper_body_scale_enabled:
                calibrated = self.retarget.calibrate_upper_body_scale(body_pos, body_rot)
                state = "updated" if calibrated else "skipped"
                print(f"[Worker] upper_body_scale calibration {state}")

        qpos_curr = self.retarget.retarget_from_full_arrays(
            body_pos,
            body_rot,
            offset_to_ground=False,
            align_body_min_z=self.enable_height_alignment
            and self.height_alignment_xrobot_body_min_each_frame,
            # XRobot body-min alignment normalizes the input human targets to
            # ground z=0; height_alignment_target_z is for robot foot-proxy mode.
            body_min_target_z=TELEOP_HEIGHT_ALIGNMENT_TARGET_Z,
        )
        if qpos_curr is None:
            return None

        qpos_curr = np.asarray(qpos_curr, dtype=np.float32).reshape(-1)
        if qpos_curr.shape[0] < self.qpos_size:
            raise ValueError(f"retarget qpos too short: {qpos_curr.shape[0]}")
        if self.enable_height_alignment and self.height_alignment_xrobot_body_min_each_frame:
            qpos_curr = qpos_curr[: self.qpos_size]
            z_shift = 0.0
        else:
            qpos_curr, z_shift = self._apply_height_alignment(qpos_curr[: self.qpos_size])
        human_positions = None
        human_rotations_wxyz = None
        if self.send_human_motion:
            human_positions, human_rotations_wxyz = self._copy_human_arrays(z_shift=z_shift)

        return {
            "type": "retarget_result",
            "seq": int(packet["seq"]),
            "recv_ns": int(packet["recv_ns"]),
            "qpos": qpos_curr.astype(np.float32, copy=True),
            "human_positions": human_positions,
            "human_rotations_wxyz": human_rotations_wxyz,
        }
