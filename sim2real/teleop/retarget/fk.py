from __future__ import annotations

import mujoco as mj
import numpy as np

from utils.math import quat_normalize_safe_np, quat_wxyz_to_xyzw_np, quat_xyzw_to_wxyz_np

from .params import resolve_robot_xml_path


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == int(mj.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mj.mjtJoint.mjJNT_BALL):
        return 4
    if joint_type in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
        return 1
    raise ValueError(f"Unsupported joint type: {joint_type}")


class LocalKinematicsModel:
    def __init__(self, target_robot: str) -> None:
        self.target_robot = str(target_robot).strip().lower()
        self.xml_file = str(resolve_robot_xml_path(self.target_robot))
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        self.data = mj.MjData(self.model)

        self._has_free_root = (
            self.model.njnt > 0
            and int(self.model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE)
            and int(self.model.jnt_qposadr[0]) == 0
        )
        self._qpos_template = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        if self._has_free_root and self._qpos_template.shape[0] >= 7:
            self._qpos_template[0:3] = 0.0
            self._qpos_template[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        joint_names: list[str] = []
        joint_qpos_indices: list[int] = []
        for joint_id in range(self.model.njnt):
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == int(mj.mjtJoint.mjJNT_FREE):
                continue
            qpos_width = _joint_qpos_width(joint_type)
            if qpos_width != 1:
                joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
                raise NotImplementedError(
                    f"LocalKinematicsModel only supports 1-DoF non-free joints; "
                    f"got joint '{joint_name}' with qpos width {qpos_width}"
                )
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None:
                raise ValueError(f"Failed to resolve joint name at index {joint_id}")
            joint_names.append(str(joint_name))
            joint_qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))

        body_ids = np.arange(1, self.model.nbody, dtype=np.int32)
        body_names: list[str] = []
        for body_id in body_ids:
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, int(body_id))
            if body_name is None:
                raise ValueError(f"Failed to resolve body name at index {body_id}")
            body_names.append(str(body_name))

        self.joint_names = tuple(joint_names)
        self.body_names = tuple(body_names)
        self._joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=np.int32)
        self._body_ids = body_ids

    def forward_kinematics(
        self,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        dof_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        root_pos_arr = np.asarray(root_pos, dtype=np.float64)
        root_rot_arr = np.asarray(root_rot, dtype=np.float64)
        dof_pos_arr = np.asarray(dof_pos, dtype=np.float64)

        if root_pos_arr.ndim == 1:
            root_pos_arr = root_pos_arr.reshape(1, 3)
        if root_rot_arr.ndim == 1:
            root_rot_arr = root_rot_arr.reshape(1, 4)
        if dof_pos_arr.ndim == 1:
            dof_pos_arr = dof_pos_arr.reshape(1, -1)

        frame_count = int(dof_pos_arr.shape[0])
        if root_pos_arr.shape != (frame_count, 3):
            raise ValueError(f"root_pos must have shape ({frame_count}, 3), got {root_pos_arr.shape}")
        if root_rot_arr.shape != (frame_count, 4):
            raise ValueError(f"root_rot must have shape ({frame_count}, 4), got {root_rot_arr.shape}")
        if dof_pos_arr.shape[1] != len(self.joint_names):
            raise ValueError(
                f"dof_pos must have {len(self.joint_names)} columns for '{self.target_robot}', "
                f"got {dof_pos_arr.shape[1]}"
            )

        root_rot_wxyz = quat_normalize_safe_np(quat_xyzw_to_wxyz_np(root_rot_arr))
        body_pos = np.empty((frame_count, len(self.body_names), 3), dtype=np.float32)
        body_rot = np.empty((frame_count, len(self.body_names), 4), dtype=np.float32)

        for frame_idx in range(frame_count):
            self.data.qpos[:] = self._qpos_template
            if self._has_free_root:
                self.data.qpos[0:3] = root_pos_arr[frame_idx]
                self.data.qpos[3:7] = root_rot_wxyz[frame_idx]
            self.data.qpos[self._joint_qpos_indices] = dof_pos_arr[frame_idx]
            mj.mj_forward(self.model, self.data)
            body_pos[frame_idx] = np.asarray(self.data.xpos[self._body_ids], dtype=np.float32)
            body_rot[frame_idx] = quat_wxyz_to_xyzw_np(
                np.asarray(self.data.xquat[self._body_ids], dtype=np.float64)
            ).astype(np.float32)

        return body_pos, body_rot
