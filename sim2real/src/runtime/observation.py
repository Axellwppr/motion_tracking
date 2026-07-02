import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING

from runtime.math_utils import _clamp_indices, _quat_apply_inv

if TYPE_CHECKING:
    from runtime.policy import Policy


def _require_cfg(policy, key: str):
    cfg = getattr(policy, "config", None)
    if cfg is None or not hasattr(cfg, key):
        raise KeyError(f"Missing required config key '{key}' for observation setup.")
    return getattr(cfg, key)


def _policy_future_steps(policy) -> np.ndarray:
    steps = np.asarray(_require_cfg(policy, "future_steps"), dtype=np.int32).reshape(-1)
    if steps.size == 0:
        raise ValueError("future_steps must not be empty.")
    return steps


class BaseObs:
    @property
    def size(self) -> int: ...
    def update(self): ...
    def compute(self) -> np.ndarray: ...

class TrackingCommandObsRaw:
    def __init__(self, ctrl, policy):
        self.ctrl = ctrl
        self.policy = policy

        self.future_steps = _policy_future_steps(policy)

    @property
    def size(self) -> int:
        n_fut = len(self.future_steps)
        return (n_fut - 1) * 3 + n_fut * 6
        # return (n_fut - 1) * 3 + n_fut * 3

    def reset(self):
        pass

    def update(self):
        pass

    def compute(self) -> np.ndarray:
        if (self.policy.ref_joint_pos is None or
            self.policy.ref_root_quat is None or
            self.policy.ref_root_pos is None):
            raise ValueError("Ref data not available yet.")

        base = self.policy.ref_idx
        T = self.policy.ref_len
        fut_idx = _clamp_indices(base + self.future_steps, T)

        root_pos_w = self.policy.ref_root_pos[fut_idx].copy()
        root_quat_w = self.policy.ref_root_quat[fut_idx].copy()

        pos_diff_w = root_pos_w[1:] - root_pos_w[0:1]
        pos_diff_b = _quat_apply_inv(root_quat_w[0], pos_diff_w)

        q_cur = self.ctrl.quat
        r_cur = R.from_quat(q_cur, scalar_first=True)
        r_ref = R.from_quat(root_quat_w, scalar_first=True)
        rel_rot = (r_cur.inv() * r_ref).as_matrix()
        if rel_rot.ndim == 2:
            rel_rot = rel_rot[None, ...]
        rot6d = rel_rot[:, :, :2].transpose(0, 2, 1).reshape(-1).astype(np.float32)
        # rel_rot = (r_cur.inv() * r_ref).as_rotvec().astype(np.float32)

        obs = np.concatenate(
            [
                pos_diff_b.reshape(-1),
                rot6d.reshape(-1),
                # rel_rot.reshape(-1),
            ],
            axis=-1,
        )
        return obs.astype(np.float32)

class TargetRootZObs:
    def __init__(self, policy):
        self.policy = policy
        self.future_steps = _policy_future_steps(policy)

    @property
    def size(self) -> int:
        return len(self.future_steps)

    def reset(self):
        pass

    def update(self):
        pass

    def compute(self) -> np.ndarray:
        if self.policy.ref_root_pos is None:
            raise ValueError("Ref data not available yet.")
        base = self.policy.ref_idx
        T = self.policy.ref_len
        fut_idx = _clamp_indices(base + self.future_steps, T)
        root_pos_w = self.policy.ref_root_pos[fut_idx]
        return root_pos_w[:, 2].astype(np.float32)

class TargetJointPosObs:
    def __init__(self, policy):
        self.policy = policy
        self.future_steps = _policy_future_steps(policy)

    @property
    def size(self) -> int:
        n_j = getattr(self.policy, "n_joints", 0)
        return len(self.future_steps) * n_j * 2

    def reset(self):
        pass

    def update(self):
        pass

    def compute(self) -> np.ndarray:
        if self.policy.ref_joint_pos is None:
            raise ValueError("Ref data not available yet.")
        base = self.policy.ref_idx
        T = self.policy.ref_len
        fut_idx = _clamp_indices(base + self.future_steps, T)
        tgt_joints = self.policy.ref_joint_pos[fut_idx]
        cur_joints = self.policy.controller.qj.astype(np.float32).reshape(1, -1)
        tgt_minus_cur = tgt_joints - cur_joints
        return np.concatenate(
            [
                tgt_joints.reshape(-1),
                tgt_minus_cur.reshape(-1),
            ],
            axis=-1,
        ).astype(np.float32)

class TargetProjectedGravityBObs:
    def __init__(self, policy):
        self.policy = policy
        self.future_steps = _policy_future_steps(policy)

    @property
    def size(self) -> int:
        return len(self.future_steps) * 3

    def reset(self):
        pass

    def update(self):
        pass

    def compute(self) -> np.ndarray:
        if (hasattr(self.policy, "ref_root_quat") and self.policy.ref_root_quat is None) and \
           (hasattr(self.policy, "ref_root_quat_rp") and self.policy.ref_root_quat_rp is None):
            raise ValueError("Ref data not available yet.")
        base = self.policy.ref_idx
        T = self.policy.ref_len
        fut_idx = _clamp_indices(base + self.future_steps, T)
        if hasattr(self.policy, "ref_root_quat"):
            root_quat_w = self.policy.ref_root_quat[fut_idx]
        elif hasattr(self.policy, "ref_root_quat_rp"):
            root_quat_w = self.policy.ref_root_quat_rp[fut_idx]
        g_world = np.array([0., 0., -1.], dtype=np.float32).reshape(1, 3)
        g_local = _quat_apply_inv(root_quat_w, g_world)
        return g_local.reshape(-1).astype(np.float32)


class RootAngVelBHistory(BaseObs):
    def __init__(self, ctrl, policy):
        self.ctrl = ctrl
        self.history_steps = list(_require_cfg(policy, "root_angvel_history_steps"))
        self.max_step = max(self.history_steps)
        self.hist = np.zeros((self.max_step + 1, 3), dtype=np.float32)

    @property
    def size(self):
        return len(self.history_steps) * 3

    def reset(self):
        self.hist[:] = 0.0

    def update(self):
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = self.ctrl.gyro.copy()

    def compute(self):
        return self.hist[self.history_steps].reshape(-1)

class RootLinAccBHistory(BaseObs):
    def __init__(self, ctrl, policy):
        self.ctrl = ctrl
        self.history_steps = list(_require_cfg(policy, "root_linacc_history_steps"))
        self.max_step = max(self.history_steps)
        self.hist = np.zeros((self.max_step + 1, 3), dtype=np.float32)

    @property
    def size(self):
        return len(self.history_steps) * 3

    def reset(self):
        self.hist[:] = 0.0

    def update(self):
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = self.ctrl.linacc.copy()

    def compute(self):
        return self.hist[self.history_steps].reshape(-1)

class ProjectedGravityBHistory(BaseObs):
    def __init__(self, ctrl, policy):
        self.ctrl = ctrl
        self.history_steps = list(_require_cfg(policy, "projected_gravity_history_steps"))
        self.max_step = max(self.history_steps)
        self.hist = np.zeros((self.max_step + 1, 3), dtype=np.float32)

    @property
    def size(self):
        return len(self.history_steps) * 3

    def reset(self):
        self.hist[:] = 0.0

    def update(self):
        self.hist = np.roll(self.hist, 1, axis=0)
        g_world = np.array([0., 0., -1.], dtype=np.float32)
        g_body = _quat_apply_inv(self.ctrl.quat, g_world)
        g_body /= np.linalg.norm(g_body) + 1e-8
        self.hist[0] = g_body

    def compute(self):
        return self.hist[self.history_steps].reshape(-1)


class JointPos(BaseObs):
    def __init__(self, ctrl, policy):
        self.ctrl = ctrl
        self.pos_steps = list(_require_cfg(policy, "joint_pos_history_steps"))
        
        self.num_joints = len(ctrl.config.policy_joint_names)
        self.max_step = max(self.pos_steps)
        self.hist = np.zeros((self.max_step + 1, self.num_joints), dtype=np.float32)

    @property
    def size(self):
        return len(self.pos_steps) * self.num_joints
    
    def reset(self):
        self.hist[:] = 0.0

    def update(self):
        self.hist = np.roll(self.hist, 1, axis=0)
        cur = self.ctrl.qj.copy()
        self.hist[0] = cur

    def compute(self):
        pos = self.hist[self.pos_steps].reshape(-1)
        return pos


class JointVel(BaseObs):
    def __init__(self, ctrl, policy):
        self.ctrl = ctrl
        self.vel_steps = list(_require_cfg(policy, "joint_vel_history_steps"))

        self.num_joints = len(ctrl.config.policy_joint_names)
        self.max_step = max(self.vel_steps)
        self.hist = np.zeros((self.max_step + 1, self.num_joints), dtype=np.float32)

    @property
    def size(self):
        return len(self.vel_steps) * self.num_joints

    def reset(self):
        self.hist[:] = 0.0

    def update(self):
        self.hist = np.roll(self.hist, 1, axis=0)
        cur = self.ctrl.dqj.copy()
        self.hist[0] = cur

    def compute(self):
        vel = self.hist[self.vel_steps].reshape(-1)
        return vel


class PrevActions(BaseObs):
    def __init__(self, policy: "Policy", old_style=False):
        self.policy = policy
        self.steps = int(_require_cfg(policy, "prev_action_steps"))
        self.action_dim = self.policy.last_action.shape[0]
        self.buf = np.zeros((self.steps, self.action_dim), dtype=np.float32)
        self.old_style = old_style

    @property
    def size(self):
        return self.action_dim * self.steps

    def reset(self):
        self.buf[:] = 0.0

    def update(self):
        self.buf = np.roll(self.buf, 1, axis=0)
        if self.old_style:
            self.buf[0, :] = self.policy.applied_action
        else:
            self.buf[0, :] = self.policy.last_action

    def compute(self):
        return self.buf.reshape(-1)

class BootIndicator(BaseObs):
    def __init__(self, policy):
        self.max_value = int(getattr(policy.config, "boot_indicator_max", 25))
        if self.max_value <= 0:
            raise ValueError(f"boot_indicator_max must be positive, got {self.max_value}")
        self.value = 0

    @property
    def size(self):
        return 1

    def reset(self):
        self.value = self.max_value

    def update(self):
        self.value = max(self.value - 1, 0)

    def compute(self):
        return np.array([self.value / self.max_value], dtype=np.float32)


class ComplianceFlagObs(BaseObs):
    def __init__(self, policy):
        self.policy = policy
        self.force_threshold = float(_require_cfg(policy, "compliance_flag_threshold"))
        self.kp = self.force_threshold / 0.05
        self.v = float(_require_cfg(policy, "compliance_flag_value"))

    @property
    def size(self):
        return 3

    def compute(self):
        return np.array([self.v, self.v * self.force_threshold, self.v * self.kp], dtype=np.float32)
