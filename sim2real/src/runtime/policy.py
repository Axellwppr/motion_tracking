import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
from common.joint_mapper import JointMapper
from common.utils import DictToClass
from runtime.motion_sources import MotionSourceBase, UDPMotionSource, VRMotionSource

def benchmark_onnx(module, sample_input, runs=100, warmup=10, desc=""):
    for _ in range(warmup):
        _ = module(sample_input)

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = module(sample_input)
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000.0)

    mean = statistics.mean(ts)
    stdev = statistics.pstdev(ts)
    p50 = np.percentile(ts, 50)
    p90 = np.percentile(ts, 90)
    p95 = np.percentile(ts, 95)
    p99 = np.percentile(ts, 99)

    print(f"[{desc}] runs={runs}, warmup={warmup}")
    print(f"mean={mean:.3f} ms, stdev={stdev:.3f} ms")
    print(f"p50={p50:.3f} ms, p90={p90:.3f} ms, p95={p95:.3f} ms, p99={p99:.3f} ms")
    return {"mean": mean, "stdev": stdev, "p50": p50, "p90": p90, "p95": p95, "p99": p99}


class ONNXModule:
    CPU_AFFINITY = (4, 5, 6, 7)
    CPU_THREADS = len(CPU_AFFINITY)

    @classmethod
    def _bind_process_to_first_cpus(cls) -> None:
        if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
            return
        try:
            allowed = sorted(os.sched_getaffinity(0))
            target = [cpu for cpu in cls.CPU_AFFINITY if cpu in allowed]
            if len(target) != len(cls.CPU_AFFINITY):
                print(
                    f"[ONNXModule] Requested CPUs {list(cls.CPU_AFFINITY)} but only "
                    f"{target} are available in current affinity mask {allowed}"
                )
            if len(target) >= 1:
                os.sched_setaffinity(0, set(target))
                print(f"[ONNXModule] Bound process affinity to CPUs {target}")
        except Exception as e:
            print(f"[ONNXModule] Failed to set process affinity: {e}")

    def __init__(self, path: str):
        self._bind_process_to_first_cpus()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.CPU_THREADS
        sess_options.inter_op_num_threads = 1
        # sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.ort_session = ort.InferenceSession(
            path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        meta_path = path.replace(".onnx", ".json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.in_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]]
        self.out_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]]

    def __call__(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        args = {
            inp.name: input[key]
            for inp, key in zip(self.ort_session.get_inputs(), self.in_keys)
            if key in input
        }
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.out_keys, outputs)}
        return outputs

# =========================================
# Policy Base
# =========================================
class Policy:
    FADE_OUT_DURATION = 2.0  # s

    def __init__(self, name: str, policy_cfg: DictToClass, controller):
        self.name = name
        self.controller = controller

        self.config = policy_cfg

        # Resolve policy path relative to repo root for robustness
        p = Path(policy_cfg.policy_path)
        cfg_dir = Path(getattr(policy_cfg, "_config_dir"))
        self.policy_path = str(p if p.is_absolute() else (cfg_dir / p))
        self.action_joint_names = list(policy_cfg.action_joint_names)
        self.action_scale = np.array(policy_cfg.action_scale, dtype=np.float32)
        self.action_clip = float(policy_cfg.action_clip)

        assert len(self.action_joint_names) == len(self.action_scale), (
            f"[{self.name}] action_joint_names ({len(self.action_joint_names)}) "
            f"!= action_scale ({len(self.action_scale)})"
        )

        self.module = ONNXModule(self.policy_path)

        self.mapper_action = JointMapper(
            self.action_joint_names,
            self.controller.config.policy_joint_names
        )
        map_info = self.mapper_action.get_mapping_info()
        print(f"[Policy:{self.name}] Action mapping: {map_info['mapped_joints']}/{map_info['from_space_size']} mapped")
        if map_info['unmapped_from_joints']:
            print(f"[Policy:{self.name}] Unmapped policy action joints: {map_info['unmapped_from_joints']}")
        if map_info['unmapped_to_joints']:
            print(f"[Policy:{self.name}] Unmapped controller joints: {map_info['unmapped_to_joints']}")

        self.policy_input: Optional[Dict[str, np.ndarray]] = None
        self.applied_action = np.zeros(len(self.action_joint_names), dtype=np.float32)
        self.last_action = np.zeros(len(self.action_joint_names), dtype=np.float32)

        self._fading_deadline: Optional[float] = None
        self._active: bool = False

        self.obs_modules = []
        self.num_obs = 0
        self._build_obs_modules()

        self.policy_input = {
            "policy": np.zeros((1, self.num_obs), dtype=np.float32),
            "is_init": np.ones((1,), dtype=bool)
        }
        input_shape = self.module.ort_session.get_inputs()[0].shape
        expected_obs_dim = input_shape[-1] if len(input_shape) > 0 else None
        if isinstance(expected_obs_dim, int) and expected_obs_dim != self.num_obs:
            raise ValueError(
                f"[Policy:{self.name}] Observation dim mismatch: built={self.num_obs}, "
                f"onnx expects {expected_obs_dim}. Please align tracking.yaml observation settings."
            )
        benchmark_onnx(self.module, self.policy_input, runs=100, warmup=200, desc="model@cuda")

    # -------- lifecycle ----------
    def fade_in(self):
        self.reset()
        self._active = True
        self._fading_deadline = None
        print(f"[Policy:{self.name}] fade_in()")

    def fade_out(self) -> float:
        self._fading_deadline = time.monotonic() + self.FADE_OUT_DURATION
        print(f"[Policy:{self.name}] fade_out() - continue until {self._fading_deadline:.3f}")
        return self._fading_deadline

    def is_fading(self) -> bool:
        return self._fading_deadline is not None

    def fading_done(self) -> bool:
        return self._fading_deadline is not None and time.monotonic() >= self._fading_deadline

    def deactivate(self):
        self._active = False
        self._fading_deadline = None
        print(f"[Policy:{self.name}] deactivated")

    # -------- abstract hooks ----------
    def _build_obs_modules(self):
        raise NotImplementedError

    def _reset_obs_modules(self):
        for m in self.obs_modules:
            if hasattr(m, "reset") and callable(m.reset):
                m.reset()

    def update_obs(self):
        obs_list = []
        for m in self.obs_modules:
            m.update()
            val = m.compute()
            obs_list.append(val)
        if self.policy_input is None:
            self.policy_input = {
                "policy": np.zeros((1, self.num_obs), dtype=np.float32),
                "is_init": np.ones((1,), dtype=bool),
            }
        self.policy_input["policy"][0, :] = np.concatenate(obs_list, axis=0)

    def compute_action(self) -> np.ndarray:
        try:
            out = self.module(self.policy_input)
        except Exception as e:
            print(f"[Policy:{self.name}] ONNX forward failed: {e}")
            return np.zeros(self.controller.dof_size, dtype=np.float32)

        if ("next", "adapt_hx") in out:
            self.policy_input["adapt_hx"][:] = out["next", "adapt_hx"]
        self.policy_input["is_init"][:] = False

        action = out["action"].copy()[0].astype(np.float32).clip(-self.action_clip, self.action_clip)
        self.last_action[:] = action
        self.applied_action[:] = action * self.action_scale

        mapped_action = self.mapper_action.map_action_from_to(self.applied_action)
        return mapped_action

    def reset(self):
        self.policy_input = None
        self.applied_action[:] = 0.0
        self.last_action[:] = 0.0
        self._reset_obs_modules()

    def post_step(self):
        """Hook called once after each policy inference/application step."""
        return

# =========================================
# Policy Subclasses
# =========================================
class TrackingPolicyRaw(Policy):
    @staticmethod
    def _parse_future_steps(policy_cfg: DictToClass):
        if not hasattr(policy_cfg, "future_steps"):
            raise KeyError("Missing required config key 'future_steps'.")
        future_steps = np.asarray(getattr(policy_cfg, "future_steps"), dtype=np.int32).reshape(-1)
        if future_steps.size == 0:
            raise ValueError("[TrackingPolicyRaw] future_steps must not be empty.")
        if int(future_steps[0]) != 0:
            raise ValueError(f"[TrackingPolicyRaw] future_steps[0] must be 0, got {future_steps.tolist()}")

        seen_negative = False
        for s in future_steps[1:]:
            if int(s) < 0:
                seen_negative = True
            elif seen_negative:
                raise ValueError(
                    "[TrackingPolicyRaw] future_steps format must be [0, ...positive/non-negative, ...negative]. "
                    f"Got: {future_steps.tolist()}"
                )
        return future_steps

    def __init__(self, name: str, policy_cfg: DictToClass, controller):
        self.controller = controller
        # ---- Config ---------------------------------------------------------
        self.body_name = "torso_link"
        self.transition_steps = int(getattr(policy_cfg, "transition_steps", 100))
        self.future_steps = self._parse_future_steps(policy_cfg)
        self.future_history_len = int(max(0, -int(self.future_steps.min())))
        configured_tail = int(getattr(policy_cfg, "switch_tail_keep_steps", self.future_history_len))
        # Keep enough old reference so negative future_steps can access valid history right after a motion switch.
        self.switch_tail_keep_steps = max(configured_tail, self.future_history_len)
        motion_source_cfg = getattr(policy_cfg, "motion_source")
        self.motion_source = str(motion_source_cfg["type"]).strip().lower()
        if self.motion_source not in ("udp", "vr"):
            raise ValueError(f"[TrackingPolicyRaw] motion_source must be 'udp' or 'vr', got '{self.motion_source}'")
        self.ref_max_len = int(getattr(policy_cfg, "ref_max_len", 2048))
        self.use_compliance_flag_obs = bool(getattr(policy_cfg, "use_compliance_flag_obs", True))

        self.dataset_joint_names = list(getattr(policy_cfg, "dataset_joint_names", []))
        if len(self.dataset_joint_names) == 0:
            raise ValueError(
                "[TrackingPolicyRaw] dataset_joint_names must be provided in tracking.yaml."
            )
        self.obs_joint_names = controller.config.policy_joint_names
        self.n_joints = len(self.obs_joint_names)

        # ---- Reference stream ----------------------------------------------
        self.ref_joint_pos: Optional[np.ndarray] = None  # (T_ref, J)
        self.ref_root_quat: Optional[np.ndarray] = None  # (T_ref, 4)
        self.ref_root_pos: Optional[np.ndarray] = None   # (T_ref, 3)

        # ---- Playback state ------------------------------------------------
        self.ref_idx: int = 0
        self.ref_len: int = 0
        self.current_name: str = "default"
        self.current_done: bool = True  # boot: default done

        self.source: MotionSourceBase
        if self.motion_source == "udp":
            self.source = UDPMotionSource(self, policy_cfg)
        else:
            self.source = VRMotionSource(self, policy_cfg)
        self.motions = self.source.motions

        super().__init__(name, policy_cfg, controller)
        self.init_count = 0

    def fade_in(self):
        super().fade_in()
        self.source.on_fade_in()

    def fade_out(self) -> float:
        self.source.on_fade_out()
        return super().fade_out()

    def deactivate(self):
        self.source.deactivate()
        self.ref_root_pos = None
        self.ref_root_quat = None
        self.ref_joint_pos = None
        super().deactivate()

    def _build_obs_modules(self):
        from runtime.observation import (
            TrackingCommandObsRaw,
            TargetRootZObs,
            TargetJointPosObs,
            TargetProjectedGravityBObs,
            RootAngVelBHistory,
            RootLinAccBHistory,
            ProjectedGravityBHistory,
            JointPos,
            JointVel,
            PrevActions,
            BootIndicator,
            ComplianceFlagObs,
        )
        self.obs_modules = [
            BootIndicator(self),
            TrackingCommandObsRaw(self.controller, self),
            TargetJointPosObs(self),
            TargetRootZObs(self),
            TargetProjectedGravityBObs(self),
            RootAngVelBHistory(self.controller, self),
            # RootLinAccBHistory(self.controller, self),
            ProjectedGravityBHistory(self.controller, self),
            JointPos(self.controller, self),
            JointVel(self.controller, self),
            PrevActions(self),
        ]
        if self.use_compliance_flag_obs:
            self.obs_modules.insert(2, ComplianceFlagObs(self))
        self.num_obs = sum(m.size for m in self.obs_modules)

    def request_motion(self, name: str) -> bool:
        request_fn = getattr(self.source, "request_motion", None)
        if callable(request_fn):
            return bool(request_fn(name))
        return False

    def update_obs(self):
        if self.ref_len > 0 and self.ref_idx < self.ref_len - 1:
            self.ref_idx += 1
            if self.ref_idx == self.ref_len - 1:
                self.current_done = True
        super().update_obs()

    def read_current_state(self) -> Dict[str, np.ndarray]:
        q_policy = self.controller.qj.copy().astype(np.float32)

        if self.ref_root_pos is not None:
            root_pos = self.ref_root_pos[self.ref_idx]
            root_quat = self.ref_root_quat[self.ref_idx]
        else:
            if hasattr(self, "motions") and "default" in self.motions:
                root_pos = self.motions["default"]["root_pos"][0].astype(np.float32, copy=True)
            else:
                root_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32)
            root_quat = self.controller.quat.copy()
        return {
            "joint_pos": q_policy,
            "root_pos": root_pos,
            "root_quat": root_quat,
        }

    def read_ref_tail_state(self) -> Dict[str, np.ndarray]:
        if (
            self.ref_joint_pos is not None
            and self.ref_root_quat is not None
            and self.ref_root_pos is not None
            and self.ref_len > 0
        ):
            return {
                "joint_pos": self.ref_joint_pos[self.ref_len - 1].astype(np.float32, copy=True),
                "root_pos": self.ref_root_pos[self.ref_len - 1].astype(np.float32, copy=True),
                "root_quat": self.ref_root_quat[self.ref_len - 1].astype(np.float32, copy=True),
            }
        return self.read_current_state()

    def append_ref_frames(self, frames: Dict[str, np.ndarray]) -> None:
        if frames is None:
            return

        j = np.asarray(frames["joint_pos"], dtype=np.float32)
        q = np.asarray(frames["root_quat"], dtype=np.float32)
        p = np.asarray(frames["root_pos"], dtype=np.float32)

        if j.ndim == 1:
            j = j.reshape(1, -1)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p = p.reshape(1, -1)

        if j.shape[0] != q.shape[0] or j.shape[0] != p.shape[0]:
            raise ValueError(f"Frame length mismatch: joint={j.shape}, quat={q.shape}, pos={p.shape}")
        if j.shape[0] == 0:
            return
        if j.shape[1] != self.n_joints:
            raise ValueError(f"Joint dim mismatch: got={j.shape[1]}, expected={self.n_joints}")
        if q.shape[1] != 4 or p.shape[1] != 3:
            raise ValueError(f"Root dim mismatch: quat={q.shape[1]}, pos={p.shape[1]}")

        if self.ref_joint_pos is None or self.ref_root_quat is None or self.ref_root_pos is None or self.ref_len <= 0:
            self.ref_joint_pos = j.copy()
            self.ref_root_quat = q.copy()
            self.ref_root_pos = p.copy()
        else:
            self.ref_joint_pos = np.concatenate([self.ref_joint_pos, j], axis=0)
            self.ref_root_quat = np.concatenate([self.ref_root_quat, q], axis=0)
            self.ref_root_pos = np.concatenate([self.ref_root_pos, p], axis=0)

        self.ref_len = int(self.ref_joint_pos.shape[0])
        self.current_done = (self.ref_idx >= self.ref_len - 1)
        self._trim_ref_prefix()

    def _trim_ref_prefix(self) -> None:
        if (
            self.ref_joint_pos is None
            or self.ref_root_quat is None
            or self.ref_root_pos is None
            or self.ref_len <= 0
        ):
            return

        keep_hist = max(self.future_history_len, self.switch_tail_keep_steps) + 2
        drop = max(0, int(self.ref_idx) - int(keep_hist))
        if self.ref_max_len > 0:
            overflow = max(0, int(self.ref_len) - int(self.ref_max_len))
            drop = max(drop, min(overflow, max(0, int(self.ref_idx) - int(keep_hist))))
        if drop <= 0:
            return

        self.ref_joint_pos = self.ref_joint_pos[drop:]
        self.ref_root_quat = self.ref_root_quat[drop:]
        self.ref_root_pos = self.ref_root_pos[drop:]
        self.ref_idx -= drop
        self.ref_len = int(self.ref_joint_pos.shape[0])
        self.current_done = (self.ref_idx >= self.ref_len - 1)

    def post_step(self):
        self.source.post_step()
