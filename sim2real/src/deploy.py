import os
import sys
import yaml
from typing import Dict, Optional

import numpy as np

from common.udp_transport import UDPRobotHigh
from common.utils import DictToClass

from runtime.policy import Policy, TrackingPolicyRaw
from runtime.recording import PolicyRunRecorder
from pathlib import Path
from paths import SIM2REAL_ROOT, SUPPORTED_ROBOTS, controller_config_path, tracking_config_path

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def get_config(policy_cfg_path: str) -> DictToClass:
    policy_cfg_path = Path(policy_cfg_path)
    if not policy_cfg_path.is_absolute():
        policy_cfg_path = SIM2REAL_ROOT / policy_cfg_path
    with open(str(policy_cfg_path), 'r') as f:
        policy_cfg = DictToClass(yaml.load(f, Loader=yaml.FullLoader))
    policy_cfg._config_dir = str(policy_cfg_path.parent)
    return policy_cfg

class Controller:
    def __init__(self, args, ctrl_cfg):
        self.args = args
        self.config = ctrl_cfg
        self.control_dt = 1.0 / self.config.control_freq
        self.policy_joint_names = list(self.config.policy_joint_names)
        self.dof_size = len(self.policy_joint_names)

        self.qj = np.zeros(self.dof_size, dtype=np.float32)
        self.dqj = np.zeros(self.dof_size, dtype=np.float32)
        self.tau = np.zeros(self.dof_size, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)
        self.linacc = np.zeros(3, dtype=np.float32)

        self.default_qpos = np.array(self.config.default_qpos, dtype=np.float32)
        self.init_qpos = np.array(self.config.init_qpos, dtype=np.float32)
        self.kps = np.array(self.config.kps, dtype=np.float32)
        self.kds = np.array(self.config.kds, dtype=np.float32)

        self.counter = 0
        self.policy_step = 0
        self.is_alive = True

        self.transport = UDPRobotHigh(self.config.udp)
        self.cmd_q = self.init_qpos.copy()
        self.cmd_qd = np.zeros(self.dof_size, dtype=np.float32)
        self.cmd_kp = self.kps.copy()
        self.cmd_kd = self.kds.copy()
        self.cmd_enable = 0
        self.extra_command = dict(getattr(self.config, "extra_command", {}))
        self.hand_left = float(np.clip(float(self.extra_command.get("hand_left", 0.0)), 0.0, 1.0))
        self.hand_right = float(np.clip(float(self.extra_command.get("hand_right", 0.0)), 0.0, 1.0))
        if "hand_enable" in self.extra_command:
            self.set_hand_command(left=self.hand_left, right=self.hand_right, enable=int(self.extra_command.get("hand_enable", 1)))
        self.buttons = {
            "start": False,
            "stop": False,
            "A": False,
            "up": False,
            "down": False,
        }
        self.sticks = {
            "lx": 0.0,
            "ly": 0.0,
            "rx": 0.0,
            "ry": 0.0,
        }
        self.have_state = False
        self.last_state_seq: Optional[int] = None
        self.last_state_receive_time_ns: Optional[int] = None
        self.skipped_state_count = 0
        self._pending_state_arrival_ns: Optional[int] = None
        self._last_state_arrival_ns: Optional[int] = None
        self._state_interval_window_start_ns: Optional[int] = None
        self._state_interval_count = 0
        self._state_interval_sum_ms = 0.0
        self._state_interval_min_ms = float("inf")
        self._state_interval_max_ms = 0.0
        self._prev_buttons = None
        self.btn_rise = {
            "start": False,
            "stop": False,
            "A": False,
            "up": False,
            "down": False,
        }

        self.wait_for_low_state()

        tracking_cfg_path = tracking_config_path(self.args.robot, self.args.tracking_config)
        print(f"[Deploy] tracking config: {tracking_cfg_path}")
        self.policies = {
            "tracking": TrackingPolicyRaw("tracking", get_config(tracking_cfg_path), self),
        }
        self.current_policy: Optional[Policy] = None
        self.pending_policy: Optional[Policy] = None
        self.recorder: Optional[PolicyRunRecorder] = None

    def _consume_low_state(self, msg) -> bool:
        if msg is None:
            return False

        self.low_state = msg
        self.qj[:] = np.asarray(msg["q"], dtype=np.float32)
        self.dqj[:] = np.asarray(msg["dq"], dtype=np.float32)
        self.tau[:] = 0.0
        self.quat[:] = np.asarray(msg["quat_wxyz"], dtype=np.float32)
        self.gyro[:] = np.asarray(msg["gyro"], dtype=np.float32)
        self.linacc[:] = np.asarray(msg.get("linacc", np.zeros(3, dtype=np.float32)), dtype=np.float32)

        buttons = msg.get("buttons", {})
        self.buttons["start"] = bool(buttons.get("start", False))
        self.buttons["stop"] = bool(buttons.get("stop", False))
        self.buttons["A"] = bool(buttons.get("A", False))
        self.buttons["up"] = bool(buttons.get("up", False))
        self.buttons["down"] = bool(buttons.get("down", False))

        sticks = msg.get("sticks", {})
        self.sticks["lx"] = float(sticks.get("lx", 0.0))
        self.sticks["ly"] = float(sticks.get("ly", 0.0))
        self.sticks["rx"] = float(sticks.get("rx", 0.0))
        self.sticks["ry"] = float(sticks.get("ry", 0.0))

        state_receive_time_ns = msg.get("state_receive_time_ns", None)
        self.last_state_receive_time_ns = None if state_receive_time_ns is None else int(state_receive_time_ns)
        self.have_state = True
        return True

    def send_cmd(self):
        self.transport.send_command(
            q_des=self.cmd_q,
            qd_des=self.cmd_qd,
            kp=self.cmd_kp,
            kd=self.cmd_kd,
            enable=self.cmd_enable,
            extra_command=self.extra_command,
            state_receive_time_ns=self.last_state_receive_time_ns,
        )
        self._record_pending_state_arrival_interval()

    def set_hand_command(self, *, left: float, right: float, enable: int = 1):
        self.hand_left = float(np.clip(left, 0.0, 1.0))
        self.hand_right = float(np.clip(right, 0.0, 1.0))
        self.extra_command["hand_enable"] = int(enable)
        self.extra_command["hand_left"] = self.hand_left
        self.extra_command["hand_right"] = self.hand_right

    def set_zero_cmd(self):
        self.cmd_q[:] = 0.0
        self.cmd_qd[:] = 0.0
        self.cmd_kp[:] = 0.0
        self.cmd_kd[:] = 0.0
        self.cmd_enable = 0

    def set_damping_cmd(self):
        self.cmd_q[:] = 0.0
        self.cmd_qd[:] = 0.0
        self.cmd_kp[:] = 0.0
        self.cmd_kd[:] = 8.0
        self.cmd_enable = 0

    def wait_for_low_state(self):
        while not self.have_state:
            self.process_state(wait_next=True, timeout_s=1.0)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while not self.buttons["start"]:
            self.process_state(wait_next=True)
            self.set_zero_cmd()
            self.send_cmd()

    def move_to_default_qpos(self):
        print("Moving to init pos....")
        total_time = 2.0
        num_step = int(total_time / self.control_dt)

        init_dof_pos = self.qj.copy()

        for t in range(num_step):
            self.process_state(wait_next=True)
            alpha = t / num_step
            self.cmd_q[:] = init_dof_pos * (1 - alpha) + self.init_qpos * alpha
            self.cmd_qd[:] = 0.0
            self.cmd_kp[:] = self.kps
            self.cmd_kd[:] = self.kds
            self.send_cmd()

    def default_qpos_state(self):
        initial_policy: Optional[Policy] = None

        print("Press A to tracking policy...")

        while True:
            self.process_state(wait_next=True)

            self.cmd_q[:] = self.init_qpos
            self.cmd_qd[:] = 0.0
            self.cmd_kp[:] = self.kps
            self.cmd_kd[:] = self.kds
            self.send_cmd()

            if self.btn_rise["stop"]:
                raise KeyboardInterrupt

            if self.btn_rise["A"]:
                initial_policy = self.policies["tracking"]
                print("Initial policy: tracking")
                break

        self.current_policy = initial_policy
        self.current_policy.fade_in()
        self.cmd_enable = 1
        self.send_cmd()

    def process_state(self, *, wait_next: bool = False, timeout_s: float | None = None) -> bool:
        if wait_next:
            pkt = self.transport.read_next_state(after_seq=self.last_state_seq, timeout_s=timeout_s, with_meta=True)
        else:
            pkt = self.transport.read_latest_state(with_meta=True)
        if pkt is None:
            return False

        if self.last_state_seq is not None:
            seq_delta = int(pkt.seq) - int(self.last_state_seq)
            if seq_delta <= 0:
                return False
            if seq_delta > 1:
                skipped = seq_delta - 1
                self.skipped_state_count += skipped
                print(f"[Warning] skipped {skipped} bridge state packet(s): last={self.last_state_seq}, now={pkt.seq}")
        self.last_state_seq = int(pkt.seq)
        self._pending_state_arrival_ns = int(pkt.recv_time_ns)
        self._consume_low_state(pkt.data)

        now = self.buttons.copy()
        if self._prev_buttons is None:
            self._prev_buttons = now
            self.btn_rise = {k: False for k in now}
        else:
            self.btn_rise = {k: (not self._prev_buttons[k]) and now[k] for k in now}
            self._prev_buttons = now
        return True

    def _record_pending_state_arrival_interval(self) -> None:
        recv_time_ns = self._pending_state_arrival_ns
        self._pending_state_arrival_ns = None
        if recv_time_ns is None:
            return
        self._record_state_arrival_interval(recv_time_ns)

    def _record_state_arrival_interval(self, recv_time_ns: int) -> None:
        if recv_time_ns <= 0:
            return
        if self._state_interval_window_start_ns is None:
            self._state_interval_window_start_ns = recv_time_ns
        if self._last_state_arrival_ns is not None:
            interval_ms = (recv_time_ns - self._last_state_arrival_ns) * 1e-6
            if interval_ms >= 0.0:
                self._state_interval_count += 1
                self._state_interval_sum_ms += interval_ms
                self._state_interval_min_ms = min(self._state_interval_min_ms, interval_ms)
                self._state_interval_max_ms = max(self._state_interval_max_ms, interval_ms)
        self._last_state_arrival_ns = recv_time_ns

        if recv_time_ns - self._state_interval_window_start_ns < 1_000_000_000:
            return

        if self._state_interval_count > 0:
            mean_ms = self._state_interval_sum_ms / self._state_interval_count
            print(
                "[Deploy] state_interval_ms "
                f"mean={mean_ms:.3f} min={self._state_interval_min_ms:.3f} "
                f"max={self._state_interval_max_ms:.3f} n={self._state_interval_count}"
            )
        else:
            print("[Deploy] state_interval_ms n/a")

        self._state_interval_window_start_ns = recv_time_ns
        self._state_interval_count = 0
        self._state_interval_sum_ms = 0.0
        self._state_interval_min_ms = float("inf")
        self._state_interval_max_ms = 0.0

    def _apply_action(self, action_delta: np.ndarray):
        if action_delta is None or not np.all(np.isfinite(action_delta)):
            print("[Controller] action invalid; hold init PD")
            raise KeyboardInterrupt
        else:
            desired = self.default_qpos + action_delta
            target = desired

        self.cmd_q[:] = target
        self.cmd_qd[:] = 0.0
        self.cmd_kp[:] = self.kps
        self.cmd_kd[:] = self.kds
        self.cmd_enable = 1

    def _start_policy_recorder(self):
        if bool(getattr(self.args, "no_record", False)):
            return
        if self.recorder is not None:
            return
        if self.current_policy is None:
            raise RuntimeError("Cannot start recorder without an active policy")
        self.recorder = PolicyRunRecorder(
            robot=self.args.robot,
            policy=self.current_policy,
            joint_names=self.policy_joint_names,
            control_dt=self.control_dt,
            output_root=getattr(self.args, "record_dir", None),
        )
        print(f"[PolicyRunRecorder] recording={self.recorder.output_path}")

    def _save_policy_recorder(self):
        if self.recorder is not None:
            self.recorder.save()

    def run(self):
        print("Running high level...")
        self._start_policy_recorder()

        while True:
            if not self.process_state(wait_next=True, timeout_s=1.0):
                print("[Warning] no bridge state packet for 1s")
                continue

            if self.btn_rise["stop"]:
                break

            self.current_policy.update_obs()
            action = self.current_policy.compute_action()
            self._apply_action(action)
            self.send_cmd()

            self.current_policy.post_step()
            if self.recorder is not None:
                self.recorder.record_step(self, action)
            self.policy_step += 1

    def close(self):
        print("Closing...")
        self.is_alive = False
        self._save_policy_recorder()
        self.transport.close()
        sys.exit(0)

import traceback

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=list(SUPPORTED_ROBOTS), default="g1")
    parser.add_argument(
        "--tracking-config",
        default="tracking.yaml",
        help=(
            "Tracking YAML to load. Relative names resolve under config/<robot>/; "
            "use tracking_compliance.yaml for the G1 compliance policy."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Root directory for policy run logs. Defaults to assets/policy_logs.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable policy run recording.",
    )
    args = parser.parse_args()

    controller = Controller(args, get_config(controller_config_path(args.robot)))

    controller.zero_torque_state()
    controller.move_to_default_qpos()
    try:
        controller.default_qpos_state()
        controller.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"An exception occurred: {e}")
        traceback.print_exc()
    finally:
        controller.set_damping_cmd()
        controller.send_cmd()
        controller.close()
