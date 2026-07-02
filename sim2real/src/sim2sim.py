import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from sshkeyboard import listen_keyboard, stop_listening

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.joint_mapper import JointMapper
from common.udp_latest import LatestPacket
from common.udp_transport import UDPRobotLow
from common.utils import DictToClass, Timer
from paths import SUPPORTED_ROBOTS, bridge_config_path

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

Keyboard2Button = {
    "a": "A",
    "s": "start",
    "x": "stop",
    "u": "up",
    "d": "down",
}
BUTTON_KEYS = ("start", "stop", "A", "up", "down")
STICK_KEYS = ("lx", "ly", "rx", "ry")
_MISSING = object()


def _cfg_value(data, name: str, path: str, default=_MISSING):
    if isinstance(data, dict):
        if name in data:
            return data[name]
    elif hasattr(data, name):
        return getattr(data, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"{path} is required")


def _as_vector(data, *, name: str, size: int | None = None, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(data, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if size is not None and arr.size != size:
        raise ValueError(f"{name} has size {arr.size}, expected {size}")
    return arr


class Sim2Sim:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.robot = str(args.robot).lower()
        self.log_prefix = f"[{self.robot.upper()}Sim2Sim]"

        freq_cfg = _cfg_value(config, "freq", "freq")
        self.low_level_freq = int(_cfg_value(freq_cfg, "physical_hz", "freq.physical_hz"))
        if self.low_level_freq <= 0:
            raise ValueError("freq.physical_hz must be positive")
        self.state_decimation = int(_cfg_value(freq_cfg, "state_decimation", "freq.state_decimation"))
        if self.state_decimation <= 0:
            raise ValueError("freq.state_decimation must be positive")
        self.state_freq = self.low_level_freq / self.state_decimation
        self.state_dt = 1.0 / self.state_freq
        self.low_level_dt = 1.0 / self.low_level_freq
        print(
            f"{self.log_prefix} freq: physical_hz={self.low_level_freq}, "
            f"state_decimation={self.state_decimation}, state_hz={self.state_freq:.3f}"
        )

        xml_candidate = Path(args.xml_path or _cfg_value(config, "xml_path", "xml_path"))
        if xml_candidate.is_absolute():
            model_path = str(xml_candidate)
        else:
            model_path = str((Path(config._config_dir) / xml_candidate).resolve())
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.low_level_dt
        self.data = mujoco.MjData(self.model)

        self.policy_joint_names = list(_cfg_value(config, "policy_joint_names", "policy_joint_names"))
        self.mujoco_joint_names = list(
            _cfg_value(config, "mujoco_joint_names", "mujoco_joint_names", self.policy_joint_names)
        )
        self.n_policy_joints = len(self.policy_joint_names)
        self.n_mujoco_joints = len(self.mujoco_joint_names)
        if self.model.nu != self.n_mujoco_joints:
            raise ValueError(f"model.nu={self.model.nu} != configured mujoco joints={self.n_mujoco_joints}")
        if len(self.model.actuator_ctrlrange) != self.n_mujoco_joints:
            raise ValueError("actuator ctrl range size does not match configured mujoco joints")

        self.policy_to_mujoco = JointMapper(self.policy_joint_names, self.mujoco_joint_names)
        mapping_info = self.policy_to_mujoco.get_mapping_info()
        print(
            f"{self.log_prefix} policy->mujoco mapping: "
            f"{mapping_info['mapped_joints']}/{mapping_info['from_space_size']} joints mapped"
        )
        if mapping_info["unmapped_from_joints"]:
            raise ValueError(f"Unmapped policy joints: {mapping_info['unmapped_from_joints']}")
        if mapping_info["unmapped_to_joints"]:
            raise ValueError(f"Unmapped MuJoCo joints: {mapping_info['unmapped_to_joints']}")

        self.ctrl_lower = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_upper = self.model.actuator_ctrlrange[:, 1]

        self.home_q_policy = _as_vector(
            _cfg_value(config, "home_q", "home_q"),
            name="home_q",
            size=self.n_policy_joints,
        )
        self.root_qpos_home = _as_vector(
            _cfg_value(config, "root_qpos_home", "root_qpos_home"),
            name="root_qpos_home",
            size=7,
        )
        self.root_qpos_control = _as_vector(
            _cfg_value(config, "root_qpos_control", "root_qpos_control"),
            name="root_qpos_control",
            size=7,
        )
        self.viewer_fps = int(_cfg_value(config, "viewer_fps", "viewer_fps", 10))
        self.max_external_force = float(_cfg_value(config, "max_external_force", "max_external_force", 30.0))

        self.data.qpos[:7] = self.root_qpos_home
        self.data.qpos[7:] = self._policy_to_mujoco(self.home_q_policy)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._ptargets_policy = self.home_q_policy.copy()
        self._kp_policy = np.zeros(self.n_policy_joints, dtype=np.float64)
        self._kd_policy = np.zeros(self.n_policy_joints, dtype=np.float64)
        self._have_command = False
        self._have_tracking_target = False
        self._buttons = {k: False for k in BUTTON_KEYS}

        self._cmd_lock = threading.Lock()
        self._button_lock = threading.Lock()
        self._sim_lock = threading.Lock()
        self._policy_delay_lock = threading.Lock()
        self._policy_delay_count = 0
        self._policy_delay_sum_ms = 0.0
        self._policy_delay_min_ms = float("inf")
        self._policy_delay_max_ms = 0.0

        self.transport = UDPRobotLow(config.udp, on_command_packet=self.cmd_sub_handler)

        self.keyboard_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={"on_press": self.on_press, "on_release": self.on_release},
            daemon=False,
        )
        self.is_alive = True
        self.policy_queried = False

        self.render_gui = bool(config.render_gui)
        self.viewer = None
        self._viewer_tick = 0
        self._physics_tick = 0
        self.viewer_decim = max(1, self.low_level_freq // max(1, self.viewer_fps))
        self.imu_lin_acc_adr, self.imu_lin_acc_dim = self._resolve_sensor_slice("imu_lin_acc")

        signal.signal(signal.SIGINT, self.close)

    def _policy_to_mujoco(self, values: np.ndarray, default_values: np.ndarray | None = None) -> np.ndarray:
        return self.policy_to_mujoco.map_action_from_to(values, default_values=default_values)

    def _mujoco_to_policy(self, values: np.ndarray) -> np.ndarray:
        return self.policy_to_mujoco.map_state_to_from(values)

    def _set_button(self, name: str, value: bool) -> None:
        with self._button_lock:
            self._buttons[name] = value

    def _buttons_snapshot(self):
        with self._button_lock:
            return dict(self._buttons)

    def on_press(self, key):
        print(f"Key pressed: {key}")
        btn = Keyboard2Button.get(key, None)
        if btn is None:
            return
        self._set_button(btn, True)

    def on_release(self, key):
        btn = Keyboard2Button.get(key, None)
        if btn is None:
            return
        time.sleep(0.1)
        self._set_button(btn, False)

    def cmd_sub_handler(self, packet: LatestPacket):
        payload = packet.data
        q_des = np.asarray(
            payload.get("q_des", np.zeros(self.n_policy_joints, dtype=np.float32)),
            dtype=np.float64,
        )
        kp = np.asarray(payload.get("kp", np.zeros(self.n_policy_joints, dtype=np.float32)), dtype=np.float64)
        kd = np.asarray(payload.get("kd", np.zeros(self.n_policy_joints, dtype=np.float32)), dtype=np.float64)
        enable = int(payload.get("enable", 0))
        if q_des.size != self.n_policy_joints or kp.size != self.n_policy_joints or kd.size != self.n_policy_joints:
            print(f"{self.log_prefix} Ignore UDP command with unexpected DOF size")
            return
        with self._cmd_lock:
            self._ptargets_policy[:] = q_des
            self._kp_policy[:] = kp
            self._kd_policy[:] = kd
        self._record_policy_delay(payload)
        self.policy_queried |= bool(enable)
        self._have_command = True

    def _record_policy_delay(self, payload):
        state_receive_time_ns = payload.get("state_receive_time_ns", None)
        if state_receive_time_ns is None:
            return
        try:
            delay_ms = (time.perf_counter_ns() - int(state_receive_time_ns)) * 1e-6
        except (TypeError, ValueError):
            return
        if delay_ms < 0.0:
            return
        with self._policy_delay_lock:
            self._policy_delay_count += 1
            self._policy_delay_sum_ms += delay_ms
            self._policy_delay_min_ms = min(self._policy_delay_min_ms, delay_ms)
            self._policy_delay_max_ms = max(self._policy_delay_max_ms, delay_ms)

    def _consume_policy_delay_stats(self):
        with self._policy_delay_lock:
            count = self._policy_delay_count
            if count == 0:
                return None
            mean_ms = self._policy_delay_sum_ms / count
            min_ms = self._policy_delay_min_ms
            max_ms = self._policy_delay_max_ms
            self._policy_delay_count = 0
            self._policy_delay_sum_ms = 0.0
            self._policy_delay_min_ms = float("inf")
            self._policy_delay_max_ms = 0.0
        return mean_ms, min_ms, max_ms, count

    def _resolve_sensor_slice(self, sensor_name: str):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            return None, 0
        return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    def _linacc(self) -> np.ndarray:
        if self.imu_lin_acc_adr is None or self.imu_lin_acc_dim < 3:
            return np.zeros(3, dtype=np.float32)
        return self.data.sensordata[self.imu_lin_acc_adr : self.imu_lin_acc_adr + 3].copy().astype(np.float32)

    def _publish_state(self):
        with self._sim_lock:
            q = self._mujoco_to_policy(self.data.qpos[7:]).astype(np.float32)
            dq = self._mujoco_to_policy(self.data.qvel[6:]).astype(np.float32)
            quat = self.data.qpos[3:7].copy().astype(np.float32)
            gyro = self.data.qvel[3:6].copy().astype(np.float32)
            linacc = self._linacc()
        self.transport.send_state(
            q=q,
            dq=dq,
            quat_wxyz=quat,
            gyro=gyro,
            linacc=linacc,
            buttons=self._buttons_snapshot(),
            sticks={name: 0.0 for name in STICK_KEYS},
        )

    def _publish_state_if_due(self):
        self._physics_tick += 1
        if (self._physics_tick % self.state_decimation) == 0:
            self._publish_state()

    def wait_for_high_cmd(self):
        print("Waiting for high level controller...")
        state_timer = Timer(self.state_dt)
        while self.is_alive and not self._have_command:
            self._publish_state()
            state_timer.sleep()
        print("Connected to high level")
        print('Press "s" to move to default pose')
        running_zero_cmd = True
        while self.is_alive and running_zero_cmd:
            buttons = self._buttons_snapshot()
            running_zero_cmd = not bool(buttons["start"])
            self._publish_state()
            state_timer.sleep()

    def simulate_gantry(self):
        print('Moving to default pose...\nPress "a" after the robot is in default pose to begin control loop')
        timer = Timer(self.low_level_dt)
        while True:
            with self._cmd_lock:
                ptargets_mujoco = self._policy_to_mujoco(self._ptargets_policy)
            with self._sim_lock:
                self.data.qpos[:7] = self.root_qpos_home
                self.data.qvel[:6] = 0.0
                self.data.qpos[7:] = ptargets_mujoco
                self.data.qvel[6:] = 0.0
                self.data.ctrl[:] = 0.0
                mujoco.mj_forward(self.model, self.data)

            if not self._viewer_sync():
                break

            self._publish_state_if_due()

            buttons = self._buttons_snapshot()
            running_default_pos = not (bool(buttons["A"]) or bool(buttons["stop"]))
            if not running_default_pos:
                break
            timer.sleep()

    def simulate_control(self):
        print("Running control loop...")
        with self._sim_lock:
            self.data.qpos[:7] = self.root_qpos_control
            mujoco.mj_forward(self.model, self.data)

        timer = Timer(self.low_level_dt)
        time_start = time.time()
        last_log_time = time_start
        loop_count = 0

        while self.is_alive:
            if not self.policy_queried:
                self._publish_state_if_due()
                timer.sleep()
                time_start = time.time()
                last_log_time = time_start
                loop_count = 0
                continue

            with self._cmd_lock:
                ptargets_mujoco = self._policy_to_mujoco(self._ptargets_policy)
                kp_mujoco = self._policy_to_mujoco(self._kp_policy)
                kd_mujoco = self._policy_to_mujoco(self._kd_policy)

            with self._sim_lock:
                qpos = self.data.qpos[7:]
                qvel = self.data.qvel[6:]
                if not self._have_tracking_target:
                    delta = ptargets_mujoco - qpos
                    if float(np.linalg.norm(delta)) > 1e-4:
                        self._have_tracking_target = True
                if not self._have_tracking_target:
                    self.data.qpos[:7] = self.root_qpos_control
                    self.data.qvel[:6] = 0.0
                    self.data.qpos[7:] = ptargets_mujoco
                    self.data.qvel[6:] = 0.0
                    self.data.ctrl[:] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                else:
                    ctrl = kp_mujoco * (ptargets_mujoco - qpos) + kd_mujoco * (0 - qvel)
                    ctrl = np.clip(ctrl, self.ctrl_lower, self.ctrl_upper)
                    self.data.ctrl[:] = ctrl
                    self._limit_external_forces()
                    mujoco.mj_step(self.model, self.data)

            if not self._viewer_sync():
                break

            self._publish_state_if_due()

            now = time.time()
            if now - last_log_time >= 1.0:
                seconds = loop_count * self.low_level_dt
                seconds_real = now - time_start
                with self._sim_lock:
                    root_z = float(self.data.qpos[2])
                delay_stats = self._consume_policy_delay_stats()
                if delay_stats is None:
                    delay_text = "policy_delay_ms=n/a"
                else:
                    mean_ms, min_ms, max_ms, delay_count = delay_stats
                    delay_text = (
                        f"policy_delay_ms mean={mean_ms:.3f} min={min_ms:.3f} "
                        f"max={max_ms:.3f} n={delay_count}"
                    )
                print(f"Time: {seconds:.2f}, Time real: {seconds_real:.2f}, Height: {root_z:.2f}, {delay_text}")
                last_log_time = now

            loop_count += 1
            timer.sleep()

        self.close()

    def _limit_external_forces(self):
        if self.max_external_force <= 0.0:
            return
        for i in range(self.model.nbody):
            force = self.data.xfrc_applied[i, :3]
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > self.max_external_force:
                self.data.xfrc_applied[i, :3] = force * (self.max_external_force / force_magnitude)

    def _viewer_sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            self.is_alive = False
            return False
        self._viewer_tick += 1
        if (self._viewer_tick % self.viewer_decim) == 0:
            self.viewer.sync()
        return True

    def run(self):
        self.keyboard_thread.start()

        if self.render_gui:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                self.viewer = viewer
                try:
                    self.wait_for_high_cmd()
                    self.simulate_gantry()
                    self.simulate_control()
                finally:
                    self.viewer = None
        else:
            self.wait_for_high_cmd()
            self.simulate_gantry()
            self.simulate_control()

    def close(self, *args):
        if not self.is_alive:
            return
        self.is_alive = False
        self._set_button("stop", True)
        stop_listening()
        self.transport.close()
        if self.keyboard_thread.is_alive() and threading.current_thread() is not self.keyboard_thread:
            self.keyboard_thread.join(timeout=1.0)
        sys.exit(0)


def load_config(path: str) -> DictToClass:
    with open(path, "r", encoding="utf-8") as f:
        cfg = DictToClass(yaml.load(f, Loader=yaml.FullLoader))
    setattr(cfg, "_config_dir", str(Path(path).resolve().parent))
    return cfg


def main(argv=None):
    try:
        import multiprocessing as mp

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Unified UDP MuJoCo sim2sim runner")
    parser.add_argument("--robot", choices=list(SUPPORTED_ROBOTS), default="g1")
    parser.add_argument("--xml_path", type=str, default=None)
    parser.add_argument("--bridge-config", type=str, default=None)
    args = parser.parse_args(argv)

    config_path = args.bridge_config or str(bridge_config_path(args.robot))
    config = load_config(config_path)
    Sim2Sim(args, config).run()


if __name__ == "__main__":
    main()
