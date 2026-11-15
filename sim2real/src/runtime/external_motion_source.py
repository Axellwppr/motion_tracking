import time

import numpy as np

from motion_channel import MotionChannel
from runtime.motion_sources import MotionSourceBase


class ExternalMotionSource(MotionSourceBase):
    """Append reference segments prepared by another process (see motion_channel.py).

    Segments arrive already continuing the reference tail; this source holds
    the tail long enough for the policy's look-ahead to reach the new frames
    (copies of the tail, ``horizon + 1`` minus what is still queued), appends
    them, and reports the reference cursor and robot state back. A segment
    planned against an older reference (revision) is rejected.
    """

    def __init__(self, policy, policy_cfg):
        super().__init__(policy, policy_cfg)
        self.channel = MotionChannel(policy_cfg.motion_source["external"]["directory"])
        self.horizon = int(max(0, np.max(policy.future_steps)))
        self.tick = 0
        self.revision = 0  # accepted segments so far
        self.ack = 0
        self.result = 0
        self.end = 0  # absolute frame index of the reference tail

    def on_fade_in(self):
        super().on_fade_in()
        self.end = self.policy.ref_len - 1

    def post_step(self):
        self.tick += 1
        request = self.channel.take_request(self.ack)
        if request is not None:
            self.result = 1 if self._append(request) else -1
            self.ack = request.serial
        self._publish()

    def _append(self, request):
        policy = self.policy
        if request.revision != self.revision:
            return False
        remaining = policy.ref_len - 1 - policy.ref_idx
        tail = np.concatenate([policy.ref_root_pos[-1], policy.ref_root_quat[-1], policy.ref_joint_pos[-1]])
        hold = np.repeat(tail[None], max(0, self.horizon + 1 - remaining), axis=0)
        frames = np.vstack([hold, request.frames])
        policy.append_ref_frames({"root_pos": frames[:, :3], "root_quat": frames[:, 3:7], "joint_pos": frames[:, 7:]})
        self.end += len(frames)
        self.revision += 1
        return True

    def _publish(self):
        policy, controller = self.policy, self.policy.controller
        remaining = policy.ref_len - 1 - policy.ref_idx

        def pose(i):
            return np.concatenate([policy.ref_root_pos[i], policy.ref_root_quat[i], policy.ref_joint_pos[i]])

        self.channel.write_state(
            tick=self.tick, revision=self.revision, ack=self.ack, result=self.result,
            index=self.end - remaining, length=self.end + 1,
            stamp=controller.last_state_receive_time_ns or 0, received=time.monotonic(),
            q=controller.low_state["q"], dq=controller.low_state["dq"], target=controller.cmd_q,
            quat=controller.quat, gyro=controller.gyro,
            reference=pose(policy.ref_idx), tail=pose(-1),
        )
