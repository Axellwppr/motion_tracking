"""Reference hand-off between deploy.py and an external process, through files.

Three memory-mapped arrays in one directory:

* ``request.npy``  header of the pending segment: serial, revision, count
* ``frames.npy``   the segment frames, ``[count, 36]``: root xyz, root wxyz, joints (policy order)
* ``state.npy``    one row published by deploy.py every control step

The external process writes a request; deploy.py appends it at its next step
and acknowledges the serial in the state (``result`` 1 accepted, -1 rejected).
Only one request may be pending. ``flock`` orders the writes on every CPU
architecture; deploy.py never blocks on it and simply retries next step.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from pathlib import Path

import numpy as np

POSE = 36
MAX_FRAMES = 4096
REQUEST = np.dtype([("serial", "i8"), ("revision", "i8"), ("count", "i8")])


def state_dtype(n_joints):
    return np.dtype([
        ("tick", "i8"), ("revision", "i8"), ("ack", "i8"), ("result", "i8"),
        ("index", "i8"), ("length", "i8"), ("stamp", "i8"), ("received", "f8"),
        ("q", "f8", (n_joints,)), ("dq", "f8", (n_joints,)), ("target", "f8", (n_joints,)),
        ("quat", "f8", (4,)), ("gyro", "f8", (3,)), ("reference", "f8", (POSE,)), ("tail", "f8", (POSE,)),
    ])


@dataclass(frozen=True)
class Request:
    serial: int
    revision: int
    frames: np.ndarray


class MotionChannel:
    def __init__(self, directory, create=False, n_joints=29):
        path = Path(directory)
        if create:
            path.mkdir(parents=True, exist_ok=False)
        mode = "w+" if create else "r+"
        self.state = np.lib.format.open_memmap(path / "state.npy", mode=mode, dtype=state_dtype(n_joints), shape=(1,))
        self.header = np.lib.format.open_memmap(path / "request.npy", mode=mode, dtype=REQUEST, shape=(1,))
        self.frames = np.lib.format.open_memmap(path / "frames.npy", mode=mode, dtype="f4", shape=(MAX_FRAMES, POSE))
        self._state_lock = (path / "state.lock").open("a+b")
        self._request_lock = (path / "request.lock").open("a+b")
        if create:
            (path / "ready").write_text("1\n")

    @staticmethod
    @contextmanager
    def _locked(handle, exclusive, wait):
        flags = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if wait else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle, flags)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

    # ------------------------------------------------------------ deploy side
    def write_state(self, **values):
        """Publish one state row; skipped (returns False) while a reader holds the lock."""
        with self._locked(self._state_lock, exclusive=True, wait=False) as acquired:
            if acquired:
                for key, value in values.items():
                    self.state[key][0] = value
            return acquired

    def take_request(self, acknowledged):
        """The pending request newer than ``acknowledged``, or None."""
        with self._locked(self._request_lock, exclusive=False, wait=False) as acquired:
            if not acquired:
                return None
            header = self.header[0].copy()
            if int(header["serial"]) <= acknowledged:
                return None
            count = int(header["count"])
            return Request(int(header["serial"]), int(header["revision"]), np.array(self.frames[:count], dtype=np.float64))

    # ----------------------------------------------------------- client side
    def read(self):
        """The latest state row (a copy), or None if deploy has not published yet."""
        with self._locked(self._state_lock, exclusive=False, wait=True):
            row = self.state[0].copy()
        return row if int(row["tick"]) > 0 else None

    def pending(self):
        """True while a submitted request has not been acknowledged."""
        row = self.read()
        return int(self.header["serial"][0]) > (int(row["ack"]) if row is not None else 0)

    def submit(self, frames, revision):
        """Publish a segment; returns its serial. Raises RuntimeError if one is still pending."""
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != POSE or not 0 < len(frames) <= MAX_FRAMES:
            raise ValueError(f"segment must be [1..{MAX_FRAMES}, {POSE}] frames")
        if self.pending():
            raise RuntimeError("previous segment not yet acknowledged")
        with self._locked(self._request_lock, exclusive=True, wait=True):
            serial = int(self.header["serial"][0]) + 1
            self.frames[: len(frames)] = frames
            self.header["revision"][0] = revision
            self.header["count"][0] = len(frames)
            self.header["serial"][0] = serial  # last: the request becomes visible
        return serial

    def close(self):
        self._state_lock.close()
        self._request_lock.close()
