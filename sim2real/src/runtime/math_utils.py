import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def _quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize quaternion(s) in wxyz order with numerical stability."""
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return q / n


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """Return quaternion conjugate in wxyz order."""
    q = np.asarray(q)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_inv_wxyz(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return quaternion inverse in wxyz order."""
    q = np.asarray(q)
    conj = _quat_conjugate_wxyz(q)
    n2 = np.sum(q * q, axis=-1, keepdims=True)
    n2 = np.maximum(n2, eps)
    return conj / n2


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (wxyz order) with broadcasting support."""
    a = np.asarray(a)
    b = np.asarray(b)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast(a, b).shape[:-1] + (4,), dtype=np.result_type(a, b))
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def yaw_quat_np(quat: np.ndarray) -> np.ndarray:
    """Extract yaw-only component (wxyz order) from quaternion(s)."""
    q = np.asarray(quat)
    assert q.shape[-1] == 4, "quat shape must be (..., 4) in wxyz order"
    shp = q.shape
    qv = q.reshape(-1, 4)

    w = qv[:, 0]
    x = qv[:, 1]
    y = qv[:, 2]
    z = qv[:, 3]

    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw

    q_yaw = np.zeros_like(qv, dtype=np.result_type(q, np.float32))
    q_yaw[:, 0] = np.cos(half)
    q_yaw[:, 3] = np.sin(half)
    q_yaw = _quat_normalize_wxyz(q_yaw)

    return q_yaw.reshape(shp)


def _quat_apply_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply inverse quaternion rotation (wxyz order) to vector(s)."""
    rot = R.from_quat(q_wxyz, scalar_first=True)
    return rot.inv().apply(v)


def _wrap_to_pi(a):
    """Wrap angle(s) to [-pi, pi]."""
    a_arr = np.asarray(a)
    v = (a_arr + np.pi) % (2.0 * np.pi) - np.pi
    v = np.where(v <= -np.pi + 1e-12, np.pi, v)
    if np.isscalar(a):
        return float(v)
    return v


def _clamp_indices(idx: np.ndarray, n: int) -> np.ndarray:
    """Clamp indices to valid range [0, n-1]."""
    return np.clip(idx, 0, max(0, n - 1))


def _slerp(q0: np.ndarray, q1: np.ndarray, steps: int) -> np.ndarray:
    """Slerp from q0 to q1 (exclusive endpoints) returning shape (steps, 4)."""
    if steps <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    key_times = [0.0, 1.0]
    key_rots = R.from_quat(np.stack([q0, q1], axis=0), scalar_first=True)
    slerp = Slerp(key_times, key_rots)
    t = np.linspace(0.0, 1.0, steps + 2, endpoint=True)[1:-1]
    out = slerp(t)
    return out.as_quat(scalar_first=True).astype(np.float32)


def _linspace_rows(a: np.ndarray, b: np.ndarray, steps: int) -> np.ndarray:
    """Row-wise linear interpolation from a to b (exclusive endpoints)."""
    if steps <= 0:
        return np.zeros((0, a.shape[-1]), dtype=np.float32)
    t = np.linspace(0.0, 1.0, steps + 2, endpoint=True)[1:-1][:, None]
    return (a[None, :] * (1.0 - t) + b[None, :] * t).astype(np.float32)


def _yaw_component_wxyz(q: np.ndarray) -> np.ndarray:
    """Return yaw-only quaternion component."""
    return yaw_quat_np(q).astype(np.float32, copy=False)


def _remove_yaw_keep_rp_wxyz(q: np.ndarray) -> np.ndarray:
    """Remove yaw from quaternion(s) while keeping roll/pitch."""
    q = np.asarray(q, dtype=np.float32)
    q_yaw = yaw_quat_np(q)
    q_yaw_inv = _quat_inv_wxyz(q_yaw)
    q_rp = _quat_mul_wxyz(q_yaw_inv, q)
    return _quat_normalize_wxyz(q_rp).astype(np.float32, copy=False)


def _zero_z(pos: np.ndarray) -> np.ndarray:
    """Zero out the Z-component of position array."""
    p = np.asarray(pos, dtype=np.float32).copy()
    p[..., 2] = 0.0
    return p


__all__ = [
    "_quat_normalize_wxyz",
    "_quat_conjugate_wxyz",
    "_quat_inv_wxyz",
    "_quat_mul_wxyz",
    "yaw_quat_np",
    "_quat_apply_inv",
    "_wrap_to_pi",
    "_clamp_indices",
    "_slerp",
    "_linspace_rows",
    "_yaw_component_wxyz",
    "_remove_yaw_keep_rp_wxyz",
    "_zero_z",
]
