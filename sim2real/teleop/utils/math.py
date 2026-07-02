from __future__ import annotations

import numpy as np


"""Quaternion convention in this module is always wxyz."""


def quat_xyzw_to_wxyz_np(quat: np.ndarray) -> np.ndarray:
    quat_arr = np.asarray(quat)
    return quat_arr[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw_np(quat: np.ndarray) -> np.ndarray:
    quat_arr = np.asarray(quat)
    return quat_arr[..., [1, 2, 3, 0]]


def quat_normalize_np(
    quat: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    quat_arr = np.asarray(quat)
    dtype = np.result_type(quat_arr, np.float64)
    normalized = quat_arr.astype(dtype, copy=False)
    norms = np.linalg.norm(normalized, axis=-1, keepdims=True)
    return normalized / np.maximum(norms, eps)


def quat_normalize_safe_np(
    quat: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    quat_arr = np.asarray(quat)
    dtype = np.result_type(quat_arr, np.float64)
    normalized = quat_arr.astype(dtype, copy=False)
    norms = np.linalg.norm(normalized, axis=-1, keepdims=True)
    safe = normalized / np.maximum(norms, eps)
    identity = np.zeros_like(safe)
    identity[..., 0] = 1.0
    return np.where(np.isfinite(norms) & (norms >= eps), safe, identity)


def quat_mul_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    dtype = np.result_type(x_arr, y_arr, np.float64)
    x_arr = x_arr.astype(dtype, copy=False)
    y_arr = y_arr.astype(dtype, copy=False)

    x0, x1, x2, x3 = x_arr[..., 0:1], x_arr[..., 1:2], x_arr[..., 2:3], x_arr[..., 3:4]
    y0, y1, y2, y3 = y_arr[..., 0:1], y_arr[..., 1:2], y_arr[..., 2:3], y_arr[..., 3:4]

    return np.concatenate(
        [
            x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
            x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
            x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
            x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0,
        ],
        axis=-1,
    )


def quat_slerp_np(quat0: np.ndarray, quat1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = quat_normalize_safe_np(quat0).reshape(4)
    q1 = quat_normalize_safe_np(quat1).reshape(4)
    t = float(np.clip(alpha, 0.0, 1.0))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return quat_normalize_safe_np(q0 + t * (q1 - q0))

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    if abs(sin_theta_0) < 1e-8:
        return quat_normalize_safe_np(q0)

    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return quat_normalize_safe_np(s0 * q0 + s1 * q1)


def quat_apply_np(
    quat: np.ndarray,
    vec: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    quat_arr = (
        quat_normalize_np(quat)
        if normalize
        else np.asarray(quat, dtype=np.result_type(quat, vec, np.float64))
    )
    vec_arr = np.asarray(vec, dtype=quat_arr.dtype)
    quat_xyz = quat_arr[..., 1:]
    quat_w = quat_arr[..., 0:1]
    uv = np.cross(quat_xyz, vec_arr)
    uuv = np.cross(quat_xyz, uv)
    return vec_arr + 2.0 * (quat_w * uv + uuv)


def rotmat_to_quat_np(rotmat: np.ndarray) -> np.ndarray:
    rotmat_arr = np.asarray(rotmat)
    if rotmat_arr.shape[-2:] != (3, 3):
        raise ValueError(f"rotmat must have shape (..., 3, 3), got {rotmat_arr.shape}")

    flat = rotmat_arr.astype(np.result_type(rotmat_arr, np.float64), copy=False).reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=flat.dtype)

    m00 = flat[:, 0, 0]
    m01 = flat[:, 0, 1]
    m02 = flat[:, 0, 2]
    m10 = flat[:, 1, 0]
    m11 = flat[:, 1, 1]
    m12 = flat[:, 1, 2]
    m20 = flat[:, 2, 0]
    m21 = flat[:, 2, 1]
    m22 = flat[:, 2, 2]

    trace = m00 + m11 + m22
    mask0 = trace > 0.0
    mask1 = (~mask0) & (m00 > m11) & (m00 > m22)
    mask2 = (~mask0) & (~mask1) & (m11 > m22)
    mask3 = ~(mask0 | mask1 | mask2)

    if np.any(mask0):
        s = np.sqrt(np.maximum(trace[mask0] + 1.0, 0.0)) * 2.0
        out[mask0, 0] = 0.25 * s
        out[mask0, 1] = (m21[mask0] - m12[mask0]) / s
        out[mask0, 2] = (m02[mask0] - m20[mask0]) / s
        out[mask0, 3] = (m10[mask0] - m01[mask0]) / s

    if np.any(mask1):
        s = np.sqrt(np.maximum(1.0 + m00[mask1] - m11[mask1] - m22[mask1], 0.0)) * 2.0
        out[mask1, 0] = (m21[mask1] - m12[mask1]) / s
        out[mask1, 1] = 0.25 * s
        out[mask1, 2] = (m01[mask1] + m10[mask1]) / s
        out[mask1, 3] = (m02[mask1] + m20[mask1]) / s

    if np.any(mask2):
        s = np.sqrt(np.maximum(1.0 + m11[mask2] - m00[mask2] - m22[mask2], 0.0)) * 2.0
        out[mask2, 0] = (m02[mask2] - m20[mask2]) / s
        out[mask2, 1] = (m01[mask2] + m10[mask2]) / s
        out[mask2, 2] = 0.25 * s
        out[mask2, 3] = (m12[mask2] + m21[mask2]) / s

    if np.any(mask3):
        s = np.sqrt(np.maximum(1.0 + m22[mask3] - m00[mask3] - m11[mask3], 0.0)) * 2.0
        out[mask3, 0] = (m10[mask3] - m01[mask3]) / s
        out[mask3, 1] = (m02[mask3] + m20[mask3]) / s
        out[mask3, 2] = (m12[mask3] + m21[mask3]) / s
        out[mask3, 3] = 0.25 * s

    out = quat_normalize_np(out)
    return out.reshape(rotmat_arr.shape[:-2] + (4,))
