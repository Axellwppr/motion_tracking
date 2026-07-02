from __future__ import annotations

from typing import Any, Iterable

import mujoco as mj
import numpy as np
from mjviser.scene import ViserMujocoScene
from viser import ViserServer
from .params import resolve_robot_xml_path

DEFAULT_GROUND_PLANE_OPACITY = 0.8
DEFAULT_ROBOT_OPACITY = 0.35
DEFAULT_HUMAN_POINT_COLOR = (255, 209, 26)
DEFAULT_HUMAN_POINT_SIZE = 0.012
DEFAULT_HUMAN_POINT_SHAPE = "circle"
DEFAULT_ROBOT_AXIS_LENGTH = 0.06
DEFAULT_ROBOT_AXIS_RADIUS = max(DEFAULT_ROBOT_AXIS_LENGTH * 0.08, 0.002)
DEFAULT_HUMAN_AXIS_LENGTH = 0.12
DEFAULT_HUMAN_AXIS_RADIUS = max(DEFAULT_HUMAN_AXIS_LENGTH * 0.08, 0.003)


def _as_positions_array(values: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {arr.shape}")
    return arr


def _as_quats_wxyz_array(values: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4), got {arr.shape}")
    return arr


def _as_scales_array(
    values: np.ndarray | Iterable[float] | Iterable[Iterable[float]] | None,
    *,
    count: int,
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full((count,), float(arr), dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == count:
        return arr
    if arr.ndim == 2 and arr.shape == (count, 3):
        return arr
    raise ValueError(f"scales must have shape ({count},) or ({count}, 3), got {arr.shape}")


def _broadcast_colors_uint8(
    colors: np.ndarray | Iterable[int] | Iterable[Iterable[int]],
    *,
    count: int,
) -> np.ndarray:
    arr = np.asarray(colors)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"colors must have shape (3,) or ({count}, 3), got {arr.shape}")
        arr = np.broadcast_to(arr.reshape(1, 3), (count, 3))
    if arr.ndim != 2 or arr.shape != (count, 3):
        raise ValueError(f"colors must have shape (3,) or ({count}, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class MJViserViewer:
    def __init__(
        self,
        target_robot: str,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.target_robot = str(target_robot).strip().lower()
        self.xml_file = str(resolve_robot_xml_path(self.target_robot))
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        self._set_robot_mesh_opacity(DEFAULT_ROBOT_OPACITY)
        self.data = mj.MjData(self.model)
        self._qpos_template = np.asarray(self.model.qpos0, dtype=np.float64).copy()

        self.server = ViserServer(host=host, port=int(port), label=f"teleop-{self.target_robot}", verbose=True)
        self.mj_scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        self.mj_scene.create_visualization_gui(camera_distance=3.0)
        self._set_ground_plane_opacity(DEFAULT_GROUND_PLANE_OPACITY)

        self._body_id_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._axes_handles: dict[str, Any] = {}
        self._point_handles: dict[str, Any] = {}
        self._scene_offset = np.zeros((3,), dtype=np.float32)

    def _is_fixed_body(self, body_id: int) -> bool:
        root_id = int(self.model.body_rootid[body_id])
        return bool(self.model.body_weldid[body_id] == 0 and self.model.body_mocapid[root_id] < 0)

    def _set_robot_mesh_opacity(self, opacity: float) -> None:
        opacity_value = float(np.clip(opacity, 0.0, 1.0))
        material_ids: set[int] = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if self._is_fixed_body(body_id) or int(self.model.geom_type[geom_id]) == int(
                mj.mjtGeom.mjGEOM_PLANE
            ):
                continue
            if float(self.model.geom_rgba[geom_id, 3]) > 0.0:
                self.model.geom_rgba[geom_id, 3] = min(
                    float(self.model.geom_rgba[geom_id, 3]),
                    opacity_value,
                )
            mat_id = int(self.model.geom_matid[geom_id])
            if mat_id >= 0:
                material_ids.add(mat_id)

        for mat_id in material_ids:
            if float(self.model.mat_rgba[mat_id, 3]) > 0.0:
                self.model.mat_rgba[mat_id, 3] = min(float(self.model.mat_rgba[mat_id, 3]), opacity_value)

    def _overlay_path(self, name: str) -> str:
        clean = str(name).strip().strip("/")
        if not clean:
            raise ValueError("overlay name must not be empty")
        return f"/teleop/{clean}"

    def _resolve_body_ids(self, body_names: list[str] | tuple[str, ...]) -> np.ndarray:
        key = tuple(str(name) for name in body_names)
        cached = self._body_id_cache.get(key)
        if cached is not None:
            return cached

        ids = np.empty((len(key),), dtype=np.int32)
        for idx, body_name in enumerate(key):
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Unknown body name for '{self.target_robot}': {body_name}")
            ids[idx] = body_id
        self._body_id_cache[key] = ids
        return ids

    def _set_ground_plane_opacity(self, opacity: float) -> None:
        handle_from_name = getattr(self.server.scene, "_handle_from_node_name", None)
        if handle_from_name is None:
            return
        opacity_value = float(np.clip(opacity, 0.0, 1.0))
        for node_name, handle in handle_from_name.items():
            if not node_name.startswith("/fixed_bodies/"):
                continue
            if hasattr(handle, "plane_opacity"):
                handle.plane_opacity = opacity_value

    def update_qpos(self, qpos: np.ndarray) -> None:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.shape[0] > self.model.nq:
            raise ValueError(f"qpos is longer than model.nq ({q.shape[0]} > {self.model.nq})")
        self.data.qpos[:] = self._qpos_template
        self.data.qpos[: q.shape[0]] = q
        mj.mj_forward(self.model, self.data)
        self.mj_scene.update_from_mjdata(self.data)
        self._scene_offset = np.asarray(getattr(self.mj_scene, "_scene_offset", np.zeros(3)), dtype=np.float32).copy()

    def get_body_transforms(self, body_names: list[str] | tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        body_ids = self._resolve_body_ids(body_names)
        positions = np.asarray(self.data.xpos[body_ids], dtype=np.float32).copy()
        quats_wxyz = np.asarray(self.data.xquat[body_ids], dtype=np.float32).copy()
        return positions, quats_wxyz

    def draw_bodies(
        self,
        name: str,
        body_names: list[str] | tuple[str, ...],
        *,
        visible: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions, quats_wxyz = self.get_body_transforms(body_names)
        positions = positions + self._scene_offset
        self.set_axes(
            f"{name}/axes",
            positions,
            quats_wxyz,
            axes_length=DEFAULT_ROBOT_AXIS_LENGTH,
            axes_radius=DEFAULT_ROBOT_AXIS_RADIUS,
            visible=bool(visible and positions.shape[0] > 0),
        )
        return positions, quats_wxyz

    def draw_human_data(
        self,
        name: str,
        human_positions: np.ndarray | Iterable[Iterable[float]] | None,
        human_rotations_wxyz: np.ndarray | Iterable[Iterable[float]] | None = None,
        *,
        visible: bool = True,
    ) -> np.ndarray:
        if human_positions is None:
            positions = np.zeros((0, 3), dtype=np.float32)
        else:
            positions = _as_positions_array(human_positions)
        positions = positions + self._scene_offset

        if human_rotations_wxyz is not None:
            quats_wxyz = _as_quats_wxyz_array(human_rotations_wxyz)
            self.set_points(
                f"{name}/points",
                np.zeros((0, 3), dtype=np.float32),
                colors=DEFAULT_HUMAN_POINT_COLOR,
                visible=False,
            )
            self.set_axes(
                f"{name}/axes",
                positions,
                quats_wxyz,
                axes_length=DEFAULT_HUMAN_AXIS_LENGTH,
                axes_radius=DEFAULT_HUMAN_AXIS_RADIUS,
                visible=bool(visible and positions.shape[0] > 0),
            )
            return positions

        self.set_axes(
            f"{name}/axes",
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            visible=False,
        )
        self.set_points(
            f"{name}/points",
            positions,
            colors=DEFAULT_HUMAN_POINT_COLOR,
            point_size=DEFAULT_HUMAN_POINT_SIZE,
            point_shape=DEFAULT_HUMAN_POINT_SHAPE,
            visible=bool(visible and positions.shape[0] > 0),
        )
        return positions

    def set_axes(
        self,
        name: str,
        positions: np.ndarray | Iterable[Iterable[float]],
        quats_wxyz: np.ndarray | Iterable[Iterable[float]],
        *,
        scales: np.ndarray | Iterable[float] | Iterable[Iterable[float]] | None = None,
        axes_length: float = 0.06,
        axes_radius: float = 0.004,
        visible: bool = True,
    ) -> None:
        overlay_path = self._overlay_path(name)
        pos_arr = _as_positions_array(positions)
        quat_arr = _as_quats_wxyz_array(quats_wxyz)
        if pos_arr.shape[0] != quat_arr.shape[0]:
            raise ValueError(
                f"positions/quaternions count mismatch: {pos_arr.shape[0]} vs {quat_arr.shape[0]}"
            )
        if pos_arr.shape[0] == 0 or not visible:
            handle = self._axes_handles.get(overlay_path)
            if handle is not None:
                handle.visible = False
            return

        scale_arr = _as_scales_array(scales, count=pos_arr.shape[0])
        handle = self._axes_handles.get(overlay_path)
        if handle is None:
            handle = self.server.scene.add_batched_axes(
                overlay_path,
                batched_wxyzs=quat_arr,
                batched_positions=pos_arr,
                batched_scales=scale_arr,
                axes_length=float(axes_length),
                axes_radius=float(axes_radius),
                visible=bool(visible),
            )
            self._axes_handles[overlay_path] = handle
            return

        handle.batched_positions = pos_arr
        handle.batched_wxyzs = quat_arr
        handle.batched_scales = scale_arr
        handle.axes_length = float(axes_length)
        handle.axes_radius = float(axes_radius)
        handle.visible = bool(visible)

    def set_points(
        self,
        name: str,
        positions: np.ndarray | Iterable[Iterable[float]],
        colors: np.ndarray | Iterable[int] | Iterable[Iterable[int]],
        *,
        point_size: float = 0.01,
        point_shape: str = "circle",
        visible: bool = True,
    ) -> None:
        overlay_path = self._overlay_path(name)
        pos_arr = _as_positions_array(positions)
        if pos_arr.shape[0] == 0 or not visible:
            handle = self._point_handles.get(overlay_path)
            if handle is not None:
                handle.visible = False
            return

        color_arr = _broadcast_colors_uint8(colors, count=pos_arr.shape[0])
        handle = self._point_handles.get(overlay_path)
        if handle is None:
            handle = self.server.scene.add_point_cloud(
                overlay_path,
                points=pos_arr,
                colors=color_arr,
                point_size=float(point_size),
                point_shape=str(point_shape),
                precision="float32",
                visible=bool(visible),
            )
            self._point_handles[overlay_path] = handle
            return

        handle.points = pos_arr
        handle.colors = color_arr
        handle.point_size = float(point_size)
        handle.point_shape = str(point_shape)
        handle.precision = "float32"
        handle.visible = bool(visible)

    def remove(self, name: str) -> None:
        overlay_path = self._overlay_path(name)
        overlay_prefix = overlay_path + "/"

        for path in [path for path in list(self._axes_handles) if path == overlay_path or path.startswith(overlay_prefix)]:
            self._axes_handles.pop(path).remove()

        for path in [path for path in list(self._point_handles) if path == overlay_path or path.startswith(overlay_prefix)]:
            self._point_handles.pop(path).remove()

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass
