from __future__ import annotations

from types import ModuleType


def load_xrobotoolkit_sdk() -> ModuleType:
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'xrobotoolkit_sdk'. Install the patched SDK in the active Python environment."
        ) from exc

    required_attrs = (
        "register_frame_callback",
        "clear_frame_callback",
        "has_frame_callback",
    )
    missing = tuple(name for name in required_attrs if not hasattr(xrt, name))
    if missing:
        missing_str = ", ".join(missing)
        raise ImportError(
            "Installed xrobotoolkit_sdk does not expose the required APIs: "
            f"{missing_str}. Reinstall the patched XRoboToolkit-PC-Service-Pybind build."
        )

    return xrt
