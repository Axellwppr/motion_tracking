from pathlib import Path

# Root of the sim2real package (one level up from this file's directory)
SIM2REAL_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = SIM2REAL_ROOT / "config"
SUPPORTED_ROBOTS = ("g1", "l7")
RETARGET_IK_FILENAMES = {
    "g1": "xrobot_to_g1.json",
    "l7": "xrobot_to_l7.json",
}
ROBOT_XML_FILENAMES = {
    "g1": "g1.xml",
    "l7": "l7.xml",
}

# Centralized directory for data assets (ckpts, data, g1, plot, visuals, npy logs)
ASSETS_DIR = SIM2REAL_ROOT / "assets"

# Ensure the assets directory exists when imported in scripts that write outputs
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def to_assets_path(rel: str | Path) -> Path:
    """Return an absolute path under ASSETS_DIR for a given relative path."""
    p = Path(rel)
    return p if p.is_absolute() else (ASSETS_DIR / p)


def robot_config_dir(robot: str | Path) -> Path:
    robot_key = str(robot).strip().lower()
    return CONFIG_DIR / robot_key


def robot_config_path(robot: str | Path, name: str | Path) -> Path:
    name_path = Path(name)
    return name_path if name_path.is_absolute() else (robot_config_dir(robot) / name_path)


def controller_config_path(robot: str | Path) -> Path:
    return robot_config_path(robot, "controller.yaml")


def tracking_config_path(robot: str | Path, name: str | Path = "tracking.yaml") -> Path:
    name_path = Path(name)
    if name_path.is_absolute():
        return name_path
    if len(name_path.parts) > 1 and name_path.parts[0] == "config":
        return SIM2REAL_ROOT / name_path
    return robot_config_path(robot, name_path)


def bridge_config_path(robot: str | Path) -> Path:
    return robot_config_path(robot, "bridge.yaml")


def robot_assets_dir(robot: str | Path) -> Path:
    return robot_config_path(robot, "assets")


def robot_xml_path(robot: str | Path) -> Path:
    robot_key = str(robot).strip().lower()
    return robot_assets_dir(robot_key) / ROBOT_XML_FILENAMES[robot_key]


def retarget_config_dir(robot: str | Path) -> Path:
    return robot_config_path(robot, "retarget")


def retarget_teleop_config_path(robot: str | Path) -> Path:
    return retarget_config_dir(robot) / "teleop.yaml"


def retarget_ik_config_path(robot: str | Path) -> Path:
    robot_key = str(robot).strip().lower()
    return retarget_config_dir(robot_key) / RETARGET_IK_FILENAMES[robot_key]
