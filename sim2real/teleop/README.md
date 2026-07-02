# XR Teleop Setup

Supplement for installing and validating realtime VR motion input.
Runtime startup order and button usage are documented in the top-level `README.md`.

Use this document when `sim2real/config/<robot>/tracking.yaml` has:

```yaml
motion_source:
  type: "vr"
```

The teleop bridge supports `g1` and `l7`.
It reads XR/PICO body/controller data, retargets the latest human pose, and serves robot poses to `src/deploy.py` over ZMQ.

## System Map

Default layout:

```text
PICO headset
  XRoboToolkit PICO app
  body trackers / controllers
        |
        | XRoboToolkit network stream over LAN
        v
Teleop host: x86 Ubuntu workstation or G1 onboard computer
  XRoboToolkit PC service
        |
        | local XRoboToolkit SDK / Python binding
        v
  serve_xrobot_teleop.py
        |
        | retargeted robot qpos + controller buttons over ZMQ
        | request: tcp://127.0.0.1:28701
        | reply:   tcp://127.0.0.1:28702
        | buttons: tcp://127.0.0.1:28703
        v
  src/deploy.py
        |
        | robot command path documented in the top-level README
        v
  simulator or real robot bridge

Browser viewer
  http://localhost:8080 reads retarget visualization from serve_xrobot_teleop.py
```

The default config assumes `serve_xrobot_teleop.py` and `src/deploy.py` run on
the same teleop host, because `motion_source.vr` connects to `127.0.0.1`.
For remote ZMQ, change `sim2real/config/<robot>/tracking.yaml` to use the
teleop host IP instead.

Install/check these pieces in order:

| Piece | Where it runs | What it does | Ready when |
| --- | --- | --- | --- |
| XRoboToolkit PICO app | PICO headset | sends body, tracker, and controller data | the app connects to the PC service and reports working |
| XRoboToolkit PC service | teleop host | receives the PICO stream | `/opt/apps/roboticsservice/runService.sh` is running |
| `xrobotoolkit_sdk` Python binding | `sim2real/.venv` on teleop host | lets Python read the PC service stream | `import xrobotoolkit_sdk` works |
| `serve_xrobot_teleop.py` | teleop host | retargets XR human pose to G1/L7 robot qpos | it prints ZMQ endpoints and optional `viewer_url` |
| `src/deploy.py` with `motion_source.type: "vr"` | same host by default | requests retarget frames and runs the policy | it prints `VRMotionSource` connected endpoints |
| Browser viewer | browser on teleop host or forwarded port | visualizes retarget result only | `http://localhost:8080` opens |

## External Requirements

This repo does not bundle the XRoboToolkit `.deb` packages or PICO `.apk`.
Download them separately before setting up live VR input.

Hardware:

- PICO headset supported by XRoboToolkit, with two controllers
- two PICO motion trackers for full-body lower-body tracking
- a low-latency network where the PICO can reach the teleop host

Teleop host:

- one Ubuntu machine that runs both `serve_xrobot_teleop.py` and XRoboToolkit PC service
- `x86_64` Ubuntu 22.04 / 24.04 workstation for local teleop, or G1 onboard `aarch64` for onboard teleop
- `uv`, `git`, and the native build dependencies listed below
- `adb` if installing the PICO APK from the Ubuntu host

Download from XR-Robotics:

- XRoboToolkit PC service source: <https://github.com/XR-Robotics/XRoboToolkit-PC-Service>
- XRoboToolkit PC service releases: <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0>
- Ubuntu 22.04 x86_64 PC service deb: <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb>
- Ubuntu 24.04 x86_64 PC service deb: <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb>
- G1/Jetson-style aarch64 headless PC service deb: <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb>
- XRoboToolkit PICO app: <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk>
- XRoboToolkit PICO app releases: <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases>

Teleop needs two separate XRoboToolkit pieces:

- system service: `/opt/apps/roboticsservice/runService.sh` receives the PICO stream
- Python binding: `xrobotoolkit_sdk` is imported by `serve_xrobot_teleop.py`

`install_xrobottoolkit_sdk.sh` only installs the Python binding.
It does not install or start the system service.

## Install XRoboToolkit PICO App

Enable developer mode on the PICO headset first.
Then install the XRoboToolkit APK either from an Ubuntu host with `adb`:

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk
adb install -g XRoboToolkit-PICO-1.1.1.apk
```

or by opening the APK link in the PICO browser and installing it from downloads.

## Install XRoboToolkit PC Service

Install this on the same machine that runs `serve_xrobot_teleop.py` and talks
directly to the PICO headset.
For local workstation teleop, that is the x86 Ubuntu machine.
For G1 onboard teleop, that is the G1 onboard computer.

### Local x86 Ubuntu Host

Ubuntu 22.04:

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

Ubuntu 24.04:

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

Verify that the service entry points exist:

```bash
test -x /opt/apps/roboticsservice/runService.sh
test -x /opt/apps/roboticsservice/RoboticsServiceProcess
```

Start the service before connecting the PICO client:

```bash
cd /opt/apps/roboticsservice
bash runService.sh
```

### G1 Onboard Host

On G1, use the `arm64` / `aarch64` headless PC service package:

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb
sudo dpkg -i XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb
```

Then verify and start it the same way:

```bash
test -x /opt/apps/roboticsservice/runService.sh
test -x /opt/apps/roboticsservice/RoboticsServiceProcess

cd /opt/apps/roboticsservice
bash runService.sh
```

Some upstream arm64/headless service builds may require a newer `GLIBC` / `GLIBCXX`
than the G1 Ubuntu 20.04 system provides.
If `RoboticsServiceProcess` fails with symbol-version errors, use a G1-compatible
package or rebuild the PC service from source against the G1 system Qt5, gRPC, and
protobuf libraries.
Reinstalling the deb may overwrite those rebuilt binaries.

## Python Environment

From the `sim2real` directory:

```bash
cd <repo>/sim2real
uv sync
```

## Native Build Dependencies

Install the native packages needed to build the XRoboToolkit Python binding:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config \
  libprotobuf-dev protobuf-compiler \
  libgrpc++-dev libgrpc-dev protobuf-compiler-grpc \
  libgl1 libegl1 libxrender1 libxext6
```

## Install XRoboToolkit Python Binding

```bash
cd <repo>/sim2real
bash install_xrobottoolkit_sdk.sh
```

The installer clones XRoboToolkit dependencies under `teleop/deps/`, builds the native SDK, and installs `xrobotoolkit_sdk` into `sim2real/.venv`.
Use this same script on both x86 and G1/arm64.
On G1, the source-build path avoids incompatible upstream prebuilt `.so` files.

## Verify Python Imports

```bash
cd <repo>/sim2real
uv run python - <<'PY'
import xrobotoolkit_sdk
import zmq
print("xrobotoolkit_sdk:", xrobotoolkit_sdk.__file__)
print("pyzmq: OK")
PY
```

## Verify XR Stream

Start XRoboToolkit PC service, connect the PICO client, and enable body/controller streaming.
Then run:

```bash
cd <repo>/sim2real
uv run python - <<'PY'
import xrobotoolkit_sdk as xrt
xrt.init()
print("body:", xrt.is_body_data_available())
print("headset:", xrt.get_headset_pose())
xrt.close()
PY
```

If body data is unavailable, check the PICO connection, tracker calibration, PC service status, and network route between the headset and host.

## Connection Troubleshooting Checklist

When the PICO cannot connect or body data stays unavailable, check:

- The XRoboToolkit PC service is running on the host that the PICO connects to.
- The host and PICO are on the same reachable LAN.
- The IP configured in the PICO XRoboToolkit client is the host IP that the PICO can route to.
- Firewalls such as `iptables`, `nftables`, or `ufw` allow the XRoboToolkit traffic.
- VPN or TUN interfaces are disabled, or their routing does not hide the host IP from the PICO.
- Proxy variables such as `http_proxy` / `https_proxy` are unset, or `127.0.0.1` is included in `no_proxy`.
- The PICO trackers are paired, calibrated, and body tracking is enabled.
- `xrobotoolkit_sdk` imports from `sim2real/.venv`, not from an unrelated Python environment.

If the Python binding raises `AttributeError` for callback APIs such as
`register_frame_callback`, reinstall it with this repo's `install_xrobottoolkit_sdk.sh`.

## Teleop Server Smoke Test

After setup succeeds, this command should start the ZMQ teleop server:

```bash
cd <repo>/sim2real/teleop
uv run python serve_xrobot_teleop.py --robot g1
```

Use `--robot l7` for L7.
Defaults come from `../config/<robot>/retarget/teleop.yaml`.

Keep a stable standing posture during startup so the retarget stack can estimate root height.

## Browser Retarget Viewer

By default, `config/<robot>/retarget/teleop.yaml` has:

```yaml
server:
  visualize: true
```

When visualization is enabled, `serve_xrobot_teleop.py` starts a browser viewer and prints:

```text
viewer_url: http://localhost:8080
```

Open that URL on the machine running the teleop server to inspect the retarget result.
If the teleop server runs on a remote machine, forward port `8080` or open the browser on that machine.

For the full sim2sim/sim2real startup order and XR button usage, return to the top-level `README.md`.

## ZMQ Endpoints

Default endpoints:

- request: `tcp://*:28701`
- reply: `tcp://*:28702`
- controller buttons: `tcp://*:28703`

`src/deploy.py` reads these addresses from `config/<robot>/tracking.yaml` under `motion_source.vr`.

Reply messages are multipart:

1. JSON header with `start`, `num_frames`, and `qpos_size`
2. contiguous `float32` payload shaped as `(num_frames, qpos_size)`

Frame layout:

- `qpos[0:3]`: root position
- `qpos[3:7]`: root quaternion in `wxyz`
- `qpos[7:]`: robot joint positions

## Record And Replay

Record raw XR data:

```bash
cd <repo>/sim2real/teleop
uv run python record_xrobot_motion.py
```

Replay a recording through the local retarget stack:

```bash
uv run python replay_xrobot_motion.py /path/to/xrobot_raw_segment.npz --robot g1
```

`replay_xrobot_motion.py --zmq` serves a recording through the same request/reply protocol used by live teleop.
