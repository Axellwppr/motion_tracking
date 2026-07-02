# Motion Tracking Sim2Real

Release tree for motion-tracking sim2sim and sim2real runtime code.

This branch supports:

- G1 sim2sim
- L7 sim2sim
- G1 sim2real

It does not include training code or dataset generation code.
The L7 hardware sim2real bridge is not part of this branch.

## Directory Layout

```text
sim2real/
  src/                 Python policy runtime, sim2sim runner, motion selector
  config/g1/           G1 runtime config, MuJoCo assets, retarget config, ckpt/motion assets
  config/l7/           L7 runtime config, MuJoCo assets, retarget config, ckpt/motion assets
  teleop/              XR/PICO realtime teleop bridge

g1_sim2real/
  src/                 G1 C++ low-level UDP/DDS bridge
  config/              G1 bridge config
  scripts/             G1 bridge build/run helpers
  third_party/         Vendored C++ dependencies
```

## Supported Modes

| Mode | Robot | Low-level side | Motion input |
| --- | --- | --- | --- |
| sim2sim | G1 | MuJoCo simulator | UDP `.npz` playback or VR realtime |
| sim2sim | L7 | MuJoCo simulator | UDP `.npz` playback or VR realtime |
| sim2real | G1 | Unitree G1 C++ bridge | UDP `.npz` playback or VR realtime |

## Checkpoints And Motions

This release includes G1/L7 PMG checkpoints and a G1 compliance checkpoint.
Checkpoint and motion assets live under each robot config directory:

```text
sim2real/config/g1/ckpts/G1_PMG/
sim2real/config/g1/ckpts/G1_Compliance/
sim2real/config/g1/motions/*.npz

sim2real/config/l7/ckpts/L7_PMG/
sim2real/config/l7/motions/*.npz
```

Each deploy checkpoint directory contains:

```text
policy.json
policy.onnx
policy.onnx.data
```

The default `config/g1/tracking.yaml` uses `G1_PMG`.
To run the G1 compliance policy, use `config/g1/tracking_compliance.yaml`.
The compliance YAML switches the policy path, removes future steps `5` and `6`,
and enables the compliance flag observation.

## Common Setup

Install the Python runtime:

```bash
cd <repo>/sim2real
uv sync
```

All Python commands below should be run from `sim2real/`.
Supported robot keys are `g1` and `l7`.

Choose the motion input in `sim2real/config/<robot>/tracking.yaml`:

```yaml
motion_source:
  type: "udp"  # local motion .npz playback
```

or:

```yaml
motion_source:
  type: "vr"   # realtime XR/PICO input
```

## Control Buttons

The policy controller always uses the same state flow:

1. wait in zero torque
2. receive `start`
3. move to the default pose
4. receive `A`
5. enter the tracking policy
6. receive `stop` to exit

Button sources:

| Context | `start` | `A` | `stop` |
| --- | --- | --- | --- |
| sim2sim terminal | keyboard `s` | keyboard `a` | keyboard `x` |
| G1 bridge terminal | keyboard `s` | keyboard `a` | keyboard `x` |
| G1 robot remote | remote `start` | remote `A` | remote `select` |

For sim2sim, keep the simulator window focused when pressing `s`, `a`, or `x`.

VR has an additional live-stream control layer:

| XR/PICO controller | Runtime meaning |
| --- | --- |
| right-hand `A` | start or resume realtime VR motion streaming |
| left-hand `X` | pause/stop realtime VR motion streaming |

The XR buttons control the motion source only.
They do not replace the simulator/robot `start` and `A` buttons used to enter the tracking policy.

## UDP Motion Playback

Use this when `motion_source.type` is `udp`.
The runtime plays motions listed in `sim2real/config/<robot>/tracking.yaml`.

`src/motion_select.py` sends UDP commands to choose which configured motion to append:

```bash
cd <repo>/sim2real
uv run src/motion_select.py --robot g1
```

Use `--robot l7` for L7.
For G1 compliance, run the selector with the same tracking YAML as the policy:

```bash
uv run src/motion_select.py --robot g1 --tracking-config tracking_compliance.yaml
```

Selector commands:

- motion index or motion name: select a motion from `tracking.yaml`
- `list`: show available motions
- empty line: resend previous choice
- `r`: reload `tracking.yaml`
- `q`: quit

During policy execution, switch back to `default` before selecting another non-default motion.

## VR Realtime Motion

Use this when `motion_source.type` is `vr`.

First complete the XR/PICO setup and component map in `sim2real/teleop/README.md`.
At runtime, start the teleop bridge before entering the tracking policy:

```bash
cd <repo>/sim2real/teleop
uv run python serve_xrobot_teleop.py --robot g1
```

Use `--robot l7` for L7.
The teleop bridge publishes retargeted robot poses over ZMQ.
`src/deploy.py` reads the ZMQ addresses from `motion_source.vr` in `tracking.yaml`.

Recommended VR operation order:

1. start XRoboToolkit PC service
2. calibrate trackers/controllers in the PICO client
3. connect the PICO client to the PC service
4. start `serve_xrobot_teleop.py`
5. start sim2sim or G1 sim2real as described below
6. enter the tracking policy with simulator/robot `start`, then simulator/robot `A`
7. press right-hand XR `A` to start realtime VR streaming
8. press left-hand XR `X` to pause/stop VR streaming

Keep a stable standing posture during teleop startup so the retarget stack can estimate root height.

## G1/L7 Sim2Sim

Use this flow for both G1 and L7 simulation.
The examples below use G1; replace `g1` with `l7` in every command for L7.

Terminal 1 starts MuJoCo:

```bash
cd <repo>/sim2real
uv run src/sim2sim.py --robot g1
```

Terminal 2 starts the policy controller:

```bash
cd <repo>/sim2real
uv run src/deploy.py --robot g1
```

For G1 compliance instead of default G1 PMG:

```bash
uv run src/deploy.py --robot g1 --tracking-config tracking_compliance.yaml
```

After both terminals are running:

1. focus the MuJoCo/sim2sim window
2. press `s` to move from zero torque to the default pose
3. wait for the default-pose transition to finish
4. press `a` to enter the tracking policy
5. use the selected motion source:
   - UDP: use `motion_select.py` to choose motions
   - VR: press right-hand XR `A` to start live motion, left-hand XR `X` to pause
6. press `x` in the sim2sim window to stop

For UDP sim2sim, start `motion_select.py` in a third terminal after `deploy.py` is running.
For VR sim2sim, start `serve_xrobot_teleop.py` before pressing simulator `a`.

## G1 Sim2Real

G1 sim2real uses the standalone C++ bridge in `g1_sim2real/` plus the Python policy runtime in `sim2real/`.

Build the G1 bridge:

```bash
cd <repo>/g1_sim2real
bash scripts/build.sh
```

### G1 Network Setup

Recommended first setup: run both the Python policy runtime and the G1 bridge on your local machine, and connect the local machine to G1 with an Ethernet cable.

1. Connect G1 and the local machine with Ethernet.
2. Configure the local Ethernet interface with a static IP in the G1 subnet, for example:
   - IP address: `192.168.123.201`
   - netmask: `255.255.255.0`
3. Find the local Ethernet interface name, for example `enp3s0`, `enx...`, or similar.
4. Use that interface name as `G1_NET` when starting the bridge.

For example:

```bash
cd <repo>/g1_sim2real
G1_NET=enp3s0 bash scripts/run_bridge.sh
```

If you need onboard inference on G1, copy both runtime components to the G1 onboard computer:

- `sim2real/`: Python policy runtime
- `g1_sim2real/`: G1 C++ bridge

Then run both the policy runtime and bridge on G1.
In that setup, use the onboard Ethernet interface:

```bash
cd <repo>/g1_sim2real
G1_NET=eth0 bash scripts/run_bridge.sh
```

The G1 onboard CPU is relatively weak, so it is recommended to pin the runtime
processes to separate CPU cores. By default, the policy runtime pins ONNX
inference to cores `4-7`. The following split has worked well:

Terminal 1 starts the VR teleop bridge:

```bash
cd <repo>/sim2real
taskset -c 1 uv run teleop/serve_xrobot_teleop.py --robot g1
```

Terminal 2 starts the G1 low-level bridge:

```bash
cd <repo>/g1_sim2real
G1_NET=eth0 taskset -c 2-3 bash scripts/run_bridge.sh
```

Terminal 3 starts the Python policy controller:

```bash
cd <repo>/sim2real
taskset -c 4-7 uv run src/deploy.py --robot g1 --no-record
```

### G1 Runtime Startup

Terminal 1 starts the G1 low-level bridge:

```bash
cd <repo>/g1_sim2real
G1_NET=<dds-interface> bash scripts/run_bridge.sh
```

Terminal 2 starts the Python policy controller:

```bash
cd <repo>/sim2real
uv run src/deploy.py --robot g1
```

For G1 compliance instead of default G1 PMG:

```bash
uv run src/deploy.py --robot g1 --tracking-config tracking_compliance.yaml
```

After both terminals are running and the robot is ready:

1. confirm the robot is powered, supported, and safe to move
2. wait until `src/deploy.py --robot g1` reports that it has connected to robot state
3. press G1 remote `start` to move from zero torque to the default pose
4. after the default-pose transition, place/confirm the robot is safely on the ground
5. press G1 remote `A` to enter the tracking policy
6. use the selected motion source:
   - UDP: use `motion_select.py` to choose motions
   - VR: press right-hand XR `A` to start live motion, left-hand XR `X` to pause
7. press the configured stop/select button to exit

For UDP G1 sim2real, start `motion_select.py` in another terminal after `deploy.py` is running.
For VR G1 sim2real, start `serve_xrobot_teleop.py` before pressing remote `A`.

**CAUTION**: Always test a motion in sim2sim before running it on hardware.
