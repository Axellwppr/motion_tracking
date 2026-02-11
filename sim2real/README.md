
# Sim2Real Deployment

### Install & run with uv (recommended)
`uv` handles isolated envs + lockfiles; it will auto-read `pyproject.toml` in this folder.

```bash
cd sim2real
# create .venv and install deps from pyproject/uv.lock if present
uv sync

# run scripts through uv to pick up the venv automatically
uv run src/sim2sim.py --xml_path assets/g1/g1.xml      # sim bridge
uv run src/deploy.py --net lo --sim2sim                # controller (sim)
uv run src/motion_select.py                            # motion selector
```

## Install
- Create conda environment
  ```
  conda create -n gentle python=3.10
  conda activate gentle
  ```
- Install the Unitree SDK2 Python bindings in virtual environment (follow the [official Unitree guide](https://github.com/unitreerobotics/unitree_sdk2_python))
- Install Python deps:
  ```bash
  pip install -r requirements.txt
  ```



## Run Sim2Sim
1. Start the simulator (state publisher + keyboard bridge):
   ```bash
   python3 src/sim2sim.py --xml_path assets/g1/g1.xml
   ```
   Leave the terminal focused so the keyboard mapping works.
2. In another terminal launch the high-level controller:
   ```bash
   python3 src/deploy.py --net lo --sim2sim
   ```
3. Flow:
   - Controller waits in zero-torque mode until it receives the simulated state.
   - Press `s` in the sim terminal to let the robot move to the default pose.
   - Press `a` in the sim terminal to start tracking policy
   - See [Motion Switching](#motion-switching) to replay different motions.
   - Press `x` to exit gracefully.

## Run Sim2Real
1. Power on G1 and connect to your PC via Ethernet cable.
   - Set your PC's Ethernet interface to a static IP in the `192.168.123.x` subnet.
2. Launch the controller pointing at the appropriate interface:
   ```bash
   python3 src/deploy.py --net <robot_iface> --real
   ```
3. The state machine matches Sim2Sim but with `physical remote controller` input
   - Zero torque
   - (Press `start`) → move to default pose
   - Place robot on the ground
   - (Press `A`) → run the active policy
   - See [Motion Switching](#motion-switching) to replay different motions.
   - (Press `select`) → exit gracefully

**⚠️ Always test motions in Sim2Sim before running them on the real robot.**

**⚠️ Do not blindly trust the RL policy. Always have emergency stop measures and qualified safety personnel on site.**

## Motion Switching
- The tracking policy accepts motion change commands while it is active.
- Open a terminal and run the motion selector:
  ```bash
  python3 src/motion_select.py
  ```
- Usage tips:
  - Type the motion name or its index (`list` prints the menu). Press Enter with an empty line to resend the previous choice.
  - `r` reloads the YAML file if you edit it; `q` exits the selector.
- Selection rules:
  - The policy only starts a new motion when the current clip has finished and the robot is in the `default` clip (or you explicitly request `default`).
  - Sending `default` always fades back to the idle pose.
