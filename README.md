# G1 Compliance Motion Tracking

This branch contains the G1 tracking + compliance implementation for whole-body
motion tracking. It is inspired by the Gentle Humanoid codebase and is kept as
the `compliance` branch of
[`Axellwppr/motion_tracking`](https://github.com/Axellwppr/motion_tracking).

The focus of this branch is:

* training a G1 whole-body motion tracking policy,
* supporting upper-body **compliance-aware** behavior for contact-rich interaction,
* supporting live **VR teleoperation** through the bundled sim2real/teleop stack.

For the HEFT PMG/WPC training implementation, use the `main` branch. For the
released deployment runtime and checkpoints, use the `sim2real` branch.

The simulation and training backend is based on **mjlab**.

A demo of the G1 compliance policy is available
[here](https://motion-tracking.axell.top).

https://github.com/user-attachments/assets/263dd3cc-8d23-4d67-bd36-37fe89f525de

https://github.com/user-attachments/assets/4d210dbf-8023-4270-b094-ab6a2353deda

Instructions for this branch's deployment and runtime usage are available in the
[`sim2real`](./sim2real) folder.

## Installation

If you do not have `uv` installed, you can install it following the instructions in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# this project uses uv for dependency and environment management
uv sync
```

## Motion Dataset Preparation

<details>
<summary><b>Quick Start: Download Preprocessed Dataset (Google Drive)</b></summary>

Download link:

- [Google Drive Dataset](https://drive.google.com/drive/folders/1-FBUxllaYwqGIUSCaWg_4inD-u5Tdvi9?usp=sharing)

After downloading, extract it into the repository `dataset/` directory, and you should have the following structure:

```text
dataset/
  amass_all/
    meta.json
    ...
  lafan_all/
    ...
```

</details>

<details>
<summary><b>Build Dataset from AMASS/LAFAN with GMR</b></summary>

### Retargeting with GMR

We use GMR to retarget the [AMASS](https://amass.is.tue.mpg.de/) and [LAFAN](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) datasets. The output format is a dataset containing a series of npz files with the following fields:

- `fps`: Frame rate
- `root_pos`: Root position
- `root_rot`: Root rotation in quaternion format (xyzw)
- `dof_pos`: Degrees of freedom positions
- `local_body_pos`: Local body positions
- `local_body_rot`: Local body rotations
- `body_names`: List of body names
- `joint_names`: List of joint names

You can use the [modified version of GMR](https://github.com/Axellwppr/GMR) to directly export npz files that meet the requirements.

You should organize the processed datasets in the following structure:
```
<dataset_root>/
    AMASS/ACCAD/Female1General_c3d/A1_-_Stand_stageii.npz
    ...
    LAFAN/walk1_subject1.npz
    ...
```

### Dataset Building

Modify `DATASET_ROOT` in `generate_dataset.sh` to point to your dataset root directory, then run the script to generate the dataset:
```
bash generate_dataset.sh
```

The dataset will be generated in the `dataset/` directory, and the code will automatically load these datasets. You can also use the `MEMATH` environment variable to specify the dataset root path.

</details>

## Training

You can use the provided `train.sh` script to run the full training pipeline. Modify the global configuration section in `train.sh` to set your WandB account and other parameters, then run:

```bash
bash train.sh
```

Under standard settings, training takes approximately 15 hours on 4× A100 GPUs.
If GPU memory is constrained, it is recommended to appropriately tune the `NPROC` and `num_envs` parameters in `train.sh` and `cfg/task/G1/G1.yaml`, respectively.
Such adjustments may increase training time and could affect training performance to some extent.

## Evaluation

```bash
uv run scripts/eval.py --run_path ${wandb_run_path} -p # p for play
uv run scripts/eval.py --run_path ${wandb_run_path} -p --export # export the policy to onnx (sim2real)
```

If you export a deployment policy, the exported checkpoint will be written under `scripts/exports/<task-name>-<timestamp>/`.

To use it in the deployment runtime:

1. Copy the exported policy folder (including `policy.onnx`, `policy.pt`, and `policy.json`) into `sim2real/assets/ckpts/`.
2. Update `sim2real/config/tracking.yaml` so that `policy_path` points to the new ONNX file.

For the actual deployment-side test procedure:

- see [`sim2real/README.md`](./sim2real/README.md) for `sim2sim/sim2real` testing

That README also explains how to use the UDP motion selector and the VR motion source during deployment.
