# Whole Body Motion Tracking

This repository is an extension of the [GentleHumanoid](https://gentle-humanoid.axell.top) training repository, supporting universal, robust, and highly dynamic whole-body motion tracking policy training.

Main features:

*  A **universal** whole-body motion tracking policy,  training and evaluation pipeline.

* Dataset support including preprocessing for AMASS and LAFAN.

A **demo** of the pretrained policies, shows a single model generalizing across diverse and highly dynamic motions, is available [here](https://motion-tracking.axell.top).

Instructions for **real-robot deployment** and the use of **pretrained models** on new motion sequences are available in `sim2real` folder.

## Installation

```bash
# 1) clone IsaacLab for editable installation
git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
git -C third_party/IsaacLab checkout v2.2.0
# 2) install dependencies via uv
uv sync
```

## Motion Dataset Preparation

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

## Training

You can use the provided `train.sh` script to run the full training pipeline. Modify the global configuration section in `train.sh` to set your WandB account and other parameters, then run:

```bash
bash train.sh
```

Under standard settings, training takes approximately 14 hours on 4× A100 GPUs.
If GPU memory is constrained, it is recommended to appropriately tune the `NPROC` and `num_envs` parameters in `train.sh` and `cfg/task/G1/G1.yaml`, respectively.
Such adjustments may increase training time and could affect training performance to some extent.

## Evaluation

```bash
uv run scripts/eval.py --run_path ${wandb_run_path} -p # p for play
uv run scripts/eval.py --run_path ${wandb_run_path} -p --export # export the policy to onnx (sim2real)
```
