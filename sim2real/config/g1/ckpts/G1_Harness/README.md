# G1_Harness

Whole-body tracking checkpoint for the HITCH harness on a Unitree G1 (29 joints, 50 Hz).

> **Not for teleoperation.** Do not drive this checkpoint from VR / XR / PICO input or from retargeted human
> motion. Use `G1_PMG` or `G1_Compliance` for those.

This checkpoint is trained specifically for the harness:

1. **Hands.** It is trained for a G1 that carries Dex3 hands or rubber hands.
2. **Clean references only.** It is trained on the smooth, physically consistent reference motion that the
   harness plans. It has not seen the noise, jitter and retargeting artefacts of live teleoperation input, so it
   is not compatible with teleoperation and may make the robot fall when driven that way.

## Files

| File | Content |
| --- | --- |
| `policy.onnx`, `policy.json` | the policy and its input / output keys |
| `deploy.yaml` | joint order, gains, default pose, action scale and observation layout |

`deploy.yaml` replaces a hand-written `tracking.yaml`: the harness reads it and writes the controller and
tracking YAML for `src/deploy.py` itself, with `motion_source.type: "external"`.
