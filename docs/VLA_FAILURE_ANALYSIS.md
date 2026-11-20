# Failed VLA pilot and replacement check

November 20, 2026. The current visual policy is not a working bowl controller.
The failed checkpoint should not have been offered as a usable default. Speed,
training-loss and infrastructure checks did not establish task competence.

## Reported oscillation

The user's run is preserved locally in
`outputs/vla/ui-runs/20261120-034655-af38cb76/` (seed 20003). It ran the two-demo
`bowl-dense-pilot-v1` checkpoint and timed out after 240 ticks without placement.
No actuator-bound or target-slew interventions occurred, so those guards did not
cause this particular motion. Median raw target change at replanning boundaries
was 0.0695 rad, versus 0.0252 rad within chunks. The gripper target stayed between
0.3369 and 0.4207 rad throughout; the expert's closing command is -0.17 rad.
The learned policy never commanded a proper close in that run.

This is a learned-control failure. The simulation waits during inference; changing
browser playback speed does not correct those targets. No expert fallback or
animation substitutes for the failed policy.

## Bounded replacement experiment

- Collected 50 independent seeded expert episodes (21000..21049); all 50 physically
  completed placement in the unchanged SO-101 scene.
- Split whole episodes into 40 training and 10 validation scenes. Training contains
  5,254 action ticks, with the gripper closed in 52.5% of labels.
- Added an explicit relative-arm action format: five arm targets are offsets from
  the observation's measured pose; the gripper target stays absolute. Normalization
  uses training data only, and chunk tails remain masked. Old absolute checkpoints
  retain their original interpretation.
- Trained the 39.2M-parameter policy for 3,000 updates, batch eight, with all training
  windows and frozen vision/text encoders. The optimization/evaluation loop took
  834.7 seconds on the M3, excluding encoder-feature preparation. Final validation
  normalized action MSE was 0.0471; mean joint-target error was 0.0279 rad over 32
  spaced validation windows. These are action errors, not successful placements.

Dataset: `outputs/vla/datasets/bowl-50-relative-v1/`. Model and training trace:
`outputs/vla/training/bowl-50-relative-v1/`.

## Actual simulation results

| Check | Scenes | Placements | Collisions |
|---|---|---:|---:|
| Guarded expert feasibility | 25000..25009 | 10/10 | 0 |
| Replacement, execute four ticks before replanning | 25000..25009 | 0/10 | 2 |
| Replacement, repeat the user's seed | 20003 | 0/1 | 0 |
| Replacement, replan every tick | 25000, 25005, 20003 | 0/3 | 0 |

The one-tick tests overlap the other tests and are not additional independent
scenes. Each episode had the same 240-tick budget and joint guards. No final Qwen
benchmark seeds were used. The replacement commands the gripper closed and can
move the bowl, but still misses grasps, loses placement alignment or stalls.
More frequent replanning did not establish successful control in the checked cases.
Adding demonstrations and changing targets did not solve the controller.

Local reports are under `outputs/vla/rollouts/bowl-50-relative-development-v1/`,
`bowl-50-relative-regression-v1/` and `bowl-50-relative-every-tick-v1/`.
The guarded expert comparison is `outputs/vla/evaluation/bowl-50-v1/expert.json`.
Generated images, models and traces remain ignored local artifacts.

## UI correction

The panel is now **VLA development diagnostics**. Failed checkpoints show their
actual score and are not auto-selected. With no qualifying model it says
**No usable VLA yet**. Explicit diagnostic selection remains possible; expert and
Qwen controls remain separate.

A recommendation requires at least ten distinct unseen-to-dataset development
scenes, at least 80% placement success, no collisions, matching checkpoint-weight
hashes and the same execution horizon used by the UI. Training and validation
seeds, stopped runs and repeated trials of one seed do not satisfy that count.
The current replacement fails this check and is not recommended. This gate is a
basic simulation development criterion, not a physical validation or final benchmark.
CLI training and evaluation share the lab job lock and publish progress.

The remaining engineering problem is precise, closed-loop reach/grasp/lift/place
behavior. It needs investigation at those transitions and checked corrective
examples, not another declaration of readiness from action loss alone. No working
VLA has been delivered by this experiment.
