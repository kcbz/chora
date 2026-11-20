# VLA conditioning and learned rollout diagnostics

Implemented November 19, 2026: recorded-observation diagnostics, two explicit
model-conditioning changes and a bounded learned-policy simulation runner.
The current pilot still does not complete bowl placement.

## Findings and corrections

The original model's eight-window normalized action MSE plateaued near 0.14.
Two measured effects explained much of the problem:

1. Joint state was weakly used. At the original 300-update checkpoint, permuting
   the eight camera observations increased MSE from 0.1534 to 1.7835. Permuting
   joint position/velocity changed it only to 0.15345. The original design supplied
   one joint token alongside 256 visual tokens and text. The correction broadcasts
   the projected joint state across decoder memory after fusion, giving it a direct
   path to the action decoder. This does not add privileged object coordinates.
2. Action order collapsed. Predicted movement within a 16-action chunk had
   normalized variance around 0.00018, while the expert's within-chunk variance
   was 0.1193. The model was largely predicting an average pose repeatedly. The
   correction adds fixed sinusoidal action-position features alongside learned
   positions. The input now gives the decoder a stronger distinction between
   earlier and later targets.

Both changes are explicit `configs/vla.yaml` model flags: `proprio_broadcast`
and `sinusoidal_action_positions`. Old checkpoints default to `false` when those
fields are absent, preserving their original inference behavior. Checkpoint tests
cover both new flags. Source/model changes intentionally invalidate strict training
resume contracts; use a fresh training run for changed architectures.

Train/eval predictions matched exactly in the checked cases. Increasing Euler
flow steps from four to sixteen did not remove the original plateau. Runtime
initial joint state matched the recording exactly; the first corrected prediction
matched the offline one within 7.5e-8 rad. Those checks found no substantial initial
state, normalization, actuator-order or train/eval discrepancy.

These are sensitivity/fit diagnostics on a small training set, not proof of
general visual grounding. In the corrected 1000-update model, camera permutation
raised MSE from 0.01664 to 0.04114; joint permutation raised it to 1.8314.
Replacing the earlier frame with the current frame barely changed error on these
samples. Do not infer robust use of visual history or language from this pilot.

## Matched training checks

All runs used the same two training episodes, one validation episode, batch four,
learning rate 0.0003, frozen vision/text encoders and four flow steps.

| Model / fit set | Updates | Training MSE | Validation MSE |
|---|---:|---:|---:|
| Original, eight windows | 300 | 0.1534 | 0.6126 |
| Joint broadcast only, same eight windows | 300 | 0.1432 | 0.4725 |
| Both corrections, same eight windows | 300 | 0.0550 | 0.4764 |
| Original, same eight windows | 1000 | 0.1361 | 0.9348 |
| Both corrections, same eight windows | 1000 | 0.01664 | 0.4459 |
| Both corrections, all 263 training windows | 1000 | 0.0467 | 0.4070 |

The matched 1000-update eight-window error decreased by about 88%. Corrected
within-chunk prediction variance reached 0.1329, rather than remaining nearly
constant. The all-window pilot evaluates 32 spaced training windows, so its
training MSE is not directly comparable with the eight-window metric. Every
validation row uses 32 correlated windows from episode 20002; that is not an
estimate over 32 independent scenes. Checkpoints were fixed-budget endpoints,
not selected by final-test results.

## Bounded actual rollouts

`chora.vla.runner` executes the model's actual predicted joint targets in MuJoCo.
It never calls the expert to replace an action. Student input remains clean RGB,
measured joint position/velocity and the cached instruction. Privileged bowl state
is used only by the environment scorer and reported outcomes.

- Joint targets pass through actuator limits and a 3 rad/s target slew limit
  (0.12 rad per 40 ms tick). Raw and applied targets and intervention counts are
  recorded separately; clipping is not hidden.
- Nonfinite predictions, wrong action shapes, skipped observations, unexpected
  timestamps and instruction changes stop/invalidate the queue. Only the cached
  bowl instruction is supported currently.
- Timestamped current/past camera features are reused when available. A prediction
  yields 16 actions and the default runner executes four before replanning.
- Environment success, collision and tick limits end the episode. The same guarded
  expert succeeds on both diagnostic seeds, verified by tests.
- Source/runtime/physics provenance must match the recording. The runner may use
  a shorter diagnostic budget without weakening the other provenance checks.

The runner is **synchronous**: simulation waits during inference. It is not the
planned asynchronous delay-aware controller, a real-time guarantee or a physical
robot safety system. The learned checkpoints remain experimental and unapproved
for lab default-policy selection.

Fixed diagnostic scenes: 20000 (a training scene) and 20003 (a separate development
scene), with a 240-tick/9.6 simulated-second limit. No final test seeds were used.

| Policy | Replan every | Seed 20000 | Seed 20003 | Placements |
|---|---:|---|---|---:|
| Original 300-update sparse model | 4 ticks | Time limit | Time limit | 0/2 |
| Joint-only 300-update sparse model | 4 ticks | Collision at tick 45 | Collision at tick 36 | 0/2 |
| Original 1000-update sparse model | 4 ticks | Time limit | Time limit | 0/2 |
| Corrected 1000-update sparse model | 4 ticks | Collision at tick 29 | Collision at tick 110 | 0/2 |
| Corrected 1000-update full-trajectory pilot | 4 ticks | Time limit, no collision | Time limit, no collision | 0/2 |
| Same full-trajectory pilot | 1 tick | Time limit, no collision | Time limit, no collision | 0/2 |

The full-trajectory pilot moves toward the bowl but does not complete the grasp
and placement. More frequent replanning did not fix the stalls in these two
scenes, and caused more slew interventions. Two training demonstrations remain
far too little evidence to claim a usable learned controller. The tests also
show why offline action MSE cannot substitute for task-success evaluation.

## Updated M3 speed check

The updated architecture still has 39.2M policy parameters. In the subsequent
20-sample hardware check, four-step cached-history inference measured 38.5 ms
median / 42.8 ms p99; one-step measured 30.3 ms median. Live two-camera capture
plus four-step inference measured 74.9 ms median. These differ from the earlier
[initial feasibility run](VLA_M3_FEASIBILITY.md); configuration and machine
conditions changed, so this is not an isolated speed-regression experiment.
There was no thermal soak or live paced control-loop benchmark. The four-step
tail already exceeds a 40 ms tick before capture, and serial capture/inference
does not support a claim of 25 Hz real-time end-to-end operation.

## Commands and artifacts

```bash
.venv-vla/bin/python -m chora.vla.diagnose \
  --checkpoint outputs/vla/training/RUN/checkpoints/step-001000 \
  --dataset outputs/vla/datasets/bowl-pilot-v1 \
  --output outputs/vla/training/RUN/diagnostics.json

.venv-vla/bin/python -m chora.vla.runner \
  --checkpoint outputs/vla/training/RUN/checkpoints/step-001000 \
  --dataset outputs/vla/datasets/bowl-pilot-v1 \
  --seeds 20000 20003 --max-ticks 240 --execute 4
```

The runner creates a unique directory under `outputs/vla/rollouts/` or accepts a
new `--output` directory. It writes `report.json` and per-episode NPZ files with
actual camera frames, raw/applied actions, joint observations, timestamps and
final full qpos. A report status of `complete` means the diagnostic finished,
not that the task succeeded; inspect `successes`, trial reasons and interventions.

Recorded local runs include:

- Training: `bowl-joint-conditioning-v1`, `bowl-action-order-v1`,
  `bowl-action-order-1000-v1`, `bowl-dense-pilot-v1` under `outputs/vla/training/`.
- Rollouts: `bowl-baseline-v1`, `bowl-joint-conditioning-v1`,
  `bowl-baseline-1000-v1`, `bowl-action-order-1000-v1`,
  `bowl-dense-pilot-v1`, `bowl-dense-every-tick-v1` under `outputs/vla/rollouts/`.
- `bowl-dense-pilot-v1/episode-20000.mp4` and `episode-20003.mp4`: actual saved
  scene/wrist frames played at simulated time, with explicit unsuccessful outcome
  labels. They are not re-rendered or edited robot motions.
- Updated hardware report: `outputs/vla/feasibility/m3-conditioning-order-v1/`.

All generated weights, frames and videos remain ignored local artifacts.

## Next work

The representation problems are substantially improved, but robust control is
still unresolved. Next collect a modest, more varied set of full expert
demonstrations and verify recovery from small disturbances before collecting
correction examples. Keep training/validation/final-test scenes separated and
evaluate actual rollouts after training. Do not expand to two-arm tiling or claim
task success based on these two-scene failed pilots.
