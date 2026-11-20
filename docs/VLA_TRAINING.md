# VLA pilot dataset and training

The current implementation supports development-pilot imitation training and a
tiny-data fit check. A [bounded synchronous learned-policy runner](VLA_POLICY_DIAGNOSTICS.md)
now supplies actual development task outcomes; those pilots still fail placement.
Production training splits and a competent controller remain. A separate
experimental lab panel can now run existing checkpoints; training controls remain
command-line tools.

## Prepare a dataset

Use visual recordings from [VLA recording](VLA.md). Preparation verifies recording
checksums, image/state/action alignment, camera/runtime contracts and timestamps.
It rejects repeated scene seeds, incompatible recordings and missing images.
Failed demonstrations are audited as exclusions, not silently counted as targets.

```bash
.venv-vla/bin/python -m chora.vla.dataset \
  outputs/vla/runs/bowl-tick-pilot-v1 \
  --output outputs/vla/datasets/bowl-pilot-v1
```

Both output directories must be new when creating recordings/datasets. Multiple
recording directories may be passed together. Episode splits are deterministic
with `--split-seed` and `--validation-fraction`. At least two successful episodes
are required. Pilot preparation accepts development seeds 20000..29999; it never
consumes original final-benchmark seeds.

The dataset references the source episode files through relative paths, preserving
checksums instead of duplicating images. Keep source recordings with the dataset.
The dataset fingerprint covers the episode assignments and normalization. Moving
the containing workspace together preserves relative paths.

Splitting happens before window creation. Each action chunk starts at one tick,
contains up to 16 actual targets and never crosses an episode boundary. Incomplete
tails are zero-padded in normalized coordinates with a loss/evaluation mask.
Earlier camera history is four ticks back, clamped to frame zero at the start
of an episode. No history comes from a prior episode or future observation.

Joint position, velocity and absolute action normalization each use means and
standard deviations from training episodes only, with a 0.01 scale floor.
Validation uses those saved statistics without fitting its own. Final post-action
observations are not included in the normalization fit. Six named scalar joints
are supported; this is not yet the paired-arm dataset adapter.

## Run a tiny fit check

```bash
.venv-vla/bin/python -m chora.vla.train \
  --dataset outputs/vla/datasets/bowl-pilot-v1 \
  --device mps --steps 300 --batch-size 4 --overfit-windows 8
```

The default deliberately selects eight evenly spaced training windows, including
episode tails, to check learning without a large collection. `--overfit-windows 0`
uses all windows in the training split; it does not change the pilot dataset into
a final benchmark. Every update samples training windows only, with Gaussian
noise/time samples for flow matching, gradient clipping and AdamW. The current
trainer uses a fixed learning rate (default 0.0003).

Frozen DINOv2 features are computed once per source episode and cached under
`.cache/vla/features/`. Cache keys include input checksums, actual vision-weight
hash, encoder revision, pooling, model source and PyTorch version. Existing cache
files are checksum-verified. Train and validation can both cache pretrained image
features without fitting on validation data. A small feature-file LRU bounds host
memory as episodes change. This cache is incompatible with image augmentation or
vision fine-tuning; those are not implemented by this trainer.

The instruction embedding is cached. This bowl dataset contains one instruction;
learning it does not establish language grounding, instruction switching or
generalization to other tasks.

## Reports and checkpoints

Runs live in unique `outputs/vla/training/` directories. `report.json` records
split seeds, model/dataset/source signatures, initial/final metrics and status.
`metrics.jsonl` stores flow loss/gradient norm per update. `evaluations.jsonl`
stores deterministic action-prediction metrics at evaluation intervals.

Evaluation uses the fit windows (or up to 32 spaced training windows) and up to
32 validation windows. Reported errors are masked normalized action MSE and mean
absolute joint-target error in radians. They are not placement success rates.
Windows from one episode are correlated, not independent robot trials.

Checkpoints include model/vision tensors and cached text in safetensors, model
configuration and normalization in JSON, plus AdamW tensors and the dedicated
noise RNG in `optimizer.safetensors`. Optimizer groups, NumPy sampling RNG and step
are in `optimizer.json`. No executable pickle is needed. Checkpoints publish by
directory rename after all files are written; incomplete `.partial` directories
are not resume targets. Checkpoints are marked `runtime_approved: false` and
`closed_loop_evaluated: false`.

There is no validation-based checkpoint selection yet: runs use a fixed update
budget and retain interval checkpoints and the final one. Validation diagnostics
do not feed gradients or select a published final-test result.

## Resume into a new run

```bash
.venv-vla/bin/python -m chora.vla.train \
  --dataset outputs/vla/datasets/bowl-pilot-v1 \
  --device mps --steps 1000 --batch-size 4 --overfit-windows 8 \
  --resume outputs/vla/training/RUN/checkpoints/step-000300
```

`--steps` is the total desired update count, so this adds 700 updates. Resuming
creates a new run and retains the previous one. Dataset fingerprint, source,
model settings, learning rate, batch size, seed and fit-window count must match.
Learning histories remain in both runs; a resumed report's initial metric is its
resume point, not the original random model. Keep the pinned environment too;
cross-runtime numerical reproducibility is not guaranteed.

An exact CPU optimizer/RNG continuation is covered by tests. The initial M3 run
also verified step 250 -> 300 against uninterrupted training: all 365 model/text
tensors, 421 optimizer/RNG tensors and optimizer metadata matched exactly in that
specific test. This is not a guarantee of bitwise results on other hardware.

## Initial M3 learning checks

Development episodes 20000 and 20001 were assigned to training; 20002 to
validation. The validation metrics below use 32 correlated windows from that one
episode, not 32 independent trials. All runs used frozen encoders, batch four,
four-step deterministic action sampling and learning rate 0.0003.

| Fit check | Updates | Training normalized action MSE | Validation normalized action MSE |
|---|---:|---:|---:|
| Eight windows | 300 | 0.1534 | 0.6126 |
| Same eight windows, continued | 1000 total | 0.1361 | 0.9348 |
| One fixed window, fresh model | 300 | 0.0197 | 1.1277 |

The eight-window random-model error began at 1.3733; the single-window error began
at 0.9067. The single-window check reduced prediction error by 97.8%, showing that
the optimizer/model can learn an actual recorded action chunk. It is deliberately
memorization, not evidence that visual input is causally used or that a task was
learned. The eight-window continuation plateaued and validation worsened. Do not
treat it as a successful multi-observation overfit or a reason to scale training
without diagnosing the remaining conditioning/prediction error.

The first 300-update eight-window loop took 28.1 seconds; the 700-update
continuation took another 53.9 seconds. Frozen feature preparation/model loading
are excluded from those loop times. Mean training joint-target errors were
0.0886 rad after 300 eight-window updates and 0.0351 rad in the single-window check;
these are not sufficiently precise to assert successful physical placement.

Local ignored runs: `outputs/vla/training/bowl-overfit-v1/`,
`bowl-overfit-1000-v1/`, `bowl-single-window-v1/` and the resume verification in
`bowl-overfit-resume-v1/resume-verification.json`. None has been executed as a
learned control policy or selected on final test scenes.

## Verify

```bash
.venv-vla/bin/python -m pytest -q \
  tests/test_vla_dataset.py tests/test_vla_training.py tests/test_vla_model.py
```

Tests cover disjoint episode splits, training-only normalization, chunk masks,
reset-local history, corruption/duplicate/overwrite rejection, exact optimizer/RNG
continuation and the neural model contracts.

Subsequent [policy diagnostics](VLA_POLICY_DIAGNOSTICS.md) identified and corrected
weak joint conditioning and action-sequence averaging, then tested actual bounded
rollouts. They improved the fit but did not establish successful placement. More
varied demonstrations and checked recovery examples are next. A low training error
alone cannot justify scaling to two-arm tiling or claiming a learned bowl skill.

## Relative-arm target format

Dataset preparation now accepts `--action-representation relative_arm`. This
creates protocol `chora-vla-pilot-chunks-v2` and fits action normalization on
training windows at the declared chunk length. At each window, the first five
joint targets are offsets from that window's measured starting joint pose. The
gripper target remains absolute. At inference, the measured pose is added back
once for all actions in that chunk before actuator/slew guards are applied.

This is a different supervised target representation, not visual smoothing or
an alteration of physics. Old v1 datasets and checkpoints keep absolute targets.
The normalization declares its representation; an unknown representation or
incompatible chunk length is rejected. Train/validation splitting and masks stay
unchanged. Both encoders remain frozen in this implementation.

CLI training now participates in the lab's pipeline lock and publishes progress,
so normal lab actions wait while visual policy training uses the GPU. A training
run finishing still requires a separate actual rollout evaluation.
