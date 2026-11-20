# Camera-based VLA integration plan

Status: in progress, November 19, 2026. Fixed-tick SO-101 recording/replay and the
neural model/M3 feasibility milestone are implemented; see [VLA development](VLA.md)
and [M3 measurements](VLA_M3_FEASIBILITY.md). The pilot dataset/trainer and bounded
fit checks are implemented; see [training](VLA_TRAINING.md). The bounded
synchronous policy runner is implemented and its pilots still fail
placement; see [diagnostics](VLA_POLICY_DIAGNOSTICS.md). A competent policy,
production datasets, delay-aware runner, UI integration and two-arm VLA policies
remain planned.

## Objective and boundaries

Add a compact vision-language-action policy to Chora that consumes camera frames,
robot proprioception and an instruction, and predicts short chunks of joint targets.
Establish the pipeline on SO-101 bowl placement, then extend it to the two-arm FR3
tiling scene, individual tiling skills and a continuous multi-stage installation episode.
Preserve the Qwen macro-action pipeline as a separate baseline.

Reference: `dhruvsirohi-arch/distill_frontier`, branch `vla-mujoco`, particularly
`docs/vla-latency/index.html`, `RESULTS.md` and the VLA model card. The reference
reports a 39.7M-parameter DINOv2-S / CLIP / transformer / flow-matching policy,
16-action chunks at 25 Hz and a four-tick execution horizon. Its RTX 4090 median
inference is 10.3 ms, or 6.7 ms with compiled fp32. These are external reported
results, not Chora targets already achieved or expected M3 measurements.

No license was found for the reference repository. Default to independently written
Chora implementation using published methods and appropriately licensed dependencies.
Before directly importing reference code or redistributing weights, establish permission
covering the company-owned work and record scope and attribution in THIRD_PARTY.md.
The reference weights are absent from Git and task-specific; they are optional comparison
artifacts, not required to build Chora's own pipeline.

## 1. Fixed-tick simulation and recording

Create `chora/vla/` with task, observation and action contracts. Add adapters around
existing models and dynamics rather than duplicate the scenes. Start with SO-101.

- `reset(seed)` and `step_tick(action)` advance a configured control interval;
  initial target 25 Hz, with MuJoCo substeps and explicit timestamp accounting.
- Observations: scene and wrist RGB images (initially 224 x 224), named joint positions
  and velocities, actuator/tool state, instruction, episode ID and simulation timestamp.
  Images come from fixed policy cameras without UI overlays; browser orbit is independent.
- Actions: bounded joint position targets for SO-101 and its gripper. Robot adapters
  own actuator mapping, units, limits and controller behavior. Do not assume actuator
  indices equal qpos indices. Reject nonfinite/wrong-shape commands and handle timeouts.
- Instrument the existing expert inside its movement executor: record the actual target
  and resulting observation at each control tick. Recording only macro endpoints or
  interpolating them after the rollout is insufficient. Keep policy rendering read-only.
- Privileged object/material state stays in expert generation and scoring, never silently
  enters student inputs. Log low-level clipping, collisions and controller interventions.

Acceptance: deterministic fixed-seed reset and tick stepping; consistent image/state/action
alignment; a recorded expert episode can replay with comparable placement and contact
outcomes; policy camera recording does not alter physics. Existing Qwen tests still pass.

## 2. M3 feasibility gate and model skeleton

Add a separate optional VLA dependency group and environment/lock, leaving the existing
MLX environment intact. Use PyTorch MPS initially, CPU for diagnostics and CUDA optionally.
Pin compatible versions after local smoke tests; do not assume the reference's lock is portable.

Build a small DINOv2-S visual encoder, frozen CLIP instruction encoder, multimodal fusion
transformer and flow-matching action head. Start with a 16-tick chunk, four-tick execution
horizon and four flow steps. Cache instruction embeddings and reuse eligible history-frame
tokens. Respect upstream weight licenses and record model revisions/checksums.

Implement device-aware synchronization, random generation, precision and checkpoint loading.
Use fp32 as the initial correctness baseline; enable other precision/compile modes only where
supported and measured. Include image normalization, joint normalization, padding and
per-robot action masks in the saved model contract.

Before a large run, test forward pass, backward pass, save/load and one synthetic inference
loop on the M3. Measure model/optimizer peak memory, rendering, image preprocessing, inference
p50/p95/p99 and training step time. GPU timing must synchronize the selected backend.
The 40 ms control interval is an engineering target, not a promise. If missed, first measure
smaller image/history/head settings and frame caching; document any lower replan frequency
and its effect on success. Do not claim faster reaction from a longer action chunk.

Acceptance: a compatible model trains and loads locally within a documented memory budget;
produce an M3 feasibility report with settings and a measured training-time estimate. Pause
large collection/training if basic compatibility or memory fails; fix the measured bottleneck.

## 3. Visual demonstrations and training

Keep VLA outputs under experiment-scoped `vla/` directories with unique run IDs, separate
from Qwen data, adapters and published results. Save source/config fingerprints, camera
calibration, control rate, joint ordering, instruction, timestamps, tool state and checksums.
Use chunk-friendly episode storage; choose format after measuring image I/O and disk cost.

Split by episode/scene seed before constructing windows. Reserve development and final test
seeds separately; compute normalization only on training data. Chunk tails have loss masks
and never cross resets or dataset splits. Start with a small successful expert collection,
prove a tiny dataset can be overfit, then scale based on learning curves. Record failures
and curation counts rather than silently dropping them.

Train imitation learning with flow-matching loss, initially frozen vision/text backbones and
small batches with accumulation. Evaluate closed-loop on development seeds during training;
select checkpoints without final test feedback. Save optimizer, random state, split/normalizer
and config for resume. Unfreeze vision blocks only when needed and within measured budget.
This phase is supervised imitation, not RL or Qwen LoRA training.

Acceptance: overfit smoke test, reproducible resume, no split leakage, and a camera-only student
that completes unseen bowl placements. Report success with episode counts/uncertainty,
collisions and invalid actions against the expert and original Qwen on comparable scenes;
do not announce parity before it is measured.

## 4. Closed-loop runner and latency evaluation

Run fixed-rate joint control independently of model inference. The policy emits bounded
chunks; a runner owns their timestamps, current queue, observation history and cancellation
on instruction change. One owner mutates MuJoCo state. Coordinate renderer/context ownership;
do not share mutable MjData between inference and simulation threads.

Provide two explicitly labeled evaluation modes:

1. Synchronous throughput: simulator waits for inference, useful for reproducibility and
   comparison with the reference report.
2. Delay-aware closed loop: physics continues with the last valid controller target/queue
   while inference runs. Timestamp observations, consume actions against an explicit chunk
   start time and discard expired chunks. When the action horizon runs out, invoke a logged
   bounded hold/stop policy instead of concealing the missed deadline.

Separate compute timing from capture/render/preprocess and action-application timing. Report
cold start, warm p50/p95/p99/max, observation age at actuation, deadline misses, simulation
real-time factor, memory and task success. Include rendering on/off and a bounded sustained
run for thermal drift. Sweep flow steps (1/2/4), execution horizons (1/2/4/8) and controlled
delay/jitter; reserve the final test set until settings are selected. Static bowls alone
will not expose reaction limits: add a separately scored moving-target or perturbation test.

Acceptance: tests prove physics advances during a deliberately delayed inference, that stale
chunks and underruns are handled, and that instruction changes invalidate queued actions.
Publish M3 results with full hardware/settings, not a comparison that equates unlike models.
Browser 1x/2x/4x playback remains a display control, independent of inference/control rates.

## 5. Lab, gallery and reports

Integrate through existing jobs, experiment registry, navbar, appearance components and gallery.
Offer explicit policy choices: Scripted expert, Qwen macro policy and Visual VLA. Disable actions
with actionable reasons when data, weights or task adapters are unavailable. Show collection,
training and evaluation progress, cancellation and per-run errors with existing running indicators.
Display success, latency, missed deadlines and policy camera views in reports. Identify whether
an episode is synchronous or delay-aware. Link model/dataset provenance and material limitations.

Acceptance: user can collect, train, evaluate and watch VLA runs from the lab without CLI-only
steps; interrupted jobs resume or fail clearly; existing appearance and baseline artifacts survive.

## 6. Two-arm tiling skills and full sequence

Reuse the same tick/record/train/evaluate contracts with a separate FR3-pair robot adapter and
separately trained checkpoints. Start with scene plus wrist views for each arm; benchmark the
additional camera cost. Use a coordinated action vector for both arms with task-specific tool
channels, per-channel limits and masks. FR3s use torque-bounded tracking of joint references;
never send position targets straight to torque actuators.

First learn dry tile pickup/transfer/placement, then adhesive application and bedding separately,
then overlapping application and tile pickup/placement. Distill the expert's actual simultaneous
motion, not a serialized list of commands. Score each arm and their shared outcome: grip,
collision, coverage, workable adhesive, placement, spacing, load limits and elapsed sim time.
Document which tool channels are continuous or discrete and how discrete commands are trained,
validated and applied; do not treat vacuum/dispensing as extra arm joints.

Retain the eight-stage tiling task interface and persistent scene. Expose instruction-conditioned
skills with explicit completion checks and cancellation. A stage coordinator initially sequences
learned skills; the arms can overlap inside compatible stages. The coordinator and physics
scorer are separate from the learned policy. Later train longer-horizon chained episodes and
measure recovery; describe staged skill execution and a learned end-to-end policy accurately.

Current tiling implements adhesive/bedding prototypes, not every installation stage. Inspection,
material preparation, back preparation, finishing, setting and joint grouting need explicit scene,
tool, consumable and scoring implementations before demonstrations for those stages can exist.
Simulation-only assumptions remain visible. Physical calibration is not a gate for this plan;
VLA training does not validate grout rheology, cure, fracture or hardware realism. Floor work
comes first; wall tiling requires its own support/adhesion dynamics, expert and evaluation.

Acceptance: unseen-seed paired-arm skills, then one persistent multi-tile episode with meaningful
concurrency and stage-level results. The eventual full installation benchmark includes every
implemented stage and failures, not only a successful-looking video.

## Delivery order and effort

Deliver in reviewable milestones: fixed-tick expert replay; M3 model feasibility; trained visual
bowl baseline; delay-aware benchmark and UI; paired-arm dry placement; adhesive/bedding overlap;
remaining installation stages and full chaining. Each milestone produces runnable artifacts and
measurements before expanding scope.

The first milestones are substantial development, not a switch in configuration. Estimate compute
and later delivery time from the M3 smoke test and initial dataset throughput. Full tiling is the
largest and least predictable phase because it includes missing environment behavior as well as
policy learning. No 4090 speed guarantee, trained tiling policy or completion date is implied.
