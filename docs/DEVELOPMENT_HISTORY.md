# Development history

This file records project evolution and historical results. It is not a guarantee
of current capabilities. Consult the task-specific documents and saved run traces
for the scope and limitations of each implementation.

## 2026-11-17–18 — Local learning baseline

- Established the Apple Silicon Python environment, MuJoCo simulation, local lab
  and MLX/Qwen imitation-learning pipeline.
- Built SO-101 bowl pickup and placement with contact-based grasping, scripted
  expert demonstrations, curation, adapter training and held-out evaluation.
- The saved original bowl student scored 82/100. This result belongs to that
  recorded experiment and does not apply to other scenes or robots. Trained
  weights, datasets and generated reports remain local artifacts.
- The project direction was inspired by the supplied prompt and
  [distill_frontier](https://dhruvsirohi-arch.github.io/distill_frontier/).
  Its source code and reported results were not imported into Chora.

## 2026-11-18 — Lab and task development

- Added orbit/pan/zoom controls, shared navigation, the octahedron logo and
  configurable light/dark appearance across the lab and reports.
- Replaced scene-copy creation with a fixed scene catalog and isolated outputs.
  Original bowl records were preserved as new tasks were introduced.
- Evaluated additional robot models and introduced native FR3 and UR10e adapters
  with bounded torque control. Imported surface assets did not provide measured
  mechanics or hardware-qualified tool attachments.
- Replaced early rigid-half cutting illustrations with material-development
  models using coarse FEM, skin and predefined cohesive release. Historical
  rigid-half scores do not validate these replacements. Cutting remains
  uncalibrated and does not model flowing tomato interiors or realistic fracture
  in all foods. See [MATERIAL_MODELS.md](MATERIAL_MODELS.md).
- Added a two-arm prepared tile-cleanup trial and source/assumption records for
  tile, adhesive and grout. The cleanup coupon was not a full installation scene.

## 2026-11-19 — Shared tiling workcell

- Introduced a persistent workcell with two independently mounted FR3 arms, a tile
  rack and a 2 × 2 placement pattern. Dry transfer preceded adhesive/bedding.
- Added finite adhesive volume, material age, assumed support/shear response,
  ridge redistribution and tracked squeeze-out.
- Replaced serial application/placement and repeated full parking moves with
  shared-timeline tile acquisition and adhesive application, local clearance,
  placement and bedding. The applicator parks after its final patch.
- The recorded coordinated fixture completed 53 actions in 140.708 simulated
  seconds, versus 73 actions in 210.749 seconds for the serial fixture. It recorded
  60.371 seconds of simultaneous arm motion and 14.708 seconds of pickup/lift
  motion during dispensing. All four tiles passed the declared model checks;
  these are deterministic regression results, not hardware validation or a
  held-out expert benchmark.
- Added viewport 1×/2×/4× target playback controls and measured-speed feedback.
  Physics timesteps and material clocks remain independent of viewing speed.
- Added photo-based tile appearance, cosmetic bevel/relief and a continuous mortar
  visualization driven by the volume grid. These do not establish realistic wet
  material mechanics. See [TILING_TASK_INTERFACE.md](TILING_TASK_INTERFACE.md).

## 2026-11-19 — Repository and publication preparation

- Renamed the project workspace to `chora` and connected the GitHub origin.
- Hardened local HTTP requests, corrected robot setup and packaged web resources,
  and added checksum-verified runtime texture retrieval.
- Excluded the unresolved knife binaries, publisher PDFs, generated logs and
  texture caches from publishable history while retaining development copies.
  Fresh clones explicitly mark knife-dependent scenes unavailable.
- Kept Coherence-authored code proprietary. Any future release license remains
  TBD; third-party materials retain their own terms.
- Separated the project overview, experimental reference, history and attribution
  documentation. Generated runtime logs remain outside version control.

## Current boundaries

The bowl task has the supported imitation-learning pipeline. Tiling does not yet
have a trained student, cured bond, grout finishing or tile-fracture model.
Physical calibration is deferred. Earlier demonstrations, visual improvements
and simulated task success must not be described as evidence of hardware safety
or validated real material behavior.

## 2026-11-19 — Visual policy recording foundation

- Added an independently written fixed-tick SO-101 adapter, ideal scene/wrist policy
  cameras, bounded named joint targets and an expert that executes IK references
  at 25 Hz. The Qwen macro pipeline and saved baseline were retained.
- Added isolated development-pilot recording with N actions/N+1 observations,
  timestamps, source/runtime/config and camera provenance, checksums and exact
  joint-trajectory/final-state physics replay. Student observations exclude bowl
  and target coordinates; expert and scorer can still access them.
- The initial 224-pixel pilot completed 3/3 development episodes (seeds 20000–20002)
  with successful placement and verified replay. This is a recording-path check,
  not a trained VLA result or a held-out success estimate.
- Verification: 37 focused non-rendering checks passed with five opt-in checks
  skipped; the simulation/VLA suite with graphics enabled passed all 30 checks.
  Recording with cameras and without cameras produced identical trajectories.
- Neural VLA implementation, M3 model performance measurements, training and lab
  integration remain subsequent milestones. No reference-repository code or
  weights were imported. See [VLA.md](VLA.md).

## 2026-11-19 — Neural VLA and M3 feasibility

- Added a separate pinned PyTorch environment and independently implemented 39.2M
  visual policy: frozen DINOv2-S, cached CLIP text, multimodal fusion and a
  flow-matching action-chunk decoder. No reference-repository code was copied.
- Verified offline model contracts, finite MPS gradients and safe checkpoint
  save/load with identical predictions. The original MLX environment remains
  separate. Three training updates are only a compatibility smoke test.
- On Apple M3, warm four-step inference with earlier-image features cached measured
  33.1 ms median; one-step inference 25.7 ms. Recomputing history took 52.6 ms.
  Live two-camera capture plus four-step inference took 60.3 ms, exceeding a
  40 ms tick. All figures are batch one, fp32, 50 repeated observations per setting.
- These measurements support proceeding with dataset/trainer development, not a
  claim of trained task success or a proven real-time closed-loop system.
  Details: [VLA_M3_FEASIBILITY.md](VLA_M3_FEASIBILITY.md).

## 2026-11-19 — VLA pilot dataset and trainer

- Added whole-episode training/validation splitting before chunking, training-only
  normalization, reset-local camera history and masked action-chunk tails. Source
  checksums and dataset fingerprints protect the saved data contract.
- Added a frozen-vision feature cache and flow-matching trainer with per-update
  diagnostics, validation action errors, safe checkpoints and optimizer/sampler/RNG
  resume. Existing Qwen datasets, checkpoints and benchmark outputs were retained.
- Verified an actual M3 resume from update 250 to 300: all model, optimizer and RNG
  tensors matched the uninterrupted run exactly in that test. Nine focused data,
  optimizer and model checks passed; the combined VLA/simulation suite passed
  36 checks with three graphics checks skipped.
- A single recorded action chunk reduced normalized prediction error by 97.8%
  over 300 updates. Eight-window training improved initially but plateaued by
  1000 updates while validation error worsened. This is not a successful learned
  bowl controller or a multi-window overfit result; no learned rollout was scored.
  Measurements and next diagnostics are in [VLA_TRAINING.md](VLA_TRAINING.md).

## 2026-11-19 — VLA conditioning and actual learned rollouts

- Diagnosed weak use of joint state and nearly constant predictions across each
  action chunk. Added explicit joint-state broadcast and fixed sinusoidal action
  positions, retaining legacy checkpoint behavior through default-false flags.
- At a matched 1000-update eight-window budget, normalized training action MSE
  decreased from 0.1361 to 0.01664. Predicted within-chunk motion recovered; this
  offline improvement did not establish successful robot control.
- Added a bounded synchronous learned-policy runner with timestamped history and
  action queues, logged actuator/slew limits, invalid-output stops, provenance
  checks and actual camera/state traces. It has no expert fallback and is not
  asynchronous or delay-aware. The same guarded expert completes both tested
  development scenes.
- Sparse learned checkpoints stalled or collided. Training on all 263 windows
  from the two pilot demonstrations eliminated collisions in the two checked
  scenes but still completed 0/2 placements, with either four-tick or every-tick
  replanning. Recorded videos show the actual unsuccessful learned behavior.
- Verified exact initial runtime/recorded joint alignment and first-prediction
  agreement within 7.5e-8 rad. The focused VLA/simulation suite passed 47 tests
  with three opt-in graphics tests skipped. Model/optimizer/task artifacts from
  the original Qwen baseline remain separate.
- Updated M3 measurements and detailed limitations are recorded in
  [VLA_POLICY_DIAGNOSTICS.md](VLA_POLICY_DIAGNOSTICS.md). Broader demonstrations
  and checked recovery examples remain next; no competent VLA policy is claimed.

## 2026-11-20 — Browser VLA tests and task-sized Git workflow

- Moved completed work into task-sized commits and pushes, and documented the
  ongoing development workflow. Corrected the isolated VLA dependency lock.
- Sized scene selects to their chosen label and inset a shared chevron, verified
  in dark/light appearance and at mobile width.
- Added an experimental Visual VLA panel: discovered local checkpoints, development
  seed selection, live scene/wrist frames, loading progress, Stop and run reports.
  The lab launches the separate VLA environment and prevents conflicting jobs.
- Verified an actual browser-driven cancellation and a complete 240-tick test;
  the UI truthfully reported unsuccessful placement at the time limit. Existing
  Qwen scores and control paths remain separate. Visual data/training commands
  remain CLI-only, and no successful VLA placement is claimed.

## 2026-11-20 — Failed-pilot correction and 50-demonstration check

- Investigated the reported shaking: raw learned targets changed abruptly between
  replans, and the two-demo policy never commanded its gripper closed in that run.
  Removed failed/untested checkpoints from automatic default selection and added
  score labels tied to unseen-scene, weight and execution-setting evidence.
- Collected 50 successful expert episodes, split 40/10, introduced versioned
  relative-arm targets and trained a bounded 3000-update replacement. It still
  completed 0/10 development placements, plus 0/1 on the user's seed; every-tick
  replanning scored 0/3 on an overlapping subset. This did not fix the controller.
- Kept all generated results separate from the working Qwen baseline. Serialized
  CLI evaluation with lab GPU jobs and exposed its real progress/outcomes.
  See [VLA_FAILURE_ANALYSIS.md](VLA_FAILURE_ANALYSIS.md) for evidence and limits.

## 2026-11-20 — Remove internal experiments from the product interface

- Removed the dropdown of overfit/conditioning/action-order training runs from
  the main UI and stopped returning unqualified runs as model choices there.
- The Visual policy section now shows a compact In development status while no
  model qualifies. Failed experiment files remain available to developer CLI
  tools and documentation, without occupying a product model picker.
- Restricted UI execution to the single qualifying model; direct requests cannot
  launch a failed diagnostic checkpoint. This does not change the failed VLA's
  task performance or claim a working controller.


## 2026-11-20 — Select control methods in the lab

- Put Scene and Control method together. The expert, Qwen and VLA use the same
  simulation workspace. Method selection determines available controls and
  which results the gallery and reports show.
- Added status displays for planned visual inspection and skill orchestration.
  Removed the separate visual-policy panel and the stale HTML lab snapshot.

## 2026-11-20 — Make training and saved results easier to find

- Grouped the two lab selectors, added scene-specific Qwen availability, and
  linked the run controls to the learning pipeline for trainable scenes.
- Fixed episode media links in scoped galleries by resolving them against the
  scene's output folder. Missing media leaves the outcome and action trace visible.


## 2026-11-20 — Distinct bowl scenes and full-width viewer

- Replaced the two similar variants in the picker with a 95 mm shelf and a
  moving landing platform. Both experts completed 50 of 50 check episodes.
- The carriage has a simulated slide joint and the scorer checks the bowl
  relative to the moving target. Standard expert/Qwen traces still replay.
  All 53 archived standard-scene VLA rollouts passed physics replay.
- Matched the viewport to the scene feed's aspect ratio so the image fills the
  card without cropping.

- Replaced the shelf bowl with a solid block and moved both the pickup and
  cobalt shelf farther out. The block expert passed 100 held-out scenes.
- Made the carriage teal and revised the moving expert's forecast and release
  clearance. It passed 100 fresh held-out scenes across all six position and
  direction combinations, plus 49 of 50 routine check scenes. Moving-platform
  curation now accounts for rail position and direction.

- Unified the SO-101 presentation palette for platforms, moved the training
  shortcut into the setup panel, and added active-job feedback there.
- Enabled fresh Qwen retraining with archival of the previous model and scores.
  Evaluations now save a final-state preview for every scored episode.
- Renamed the planned hierarchical mode Skill Orchestrator and recorded the
  first tiling pilot architecture.


## 2026-11-20 — Visual inspection and comparable expert benchmarks

- Added local VLM inspection of verified overhead frames. The pilot matched
  geometry on 11/12 bowl and 9/12 block examples.
- Ran the same 100 held-out expert seeds for the block and moving bowl. Each
  completed all 100. The Standard expert benchmark remains 97/100.
- Added a direct-action VLA pilot. It placed the bowl on one familiar seed
  and collided on the second. Faster replanning timed out on both seeds. The
  checkpoint did not qualify for use from the lab.
