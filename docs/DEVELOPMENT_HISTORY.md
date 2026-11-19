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
