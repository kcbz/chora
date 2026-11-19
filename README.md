# Chora by Coherence

A local robot learning lab for Apple Silicon. An official SO-101 arm learns bowl
pick-and-place from a scripted expert in MuJoCo. Qwen reads structured scene
observations and emits high-level JSON actions. A numerical IK controller executes
the motion; a free-body bowl is held by actual contact dynamics, without a weld or
teleportation.

## Setup and publication

Chora's own code is proprietary; see [LICENSE](LICENSE). Third-party licenses
are recorded separately in [THIRD_PARTY.md](THIRD_PARTY.md).

For a fresh checkout, run `./setup.sh`. It retrieves all active robot models and
checksum-verified tiling textures. The unresolved imported knife is intentionally
not distributed; cutting scenes require a separately cleared local asset.
See [asset setup and wheel/workspace details](docs/ASSETS.md) and
[local development security](docs/SECURITY.md).

## Open the lab

From this folder:

```bash
source .venv/bin/activate
python -m chora serve
```

Open **http://127.0.0.1:8765**. Select the expert, choose a scene seed, and press
**Run episode**. The front and top cameras show the real simulation. The action
trace and pipeline cards display measured run outputs, not preset scores.

The live preview renders at 1280×960 with high-quality JPEG frames. Drag on the
scene to orbit, hold Shift while dragging (or right-drag) to pan, and scroll to
zoom. The **Front view** and **Top view** buttons return to fixed camera presets.
Camera controls work while paused and during playback, and never affect physics.

The **Appearance** selector at the top right of the navbar offers **Light**, **Dark**,
**Auto**, and **Customize…**. Light defaults to Porcelain; dark defaults to
Graphite. Auto follows the device's local clock: light from 07:00 to 19:00 and
dark overnight. Customize lets you assign any of the five palettes to either
mode and change both switching times. Choices persist in this browser and sync
across its Chora tabs. Vertical orbit dragging follows the revised direction.

The gallery, experiment report, and curation report share these appearance
settings and the navbar. Open them from **Gallery**, **Reports**, or the lab's
footer. Markdown and JSON exports remain available as downloads.

Each pipeline card explains its task and shows its own running indicator,
progress, and final status. Saved results stay labelled as the last completed
run while a new one is in progress. **Run details** expands the job log; it is
hidden when there is no log. Repeating collection reuses existing demonstrations
and explicitly reports when no new collection is needed.

## Fixed scenes

The selector contains nine fixed scenes. There is no scene-creation or copy UI.
The two accidental copy presets and their generated pages were removed; original
training data, adapters and historical evaluations were preserved.

| Scene | Setup |
| --- | --- |
| `so101-bowl` | Original trained bowl experiment, standard 55 mm platform |
| `so101-bowl-low-left` | Different bowl starts; platform at (150, 105) mm, 35 mm high |
| `so101-bowl-high-right` | Different bowl starts; platform at (190, 75) mm, 65 mm high |
| `ur10e-hammer` | Native UR10e with scanned claw hammer; uncalibrated impact trial |
| `fr3-cut-apple` | Local-only imported knife (license unresolved); tetrahedral apple flesh and elastic skin, material trial |
| `fr3-cut-tomato` | Local-only imported knife (license unresolved); tetrahedral tomato flesh and elastic skin, material trial |
| `fr3-cut-banana` | Local-only imported knife (license unresolved); tetrahedral banana flesh and elastic skin, material trial |
| `fr3-duo-tile-wipe` | Native two-arm FR3 Duo: suction workholding and sponge-contact coverage |
| `fr3-tile-installation` | Coordinated FR3 pair: adhesive application overlaps tile acquisition, followed by bedding |

Each scene has isolated outputs in `outputs/experiments/<id>/`; only the original
bowl scene currently has a trained student. Its old 82/100 result does not apply
to the new bowl layouts or tool tasks.

Tool scenes now use native FR3 (knife) and UR10e (hammer) articulated models,
with bounded torque controllers and visible installed-tool adapters. The hammer's
commanded downward task force is removed before contact; joint gravity and
orientation control remain enabled. This is not hardware impact qualification.
The new two-arm FR3 Duo task moves a free tile held by finite contact adhesion,
then wipes its face with an installed sponge. Coverage dots are contact samples,
not grout residue. See [real assets and the tiling plan](docs/REAL_OBJECTS_AND_TILING.md).

Food now uses **volumetric tetrahedral FEM** plus a separate elastic skin,
with different illustrative bulk stiffness for apple, tomato and banana. The
visual skin is bound to deforming nodes, with no exposed pre-cut faces before
damage. Contact work releases local cohesive bonds on a predefined plane.
This is a material-development trial: **no validated cutting score and no student
training**. Tomato flesh is a soft solid, not flowing juice. Apple and banana surface geometry/textures now come from CC0 Poly Haven assets;
interiors and the tomato surface remain procedural. The coarse FEM envelope is
still an approximation and does not inherit measured mechanics from a scan. See [material details](docs/MATERIAL_MODELS.md).
Recordings and measured quantities are available on each tool scene's Gallery
and Reports pages. The earlier 25/25 rigid-half results do not validate these
replacement models and are not carried over.

The previous mounted-hammer demonstration, transfer primitives and scene variations
remain historical source/output records but are excluded from the fixed selector.
New scenes can be added in source after their controller/physics are implemented;
the server rejects copy creation and selection of archived presets.

Use `--experiment <scene-id>` before a command, for example:

```bash
.venv/bin/python -m chora --experiment ur10e-hammer expert --record
```

### Different arms and two-arm systems

Active adapters are SO-101 for miniature bowls, FR3 for cooking material trials,
UR10e for hammering trials, and FR3 Duo for paired workholding/wiping. Tool masses,
contact laws and material values still require calibration. The original bowl
student does not control these new robots. Shared robot/control code lives in
`chora/arm_control.py`; original source robot assets are unchanged.

A new robot adapter must define its model, joints, limits, tool frames, motion
controller/IK, and gripper mapping. A task adapter defines observations, actions,
an expert, and success/failure checks. Two-arm adapters also need both mount poses,
arm identities, a shared collision world, coordinated actions, and task-specific
success checks. Register only implemented adapters in `chora/experiments.py` and
route the simulator/controller factory to the new implementation; the current
simulator explicitly rejects another robot type. Keep all new task results isolated.

Useful next task families include part sorting, peg insertion, fixture-assisted
fastening, ingredient sorting, stirring, and pouring. They require their own
geometry, controllers, and measured evaluations before claiming task competence.

The server binds to your own machine only. Stop it with Ctrl-C. You can also
double-click `Launch Chora.command` in Finder. Keep its Terminal window open.

## Environment

This project uses Python 3.12 in `.venv`, isolated from system Python. MLX handles
4-bit QLoRA training and inference on the M3 GPU. The simulation runs on the CPU;
rendering uses macOS OpenGL. A 16 GB M3 is the initial target. Model assets use
about 1 GB; raw data, videos, and dependencies require additional disk space.

To recreate the environment on an Apple Silicon Mac with Python 3.12 and Git:

```bash
./setup.sh
```

`requirements-lock.txt` records the exact installed packages from the first run.
`assets/menagerie` contains a sparse checkout of the official SO-101 model. See
`THIRD_PARTY.md` for sources, licenses, and the tested revision. We do not modify
the model files. Scene assembly adds a pedestal, table, target markings, bowl,
cameras, and a tool site. The scene retains the official yellow robot materials.

## Run each stage

```bash
# Environment and CPU tests
.venv/bin/python -m chora doctor
.venv/bin/python -m pytest -q

# Expert qualification: at least 90% of 50 scenes must pass
.venv/bin/python -m chora expert

# Optional native viewer: macOS requires the mjpython launcher
.venv/bin/mjpython -m chora expert --live --seed 0

# Record a real episode
.venv/bin/python -m chora expert --record --seed 0

# Collect demonstrations (resumes existing seeds without overwriting raw data)
.venv/bin/python -m chora generate --episodes 1200
.venv/bin/python -m chora curate

# Download model (~1 GB); subsequent training is local
.venv/bin/python -m chora download
.venv/bin/python -m chora train

# Continue a saved run, preserving previous adapters and metrics
.venv/bin/python -m chora train --resume --batch-size 4 --iterations 1200

# Freeze an adapter chosen on development/validation data, before testing
.venv/bin/python -m chora select --checkpoint 200

# Compare policies on the same 100 reserved test seeds
.venv/bin/python -m chora evaluate --policy expert
.venv/bin/python -m chora evaluate --policy base
.venv/bin/python -m chora evaluate --policy student
```

`./run_all.sh` runs the tests, qualification, data stages, model download, training,
and expert/student evaluation in order. It stops on a failure. No paid APIs are
used. Set optional `HF_TOKEN` in your environment if Hugging Face rate-limits the
download; never put credentials in source files.

For the rendering invariance test, run on the Mac with graphics access:

```bash
CHORA_RENDER_TEST=1 .venv/bin/python -m pytest -q
```

With the lab running, `CHORA_HTTP_TEST=1 CHORA_RENDER_TEST=1 CHORA_MLX_TEST=1
.venv/bin/python -m pytest -q` also checks the HTTP routes and the training loss
optimization. `python -m chora preview --checkpoint 200` records an immutable
checkpoint on development seeds 20000–20004, never the final test seeds.

macOS sandboxed/headless processes may have no Metal device or permission to open
localhost ports. Run these commands in a normal Terminal session in that case.

## Experiment design

- **Observation:** bowl position/yaw, target bounds, TCP position, gripper state,
  contact-based grasp indication, decision count, and previous action result.
- **Actions:** `move_to`, `open_gripper`, `close_gripper`, `done`. Strict JSON
  schemas reject unknown actions, extra fields, booleans, NaNs, and infinities.
- **Units:** simulation/data use metres; prompts, history, and student actions
  use integer millimetres. Thinking is disabled for both training and inference.
- **Controller:** top-down rim grasp, bounded numerical IK, complete straight-path
  validation before movement, joint position actuators, floor/table collision guard.
- **Success:** the entire bowl lies inside the target, upright within 15 degrees,
  resting on the table, nearly stationary, released, with no arm contact, throughout
  a 0.5-second physics interval. Every episode is capped at 15 decisions.
- **Curation:** successes only; discard invalid/rejected/collision/failed-grasp
  episodes, edge placements, and near-duplicate initial scenes. Rank by placement
  margin, path length, and decision count; cap each 4×4 position × 4 yaw bin.
- **Split:** whole episodes, stratified by bin. Generation uses seeds 0–9999;
  qualification uses 10000 onward; final evaluation uses 100000–100099. Final
  test scenes never select adapters or tune hyperparameters.
- **Training:** Qwen3-1.7B 4-bit, QLoRA rank 16, last 8 layers, batch size 1,
  assistant-only token loss, gradient checkpointing, greedy inference. The output
  vocabulary projection is computed only at assistant positions; a test checks
  loss and gradient equivalence against the full projection. Report
  validation loss every 100 steps, save adapters every 50. A 50-minute budget
  saves and stops at the next 10-step reporting interval. It is not an exact
  per-second deadline. Download/loading/tokenization are outside the training timer.

`configs/default.yaml` holds experiment settings. Changing physics, spawn ranges,
or the expert requires a fresh dataset; move previous `outputs/` aside first.
To restart from the base model, archive `outputs/adapter`, `outputs/training.json`,
and `outputs/training_metrics.jsonl` before starting. `--resume` automatically
archives the previous run under `outputs/training_runs/` and continues from its
weights with a fresh optimizer at half the configured learning rate. Pipeline stages use a file lock
to prevent simultaneous writes. Existing adapters are never silently overwritten.

## Outputs

| File | Contents |
| --- | --- |
| `outputs/oracle.json` | Expert qualification results |
| `outputs/data/raw.jsonl` | Immutable, resumable demonstration episodes |
| `outputs/data/curated/` | Training/validation chat pairs and selected episodes |
| `outputs/curation.json` | Counts, exclusions, bins, seed split, raw-data hash |
| `outputs/curation_report.md`, `coverage.png` | Curation summary and coverage plot |
| `outputs/adapter/` | Trained adapter and intermediate checkpoints |
| `outputs/training.json`, `training_metrics.jsonl` | Real training progress and metrics |
| `outputs/selection.json`, `training_curve.png` | Frozen adapter provenance and loss curve |
| `outputs/REPORT.md` | Summary of the measured experiment, including all training attempts |
| `outputs/results.json` | Separate expert, base, and student benchmark results |
| `outputs/evaluation/<policy>/` | Per-episode records, summary, selected MP4s |
| `outputs/eval_gallery.html` | Filterable gallery of all evaluated policies |
| `outputs/experiment_report.html`, `outputs/curation_report.html` | Themed experiment and dataset report pages |

Evaluate with `--record 100` to record every episode, or `--record 0` for no videos.
Use `python -m chora replay --policy student --failures 2` to refresh saved clips with current visual settings. Every observation, action result, and final outcome must match the scored trace before a clip is replaced. This also records the first two failed episodes for inspection.

`python -m chora thumbnails --policy all` renders final-state thumbnails for every
scored episode, checking each saved observation and outcome before publishing it.
The gallery shows a compact episode wall: hover recorded clips to preview, click
for video or a final-state still and the action trace. It clearly distinguishes
episodes without movies. The untrained-baseline tab stays disabled until a base-model
evaluation exists.

An evaluation with `--episodes 5` is labelled a partial benchmark, not a 100-trial
result. Native video files can be viewed through the lab's `/outputs/` routes.

## Scope

This first experiment is **imitation learning / behaviour cloning**, rather than
reward-based reinforcement learning. The agent receives exact simulator object
coordinates; it does not yet perceive camera images. One object and target are
used with randomized starts and small placement variations. The expert is a
scripted observation-based planner, not a frontier-model teacher. Strong expert
performance does not guarantee a strong student. Camera perception, recovery
data/DAgger, additional tasks, and sim-to-real are subsequent experiments.

The browser lab, simulator, data pipeline, and MLX integration were independently
implemented for Coherence. The project direction was inspired by the supplied
prompt and [distill_frontier](https://dhruvsirohi-arch.github.io/distill_frontier/).
No source code or reported results from that project are included.

## UI sources

`chora/web/lab-template.html` and `lab.js` define the lab. `app.css`,
`navbar.html`, `appearance-dialog.html`, and the appearance JavaScript files
are shared by all HTML pages through `chora/ui.py`. One canonical
`logo-octahedron.svg` supplies the brand mark. CSS and JavaScript load from shared
`/static/` resources; served reports refresh their navbar and appearance dialog
from the same components on every request. `index.html` is a generated
snapshot. `python -m chora report` refreshes generated gallery/report artifacts.
Job lifecycle and progress live in `outputs/active_job.json` and `outputs/jobs/`.

## Robot-arm selection study

A separate comparison of Panda, FR3, UR5e, UR10e, iiwa 14 and dual FR3 is available
in [docs/ROBOT_ARM_EVALUATION.md](docs/ROBOT_ARM_EVALUATION.md) and through
[the local report](http://127.0.0.1:8765/outputs/robot_evaluation/index.html).
It evaluates poses, collisions and hypothetical loads and does not establish
hardware/task success. The follow-up implementation uses FR3, UR10e and FR3 Duo
in the new scenes; see [real objects and tiling](docs/REAL_OBJECTS_AND_TILING.md).

## Source-backed tiling material baseline

[The tiling material workbench](http://127.0.0.1:8765/materials/tiling) now records
QuarryBasics Ironspot 31IS actual dimensions, LATICRETE 254 PLATINUM and PERMACOLOR
product data separately from simulation assumptions. [Implementation/status](docs/TILING_BASELINE.md)
documents the remaining installation and fracture work. A finite-volume 2x2
material sequence and contact-coupled two-arm residue cleanup are implemented;
full floor/wall installation and a tiling student are not. Tile mass is estimated,
not manufacturer-measured. Physical calibration is deferred, not a prerequisite
for simulation-only development.

The **FR3 pair · Adhesive and tile bedding** scene extends the shared tiling
workcell: arm B applies each finite adhesive patch while arm A acquires and lifts its
tile on the same physics timeline. Local clearance allows placement and bedding;
the applicator parks after its final patch. The spatial mortar support,
shear and volume transport laws are explicit simulation assumptions. No cured
bond, grout finishing or fracture is implemented, and Qwen training remains
disabled. See [the task interface](docs/TILING_TASK_INTERFACE.md) for model limits,
checks and the remaining stages.

Live viewing offers **1×, 2× and 4×** at the upper right of the simulation viewport.
Multipliers are targets relative to simulated time; the UI reports achieved speed
because physics, rendering and model inference can limit throughput. Speeds can
change during an episode. The viewer reduces rendering frequency at higher
speeds; physics timestep, actions, material clocks and force limits stay the same.

