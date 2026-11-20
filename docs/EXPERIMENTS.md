# Experiments and developer reference

Run the commands below from the project root after setup. This reference covers
the supported learning pipeline and technical details; development changes and
historical results live in [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md).

The trained pipeline applies to the SO-101 bowl task. Material-development and
tiling scenes are simulation trials; their student pipelines remain disabled.
Knife-dependent scenes require a separately cleared local asset.

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

## Task and robot adapters

An adapter defines the robot model, joints, limits, tool frames, controller/IK
and gripper mapping. A task defines observations/actions, an expert and explicit
success/failure checks. Two-arm tasks also require mount poses, arm identities,
a shared collision world and coordinated actions. Register implemented adapters
in `chora/experiments.py` and the simulator/controller factories. Keep outputs
and benchmark claims separate by task.

## Pick-and-place scenes and their students

- **Standard platform** is the original fixed-target scene and uses the existing
  trained Qwen adapter.
- **Block · Raised shelf** uses a solid block and a 95 mm cobalt shelf. Both
  are farther from the arm than in the first shelf preset. The expert pinches
  the block and carries it across the work area.
- **Moving platform** uses a motorized slide with up to 32 mm of travel to either
  side of center. The marked goal moves with the platform. A successful release
  stays upright, fully inside the goal, and stable relative to the platform
  throughout the settle check. The structured observation includes the measured
  platform velocity and a predicted goal position to plan the release. Each
  seed selects a starting phase and a peak speed between 16 and 24 mm/s.
  The forecast uses measured position, velocity, and the known rail frequency.
  Curation groups starts by pickup pose, rail position, and direction.

Each scene has separate demonstrations, adapters, and evaluation results. In the
lab, select **Qwen · State-based imitation** and choose **Go to training**. Pass
the expert check, collect and curate demonstrations, then train the student.
After training, **Run episode** becomes available. **Evaluate student** measures
generalization on reserved scenes.

## UI architecture

`chora/web/lab-template.html` and `lab.js` define the lab. Shared navigation,
appearance controls and `logo-octahedron.svg` are assembled through `chora/ui.py`.
CSS and JavaScript load through `/static/`. Reports refresh their navigation and
appearance components from the same source. The scene and control method in the
lab URL carry through to Gallery and Reports. `python -m chora report` refreshes
the generated report and gallery artifacts.

## Runtime diagnostics

MuJoCo can create `MUJOCO_LOG.TXT` for warnings and errors in the process working
directory. It is a generated diagnostic file, not a source/configuration file or
a changelog. It stays ignored by Git. Existing diagnostic copies may be archived
under `outputs/logs/`; the engine may create another root copy on a later warning.
Logs can be removed when they are no longer needed for debugging. Removing a log
does not resolve the model or solver warning that produced it.

## Related references

- [Asset setup and wheel/workspace behavior](ASSETS.md)
- [Material models and their limitations](MATERIAL_MODELS.md)
- [Tiling task and coordination](TILING_TASK_INTERFACE.md)
- [Robot evaluation](ROBOT_ARM_EVALUATION.md)
- [Third-party sources and notices](THIRD_PARTY.md)
