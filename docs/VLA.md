# Visual VLA development

Implemented: fixed-tick SO-101 expert recording/replay and a compact neural VLA
skeleton with measured M3 inference, training-step and checkpoint smoke tests.
A pilot chunk dataset and resumable imitation trainer are also implemented; see
[VLA training](VLA_TRAINING.md). A competent closed-loop visual policy, production
training/evaluation splits remain to be implemented. The lab now includes an
experimental VLA test panel; collection/training controls remain command-line tools.
The [bounded synchronous runner](VLA_POLICY_DIAGNOSTICS.md) is implemented, but
the current pilot has not completed bowl placement in its development rollouts.
See the [integration plan](VLA_INTEGRATION_PLAN.md) for the remaining milestones.

## Record a development pilot

Use the existing Chora environment and SO-101 assets; no PyTorch/model download is
needed for recording. Run from the project root on a Mac with graphics access:

```bash
.venv/bin/python -m chora vla-record --episodes 3
```

This creates a unique run under `outputs/vla/runs/` for the selected bowl experiment.
The optional `--output` must name a new directory. `--image-size` defaults to 224.
`--state-only` omits images for physics diagnostics and does not produce visual
training data. Other robots/tasks are rejected explicitly.

Pilots use development seeds 20000..29999, beginning at 20000 by default. They are
not the final benchmark or a training split. A later dataset collector will own
separate train/validation/final-test episode splits.

The adapter executes six named position-actuator targets at 25 Hz, eight MuJoCo
steps per tick at the existing 5 ms physics timestep. It translates the existing
expert's Cartesian planner into IK targets actually executed at each tick. The
timing and labels therefore differ from Qwen's macro-action dataset. Existing
Qwen data and checkpoints are not migrated or overwritten.

Policy input is two RGB views (`front`, `wrist`), measured joint positions and
velocities, an instruction, tick index and simulation timestamp. Policy frames
have no UI overlays. Target coordinates and bowl state are available to the expert
and scorer separately, not to student observations. The wrist camera is an ideal
simulated optical camera, not a claim about installed physical camera hardware.

## Files and alignment

Each run contains:

- `manifest.json`: runtime/source/config fingerprints, named actuator/joint order,
  camera mounts/FOV/resolution, timing contract, collection status and outcomes.
- `episode-<seed>.npz`: arrays loaded with `allow_pickle=False`: N applied actions,
  N+1 joint position/velocity observations and timestamps, N+1 frames per camera,
  expert phase labels, and final full qpos/qvel for replay checks.
- `episode-<seed>.json`: outcome, seed, frame count, image availability and NPZ checksum.

Transition alignment is `observation[t] -> action[t] -> observation[t+1]`.
The initial timestamp is after the existing reset/settling phase. Every subsequent
observation is exactly 40 ms later. All six targets are radians, including the
SO-101's angular gripper joint. Out-of-range, nonfinite and malformed targets are
rejected before advancing physics; there is no silent action clipping.

Placement is scored over the configured stable interval. Collision and tick limits
terminate the tick environment; unsuccessful expert exhaustion is retained as a
failure. Rendering does not drive time or change state. These are synchronous
expert demonstrations; a delay-aware inference runner has not been implemented.

## Verify replay

```bash
.venv/bin/python -m chora vla-replay outputs/vla/runs/RUN/episode-20000.npz
```

Replay checks checksum and source/runtime/config provenance, executes the stored
joint targets without calling the expert, compares every joint observation and
timestamp, and verifies final full physics state and outcome. It does not regenerate
or compare pixels. If the runtime or relevant source changed, regenerate the pilot
rather than silently treating it as a compatible trajectory.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_vla_ticks.py tests/test_simulation.py
CHORA_RENDER_TEST=1 .venv/bin/python -m pytest -q tests/test_vla_ticks.py tests/test_simulation.py
```

The opt-in graphics checks require a usable Mac graphics context. They compare
image-enabled and state-only expert trajectories and test both policy cameras.
Pilot success demonstrates a working data path, not learned-policy performance
or the RTX 4090 latency reported by the reference project.

## Isolated neural VLA environment

Keep the existing `.venv` for the lab and MLX/Qwen. The VLA uses `.venv-vla`:

```bash
uv venv --python .venv/bin/python .venv-vla
uv pip install --python .venv-vla/bin/python -r requirements-vla-lock.txt
.venv-vla/bin/python -m pytest -q tests/test_vla_model.py
```

The lock records the tested macOS arm64 Python 3.12 environment. The package also
defines an optional `vla` dependency group; it is not needed for normal Chora use.
Environment creation should be done once; do not recreate an active environment.

`configs/vla.yaml` pins the official DINOv2-S and CLIP model revisions and neural
architecture. The independent Chora implementation is 39.2M policy parameters:
frozen DINOv2 image features, CLIP text features computed/cached separately, four
fusion-transformer layers and four action-decoder layers. It combines current and
earlier scene/wrist views, plus measured joint position/velocity. Images are
ImageNet-normalized and patch features pool to 8 x 8 tokens per view. It predicts
16 absolute joint-target actions through a flow-matching velocity field. Frozen
CLIP adds 63.2M offline text-encoder parameters, not included in the 39.2M policy.
This is an independently implemented architecture in the same family as the
reference, not a checkpoint-compatible copy or a reproduction of its results.

## Run the feasibility benchmark

```bash
.venv-vla/bin/python -m chora.vla.benchmark \
  --episode outputs/vla/runs/RUN/episode-20000.npz \
  --device mps --reps 50 --warmup 10 --train-steps 3 --render
```

Omit `--render` for inference/training only. `--device cpu` is a diagnostic path;
`auto` selects MPS, CUDA or CPU when available. Outputs get a unique directory in
`outputs/vla/feasibility/`, or use `--output` with a new directory name.

First use downloads the pinned upstream encoder weights into Chora's ignored
Hugging Face cache. DINOv2 loads safetensors. The official CLIP repository uses
PyTorch weights, loaded with restricted `weights_only=True`; no remote Python
implementation is executed. The policy checkpoint uses safetensors and JSON,
including vision weights, architecture, normalization, cached instruction and
model revisions. Loading it does not download encoders. It is not yet an optimizer
resume checkpoint or a runtime language-switching policy wrapper.

The benchmark checks real recorded images, action shape/finiteness, finite
training gradients and parameter updates, then prediction equivalence after
checkpoint save/load. It sweeps 1/2/4 flow steps and cached/uncached earlier visual
features. The cache is explicitly supplied in this harness; a timestamp-aware
cache and asynchronous controller still need implementation in the runner.

The three training updates are a hardware smoke test on one pilot window, not
task training. Vision/text are frozen. No learned actions are sent to a robot or
scored as bowl-placement success. Changing flow steps is only a speed comparison
until we measure trained-policy success at each setting.

See [M3 feasibility results](VLA_M3_FEASIBILITY.md) for measured timings and limits.
The [pilot trainer](VLA_TRAINING.md) provides episode-level splits, masked chunk
datasets and resumable training. [Policy diagnostics](VLA_POLICY_DIAGNOSTICS.md)
cover conditioning fixes and bounded development rollouts. More varied training
and recovery examples are needed; a competent controller and delay-aware runner
are not yet established.

## Test from the browser

Start the normal lab with `.venv/bin/python -m chora serve` and open
[http://127.0.0.1:8765/](http://127.0.0.1:8765/). Select **Bowl · Standard platform**,
then use the **VLA development diagnostics** panel below the main simulation. Choose a discovered
checkpoint, set a development seed in 20000..29999 and click **Test VLA**.
No failed or untested checkpoint is selected as a usable default. A recommended
default requires at least ten distinct scenes outside that dataset's training and
validation episodes, at least 80% placement success and no collisions in those
trials. Reported weight hashes must match the checkpoint. This is a basic simulation
development check, not a final benchmark or physical validation.

The server launches one bounded synchronous episode in `.venv-vla`, showing live
scene/wrist views and progress, inference timing and the actual outcome. **Stop**
requests cancellation; **Open run results** links the saved report. Viewer orbit
and playback-speed controls belong to the separate main simulation, not these
fixed policy-camera views. Test results do not update the Qwen benchmark scores.

Only completed local training checkpoints with matching datasets appear; arbitrary
paths cannot be supplied through the browser. Missing dependencies/checkpoints or
unsupported scenes disable testing with an explanation. Concurrent live episodes,
scene switching and pipeline jobs are blocked during a VLA test. Source/runtime
provenance and all existing action guards still apply. No expert fallback is used.

UI runs live under `outputs/vla/ui-runs/`, with atomic camera/progress snapshots,
the regular report/episode traces and a sibling process log. This is inference
and inspection UI, not a visual-policy data collection/training interface.
Restart an already-running lab after updating server code.

Failed checkpoints are labelled with their actual success counts. Repeated runs
of one seed do not count as independent scenes; stopped runs are excluded. Without
a qualifying checkpoint, the UI says **No usable VLA yet** and diagnostics require
an explicit checkpoint choice. The expert and Qwen baseline remain available in
the main controls. These defaults address the observed oscillating two-demo pilot.
