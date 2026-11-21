# Visual VLA development

Chora has fixed-tick SO-101 expert recording, replay, and an experimental
visual imitation trainer. Its trained pilot failed the closed-loop placement
tests. The lab shows its availability in the VLA control method. See
[VLA training](VLA_TRAINING.md) for the experimental pipeline and
[the failure analysis](VLA_FAILURE_ANALYSIS.md) for the measured outcomes.

## Verify the reference policy on a Mac

The reported moving-pad results come from Dhruv's M2 policy in
`distill_frontier` at tag `vla-v1`. Its model card lists the checksum for
`vla-v1-m2.pt`, which must be obtained from the training machine.
With the reference checkout next to Chora, run from the Chora root:

```bash
.venv-vla/bin/python scripts/verify_reference_vla.py --expert --episodes 3
.venv-vla/bin/python scripts/verify_reference_vla.py --checkpoint /path/to/vla-v1-m2.pt
```

The first command exercises the reference physics, scripted expert and both
cameras. The second verifies the checkpoint checksum and runs the original
policy in its checkpoint-configured scene using Apple Metal. Increase
`--episodes` to 100 to evaluate the reference held-out seeds. The script reports
placement outcomes and replan latency after warmup and leaves the lab untouched.

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

## Direct-action familiar-scene pilot

The `--direct-pilot` option keeps the two camera streams, joint state and
instruction embedding. It uses a smaller fusion and action decoder and fits
the 16 joint-target predictions directly, with one decoder pass per replan.

The initial run fit every window from two Standard bowl demonstrations for
1,200 updates. Its rollout at four ticks per replan placed the bowl on seed
20000 and collided on seed 20001. Replanning every one or two ticks timed out on both.
The one placement was on a training scene. This checkpoint remains outside
the browser's qualified model selection.

To reproduce from the existing three-episode pilot dataset:

```bash
.venv-vla/bin/python -m chora.vla.train --dataset outputs/vla/datasets/bowl-pilot-v1 \
  --direct-pilot --steps 1200 --batch-size 8 --overfit-windows 0 --eval-every 1200 \
  --device mps --output outputs/vla/training/bowl-direct-familiar-v1
.venv-vla/bin/python -m chora.vla.runner \
  --checkpoint outputs/vla/training/bowl-direct-familiar-v1/checkpoints/step-001200 \
  --dataset outputs/vla/datasets/bowl-pilot-v1 --device mps --seeds 20000 20001 --execute 4
```

## Visual policy in the browser

In the lab select **Bowl · Standard platform**, then **VLA · Camera-based
control**. With an evaluated model, the main viewer shows its front and wrist
cameras and Experiment controls handles the seed, run, stop, and outcome.
Without one, the same viewer remains available as a scene preview. Gallery
and Reports are scoped to this method.

A model must satisfy the basic development check: at least ten distinct scenes
outside the dataset's train/validation episodes, at least 80% placement success,
no collisions, matching weight hashes and the deployed execution horizon. This
is a simulation development criterion, not physical validation. The current
replacement scored 0/10, so no model is exposed for UI execution. Browser requests
cannot launch an unqualified checkpoint, even if its ID is supplied directly.

Training runs and failed diagnostics remain in local outputs and developer CLI
commands. Their names (such as overfit, joint conditioning and action order)
describe internal experiments, not robot tasks, scenes or usable model choices.
See [failure analysis](VLA_FAILURE_ANALYSIS.md) for the actual results. The expert
and Qwen baseline remain in the main controls.

When a qualifying model exists, run the normal lab with
`.venv/bin/python -m chora serve`, select the standard bowl scene, and use its
single visual-policy run control. Live scene/wrist frames, Stop and outcome
reporting remain supported. Restart the server after updating backend code.
