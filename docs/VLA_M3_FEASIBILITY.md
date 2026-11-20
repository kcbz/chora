# VLA feasibility on Apple M3

Measured November 19, 2026. Hardware: Apple M3, 16 GB unified memory, macOS
26.6.2, Python 3.12.11, PyTorch 2.15.0 MPS, fp32, batch size 1. No MPS CPU
fallback was enabled. Separate environment: `.venv-vla`.

## What ran

An independently implemented 39,230,982-parameter visual policy, with
17,174,406 trainable parameters. DINOv2-S and the separately cached CLIP text
encoder are frozen. Fusion and flow/action modules are initially random; three
optimizer updates are hardware smoke tests, not a trained skill. Inputs are real
recorded 224 x 224 scene/wrist frames, two-frame history, joint position/velocity
and the bowl instruction. Output is a 16 x 6 joint-target chunk.

## Warm inference

50 samples per row after 10 warm-ups, synchronized on MPS before and after each
sample. Includes RGB conversion/upload/normalization, current-view encoding,
fusion, action decoding and copying actions to CPU. Instruction encoding is
cached and excluded. Earlier-frame features are cached except in the last row.

| Settings | p50 ms | p95 ms | p99 ms | Over 40 ms |
|---|---:|---:|---:|---:|
| 4 flow steps, cached earlier views | 33.1 | 33.7 | 33.7 | 0/50 |
| 2 flow steps, cached earlier views | 28.0 | 28.4 | 28.8 | 0/50 |
| 1 flow step, cached earlier views | 25.7 | 26.4 | 26.6 | 0/50 |
| 4 flow steps, recompute earlier views | 52.6 | 53.7 | 53.8 | 50/50 |

These are repeated observations, not 50 episodes. Fifty samples cannot establish
production tail latency, thermal stability or task success. Flow-step accuracy
tradeoffs have not been measured. Initial-call timing was 47.3 ms in this
run and about 425 ms in the earlier process, so warming the path matters.

## Camera cost and scheduling

Two live MuJoCo camera renders took 25.1 ms median. Capture plus
four-step cached-history inference took 60.3 ms median / 61.2 ms p99;
all 50 combined measurements exceeded a 40 ms tick. The scene was fixed and no
physics stepping, actuator application or control-loop pacing was included.

Inference alone fits the 25 Hz tick in this short run. A serial capture + inference
operation on every tick does not. Start with four-tick action execution (160 ms
between replans), then build/test independent fixed-rate control, timestamp-aware
history caching and inference scheduling. The 160 ms horizon is a proposed
starting setting, not a demonstrated closed-loop deadline guarantee. A moving
world and contention between rendering and inference still need evaluation.

## Training and checkpoint checks

All three batch-one optimizer updates had finite nonzero gradients. The first
step took 156.3 ms and the next two 78.2 / 79.3 ms. Each included
frozen vision encoding, trainable fusion/head forward, backward, gradient clipping
and AdamW update. These few steps on one chunk do not estimate large-batch
throughput or the number of updates needed to learn the task.

The safe tensor/JSON checkpoint loaded without downloading models again. On the
same device its action prediction matched with maximum absolute error 0.0.
Offline contract tests cover head parameter updates, frozen vision gradients,
padding isolation, action shapes and exact checkpoint reload.

Observed after training: process peak RSS 1136 MiB, MPS tensor allocation
350 MiB and MPS driver allocation 1252 MiB. MPS figures are snapshots,
not allocator peak measurements. Unified-memory metrics overlap; do not add them
into a total. Larger batches, vision fine-tuning and two-arm cameras need their
own measurements.

## Artifacts and reproduction

Local ignored run: `outputs/vla/feasibility/m3-v2/`, containing `report.json` and
`checkpoint/`. The report stores individual latency samples, runtime/settings,
source hashes, input checksum and checkpoint checksum. An earlier `m3-v1` run
contains the initial inference-only measurements. Neither checkpoint is a
competent policy or should be used to control the scene.

```bash
.venv-vla/bin/python -m chora.vla.benchmark \
  --episode outputs/vla/runs/bowl-tick-pilot-v1/episode-20000.npz \
  --device mps --reps 50 --warmup 10 --train-steps 3 --render
```

On a fresh checkout first follow [VLA setup and recording](VLA.md); local pilot
and benchmark artifacts are not committed. The source uses official
[DINOv2 interfaces](https://huggingface.co/docs/transformers/model_doc/dinov2),
[CLIP text interfaces](https://huggingface.co/docs/transformers/model_doc/clip) and
[PyTorch MPS synchronization/memory APIs](https://docs.pytorch.org/docs/stable/mps.html).
These measurements are Chora's own, not the reference project's RTX 4090 results.

Next milestone: a split-safe chunk dataset and trainer, followed by a closed-loop
policy runner and development-scene evaluations. Do not scale data collection
before an overfit smoke test and initial closed-loop rollout work.
