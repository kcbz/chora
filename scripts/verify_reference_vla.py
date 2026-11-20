"""Run distill_frontier's original M2 policy against its own held-out scenes."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

CHORA = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "2dfa2e0c881b1396a1d77fae2f3b787580058af0"
MODEL_DIR = Path("outputs/vla/models/vla-v1")
CHECKPOINT = "vla-v1-m2.pt"
CHECKPOINT_SHA256 = "834acd779bb9878f36d2745dd01e96cf4e32d0d73b89d290deeed30b5e3ae2bb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=CHORA.parent / "distill_frontier")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", type=Path, help=f"path to the original {CHECKPOINT}")
    mode.add_argument("--expert", action="store_true", help="check the scene and cameras without weights")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--episodes", type=int, default=1, help="first N held-out seeds, up to 100")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error("--episodes must be between 1 and 100")

    reference = args.reference.resolve()
    if not reference.is_dir():
        parser.error(f"distill_frontier checkout not found at {reference}")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=reference, text=True
        )
    except subprocess.CalledProcessError:
        parser.error(f"could not inspect the Git checkout at {reference}")
    if head != REFERENCE_COMMIT or changes:
        parser.error(f"use a clean distill_frontier checkout at vla-v1 ({REFERENCE_COMMIT[:7]})")

    if not args.expert:
        checkpoint = (args.checkpoint or reference / MODEL_DIR / CHECKPOINT).resolve()
        if not checkpoint.is_file():
            parser.error(f"missing original M2 weights at {checkpoint}; supply them with --checkpoint")
        with checkpoint.open("rb") as weights:
            digest = hashlib.file_digest(weights, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            parser.error(f"SHA-256 mismatch for {checkpoint}; expected {CHECKPOINT_SHA256}")

    cache = CHORA / ".cache" / "huggingface" / "hub"
    if cache.is_dir() and "HF_HOME" not in os.environ:
        os.environ.setdefault("HF_HUB_CACHE", str(cache))
    sys.path.insert(0, str(reference))
    from viz.viz import setup_gl

    setup_gl("record")
    from vla.common import load_vla_config
    from vla.env import TickEnv

    policy = None
    if args.expert:
        cfg = load_vla_config(str(reference / "configs" / "vla_m2.yaml"))
        from vla.expert import TickExpert
        from vla.run_expert import run_episode
    else:
        import numpy as np
        import torch
        from vla.policy import load_policy, rollout

        if args.device == "mps" and not torch.backends.mps.is_available():
            parser.error("PyTorch could not access the Mac GPU; use a local graphics session or --device cpu")
        cfg = load_vla_config()
        # The reference checkpoint bundles both the exact scene config and trained weights.
        # The checksum above verifies the artifact before the reference loader unpickles it.
        cfg["vla"] = torch.load(checkpoint, map_location="cpu", weights_only=False)["vla_cfg"]
        policy = load_policy(checkpoint, device=args.device)

    robot_xml = Path(cfg["paths"]["robot_xml"])
    if not (reference / robot_xml).is_file():
        shared_xml = CHORA / robot_xml
        if not shared_xml.is_file():
            parser.error(f"SO-101 asset missing at {reference / robot_xml} and {shared_xml}")
        cfg["paths"]["robot_xml"] = str(shared_xml)

    start = cfg["vla"]["seeds"]["eval_start"]
    env = TickEnv(cfg, render=True)
    successes = 0
    try:
        if args.expert:
            expert = TickExpert(cfg, env)
            print("M2 scripted expert", flush=True)
        else:
            print(f"{CHECKPOINT} on {args.device}", flush=True)
            # Match the reference evaluator's full warmup before collecting latency.
            rollout(env, policy, start)
            policy.latencies_ms.clear()
        for seed in range(start, start + args.episodes):
            if args.expert:
                info, _ = run_episode(env, expert, seed)
            else:
                info = rollout(env, policy, seed)
            success = bool(info["success"])
            successes += success
            result = "success" if success else info["failure_type"]
            print(f"seed {seed}: {result} ({env.stats.ticks} ticks)", flush=True)
        print(f"{successes}/{args.episodes} succeeded")
        if policy is not None:
            p50, p99 = np.percentile(policy.latencies_ms, [50, 99])
            print(f"Policy replan latency: {p50:.1f} ms p50, {p99:.1f} ms p99")
    finally:
        env.renderer.close()
    return int(successes != args.episodes)


if __name__ == "__main__":
    raise SystemExit(main())
