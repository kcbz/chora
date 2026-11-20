"""M3/CPU/CUDA feasibility on recorded frames; never a task-success benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

from ..common import OUTPUT, ROOT
from .env import BowlTickEnv
from .model import ModelConfig, VisualActionModel, load_backbones, load_checkpoint, preprocess_rgb, save_checkpoint


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def distribution(values):
    return {"n": len(values), **{k: float(np.percentile(values, p)) for k, p in
                               (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99), ("max_ms", 100))}}


def run(*, episode, output=None, device="auto", reps=50, warmup=10, train_steps=3, config_path=None, render=False):
    if not 5 <= reps <= 1000 or not 1 <= warmup <= 100 or not 1 <= train_steps <= 20:
        raise ValueError("Use reps 5..1000, warmup 1..100 and train_steps 1..20")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unknown device")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; run with graphics/device access or select CPU explicitly")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(42)
    settings = yaml.safe_load(Path(config_path or ROOT / "configs/vla.yaml").read_text())
    cfg = ModelConfig(**settings["model"])
    if (cfg.history, cfg.cameras, cfg.action_dim) != (2, 2, 6):
        raise ValueError("This feasibility harness expects two-frame/two-camera SO-101 data")
    episode = Path(episode)
    manifest = json.loads((episode.parent / "manifest.json").read_text())
    meta = json.loads(episode.with_suffix(".json").read_text())
    if hashlib.sha256(episode.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Pilot checksum mismatch")
    if manifest["protocol"] != "chora-visual-bowl-ticks-v1":
        raise ValueError("Unsupported demonstration protocol")
    if manifest["image_size"] != cfg.image_size or manifest["camera_names"] != ["front", "wrist"]:
        raise ValueError("Pilot camera contract differs from the model")
    if manifest["control_hz"] != 25:
        raise ValueError("This benchmark expects the 25 Hz pilot")
    path = Path(output) if output else OUTPUT / "vla/feasibility" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "scope": "hardware/model feasibility; untrained action head; no task-success claim",
              "device": device, "torch": torch.__version__, "python": platform.python_version(),
              "platform": platform.platform(), "precision": "fp32", "batch_size": 1,
              "settings": settings, "episode": str(episode), "episode_sha256": meta["sha256"],
              "warmup": warmup, "reps": reps, "control_period_ms": 40,
              "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), Path(__file__).with_name("model.py"))},
              "note": "Repeated recorded observation; instruction cached; no live simulation, camera capture or actuation. No thermal soak."}
    def write():
        (path / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    write()
    try:
        env = BowlTickEnv(manifest["config"], images=False)
        try:
            if env.joint_names != manifest["joint_names"]:
                raise ValueError("Joint ordering changed")
            center = env.bounds.mean(1).astype(np.float32)
            scale = ((env.bounds[:, 1] - env.bounds[:, 0]) / 2).astype(np.float32)
        finally:
            env.close()
        with np.load(episode, allow_pickle=False) as data:
            i = min(32, len(data["actions"]) - cfg.chunk)
            if i < 4:
                raise ValueError("Pilot is too short")
            current_rgb = torch.from_numpy(np.stack([data[f"rgb_{c}"][i] for c in manifest["camera_names"]])[None])
            earlier_rgb = torch.from_numpy(np.stack([data[f"rgb_{c}"][i-4] for c in manifest["camera_names"]])[None])
            positions = (data["joint_positions"][i] - center) / scale
            velocities = data["joint_velocities"][i] / scale
            proprio = torch.tensor(np.r_[positions, velocities][None], dtype=torch.float32, device=device)
            targets = torch.tensor(((data["actions"][i:i+cfg.chunk] - center) / scale)[None], dtype=torch.float32, device=device)
        start = time.perf_counter()
        vision, text_encoder, tokenizer = load_backbones(settings)
        model = VisualActionModel(cfg, vision).to(device).eval()
        text_encoder = text_encoder.to(device)
        tokens = tokenizer([manifest["instruction"]], return_tensors="pt", truncation=True, max_length=77)
        with torch.inference_mode():
            text = text_encoder(**{k: v.to(device) for k, v in tokens.items()}).last_hidden_state.clone()
            text_valid = tokens["attention_mask"].bool().to(device)
        # Do not retain inference-mode tensors for the subsequent autograd test.
        text, text_valid = text.clone(), text_valid.clone()
        synchronize(device)
        result["load_and_instruction_seconds"] = time.perf_counter() - start
        result["parameters"] = {"policy": sum(p.numel() for p in model.parameters()),
                                 "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 "frozen_text_encoder_offline": sum(p.numel() for p in text_encoder.parameters())}
        del text_encoder
        print(f"Loaded {result['parameters']['policy']/1e6:.1f}M policy on {device}", flush=True)

        with torch.no_grad():
            earlier = model.encode_images(preprocess_rgb(earlier_rgb, device))

        def infer(steps, cold_history=False, rgb=None):
            with torch.no_grad():
                current = model.encode_images(preprocess_rgb(current_rgb if rgb is None else rgb, device))
                old = model.encode_images(preprocess_rgb(earlier_rgb, device)) if cold_history else earlier
                visual = torch.stack((old, current), dim=1)
                memory, pad = model.condition(visual, text, text_valid, proprio)
                actions = model.sample(memory, pad, steps=steps)
                return actions.cpu()

        synchronize(device)
        start = time.perf_counter()
        prediction = infer(cfg.flow_steps)
        synchronize(device)
        result["first_policy_ms"] = (time.perf_counter() - start) * 1000
        if prediction.shape != (1, cfg.chunk, cfg.action_dim) or not torch.isfinite(prediction).all():
            raise RuntimeError("Invalid inference result")
        result["inference"] = {}
        for label, steps, cold in [("cached_history_f4", 4, False), ("cached_history_f2", 2, False),
                                    ("cached_history_f1", 1, False), ("uncached_history_f4", 4, True)]:
            for _ in range(warmup):
                infer(steps, cold)
            samples = []
            for _ in range(reps):
                synchronize(device)
                start = time.perf_counter()
                infer(steps, cold)
                synchronize(device)
                samples.append((time.perf_counter() - start) * 1000)
            stats = distribution(samples)
            stats.update(samples_ms=samples, over_40ms=sum(s > 40 for s in samples))
            result["inference"][label] = stats
            print(f"{label}: p50 {stats['p50_ms']:.1f}ms, p99 {stats['p99_ms']:.1f}ms", flush=True)
            write()

        if render:
            live = BowlTickEnv(manifest["config"], images=True, image_size=cfg.image_size)
            try:
                live.reset(meta["seed"])
                camera_samples, combined_samples = [], []
                for i in range(warmup + reps):
                    synchronize(device)
                    start = time.perf_counter()
                    obs = live.observe()
                    rgb = torch.from_numpy(np.stack([obs["images"][c] for c in manifest["camera_names"]])[None])
                    capture_ms = (time.perf_counter() - start) * 1000
                    infer(cfg.flow_steps, rgb=rgb)
                    synchronize(device)
                    total_ms = (time.perf_counter() - start) * 1000
                    if i >= warmup:
                        camera_samples.append(capture_ms)
                        combined_samples.append(total_ms)
                result["render_timing"] = {"scene_and_wrist_capture": distribution(camera_samples),
                                            "capture_plus_policy": distribution(combined_samples),
                                            "combined_over_40ms": sum(s > 40 for s in combined_samples),
                                            "note": "Two live MuJoCo renders at a fixed scene state plus cached-history inference; no physics stepping, action application, pacing or closed-loop task evaluation."}
                print(f"Capture + policy: p50 {np.median(combined_samples):.1f}ms", flush=True)
            finally:
                live.close()

        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        valid = torch.ones((1, cfg.chunk), dtype=torch.bool, device=device)
        training = []
        for step in range(train_steps):
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            old = model.encode_images(preprocess_rgb(earlier_rgb, device))
            current = model.encode_images(preprocess_rgb(current_rgb, device))
            memory, pad = model.condition(torch.stack((old, current), dim=1), text, text_valid, proprio)
            loss = model.flow_loss(memory, pad, targets, valid)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Training loss/gradient is nonfinite or zero")
            optimizer.step()
            synchronize(device)
            training.append({"loss": float(loss.detach().cpu()), "gradient_norm": float(norm.cpu()),
                             "step_ms": (time.perf_counter() - start) * 1000})
            print(f"Training smoke {step+1}/{train_steps}: loss {training[-1]['loss']:.4f}", flush=True)
        result["training_smoke"] = training
        result["memory"] = {"process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if platform.system() == "Darwin" else 1024),
                            "note": "MPS readings are snapshots after training, not allocator peak measurements; RSS is process lifetime high-water."}
        if device == "mps":
            result["memory"].update(mps_tensor_mb=torch.mps.current_allocated_memory()/2**20,
                                      mps_driver_mb=torch.mps.driver_allocated_memory()/2**20,
                                      mps_recommended_mb=torch.mps.recommended_max_memory()/2**20)
        if device == "cuda":
            result["memory"].update(cuda_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
        model.eval()
        expected = infer(cfg.flow_steps)
        save_checkpoint(path / "checkpoint", model, metadata={"settings": settings, "trained_policy": False,
                         "instruction": manifest["instruction"], "joint_names": manifest["joint_names"],
                         "camera_names": manifest["camera_names"], "control_hz": manifest["control_hz"],
                         "purpose": "few-step hardware smoke, not a competent policy"},
                        text_features=text, text_valid=text_valid,
                        normalization={"position_center": center.tolist(), "position_scale": scale.tolist(),
                                       "velocity_scale": scale.tolist(), "action_type": "absolute joint position",
                                       "method": "actuator range, not dataset-fitted statistics"})
        loaded, loaded_text, loaded_mask, _ = load_checkpoint(path / "checkpoint", device)
        with torch.no_grad():
            old = loaded.encode_images(preprocess_rgb(earlier_rgb, device))
            cur = loaded.encode_images(preprocess_rgb(current_rgb, device))
            mem, mask = loaded.condition(torch.stack((old, cur), dim=1), loaded_text, loaded_mask, proprio)
            actual = loaded.sample(mem, mask).cpu()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        result.update(status="complete", checkpoint_roundtrip=True,
                      checkpoint_max_abs_error=float((actual-expected).abs().max()),
                      checkpoint_sha256=hashlib.sha256((path/"checkpoint/model.safetensors").read_bytes()).hexdigest())
        try:
            result["hardware"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            result["hardware"] = platform.processor()
        write()
        return path
    except BaseException as error:
        result.update(status="failed", error=str(error))
        write()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--render", action="store_true", help="Also measure two live camera renders plus inference (needs graphics)")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
