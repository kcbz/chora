"""Frozen-encoder VLA imitation trainer and deliberately small overfit checks."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from ..common import OUTPUT, ROOT, exclusive_job, job_progress
from .dataset import ChunkDataset, digest
from .model import ModelConfig, VisualActionModel, load_backbones, load_checkpoint, preprocess_rgb, save_checkpoint


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def feature_files(dataset, model, settings, device):
    """Cache only frozen image features. No scene state or action labels enter vision."""
    vision_hash = hashlib.sha256()
    for name, tensor in model.vision.state_dict().items():
        vision_hash.update(name.encode())
        vision_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    root = ROOT / ".cache/vla/features"
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for index, episode in enumerate(dataset.episodes):
        key = digest({"episode": episode["sha256"], "vision": vision_hash.hexdigest(),
                      "vision_revision": settings["vision"], "grid": model.cfg.pooled_grid,
                      "model_source": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
                      "torch": torch.__version__})
        path, note = root / (key + ".safetensors"), root / (key + ".json")
        if path.exists() and note.exists():
            expected = json.loads(note.read_text())["sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen feature cache checksum mismatch")
        else:
            arrays = dataset.episode_arrays(index, images=True)
            rgb = np.stack([arrays[f"rgb_{c}"] for c in dataset.manifest["contract"]["camera_names"]], axis=1)
            encoded = []
            with torch.no_grad():
                for start in range(0, len(rgb), 4):
                    batch = preprocess_rgb(torch.from_numpy(rgb[start:start+4]), device)
                    encoded.append(model.encode_images(batch).cpu())
            features = torch.cat(encoded).contiguous()
            temp = path.with_suffix("." + uuid.uuid4().hex + ".partial")
            save_file({"features": features}, str(temp))
            temp.replace(path)
            note.write_text(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "episode_sha256": episode["sha256"], "shape": list(features.shape)}, indent=2) + "\n")
            print(f"Cached frozen vision: {episode['seed']} ({len(rgb)} frames)", flush=True)
        paths[episode["seed"]] = path
    return paths


class Batches:
    def __init__(self, dataset, features, device):
        self.dataset, self.features, self.device = dataset, features, device
        self.loaded = OrderedDict()

    def __call__(self, indices):
        samples, visual = [], []
        for i in indices:
            sample = self.dataset[int(i)]
            seed = sample["seed"]
            if seed not in self.loaded:
                if len(self.loaded) >= 2:
                    self.loaded.popitem(last=False)
                self.loaded[seed] = load_file(str(self.features[seed]))["features"]
            self.loaded.move_to_end(seed)
            f = self.loaded[seed]
            visual.append(torch.stack((f[sample["history_tick"]], f[sample["tick"]])))
            samples.append(sample)
        result = {key: torch.from_numpy(np.stack([s[key] for s in samples])).to(self.device)
                  for key in ("proprio", "targets", "valid")}
        result["visual"] = torch.stack(visual).to(self.device)
        return result


def evaluate_windows(model, batches, indices, text, text_valid, batch_size):
    model.eval()
    squared, absolute, count = 0., 0., 0
    scales = torch.tensor(batches.dataset.norm["action_scale"], device=batches.device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = batches(indices[start:start+batch_size])
            n = len(batch["targets"])
            memory, pad = model.condition(batch["visual"], text.expand(n, -1, -1), text_valid.expand(n, -1), batch["proprio"])
            predicted = model.sample(memory, pad)
            error = (predicted - batch["targets"]) * batch["valid"][:, :, None]
            squared += float(error.square().sum().cpu())
            absolute += float((error.abs() * scales).sum().cpu())
            count += int(batch["valid"].sum().cpu()) * model.cfg.action_dim
    return {"normalized_action_mse": squared / count, "mean_joint_target_error_rad": absolute / count,
            "windows": len(indices), "valid_scalars": count}


def save_training_state(path, optimizer, generator, *, step, sampler, signature):
    state = optimizer.state_dict()
    tensors = {f"state.{i}.{key}": value.detach().cpu().contiguous()
               for i, row in state["state"].items() for key, value in row.items()}
    tensors["noise_rng"] = generator.get_state()
    save_file(tensors, str(Path(path) / "optimizer.safetensors"))
    payload = {"step": step, "sampler": sampler.bit_generator.state,
               "param_groups": state["param_groups"], "signature": signature}
    (Path(path) / "optimizer.json").write_text(json.dumps(payload, indent=2) + "\n")


def restore_training_state(path, optimizer, generator, sampler, signature):
    path = Path(path)
    meta = json.loads((path / "optimizer.json").read_text())
    if meta["signature"] != signature:
        raise ValueError("Resume dataset/model/training contract changed")
    tensors = load_file(str(path / "optimizer.safetensors"))
    state = {}
    for key, tensor in tensors.items():
        if key.startswith("state."):
            _, index, name = key.split(".", 2)
            state.setdefault(int(index), {})[name] = tensor
    optimizer.load_state_dict({"state": state, "param_groups": meta["param_groups"]})
    generator.set_state(tensors["noise_rng"])
    sampler.bit_generator.state = meta["sampler"]
    return meta["step"]


@exclusive_job("vla-train")
def run(dataset, *, steps=300, batch_size=4, overfit_windows=8, output=None, device="auto",
        learning_rate=.0003, seed=42, eval_every=50, resume=None, direct_pilot=False):
    if not 1 <= steps <= 100000 or not 1 <= batch_size <= 16 or not 0 <= overfit_windows <= 1024 or eval_every < 1:
        raise ValueError("Invalid training size/settings")
    if not 0 < learning_rate <= .01:
        raise ValueError("Invalid learning rate")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unknown device")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(seed)
    settings = yaml.safe_load((ROOT / "configs/vla.yaml").read_text())
    if direct_pilot:
        settings["model"].update(direct_actions=True, pooled_grid=4, width=128, heads=4,
                                 fusion_layers=2, decoder_layers=2, flow_steps=1)
    cfg = ModelConfig(**settings["model"])
    train = ChunkDataset(dataset, "train", chunk=cfg.chunk)
    validation = ChunkDataset(dataset, "validation", chunk=cfg.chunk)
    contract = train.manifest["contract"]
    if (cfg.history, cfg.cameras, cfg.action_dim) != (2, 2, 6) or contract["image_size"] != cfg.image_size:
        raise ValueError("Trainer expects SO-101 camera/history contract")
    if not len(train) or not len(validation):
        raise ValueError("Need nonempty train and validation splits")
    pool = np.linspace(0, len(train)-1, min(overfit_windows, len(train)), dtype=int) if overfit_windows else np.arange(len(train))
    train_eval = pool if len(pool) <= 32 else pool[np.linspace(0, len(pool)-1, 32, dtype=int)]
    val_eval = np.linspace(0, len(validation)-1, min(32, len(validation)), dtype=int)
    signature = {"dataset": train.manifest["fingerprint"], "settings": settings, "batch_size": batch_size,
                 "overfit_windows": overfit_windows, "learning_rate": learning_rate, "seed": seed,
                 "history_offset": train.history_offset,
                 "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ("train.py", "dataset.py", "model.py")}}
    path = Path(output) if output else OUTPUT / "vla/training" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "mode": "overfit_smoke" if overfit_windows else "pilot_training",
              "device": device, "steps_requested": steps, "signature": signature,
              "closed_loop_evaluated": False, "train_windows_available": len(train),
              "train_windows_used": len(pool), "validation_windows_available": len(validation),
              "train_seeds": [e["seed"] for e in train.episodes],
              "validation_seeds": [e["seed"] for e in validation.episodes],
              "warning": "Development pilot training, not held-out task success. No final-test seeds are used."}
    def write():
        (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write()
    try:
        job_progress(0,steps,"Loading visual policy encoders…")
        if resume:
            model, text, text_valid, saved = load_checkpoint(resume, device)
            if saved["normalization"] != train.norm or saved["metadata"]["instruction"] != contract["instruction"]:
                raise ValueError("Checkpoint normalization/instruction changed")
        else:
            vision, text_encoder, tokenizer = load_backbones(settings)
            model = VisualActionModel(cfg, vision).to(device)
            text_encoder = text_encoder.to(device)
            tokens = tokenizer([contract["instruction"]], return_tensors="pt", truncation=True, max_length=77)
            with torch.no_grad():
                text = text_encoder(**{k: v.to(device) for k, v in tokens.items()}).last_hidden_state.detach()
            text_valid = tokens["attention_mask"].bool().to(device)
            del text_encoder
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        noise_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
        sampler = np.random.default_rng(seed)
        start_step = restore_training_state(resume, optimizer, noise_rng, sampler, signature) if resume else 0
        if start_step >= steps:
            raise ValueError("Requested total steps must exceed resumed step")
        job_progress(start_step,steps,"Preparing frozen camera features…")
        train_features = feature_files(train, model, settings, device)
        val_features = feature_files(validation, model, settings, device)
        train_batches, val_batches = Batches(train, train_features, device), Batches(validation, val_features, device)
        initial = evaluate_windows(model, train_batches, train_eval, text, text_valid, batch_size)
        report["initial_train"] = initial
        report["start_step"] = start_step
        write()
        started = time.perf_counter()
        last = None
        for step in range(start_step+1, steps+1):
            model.train()
            batch = train_batches(sampler.choice(pool, size=batch_size, replace=True))
            optimizer.zero_grad(set_to_none=True)
            memory, pad = model.condition(batch["visual"], text.expand(batch_size, -1, -1),
                                           text_valid.expand(batch_size, -1), batch["proprio"])
            if cfg.direct_actions:
                loss = model.direct_loss(memory, pad, batch["targets"], batch["valid"])
            else:
                noise = torch.randn(batch["targets"].shape, generator=noise_rng).to(device)
                t = torch.rand(batch_size, generator=noise_rng).to(device)
                loss = model.flow_loss(memory, pad, batch["targets"], batch["valid"], noise=noise, t=t)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError("Nonfinite training loss/gradient")
            optimizer.step()
            if step == start_step+1 or step % 25 == 0 or step == steps:
                job_progress(step,steps,f"Training visual policy · update {step:,} of {steps:,}")
            row = {"step": step, ("action_loss" if cfg.direct_actions else "flow_loss"): float(loss.detach().cpu()), "gradient_norm": float(norm.cpu())}
            with (path / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step % eval_every == 0 or step == steps:
                row["train"] = evaluate_windows(model, train_batches, train_eval, text, text_valid, batch_size)
                row["validation"] = evaluate_windows(model, val_batches, val_eval, text, text_valid, batch_size)
                sync(device)
                row["elapsed_seconds"] = time.perf_counter() - started
                checkpoint = path / "checkpoints" / f"step-{step:06d}"
                partial = checkpoint.with_name(checkpoint.name + ".partial")
                model.eval()
                save_checkpoint(partial, model, metadata={"settings": settings, "instruction": contract["instruction"],
                                "joint_names": contract["joint_names"], "camera_names": contract["camera_names"],
                                "control_hz": contract["control_hz"], "history_offset": train.history_offset,
                                "dataset_fingerprint": train.manifest["fingerprint"], "training_mode": report["mode"],
                                "runtime_approved": False, "closed_loop_evaluated": False, "step": step},
                                text_features=text, text_valid=text_valid, normalization=train.norm)
                save_training_state(partial, optimizer, noise_rng, step=step, sampler=sampler, signature=signature)
                partial.rename(checkpoint)
                row["checkpoint"] = str(checkpoint)
                last = row
                report.update(step=step, latest=row)
                write()
                with (path / "evaluations.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"VLA step {step}/{steps}: train MSE {row['train']['normalized_action_mse']:.4f}, "
                      f"validation MSE {row['validation']['normalized_action_mse']:.4f}", flush=True)
        report.update(status="complete", final=last,
                      train_mse_ratio=last["train"]["normalized_action_mse"] / initial["normalized_action_mse"],
                      checkpoint_selection="fixed budget final checkpoint; validation not used to fit/select")
        write()
        job_progress(steps,steps,"Visual policy training finished; closed-loop evaluation is still required.")
        return path
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overfit-windows", type=int, default=8, help="0 uses all training windows")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--direct-pilot", action="store_true", help="Fit a compact direct action decoder")
    parser.add_argument("--resume", help="Resume a completed optimizer/model checkpoint into a new run")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
