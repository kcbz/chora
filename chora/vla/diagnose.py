"""Recorded-observation prediction and conditioning diagnostics, not rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import ChunkDataset
from .model import load_checkpoint
from .train import Batches, feature_files


def run(checkpoint, dataset, output, *, device="mps", windows=8):
    model, text, text_valid, saved = load_checkpoint(checkpoint, device)
    data = ChunkDataset(dataset, "train", chunk=model.cfg.chunk,
                        history_offset=saved["metadata"].get("history_offset", 4))
    if saved["metadata"]["dataset_fingerprint"] != data.manifest["fingerprint"]:
        raise ValueError("Checkpoint and dataset fingerprints differ")
    if not 2 <= windows <= 32:
        raise ValueError("Use 2..32 diagnostic windows")
    ids = np.linspace(0, len(data)-1, min(windows, len(data)), dtype=int)
    batch = Batches(data, feature_files(data, model, saved["metadata"]["settings"], device), device)(ids)
    n = len(ids)
    scales = torch.tensor(data.norm["action_scale"], dtype=torch.float32, device=device)
    centers = torch.tensor(data.norm["action_center"], dtype=torch.float32, device=device)
    def metrics(pred):
        error = pred-batch["targets"]
        valid = batch["valid"][:, :, None]
        count = valid.sum()*model.cfg.action_dim
        return {"normalized_mse": float((error.square()*valid).sum().cpu()/count.cpu()),
                "joint_mae_rad": ((error.abs()*scales*valid).sum((0, 1))/valid.sum()).cpu().tolist(),
                "first_action_mae_rad": float((error[:, 0].abs()*scales).mean().cpu()),
                "first_four_mae_rad": float((error[:, :4].abs()*scales*valid[:, :4]).sum().cpu()/(valid[:, :4].sum()*model.cfg.action_dim).cpu())}
    report = {"checkpoint": str(checkpoint), "dataset_fingerprint": data.manifest["fingerprint"],
              "joint_names": data.manifest["contract"]["joint_names"],
              "proprio_broadcast": model.cfg.proprio_broadcast,
              "sinusoidal_action_positions": model.cfg.sinusoidal_action_positions,
              "scope": "training-window sensitivity; permutations are not task success or causal proof",
              "variants": {}, "cases": []}
    with torch.no_grad():
        normal = None
        for label, visual, proprio in (("normal", batch["visual"], batch["proprio"]),
                                       ("permuted_cameras", batch["visual"].roll(1, 0), batch["proprio"]),
                                       ("permuted_joints", batch["visual"], batch["proprio"].roll(1, 0)),
                                       ("current_frame_as_history", batch["visual"][:, -1:].expand(-1, 2, -1, -1, -1), batch["proprio"])):
            memory, pad = model.condition(visual, text.expand(n, -1, -1), text_valid.expand(n, -1), proprio)
            for steps in ((1, 4, 16) if label == "normal" else (4,)):
                pred = model.sample(memory, pad, steps=steps)
                report["variants"][f"{label}_f{steps}"] = metrics(pred)
                if label == "normal" and steps == 4:
                    normal = pred
        model.train()
        memory, pad = model.condition(batch["visual"], text.expand(n, -1, -1), text_valid.expand(n, -1), batch["proprio"])
        train_pred = model.sample(memory, pad)
        report["train_eval_max_difference"] = float((train_pred-normal).abs().max().cpu())
        model.eval()
        for row, i in enumerate(ids):
            sample = data[int(i)]
            valid = batch["valid"][row]
            pred = normal[row]*scales+centers
            target = batch["targets"][row]*scales+centers
            report["cases"].append({"seed": sample["seed"], "tick": sample["tick"],
                                    "valid_ticks": int(valid.sum()),
                                    "predicted_joint_targets": pred[valid].cpu().tolist(),
                                    "expert_joint_targets": target[valid].cpu().tolist()})
        # This uses future expert labels only as an explanatory offline reference.
        # It is never an executable policy or a source of student observations.
        oracle_error, variation, count = 0., 0., 0
        for row in range(n):
            valid = batch["valid"][row]
            targets = batch["targets"][row][valid]
            predictions = normal[row][valid]
            oracle_error += float((targets-targets.mean(0)).square().sum().cpu())
            variation += float((predictions-predictions.mean(0)).square().sum().cpu())
            count += targets.numel()
        report["temporal_diagnostic"] = {"oracle_constant_chunk_mse": oracle_error/count,
                                         "prediction_within_chunk_variance": variation/count,
                                         "note": "Future-label oracle for diagnosis only; never used by the runner."}
    path = Path(output)
    with path.open("x") as f:
        f.write(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--windows", type=int, default=8)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
