"""Episode-isolated chunk windows and training-only normalization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .record import PROTOCOL, sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(recordings, output, *, validation_fraction=.2, split_seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    output = Path(output).resolve()
    episodes, seen = [], set()
    contract = None
    for directory in recordings:
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["protocol"] != PROTOCOL or manifest["status"] != "complete" or not manifest["images_recorded"]:
            raise ValueError("Need complete visual pilot recordings")
        current = {k: manifest[k] for k in ("fingerprint", "image_size", "control_hz", "joint_names", "camera_names", "instruction")}
        if contract is not None and contract != current:
            raise ValueError("Recording contracts differ; do not mix scenes or runtimes silently")
        contract = current
        for row in manifest["episodes"]:
            seed = row["seed"]
            if not 20000 <= seed < 30000:
                raise ValueError("This pilot dataset accepts development seeds only")
            if seed in seen:
                raise ValueError("Duplicate episode seed; split by scene before windowing")
            seen.add(seed)
            file = directory / f"episode-{seed}.npz"
            if sha256(file) != row["sha256"]:
                raise ValueError("Recording checksum mismatch")
            # Keep failures in the audit, but not the imitation-learning targets.
            if not row["success"]:
                episodes.append({"seed": seed, "split": "excluded_failure", "sha256": row["sha256"]})
                continue
            with np.load(file, allow_pickle=False) as data:
                n = len(data["actions"])
                if n < 1 or data["actions"].shape != (n, 6):
                    raise ValueError("Invalid action array")
                for key in ("joint_positions", "joint_velocities", "timestamps"):
                    if len(data[key]) != n + 1 or not np.isfinite(data[key]).all():
                        raise ValueError("Invalid state/timestamp alignment")
                if data["joint_positions"].shape != (n+1, 6) or data["joint_velocities"].shape != (n+1, 6):
                    raise ValueError("Expected six named scalar joints")
                if not np.isfinite(data["actions"]).all() or not np.allclose(np.diff(data["timestamps"]), 1 / contract["control_hz"], atol=1e-10, rtol=0):
                    raise ValueError("Invalid action values or control timestamps")
                for camera in contract["camera_names"]:
                    rgb = data[f"rgb_{camera}"]
                    if rgb.dtype != np.uint8 or rgb.shape != (n + 1, contract["image_size"], contract["image_size"], 3):
                        raise ValueError("Invalid RGB alignment")
            episodes.append({"seed": seed, "path": os.path.relpath(file, output),
                             "sha256": row["sha256"], "ticks": n})
    successes = [e for e in episodes if e.get("split") != "excluded_failure"]
    if len(successes) < 2:
        raise ValueError("Need at least two successful episodes for disjoint splits")
    order = np.random.default_rng(split_seed).permutation(len(successes))
    count = min(len(successes) - 1, max(1, round(len(successes) * validation_fraction)))
    validation = set(order[:count].tolist())
    for i, e in enumerate(successes):
        e["split"] = "validation" if i in validation else "train"
    train_arrays = {key: [] for key in ("joint_positions", "joint_velocities", "actions")}
    for episode in successes:
        if episode["split"] != "train":
            continue
        with np.load((output / episode["path"]).resolve(), allow_pickle=False) as data:
            for key in train_arrays:
                train_arrays[key].append(data[key] if key == "actions" else data[key][:-1])
    normalization = {"method": "training episodes only; standard deviation floor 0.01"}
    for key, prefix in (("joint_positions", "position"), ("joint_velocities", "velocity"), ("actions", "action")):
        values = np.concatenate(train_arrays[key])
        normalization[prefix + "_center"] = values.mean(0).tolist()
        normalization[prefix + "_scale"] = np.maximum(values.std(0), .01).tolist()
    value = {"protocol": "chora-vla-pilot-chunks-v1", "purpose": "development training/overfit; no final-test results",
             "contract": contract, "episodes": episodes, "normalization": normalization,
             "split_seed": split_seed, "validation_fraction": validation_fraction}
    value["fingerprint"] = digest(value)
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset.json").write_text(json.dumps(value, indent=2) + "\n")
    return output


class ChunkDataset:
    def __init__(self, directory, split, *, chunk=16, history_offset=4, stride=1):
        if split not in {"train", "validation"} or min(chunk, history_offset, stride) < 1:
            raise ValueError("Invalid split or chunk settings")
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "dataset.json").read_text())
        payload = {k: v for k, v in self.manifest.items() if k != "fingerprint"}
        if self.manifest["protocol"] != "chora-vla-pilot-chunks-v1" or digest(payload) != self.manifest["fingerprint"]:
            raise ValueError("Dataset manifest integrity mismatch")
        self.chunk, self.history_offset = chunk, history_offset
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == split]
        self.norm = self.manifest["normalization"]
        self.windows = [(i, tick) for i, episode in enumerate(self.episodes) for tick in range(0, episode["ticks"], stride)]
        self._states = {}
        for e in self.episodes:
            if sha256(self.directory / e["path"]) != e["sha256"]:
                raise ValueError("Source episode checksum changed")

    def __len__(self):
        return len(self.windows)

    def episode_arrays(self, index, *, images=False):
        e = self.episodes[index]
        if index not in self._states:
            with np.load(self.directory / e["path"], allow_pickle=False) as data:
                self._states[index] = {k: data[k] for k in ("actions", "joint_positions", "joint_velocities")}
        result = self._states[index]
        if images:
            with np.load(self.directory / e["path"], allow_pickle=False) as data:
                result = {**result, **{f"rgb_{c}": data[f"rgb_{c}"] for c in self.manifest["contract"]["camera_names"]}}
        return result

    def __getitem__(self, index):
        episode_index, tick = self.windows[index]
        arrays = self.episode_arrays(episode_index)
        n = self.episodes[episode_index]["ticks"]
        valid = np.arange(self.chunk) < n - tick
        targets = np.zeros((self.chunk, 6), dtype=np.float32)
        actual = arrays["actions"][tick:min(n, tick+self.chunk)]
        targets[:len(actual)] = (actual - self.norm["action_center"]) / self.norm["action_scale"]
        q = (arrays["joint_positions"][tick] - self.norm["position_center"]) / self.norm["position_scale"]
        v = (arrays["joint_velocities"][tick] - self.norm["velocity_center"]) / self.norm["velocity_scale"]
        return {"episode_index": episode_index, "seed": self.episodes[episode_index]["seed"],
                "tick": tick, "history_tick": max(0, tick-self.history_offset),
                "proprio": np.r_[q, v].astype(np.float32), "targets": targets, "valid": valid}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.recordings, args.output, validation_fraction=args.validation_fraction, split_seed=args.split_seed))


if __name__ == "__main__":
    main()
