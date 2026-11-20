"""Synchronous, bounded simulation diagnostics for a learned action-chunk policy.

Physics waits during inference. This is explicitly not the future asynchronous
delay-aware runner, and never falls back to expert or privileged object state.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from ..common import OUTPUT
from .dataset import ChunkDataset, digest
from .env import BowlTickEnv
from .model import load_checkpoint, preprocess_rgb
from .record import provenance
from .train import sync


class ActionGuard:
    def __init__(self, bounds, initial, period, max_speed=3.):
        self.bounds = np.asarray(bounds, dtype=float)
        self.last = np.asarray(initial, dtype=float).copy()
        if not math.isfinite(max_speed) or max_speed <= 0 or not math.isfinite(period) or period <= 0:
            raise ValueError("Positive finite target speed and period required")
        if (self.bounds.shape != (len(self.last), 2) or not np.isfinite(self.bounds).all()
                or not np.isfinite(self.last).all() or np.any(self.bounds[:, 0] > self.bounds[:, 1])
                or np.any(self.last < self.bounds[:, 0]) or np.any(self.last > self.bounds[:, 1])):
            raise ValueError("Malformed actuator bounds")
        self.max_delta = max_speed*period

    def apply(self, raw):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != self.last.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid learned target")
        bounded = np.clip(raw, self.bounds[:, 0], self.bounds[:, 1])
        applied = np.clip(bounded, self.last-self.max_delta, self.last+self.max_delta)
        events = {"bounds_limited": int(np.count_nonzero(raw != bounded)),
                  "slew_limited": int(np.count_nonzero(bounded != applied))}
        self.last = applied.copy()
        return applied, events


def verify_recording_contract(env, source, expected):
    # A diagnostic may stop earlier than collection without changing physics.
    # Every other provenance field must still match the saved recording.
    if not 1 <= env.max_ticks <= source["max_ticks"]:
        raise ValueError("Diagnostic budget must not exceed the recording budget")
    current = provenance(env)
    current.pop("fingerprint")
    current["max_ticks"] = source["max_ticks"]
    if source["fingerprint"] != expected or digest(current) != expected:
        raise ValueError("Recording physics/runtime provenance changed")


class ChunkPolicy:
    def __init__(self, model, text, text_valid, metadata, *, device="mps", execute=4):
        if not 1 <= execute <= model.cfg.chunk:
            raise ValueError("Execution horizon must fit within the predicted chunk")
        if model.cfg.history != 2:
            raise ValueError("Runner currently supports two-frame history")
        self.model, self.text, self.text_valid = model.eval(), text, text_valid
        self.saved, self.device, self.execute = metadata, device, execute
        self.norm = metadata["normalization"]
        for key in ("position_center", "position_scale", "velocity_center", "velocity_scale", "action_center", "action_scale"):
            if len(self.norm.get(key, [])) != model.cfg.action_dim:
                raise ValueError("Checkpoint lacks training normalization")
        self.camera_names = metadata["metadata"]["camera_names"]
        self.history_offset = metadata["metadata"]["history_offset"]
        self.reset()

    def reset(self):
        self.observations = deque(maxlen=self.history_offset+1)
        self.features = OrderedDict()
        self.queue = deque()
        self.last_tick = -1
        self.last_timestamp = None
        self.inferences = []

    def act(self, obs):
        tick = obs["tick"]
        if tick != self.last_tick+1:
            self.queue.clear()
            raise ValueError("Nonconsecutive observation; discard queue and reset")
        if obs["instruction"] != self.saved["metadata"]["instruction"]:
            self.queue.clear()
            raise ValueError("This checkpoint caches one instruction; language switching is unavailable")
        period = 1/self.saved["metadata"]["control_hz"]
        if self.last_timestamp is not None and not math.isclose(obs["timestamp"]-self.last_timestamp, period, abs_tol=1e-8):
            self.queue.clear()
            raise ValueError("Observation timestamp mismatch")
        self.last_tick, self.last_timestamp = tick, obs["timestamp"]
        self.observations.append(obs)
        if not self.queue:
            sync(self.device)
            start = time.perf_counter()
            def encode(o):
                if o["tick"] not in self.features:
                    rgb = torch.from_numpy(np.stack([o["images"][name] for name in self.camera_names])[None])
                    self.features[o["tick"]] = self.model.encode_images(preprocess_rgb(rgb, self.device))
                return self.features[o["tick"]]
            with torch.no_grad():
                old, current = encode(self.observations[0]), encode(obs)
                q = (obs["joint_positions"]-self.norm["position_center"])/self.norm["position_scale"]
                v = (obs["joint_velocities"]-self.norm["velocity_center"])/self.norm["velocity_scale"]
                state = torch.tensor(np.r_[q, v][None], dtype=torch.float32, device=self.device)
                memory, pad = self.model.condition(torch.stack((old, current), 1), self.text, self.text_valid, state)
                normalized = self.model.sample(memory, pad)[0].cpu().numpy()
            raw = normalized*np.asarray(self.norm["action_scale"])+self.norm["action_center"]
            if raw.shape != (self.model.cfg.chunk, self.model.cfg.action_dim) or not np.isfinite(raw).all():
                raise ValueError("Nonfinite or malformed learned action chunk")
            sync(self.device)
            self.inferences.append({"tick": tick, "observation_timestamp": obs["timestamp"],
                                    "milliseconds": (time.perf_counter()-start)*1000})
            self.queue.extend((tick+i, action.copy()) for i, action in enumerate(raw[:self.execute]))
            while len(self.features) > self.history_offset+1:
                self.features.popitem(last=False)
        expected, action = self.queue.popleft()
        if expected != tick:
            self.queue.clear()
            raise ValueError("Expired action chunk")
        return action


def run(checkpoint, dataset, *, output=None, device="mps", seeds=(20000, 20003), max_ticks=240, execute=4, ui_progress=False):
    seeds = list(seeds)
    if not 1 <= len(seeds) <= 10 or any(not 20000 <= seed < 30000 for seed in seeds):
        raise ValueError("Diagnostics use 1..10 development seeds in 20000..29999")
    model, text, text_valid, saved = load_checkpoint(checkpoint, device)
    data = ChunkDataset(dataset, "train")
    meta = saved["metadata"]
    if meta.get("dataset_fingerprint") != data.manifest["fingerprint"]:
        raise ValueError("Checkpoint/dataset mismatch")
    first = (data.directory/data.episodes[0]["path"]).resolve()
    source = json.loads((first.parent/"manifest.json").read_text())
    env = BowlTickEnv(source["config"], hz=meta["control_hz"], image_size=model.cfg.image_size, max_ticks=max_ticks)
    path = Path(output) if output else OUTPUT/"vla/rollouts"/(time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8])
    try:
        if env.joint_names != meta["joint_names"] or list(env.camera_names) != meta["camera_names"]:
            raise ValueError("Robot or camera ordering mismatch")
        verify_recording_contract(env, source, data.manifest["contract"]["fingerprint"])
        path.mkdir(parents=True, exist_ok=False)
        policy = ChunkPolicy(model, text, text_valid, saved, device=device, execute=execute)
        last_publish = [0.]
        def publish_ui(obs, phase, force=False):
            if not ui_progress or (not force and time.monotonic()-last_publish[0] < .1):
                return
            from PIL import Image
            image = Image.fromarray(np.concatenate([obs["images"][c] for c in env.camera_names], axis=1))
            temporary = path/"latest.tmp.jpg"
            image.save(temporary, format="JPEG", quality=88)
            temporary.replace(path/"latest.jpg")
            progress = dict(phase=phase, seed=env.sim.seed, tick=env.tick, max_ticks=max_ticks,
                            simulation_seconds=env.tick*env.period, outcome=env.reason,
                            success=env.success, inference_count=len(policy.inferences),
                            inference_ms=policy.inferences[-1]["milliseconds"] if policy.inferences else None,
                            frame_revision=time.time_ns())
            temporary = path/"progress.tmp.json"
            temporary.write_text(json.dumps(progress))
            temporary.replace(path/"progress.json")
            last_publish[0] = time.monotonic()
        results = []
        report = {"status": "running", "checkpoint": str(checkpoint), "dataset_fingerprint": data.manifest["fingerprint"],
                  "mode": "synchronous development diagnostics; physics waits for inference",
                  "max_ticks": max_ticks, "execute_horizon": execute, "target_slew_limit_rad_s": 3.,
                  "no_expert_fallback": True, "seeds": seeds, "trials": results,
                  "note": "No claim of delay-aware control, physical safety, benchmark success or runtime approval."}
        def write():
            (path/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        write()
        try:
            for seed in seeds:
                obs = env.reset(seed)
                policy.reset()
                publish_ui(obs, "running", force=True)
                guard = ActionGuard(env.bounds, env.sim.data.ctrl, env.period)
                frames = [np.concatenate([obs["images"][c] for c in env.camera_names], axis=1)]
                actions, applied, states, interventions = [], [], [obs["joint_positions"]], []
                timestamps = [obs["timestamp"]]
                stop = None
                start = time.perf_counter()
                while not env.done:
                    if ui_progress and path.with_suffix(".stop").exists():
                        env.done, env.reason = True, "stopped"
                        break
                    try:
                        raw = policy.act(obs)
                        target, events = guard.apply(raw)
                        obs, _, _ = env.step_tick(target)
                    except ValueError as error:
                        stop = str(error)
                        env.done, env.success, env.reason = True, False, "policy_error"
                        break
                    publish_ui(obs, "running")
                    actions.append(raw)
                    applied.append(target)
                    states.append(obs["joint_positions"])
                    timestamps.append(obs["timestamp"])
                    interventions.append(events)
                    frames.append(np.concatenate([obs["images"][c] for c in env.camera_names], axis=1))
                trial = {"seed": seed, **env.info(), "stop_error": stop,
                         "elapsed_wall_seconds": time.perf_counter()-start,
                         "bounds_limited_scalars": sum(e["bounds_limited"] for e in interventions),
                         "slew_limited_scalars": sum(e["slew_limited"] for e in interventions),
                         "inferences": policy.inferences}
                np.savez_compressed(path/f"episode-{seed}.npz", frames=np.asarray(frames),
                                    raw_actions=np.asarray(actions), applied_actions=np.asarray(applied),
                                    joint_positions=np.asarray(states), timestamps=np.asarray(timestamps),
                                    final_qpos=env.sim.data.qpos.copy())
                publish_ui(obs, "complete", force=True)
                results.append(trial)
                write()
                print(f"Learned rollout {seed}: {trial['reason']}, {trial['ticks']} ticks, "
                      f"slew interventions {trial['slew_limited_scalars']}", flush=True)
            report.update(status="complete", successes=sum(e["success"] for e in results))
            write()
        except BaseException as error:
            report.update(status="failed", error=str(error))
            write()
            raise
    finally:
        env.close()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20000, 20003])
    parser.add_argument("--max-ticks", type=int, default=240)
    parser.add_argument("--execute", type=int, default=4)
    parser.add_argument("--ui-progress", action="store_true", help="Publish atomic camera/progress files for the local lab")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
