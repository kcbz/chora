"""Versioned tick demonstrations and physics replay, without model dependencies."""
from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from pathlib import Path

import mujoco
import numpy as np

from ..common import OUTPUT, ROOT
from .env import BowlTickEnv, TickExpert

PROTOCOL = "chora-visual-bowl-ticks-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(env):
    sources = [Path(__file__), Path(__file__).with_name("env.py"),
               Path(__file__).parents[1] / "sim.py", Path(__file__).parents[1] / "policy.py"]
    robot = ROOT / "assets/menagerie/robotstudio_so101"
    sources += sorted(p for p in robot.rglob("*") if p.is_file() and p.suffix in {".xml", ".stl", ".obj", ".png"})
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in sources}
    m = env.sim.model
    cameras = {}
    for name in env.camera_names:
        c = m.camera(name).id
        cameras[name] = {"parent_body": m.body(int(m.cam_bodyid[c])).name,
                         "local_position_m": m.cam_pos[c].tolist(),
                         "local_quaternion_wxyz": m.cam_quat[c].tolist(),
                         "vertical_fov_degrees": float(m.cam_fovy[c]),
                         "image_size": [env.image_size, env.image_size],
                         "axes": "MuJoCo/OpenGL: view along local -Z, image up +Y"}
    record = {"protocol": PROTOCOL, "config": env.cfg, "source_hashes": hashes,
              "mujoco": mujoco.__version__, "numpy": np.__version__,
              "python": platform.python_version(), "control_hz": env.hz,
              "physics_timestep": env.sim.model.opt.timestep,
              "substeps": env.substeps, "max_ticks": env.max_ticks,
              "joint_names": env.joint_names, "actuator_names": env.actuator_names,
              "action_units": "radians; position targets in named actuator order",
              "camera_names": list(env.camera_names), "cameras": cameras,
              "instruction": env.observe()["instruction"],
              "expert": "existing bowl planner with fixed-tick IK execution",
              "termination": "configured stable placement interval, collision, expert exhaustion or tick limit",
              "physics_calibration": "unchanged SO-101 bowl benchmark; not real-world validated"}
    record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def record_episode(env, seed, path):
    """Store N actions and N+1 pre/post observations. No pickle/object arrays."""
    path = Path(path)
    obs = env.reset(seed)
    expert = TickExpert(env, seed)
    states, velocities, timestamps, actions, phases = [], [], [], [], []
    images = {name: [] for name in obs["images"]}

    def capture(o):
        states.append(o["joint_positions"])
        velocities.append(o["joint_velocities"])
        timestamps.append(o["timestamp"])
        for name in images:
            images[name].append(o["images"][name])

    capture(obs)
    while not env.done:
        action = expert.act()
        obs, _, _ = env.step_tick(action)
        actions.append(action)
        phases.append(expert.phase)
        capture(obs)
        if expert.exhausted and not expert.pending and not env.done:
            env.done, env.reason = True, "expert_exhausted"
    arrays = dict(joint_positions=np.asarray(states), joint_velocities=np.asarray(velocities),
                  timestamps=np.asarray(timestamps), actions=np.asarray(actions), phases=np.asarray(phases),
                  final_qpos=env.sim.data.qpos.copy(), final_qvel=env.sim.data.qvel.copy())
    arrays.update({f"rgb_{name}": np.asarray(frames) for name, frames in images.items()})
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)
    metadata = {"seed": seed, **env.info(), "frames": len(states), "images_recorded": bool(images),
                "sha256": sha256(path), "observation_alignment": "obs[t], action[t], obs[t+1]"}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def collect(*, episodes=3, seed_start=20000, output=None, images=True, image_size=224):
    if isinstance(episodes, bool) or not isinstance(episodes, int) or not 1 <= episodes <= 100:
        raise ValueError("Pilot collection requires 1..100 episodes")
    # Dedicated development range; do not touch existing training/final benchmark seeds.
    if not 20000 <= seed_start <= 29999 or seed_start + episodes > 30000:
        raise ValueError("VLA pilot seeds must stay in development range 20000..29999")
    path = Path(output) if output is not None else OUTPUT / "vla/runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    env = BowlTickEnv(images=images, image_size=image_size)
    try:
        path.mkdir(parents=True, exist_ok=False)
        manifest = provenance(env)
        manifest.update(status="running", purpose="development pilot; not a training/final-test split",
                        images_recorded=images, image_size=image_size)
        manifest_path = path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        trials = []
        try:
            for seed in range(seed_start, seed_start + episodes):
                result = record_episode(env, seed, path / f"episode-{seed}.npz")
                trials.append(result)
                print(f"VLA pilot {len(trials)}/{episodes}: seed {seed}, {result['reason']}, {result['ticks']} ticks", flush=True)
        except BaseException as error:
            manifest.update(status="failed", error=str(error), completed_episodes=trials)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        manifest.update(status="complete", episodes=trials, successes=sum(t["success"] for t in trials))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        env.close()
    return path


def replay(path):
    """Verify every recorded state transition and final full physics state."""
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    metadata = json.loads(path.with_suffix(".json").read_text())
    if manifest["protocol"] != PROTOCOL or sha256(path) != metadata["sha256"]:
        raise ValueError("Unsupported protocol or recording checksum mismatch")
    env = BowlTickEnv(manifest["config"], hz=manifest["control_hz"], images=False,
                      image_size=manifest["image_size"], max_ticks=manifest["max_ticks"])
    try:
        current = provenance(env)
        if current["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Simulation, source or runtime provenance changed; regenerate the pilot")
        with np.load(path, allow_pickle=False) as data:
            obs = env.reset(metadata["seed"])
            def check(i):
                np.testing.assert_allclose(obs["joint_positions"], data["joint_positions"][i], atol=1e-10, rtol=0)
                np.testing.assert_allclose(obs["joint_velocities"], data["joint_velocities"][i], atol=1e-10, rtol=0)
                np.testing.assert_allclose(obs["timestamp"], data["timestamps"][i], atol=1e-10, rtol=0)
            if len(data["joint_positions"]) != len(data["actions"]) + 1:
                raise ValueError("Invalid observation/action alignment")
            check(0)
            for i, action in enumerate(data["actions"]):
                obs, _, _ = env.step_tick(action)
                check(i + 1)
            np.testing.assert_allclose(env.sim.data.qpos, data["final_qpos"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(env.sim.data.qvel, data["final_qvel"], atol=1e-10, rtol=0)
        if metadata["reason"] == "expert_exhausted" and not env.done:
            env.done, env.reason = True, "expert_exhausted"
        if env.info() != {k: metadata[k] for k in env.info()}:
            raise AssertionError("Replay outcome differs from recording")
        return {"verified": True, **env.info(), "path": str(path)}
    finally:
        env.close()
