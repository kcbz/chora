import os

import numpy as np
import pytest

from chora.vla.env import BowlTickEnv, TickExpert
from chora.vla.record import collect, replay


@pytest.fixture
def env():
    e = BowlTickEnv(images=False)
    yield e
    e.close()


def test_fixed_ticks_named_mapping_and_observation_boundary(env):
    before = env.reset(20000)
    q = before["joint_positions"].copy()
    after, done, _ = env.step_tick(q)
    assert not done
    assert after["timestamp"] - before["timestamp"] == pytest.approx(.04)
    assert env.sim.physics_steps == 60 + 8
    assert set(after) == {"timestamp", "tick", "joint_positions", "joint_velocities", "instruction", "images"}
    assert env.joint_names[-1] == "gripper"
    assert "bowl" not in after and "target" not in after
    assert not np.shares_memory(after["joint_positions"], env.sim.data.qpos)


@pytest.mark.parametrize("bad", [[0]*5, [float("nan")]*6, [True]*6, ["0"]*6, [100]*6])
def test_bad_targets_cannot_advance_physics(env, bad):
    before = env.sim.data.qpos.copy()
    t = env.sim.data.time
    with pytest.raises(ValueError):
        env.step_tick(bad)
    assert env.sim.data.time == t
    np.testing.assert_array_equal(before, env.sim.data.qpos)


def test_nonintegral_control_period_rejected():
    with pytest.raises(ValueError, match="integer"):
        BowlTickEnv(hz=30, images=False)


def test_tick_limit_and_reset(env):
    env.max_ticks = 1
    _, done, info = env.step_tick(env.sim.data.ctrl.copy())
    assert done and info["reason"] == "tick_limit" and not info["success"]
    with pytest.raises(RuntimeError):
        env.step_tick(env.sim.data.ctrl.copy())
    env.reset(20000)
    assert env.tick == 0 and not env.done


def test_expert_record_replays_every_state_and_does_not_overwrite(tmp_path):
    run = collect(episodes=1, output=tmp_path / "pilot", images=False)
    file = run / "episode-20000.npz"
    result = replay(file)
    assert result["verified"] and result["success"]
    with np.load(file, allow_pickle=False) as data:
        assert len(data["timestamps"]) == len(data["actions"]) + 1
        np.testing.assert_allclose(np.diff(data["timestamps"]), .04, atol=1e-12)
        assert not any(k.startswith("rgb_") for k in data)
    with pytest.raises(FileExistsError):
        collect(episodes=1, output=run, images=False)
    with file.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        replay(file)


@pytest.mark.parametrize("seed", [0, 4, 31])
def test_fixed_tick_expert_physically_places_bowl(env, seed):
    env.reset(seed)
    p = TickExpert(env, seed)
    while not env.done:
        env.step_tick(p.act())
        if p.exhausted and not p.pending:
            break
    assert env.success and env.sim.model.neq == 0


@pytest.mark.skipif(os.environ.get("CHORA_RENDER_TEST") != "1", reason="Opt-in GPU rendering")
def test_policy_cameras_are_read_only_and_record_aligned_frames(tmp_path):
    e = BowlTickEnv(images=True, image_size=64)
    try:
        e.reset(20000)
        q = e.sim.data.qpos.copy()
        obs = e.observe()
        assert set(obs["images"]) == {"front", "wrist"}
        for frame in obs["images"].values():
            assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
            assert frame.std() > 1
        np.testing.assert_array_equal(e.sim.data.qpos, q)
    finally:
        e.close()
    state_run = collect(episodes=1, output=tmp_path / "state", images=False, image_size=64)
    image_run = collect(episodes=1, output=tmp_path / "image", images=True, image_size=64)
    with np.load(state_run / "episode-20000.npz") as plain, np.load(image_run / "episode-20000.npz") as visual:
        for name in plain:
            np.testing.assert_array_equal(plain[name], visual[name])
        assert len(visual["rgb_wrist"]) == len(visual["actions"]) + 1
    assert replay(image_run / "episode-20000.npz")["verified"]
