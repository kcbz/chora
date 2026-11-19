import json
import math
import os

import numpy as np
import pytest

from chora.common import OUTPUT, read_json
from chora.episodes import rollout
from chora.policy import Expert, messages
from chora.sim import Simulation, validate_action


@pytest.fixture(scope="module")
def sim():
    return Simulation()


@pytest.mark.parametrize("action",[
    None, {}, {"action":"torque"}, {"action":"done","x":0},
    {"action":"move_to","x":True,"y":0,"z":0},
    {"action":"move_to","x":math.nan,"y":0,"z":0},
    {"action":"move_to","x":0,"y":0,"z":float("inf")},
    {"action":"move_to","x":0,"y":0},
])
def test_untrusted_actions_rejected(action):
    with pytest.raises(ValueError):
        validate_action(action)


def test_unreachable_move_has_no_physical_effect(sim):
    sim.reset(42)
    q = sim.data.qpos.copy()
    obs = sim.step({"action":"move_to","x":20,"y":0,"z":.1})
    assert obs["last_result"]=="rejected_unreachable"
    np.testing.assert_array_equal(q,sim.data.qpos)


def test_remote_gripper_close_does_not_attach_bowl(sim):
    sim.reset(7)
    assert sim.model.neq == 0, "Physical benchmark must not use a weld grasp"
    bowl_before = sim.bowl
    obs = sim.step({"action":"close_gripper"})
    assert obs["last_result"]=="failed_grasp"
    assert not obs["grasped"]
    np.testing.assert_allclose(sim.bowl,bowl_before,atol=.0001)


def test_done_requires_real_placement(sim):
    sim.reset(3)
    assert sim.step({"action":"done"})["last_result"]=="not_placed"
    assert not sim.success
    with pytest.raises(RuntimeError):
        sim.step({"action":"done"})


def test_step_limit_ends_bad_policy(sim):
    sim.reset(3)
    for _ in range(sim.cfg["max_steps"]):
        sim.step({"action":"bad"})
    assert sim.finished and sim.result=="step_limit"


def test_physical_oracle_and_exact_reproducibility(sim):
    e1 = rollout(sim,Expert(31),31)
    q1 = sim.data.qpos.copy()
    e2 = rollout(sim,Expert(31),31)
    assert e1["success"] and e2["success"]
    assert e1==e2
    np.testing.assert_array_equal(q1,sim.data.qpos)


def test_prompt_has_consistent_integer_units(sim):
    obs = sim.reset(8)
    action = {"action":"move_to","x":.154,"y":-.073,"z":.125}
    chat = messages(obs,[action],action)
    user = json.loads(chat[1]["content"])
    assistant = json.loads(chat[2]["content"])
    assert assistant=={"action":"move_to","x":154,"y":-73,"z":125}
    assert user["history"][-1]==assistant
    assert user["observation"]["target"][2]==55


def test_episode_split_and_test_set_isolation():
    report = read_json(OUTPUT/"curation.json")
    if not report:
        pytest.skip("No generated dataset yet")
    train,val = set(report["train_seeds"]),set(report["validation_seeds"])
    assert train and val and train.isdisjoint(val)
    assert all(0<=s<10000 for s in train|val)
    assert report["selected_episodes"]==len(train|val)


@pytest.mark.skipif(os.environ.get("CHORA_RENDER_TEST")!="1",reason="Set CHORA_RENDER_TEST=1 for the GPU rendering test")
def test_camera_does_not_change_physics(sim):
    from chora.visualization import Camera
    plain = rollout(sim,Expert(4),4)
    q = sim.data.qpos.copy()
    camera = Camera(sim,320,240)
    frames = []
    def capture(s):
        frames.append(camera.render(s).shape)
    sim.on_frame = capture
    try:
        rendered = rollout(sim,Expert(4),4)
    finally:
        sim.on_frame = None
        camera.close()
    assert frames and frames[0]==(240,320,3)
    assert plain==rendered
    np.testing.assert_array_equal(q,sim.data.qpos)


@pytest.mark.skipif(os.environ.get("CHORA_RENDER_TEST")!="1",reason="Set CHORA_RENDER_TEST=1 for the GPU rendering test")
def test_free_camera_changes_only_the_view(sim):
    from chora.visualization import Camera
    sim.reset(8)
    q = sim.data.qpos.copy()
    camera = Camera(sim,320,240)
    try:
        front = camera.render(sim,"front")
        camera.set_orbit({"azimuth":45,"elevation":-35,"distance":.55,"lookat":[.12,0,.16]})
        orbit = camera.render(sim,"orbit")
        assert not np.array_equal(front,orbit)
        np.testing.assert_array_equal(q,sim.data.qpos)
    finally:
        camera.close()
