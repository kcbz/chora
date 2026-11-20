import copy

import pytest

from chora import experiments
from chora.episodes import bin_key, rollout
from chora.policy import make_expert, messages
from chora.sim import Simulation, make_simulation


def test_fixed_scenes_replace_copy_creation(tmp_path,monkeypatch):
    original=experiments.load_config()
    scenes=experiments.catalog()
    assert [s['id'] for s in scenes]==list(experiments.FIXED_SCENES)
    bowls=[s for s in scenes if s['task']=='bowl_pick_place']
    assert {scene["id"] for scene in bowls} == {
        experiments.DEFAULT, "so101-bowl-moving-platform",
    }
    raised = next(scene for scene in scenes if scene["id"] == "so101-part-raised-shelf")
    assert raised["simulation"]["object_kind"] == "part"
    assert raised["simulation"]["table_height"] == .095
    assert raised["simulation"]["spawn_y"][1] < original["simulation"]["spawn_y"][0]
    monkeypatch.setattr(experiments,'PRESETS',tmp_path)
    with pytest.raises(ValueError,match='copies'):
        experiments.create_experiment({'id':'my-copy','name':'Copy'})
    assert not list(tmp_path.glob('*.yaml'))
    assert experiments.load_config()==original


@pytest.mark.parametrize("changes",[
    {"id":"../outside"},{"robot":"unknown-arm"},{"task":"cook-everything"},
    {"simulation":{"spawn_x":[.205,.145]}},
    {"simulation":{"table_height":.09,"target":[.175,.09,.09]}},
])
def test_invalid_or_unimplemented_setups_are_rejected(tmp_path,monkeypatch,changes):
    monkeypatch.setattr(experiments,"PRESETS",tmp_path)
    payload={"id":"my-experiment","name":"Test","robot":"so101","task":"bowl_pick_place","simulation":{}}
    payload.update(changes)
    with pytest.raises(ValueError):experiments.create_experiment(payload)
    assert not list(tmp_path.glob("*.yaml"))


@pytest.mark.parametrize("preset",["so101-construction-staging","so101-kitchen-staging"])
def test_staging_tasks_use_physical_grasps_and_task_specific_observations(preset):
    cfg=experiments.load_config(preset);sim=Simulation(cfg)
    assert sim.model.neq==0
    for seed in (0,1,2):
        episode=rollout(sim,make_expert(seed,cfg),seed)
        assert episode["success"]
        observation=episode["steps"][0]["observation"]
        assert "object" in observation and "bowl" not in observation
        prompt=messages(observation,[])[0]["content"]
        assert ("construction part" if preset.endswith("construction-staging") else "ingredient portion") in prompt


def test_original_observation_protocol_and_config_remain_compatible():
    sim=Simulation(experiments.load_config())
    observation=sim.reset(0)
    assert "bowl" in observation and "object" not in observation
    assert "object_kind" not in sim.cfg


def test_unlicensed_local_knife_is_explicitly_unavailable_in_a_clean_workspace(tmp_path,monkeypatch):
    import shutil
    shutil.copytree(experiments.ROOT/'configs',tmp_path/'configs')
    monkeypatch.setattr(experiments,'ROOT',tmp_path)
    monkeypatch.setattr(experiments,'PRESETS',tmp_path/'configs/experiments')
    definition=experiments.public_definition('fr3-cut-tomato')
    assert definition['available'] is False
    assert 'local-only knife' in definition['unavailable_reason']
    assert experiments.public_definition()['available'] is True


def test_expert_intercepts_and_bowl_stays_on_the_moving_platform():
    cfg = experiments.load_config("so101-bowl-moving-platform")
    sim = make_simulation(cfg)
    assert not sim.is_placed()
    result = rollout(sim, make_expert(10000, cfg), 10000)
    targets = [step["observation"]["target"][1] for step in result["steps"]]
    assert max(targets) - min(targets) > .02
    assert result["success"] and result["collisions"] == 0
    assert sim.model.neq == 0
    assert sim.is_placed()
    offset = sim.bowl[:2] - sim.target[:2]
    sim._advance(.5)
    assert sim.is_placed()
    assert abs((sim.bowl[:2] - sim.target[:2] - offset)[1]) < .003


def test_platform_forecast_and_selection_use_its_current_motion():
    cfg = experiments.load_config("so101-bowl-moving-platform")
    sim = make_simulation(cfg)
    bins = set()
    for seed in range(12):
        obs = sim.reset(seed)
        bins.add(bin_key({"steps": [{"observation": obs}]}, cfg)[-2:])
        sim._advance(sim.cfg["move_seconds"] + .12 + .10)
        assert obs["target_at_release"][1] == pytest.approx(sim.target[1], abs=.003)
    assert {direction for _, direction in bins} == {0, 1}
    assert {position for position, _ in bins} == {0, 1, 2}
