from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from chora.vla.runner import ActionGuard, ChunkPolicy, verify_recording_contract


def test_guard_logs_bounds_and_slew_and_rejects_nonfinite_without_mutation():
    guard = ActionGuard(np.tile([-1, 1], (6, 1)), np.zeros(6), .04)
    action, events = guard.apply(np.ones(6)*10)
    np.testing.assert_allclose(action, .12)
    assert events == {"bounds_limited": 6, "slew_limited": 6}
    before = guard.last.copy()
    with pytest.raises(ValueError):
        guard.apply(np.ones(6)*np.nan)
    np.testing.assert_array_equal(before, guard.last)
    with pytest.raises(ValueError):
        ActionGuard(np.tile([-1, 1], (6, 1)), np.ones(6)*2, .04)


class FakeModel:
    cfg = SimpleNamespace(chunk=16, history=2, action_dim=6)
    encodes = 0
    inferences = 0
    bad = False

    def eval(self):
        return self

    def encode_images(self, images):
        self.encodes += 1
        return torch.zeros(1, 2, 1, 3)

    def condition(self, visual, text, mask, state):
        assert visual.shape == (1, 2, 2, 1, 3)
        assert state.shape == (1, 12)
        return torch.zeros(1, 1, 3), mask

    def sample(self, memory, pad):
        self.inferences += 1
        return torch.full((1, 16, 6), float("nan") if self.bad else 0.)


def obs(tick, instruction="bowl"):
    return {"tick": tick, "timestamp": .3+tick*.04, "instruction": instruction,
            "joint_positions": np.zeros(6), "joint_velocities": np.zeros(6),
            "images": {name: np.zeros((28, 28, 3), dtype=np.uint8) for name in ("front", "wrist")}}


def policy():
    norm = {key+"_"+suffix: [value]*6 for key in ("position", "velocity", "action")
            for suffix, value in (("center", 0.), ("scale", 1.))}
    meta = {"normalization": norm, "metadata": {"camera_names": ["front", "wrist"],
              "history_offset": 4, "instruction": "bowl", "control_hz": 25}}
    model = FakeModel()
    return model, ChunkPolicy(model, torch.zeros(1, 1, 3), torch.ones(1, 1, dtype=torch.bool), meta, device="cpu")


def test_queue_replans_and_reuses_timestamped_history():
    model, p = policy()
    for tick in range(5):
        np.testing.assert_array_equal(p.act(obs(tick)), np.zeros(6))
    assert model.inferences == 2
    assert model.encodes == 2  # frame zero reused as history, not encoded twice.
    p.reset()
    assert not p.queue and not p.features and p.last_tick == -1


@pytest.mark.parametrize("corrupt", ["skip", "instruction", "timestamp"])
def test_stale_observations_and_instruction_changes_invalidate_queue(corrupt):
    _, p = policy()
    p.act(obs(0))
    o = obs(2 if corrupt == "skip" else 1, "other" if corrupt == "instruction" else "bowl")
    if corrupt == "timestamp":
        o["timestamp"] += .2
    with pytest.raises(ValueError):
        p.act(o)
    assert not p.queue


def test_nonfinite_predictions_cannot_enter_queue():
    model, p = policy()
    model.bad = True
    with pytest.raises(ValueError, match="Nonfinite"):
        p.act(obs(0))
    assert not p.queue


@pytest.mark.parametrize("seed", [20000, 20003])
def test_same_guard_does_not_prevent_expert_placement(seed):
    from chora.vla.env import BowlTickEnv, TickExpert
    env = BowlTickEnv(images=False, max_ticks=240)
    try:
        env.reset(seed)
        expert = TickExpert(env, seed)
        guard = ActionGuard(env.bounds, env.sim.data.ctrl, env.period)
        while not env.done:
            applied, _ = guard.apply(expert.act())
            env.step_tick(applied)
            if expert.exhausted and not expert.pending:
                break
        assert env.success
    finally:
        env.close()


def test_shorter_diagnostic_budget_preserves_physics_contract():
    from chora.vla.env import BowlTickEnv
    from chora.vla.record import provenance
    env = BowlTickEnv(images=False)
    try:
        source = provenance(env)
        expected = source["fingerprint"]
        env.max_ticks = 240
        verify_recording_contract(env, source, expected)
        env.max_ticks = 501
        with pytest.raises(ValueError, match="budget"):
            verify_recording_contract(env, source, expected)
        env.max_ticks = 240
        env.cfg["simulation"]["table_height"] += .001
        with pytest.raises(ValueError, match="provenance"):
            verify_recording_contract(env, source, expected)
    finally:
        env.close()
