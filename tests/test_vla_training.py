import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chora.vla.train import restore_training_state, save_training_state


def test_optimizer_and_random_state_resume_matches_uninterrupted_step(tmp_path):
    torch.manual_seed(1)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(4)
    sampler = np.random.default_rng(5)

    def step(m, opt, rng, sam):
        x = torch.tensor(sam.normal(size=(3, 2)), dtype=torch.float32)
        y = torch.randn((3, 1), generator=rng)
        opt.zero_grad()
        (m(x)-y).square().mean().backward()
        opt.step()

    step(model, optimizer, generator, sampler)
    saved_weights = {k: v.clone() for k, v in model.state_dict().items()}
    signature = {"dataset": "example", "model": "tiny"}
    save_training_state(tmp_path, optimizer, generator, step=1, sampler=sampler, signature=signature)
    step(model, optimizer, generator, sampler)
    expected = {k: v.clone() for k, v in model.state_dict().items()}

    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(saved_weights)
    opt = torch.optim.AdamW(restored.parameters(), lr=.001)
    rng, sam = torch.Generator(), np.random.default_rng()
    assert restore_training_state(tmp_path, opt, rng, sam, signature) == 1
    step(restored, opt, rng, sam)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="contract"):
        restore_training_state(tmp_path, opt, rng, sam, {"dataset": "changed"})
