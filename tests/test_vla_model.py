"""Offline tiny-model contract tests; pretrained weights are not downloaded."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Dinov2Config, Dinov2Model

from chora.vla.model import ModelConfig, VisualActionModel, load_checkpoint, preprocess_rgb, save_checkpoint


@pytest.fixture
def model():
    torch.manual_seed(7)
    cfg = ModelConfig(image_size=28, pooled_grid=2, width=24, heads=4,
                      fusion_layers=1, decoder_layers=1, text_dim=16, chunk=4)
    vision = Dinov2Model(Dinov2Config(hidden_size=24, num_hidden_layers=1,
                                     num_attention_heads=4, intermediate_size=48, image_size=28))
    return VisualActionModel(cfg, vision)


def inputs(model):
    images = preprocess_rgb(torch.randint(0, 256, (1, 2, 28, 28, 3), dtype=torch.uint8), "cpu")
    visual = torch.stack((model.encode_images(images), model.encode_images(images)), dim=1)
    return visual, torch.randn(1, 3, 16), torch.tensor([[True, True, False]]), torch.randn(1, 12)


def test_fusion_action_chunk_and_backprop_keep_vision_frozen(model):
    model.train()
    memory, pad = model.condition(*inputs(model))
    target = torch.randn(1, 4, 6)
    valid = torch.tensor([[True, True, True, False]])
    before = model.velocity_head.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    loss = model.flow_loss(memory, pad, target, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.vision.parameters())
    assert not model.vision.training
    optimizer.step()
    assert not torch.equal(before, model.velocity_head.weight)
    model.eval()
    with torch.no_grad():
        output = model.sample(memory.detach(), pad)
    assert output.shape == (1, 4, 6) and torch.isfinite(output).all()


def test_padding_cannot_change_valid_loss_or_language_condition(model):
    model.eval()
    args = inputs(model)
    memory, pad = model.condition(*args)
    targets = torch.randn(1, 4, 6)
    valid = torch.tensor([[True, True, False, False]])
    noise, t = torch.randn_like(targets), torch.tensor([.4])
    first = model.flow_loss(memory, pad, targets, valid, noise=noise, t=t)
    targets[:, 2:] = 1e6
    second = model.flow_loss(memory, pad, targets, valid, noise=noise, t=t)
    torch.testing.assert_close(first, second)
    other_text = args[1].clone()
    other_text[:, 2] = 1e6  # Padded CLIP token excluded from both fusion and decoder memory.
    other, other_pad = model.condition(args[0], other_text, args[2], args[3])
    torch.testing.assert_close(model.sample(memory, pad), model.sample(other, other_pad))
    with pytest.raises(ValueError):
        model.flow_loss(memory, pad, targets, torch.zeros_like(valid))


def test_checkpoint_roundtrip_without_network(model, tmp_path):
    model.eval()
    visual, text, valid, proprio = inputs(model)
    with torch.no_grad():
        memory, pad = model.condition(visual, text, valid, proprio)
        expected = model.sample(memory, pad)
    save_checkpoint(tmp_path / "checkpoint", model, metadata={"trained_policy": False},
                    text_features=text, text_valid=valid, normalization={})
    restored, rt, rv, metadata = load_checkpoint(tmp_path / "checkpoint")
    with torch.no_grad():
        memory, pad = restored.condition(visual, rt, rv, proprio)
        actual = restored.sample(memory, pad)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert metadata["metadata"]["trained_policy"] is False
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "checkpoint", model, metadata={}, text_features=text,
                        text_valid=valid, normalization={})


def test_preprocessing_and_config_contract():
    rgb = torch.zeros(1, 2, 28, 28, 3, dtype=torch.uint8)
    processed = preprocess_rgb(rgb, "cpu")
    assert processed.shape == (1, 2, 3, 28, 28)
    torch.testing.assert_close(processed[0, 0, :, 0, 0], -torch.tensor([.485, .456, .406]) / torch.tensor([.229, .224, .225]))
    with pytest.raises(ValueError):
        preprocess_rgb(rgb.float(), "cpu")
    with pytest.raises(ValueError):
        ModelConfig(width=25, heads=4)


def test_joint_broadcast_is_explicit_and_preserves_legacy_loading(model):
    model.eval()
    args = inputs(model)
    with torch.no_grad():
        legacy, pad = model.condition(*args)
        joint = model.proprio_projection(args[3])[:, None]
        model.cfg.proprio_broadcast = True
        broadcast, new_pad = model.condition(*args)
    torch.testing.assert_close(broadcast, legacy + joint)
    torch.testing.assert_close(new_pad, pad)
    assert ModelConfig().proprio_broadcast is False


def test_explicit_action_order_survives_checkpoint_and_is_not_trainable(model, tmp_path):
    model.cfg.sinusoidal_action_positions = True
    model.cfg.proprio_broadcast = True
    assert not model.fixed_action_position.requires_grad
    assert not torch.equal(model.fixed_action_position[:, 0], model.fixed_action_position[:, 1])
    assert "fixed_action_position" not in model.state_dict()
    model.eval()
    visual, text, valid, proprio = inputs(model)
    with torch.no_grad():
        mem, pad = model.condition(visual, text, valid, proprio)
        expected = model.sample(mem, pad)
    save_checkpoint(tmp_path/"ordered", model, metadata={}, text_features=text, text_valid=valid, normalization={})
    restored, rt, rv, _ = load_checkpoint(tmp_path/"ordered")
    assert restored.cfg.sinusoidal_action_positions and restored.cfg.proprio_broadcast
    with torch.no_grad():
        mem, pad = restored.condition(visual, rt, rv, proprio)
        actual = restored.sample(mem, pad)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
