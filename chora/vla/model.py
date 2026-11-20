"""Independent compact VLA: frozen DINOv2/CLIP features and a flow action head.

No reference-repository code is imported. Hardware smoke tests do not establish
task competence: the fusion/action modules must learn from demonstrations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    image_size: int = 224
    cameras: int = 2
    history: int = 2
    pooled_grid: int = 8
    width: int = 384
    heads: int = 6
    fusion_layers: int = 4
    decoder_layers: int = 4
    chunk: int = 16
    action_dim: int = 6
    text_dim: int = 512
    flow_steps: int = 4
    proprio_broadcast: bool = False
    sinusoidal_action_positions: bool = False

    def __post_init__(self):
        flags = {"proprio_broadcast", "sinusoidal_action_positions"}
        if any(not isinstance(getattr(self, k), bool) for k in flags):
            raise ValueError("Conditioning flags must be boolean")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for k, v in asdict(self).items() if k not in flags):
            raise ValueError("Model settings must be positive integers")
        if self.width % self.heads or self.image_size % 14:
            raise ValueError("Width must divide into heads and image size into 14px patches")
        if self.pooled_grid > self.image_size // 14:
            raise ValueError("Pooling cannot enlarge the patch grid")


def preprocess_rgb(rgb, device):
    """B,V,H,W,3 uint8 to ImageNet-normalized DINOv2 input, B,V,3,H,W."""
    if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ValueError("Expected B,V,H,W,3 uint8 RGB")
    x = rgb.to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255
    mean = x.new_tensor([.485, .456, .406])[None, None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, None, :, None, None]
    return (x - mean) / std


class VisualActionModel(nn.Module):
    def __init__(self, cfg: ModelConfig, vision):
        super().__init__()
        self.cfg, self.vision = cfg, vision
        self.vision.requires_grad_(False).eval()
        w = cfg.width
        self.visual_projection = nn.Linear(vision.config.hidden_size, w)
        n = cfg.history * cfg.cameras * cfg.pooled_grid ** 2
        self.visual_position = nn.Parameter(torch.randn(1, n, w) * .01)
        self.text_projection = nn.Linear(cfg.text_dim, w)
        self.proprio_projection = nn.Linear(2 * cfg.action_dim, w)
        layer = nn.TransformerEncoderLayer(w, cfg.heads, 4 * w, dropout=0.,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, cfg.fusion_layers, norm=nn.LayerNorm(w), enable_nested_tensor=False)
        layer = nn.TransformerDecoderLayer(w, cfg.heads, 4 * w, dropout=0.,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers, norm=nn.LayerNorm(w))
        self.action_projection = nn.Linear(cfg.action_dim, w)
        self.time_projection = nn.Sequential(nn.Linear(3, w), nn.SiLU(), nn.Linear(w, w))
        self.action_position = nn.Parameter(torch.randn(1, cfg.chunk, w) * .01)
        phase = torch.arange(cfg.chunk, dtype=torch.float32)[:, None] * torch.exp(
            torch.arange(0, w, 2, dtype=torch.float32) * (-math.log(10000.)/w))[None]
        fixed = torch.zeros(1, cfg.chunk, w)
        fixed[0, :, 0::2] = phase.sin()
        fixed[0, :, 1::2] = phase.cos()[:, :w//2]
        self.register_buffer("fixed_action_position", fixed, persistent=False)
        self.velocity_head = nn.Linear(w, cfg.action_dim)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    def encode_images(self, images):
        b, v, c, h, w = images.shape
        cfg = self.cfg
        if (v, c, h, w) != (cfg.cameras, 3, cfg.image_size, cfg.image_size):
            raise ValueError("Camera/image shape differs from the model contract")
        with torch.no_grad():
            patches = self.vision(pixel_values=images.reshape(b * v, c, h, w)).last_hidden_state[:, 1:]
            side = h // 14
            patches = patches.transpose(1, 2).reshape(b * v, -1, side, side)
            pooled = F.adaptive_avg_pool2d(patches, (cfg.pooled_grid, cfg.pooled_grid))
        return pooled.flatten(2).transpose(1, 2).reshape(b, v, cfg.pooled_grid ** 2, -1)

    def condition(self, visual, text, text_valid, proprio):
        cfg = self.cfg
        b = visual.shape[0]
        if tuple(visual.shape[1:4]) != (cfg.history, cfg.cameras, cfg.pooled_grid ** 2):
            raise ValueError("Visual history shape differs from the model contract")
        if proprio.shape != (b, cfg.action_dim * 2):
            raise ValueError("Expected joint positions and velocities in named order")
        x = self.visual_projection(visual.flatten(1, 3)) + self.visual_position
        text_tokens = self.text_projection(text)
        joint_token = self.proprio_projection(proprio)[:, None]
        memory = torch.cat((x, text_tokens, joint_token), dim=1)
        pad = torch.cat((torch.zeros(b, x.shape[1], dtype=torch.bool, device=x.device),
                         ~text_valid.bool(), torch.zeros(b, 1, dtype=torch.bool, device=x.device)), dim=1)
        memory = self.fusion(memory, src_key_padding_mask=pad)
        if cfg.proprio_broadcast:
            # A single joint token otherwise competes with 256 visual tokens.
            # Supply its projection to every memory location so decoder attention
            # can use precise joint state without first isolating that one token.
            memory = memory + joint_token
        return memory, pad

    def velocity(self, actions, t, memory, pad):
        phase = torch.stack((t, torch.sin(t * torch.pi), torch.cos(t * torch.pi)), dim=-1)
        x = self.action_projection(actions) + self.action_position + self.time_projection(phase)[:, None]
        if self.cfg.sinusoidal_action_positions:
            x = x + self.fixed_action_position
        return self.velocity_head(self.decoder(x, memory, memory_key_padding_mask=pad))

    def flow_loss(self, memory, pad, targets, valid, *, noise=None, t=None):
        if targets.shape != (memory.shape[0], self.cfg.chunk, self.cfg.action_dim):
            raise ValueError("Wrong action chunk shape")
        if valid.shape != targets.shape[:2] or not bool(valid.any()):
            raise ValueError("Need at least one valid target tick")
        # Padded tail values cannot influence valid predictions via self-attention.
        targets = targets.masked_fill(~valid.bool()[:, :, None], 0.)
        if noise is None:
            noise = torch.randn_like(targets)
        if t is None:
            t = torch.rand(targets.shape[0], device=targets.device)
        interpolated = (1 - t[:, None, None]) * noise + t[:, None, None] * targets
        error = (self.velocity(interpolated, t, memory, pad) - (targets - noise)).square()
        return (error * valid[:, :, None]).sum() / (valid.sum() * self.cfg.action_dim)

    def sample(self, memory, pad, *, steps=None, noise=None):
        steps = self.cfg.flow_steps if steps is None else steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("Flow steps must be a positive integer")
        x = torch.zeros(memory.shape[0], self.cfg.chunk, self.cfg.action_dim, device=memory.device) if noise is None else noise
        for i in range(steps):
            t = x.new_full((len(x),), i / steps)
            x = x + self.velocity(x, t, memory, pad) / steps
        return x


def load_backbones(settings):
    """Only official pinned model implementations; no remote Python execution."""
    from transformers import AutoTokenizer, CLIPTextModel, Dinov2Model
    vision = Dinov2Model.from_pretrained(settings["vision"]["id"], revision=settings["vision"]["revision"],
                                        use_safetensors=True, attn_implementation="sdpa")
    text = CLIPTextModel.from_pretrained(settings["text"]["id"], revision=settings["text"]["revision"],
                                        weights_only=True, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(settings["text"]["id"], revision=settings["text"]["revision"])
    return vision, text.requires_grad_(False).eval(), tokenizer


def save_checkpoint(path, model, *, metadata, text_features, text_valid, normalization):
    """Self-contained policy tensors and cached instruction, no executable pickle."""
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    tensors = {"model." + k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    tensors.update(text_features=text_features.detach().cpu().contiguous(),
                   text_valid=text_valid.detach().cpu().contiguous())
    save_file(tensors, str(path / "model.safetensors"))
    data = {"protocol": "chora-vla-skeleton-v1", "model_config": asdict(model.cfg),
            "vision_config": model.vision.config.to_dict(), "metadata": metadata,
            "normalization": normalization}
    (path / "config.json").write_text(json.dumps(data, indent=2) + "\n")


def load_checkpoint(path, device="cpu"):
    from safetensors.torch import load_file
    from transformers import Dinov2Config, Dinov2Model
    path = Path(path)
    data = json.loads((path / "config.json").read_text())
    if data["protocol"] != "chora-vla-skeleton-v1":
        raise ValueError("Unknown VLA checkpoint protocol")
    model = VisualActionModel(ModelConfig(**data["model_config"]),
                              Dinov2Model(Dinov2Config(**data["vision_config"])))
    tensors = load_file(str(path / "model.safetensors"))
    model.load_state_dict({k.removeprefix("model."): v for k, v in tensors.items() if k.startswith("model.")}, strict=True)
    return model.to(device).eval(), tensors["text_features"].to(device), tensors["text_valid"].to(device), data
