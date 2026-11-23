"""
Emotion / rasa conditioning for the Sooktam-2 DiT (F5-TTS v1 Base backbone).

Where the emotion goes in
-------------------------
Every DiTBlock in Sooktam-2 is modulated by AdaLayerNorm driven by the timestep
embedding `t` (shape b x 1024). Class-conditional DiT (Peebles & Xie) adds the
class embedding to exactly this vector, so we do the same:

        t' = time_embed(time) + EmotionConditioner(rasa weights, intensity)

The conditioner's last layer is zero-initialised, so before training the model
is *bit-identical* to the released Sooktam-2. Only the conditioner and
(optionally) LoRA adapters on attention are trained; the 336M backbone stays
frozen.

Guidance at inference
---------------------
Three velocity predictions per ODE step:
    v_c  : ref audio + text + emotion
    v_u  : nothing (the usual F5 unconditional branch, null emotion)
    v_ne : ref audio + text, null emotion
    v = v_c + cfg_text * (v_c - v_u) + cfg_emo * (v_c - v_ne)
cfg_emo is an "emotion strength" knob that pushes away from the reference
clip's own (usually neutral) prosody.
"""
from __future__ import annotations

import inspect
import math
from random import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_ckpt

from .rasa import NUM_BASIS


# --------------------------------------------------------------------------- #
# Conditioner
# --------------------------------------------------------------------------- #
class EmotionConditioner(nn.Module):
    """
    Two injection points (both zero-initialised):
      * global:  added to the timestep vector -> AdaLN modulation of every DiT block   (v1 and v2)
      * input:   added to every frame of the input embedding (before block 1)          (v2, input_proj=True)
    """

    def __init__(self, dim: int, n_basis: int = NUM_BASIS, emb_dim: int = 256, input_proj: bool = False):
        super().__init__()
        self.n_basis = n_basis
        self.table = nn.Parameter(torch.randn(n_basis + 1, emb_dim) * 0.02)  # last row = null
        self.mlp = nn.Sequential(nn.Linear(emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.inp = None
        if input_proj:
            self.inp = nn.Sequential(nn.Linear(emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.inp[-1].weight)
            nn.init.zeros_(self.inp[-1].bias)

    def mix(self, weights, null_mask=None, intensity=None):
        """
        weights:   (b, n_basis) mixture over basis emotions (rows sum to 1)
        null_mask: (b,) bool, True -> use the null embedding (CFG / no label)
        intensity: (b,) float, 0 = null, 1 = trained emotion, >1 = exaggerated
        """
        null = self.table[-1]
        emo = weights.to(self.table.dtype) @ self.table[:-1]
        if intensity is not None:
            emo = null + intensity.to(emo.dtype)[:, None] * (emo - null)
        if null_mask is not None:
            emo = torch.where(null_mask[:, None], null.expand_as(emo), emo)
        return emo

    def both(self, weights, null_mask=None, intensity=None):
        emo = self.mix(weights, null_mask, intensity)
        return self.mlp(emo), (self.inp(emo) if self.inp is not None else None)

    def forward(self, weights, null_mask=None, intensity=None):
        return self.mlp(self.mix(weights, null_mask, intensity))


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + self.lora_B(self.lora_A(self.drop(x))) * self.scale


def set_lora_enabled(module: nn.Module, enabled: bool):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


DEFAULT_LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0")


def inject_lora(transformer: nn.Module, targets=DEFAULT_LORA_TARGETS, r=16, alpha=16, dropout=0.0):
    replaced = []
    for name, module in list(transformer.transformer_blocks.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in targets):
            parent_name, _, child = name.rpartition(".")
            parent = transformer.transformer_blocks.get_submodule(parent_name) if parent_name else transformer.transformer_blocks
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)
    if not replaced:
        names = [n for n, m in transformer.transformer_blocks.named_modules() if isinstance(m, nn.Linear)][:20]
        raise RuntimeError(f"No LoRA targets matched {targets}. First Linear layers: {names}")
    return replaced


# --------------------------------------------------------------------------- #
# Forward pass with emotion (works for Sooktam-2's DiT and upstream F5-TTS DiT)
# --------------------------------------------------------------------------- #
def _block_kwargs(transformer):
    if not hasattr(transformer, "_emo_block_kw"):
        params = inspect.signature(transformer.transformer_blocks[0].forward).parameters
        transformer._emo_block_kw = {"skip_flash_attn": True} if "skip_flash_attn" in params else {}
        p2 = inspect.signature(transformer.get_input_embed).parameters
        transformer._emo_embed_has_mask = "audio_mask" in p2
    return transformer._emo_block_kw


def embed_inputs(transformer, x, cond, text, drop_audio_cond, drop_text, mask=None, cache=False):
    _block_kwargs(transformer)
    kw = dict(drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache)
    if transformer._emo_embed_has_mask:
        kw["audio_mask"] = mask
    return transformer.get_input_embed(x, cond, text, **kw)


def run_blocks(transformer, x, t, mask=None, grad_ckpt=False):
    seq_len = x.shape[1]
    rope = transformer.rotary_embed.forward_from_seq_len(seq_len)
    bkw = _block_kwargs(transformer)
    residual = x if transformer.long_skip_connection is not None else None
    for block in transformer.transformer_blocks:
        if grad_ckpt and torch.is_grad_enabled():
            x = torch_ckpt(lambda _x, _t, _b=block: _b(_x, _t, mask=mask, rope=rope, **bkw), x, t, use_reentrant=False)
        else:
            x = block(x, t, mask=mask, rope=rope, **bkw)
    if residual is not None:
        x = transformer.long_skip_connection(torch.cat((x, residual), dim=-1))
    x = transformer.norm_out(x, t)
    return transformer.proj_out(x)


def emotion_forward(transformer, x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False,
                    emo_weights=None, emo_null=None, intensity=None, grad_ckpt=False):
    b = x.shape[0]
    if time.ndim == 0:
        time = time.repeat(b)
    t = transformer.time_embed(time)
    x_shift = None
    if emo_weights is not None:
        t_shift, x_shift = transformer.emotion.both(emo_weights, emo_null, intensity)
        t = t + t_shift.to(t.dtype)
    h = embed_inputs(transformer, x, cond, text, drop_audio_cond, drop_text, mask=mask, cache=False)
    if x_shift is not None:
        h = h + x_shift.to(h.dtype)[:, None, :]
    return run_blocks(transformer, h, t, mask=mask, grad_ckpt=grad_ckpt)


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def attach_emotion(cfm, lora_r: int = 16, lora_alpha: int = 16, lora_targets=DEFAULT_LORA_TARGETS, emb_dim=256,
                   input_proj: bool = False):
    """Freeze Sooktam-2 and add the trainable emotion pathway. Returns list of trainable param names."""
    tr = cfm.transformer
    for p in cfm.parameters():
        p.requires_grad_(False)
    if not hasattr(tr, "emotion"):
        tr.emotion = EmotionConditioner(tr.dim, emb_dim=emb_dim, input_proj=input_proj).to(next(tr.parameters()).device)
    if lora_r and lora_r > 0 and not getattr(tr, "_lora_injected", False):
        inject_lora(tr, targets=lora_targets, r=lora_r, alpha=lora_alpha)
        tr._lora_injected = True
        tr.to(next(tr.parameters()).device)
    for n, p in tr.named_parameters():
        if n.startswith("emotion.") or "lora_" in n:
            p.requires_grad_(True)
    return [n for n, p in tr.named_parameters() if p.requires_grad]


def adapter_state_dict(transformer):
    return {n: p.detach().cpu() for n, p in transformer.named_parameters() if n.startswith("emotion.") or "lora_" in n}


def load_adapter(transformer, sd, strict=True):
    own = dict(transformer.named_parameters())
    missing = [k for k in own if (k.startswith("emotion.") or "lora_" in k) and k not in sd]
    unexpected = [k for k in sd if k not in own]
    if strict and (missing or unexpected):
        raise RuntimeError(f"adapter mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    with torch.no_grad():
        for k, v in sd.items():
            if k in own:
                own[k].copy_(v.to(own[k].device, own[k].dtype))


# --------------------------------------------------------------------------- #
# Training loss (replicates CFM.forward but with an explicit loss mask + emotion)
# --------------------------------------------------------------------------- #
def lens_to_mask(lens, length=None):
    length = length or int(lens.amax())
    return torch.arange(length, device=lens.device)[None, :] < lens[:, None]


def flow_matching_loss(cfm, mel, text, lens, loss_mask, emo_weights, emo_null,
                       audio_drop_prob=0.3, cond_drop_prob=0.2, grad_ckpt=True,
                       text_drop_keep_emo_prob=0.0, return_parts=False):
    """
    mel:       (b, n, 100) target log-mel (already padded)
    text:      (b, nt) long token ids, padded with -1
    lens:      (b,) valid frames
    loss_mask: (b, n) True where the model must generate (the infill span)
    text_drop_keep_emo_prob: v2 - hide the words but keep prompt audio + rasa label, so the label is the
                             only source of emotional prosody (Rasa text is emotion-specific, see diagnose).
    """
    tr = cfm.transformer
    b, n, _ = mel.shape
    valid = lens_to_mask(lens, n)
    loss_mask = loss_mask & valid

    x1 = mel
    x0 = torch.randn_like(x1)
    time = torch.rand((b,), dtype=x1.dtype, device=x1.device)
    tt = time[:, None, None]
    phi = (1 - tt) * x0 + tt * x1
    flow = x1 - x0
    cond = torch.where(loss_mask[..., None], torch.zeros_like(x1), x1)

    drop_audio, drop_text = False, False
    emo_null = emo_null.clone()
    r = random()
    if r < cond_drop_prob:                                   # fully unconditional branch -> null emotion too
        drop_audio, drop_text = True, True
        emo_null[:] = True
    elif r < cond_drop_prob + text_drop_keep_emo_prob:       # words hidden, label + prompt kept
        drop_text = True
    else:
        drop_audio = random() < audio_drop_prob

    pred = emotion_forward(tr, phi, cond, text, time, mask=valid, drop_audio_cond=drop_audio,
                           drop_text=drop_text, emo_weights=emo_weights, emo_null=emo_null, grad_ckpt=grad_ckpt)
    err = F.mse_loss(pred.float(), flow.float(), reduction="none").mean(-1)
    loss = (err * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    if not return_parts:
        return loss
    return loss, dict(pred=pred, phi=phi, time=time, loss_mask=loss_mask, emo_null=emo_null,
                      drop_text=drop_text, drop_audio=drop_audio)


def one_step_estimate(parts):
    """x1_hat = phi + (1 - t) * v   (the model's current guess of the clean mel)"""
    t = parts["time"][:, None, None].to(parts["pred"].dtype)
    return parts["phi"] + (1 - t) * parts["pred"]


# --------------------------------------------------------------------------- #
# Sampler with emotion guidance
# --------------------------------------------------------------------------- #
def _timesteps(steps, device, dtype, sway):
    t = None
    try:
        from f5_tts.model.utils import get_epss_timesteps  # present in Sooktam-2 / recent F5
        t = get_epss_timesteps(steps, device=device, dtype=dtype)
    except Exception:
        pass
    if t is None:
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    if sway is not None:
        t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


@torch.no_grad()
def sample_with_emotion(cfm, cond_mel, text, duration, emo_weights, intensity=1.0,
                        steps=32, cfg_text=2.0, cfg_emo=1.0, sway=-1.0, seed=None):
    """
    cond_mel: (1, n_ref, 100) reference log-mel
    text:     (1, nt) token ids (ref text + gen text), -1 padded
    duration: total frames (ref + generated)
    emo_weights: (1, n_basis) or None (=> plain Sooktam-2 behaviour)
    Returns full mel (1, duration, 100); caller cuts off the first n_ref frames.
    """
    tr = cfm.transformer
    dev = cond_mel.device
    dtype = next(tr.parameters()).dtype
    cond_mel = cond_mel.to(dtype)
    n_ref = cond_mel.shape[1]
    duration = int(max(duration, n_ref + 1))
    cond = F.pad(cond_mel, (0, 0, 0, duration - n_ref))
    cond_mask = (torch.arange(duration, device=dev) < n_ref)[None, :, None]
    step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

    use_emo = emo_weights is not None and hasattr(tr, "emotion")
    if use_emo:
        w = emo_weights.to(dev)
        inten = torch.full((1,), float(intensity), device=dev)
        e_c, s_c = tr.emotion.both(w, None, inten)
        e_n, s_n = tr.emotion.both(w, torch.ones(1, dtype=torch.bool, device=dev))
    branches = 3 if (use_emo and cfg_emo > 0) else 2

    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(1, duration, cfm.num_channels, device=dev, dtype=dtype)
    ts = _timesteps(steps, dev, dtype, sway)

    tr.clear_cache()
    for i in range(len(ts) - 1):
        t_cur = ts[i]
        te = tr.time_embed(t_cur.repeat(1))
        hc0 = embed_inputs(tr, x, step_cond, text, False, False, cache=True)
        hu = embed_inputs(tr, x, step_cond, text, True, True, cache=True)
        hc = hc0
        if use_emo and s_c is not None:
            hc, hu = hc0 + s_c[:, None, :], hu + s_n[:, None, :]
        hs, tvec = [hc, hu], [te + (e_c if use_emo else 0), te + (e_n if use_emo else 0)]
        if branches == 3:
            hs.append(hc0 + s_n[:, None, :] if s_c is not None else hc0)
            tvec.append(te + e_n)
        out = run_blocks(tr, torch.cat(hs), torch.cat(tvec).to(dtype))
        outs = out.chunk(branches)
        v = outs[0] + cfg_text * (outs[0] - outs[1])
        if branches == 3:
            v = v + cfg_emo * (outs[0] - outs[2])
        x = x + (ts[i + 1] - t_cur) * v
    tr.clear_cache()
    return x
