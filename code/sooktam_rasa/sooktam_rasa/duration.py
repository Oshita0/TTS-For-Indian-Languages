"""
Learned cross-attention duration predictor (replaces Sooktam-2's ratio heuristic).

Sooktam-2 (fixed) does:   gen_frames = ref_frames / len(ref_text) * len(gen_text)
which assumes every grapheme takes the same time. That left ~25% error on
Dravidian scripts in the bug report (Sec. 10).

This model keeps that formula's structure but learns its parts:

    u_i    = softplus(MLP(TextEnc(token_i, lang)))      intrinsic "duration weight" of each CLS token
    tau    = ref_frames / sum_{j in ref} u_j            speaker tempo, measured on the reference
    c_i    = Head(CrossAttn(tgt token_i  ->  RefMelEnc(ref mel)) + emotion)    zero-initialised residual
    frames = tau * sum_{i in tgt} u_i * exp(c_i)

At init (u = 1, c = 0) it equals the token-count heuristic, so training can
only improve on the baseline. Emotion enters c_i (sad speech is slower, angry
faster). Cross-attention over the reference mel lets the model pick up the
reference's articulation/pausing style, not just its average rate.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .rasa import LANGS, NUM_BASIS, UNK_LANG_ID


def sinusoid(n, d, device):
    pos = torch.arange(n, device=device).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class CrossAttnDurationPredictor(nn.Module):
    def __init__(self, vocab_size, n_mels=100, d=256, heads=4, text_layers=3, mel_layers=2, xattn_layers=2,
                 dropout=0.1, lang_drop=0.2):
        super().__init__()
        self.lang_drop = lang_drop
        self.tok = nn.Embedding(vocab_size + 1, d, padding_idx=0)       # ids are shifted by +1, 0 = pad
        self.lang = nn.Embedding(len(LANGS) + 1, d)                     # last = unknown
        enc = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.text_enc = nn.TransformerEncoder(enc(), text_layers)
        self.u_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.u_head[-1].weight)
        nn.init.constant_(self.u_head[-1].bias, math.log(math.e - 1))  # softplus(.) = 1 at init

        self.mel_in = nn.Sequential(nn.Conv1d(n_mels, d, 5, stride=2, padding=2), nn.GELU(),
                                    nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU())
        self.mel_enc = nn.TransformerEncoder(enc(), mel_layers)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.xattn = nn.TransformerDecoder(dec, xattn_layers)
        self.emo = nn.Linear(NUM_BASIS, d)
        self.c_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.c_head[-1].weight)
        nn.init.zeros_(self.c_head[-1].bias)

    def _encode_text(self, ids, lang):
        mask = ids == 0
        h = self.tok(ids) + sinusoid(ids.shape[1], self.tok.embedding_dim, ids.device)[None]
        h = h + self.lang(lang)[:, None]
        h = self.text_enc(h, src_key_padding_mask=mask)
        u = F.softplus(self.u_head(h).squeeze(-1)).masked_fill(mask, 0.0)
        return h, u, mask

    def forward(self, ref_mel, ref_mel_lens, ref_ids, tgt_ids, lang, emo_weights, ref_lang=None):
        if self.training and self.lang_drop > 0:
            lang = torch.where(torch.rand_like(lang.float()) < self.lang_drop, torch.full_like(lang, UNK_LANG_ID), lang)
        ref_lang = lang if ref_lang is None else ref_lang

        _, u_ref, _ = self._encode_text(ref_ids, ref_lang)
        tau = ref_mel_lens.float() / u_ref.sum(1).clamp(min=1e-3)                      # frames per unit

        m = self.mel_in(ref_mel.transpose(1, 2)).transpose(1, 2)                         # 4x downsampled
        m_lens = torch.div(ref_mel_lens + 3, 4, rounding_mode="floor").clamp(min=1, max=m.shape[1])
        m_mask = torch.arange(m.shape[1], device=m.device)[None] >= m_lens[:, None]
        m = self.mel_enc(m + sinusoid(m.shape[1], m.shape[2], m.device)[None], src_key_padding_mask=m_mask)

        h, u_tgt, t_mask = self._encode_text(tgt_ids, lang)
        h = h + self.emo(emo_weights)[:, None]
        h = self.xattn(h, m, tgt_key_padding_mask=t_mask, memory_key_padding_mask=m_mask)
        c = self.c_head(h).squeeze(-1).clamp(-3, 3).masked_fill(t_mask, 0.0)
        per_token = tau[:, None] * u_tgt * torch.exp(c)
        return per_token.sum(1), per_token, tau

    @staticmethod
    def loss(pred_frames, true_frames, tau=None, tau_reg=0.0):
        l = F.l1_loss(torch.log(pred_frames.clamp(min=1.0)), torch.log(true_frames.clamp(min=1.0)))
        if tau_reg > 0 and tau is not None:
            # keeps speaker tempo near a shared scale so per-language u's stay comparable across scripts
            l = l + tau_reg * (torch.log(tau) - torch.log(tau).mean().detach()).pow(2).mean()
        return l


# baselines from the bug report ------------------------------------------------
def heuristic_frames(ref_frames, ref_len, tgt_len):
    return ref_frames / ref_len.clamp(min=1) * tgt_len


def abs_pct_err(pred, true):
    return ((pred - true).abs() / true.clamp(min=1)).mul(100)
