"""
Load Sooktam-2 (+ emotion adapter, + learned duration model) and synthesise speech in a chosen rasa.

    tts = EmotionalSooktam.load("/content/sooktam2", adapter="ckpt/emotion/best.pt",
                                duration_model="ckpt/duration/best.pt")
    wav, sr = tts.synthesize("ref.wav", "reference transcript", "बोलने वाला वाक्य", "hindi",
                             rasa="karuna", intensity=1.0, cfg_emo=1.0, out_path="karuna.wav")
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .emotion_dit import attach_emotion, load_adapter, sample_with_emotion, set_lora_enabled
from .rasa import LANG_TO_ID, NAVARASA, UNK_LANG_ID, rasa_weights

HOP, SR = 256, 24000


def load_sooktam(sooktam_dir, ckpt=None, device=None):
    """Build Sooktam-2 through its own F5TTS API (needs `pip install -e <sooktam_dir>`)."""
    from f5_tts.api import F5TTS

    if ckpt is None:
        for c in ("sooktam.safetensors", "model_1250000.pt"):
            if os.path.exists(os.path.join(sooktam_dir, c)):
                ckpt = c
                break
    ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(sooktam_dir, ckpt)
    return F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt_path,
                 vocab_file=os.path.join(sooktam_dir, "vocab.txt"), device=device)


class EmotionalSooktam:
    def __init__(self, f5, dur_model=None):
        self.f5, self.cfm, self.vocoder = f5, f5.ema_model, f5.vocoder
        self.device = f5.device
        self.dur = dur_model
        self.vmap = self.cfm.vocab_char_map

    @classmethod
    def load(cls, sooktam_dir, adapter=None, duration_model=None, ckpt=None, device=None):
        f5 = load_sooktam(sooktam_dir, ckpt, device)
        if adapter:
            blob = torch.load(adapter, map_location="cpu")
            cfg = blob.get("config", {})
            attach_emotion(f5.ema_model, lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 16),
                           input_proj=cfg.get("emo_input_proj", False))
            load_adapter(f5.ema_model.transformer, blob["adapter"])
        f5.ema_model.eval()
        dur = None
        if duration_model:
            from .duration import CrossAttnDurationPredictor
            blob = torch.load(duration_model, map_location="cpu")
            dur = CrossAttnDurationPredictor(**blob["model_kwargs"])
            dur.load_state_dict(blob["model"])
            dur.to(f5.device).eval()
        return cls(f5, dur)

    # ------------------------------------------------------------------ #
    def _tokens(self, text, language):
        from f5_tts.infer.utils_infer import tokenize_texts
        return tokenize_texts([text], tokenizer="cls", cls_language=language)[0]

    def _ids(self, toks):
        return [self.vmap.get(t, 0) for t in toks]

    def _ref(self, ref_file, ref_text, target_rms=0.1):
        import torchaudio
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, show_info=lambda *a: None)
        audio, sr = torchaudio.load(ref_file)
        audio = audio.mean(0, keepdim=True)
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != SR:
            audio = torchaudio.functional.resample(audio, sr, SR)
        mel = self.cfm.mel_spec(audio.to(self.device)).permute(0, 2, 1)  # (1, T, 100)
        return mel, ref_text, float(rms)

    @torch.no_grad()
    def predict_gen_frames(self, ref_mel, ref_text, gen_text, language, ref_language=None, emo_w=None,
                           method="learned", speed=1.0):
        n_ref = ref_mel.shape[1]
        if isinstance(method, (int, float)) and not isinstance(method, bool):
            return int(method * SR / HOP)
        if method == "learned" and self.dur is not None:
            ref_ids = torch.tensor(self._ids(self._tokens(ref_text, ref_language or language)), device=self.device)[None] + 1
            tgt_ids = torch.tensor(self._ids(self._tokens(gen_text, language)), device=self.device)[None] + 1
            lang = torch.tensor([LANG_TO_ID.get(language, UNK_LANG_ID)], device=self.device)
            rlang = torch.tensor([LANG_TO_ID.get(ref_language or language, UNK_LANG_ID)], device=self.device)
            frames, _, _ = self.dur(ref_mel.float(), torch.tensor([n_ref], device=self.device), ref_ids, tgt_ids, lang,
                                    emo_w.to(self.device), ref_lang=rlang)
            return int(frames.item() / speed)
        if method in ("learned", "grapheme"):
            import grapheme
            return int(n_ref / grapheme.length(ref_text) * grapheme.length(gen_text) / speed)
        if method == "bytes":  # original (buggy) Sooktam-2 formula, for comparison
            return int(n_ref / len(ref_text.encode()) * len(gen_text.encode()) / speed)
        raise ValueError(method)

    @torch.no_grad()
    def synthesize(self, ref_file, ref_text, gen_text, language, rasa="shanta", intensity=1.0,
                   cfg_text=2.0, cfg_emo=1.0, nfe_step=32, sway=-1.0, speed=1.0, duration="learned",
                   ref_language=None, seed=None, out_path=None):
        """
        rasa: one of NAVARASA keys, a dict mixture e.g. {"karuna": .7, "shanta": .3}, or None (plain Sooktam-2)
        duration: "learned" | "grapheme" | "bytes" | float seconds for the generated part
        """
        ref_mel, ref_text, rms = self._ref(ref_file, ref_text)
        emo_w = rasa_weights(rasa)[None] if rasa is not None else None
        if emo_w is not None and not hasattr(self.cfm.transformer, "emotion"):
            raise RuntimeError("No emotion adapter loaded; pass adapter=... or rasa=None")
        n_ref = ref_mel.shape[1]
        gen_frames = self.predict_gen_frames(ref_mel, ref_text, gen_text, language, ref_language,
                                             emo_w if emo_w is not None else rasa_weights("shanta")[None],
                                             duration, speed)
        ids = torch.tensor(self._ids(self._tokens(ref_text + gen_text, language)), device=self.device)[None]
        # rasa=None means "plain Sooktam-2": switch LoRA off too, so it is the true base model
        set_lora_enabled(self.cfm.transformer, emo_w is not None)
        try:
            mel = sample_with_emotion(self.cfm, ref_mel, ids, n_ref + gen_frames, emo_w, intensity=intensity,
                                      steps=nfe_step, cfg_text=cfg_text, cfg_emo=cfg_emo, sway=sway, seed=seed)
        finally:
            set_lora_enabled(self.cfm.transformer, True)
        gen = mel[:, n_ref:].permute(0, 2, 1).float()
        wav = self.vocoder.decode(gen).squeeze().cpu().numpy().astype(np.float32)
        if rms < 0.1:
            wav = wav * rms / 0.1
        if out_path:
            import soundfile as sf
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            sf.write(out_path, wav, SR)
        return wav, SR

    def navarasa_sweep(self, ref_file, ref_text, gen_text, language, out_dir, rasas=None, **kw):
        paths = {}
        for r in rasas or list(NAVARASA):
            p = os.path.join(out_dir, f"{language}_{r}.wav")
            self.synthesize(ref_file, ref_text, gen_text, language, rasa=r, out_path=p, **kw)
            paths[r] = p
        return paths
