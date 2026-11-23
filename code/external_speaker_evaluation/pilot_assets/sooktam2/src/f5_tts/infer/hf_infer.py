"""Inference helpers intended for Hugging Face usage (no HTTP server required)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable, Optional, Tuple

from f5_tts.api import F5TTS

ENV_DEFAULTS = {
    "model": os.environ.get("F5TTS_MODEL", "F5TTS_v1_Base"),
    "ckpt_file": os.environ.get("F5TTS_CKPT", ""),
    "vocab_file": os.environ.get("F5TTS_VOCAB", ""),
    "ode_method": os.environ.get("F5TTS_ODE_METHOD", "euler"),
    "use_ema": os.environ.get("F5TTS_USE_EMA", "true").lower() != "false",
    "vocoder_local_path": os.environ.get("F5TTS_VOCODER_PATH"),
    "device": os.environ.get("F5TTS_DEVICE"),
    "hf_cache_dir": os.environ.get("F5TTS_HF_CACHE_DIR"),
}


@lru_cache(maxsize=2)
def load_tts(
    model: str = ENV_DEFAULTS["model"],
    ckpt_file: str = ENV_DEFAULTS["ckpt_file"],
    vocab_file: str = ENV_DEFAULTS["vocab_file"],
    ode_method: str = ENV_DEFAULTS["ode_method"],
    use_ema: bool = ENV_DEFAULTS["use_ema"],
    vocoder_local_path: Optional[str] = ENV_DEFAULTS["vocoder_local_path"],
    device: Optional[str] = ENV_DEFAULTS["device"],
    hf_cache_dir: Optional[str] = ENV_DEFAULTS["hf_cache_dir"],
) -> F5TTS:
    """Load and cache an F5TTS model for inference."""
    return F5TTS(
        model=model,
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        ode_method=ode_method,
        use_ema=use_ema,
        vocoder_local_path=vocoder_local_path,
        device=device,
        hf_cache_dir=hf_cache_dir,
    )


def synthesize(
    tts: F5TTS,
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    *,
    target_rms: float = 0.1,
    cross_fade_duration: float = 0.15,
    sway_sampling_coef: float = -1.0,
    cfg_strength: float = 2.0,
    nfe_step: int = 32,
    speed: float = 1.0,
    fix_duration: Optional[float] = None,
    remove_silence: bool = False,
    seed: Optional[int] = None,
    tokenizer: str = "pinyin",
    cls_language: Optional[str] = None,
    cls_tokenizer_fn: Optional[Callable[[str, str], list]] = None,
    cls_server_url: Optional[str] = None,
    cls_timeout: float = 5.0,
    file_wave: Optional[str] = None,
    file_spec: Optional[str] = None,
    show_info=None,
    progress=None,
) -> Tuple[Any, int, Optional[Any]]:
    """Run inference and return (wav, sample_rate, spectrogram)."""
    if show_info is None:
        show_info = lambda *args, **kwargs: None

    return tts.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=gen_text,
        show_info=show_info,
        progress=progress,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        sway_sampling_coef=sway_sampling_coef,
        cfg_strength=cfg_strength,
        nfe_step=nfe_step,
        speed=speed,
        fix_duration=fix_duration,
        remove_silence=remove_silence,
        file_wave=file_wave,
        file_spec=file_spec,
        seed=seed,
        tokenizer=tokenizer,
        cls_language=cls_language,
        cls_tokenizer_fn=cls_tokenizer_fn,
        cls_server_url=cls_server_url,
        cls_timeout=cls_timeout,
    )
