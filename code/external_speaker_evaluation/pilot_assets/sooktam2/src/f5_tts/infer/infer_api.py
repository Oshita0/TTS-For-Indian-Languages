"""FastAPI server for F5-TTS inference.

Launch with a custom checkpoint:
    python src/f5_tts/infer/infer_api.py --ckpt-file ckpts/my_model.safetensors --vocab-file ckpts/vocab.txt

The API exposes:
- GET  /health         -> basic readiness info
- POST /v1/tts         -> synthesize speech (JSON body)
"""

import base64
import io
import os
import tempfile
import threading
from functools import lru_cache
from typing import Optional

import click
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import save_spectrogram

# Allow configuration through environment variables for quick overrides
ENV_DEFAULTS = {
    "model": os.environ.get("F5TTS_API_MODEL", "F5TTS_v1_Base"),
    "ckpt_file": os.environ.get(
        "F5TTS_API_CKPT",
        "/workspace/personal/team_folders/F5-TTS-common/ckpts/F5TTS_v1_Base_vocos_cls_speech_db_wer_filtered_12_langs_train_finetune_cls/"
        "model_1250000.pt",
    ),
    "vocab_file": os.environ.get(
        "F5TTS_API_VOCAB",
        "/workspace/personal/team_folders/F5-TTS-common/ckpts/F5TTS_v1_Base_vocos_cls_speech_db_wer_filtered_12_langs_train_finetune_cls/"
        "vocab.txt",
    ),
    "ode_method": os.environ.get("F5TTS_API_ODE_METHOD", "euler"),
    "use_ema": os.environ.get("F5TTS_API_USE_EMA", "true").lower() != "false",
    "vocoder_local_path": os.environ.get("F5TTS_API_VOCODER_PATH"),
    "device": os.environ.get("F5TTS_API_DEVICE"),
    "hf_cache_dir": os.environ.get("F5TTS_API_HF_CACHE_DIR"),
    "en_model": os.environ.get("F5TTS_API_EN_MODEL", os.environ.get("F5TTS_API_MODEL", "F5TTS_v1_Base")),
    "en_ckpt_file": os.environ.get(
        "F5TTS_API_EN_CKPT",
        "/workspace/personal/team_folders/vansh.pundir/F5-TTS/ckpts/"
        "F5TTS_v1_Base_12_lang_vocos_char_speech_db_only_TTS_12_langs_eval_v3_char_dedup_validation/"
        "model_550000.pt",
    ),
    "en_vocab_file": os.environ.get(
        "F5TTS_API_EN_VOCAB",
        "/workspace/personal/team_folders/vansh.pundir/F5-TTS/ckpts/"
        "F5TTS_v1_Base_12_lang_vocos_char_speech_db_only_TTS_12_langs_eval_v3_char_dedup_validation/"
        "vocab.txt",
    ),
    "cls_url": os.environ.get("F5TTS_CLS_URL", "http://localhost:8061/process"),
    "cls_timeout": float(os.environ.get("F5TTS_CLS_TIMEOUT", "5.0")),
}


class InferenceRequest(BaseModel):
    ref_audio_path: Optional[str] = Field(
        default=None, description="Path to reference audio reachable by the server."
    )
    ref_audio_base64: Optional[str] = Field(
        default=None, description="Base64-encoded reference audio (recommended: WAV/FLAC)."
    )
    ref_text: str = Field(
        default="",
        description="Transcript of the reference audio. Leave blank to auto-transcribe (requires ASR).",
    )
    gen_text: str = Field(..., description="Text to synthesize.")
    target_rms: float = Field(default=0.1, description="Minimum RMS applied to reference audio.")
    cross_fade_duration: float = Field(default=0.15, description="Seconds to overlap between chunks.")
    sway_sampling_coef: float = Field(default=-1.0, description="Sway sampling coefficient.")
    cfg_strength: float = Field(default=2.0, description="Classifier-free guidance strength.")
    nfe_step: int = Field(default=32, description="Number of function evaluations.")
    speed: float = Field(default=1.0, description="Generation speed multiplier.")
    fix_duration: Optional[float] = Field(
        default=None, description="Force output duration (seconds). Leave None for automatic."
    )
    remove_silence: bool = Field(default=False, description="Remove leading/trailing silence from output.")
    seed: Optional[int] = Field(default=None, description="Set for deterministic output.")
    return_spectrogram: bool = Field(default=False, description="Also return spectrogram as base64 PNG.")
    tokenizer: Optional[str] = Field(
        default=None,
        description="Optional tokenizer override: char | cls | pinyin. If omitted, uses legacy pinyin behavior.",
    )
    cls_language: Optional[str] = Field(
        default=None,
        description="CLS language name (e.g., hindi, english). Used only when tokenizer=cls.",
    )

    @model_validator(mode="after")
    def ensure_audio_source(self):
        if not self.ref_audio_path and not self.ref_audio_base64:
            raise ValueError("Provide either ref_audio_path or ref_audio_base64.")
        if not self.gen_text or not self.gen_text.strip():
            raise ValueError("gen_text cannot be empty.")
        return self


def _encode_wav_base64(wav, sample_rate: int) -> str:
    """Encode waveform to a base64 WAV string."""
    with io.BytesIO() as buffer:
        sf.write(buffer, wav, sample_rate, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_spec_base64(spec) -> str:
    """Save spectrogram to a temp file and encode it as base64 PNG."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_spectrogram(spec, tmp_path)
        with open(tmp_path, "rb") as img:
            return base64.b64encode(img.read()).decode("ascii")
    finally:
        os.remove(tmp_path)


def _write_temp_audio(data: bytes) -> str:
    """Persist uploaded audio bytes to a temp file for downstream processing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        return tmp.name


@lru_cache(maxsize=4)
def _load_model(
    model: str = ENV_DEFAULTS["model"],
    ckpt_file: str = ENV_DEFAULTS["ckpt_file"],
    vocab_file: str = ENV_DEFAULTS["vocab_file"],
    ode_method: str = ENV_DEFAULTS["ode_method"],
    use_ema: bool = ENV_DEFAULTS["use_ema"],
    vocoder_local_path: Optional[str] = ENV_DEFAULTS["vocoder_local_path"],
    device: Optional[str] = ENV_DEFAULTS["device"],
    hf_cache_dir: Optional[str] = ENV_DEFAULTS["hf_cache_dir"],
):
    """Cache TTS models by configuration to avoid reloading across requests."""
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


def create_app(
    model: str = ENV_DEFAULTS["model"],
    ckpt_file: str = ENV_DEFAULTS["ckpt_file"],
    vocab_file: str = ENV_DEFAULTS["vocab_file"],
    en_model: str = ENV_DEFAULTS["en_model"],
    en_ckpt_file: str = ENV_DEFAULTS["en_ckpt_file"],
    en_vocab_file: str = ENV_DEFAULTS["en_vocab_file"],
    ode_method: str = ENV_DEFAULTS["ode_method"],
    use_ema: bool = ENV_DEFAULTS["use_ema"],
    vocoder_local_path: Optional[str] = ENV_DEFAULTS["vocoder_local_path"],
    device: Optional[str] = ENV_DEFAULTS["device"],
    hf_cache_dir: Optional[str] = ENV_DEFAULTS["hf_cache_dir"],
):
    """Build a FastAPI app wired to a single F5TTS instance."""
    tts_hi = _load_model(
        model=model,
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        ode_method=ode_method,
        use_ema=use_ema,
        vocoder_local_path=vocoder_local_path,
        device=device,
        hf_cache_dir=hf_cache_dir,
    )
    infer_lock_hi = threading.Lock()
    infer_lock_en = threading.Lock()

    app = FastAPI(title="F5-TTS API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": tts_hi.device,
            "mel_spec_type": tts_hi.mel_spec_type,
            "use_ema": tts_hi.use_ema,
            "supported_langs": ["hi", "en"],
        }

    @app.post("/v1/tts")
    def infer(payload: InferenceRequest, lang: str = Query("hi", description="Language code: hi|en")):
        lang_key = (lang or "hi").strip().lower()
        if lang_key == "hi":
            tts = tts_hi
            infer_lock = infer_lock_hi
        elif lang_key == "en":
            tts = _load_model(
                model=en_model,
                ckpt_file=en_ckpt_file,
                vocab_file=en_vocab_file,
                ode_method=ode_method,
                use_ema=use_ema,
                vocoder_local_path=vocoder_local_path,
                device=device,
                hf_cache_dir=hf_cache_dir,
            )
            infer_lock = infer_lock_en
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported lang '{lang}'. Use 'hi' for Hindi or 'en' for English.",
            )

        if lang_key == "hi":
            tokenizer_used = "cls"
        elif lang_key == "en":
            tokenizer_used = "char"
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported lang for hard-coded tokenizer. Use 'hi' or 'en'.",
            )

        cls_language = None
        if tokenizer_used == "cls":
            if payload.cls_language and payload.cls_language.strip():
                cls_language = payload.cls_language.strip().lower()
            else:
                cls_language = "hindi" if lang_key == "hi" else "english" if lang_key == "en" else None
            if not cls_language:
                raise HTTPException(
                    status_code=400,
                    detail="cls_language is required when tokenizer=cls and lang is not hi/en.",
                )

        cleanup_path = None
        if payload.ref_audio_path:
            ref_audio = payload.ref_audio_path
            if not os.path.exists(ref_audio):
                raise HTTPException(status_code=400, detail=f"ref_audio_path not found: {ref_audio}")
        else:
            try:
                audio_bytes = base64.b64decode(payload.ref_audio_base64)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Invalid ref_audio_base64: {exc}") from exc
            ref_audio = _write_temp_audio(audio_bytes)
            cleanup_path = ref_audio

        try:
            with infer_lock:
                try:
                    wav, sr, spec = tts.infer(
                        ref_file=ref_audio,
                        ref_text=payload.ref_text,
                        gen_text=payload.gen_text,
                        show_info=lambda *args, **kwargs: None,
                        progress=None,
                        target_rms=payload.target_rms,
                        cross_fade_duration=payload.cross_fade_duration,
                        sway_sampling_coef=payload.sway_sampling_coef,
                        cfg_strength=payload.cfg_strength,
                        nfe_step=payload.nfe_step,
                        speed=payload.speed,
                        fix_duration=payload.fix_duration,
                        remove_silence=payload.remove_silence,
                        seed=payload.seed,
                        tokenizer=tokenizer_used,
                        cls_language=cls_language,
                        cls_server_url=ENV_DEFAULTS["cls_url"],
                        cls_timeout=ENV_DEFAULTS["cls_timeout"],
                    )
                except Exception as exc:  # noqa: BLE001
                    if tokenizer_used == "cls":
                        raise HTTPException(
                            status_code=502,
                            detail=f"CLS tokenization failed: {exc}",
                        ) from exc
                    raise
        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                os.remove(cleanup_path)

        response = {
            "audio_base64": _encode_wav_base64(wav, sr),
            "sample_rate": sr,
            "seed": getattr(tts, "seed", payload.seed),
        }
        if payload.return_spectrogram and spec is not None:
            response["spectrogram_base64"] = _encode_spec_base64(spec)

        return response

    return app


app = create_app()


@click.command()
@click.option("--model", default=ENV_DEFAULTS["model"], show_default=True, help="Model config name to load.")
@click.option("--ckpt-file", default=ENV_DEFAULTS["ckpt_file"], show_default=True, help="Checkpoint file path.")
@click.option("--vocab-file", default=ENV_DEFAULTS["vocab_file"], show_default=True, help="Custom vocab file path.")
@click.option("--ode-method", default=ENV_DEFAULTS["ode_method"], show_default=True, help="ODE method for sampler.")
@click.option(
    "--use-ema/--no-use-ema",
    default=ENV_DEFAULTS["use_ema"],
    show_default=True,
    help="Load EMA weights from checkpoint.",
)
@click.option(
    "--vocoder-local-path",
    default=ENV_DEFAULTS["vocoder_local_path"],
    show_default=True,
    help="Local vocoder directory (skips HF download).",
)
@click.option("--device", default=ENV_DEFAULTS["device"], show_default=True, help="Force device: cpu|cuda|mps|xpu.")
@click.option(
    "--hf-cache-dir",
    default=ENV_DEFAULTS["hf_cache_dir"],
    show_default=True,
    help="HuggingFace cache directory override.",
)
@click.option("--host", default="0.0.0.0", show_default=True, help="API host.")
@click.option("--port", default=8060, show_default=True, help="API port.", type=int)
@click.option("--root-path", default="", show_default=True, help="Set FastAPI root_path when behind a proxy.")
def main(
    model,
    ckpt_file,
    vocab_file,
    ode_method,
    use_ema,
    vocoder_local_path,
    device,
    hf_cache_dir,
    host,
    port,
    root_path,
):
    """Run the FastAPI server for HTTP inference."""
    api_app = create_app(
        model=model,
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        ode_method=ode_method,
        use_ema=use_ema,
        vocoder_local_path=vocoder_local_path,
        device=device,
        hf_cache_dir=hf_cache_dir,
    )
    uvicorn.run(api_app, host=host, port=port, root_path=root_path)


if __name__ == "__main__":
    main()
