"""Hugging Face AutoModel integration for F5-TTS (inference-only)."""

from __future__ import annotations

import os
from typing import Any, List, Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from f5_tts.api import F5TTS

logger = logging.get_logger(__name__)


class F5TTSConfig(PretrainedConfig):
    model_type = "f5_tts"

    def __init__(
        self,
        model_name: str = "F5TTS_v1_Base",
        ckpt_file: str = "",
        vocab_file: str = "",
        ode_method: str = "euler",
        use_ema: bool = True,
        vocoder_local_path: Optional[str] = None,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.ode_method = ode_method
        self.use_ema = use_ema
        self.vocoder_local_path = vocoder_local_path
        self.device = device
        self.hf_cache_dir = hf_cache_dir

        if "auto_map" not in kwargs:
            # Keep AutoTokenizer as a string to satisfy Hub config validators.
            self.auto_map = {
                "AutoConfig": "hf_auto.F5TTSConfig",
                "AutoModel": "hf_auto.F5TTSAutoModel",
                "AutoTokenizer": "hf_auto.F5TTSTokenizer",
            }


class F5TTSTokenizer(PreTrainedTokenizer):
    """Minimal character-level tokenizer backed by vocab.txt (inference helper)."""

    vocab_files_names = {"vocab_file": "vocab.txt"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, **kwargs) -> None:
        self.vocab_file = vocab_file
        tokens = self._load_vocab_tokens(vocab_file)
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.ids_to_tokens = {idx: tok for tok, idx in self.vocab.items()}

        if kwargs.get("unk_token") is None:
            kwargs["unk_token"] = "<unk>"
        super().__init__(**kwargs)

        if self.unk_token not in self.vocab:
            unk_id = len(self.vocab)
            self.vocab[self.unk_token] = unk_id
            self.ids_to_tokens[unk_id] = self.unk_token

    @staticmethod
    def _load_vocab_tokens(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle]

    def get_vocab(self) -> dict:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        os.makedirs(save_directory, exist_ok=True)
        filename = (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
        path = os.path.join(save_directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            for idx in range(len(self.ids_to_tokens)):
                handle.write(f"{self.ids_to_tokens[idx]}\n")
        return (path,)


def load_tokenizer(
    repo_or_path: str = "bharatgenai/sooktam2",
    vocab_file: str = "vocab.txt",
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> F5TTSTokenizer:
    """Load the character-level tokenizer from a local folder or Hugging Face."""
    resolved = F5TTSAutoModel._resolve_file(
        vocab_file,
        repo_or_path,
        cache_dir,
        revision,
        token,
        local_files_only,
    )
    return F5TTSTokenizer(resolved)


class F5TTSAutoModel(PreTrainedModel):
    config_class = F5TTSConfig

    def __init__(self, config: F5TTSConfig, ckpt_file: str = "", vocab_file: str = "", **kwargs) -> None:
        super().__init__(config)
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.tts = F5TTS(
            model=config.model_name,
            ckpt_file=ckpt_file or config.ckpt_file,
            vocab_file=vocab_file or config.vocab_file,
            ode_method=config.ode_method,
            use_ema=config.use_ema,
            vocoder_local_path=config.vocoder_local_path,
            device=config.device,
            hf_cache_dir=config.hf_cache_dir,
        )

    @staticmethod
    def _resolve_file(
        filename: str,
        repo_or_path: Optional[str],
        cache_dir: Optional[str],
        revision: Optional[str],
        token: Optional[str],
        local_files_only: bool,
    ) -> str:
        if not filename:
            return ""
        if os.path.isfile(filename):
            return filename
        if repo_or_path and os.path.isdir(repo_or_path):
            candidate = os.path.join(repo_or_path, filename)
            if os.path.isfile(candidate):
                return candidate
        if not repo_or_path:
            return filename
        return hf_hub_download(
            repo_id=repo_or_path,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[str], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config_kwargs = {
                "cache_dir": kwargs.get("cache_dir"),
                "revision": kwargs.get("revision"),
                "token": kwargs.get("token"),
                "local_files_only": kwargs.get("local_files_only", False),
                "trust_remote_code": kwargs.get("trust_remote_code"),
            }
            try:
                config = F5TTSConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
            except Exception:  # noqa: BLE001
                logger.warning("F5TTSConfig not found, using defaults.")
                config = F5TTSConfig()

        ckpt_file = kwargs.pop("ckpt_file", None) or config.ckpt_file
        vocab_file = kwargs.pop("vocab_file", None) or config.vocab_file

        cache_dir = kwargs.get("cache_dir") or config.hf_cache_dir
        revision = kwargs.get("revision")
        token = kwargs.get("token")
        local_files_only = kwargs.get("local_files_only", False)

        ckpt_file = cls._resolve_file(
            ckpt_file,
            pretrained_model_name_or_path,
            cache_dir,
            revision,
            token,
            local_files_only,
        )
        vocab_file = cls._resolve_file(
            vocab_file,
            pretrained_model_name_or_path,
            cache_dir,
            revision,
            token,
            local_files_only,
        )

        return cls(config, ckpt_file=ckpt_file, vocab_file=vocab_file)

    def forward(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError("Use .infer(...) or .tts.infer(...) for generation.")

    def infer(self, *args, **kwargs):
        return self.tts.infer(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)


def register_f5tts_auto() -> None:
    """Register F5-TTS with Hugging Face AutoConfig/AutoModel/AutoTokenizer (local usage)."""
    AutoConfig.register(F5TTSConfig.model_type, F5TTSConfig)
    AutoModel.register(F5TTSConfig, F5TTSAutoModel)
    AutoTokenizer.register(F5TTSConfig, F5TTSTokenizer)
