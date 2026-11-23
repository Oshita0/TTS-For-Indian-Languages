"""
Navarasa <-> emotion label space for Sooktam-2.

Rasa (AI4Bharat, Interspeech 2024) records the six Ekman emotions plus neutral
reading styles. Seven of the nine rasas of the Natyashastra map onto those
recordings directly. The other two (Shringara, Veera) have no recordings, so
they are defined as mixtures of the learned basis embeddings. That makes them
zero-shot / experimental and they must be judged by listening tests, not
claimed as trained classes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

# ---- basis emotions that actually have training data -------------------------
# Order matters: index i <-> row i of the emotion embedding table.
BASIS = ["shanta", "hasya", "karuna", "raudra", "bhayanaka", "bibhatsa", "adbhuta"]
BASIS_TO_ID = {n: i for i, n in enumerate(BASIS)}
NUM_BASIS = len(BASIS)
NULL_ID = NUM_BASIS  # "no emotion given" (used for classifier-free guidance)

EKMAN_OF = {
    "shanta": "neutral",
    "hasya": "happy",
    "karuna": "sad",
    "raudra": "angry",
    "bhayanaka": "fear",
    "bibhatsa": "disgust",
    "adbhuta": "surprise",
}


@dataclass(frozen=True)
class Rasa:
    name: str
    devanagari: str
    bhava: str          # sthayi-bhava (the underlying emotional state)
    gloss: str
    valence: float      # rough coordinates, -1..1, for analysis/plots only
    arousal: float
    mixture: dict       # weights over BASIS; one-hot for trained rasas


NAVARASA = {
    "shringara": Rasa("shringara", "शृंगार", "rati", "love / tenderness", 0.8, 0.3,
                      {"hasya": 0.55, "shanta": 0.45}),            # zero-shot mixture
    "hasya":     Rasa("hasya", "हास्य", "hasa", "joy / laughter", 0.8, 0.6, {"hasya": 1.0}),
    "karuna":    Rasa("karuna", "करुण", "shoka", "sorrow / compassion", -0.7, -0.4, {"karuna": 1.0}),
    "raudra":    Rasa("raudra", "रौद्र", "krodha", "anger / fury", -0.6, 0.9, {"raudra": 1.0}),
    "veera":     Rasa("veera", "वीर", "utsaha", "heroism / resolve", 0.5, 0.8,
                      {"raudra": 0.45, "hasya": 0.55}),            # zero-shot mixture
    "bhayanaka": Rasa("bhayanaka", "भयानक", "bhaya", "fear / terror", -0.7, 0.7, {"bhayanaka": 1.0}),
    "bibhatsa":  Rasa("bibhatsa", "बीभत्स", "jugupsa", "disgust", -0.8, 0.3, {"bibhatsa": 1.0}),
    "adbhuta":   Rasa("adbhuta", "अद्भुत", "vismaya", "wonder / surprise", 0.4, 0.8, {"adbhuta": 1.0}),
    "shanta":    Rasa("shanta", "शान्त", "shama", "peace / calm (neutral)", 0.1, -0.6, {"shanta": 1.0}),
}
TRAINED_RASAS = [r for r in NAVARASA if len(NAVARASA[r].mixture) == 1]
ZERO_SHOT_RASAS = [r for r in NAVARASA if len(NAVARASA[r].mixture) > 1]


def rasa_weights(rasa: str | dict) -> torch.Tensor:
    """Return a (NUM_BASIS,) weight vector. Accepts a rasa name or a custom mixture dict."""
    mix = NAVARASA[rasa].mixture if isinstance(rasa, str) else rasa
    w = torch.zeros(NUM_BASIS)
    for k, v in mix.items():
        w[BASIS_TO_ID[k]] = float(v)
    s = w.sum()
    if s <= 0:
        raise ValueError(f"empty mixture: {rasa}")
    return w / s


# ---- mapping AI4Bharat Rasa `style` strings -> basis ids ---------------------
# The dataset card does not publish the exact style strings, so we match on
# keywords and print anything unmatched (see data.inspect_styles).
_STYLE_PATTERNS = [
    ("hasya", r"happ|joy"),
    ("karuna", r"sad"),
    ("raudra", r"^ang"),          # ^ so that "UMANG" (app commands) is not read as anger
    ("bhayanaka", r"fear"),
    ("bibhatsa", r"disg"),
    ("adbhuta", r"surp"),
    ("shanta", r"wiki|neutral|^indic|book|news|read"),  # INDIC = neutral IndicTTS-text readings
]
# Conversational / command styles are real speech but not "an emotion".
# Default: drop them. Set include_other_styles=True to keep them as shanta.
_OTHER_STYLES = r"conv|alexa|umang|digi|bb|bigbasket|command"


def style_to_basis(style: str, include_other_styles: bool = False) -> int | None:
    s = (style or "").strip().lower()
    for name, pat in _STYLE_PATTERNS:
        if re.search(pat, s):
            return BASIS_TO_ID[name]
    if include_other_styles and re.search(_OTHER_STYLES, s):
        return BASIS_TO_ID["shanta"]
    return None


# Sooktam-2 CLS tokenizer language names (as used in the model card: "hindi").
SOOKTAM_LANG = {
    "Hindi": "hindi", "Marathi": "marathi", "Gujarati": "gujarati", "Bengali": "bengali",
    "Punjabi": "punjabi", "Odia": "odia", "Urdu": "urdu", "Tamil": "tamil",
    "Telugu": "telugu", "Kannada": "kannada", "Malayalam": "malayalam",
}
LANG_FAMILY = {
    "hindi": "indo-aryan", "marathi": "indo-aryan", "gujarati": "indo-aryan", "bengali": "indo-aryan",
    "punjabi": "indo-aryan", "odia": "indo-aryan", "urdu": "indo-aryan",
    "tamil": "dravidian", "telugu": "dravidian", "kannada": "dravidian", "malayalam": "dravidian",
    "english": "english",  # Sooktam-2 speaks Indian English; Rasa has no English, so this id is untrained
}
LANGS = sorted(LANG_FAMILY)
LANG_TO_ID = {l: i for i, l in enumerate(LANGS)}
UNK_LANG_ID = len(LANGS)
