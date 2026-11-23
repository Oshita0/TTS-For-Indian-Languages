---
language:
- en
- hi
- mr
- gu
- ta
- te
- kn
- bn
- ml
- or
- ur
- pa
pipeline_tag: text-to-speech
library_name: transformers
tags:
- text-to-speech
- tts
- multilingual
- indic
- f5-tts
- sooktam2
---

<p align="center">
  <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67b462a1f4f414c2b3e2bc2f/EnVeNWEIeZ6yF6ueZ7E3Y.jpeg" width="140" alt="BharatGen Logo"/>
</p>

<h1 align="center">Sooktam-2 🇮🇳</h1>
<p align="center"><em>"विविधता में ही भारत की शक्ति है, और हर भाषा उस शक्ति की आवाज़ है।"</em></p>
<p align="center"><b>Sovereign AI · Built in Bharat · For Bharat</b></p>

---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YvgkOL7mM7vcOE8IHOhHD9PprYUh5bvb)

## The Story

India is not one voice - it is a symphony. Tamil, Bengali, Urdu, Hindi, Kannada - each a living civilisation, spoken daily by hundreds of millions. Yet for too long, AI treated them as afterthoughts. Models built elsewhere, for someone else, leaving Bharat to make do with approximations of its own languages.

**BharatGen was built to end that.** We are India's sovereign AI initiative - weaving the country's languages, cultures, and voices into technology that is truly Indian. Not adapted. Not translated. *Built from the ground up, for Bharat.*

**Sooktam-2** is our answer to India's need for a voice. A Text-to-Speech model that speaks 12 languages (11 indian languages + 1 indian english) with the phonetic precision, prosody, and cultural soul they deserve - so that every Indian, in every state, can hear AI speak *their* language, in *their* accent, and feel at home.

This is **GenAI for Bharat, by Bharat.**

---

## What is Sooktam-2?

Sooktam-2 is a sovereign multilingual Text-to-Speech model built by BharatGen. It synthesises natural, expressive speech across India's major languages using reference-guided voice conditioning - preserving the speaker's voice, accent, and cultural cadence.

**Represented Languages - 12**

`Hindi` · `Marathi` · `Gujarati` · `Tamil` · `Telugu` · `Kannada` · `Bengali` · `Malayalam` · `Odia` · `Urdu` · `Punjabi` · `Indian English`

**Key Capabilities**
- Reference-guided voice cloning
- Multilingual Indic speech synthesis
- Natural prosody and expressive delivery
- Language-aware CLS tokenization for accurate Indic phonetics
- Production-quality audio output, deployment-ready at scale

---

## Quickstart

- Python version = 3.10

```bash
git clone https://huggingface.co/bharatgenai/sooktam2
cd sooktam2
sh setup-cls.sh
```

---

## Python Inference

```python
import os
from transformers import AutoModel

# --- Model ID ---
MODEL_ID = "bharatgenai/sooktam2"

# --- Your reference audio and target text ---
REF_AUDIO = "reference.wav"   # A short, clean voice clip (3–10 sec)
REF_TEXT  = "सर, मैं तब से यह कह रहा हूँ कि मैंने अपना टिकट कैंसल कर दिया है, लेकिन अब तक मेरे पैसे वापस नहीं आए हैं। आप इस मामले को देखेंगे भी या नहीं?"
GEN_TEXT  = "यह एक टेस्ट वाक्य है जिसे आवाज़ में बदलना है।"

# --- Output ---
OUT_DIR = "outputs"
OUT_WAV = os.path.join(OUT_DIR, "sooktam_cls.wav")

# --- Load model (auto-downloads checkpoint + vocab from HuggingFace) ---
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

os.makedirs(OUT_DIR, exist_ok=True)

# CLS tokenization is handled inside utils_infer via cls_tokenizer_v2
wav, sr, _ = model.infer(
    ref_file=REF_AUDIO,
    ref_text=REF_TEXT,
    gen_text=GEN_TEXT,
    tokenizer="cls",
    cls_language="hindi",
    file_wave=OUT_WAV,
)

print("Saved:", OUT_WAV, "sample_rate:", sr, "samples:", len(wav))
```

> The model and vocab download automatically from HuggingFace on first run. No manual checkpoint hunting required.

---

## Hugging Face AutoModel

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "bharatgenai/sooktam2",
    trust_remote_code=True,
)

wav, sr, _ = model.infer(
    ref_file="ref.wav",
    ref_text="Your reference transcript.",
    gen_text="Text you want to synthesise.",
    tokenizer="cls",
    cls_language="hindi",
)
```

---

## License
This post-trained checkpoint is released under the BharatGen non-commercial license.

Please refer to the [LICENSE](./LICENSE) file for detailed terms and conditions.

---

## Contributors
- Yash
- Supreet
- Isha
- Vansh
- Pranav

For any questions or feedback, please contact: contact@bharatgen.com

---

## BharatGen - Sovereign AI for a Sovereign Nation

BharatGen is India's initiative to build AI that is Indian in its roots, inclusive in its reach, and sovereign in its design. We believe that a nation of India's civilisational depth - of Sanskrit and Tamil, of Tagore and Kabir, of a billion daily conversations - should not have to borrow its voice from elsewhere.

India's languages are not a niche. They are the world's richest linguistic heritage. And now, they have a model built for them.

We are just getting started.

---

<p align="center">
  <a href="https://bharatgen.com">bharatgen.com</a> · <a href="https://huggingface.co/bharatgenai/sooktam2">HuggingFace ↗</a>
  <br/><br/>
  <b>जय हिन्द · जय भारत 🇮🇳</b>
</p>
