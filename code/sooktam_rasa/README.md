# Sooktam-2 × Navarasa: emotional, duration-aware TTS for Indian languages

UGP-2, IIT Kanpur. This continues the Sooktam-2 duration-bug work. It adds:

1. **Rasa / emotion control** inside Sooktam-2's DiT. The model is trained on AI4Bharat *Rasa*; the backbone stays frozen, and only the adapter (about 3M parameters) is trained.
2. **A learned cross-attention duration predictor.** It replaces the `ref_len / text_len` heuristic and targets the Indo-Aryan vs Dravidian gap from Sec. 10 of the bug report.
3. **An evaluation pipeline:** emotion accuracy, CER, speaker similarity, duration error, and a blind sheet for a listening test.

## How it works

```
                       time t ──► TimestepEmbed ─┐
rasa ─► weights over 7 basis emotions ─► EmotionConditioner (zero-init) ─(+)─► t' ─► AdaLN in all 22 DiT blocks
                                                             LoRA r=16 on q/k/v/out of every block (trainable)
ref mel + noisy mel + CLS text ─► InputEmbedding ─► 22 × DiTBlock (frozen) ─► velocity ─► Euler ODE ─► Vocos
```

* **Where the emotion goes in.** Sooktam-2 = F5-TTS v1 Base (dim 1024, depth 22). Every block is modulated by AdaLN driven by the timestep vector. Class-conditional DiT adds the class embedding to that same vector, and so does this code. The conditioner's last layer starts at zero, so at step 0 the output is **bit-identical** to the released model (this is tested).
* **The prompt-leak problem.** F5/Voicebox copies prosody from the reference clip, so a neutral reference tends to produce neutral speech whatever the label says. To counter this, 70% of training batches are built as `[neutral clip, same speaker] + [emotional clip]`, with the loss only on the emotional part. That is exactly the situation at inference.
* **Emotion guidance.** Three ODE branches per step:
  `v = v_c + cfg_text·(v_c − v_uncond) + cfg_emo·(v_c − v_no-emotion)`.
  `cfg_emo` is an emotion-strength knob, and `intensity` interpolates between the null embedding and the emotion embedding.
* **Navarasa.** Rasa records the 6 Ekman emotions plus neutral. Those cover 7 rasas: hasya, karuna, raudra, bhayanaka, bibhatsa, adbhuta and shanta. **Shringara and veera have no recordings anywhere.** They are defined as mixtures (`shringara = .55 hasya + .45 shanta`, `veera = .55 hasya + .45 raudra`), so they are zero-shot. Report them that way and judge them by listening tests.
* **Duration.** `frames = τ · Σ u_i · exp(c_i)`:
  * `u_i` is a learned per-token duration weight.
  * `τ = ref_frames / Σ_ref u_j` is the speaker's tempo, measured on the reference.
  * `c_i` is a zero-initialised residual from cross-attention (target tokens attend over the reference mel), plus the emotion.

  At init this equals the token-count heuristic, so training can only move it towards better.

## Colab or local?

**Colab works.** A free T4 is enough for data prep, the duration model, and a slow emotion run. Colab Pro (L4/A100) makes the emotion run 3–6× faster. You don't need a local GPU, but if you have one with 12 GB or more (or access to IIT-K's HPC), that's better for long runs because Colab sessions disconnect. Everything resumes from Google Drive either way.

Rough sizes:

| Item | Size |
|---|---|
| Model weights | 1.3 GB (`sooktam.safetensors`) |
| Rasa per language | 8–37 GB on the Hub (only chosen rows' row-groups are downloaded) |
| Cached mels | about 70 MB per hour of speech |
| Adapter checkpoint | a few MB |

## Quick start (Colab)
1. Accept the terms of `ai4bharat/Rasa` on Hugging Face. Add your read token as the Colab secret `HF_TOKEN`.
2. Upload the `code/sooktam_rasa/` project folder to `MyDrive/sooktam_rasa/`. The inner `sooktam_rasa/` Python package and `configs/` belong directly inside that folder.
3. Open `Sooktam_Navarasa_Colab.ipynb` and run it top to bottom. In order, it:
   * installs everything,
   * runs the baseline check,
   * runs `--inspect` (check the style→rasa table!),
   * builds the data cache,
   * trains the duration model, then the emotion adapter,
   * plays the navarasa sweep,
   * runs the evaluation.

Local (Linux, Python 3.10, CUDA GPU):
```bash
git clone https://huggingface.co/bharatgenai/sooktam2 && cd sooktam2 && sh setup-cls.sh && cd ..
pip install grapheme librosa soundfile pandas pyarrow
export HF_TOKEN=hf_...
python -m sooktam_rasa.prepare --languages Hindi Tamil --data_dir data --inspect
python -m sooktam_rasa.prepare --languages Hindi Tamil --data_dir data --device cuda
python -m sooktam_rasa.train_duration --set sooktam_dir='"sooktam2"' data_dir='"data"' out_dir='"ckpt/duration"'
python -m sooktam_rasa.train_emotion --config configs/emotion_a100.json --set sooktam_dir='"sooktam2"' data_dir='"data"' out_dir='"ckpt/emotion"'
```

## Files
| File | Purpose |
|---|---|
| `sooktam_rasa/rasa.py` | Navarasa definitions, the Rasa `style` → basis mapping, language families |
| `sooktam_rasa/emotion_dit.py` | Emotion conditioner, LoRA, emotion-aware DiT forward, flow-matching loss, 3-branch guided sampler |
| `sooktam_rasa/data.py` | Metadata scan, row selection, mel/CLS extraction, paired training dataset, emotion-balanced frame-budget sampler |
| `sooktam_rasa/duration.py` | Cross-attention duration predictor + byte/grapheme baselines |
| `sooktam_rasa/train_emotion.py` | Adapter training: AMP, grad checkpointing, resume, val, audio samples |
| `sooktam_rasa/train_duration.py` | Duration training + per-family error table vs the byte and grapheme formulas |
| `sooktam_rasa/infer.py` | `EmotionalSooktam.load(...).synthesize(..., rasa="karuna", intensity=1.0)` |
| `sooktam_rasa/evaluate.py` | `generate` (Sooktam env) and `score` (any env) steps, listening-test sheet |
| `tests/test_cpu.py` | End-to-end CPU test with a tiny random DiT standing in for Sooktam-2 |

## Experiments to report
1. **Duration (cheap; do first).** Mean abs % error, broken down as bytes vs grapheme vs learned, per language family. Also re-run the 10-language English-reference table from the bug report with `duration="learned"`.
   * Rasa speakers are monolingual, so the training pairs are same-language.
   * Cross-script generalisation therefore has to be *tested* on your bug-report setup, or on bilingual speakers such as those in IndicTTS.
2. **Emotion.** Emotion-judge accuracy for three conditions: base Sooktam-2, adapter without emotion guidance, and adapter with guidance. Always quote it next to the judge's ceiling on real speech. Also report:
   * CER and speaker similarity, to show intelligibility and voice are kept;
   * a confusion matrix (expect fear and surprise to be the hardest; the Rasa paper reports the same).
3. **Ablations.** `p_pair` (0 vs 0.7), `lora_r` (0 = conditioner only vs 16), and `cfg_emo` ∈ {0, 0.5, 1, 2}.
4. **Listening test.** Use `listening_test.csv` with native speakers: naturalness 1–5, plus a forced choice of the perceived rasa.

## Things to check on the first run
These couldn't be verified from outside the gated dataset or the Colab runtime.
- **Style strings in Rasa.** The card doesn't list them. `--inspect` prints the exact values and what they map to. If any emotion shows `SKIP`, add its keyword to `_STYLE_PATTERNS` in `rasa.py`.
- **CLS language names.** The code uses lowercase English names (`"hindi"`, `"tamil"`, …), as in the model card. The baseline cell and the `OOV token rate` printout will show a problem immediately.
- **Checkpoint.** `load_sooktam` prefers `sooktam.safetensors`. If the baseline cell sounds wrong, download `model_1250000.pt` and pass `ckpt="model_1250000.pt"` (the training script takes the `ckpt` setting).
- **Model size.** The checkpoint is F5TTS_v1_Base (1024×22, about 336M params, 1.34 GB fp32). The bug report says 150M; worth correcting.

## Licences
Rasa is CC-BY-4.0. Sooktam-2 is under the BharatGen non-commercial licence, so the adapters inherit that restriction.
