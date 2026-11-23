# Sooktam-2 × Navarasa — running results log (source material for the final report)

UGP-2, IIT Kanpur, Prof. Arnab Bhattacharya. Continuation of the Sooktam-2 duration-bug report (byte → grapheme fix).
Every number here was produced by the code in this repo on Google Colab Pro. Times are IST.

---

## 0. Setup
- **Base model:** `bharatgenai/sooktam2` = F5-TTS v1 Base architecture (DiT, dim 1024, depth 22, 16 heads, text_dim 512, ConvNeXt text encoder),
  mel 100 bins / 24 kHz / hop 256, Vocos vocoder, CLS tokenizer. **~336-340M parameters** (the earlier report said 150M; that is wrong).
  Checkpoint used: `sooktam.safetensors` (1.34 GB).
- **Environment:** Colab Pro; Python 3.10 venv via `uv` (Sooktam's `setup-cls.sh` pins torch 2.6 / TF 2.15, Colab ships Python 3.12/3.13).
  Hardware: T4 (data prep, duration model), L4 24 GB (emotion training; ~3.6 s/step at 16k frames/batch, 4.3 GB peak memory, bf16).
- **Baseline check (06-Nov):** plain Sooktam-2, Hindi ref → Hindi text, `bytes` and `grapheme` duration both correct and near-identical
  (expected: same script, so byte and grapheme ratios agree).

## 1. Data: AI4Bharat Rasa (Interspeech 2024, CC-BY-4.0)
- Gated HF dataset, 22 language configs, 48 kHz mono, columns: filename, text, language, gender, style, duration, wav_path, audio.
  ~1.28M rows / 388 GB in total. One male + one female speaker per language.
- Languages used: **Hindi** (Indo-Aryan) and **Tamil** (Dravidian) — to measure the family gap from the bug report.
- Style labels (from `--inspect`, metadata-only scan):

| Style | Hindi train h | Tamil train h | Mapped to |
|---|---|---|---|
| WIKI | 14.22 | 16.39 | shanta |
| NEWS | 4.02 | 4.03 | shanta |
| BOOK | 3.87 | 3.66 | shanta |
| INDIC (IndicTTS texts) | 1.90 | 1.60 | shanta |
| ANGER | 1.80 | 2.83 | raudra |
| SURPRISE | 1.79 | 2.50 | adbhuta |
| SAD | 1.77 | 2.50 | karuna |
| FEAR | 1.75 | 2.19 | bhayanaka |
| HAPPY | 1.74 | 2.84 | hasya |
| DISGUST | 1.66 | 1.89 | bibhatsa |
| CONV | 6.84 | 6.18 | skipped |
| PROPER NOUN | 2.66 | 1.33 | skipped |
| ALEXA / UMANG / BB / DIGI (commands) | 1.78 | 2.52 | skipped |

  Totals: Hindi train 25,713 rows / 45.8 h, test 2,858 / 5.0 h; Tamil train 33,005 / 50.4 h, test 3,656 / 5.7 h.
- **Bug found and fixed during inspection:** regex `ang` matched "UM**ANG**" (command style) → would have put 0.45-0.86 h of app commands
  into *raudra*; and INDIC was skipped although it is neutral read speech. Fixed to `^ang` and `^indic`.
- **Selection:** all emotional speech + 2 h neutral per speaker (train), 0.3 h neutral per speaker (test); 1-10 s clips; edge silence
  trimmed (top_db 45) + 50 ms pad; resampled to 24 kHz; Sooktam's own log-mel; CLS tokens. Only parquet row-groups containing selected
  rows are downloaded.
- **Result:** 19,971 utterances cached (≈2.6 GB fp16 mels). OOV rate of CLS tokens vs Sooktam vocab: **0.0000%**.
  Selected Tamil train hours (Female / Male): shanta 2.0/2.0, hasya 1.91/0.83, karuna 1.59/0.80, raudra 2.09/0.72,
  bhayanaka 1.36/0.78, bibhatsa 1.26/0.57, adbhuta 1.62/0.79.
- Extraction throughput ≈75 utt/min writing to Drive (bottleneck: small-file writes to Drive FUSE); copying cache to local disk for
  training: 2.6 GB in 71 s.
- **Key dataset property (07-Nov):** 18,786 unique sentences, **0 sentences recorded in more than one emotion** → Rasa's emotional data is
  non-parallel; each emotion has its own emotion-bearing sentences. (Central to the v1 negative result, §3.)

## 2. Learned cross-attention duration predictor
- Model: frames = τ · Σ u_i · exp(c_i); u_i = softplus(MLP(TextEnc(CLS token, language))), τ = ref_frames / Σ_ref u_j (speaker tempo),
  c_i = zero-init residual from cross-attention (target tokens → reference mel, 4× downsampled) + emotion embedding.
  Initialised to equal the token-count heuristic. ~4M params. 15k steps, batch 32, AdamW 3e-4 OneCycle, L1 on log-frames,
  τ-regulariser 0.01. Trained on T4 in ~40 min.
- Evaluation: held-out test split, same-speaker (ref, target) pairs, 70% neutral references; mean absolute % error of generated duration.

| Group | bytes (original Sooktam) | grapheme (bug-report fix) | CLS-token count | **learned** |
|---|---|---|---|---|
| All | 18.2% | 16.6% | 17.2% | **7.7%** |
| Indo-Aryan (Hindi) | 13.0% | 11.9% | 11.8% | **6.5%** |
| Dravidian (Tamil) | 20.3% | 18.5% | 19.4% | **8.2%** |

- Training curve (ALL): 9.1% at first checkpoint → plateau 7.7-8.2% from ~step 6k. Best 7.73%.
- Findings: 54% relative error reduction vs grapheme fix; Dravidian–Indo-Aryan gap 6.6 pts → 1.7 pts.
- Caveats: same speakers in train and test (Rasa has 2 speakers/language); same-script pairs only. Cross-script (English ref → Indic)
  still to be evaluated on the bug-report setup.

## 3. Emotion adapter v1 (negative result)
- Method: frozen Sooktam-2; EmotionConditioner (7 basis emotions + null, 256-d table → MLP → added to timestep vector → AdaLN of all
  22 blocks, zero-init) + LoRA r=16 on q/k/v/out of every block. 4.20M trainable / 340M. 70% of batches: [neutral prompt, same speaker]
  + [target], loss on target only; 30% F5-style random infill. CFG drops 0.3 audio / 0.2 all; 15% emotion-null. 3-branch guidance at
  inference (cfg_text 2, cfg_emo 1).
- Run: 8,000 steps, 16k frames/batch, bf16, L4, ~3.6 s/step (≈8 h incl. one disconnect + resume from step 1000 and from 1500).
  train 17,812 / val 2,159. Train FM loss ≈0.72 → ≈0.62. Val loss (original, *noisy* implementation — different clips each check)
  0.587-0.662.
- **Bug found:** validation sampled different batches at every check, so val numbers were not comparable and `best.pt` selection was
  noise-driven. Fixed (fixed batches + fixed noise); added 1k-step snapshots.
- **Listening (07-Nov):** step-8000 samples: all nine rasas sound the same.
- **Label-ablation diagnostic** (630 held-out clips, identical clips/prompts/noise across conditions):

| | base (adapter off) | null label | correct label | wrong label |
|---|---|---|---|---|
| best.pt (step 6000) | 0.7938 | 0.6344 | 0.6343 | 0.6366 |
| latest.pt (step 8000) | 0.7938 | 0.6345 | 0.6335 | 0.6358 |

  - Speaker/recording adaptation (base → null): **+20.1%**
  - Emotion-label gain (null → correct): **+0.02% / +0.17%**; label helps on 29% / 45% of clips
  - Right vs wrong label: +0.36%; right label wins on 56% / 60% of clips
  - Per-rasa: largest label effect on shanta (−0.9% at 8k); others within ±0.3%.
- **Interpretation:** almost all improvement is domain adaptation (LoRA fits Rasa speakers/recording conditions). The model ignores the
  rasa label because, with zero parallel sentences, the emotion is predictable from the words; the label is redundant during training.
  Also the training-time samples used emotional test sentences, so text drove all nine renditions the same way.

## 4. Emotion adapter v2
Changes to make the label necessary:
1. **Text dropout with label kept** (30% of steps): words hidden, neutral prompt + label visible.
2. **Emotion critic loss**: mel emotion classifier trained on real Rasa speech (audio only), frozen, scores the one-step estimate
   x̂1 = φ + (1−t)·v on the generated span, CE weighted by t, weight 0.15.
3. **p_pair = 1.0**: prompt always a different, neutral utterance (removes leakage through the prompt).
4. **Input-level injection**: emotion vector also added to every input frame (zero-init), lr_emotion 1e-3.
5. Validation now prints **LABEL GAIN** (null vs correct, same clips/noise) at every check; samples speak a *neutral* sentence.

### 4a. Emotion critic (mel classifier, real Rasa speech) — 07-Nov
- 5-layer Conv1d (GroupNorm, GELU, 2 stride-2 stages) + masked mean/std pooling; 1.45M params. 4,000 steps, batch 32, emotion-balanced
  sampling, random 1.5-6 s crops, noise/blur/frequency-mask augmentation, label smoothing 0.1. Train 17,812 / test 2,159. T4/L4, ~5 min.
- Test accuracy by step: 0.501 (500) → 0.608 → 0.690 → 0.759 → 0.820 → 0.878 → 0.896 → **0.906 (4000)**; chance 0.143.
- Confusion (rows true, cols predicted; shanta, hasya, karuna, raudra, bhayanaka, bibhatsa, adbhuta):

| true \ pred | shan | hasy | karu | raud | bhay | bibh | adbh |
|---|---|---|---|---|---|---|---|
| shanta | 703 | 3 | 3 | 9 | 0 | 7 | 1 |
| hasya | 2 | 231 | 5 | 5 | 0 | 0 | 20 |
| karuna | 2 | 5 | 199 | 1 | 12 | 2 | 1 |
| raudra | 2 | 4 | 1 | 266 | 0 | 12 | 7 |
| bhayanaka | 1 | 1 | 17 | 1 | 182 | 13 | 5 |
| bibhatsa | 3 | 2 | 4 | 3 | 4 | 171 | 4 |
| adbhuta | 1 | 20 | 0 | 5 | 0 | 15 | 204 |

- Main confusions: hasya↔adbhuta (high-arousal positive), karuna↔bhayanaka. Caveat: same speakers in train/test and each emotion was
  likely recorded in its own session, so some accuracy may come from session/channel cues → final evaluation uses an independent judge.

### 4b. v2 training — 07-Nov, A100
- 5.51M trainable / 342M (input-level injection adds 1.31M). 16k frames/batch, bf16, **0.77 s/step on A100** (vs 3.6 s/step on L4
  for v1), 4.5 GB peak. Data restore on a fresh runtime: sequential rsync from Drive ran at ~390 kB/s (22,135 small files);
  32-way parallel copy finished in minutes → packed into one tar on Drive for later sessions.
- Logged `critic-acc(t>.5)` was 0.85-0.95 from step 25 onward. **Design flaw identified:** for t>0.5 the one-step estimate
  x̂1 is dominated by the real (emotional) target contained in the noisy input, so the critic mostly scores reconstruction, not
  label-driven generation → this metric (and much of the critic loss) is uninformative. Text-dropout is the effective lever.
- Fixed-batch validation, correct label vs null label (on emotional-sentence clips):

| step | val (label) | val (null) | label gain |
|---|---|---|---|
| 500 | 0.6733 | 0.6631 | −1.53% |
| 1000 | 0.6441 | 0.6439 | −0.02% |
| 1500 | 0.6502 | 0.6387 | −1.80% |
| 2000 | 0.6364 | 0.6329 | −0.56% |
| 2500 | 0.6358 | 0.6310 | −0.75% |
| 3000 | 0.6350 | 0.6277 | −1.16% |
| 3500 | 0.6334 | 0.6264 | −1.12% |
| 4000 | 0.6313 | 0.6277 | −0.57% |
| 4500 | 0.6301 | 0.6255 | −0.73% |
| 5000 | 0.6298 | 0.6243 | −0.89% |
| 5500 | 0.6258 | 0.6193 | −1.05% |
| 6000 | **0.6246** | 0.6193 | −0.86% |

- Run completed 6,000 steps (~1.4 h on A100). Val loss improved monotonically (best = final step). Train FM loss ≈0.72 → ≈0.64.
  Label gain on emotional-sentence clips stays slightly negative throughout (−0.6 to −1.8%).
- Interpretation: unlike v1 (≈0), the label now changes predictions, but on clips whose words already carry the emotion it does not
  help (slightly hurts). This val metric is the wrong test for v2; the decisive test is controllability on neutral sentences (4c).
- **Listening, step 2000 samples (neutral sentence, neutral reference, cfg_emo 1):** audible differences between rasas — raudra
  louder, karuna distinguishable from shanta. First audible label control (v1 step-8000 samples were indistinguishable).
  At step 6000 (user, 07-Nov 23:23): "the voice is different for different rasa".

### 4c. Controllability on NEUTRAL sentences (v2, latest.pt, step 6000) — 07-Nov
- Protocol: test-split neutral reference clips (2 per language, one per speaker) × 3 neutral test sentences per language
  = 12 pairs (6 Hindi, 6 Tamil); each synthesised for all 7 trained rasas at cfg_emo ∈ {0, 1, 3}, intensity 1, learned duration,
  seed 0; plus base Sooktam-2 (adapter off). Judge: the mel emotion critic (90.6% on real test speech).
- Base Sooktam-2 on these neutral pairs: critic says **100% shanta**.

| condition | accuracy (chance 0.143) | emotional rasas only | mean p(target) | Hindi | Tamil |
|---|---|---|---|---|---|
| cfg_emo 0 | **0.702** | 0.653 | 0.621 | 0.786 | 0.619 |
| cfg_emo 1 | **0.762** | 0.722 | 0.676 | 0.833 | 0.690 |
| cfg_emo 3 | **0.810** | 0.778 | 0.709 | 0.881 | 0.738 |

- Confusion at cfg_emo 3 (rows target, cols predicted; shan hasy karu raud bhay bibh adbh), 12 per row:
  shanta 12/0/0/0/0/0/0; hasya 8/4/0/0/0/0/0; karuna 6/0/6/0/0/0/0; raudra 0/0/0/12/0/0/0; bhayanaka 0/0/0/0/12/0/0;
  bibhatsa 0/0/0/0/0/12/0; adbhuta 0/0/2/0/0/0/10.
  (cfg 0: hasya 4/12, karuna 3/12, raudra 12, bhayanaka 9, bibhatsa 11, adbhuta 8; cfg 1: hasya 5, karuna 4, raudra 12,
  bhayanaka 10, bibhatsa 12, adbhuta 9.)
- Findings: label control works (0.70-0.81 vs 0.14 chance; v1 produced indistinguishable outputs); emotion guidance acts as a
  strength knob (monotonic 0.70 → 0.76 → 0.81); raudra/bibhatsa/bhayanaka strongest; hasya and karuna weakest and fall back to
  shanta (too mild rather than wrong); Tamil < Hindi (0.74 vs 0.88 at cfg 3).
- **Caveat:** judge = the same critic used as a training loss → possible "teaching to the test". Pending checks: (i) same test on
  the v1 adapter, (ii) independent zero-shot emotion2vec judge + Whisper CER + ECAPA speaker similarity (`score_control.py`),
  (iii) human listening. Small n (12 per cell).

### 4d. Ablation: same controllability test on the v1 adapter (latest.pt, step 8000) — 08-Nov
- Base Sooktam-2: critic 100% shanta (same pairs).

| condition | v1 all | v1 emotional only | v1 mean p(target) | Hindi | Tamil | **v2 all** | **v2 emotional only** |
|---|---|---|---|---|---|---|---|
| cfg_emo 0 | 0.250 | 0.139 | 0.233 | 0.262 | 0.238 | 0.702 | 0.653 |
| cfg_emo 1 | 0.286 | 0.181 | 0.260 | 0.310 | 0.262 | 0.762 | 0.722 |
| cfg_emo 3 | 0.321 | 0.208 | 0.318 | 0.333 | 0.310 | 0.810 | 0.778 |

- v1 confusion at cfg 3 (shan hasy karu raud bhay bibh adbh): shanta 12/0/0/0/0/0/0; hasya 8/2/0/2/0/0/0; karuna 8/0/2/0/2/0/0;
  raudra 3/1/0/8/0/0/0; bhayanaka 7/0/0/0/2/2/1; bibhatsa 8/0/0/1/1/1/1; adbhuta 5/3/0/4/0/0/0.
- Findings: v1 collapses to shanta (emotional-only accuracy at chance without guidance); only raudra partly leaks through
  (6-8/12), consistent with anger being the most acoustically distinct emotion. v2 is ~4-5× better on emotional rasas.
  v1 was never trained with the critic yet the critic still reports mostly shanta for it → the critic does not call
  arbitrary speech emotional (partial answer to the teaching-to-the-test concern; independent judge still pending).

### 4e. Independent judge: emotion2vec+ large, zero-shot (v2 control set) — 08-Nov
- emotion2vec's own labels mapped to rasas (neutral→shanta, happy→hasya, sad→karuna, angry→raudra, fearful→bhayanaka,
  disgusted→bibhatsa, surprised→adbhuta; "other"/"unk" ignored). Same 84 v2 outputs per cfg; plus 10 real test recordings per
  language × rasa (140) as the judge's ceiling.

| condition | accuracy (all 7) | emotional only | notes |
|---|---|---|---|
| real Rasa recordings | 0.407 (57/140) | 0.350 (42/120) | judge ceiling on genuine Hindi/Tamil emotional speech |
| base Sooktam-2 | – | – | predicted 67% shanta, 33% hasya |
| v2 cfg_emo 0 | 0.333 (28/84) | 0.264 (19/72) | |
| v2 cfg_emo 1 | 0.440 (37/84) | 0.375 (27/72) | |
| v2 cfg_emo 3 | **0.583 (49/84)** | **0.542 (39/72)** | CER 0.101, ECAPA speaker sim 0.568 |

- Real-recording confusion (20 per row; shan hasy karu raud bhay bibh adbh): shanta 15/4/0/0/0/0/1; hasya 0/19/0/0/0/0/1;
  karuna 7/6/1/0/1/3/2; raudra 1/4/0/9/0/2/4; bhayanaka 1/8/2/0/5/0/4; bibhatsa 1/10/0/1/0/1/7; adbhuta 0/13/0/0/0/0/7.
  → emotion2vec is strongly biased to "happy" on Indian speech and almost never recognises sad (1/20) or disgust (1/20).
- v2 cfg 3 confusion (12 per row): shanta 10/2/0/0/0/0/0; hasya 0/10/0/2/0/0/0; karuna 3/5/4/0/0/0/0; raudra 0/1/0/11/0/0/0;
  bhayanaka 2/1/2/0/3/0/4; bibhatsa 2/1/0/3/0/4/2; adbhuta 1/4/0/0/0/0/7.
  (cfg 0 diag: 9,8,1,7,1,0,2; cfg 1 diag: 10,7,2,10,3,1,4.)
- Findings: an independent model also hears the requested rasa, monotonic in cfg_emo (0.33 → 0.44 → 0.58) → not critic gaming.
  v2 at cfg 3 exceeds the judge's score on real recordings (0.58 vs 0.41): plausibly guidance produces more prototypical /
  exaggerated emotion than the actors' natural readings — needs a human listening test. Both judges agree on raudra and adbhuta;
  they disagree on hasya (critic 4/12, e2v 10/12), but e2v's hasya bias makes that cell unreliable.
- Full summary (Whisper large-v3 CER via faster-whisper; ECAPA speaker similarity to the neutral reference clip):

| condition | e2v acc | pred = shanta | CER | speaker sim | n |
|---|---|---|---|---|---|
| real recordings | 0.407 | 0.179 | 0.067 | – | 140 |
| base Sooktam-2 | – | 0.667 | **0.101** | **0.781** | 12 |
| v2 cfg_emo 0 | 0.333 | 0.393 | 0.095 | 0.685 | 84 |
| v2 cfg_emo 1 | 0.440 | 0.333 | 0.098 | 0.653 | 84 |
| v2 cfg_emo 3 | 0.583 | 0.214 | 0.101 | 0.568 | 84 |

- Findings: (1) **no intelligibility cost** — CER 9.5-10.1% at every strength vs 10.1% for base Sooktam-2 (real recordings 6.7%
  are different, emotional sentences, not directly comparable). (2) **Emotion strength trades off against speaker similarity**
  (0.78 base → 0.69 → 0.65 → 0.57); partly inherent (emotional voice quality differs from the neutral reference and ECAPA is
  sensitive to it), partly a real identity loss at high guidance. Recommended default cfg_emo = 1.
- Next check: ECAPA similarity between REAL neutral and REAL emotional recordings of the same speaker, to calibrate how much of the
  drop is natural.

## 6. Status vs. the original resume bullets (08-Nov)
| bullet | status |
|---|---|
| Low-latency multilingual TTS with CNFs + emotional conditioning | partial: emotion conditioning done on Sooktam-2 (extended, not built); latency not measured; 2 languages |
| Resolved prosodic flattening + duration modelling across 10+ languages | partial: duration model done for Hindi + Tamil; prosodic flattening not measured directly |
| Replaced autoregressive DDPM with CNF OT-CFM | not done / not applicable (Sooktam already OT-CFM; DDPM is not autoregressive) |
| Zero-shot speaker conditioning with UnitSpeech from 3-s prompts | not done (Sooktam's reference-prompt cloning used; UnitSpeech not used) |
| Cross-attention duration prediction from reference mels (replacing MFA) | **done** (replaced the ratio heuristic; Sooktam never used MFA) |
| Lower RTF vs DDPM, <10 function evaluations | not done (all runs at NFE 32; no DDPM baseline) |
| Speaker similarity on 20+ unseen speakers, MCD, MOS | not done (speaker sim on 4 seen Rasa speakers only) |

Feasible next: NFE/RTF sweep (8/16/32), MCD vs real recordings, real-vs-real ECAPA calibration, small MOS test, 2-3 more languages.


## 5. Engineering issues met (for the methods/appendix)
- Colab Python 3.12/3.13 vs Sooktam's 3.10 pins → separate uv venv.
- `MPLBACKEND=module://matplotlib_inline...` inherited by the venv → crash; set `MPLBACKEND=Agg`.
- HF token placed in code instead of Colab Secrets → `SecretNotFoundError`.
- Laptop sleep → runtime disconnect; training resumes from 500-step checkpoints.
- Drive small-file I/O is slow → copy data cache to local disk each session (parallel copy; tar archive).
- faster-whisper + PyAV version clash on Colab Python 3.13 (`av.open(... metadata_errors=...)`) → decode audio with librosa and pass arrays.
- faster-whisper (CTranslate2) needs `libcublas.so.12`, absent from the newer Colab image → pip `nvidia-cublas-cu12` + LD_LIBRARY_PATH, or fall back to the transformers Whisper pipeline (now automatic).
- Colab file upload renames duplicates (`name (1).zip`) → silently re-used a stale zip; fixed by deleting old copies first and checking byte size.
