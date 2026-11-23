# Unseen-speaker evaluation for Sooktam-2 × Navarasa

Run the setup in `code/sooktam_rasa/Sooktam_Navarasa_Colab.ipynb` first. This evaluation uses the project Python package, Rasa manifest and trained checkpoints. Upload `unseen_speakers.py` to `/content/`. The commands below save new evaluation outputs in `MyDrive/sooktam_unseen_eval/`.

The Hugging Face repository size is for the whole corpus. Use the `prepare` command below to stream rows from the two test splits on CPU. The four-speaker pilot saves eight short WAVs, roughly 8 MB or less at 48 kHz mono. Streaming may read additional Parquet data while it searches for eligible speakers. [Hugging Face streaming guide](https://huggingface.co/docs/datasets/stream)

## 1. Get the speaker panel

Sign in to [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) and request access from your own Hugging Face account. Add a read token for that account to Colab Secrets as `HF_TOKEN`. Use the same secret as the original notebook if that account already has access. Upload `unseen_speakers.py` to `/content/` in the Colab Files panel. Run this cell after your normal Drive mount and Sooktam-2 setup.

```python
from google.colab import userdata
import os

os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
ROOT = '/content/drive/MyDrive/sooktam_rasa'
EVAL = '/content/drive/MyDrive/sooktam_unseen_eval'
CODE = '/content/sooktam_rasa'
PY = '/content/py310/bin/python'
SCRIPT = '/content/unseen_speakers.py'
!pip -q install datasets soundfile
```

The selector streams the Hindi and Tamil `test` splits, records the exact dataset revision, and saves 12 speaker IDs per language, with six male and six female speakers. Each speaker contributes two distinct 3–10 second clips. Listen to the `*-ref.wav` clips before synthesis and confirm they are clear, single-speaker, and neutral-sounding. IndicVoices-R is largely conversational and the selector has no neutral-emotion labels. Inspect transcripts for mismatches and note any exclusions before evaluating outputs. If the first panel needs replacements, prepare a new panel directory using a different fixed `--seed` and record the screening decision.

These speakers are drawn from a separate corpus. The script records the training manifest's groups and checks for literal ID overlap. Rasa's language-and-gender labels are not person identifiers, so this check cannot establish cross-dataset identity disjointness on its own. Overlap with Sooktam-2 pretraining is unknown. Report the scope as “evaluated on external IndicVoices-R speakers using a Rasa-trained adapter.”

## 2. Generate a small pilot, then the full run

Run a four-speaker pilot before generating the full set. Screen the reference audio after preparing the panel. Check a base and a v2 clip in each language before proceeding. The pilot uses the same selection rules and seed as the full panel.

```python
!python {SCRIPT} prepare --per-language 2 --panel {EVAL}/pilot_panel --training-manifest {ROOT}/data/manifest.jsonl
```

```python
!PYTHONPATH={CODE} {PY} {SCRIPT} run --panel {EVAL}/pilot_panel --out {EVAL}/pilot_small --sooktam-dir /content/sooktam2 --adapter {ROOT}/ckpt/emotion_v2/best.pt --duration-model {ROOT}/ckpt/duration/best.pt --sentences 1 --steps 8
```

The small pilot generates 20 clips using one sentence per speaker. Eight steps reduce runtime. The Sooktam-2 checkpoint and inference tensors still need to fit in GPU memory. If the normal Sooktam-2 notebook already runs on your GPU, start with this pilot. The pilot checks that the pipeline works, with just four speakers for the initial comparison.

You can score speaker similarity on CPU using only the ECAPA model. Use the project’s scoring Python and install `speechbrain` and `librosa` there if needed. This writes `summary_speaker.json` and `rows_scored_speaker.csv`.

```python
!PYTHONPATH={CODE} python {SCRIPT} score --panel {EVAL}/pilot_panel --out {EVAL}/pilot_small --device cpu --speaker-only
```

Prepare and screen the full panel when the pilot is ready.

```python
!python {SCRIPT} prepare --panel {EVAL}/panel --training-manifest {ROOT}/data/manifest.jsonl
```

```python
!PYTHONPATH={CODE} {PY} {SCRIPT} run --panel {EVAL}/panel --out {EVAL}/full --sooktam-dir /content/sooktam2 --adapter {ROOT}/ckpt/emotion_v2/best.pt --duration-model {ROOT}/ckpt/duration/best.pt
```

The full run generates 360 unique clips (24 speakers × 3 sentences × one adapter-off base clip plus four v2 emotions at `cfg_emo=1`). Add `--strong-guidance` with a separate output directory to include `cfg_emo=3` (648 clips). The base arm uses the same learned duration predictor with neutral duration weights and the same 32-step setting as v2. This isolates the adapter-enabled system on a common inference stack. Base clips are reused as the paired comparison for each target emotion and receive no target-emotion accuracy score. The run records checkpoint hashes and resumes saved clips when rerun with the same configuration.

## 3. Score and prepare a blinded listening set

Use the original project’s scoring environment, generally Colab’s default Python. If needed, install its scoring dependencies first.

```python
!pip -q install faster-whisper funasr modelscope speechbrain jiwer librosa
!PYTHONPATH={CODE} python {SCRIPT} score --panel {EVAL}/panel --out {EVAL}/full
!python {SCRIPT} listening --out {EVAL}/full
```

- `summary.json` reports ECAPA similarity, Whisper character error rate, emotion2vec agreement, and end-to-end real-time factor. The 95% intervals resample speakers within each language. System differences are computed on paired speakers. ECAPA uses the separate enrollment clip, and the report includes real-reference versus real-enrollment similarity for calibration. CER retains Indic vowel signs and marks, so its values may differ from the earlier report’s normalization.
- `rows_scored.csv` contains per-clip measurements, ASR transcripts and emotion predictions. Emotion2vec agreement is an automatic metric with known biases on this material.
- `listening/` contains up to 30 shuffled clips and a rating sheet for native Hindi and Tamil listeners. Keep `listening_key.csv` with the experimenter. Report the number of listeners and treat this small subset as a pilot.
