# Sooktam-2 × Navarasa

Undergraduate project, IIT Kanpur · Oshita Bhongade and Pratyush Shrivastav · Supervisor: Prof. Arnab Bhattacharya

Start with [the 14-page project report](report/sooktam2_navarasa_report.pdf). It covers the duration predictor, the two emotion adapter versions, the Rasa experiments, and the four-speaker IndicVoices-R pilot and LoRA ablation. The editable LaTeX source is in the same folder.

## Download the complete project

The repository contains the project source, notebook, configurations, report, dataset metadata, and saved experiment tables and logs. The **Complete project bundle** release contains the full project folder, including all model checkpoints, training caches, reference recordings, and generated speech.

Download every numbered `sooktam2_navarasa_project.zip.partNN` asset from [Releases](https://github.com/Oshita0/TTS-For-Indian-Languages/releases), put them in one folder, and combine them on macOS or Linux:

```sh
cat sooktam2_navarasa_project.zip.part* > sooktam2_navarasa_project.zip
shasum -a 256 -c sooktam2_navarasa_project.zip.sha256
unzip sooktam2_navarasa_project.zip
```

Extract into a new folder. The ZIP contains one `sooktam2_navarasa_project/` directory. Its original paths match the table below. Checkpoints, WAVs, and the training cache are in the release bundle rather than Git history. `PROJECT_FILES.csv` indexes every source file and its location.

## Where to look

| Location | Contents |
|---|---|
| `code/sooktam_rasa/` | Training and evaluation code, configurations, tests, and results log. Open `Sooktam_Navarasa_Colab.ipynb` for the workflow and `RESULTS_LOG.md` for the experiment record. |
| `code/sooktam_rasa/ckpt/` | Duration predictor, emotion critic, v1 and v2 checkpoints, training logs, diagnostic JSON, and generated audio. The `control/`, `speed/`, and `mcd/` folders hold the evaluation outputs. |
| `code/sooktam_rasa/data/` | Processed Rasa manifest, metadata, mel features, and WAVs. The training cache `data_cache.tar` is beside this folder. |
| `code/external_speaker_evaluation/` | Four-speaker pilot and LoRA ablation code, panel metadata, reference audio, and scores. |
| `code/external_speaker_evaluation/pilot_assets/results_mps4/` | Pilot settings, 20 generated clips, per-clip ECAPA scores, and summary. |
| `code/external_speaker_evaluation/pilot_assets/results_lora_only/` | Four LoRA-only WAVs and paired speaker scores for base, LoRA only, and v2 shanta. |
| `code/external_speaker_evaluation/pilot_assets/sooktam2/` | Sooktam-2 inference source, tokenizer files, and BharatGen license. |

## Reproducing the external pilot

The saved audio and tables can be inspected directly. The pilot used two Hindi and two Tamil speakers, one sentence per language, eight sampling steps, and seed 0. Generation ran with PyTorch MPS on an Apple Silicon Mac, and SpeechBrain ECAPA scoring ran on CPU. The panel includes a separate enrollment WAV for every speaker.

Use the project notebook and code README for the Python environment. For the local pilot, install the packaged Sooktam-2 source in that environment and place the pretrained `sooktam.safetensors` in `code/external_speaker_evaluation/pilot_assets/sooktam2/`. The expected SHA-256 and the two task checkpoint hashes are recorded in `pilot_assets/results_mps4/settings.json` relative to the pilot directory. [BharatGen model weights](https://huggingface.co/bharatgenai/sooktam2)

From the complete project bundle (which includes the checkpoints and reference audio), run the following for a fresh generation and scoring pass:

```sh
cd code/external_speaker_evaluation
export PYTHONPATH="../sooktam_rasa"
python unseen_speakers.py run --panel pilot_assets/panel --out pilot_assets/reproduced \
  --sooktam-dir pilot_assets/sooktam2 --adapter pilot_assets/emotion_v2.pt \
  --duration-model pilot_assets/duration_download.pt --device mps \
  --steps 8 --sentences 1 --seed 0
python unseen_speakers.py score --panel pilot_assets/panel \
  --out pilot_assets/reproduced --device cpu --speaker-only
```

`lora_only_ablation.py` checks the original panel and checkpoint hashes. Its `generate` mode fills in missing LoRA-only WAVs. Its `score` mode creates `paired_speaker_scores.csv` and expects that path to be free, so use a working copy if rerunning it. `UNSEEN_SPEAKER_EVAL.md` records the original Colab workflow and proposed larger panel. The report and saved output folders describe the completed four-speaker experiment.

## Data and model attribution

- The processed Hindi and Tamil training and evaluation files come from [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), licensed CC BY 4.0. Processing includes selection, resampling, trimming, and mel feature extraction. The report cites the Rasa paper.
- The external speaker reference and enrollment clips come from [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r), licensed CC BY 4.0. The panel JSON records the exact dataset revision and source filenames. The report cites the IndicVoices-R paper.
- Licensed by BharatGen under the BharatGen Research License. Sooktam-2 is the BharatGen backbone. The duration predictor, emotion conditioning, and LoRA adaptations are the project team's modifications.

