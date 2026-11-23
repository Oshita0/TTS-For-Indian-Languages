import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Sooktam-2 × Navarasa — emotional TTS for Indian languages
**UGP-2, IIT Kanpur.** Adds rasa/emotion control and a learned duration predictor to `bharatgenai/sooktam2`, trained on AI4Bharat **Rasa**.

| Stage | What runs | GPU time (rough) |
|---|---|---|
| 0–2 | install, baseline check | ~10 min each session |
| 3 | data cache (metadata scan + audio for chosen rows) | 30–90 min per language, once |
| 4 | duration predictor | 10–30 min |
| 5 | emotion adapter (LoRA + AdaLN conditioner, backbone frozen) | several hours; resumable |
| 6–7 | listen, evaluate | 30–60 min |

**Runtime:** `Runtime → Change runtime type → GPU`. T4 works (use `configs/emotion_t4.json`); L4/A100 (Colab Pro) is 3–6× faster (`configs/emotion_a100.json`).
Everything that matters (data cache, checkpoints, samples) lives on Google Drive, so a disconnect only costs the install step.

**Before you start:** Accept the `ai4bharat/Rasa` dataset terms on Hugging Face and add a read token as `HF_TOKEN` in Colab Secrets. Upload the project folder `code/sooktam_rasa/` to `MyDrive/sooktam_rasa/`, keeping the Python package and `configs/` directly inside that folder.
""")

code(r"""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
ROOT = '/content/drive/MyDrive/sooktam_rasa'
DATA = f'{ROOT}/data'
CKPT = f'{ROOT}/ckpt'
os.makedirs(DATA, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['MPLBACKEND'] = 'Agg'          # Colab's inline backend doesn't exist in the 3.10 venv
PY = '/content/py310/bin/python'          # Sooktam-2's pinned stack needs Python 3.10 (TF 2.15 / torch 2.6)
CODE = '/content/sooktam_rasa'
""")

md("## 1. Install (every new session)\nSooktam's `setup-cls.sh` pins Python 3.10 packages, while Colab ships 3.12, so a separate 3.10 venv is built with `uv`. The 5.4 GB training checkpoint `model_1250000.pt` is skipped; only the 1.3 GB `sooktam.safetensors` weights are downloaded.")
code(r"""
%%bash
set -e
pip -q install uv huggingface_hub
uv python install 3.10 -q
[ -d /content/py310 ] || uv venv --seed -q /content/py310 --python 3.10
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download('bharatgenai/sooktam2', local_dir='/content/sooktam2',
                  ignore_patterns=['model_1250000.pt'], token=os.environ.get('HF_TOKEN'))
EOF
cd /content/sooktam2
source /content/py310/bin/activate
sh setup-cls.sh > /content/setup_cls.log 2>&1 || (tail -30 /content/setup_cls.log; exit 1)
pip -q install grapheme librosa soundfile pandas pyarrow
project_dir=/content/drive/MyDrive/sooktam_rasa
runtime_dir=/content/sooktam_rasa
[ -f "$project_dir/sooktam_rasa/infer.py" ] && [ -d "$project_dir/configs" ] || {
    echo 'Upload code/sooktam_rasa/ as MyDrive/sooktam_rasa/ first.'
    exit 1
}
mkdir -p "$runtime_dir"
cp -R "$project_dir/sooktam_rasa" "$project_dir/configs" "$runtime_dir/"
echo INSTALL OK
""")

md("## 2. Baseline check: plain Sooktam-2 must sound right before anything is trained")
code(r"""
%%writefile /content/baseline.py
import sys; sys.path.insert(0, '/content/sooktam_rasa')
from sooktam_rasa.infer import EmotionalSooktam
tts = EmotionalSooktam.load('/content/sooktam2')
ref_text = "सर, मैं तब से यह कह रहा हूँ कि मैंने अपना टिकट कैंसल कर दिया है, लेकिन अब तक मेरे पैसे वापस नहीं आए हैं। आप इस मामले को देखेंगे भी या नहीं?"
for mode in ['bytes', 'grapheme']:
    tts.synthesize('/content/sooktam2/ref.wav', ref_text, 'यह एक टेस्ट वाक्य है जिसे आवाज़ में बदलना है।', 'hindi',
                   rasa=None, duration=mode, seed=0, out_path=f'/content/baseline_{mode}.wav')
print('ok')
""")
code(r"""
!cd /content && {PY} baseline.py 2>&1 | tail -5
from IPython.display import Audio, display
for m in ['bytes', 'grapheme']:
    print(m); display(Audio(f'/content/baseline_{m}.wav'))
""")

md("## 3. Build the data cache from Rasa\nFirst `--inspect`: it reads only the metadata columns and prints every `style` value with its hours and the rasa it maps to. **Check that the six emotions and the neutral style map correctly** before extracting. Start with one Indo-Aryan + one Dravidian language (Hindi + Tamil), so the Indo-Aryan/Dravidian gap from your report can be measured.")
code(r"""
LANGS = 'Hindi Tamil'
!cd {CODE} && {PY} -m sooktam_rasa.prepare --languages {LANGS} --data_dir {DATA} --inspect 2>&1 | grep -v Warning
""")
code(r"""
# extraction is resumable: if Colab disconnects, just run this cell again
!cd {CODE} && {PY} -m sooktam_rasa.prepare --languages {LANGS} --data_dir {DATA} --neutral_hours 2 --test_neutral_hours 0.3 --device cuda 2>&1 | grep -v Warning | tail -40
!wc -l {DATA}/manifest.jsonl
""")

md("## 4. Learned cross-attention duration predictor\nTrains on cached mels only (no TTS forward pass). It prints mean absolute % error for the byte formula (original bug), the grapheme formula (your fix), raw CLS-token counts, and the learned model, per language family.")
code(r"""
!cd {CODE} && {PY} -m sooktam_rasa.train_duration --config configs/duration.json --set sooktam_dir='"/content/sooktam2"' data_dir='"{DATA}"' out_dir='"{CKPT}/duration"' 2>&1 | grep -v "^step" | tail -40
""")

md("## 5. Train the emotion / rasa adapter\nBackbone frozen; trains an AdaLN emotion conditioner (zero-initialised, so step 0 = the original model) + LoRA r=16 on attention.\n70% of batches pair a **neutral** prompt from the same speaker with an **emotional** target, which is the inference situation: neutral reference clip + rasa label.\n\nResumable: if the session dies, rerun steps 0–1, then this cell. Listen to `samples/step_*/` as it trains.")
code(r"""
import torch
cfg = 'configs/emotion_a100.json' if torch.cuda.get_device_properties(0).total_memory > 30e9 else 'configs/emotion_t4.json'
print('using', cfg)
!cd {CODE} && {PY} -m sooktam_rasa.train_emotion --config {cfg} --set sooktam_dir='"/content/sooktam2"' data_dir='"{DATA}"' out_dir='"{CKPT}/emotion"' 2>&1 | grep -v Warning
""")
code(r"""
# listen to the latest training samples
import glob, os
from IPython.display import Audio, display
steps = sorted(glob.glob(f'{CKPT}/emotion/samples/step_*'))
print(steps[-1] if steps else 'no samples yet')
for p in sorted(glob.glob(f'{steps[-1]}/*.wav')) if steps else []:
    print(os.path.basename(p)); display(Audio(p))
""")

md("## 6. Navarasa with your own voice\nUpload a 3–10 s clip and its transcript. `intensity` scales the emotion (0 = neutral, 1 = trained, >1 = exaggerated); `cfg_emo` pushes away from the reference clip's own prosody. **Shringara and Veera are zero-shot mixtures** (no recordings exist for them), so judge them by ear.")
code(r"""
%%writefile /content/navarasa.py
import sys, json; sys.path.insert(0, '/content/sooktam_rasa')
from sooktam_rasa.infer import EmotionalSooktam
a = json.load(open('/content/navarasa_args.json'))
tts = EmotionalSooktam.load('/content/sooktam2', adapter=a['adapter'], duration_model=a['duration'])
print(tts.navarasa_sweep(a['ref'], a['ref_text'], a['text'], a['lang'], '/content/navarasa_out',
                         intensity=a['intensity'], cfg_emo=a['cfg_emo'], seed=0))
""")
code(r"""
import json
args = dict(ref='/content/sooktam2/ref.wav',
            ref_text="सर, मैं तब से यह कह रहा हूँ कि मैंने अपना टिकट कैंसल कर दिया है, लेकिन अब तक मेरे पैसे वापस नहीं आए हैं। आप इस मामले को देखेंगे भी या नहीं?",
            text='मुझे विश्वास नहीं हो रहा कि यह सब इतनी जल्दी हो गया।', lang='hindi',
            intensity=1.0, cfg_emo=1.0,
            adapter=f'{CKPT}/emotion/best.pt', duration=f'{CKPT}/duration/best.pt')
json.dump(args, open('/content/navarasa_args.json', 'w'), ensure_ascii=False)
!cd /content && {PY} navarasa.py 2>&1 | tail -3
from IPython.display import Audio, display
from glob import glob
for p in sorted(glob('/content/navarasa_out/*.wav')):
    print(p.split('/')[-1]); display(Audio(p))
""")

md("## 7. Evaluation\n**Step 1** synthesises the test set under 3 conditions (base Sooktam-2 / adapter without emotion guidance / adapter with guidance) in the 3.10 env. **Step 2** scores it in Colab's own Python: emotion accuracy (emotion2vec+ → classifier fitted on *real* Rasa speech, whose accuracy on held-out real speech is the ceiling), CER (faster-whisper large-v3), speaker similarity (ECAPA), duration error. It also writes a blind `listening_test.csv` for MOS.")
code(r"""
EVAL = f'{CKPT}/eval'
!cd {CODE} && {PY} -m sooktam_rasa.evaluate generate --data_dir {DATA} --out_dir {EVAL} --adapter {CKPT}/emotion/best.pt --duration_model {CKPT}/duration/best.pt --per_lang_rasa 10 2>&1 | tail -3
""")
code(r"""
!pip -q install faster-whisper funasr modelscope speechbrain scikit-learn jiwer librosa
!cd {CODE} && python -m sooktam_rasa.evaluate score --data_dir {DATA} --out_dir {EVAL} 2>&1 | grep -v Warning | tail -25
""")
code(r"""
import json, pandas as pd
s = json.load(open(f'{EVAL}/summary.json'))
print('emotion-judge ceiling on real speech:', s['judge_ceiling_real_speech'])
display(pd.DataFrame(s['by_condition']).T)
display(pd.DataFrame(s['by_condition_family']).T)
display(pd.DataFrame(s['by_condition_rasa']).T)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("Sooktam_Navarasa_Colab.ipynb", "w"), indent=1, ensure_ascii=False)
print("cells:", len(cells))
