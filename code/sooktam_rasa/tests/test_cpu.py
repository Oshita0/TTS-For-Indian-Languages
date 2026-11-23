"""CPU smoke tests with a tiny random DiT standing in for Sooktam-2. Run: python tests/test_cpu.py"""
import io
import json
import os
import random
import shutil
import sys
import tempfile
import types
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import torch
from f5_tts.model import CFM, DiT

import sooktam_rasa.infer as infer_mod
from sooktam_rasa import data as D
from sooktam_rasa.duration import CrossAttnDurationPredictor
from sooktam_rasa.emotion_dit import attach_emotion
from sooktam_rasa.rasa import BASIS, NAVARASA, rasa_weights, style_to_basis

torch.manual_seed(0)
random.seed(0)
tmp = tempfile.mkdtemp()
VOCAB = [" "] + [chr(c) for c in range(0x900, 0x97F)] + list("abcdefghijklmnopqrstuvwxyz.,")
open(os.path.join(tmp, "vocab.txt"), "w", encoding="utf-8").write("\n".join(VOCAB) + "\n")


def tiny_cfm():
    from f5_tts.model.utils import get_tokenizer
    vmap, vsize = get_tokenizer(os.path.join(tmp, "vocab.txt"), "custom")
    tr = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, text_dim=32, conv_layers=1, text_num_embeds=vsize, mel_dim=100)
    for p in tr.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.02)
    return CFM(transformer=tr, vocab_char_map=vmap,
               mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
                                    target_sample_rate=24000, mel_spec_type="vocos"))


# 1. style mapping ---------------------------------------------------------------------------
for s, want in [("HAPPY", "hasya"), ("Sad", "karuna"), ("ANGER", "raudra"), ("UMANG", None), ("INDIC", "shanta"), ("FEAR", "bhayanaka"),
                ("DISGUST", "bibhatsa"), ("SURPRISE", "adbhuta"), ("WIKI", "shanta"), ("CONV", None)]:
    b = style_to_basis(s)
    assert (BASIS[b] if b is not None else None) == want, (s, b)
assert abs(rasa_weights("veera").sum() - 1) < 1e-6 and len(NAVARASA) == 9
print("[ok] rasa mapping")

# 2. featurizer on a fake 48 kHz clip ---------------------------------------------------------
sr = 48000
t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
x = np.concatenate([np.zeros(sr // 4), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(sr // 4)]).astype(np.float32)
buf = io.BytesIO()
sf.write(buf, x, sr, format="WAV")
feat = D.Featurizer()
wav = feat.wav_from_bytes(buf.getvalue())
mel = feat.mel_from_wav(wav)
assert 1.4 < len(wav) / 24000 < 1.7, len(wav) / 24000   # edge silence trimmed, +50ms
assert mel.shape[1] == 100
print("[ok] featurizer", tuple(mel.shape), f"{len(wav) / 24000:.2f}s")

# 3. select_rows on fake metadata --------------------------------------------------------------
import pandas as pd
rows = []
for g in ["Male", "Female"]:
    for st in ["WIKI"] * 40 + ["HAPPY", "SAD", "ANGER", "FEAR", "DISGUST", "SURPRISE"] * 8 + ["CONV"] * 5:
        rows.append(dict(filename=f"{g}_{len(rows)}.wav", text="नमस्ते दुनिया", gender=g, style=st,
                         duration=random.uniform(2, 6), shard="s0", row_group=0, row=len(rows)))
df = pd.DataFrame(rows)
D.inspect_styles(df)
sel = D.select_rows(df, neutral_hours_per_speaker=60 / 3600)
assert (sel["basis"] == 0).groupby(sel["gender"]).sum().max() <= 20
print("[ok] select_rows")

# 4. synthetic manifest (2 languages x 2 speakers) --------------------------------------------
os.makedirs(os.path.join(tmp, "mels"), exist_ok=True)
os.makedirs(os.path.join(tmp, "wavs"), exist_ok=True)
items = []
for lang, fam in [("hindi", "indo-aryan"), ("tamil", "dravidian")]:
    for spk in ["male", "female"]:
        for k in range(30):
            basis = 0 if k < 12 else 1 + (k % 6)
            n_tok = random.randint(8, 30)
            toks = [random.choice(VOCAB[1:60]) for _ in range(n_tok)]
            rate = (2.0 if fam == "dravidian" else 1.5) * (1.3 if BASIS[basis] == "karuna" else 1.0)
            n = int(n_tok * rate * 3) + 5
            m = torch.randn(n, 100) * 0.5 - 5
            rel = f"mels/{lang}_{spk}_{k}.pt"
            torch.save(m.half(), os.path.join(tmp, rel))
            wrel = f"wavs/{lang}_{spk}_{k}.wav"
            sf.write(os.path.join(tmp, wrel), np.random.randn(n * 256).astype(np.float32) * 0.05, 24000)
            items.append(dict(id=f"{lang}/{spk}/{k}", split="test" if k % 10 == 9 else "train", lang=lang, family=fam,
                              speaker=f"{lang}_{spk}", style="x", basis=basis, text="".join(toks), tokens=toks,
                              n_frames=n, n_graphemes=n_tok, n_bytes=len("".join(toks).encode()), mel=rel, wav=wrel))
with open(os.path.join(tmp, "manifest.jsonl"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("[ok] manifest", len(items))

# 5. emotion training via the real train_emotion.main, with Sooktam swapped for the tiny model ----
class FakeVocoder:
    def decode(self, mel):  # (1, 100, T) -> (1, T*256)
        return torch.randn(1, mel.shape[-1] * 256) * 0.01

def fake_load(sooktam_dir, ckpt=None, device=None):
    cfm = tiny_cfm()
    return types.SimpleNamespace(ema_model=cfm, vocoder=FakeVocoder(), device="cpu")

infer_mod.load_sooktam = fake_load
from sooktam_rasa import train_emotion as TE
cfg = dict(TE.DEFAULTS, sooktam_dir=tmp, data_dir=tmp, out_dir=os.path.join(tmp, "ck_emo"), max_steps=30,
           warmup=5, frames_per_batch=600, grad_accum=1, log_every=10, val_every=15, save_every=15,
           sample_every=0, num_workers=0, val_batches=3, lora_r=4, amp="none")
TE.main(cfg)
assert os.path.exists(os.path.join(cfg["out_dir"], "best.pt"))
cfg["max_steps"] = 35
TE.main(cfg)   # resume path
print("[ok] emotion training + resume")
from sooktam_rasa import diagnose
sys.argv = ["x", "--data_dir", tmp, "--sooktam_dir", tmp, "--frames_per_batch", "600", "--n_batches", "3",
            "--ckpts", os.path.join(cfg["out_dir"], "best.pt"), os.path.join(cfg["out_dir"], "latest.pt"),
            "--out", os.path.join(tmp, "diag.json")]
diagnose.main()
assert os.path.exists(os.path.join(tmp, "diag.json"))
print("[ok] diagnose")

# v2: critic -> training with text-drop + critic + input injection -> diagnose
from sooktam_rasa import emo_classifier
cls_path = os.path.join(tmp, "emo_cls.pt")
sys.argv = ["x", "--data_dir", tmp, "--out", cls_path, "--steps", "40", "--batch_size", "8", "--workers", "0"]
emo_classifier.main()
cfg2 = dict(cfg, out_dir=os.path.join(tmp, "ck_v2"), max_steps=20, val_every=10, save_every=10, p_pair=1.0,
            emo_input_proj=True, text_drop_keep_emo_prob=0.3, cls_ckpt=cls_path, cls_weight=0.15, lr_emotion=1e-3,
            log_every=5)
TE.main(cfg2)
sys.argv = ["x", "--data_dir", tmp, "--sooktam_dir", tmp, "--frames_per_batch", "600", "--n_batches", "2",
            "--ckpts", os.path.join(cfg2["out_dir"], "latest.pt")]
diagnose.main()
print("[ok] v2 training + diagnose")

# 6. duration predictor ------------------------------------------------------------------------
from sooktam_rasa import train_duration as TD
dcfg = dict(TD.DEFAULTS, sooktam_dir=tmp, data_dir=tmp, out_dir=os.path.join(tmp, "ck_dur"), d=64, heads=2,
            text_layers=1, mel_layers=1, xattn_layers=1, batch_size=8, max_steps=300, val_every=150, num_workers=0)
TD.main(dcfg)
res = json.load(open(os.path.join(dcfg["out_dir"], "best_results.json")))
print("[ok] duration training; learned vs cls_tokens:", round(res["ALL"]["learned"], 1), round(res["ALL"]["cls_tokens"], 1))

# 7. synthesis path (tokenizer + ref preprocessing mocked) ---------------------------------------
f5 = fake_load(tmp)
attach_emotion(f5.ema_model, lora_r=4)
blob = torch.load(os.path.join(cfg["out_dir"], "best.pt"))
from sooktam_rasa.emotion_dit import load_adapter
load_adapter(f5.ema_model.transformer, blob["adapter"])
dblob = torch.load(os.path.join(dcfg["out_dir"], "best.pt"))
dur = CrossAttnDurationPredictor(**dblob["model_kwargs"])
dur.load_state_dict(dblob["model"])
tts = infer_mod.EmotionalSooktam(f5, dur.eval())
tts._tokens = lambda text, lang: list(text)
tts._ref = lambda f, t: (torch.randn(1, 60, 100) - 5, t + ". ", 0.1)
for rasa in ["karuna", "veera", None, {"raudra": .5, "shanta": .5}]:
    for dmode in ["learned", "grapheme", "bytes", 2.0]:
        w, sr_ = tts.synthesize("ref.wav", "नमस्ते", "आज मौसम अच्छा है", "hindi", rasa=rasa, duration=dmode,
                                nfe_step=4, seed=0, out_path=os.path.join(tmp, "out.wav"))
        assert np.isfinite(w).all() and len(w) > 0
print("[ok] synthesis (all rasa / duration modes)")
f5b = fake_load(tmp)
b2 = torch.load(os.path.join(cfg2["out_dir"], "latest.pt"), weights_only=False)
attach_emotion(f5b.ema_model, lora_r=4, input_proj=b2["config"]["emo_input_proj"])
load_adapter(f5b.ema_model.transformer, b2["adapter"])
tts2 = infer_mod.EmotionalSooktam(f5b, dur.eval())
tts2._tokens, tts2._ref = tts._tokens, tts._ref
for rasa in ["karuna", None]:
    w, _ = tts2.synthesize("ref.wav", "नमस्ते", "आज मौसम अच्छा है", "hindi", rasa=rasa, nfe_step=4, seed=0, cfg_emo=2.0)
    assert np.isfinite(w).all()
print("[ok] v2 synthesis")

# controllability test on the v2 adapter
from sooktam_rasa import controllability
infer_mod.EmotionalSooktam._tokens = lambda self, text, lang: list(text)
infer_mod.EmotionalSooktam._ref = lambda self, f, t: (torch.randn(1, 60, 100) - 5, t + ". ", 0.1)
sys.argv = ["x", "--data_dir", tmp, "--sooktam_dir", tmp, "--adapter", os.path.join(cfg2["out_dir"], "latest.pt"),
            "--critic", cls_path, "--out_dir", os.path.join(tmp, "control"), "--refs_per_lang", "1", "--texts_per_lang", "1",
            "--cfg_emo", "0", "2"]
controllability.main()
print("[ok] controllability")

from sooktam_rasa import speed_mcd
for mode, extra in [("speed", ["--critic", cls_path, "--nfe", "2", "4", "--rasas", "raudra", "karuna"]),
                    ("mcd", ["--per_class", "1", "--nfe_step", "4"])]:
    sys.argv = ["x", mode, "--data_dir", tmp, "--sooktam_dir", tmp, "--adapter", os.path.join(cfg2["out_dir"], "latest.pt"),
                "--out_dir", os.path.join(tmp, mode), "--refs_per_lang", "1", "--texts_per_lang", "1"] + extra
    speed_mcd.main()
print("[ok] speed + mcd")

# 8. eval generation step --------------------------------------------------------------------
from sooktam_rasa.evaluate import generate_eval_set, summarize
test_items = D.load_manifest(tmp, split="test")
rows = generate_eval_set(tts, test_items, tmp, os.path.join(tmp, "eval"), per_lang_rasa=2)
assert os.path.exists(os.path.join(tmp, "eval", "eval_rows.csv")) and len({r["condition"] for r in rows}) == 3
for r in rows:
    r.update(emo_correct=random.randint(0, 1), cer=0.1, spk_sim=0.5, dur_err_pct=10.0)
summarize(rows)
print("[ok] eval generation", len(rows), "rows")
shutil.rmtree(tmp)
print("\nALL TESTS PASSED")
