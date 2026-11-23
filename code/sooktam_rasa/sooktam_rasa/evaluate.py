"""
Objective evaluation of emotional Sooktam-2.

Metrics per (language, rasa):
  * emotion accuracy  - an in-domain classifier: emotion2vec+ embeddings -> logistic regression trained
                        on REAL Rasa recordings. Its accuracy on held-out real speech is reported as the
                        ceiling, so synthetic scores have something to be compared against.
  * CER               - faster-whisper large-v3 (same as the bug report; Odia has no Whisper code).
  * speaker similarity- ECAPA (speechbrain) cosine between reference clip and output.
  * duration          - generated seconds vs. ground-truth recording of the same sentence+emotion.

Conditions compared: base Sooktam-2 (no adapter), adapter with cfg_emo=0, adapter with cfg_emo>0.
MOS / MUSHRA still needs human listeners; make_listening_test() builds a blind, shuffled sheet.
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict

import numpy as np

# NOTE: scoring imports (funasr, faster_whisper, speechbrain, sklearn) are lazy, so STEP 1 runs without them.

from .rasa import BASIS, EKMAN_OF

WHISPER_CODE = {"hindi": "hi", "marathi": "mr", "gujarati": "gu", "bengali": "bn", "punjabi": "pa", "urdu": "ur",
                "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml", "english": "en", "odia": None}


# ---------------- emotion judge ---------------------------------------------------------------
class EmotionJudge:
    def __init__(self, device="cuda"):
        from funasr import AutoModel
        try:
            self.m = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True, device=device)
        except Exception as e:  # noqa: BLE001
            print("HF hub load failed, trying ModelScope:", repr(e)[:150])
            self.m = AutoModel(model="iic/emotion2vec_plus_large", hub="ms", disable_update=True, device=device)
        self.clf = None

    def embed(self, wav_paths):
        out = []
        for p in wav_paths:
            r = self.m.generate(p, granularity="utterance", extract_embedding=True)[0]
            out.append(np.asarray(r["feats"]).reshape(-1))
        return np.stack(out)

    def fit(self, wav_paths, labels, test_paths=None, test_labels=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        self.clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        self.clf.fit(self.embed(wav_paths), labels)
        if test_paths:
            acc = accuracy_score(test_labels, self.clf.predict(self.embed(test_paths)))
            print(f"emotion judge accuracy on held-out REAL speech: {acc:.3f} (ceiling for synthetic)")
            return acc

    def predict(self, wav_paths):
        return self.clf.predict(self.embed(wav_paths))


# ---------------- ASR / CER -------------------------------------------------------------------
class CER:
    """Whisper large-v3 CER. Tries faster-whisper first; if its CUDA libraries are missing (common on new Colab images),
    falls back to the Hugging Face transformers pipeline, which runs on PyTorch's own CUDA."""

    def __init__(self, device="cuda", backend="auto"):
        import numpy as np
        self.backend = None
        if backend in ("auto", "faster"):
            try:
                from faster_whisper import WhisperModel
                self.m = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
                list(self.m.transcribe(np.zeros(16000, dtype=np.float32), language="hi")[0])  # forces a GPU encode now
                self.backend = "faster"
            except Exception as e:  # noqa: BLE001
                if backend == "faster":
                    raise
                print("faster-whisper unavailable, using transformers Whisper:", repr(e)[:150])
        if self.backend is None:
            import torch
            from transformers import pipeline
            self.m = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                              torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                              device=0 if device == "cuda" else -1)
            self.backend = "hf"

    def __call__(self, wav_path, ref_text, language):
        import jiwer
        import librosa  # decode ourselves: avoids faster-whisper / PyAV version clashes
        code = WHISPER_CODE.get(language)
        if code is None:
            return float("nan")
        audio, _ = librosa.load(wav_path, sr=16000)
        if self.backend == "faster":
            segs, _ = self.m.transcribe(audio, language=code, beam_size=5)
            hyp = "".join(s.text for s in segs).strip()
        else:
            hyp = self.m({"raw": audio, "sampling_rate": 16000},
                         generate_kwargs={"language": code, "task": "transcribe"})["text"].strip()
        norm = lambda s: "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()
        return jiwer.cer(norm(ref_text), norm(hyp))


# ---------------- speaker similarity ----------------------------------------------------------
class SpeakerSim:
    def __init__(self, device="cuda"):
        from speechbrain.inference.speaker import EncoderClassifier
        self.enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                  run_opts={"device": device}, savedir="/tmp/ecapa")

    def _emb(self, path):
        import librosa
        import torch
        wav, _ = librosa.load(path, sr=16000)
        return self.enc.encode_batch(torch.tensor(wav)[None]).squeeze()

    def __call__(self, a, b):
        import torch
        return torch.nn.functional.cosine_similarity(self._emb(a), self._emb(b), dim=0).item()


# ---------------- full evaluation -------------------------------------------------------------
def pick_eval_set(items, per_lang_rasa=10, seed=0):
    """Test items that have a wav; for each, a neutral reference from the same speaker (different utt)."""
    rng = random.Random(seed)
    neutral = BASIS.index("shanta")
    by = defaultdict(list)
    refs = defaultdict(list)
    for it in items:
        if not it.get("wav"):
            continue
        by[(it["lang"], it["basis"])].append(it)
        if it["basis"] == neutral:
            refs[it["speaker"]].append(it)
    cases = []
    for (lang, basis), lst in sorted(by.items()):
        rng.shuffle(lst)
        for tgt in lst[:per_lang_rasa]:
            pool = [r for r in refs[tgt["speaker"]] if r["id"] != tgt["id"]]
            if pool:
                cases.append((rng.choice(pool), tgt))
    return cases


def generate_eval_set(tts, items, data_dir, out_dir, conditions=None, per_lang_rasa=10):
    """STEP 1 (Sooktam env): synthesise every eval case under every condition -> eval_rows.csv"""
    conditions = conditions or {
        "base_sooktam": dict(rasa=None),          # no adapter at all (LoRA switched off)
        "adapter_cfg0": dict(cfg_emo=0.0),        # label only through AdaLN
        "adapter_cfg1": dict(cfg_emo=1.0),        # + emotion guidance
    }
    os.makedirs(out_dir, exist_ok=True)
    cases = pick_eval_set(items, per_lang_rasa)
    print(f"{len(cases)} eval cases x {len(conditions)} conditions")
    rows = []
    for cname, ckw in conditions.items():
        for ref, tgt in cases:
            rasa = BASIS[tgt["basis"]]
            kw = dict(rasa=rasa, duration="learned")
            kw.update(ckw)
            stem = os.path.basename(tgt["mel"])[:-3]
            path = os.path.join(out_dir, cname, f"{tgt['lang']}_{rasa}_{stem}.wav")
            wav, sr = tts.synthesize(os.path.join(data_dir, ref["wav"]), ref["text"], tgt["text"], tgt["lang"],
                                     seed=0, out_path=path, **kw)
            rows.append(dict(condition=cname, lang=tgt["lang"], family=tgt["family"], rasa=rasa, label=tgt["basis"],
                             path=path, ref=os.path.join(data_dir, ref["wav"]), real=os.path.join(data_dir, tgt["wav"]),
                             text=tgt["text"], gen_sec=len(wav) / sr, true_sec=tgt["n_frames"] * 256 / 24000))
    _write_rows(rows, os.path.join(out_dir, "eval_rows.csv"))
    return rows


def _write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def score_eval(out_dir, items, data_dir, device="cuda"):
    """STEP 2 (any env with funasr / faster-whisper / speechbrain): score eval_rows.csv -> summary.json"""
    with open(os.path.join(out_dir, "eval_rows.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"], r["gen_sec"], r["true_sec"] = int(r["label"]), float(r["gen_sec"]), float(r["true_sec"])

    # judge trained on real recordings that are NOT eval targets
    eval_real = {r["real"] for r in rows}
    jt = [it for it in items if it.get("wav") and os.path.join(data_dir, it["wav"]) not in eval_real]
    random.Random(0).shuffle(jt)
    cut = int(len(jt) * 0.8)
    p = lambda lst: [os.path.join(data_dir, it["wav"]) for it in lst]
    lab = lambda lst: [it["basis"] for it in lst]
    judge = EmotionJudge(device)
    ceiling = judge.fit(p(jt[:cut]), lab(jt[:cut]), p(jt[cut:]), lab(jt[cut:]))

    # also score the REAL recording of each eval sentence -> per-language reference point
    real_rows = {r["real"]: dict(r, condition="real_recording", path=r["real"], gen_sec=r["true_sec"]) for r in rows}
    rows = rows + list(real_rows.values())
    preds = judge.predict([r["path"] for r in rows])
    cer, spk = CER(device), SpeakerSim(device)
    for r, pr in zip(rows, preds):
        r["emo_pred"] = BASIS[int(pr)]
        r["emo_correct"] = int(int(pr) == r["label"])
        r["cer"] = cer(r["path"], r["text"], r["lang"])
        r["spk_sim"] = spk(r["path"], r["ref"])
        r["dur_err_pct"] = abs(r["gen_sec"] - r["true_sec"]) / r["true_sec"] * 100
    _write_rows(rows, os.path.join(out_dir, "eval_scored.csv"))
    summary = summarize(rows)
    summary["judge_ceiling_real_speech"] = ceiling
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    return rows, summary


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = dict(emo_acc=("emo_correct", "mean"), cer=("cer", "mean"), spk_sim=("spk_sim", "mean"),
               dur_err_pct=("dur_err_pct", "mean"), n=("path", "count"))
    out = {"by_condition": df.groupby("condition").agg(**agg).round(4).to_dict("index"),
           "by_condition_rasa": {f"{c}|{r}": v for (c, r), v in
                                 df.groupby(["condition", "rasa"]).agg(**agg).round(4).to_dict("index").items()},
           "by_condition_family": {f"{c}|{f}": v for (c, f), v in
                                   df.groupby(["condition", "family"]).agg(**agg).round(4).to_dict("index").items()}}
    print(pd.DataFrame(out["by_condition"]).T)
    return out


def make_listening_test(rows, out_csv, n_per_condition=40, seed=0):
    """Blind sheet: shuffled file ids, listeners rate naturalness (1-5) and pick the perceived rasa."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    picked = [r for lst in by.values() for r in rng.sample(lst, min(n_per_condition, len(lst)))]
    rng.shuffle(picked)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "audio_path", "naturalness_1to5", "perceived_rasa (" + "/".join(BASIS) + ")"])
        key = []
        for i, r in enumerate(picked):
            w.writerow([f"item_{i:03d}", r["path"], "", ""])
            key.append(dict(item=f"item_{i:03d}", condition=r["condition"], rasa=r["rasa"], ekman=EKMAN_OF[r["rasa"]]))
    json.dump(key, open(out_csv.replace(".csv", "_KEY.json"), "w"), indent=2)
    print("listening sheet:", out_csv, "(keep the _KEY.json away from raters)")


if __name__ == "__main__":
    import argparse

    from .data import load_manifest

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generate", "score"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sooktam_dir", default="/content/sooktam2")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--duration_model", default=None)
    ap.add_argument("--per_lang_rasa", type=int, default=10)
    ap.add_argument("--languages", nargs="*", default=None, help="sooktam names, e.g. hindi tamil")
    a = ap.parse_args()
    test_items = load_manifest(a.data_dir, split="test", langs=a.languages)
    if a.step == "generate":
        from .infer import EmotionalSooktam
        tts = EmotionalSooktam.load(a.sooktam_dir, adapter=a.adapter, duration_model=a.duration_model)
        generate_eval_set(tts, test_items, a.data_dir, a.out_dir, per_lang_rasa=a.per_lang_rasa)
    else:
        rows, _ = score_eval(a.out_dir, test_items, a.data_dir)
        make_listening_test([r for r in rows if r["condition"] != "real_recording"] +
                            [r for r in rows if r["condition"] == "real_recording"],
                            os.path.join(a.out_dir, "listening_test.csv"))
