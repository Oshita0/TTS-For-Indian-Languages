"""
Independent scoring of the controllability outputs (run in Colab's DEFAULT Python, not the 3.10 venv).

  * emotion2vec+ large, ZERO-SHOT (its own emotion labels, no training on Rasa, never seen by our model)
  * the same judge on REAL Rasa test recordings -> what it achieves on genuine Hindi/Tamil emotional speech (ceiling)
  * CER with faster-whisper large-v3 (intelligibility), ECAPA speaker similarity to the reference clip

    pip install faster-whisper funasr modelscope speechbrain jiwer librosa
    python -m sooktam_rasa.score_control --control_dir .../emotion_v2/control --data_dir /content/data_local
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict

from .rasa import BASIS

E2V_TO_BASIS = {"neutral": "shanta", "happy": "hasya", "sad": "karuna", "angry": "raudra",
                "fearful": "bhayanaka", "disgusted": "bibhatsa", "surprised": "adbhuta"}


class Emotion2Vec:
    def __init__(self, device="cuda"):
        from funasr import AutoModel
        try:
            self.m = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True, device=device)
        except Exception as e:  # noqa: BLE001
            print("HF hub failed, trying ModelScope:", repr(e)[:120])
            self.m = AutoModel(model="iic/emotion2vec_plus_large", hub="ms", disable_update=True, device=device)

    def __call__(self, path):
        r = self.m.generate(path, granularity="utterance", extract_embedding=False)[0]
        best, best_s = None, -1.0
        for lab, s in zip(r["labels"], r["scores"]):
            eng = lab.split("/")[-1].strip().lower()
            if eng in E2V_TO_BASIS and s > best_s:   # restrict to the 7 classes we use ("other"/"unk" ignored)
                best, best_s = E2V_TO_BASIS[eng], s
        return best


def confusion(rows, key_t="target", key_p="e2v"):
    tab = {t: {p: 0 for p in BASIS} for t in BASIS}
    for r in rows:
        if r[key_t] in tab and r[key_p] in tab:
            tab[r[key_t]][r[key_p]] += 1
    return tab


def print_conf(title, tab):
    print(title)
    print(f"{'target / pred':<14}" + "".join(f"{b[:6]:>8}" for b in BASIS))
    for t in BASIS:
        if sum(tab[t].values()):
            print(f"{t:<14}" + "".join(f"{tab[t][p]:>8}" for p in BASIS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--real_per_class", type=int, default=10, help="real test clips per language x rasa")
    ap.add_argument("--no_cer", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from .data import load_manifest
    from .evaluate import CER, SpeakerSim

    rows = list(csv.DictReader(open(os.path.join(a.control_dir, "rows.csv"), encoding="utf-8")))
    rng = random.Random(0)
    real = defaultdict(list)
    for it in load_manifest(a.data_dir, split="test"):
        if it.get("wav"):
            real[(it["lang"], it["basis"])].append(it)
    real_rows = []
    for (lang, b), lst in sorted(real.items()):
        for it in rng.sample(lst, min(a.real_per_class, len(lst))):
            real_rows.append(dict(condition="real_recording", lang=lang, target=BASIS[b],
                                  path=os.path.join(a.data_dir, it["wav"]), text=it["text"], ref=""))

    e2v = Emotion2Vec(a.device)
    for r in rows + real_rows:
        r["e2v"] = e2v(r["path"])
    print("emotion2vec done", flush=True)
    if not a.no_cer:
        cer = CER(a.device)
        for r in rows + real_rows:
            r["cer"] = cer(r["path"], r["text"], r["lang"])
        print("CER done", flush=True)
    spk = SpeakerSim(a.device)
    for r in rows:
        r["spk_sim"] = spk(r["path"], r["ref"])

    out = {}
    conds = ["real_recording"] + sorted({r["condition"] for r in rows}, key=lambda c: (c != "base", c))
    print(f"\n{'condition':<16}{'e2v acc':>9}{'e2v acc emo':>13}{'pred=shanta':>13}{'CER':>8}{'spk sim':>9}{'n':>5}")
    for c in conds:
        rs = real_rows if c == "real_recording" else [r for r in rows if r["condition"] == c]
        tgt = [r for r in rs if r["target"] in BASIS]
        acc = sum(r["e2v"] == r["target"] for r in tgt) / max(1, len(tgt))
        emo = [r for r in tgt if r["target"] != "shanta"]
        acc_e = sum(r["e2v"] == r["target"] for r in emo) / max(1, len(emo))
        neu = sum(r["e2v"] == "shanta" for r in rs) / max(1, len(rs))
        cers = [float(r["cer"]) for r in rs if "cer" in r and r["cer"] == r["cer"]]
        sims = [float(r["spk_sim"]) for r in rs if "spk_sim" in r]
        mc = sum(cers) / len(cers) if cers else float("nan")
        ms = sum(sims) / len(sims) if sims else float("nan")
        accs = f"{acc:>9.3f}{acc_e:>13.3f}" if tgt else f"{'-':>9}{'-':>13}"
        print(f"{c:<16}{accs}{neu:>13.3f}{mc:>8.3f}{ms:>9.3f}{len(rs):>5}")
        out[c] = dict(e2v_acc=acc if tgt else None, e2v_acc_emotional=acc_e if tgt else None, pred_shanta=neu,
                      cer=mc, spk_sim=ms, n=len(rs), confusion=confusion(tgt) if tgt else None,
                      by_lang={l: sum(r["e2v"] == r["target"] for r in tgt if r["lang"] == l) /
                               max(1, sum(r["lang"] == l for r in tgt)) for l in sorted({r["lang"] for r in tgt})})
    for c in conds:
        if out[c]["confusion"]:
            print_conf(f"\nemotion2vec confusion: {c}", out[c]["confusion"])
    base = [r for r in rows if r["condition"] == "base"]
    if base:
        dist = {b: sum(r["e2v"] == b for r in base) / len(base) for b in BASIS}
        print("\nbase Sooktam-2, emotion2vec says:", {k: round(v, 2) for k, v in dist.items() if v})
        out["base"]["prediction_distribution"] = dist
    json.dump(out, open(os.path.join(a.control_dir, "independent_scores.json"), "w"), indent=2)
    with open(os.path.join(a.control_dir, "rows_scored.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows + real_rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows + real_rows)
    print("\nsaved", os.path.join(a.control_dir, "independent_scores.json"))


if __name__ == "__main__":
    main()
