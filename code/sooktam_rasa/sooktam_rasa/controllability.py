"""
Controllability test: can the rasa LABEL alone make speech emotional?

Neutral reference clip + NEUTRAL sentence (from the test split), synthesised once per rasa and per emotion-guidance
strength. The frozen mel emotion critic (trained on real Rasa speech, 90.6% on real test clips) classifies each output.
If the label works, the confusion matrix is diagonal; if it is ignored, every row collapses onto "shanta".

    python -m sooktam_rasa.controllability --data_dir /content/data_local --adapter .../emotion_v2/latest.pt \
        --critic .../emo_cls.pt --duration_model .../duration/best.pt --out_dir .../emotion_v2/control

Also writes all wavs + rows.csv, so the same files can be scored by the independent emotion2vec judge.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

import torch

from .data import load_manifest
from .rasa import BASIS, NUM_BASIS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--duration_model", default=None)
    ap.add_argument("--sooktam_dir", default="/content/sooktam2")
    ap.add_argument("--refs_per_lang", type=int, default=2)
    ap.add_argument("--texts_per_lang", type=int, default=3)
    ap.add_argument("--cfg_emo", type=float, nargs="+", default=[0.0, 1.0, 3.0])
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from .emo_classifier import load_classifier
    from .infer import EmotionalSooktam

    rng = random.Random(a.seed)
    tts = EmotionalSooktam.load(a.sooktam_dir, adapter=a.adapter, duration_model=a.duration_model)
    dev = tts.device
    critic = load_classifier(a.critic, dev)
    neutral = BASIS.index("shanta")
    test = [it for it in load_manifest(a.data_dir, split="test") if it["basis"] == neutral and it.get("wav")]
    langs = sorted({it["lang"] for it in test})

    cases = []
    for lang in langs:
        pool = [it for it in test if it["lang"] == lang]
        spk = sorted({it["speaker"] for it in pool})
        refs = []
        for i in range(a.refs_per_lang):  # alternate speakers
            cand = [it for it in pool if it["speaker"] == spk[i % len(spk)] and it not in refs]
            refs.append(rng.choice(cand))
        texts = rng.sample([it for it in pool if it not in refs], a.texts_per_lang)
        cases += [(lang, r, t) for r in refs for t in texts]
    print(f"{len(cases)} (reference, neutral sentence) pairs over {langs}")

    conditions = [("base", None, 0.0)] + [(f"cfg{c:g}", "label", c) for c in a.cfg_emo]
    rows = []
    for cname, mode, cfg in conditions:
        targets = [None] if mode is None else list(range(NUM_BASIS))
        for tgt in targets:
            for lang, ref, txt in cases:
                rasa = None if tgt is None else BASIS[tgt]
                path = os.path.join(a.out_dir, cname, f"{lang}_{rasa or 'base'}_{os.path.basename(ref['mel'])[:-3]}_"
                                    f"{os.path.basename(txt['mel'])[:-3]}.wav")
                wav, sr = tts.synthesize(os.path.join(a.data_dir, ref["wav"]), ref["text"], txt["text"], lang, rasa=rasa,
                                         intensity=a.intensity, cfg_emo=cfg, duration="learned" if a.duration_model else "grapheme",
                                         seed=a.seed, out_path=path)
                with torch.no_grad():
                    mel = tts.cfm.mel_spec(torch.from_numpy(wav).float()[None].to(dev)).permute(0, 2, 1)
                    probs = critic(mel).softmax(-1)[0].cpu()
                rows.append(dict(condition=cname, lang=lang, target=rasa or "none", pred=BASIS[int(probs.argmax())],
                                 p_target=float(probs[tgt]) if tgt is not None else float("nan"),
                                 p_shanta=float(probs[neutral]), seconds=len(wav) / sr, path=path, text=txt["text"],
                                 ref=os.path.join(a.data_dir, ref["wav"])))
        print(f"  done {cname}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    base_rows = [r for r in rows if r["condition"] == "base"]
    base_dist = {b: sum(r["pred"] == b for r in base_rows) / len(base_rows) for b in BASIS}
    print("\nbase Sooktam-2 on neutral sentences, critic says:", {k: round(v, 2) for k, v in base_dist.items() if v > 0})
    summary["base_prediction_distribution"] = base_dist
    for cname, mode, cfg in conditions[1:]:
        rs = [r for r in rows if r["condition"] == cname]
        acc = sum(r["pred"] == r["target"] for r in rs) / len(rs)
        emo_rs = [r for r in rs if r["target"] != "shanta"]
        acc_emo = sum(r["pred"] == r["target"] for r in emo_rs) / len(emo_rs)
        p_tgt = sum(r["p_target"] for r in rs) / len(rs)
        print(f"\n=== {cname} (cfg_emo={cfg:g}, intensity={a.intensity:g}) ===")
        print(f"controllability accuracy {acc:.3f}  (chance {1 / NUM_BASIS:.3f}; emotional rasas only {acc_emo:.3f};"
              f" mean p(target) {p_tgt:.3f})")
        hdr = "target / pred"
        print(f"{hdr:<14}" + "".join(f"{b[:6]:>8}" for b in BASIS))
        conf = []
        for t in BASIS:
            row = [sum(1 for r in rs if r["target"] == t and r["pred"] == p) for p in BASIS]
            conf.append(row)
            print(f"{t:<14}" + "".join(f"{v:>8}" for v in row))
        by_lang = {l: sum(r["pred"] == r["target"] for r in rs if r["lang"] == l) / max(1, sum(r["lang"] == l for r in rs))
                   for l in langs}
        print("by language:", {k: round(v, 3) for k, v in by_lang.items()})
        summary[cname] = dict(cfg_emo=cfg, accuracy=acc, accuracy_emotional=acc_emo, mean_p_target=p_tgt,
                              confusion=conf, by_language=by_lang)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    print("\nsaved", os.path.join(a.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
