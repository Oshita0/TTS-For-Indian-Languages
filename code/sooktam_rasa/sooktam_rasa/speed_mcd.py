"""
Two experiments for the "fast synthesis" and "MCD" claims.

speed : synthesis speed vs number of ODE steps (NFE).  For NFE in {4, 8, 16, 32}: real-time factor (end-to-end and ODE-only,
        GPU-synchronised, after warm-up) + emotion controllability (critic). Writes rows.csv in the same format as
        controllability.py, so `score_control` adds emotion2vec / CER / speaker similarity per NFE.
        Note: with the adapter, every ODE step evaluates the DiT for 3 branches (cond / uncond / no-emotion); base Sooktam-2
        uses 2. One "function evaluation" here = one ODE step.

mcd   : synthesise test sentences that have a REAL recording (same speaker, a different neutral clip as reference), for
        base Sooktam-2 and v2 (true rasa label), and compare with the real recording:
          * MCD-DTW on 13 MFCCs (c1-c13, DTW-aligned). NOTE: MFCC-based MCD; values are not directly comparable to
            SPTK mel-cepstral MCD numbers in papers, but are comparable across our conditions.
          * duration error vs the real recording
          * pitch variability: std of log-F0 in semitones (pYIN, voiced frames) -> direct test of "prosodic flattening"

    python -m sooktam_rasa.speed_mcd speed --data_dir ... --adapter ... --critic ... --duration_model ... --out_dir ...
    python -m sooktam_rasa.speed_mcd mcd   --data_dir ... --adapter ... --duration_model ... --out_dir ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from .data import load_manifest
from .rasa import BASIS

SR = 24000


# --------------------------------------------------------------------------- helpers
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def neutral_cases(data_dir, refs_per_lang, texts_per_lang, seed=0):
    rng = random.Random(seed)
    neutral = BASIS.index("shanta")
    test = [it for it in load_manifest(data_dir, split="test") if it["basis"] == neutral and it.get("wav")]
    cases = []
    for lang in sorted({it["lang"] for it in test}):
        pool = [it for it in test if it["lang"] == lang]
        spk = sorted({it["speaker"] for it in pool})
        refs = []
        for i in range(refs_per_lang):
            cand = [it for it in pool if it["speaker"] == spk[i % len(spk)] and it not in refs]
            refs.append(rng.choice(cand))
        texts = rng.sample([it for it in pool if it not in refs], texts_per_lang)
        cases += [(lang, r, t) for r in refs for t in texts]
    return cases


def write_rows(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_wav(path):
    import librosa
    y, _ = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=40)
    return y


def mfcc13(y):
    import librosa
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=14, n_fft=1024, hop_length=256, n_mels=80)[1:]  # drop c0 (energy)


def mcd_dtw(a, b):
    import librosa
    _, wp = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    wp = wp[::-1]
    diff = a[:, wp[:, 0]] - b[:, wp[:, 1]]
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt((diff ** 2).sum(0))))


def f0_semitone_std(y):
    import librosa
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=256)
    f0 = f0[voiced & np.isfinite(f0)] if f0 is not None else np.array([])
    if len(f0) < 10:
        return float("nan")
    st = 12 * np.log2(f0 / np.median(f0))
    return float(np.std(st))


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------- speed
def run_speed(a):
    from . import infer as I
    from .emo_classifier import load_classifier

    timer = {"ode": 0.0}
    orig = I.sample_with_emotion

    def timed(*args, **kw):
        _sync()
        t = time.perf_counter()
        out = orig(*args, **kw)
        _sync()
        timer["ode"] += time.perf_counter() - t
        return out

    I.sample_with_emotion = timed
    tts = I.EmotionalSooktam.load(a.sooktam_dir, adapter=a.adapter, duration_model=a.duration_model)
    critic = load_classifier(a.critic, tts.device) if a.critic else None
    cases = neutral_cases(a.data_dir, a.refs_per_lang, a.texts_per_lang, a.seed)
    rasas = a.rasas or BASIS
    dur = "learned" if a.duration_model else "grapheme"
    print(f"{len(cases)} pairs, rasas {rasas}, NFE {a.nfe}", flush=True)

    # warm-up (CUDA kernels, CLS tokenizer, reference cache)
    lang, ref, txt = cases[0]
    for _ in range(2):
        tts.synthesize(os.path.join(a.data_dir, ref["wav"]), ref["text"], txt["text"], lang, rasa="raudra", nfe_step=8, seed=0)

    rows, summary = [], {}
    conds = [(f"base_nfe{n}", None, n) for n in a.nfe] + [(f"nfe{n}", "label", n) for n in a.nfe]
    for cname, mode, nfe in conds:
        tot_t = tot_ode = tot_audio = 0.0
        hits = n_lab = 0
        for tgt in ([None] if mode is None else rasas):
            for lang, ref, txt in cases:
                path = os.path.join(a.out_dir, cname, f"{lang}_{tgt or 'base'}_{os.path.basename(ref['mel'])[:-3]}_"
                                    f"{os.path.basename(txt['mel'])[:-3]}.wav")
                timer["ode"] = 0.0
                _sync()
                t0 = time.perf_counter()
                wav, sr = tts.synthesize(os.path.join(a.data_dir, ref["wav"]), ref["text"], txt["text"], lang, rasa=tgt,
                                         cfg_emo=a.cfg_emo, nfe_step=nfe, duration=dur, seed=a.seed)
                _sync()
                dt = time.perf_counter() - t0
                secs = len(wav) / sr
                tot_t, tot_ode, tot_audio = tot_t + dt, tot_ode + timer["ode"], tot_audio + secs
                import soundfile as sf
                os.makedirs(os.path.dirname(path), exist_ok=True)
                sf.write(path, wav, sr)
                pred = ""
                if critic is not None and tgt is not None:
                    with torch.no_grad():
                        mel = tts.cfm.mel_spec(torch.from_numpy(wav).float()[None].to(tts.device)).permute(0, 2, 1)
                        pred = BASIS[int(critic(mel).argmax(-1))]
                    hits += int(pred == tgt)
                    n_lab += 1
                rows.append(dict(condition=cname, lang=lang, target=tgt or "none", pred=pred, nfe=nfe, seconds=secs,
                                 synth_time=dt, ode_time=timer["ode"], path=path, text=txt["text"],
                                 ref=os.path.join(a.data_dir, ref["wav"])))
        summary[cname] = dict(nfe=nfe, rtf_total=tot_t / tot_audio, rtf_ode=tot_ode / tot_audio,
                              ms_per_step=1000 * tot_ode / max(1, sum(1 for r in rows if r["condition"] == cname)) / nfe,
                              critic_acc=hits / n_lab if n_lab else None, audio_seconds=tot_audio)
        s = summary[cname]
        acc = f"{s['critic_acc']:.3f}" if s["critic_acc"] is not None else "  -  "
        print(f"{cname:<12} RTF total {s['rtf_total']:.4f}  RTF ODE {s['rtf_ode']:.4f}  "
              f"{s['ms_per_step']:.1f} ms/step  critic acc {acc}", flush=True)

    write_rows(rows, os.path.join(a.out_dir, "rows.csv"))
    json.dump(summary, open(os.path.join(a.out_dir, "speed_summary.json"), "w"), indent=2)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"\nGPU: {gpu}   (RTF < 1 = faster than real time; 1/RTF = x real-time)")
    print(f"{'condition':<12}{'NFE':>5}{'RTF total':>11}{'x realtime':>12}{'RTF ODE':>10}{'critic acc':>12}")
    for c, s in summary.items():
        acc = f"{s['critic_acc']:.3f}" if s["critic_acc"] is not None else "-"
        print(f"{c:<12}{s['nfe']:>5}{s['rtf_total']:>11.4f}{1 / s['rtf_total']:>11.1f}x{s['rtf_ode']:>10.4f}{acc:>12}")
    print("saved", os.path.join(a.out_dir, "speed_summary.json"), "and rows.csv (score with sooktam_rasa.score_control)")


# --------------------------------------------------------------------------- mcd
def run_mcd(a):
    from .infer import EmotionalSooktam
    import soundfile as sf

    rng = random.Random(a.seed)
    tts = EmotionalSooktam.load(a.sooktam_dir, adapter=a.adapter, duration_model=a.duration_model)
    items = [it for it in load_manifest(a.data_dir, split="test") if it.get("wav")]
    neutral = BASIS.index("shanta")
    neutral_by_spk = defaultdict(list)
    for it in items:
        if it["basis"] == neutral:
            neutral_by_spk[it["speaker"]].append(it)
    by = defaultdict(list)
    for it in items:
        by[(it["lang"], it["basis"])].append(it)
    targets = []
    for key in sorted(by):
        lst = by[key][:]
        rng.shuffle(lst)
        targets += lst[: a.per_class]
    print(f"{len(targets)} test sentences with real recordings", flush=True)

    dur = "learned" if a.duration_model else "grapheme"
    rows = []
    for k, tgt in enumerate(targets):
        pool = [r for r in neutral_by_spk[tgt["speaker"]] if r["id"] != tgt["id"]]
        if not pool:
            continue
        ref = rng.choice(pool)
        real_path = os.path.join(a.data_dir, tgt["wav"])
        real = load_wav(real_path)
        real_mf, real_f0 = mfcc13(real), f0_semitone_std(real)
        rasa = BASIS[tgt["basis"]]
        rows.append(dict(condition="real", lang=tgt["lang"], family=tgt["family"], rasa=rasa, mcd=0.0, dur_err_pct=0.0,
                         f0_std_st=real_f0, path=real_path, text=tgt["text"], target=rasa, ref=""))
        for cname, r_label in [("base", None), ("v2", rasa)]:
            path = os.path.join(a.out_dir, cname, f"{tgt['lang']}_{rasa}_{os.path.basename(tgt['mel'])[:-3]}.wav")
            wav, sr = tts.synthesize(os.path.join(a.data_dir, ref["wav"]), ref["text"], tgt["text"], tgt["lang"],
                                     rasa=r_label, cfg_emo=a.cfg_emo, nfe_step=a.nfe_step, duration=dur, seed=a.seed)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, wav, sr)
            g = load_wav(path)
            rows.append(dict(condition=cname, lang=tgt["lang"], family=tgt["family"], rasa=rasa,
                             mcd=mcd_dtw(mfcc13(g), real_mf), dur_err_pct=abs(len(g) - len(real)) / len(real) * 100,
                             f0_std_st=f0_semitone_std(g), path=path, text=tgt["text"], target=rasa,
                             ref=os.path.join(a.data_dir, ref["wav"])))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(targets)}", flush=True)

    write_rows(rows, os.path.join(a.out_dir, "mcd_rows.csv"))
    summary = {}
    print(f"\n{'condition':<10}{'MCD-DTW (dB)':>14}{'dur err %':>11}{'F0 std (st)':>13}{'n':>5}")
    for c in ["real", "base", "v2"]:
        rs = [r for r in rows if r["condition"] == c]
        summary[c] = dict(mcd=nanmean([r["mcd"] for r in rs]), dur_err_pct=nanmean([r["dur_err_pct"] for r in rs]),
                          f0_std_st=nanmean([r["f0_std_st"] for r in rs]), n=len(rs))
        s = summary[c]
        mcd = "-" if c == "real" else f"{s['mcd']:.2f}"
        de = "-" if c == "real" else f"{s['dur_err_pct']:.1f}"
        print(f"{c:<10}{mcd:>14}{de:>11}{s['f0_std_st']:>13.2f}{s['n']:>5}")
    print("\nPitch variability by rasa (F0 std, semitones) - prosodic flattening check:")
    print(f"{'rasa':<11}{'real':>7}{'base':>7}{'v2':>7}   |   MCD base / v2   (by family below)")
    for rasa in BASIS:
        vals = {c: nanmean([r["f0_std_st"] for r in rows if r["condition"] == c and r["rasa"] == rasa]) for c in ["real", "base", "v2"]}
        mc = {c: nanmean([r["mcd"] for r in rows if r["condition"] == c and r["rasa"] == rasa]) for c in ["base", "v2"]}
        summary.setdefault("by_rasa", {})[rasa] = dict(f0=vals, mcd=mc)
        print(f"{rasa:<11}{vals['real']:>7.2f}{vals['base']:>7.2f}{vals['v2']:>7.2f}   |   {mc['base']:.2f} / {mc['v2']:.2f}")
    for fam in sorted({r["family"] for r in rows}):
        mc = {c: nanmean([r["mcd"] for r in rows if r["condition"] == c and r["family"] == fam]) for c in ["base", "v2"]}
        summary.setdefault("by_family", {})[fam] = mc
        print(f"MCD {fam:<11} base {mc['base']:.2f}   v2 {mc['v2']:.2f}")
    json.dump(summary, open(os.path.join(a.out_dir, "mcd_summary.json"), "w"), indent=2)
    print("saved", os.path.join(a.out_dir, "mcd_summary.json"), "  (MFCC-based MCD; compare across our conditions only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["speed", "mcd"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--duration_model", default=None)
    ap.add_argument("--critic", default=None)
    ap.add_argument("--sooktam_dir", default="/content/sooktam2")
    ap.add_argument("--nfe", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--nfe_step", type=int, default=32, help="mcd mode")
    ap.add_argument("--rasas", nargs="*", default=None)
    ap.add_argument("--refs_per_lang", type=int, default=2)
    ap.add_argument("--texts_per_lang", type=int, default=3)
    ap.add_argument("--per_class", type=int, default=5, help="mcd: test sentences per language x rasa")
    ap.add_argument("--cfg_emo", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_speed(a) if a.mode == "speed" else run_mcd(a)


if __name__ == "__main__":
    main()
