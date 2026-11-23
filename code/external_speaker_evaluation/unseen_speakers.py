"""Prepare, synthesize and score the Sooktam-2 external-speaker evaluation."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import mean, quantiles
import time
import unicodedata


DATASET = "ai4bharat/indicvoices_r"
TEXTS = {
    "hindi": ("कमरे की खिड़की खुली है।", "मेज़ पर एक किताब रखी है।",
              "अगली बस दस मिनट में आएगी।"),
    "tamil": ("அறையின் சன்னல் திறந்திருக்கிறது.", "மேசையின் மீது ஒரு புத்தகம் இருக்கிறது.",
              "அடுத்த பேருந்து பத்து நிமிடங்களில் வரும்."),
}
RASAS = ("shanta", "hasya", "karuna", "raudra")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text):
    """Keep Indic vowel signs and other combining marks in the CER input."""
    return " ".join("".join(c for c in unicodedata.normalize("NFC", text).casefold()
                            if unicodedata.category(c)[0] in "LMN" or c.isspace()).split())


def prepare(args):
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    import numpy as np
    import soundfile as sf

    if (args.panel / "panel.json").exists():
        raise SystemExit("This panel already exists. Use a new directory to select a different panel.")
    with args.training_manifest.open(encoding="utf-8") as stream:
        seen = {row["speaker"] for line in stream if line.strip()
                for row in [json.loads(line)] if row["split"] != "test"}
    if not seen:
        raise SystemExit("No training speaker IDs found in the Rasa manifest.")
    token = os.environ.get("HF_TOKEN") or True
    revision = HfApi().dataset_info(DATASET, revision=args.revision, token=token).sha
    (args.panel / "audio").mkdir(parents=True, exist_ok=True)
    selected, used = [], set(seen)
    for lang in TEXTS:
        data = load_dataset(DATASET, lang.title(), split="test", streaming=True,
                            revision=revision, token=token).cast_column("audio", Audio(decode=False))
        data = data.shuffle(seed=args.seed, buffer_size=128)
        candidates = {}
        quotas = {"female": args.per_language // 2, "male": args.per_language // 2}
        for row in data:
            gender = str(row.get("gender", "")).strip().lower()
            gender = {"f": "female", "m": "male"}.get(gender, gender)
            speaker = str(row.get("speaker_id") or "").strip()
            transcript = str(row.get("verbatim") or row.get("text") or "").strip()
            script = (0x0900, 0x097F) if lang == "hindi" else (0x0B80, 0x0BFF)
            if (not quotas.get(gender, 0) or not speaker or speaker in used
                    or not any(script[0] <= ord(c) <= script[1] for c in transcript)
                    or not 3 <= float(row.get("duration") or 0) <= 10):
                continue
            audio = row["audio"]
            source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
            wav, sr = sf.read(source, always_2d=True, dtype="float32")
            mono = wav.mean(axis=1)
            if not 3 <= len(mono) / sr <= 10 or not np.isfinite(mono).all() or np.max(np.abs(mono)) < .001:
                continue
            clips = candidates.setdefault((speaker, gender), [])
            if clips and normalized(transcript) == normalized(clips[0][2]):
                continue
            clips.append((mono, sr, transcript, audio.get("path"), len(mono) / sr))
            if len(clips) != 2:
                continue
            used.add(speaker)
            quotas[gender] -= 1
            panel_id = f"{lang}-{1 + sum(r['lang'] == lang for r in selected):02d}"
            entry = dict(id=panel_id, speaker_id=speaker, lang=lang, gender=gender)
            for label, (samples, rate, text, original, seconds) in zip(("ref", "enrollment"), clips):
                path = f"audio/{panel_id}-{label}.wav"
                sf.write(args.panel / path, samples, rate, subtype="PCM_16")
                entry[label] = dict(path=path, text=text, source_file=original, seconds=seconds)
            selected.append(entry)
            print(f"Selected {panel_id} ({gender})", flush=True)
            if not any(quotas.values()):
                break
        if any(quotas.values()):
            raise SystemExit(f"Insufficient eligible {lang} test speakers. Remaining quotas: {quotas}")
    write_json(args.panel / "panel.json", dict(dataset=DATASET, revision=revision, split="test",
               seed=args.seed, training_speaker_groups=sorted(seen), speakers=selected))


def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    from sooktam_rasa.infer import EmotionalSooktam

    panel = read_json(args.panel / "panel.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Select a GPU runtime in Colab before synthesis.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("PyTorch cannot access the Mac GPU through MPS on this machine.")
    synchronize = {"cuda": lambda: torch.cuda.synchronize(), "mps": lambda: torch.mps.synchronize(),
                   "cpu": lambda: None}[args.device]
    checkpoints = {"adapter": args.adapter, "duration": args.duration_model,
                   "backbone": args.sooktam_dir / "sooktam.safetensors"}
    for path in checkpoints.values():
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "clips").mkdir(exist_ok=True)
    panel_hash = hashlib.sha256((args.panel / "panel.json").read_bytes()).hexdigest()
    settings = dict(panel_sha256=panel_hash, steps=args.steps, seed=args.seed, device=args.device,
                    hardware=torch.cuda.get_device_name() if args.device == "cuda" else args.device,
                    checkpoints={name: file_sha256(path) for name, path in checkpoints.items()},
                    strong_guidance=args.strong_guidance, sentences=args.sentences)
    settings_path = args.out / "settings.json"
    if settings_path.exists():
        previous = read_json(settings_path)
        previous.setdefault("sentences", 3)
        if previous != settings:
            raise SystemExit("Existing run has different settings. Choose a new --out directory.")
    write_json(settings_path, settings)
    progress = args.out / "progress.jsonl"
    rows = []
    if progress.exists():
        rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {r["clip_id"]: r for r in rows if (args.out / r["path"]).is_file()}
    done = set(completed)
    rows = list(completed.values())
    conditions = [("base", None, 0.0)] + [("v2_cfg1", rasa, 1.0) for rasa in RASAS]
    if args.strong_guidance:
        conditions += [("v2_cfg3", rasa, 3.0) for rasa in RASAS]
    expected = len(panel["speakers"]) * args.sentences * len(conditions)
    if len(done) == expected:
        write_csv(args.out / "rows.csv", rows)
        return
    model = EmotionalSooktam.load(str(args.sooktam_dir), adapter=str(args.adapter),
                                  duration_model=str(args.duration_model), device=args.device)

    def synthesize(speaker, sentence, rasa, guidance):
        ref = speaker["ref"]
        return model.synthesize(str(args.panel / ref["path"]), ref["text"], sentence, speaker["lang"],
                                rasa=rasa, cfg_emo=guidance, nfe_step=args.steps,
                                duration="learned", seed=args.seed)

    first = panel["speakers"][0]
    for _, rasa, guidance in (conditions[0], conditions[1]):
        synthesize(first, TEXTS[first["lang"]][0], rasa, guidance)
    with progress.open("a", encoding="utf-8") as log:
        for speaker in panel["speakers"]:
            for index, sentence in enumerate(TEXTS[speaker["lang"]][:args.sentences], 1):
                for condition, rasa, guidance in conditions:
                    clip_id = f"{speaker['id']}-{index}-{condition}-{rasa or 'unconditioned'}"
                    if clip_id in done:
                        continue
                    synchronize()
                    start = time.perf_counter()
                    wav, sr = synthesize(speaker, sentence, rasa, guidance)
                    synchronize()
                    elapsed = time.perf_counter() - start
                    if not len(wav) or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
                        raise RuntimeError(f"Invalid or silent synthesis: {clip_id}")
                    path = f"clips/{clip_id}.wav"
                    sf.write(args.out / path, wav, sr)
                    row = dict(clip_id=clip_id, id=speaker["id"], lang=speaker["lang"], sentence=index,
                               text=sentence, condition=condition, target=rasa or "", path=path,
                               audio_seconds=len(wav) / sr, synth_seconds=elapsed, rtf=elapsed * sr / len(wav))
                    rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
            print(f"{len(rows)}/{expected} clips ready", flush=True)
    write_csv(args.out / "rows.csv", rows)


def listening(args):
    with (args.out / "rows.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rng = random.Random(2026)
    cases = set()
    for lang in TEXTS:
        ids = sorted({r["id"] for r in rows if r["lang"] == lang})
        for speaker in rng.sample(ids, min(3, len(ids))):
            sentences = sorted({r["sentence"] for r in rows if r["id"] == speaker})
            cases.add((speaker, rng.choice(sentences)))
    subset = [r for r in rows if (r["id"], r["sentence"]) in cases and r["condition"] in ("base", "v2_cfg1")]
    rng.shuffle(subset)
    directory = args.out / "listening"
    directory.mkdir(exist_ok=False)
    sheet, key = [], []
    for i, row in enumerate(subset, 1):
        clip = f"sample-{i:03d}.wav"
        shutil.copyfile(args.out / row["path"], directory / clip)
        sheet.append(dict(clip=clip, language=row["lang"], naturalness_1_to_5="", perceived_emotion=""))
        key.append(dict(clip=clip, original_clip=row["clip_id"], condition=row["condition"], target=row["target"]))
    write_csv(directory / "ratings.csv", sheet)
    write_csv(args.out / "listening_key.csv", key)


def summary(rows, speakers, seed):
    """Mean of speaker means, with language-stratified speaker bootstrap."""
    rng = random.Random(seed)
    by_language = [[s["id"] for s in speakers if s["lang"] == lang] for lang in TEXTS]
    draws = [[speaker for ids in by_language if ids for speaker in rng.choices(ids, k=len(ids))]
             for _ in range(4000)]

    def estimate(values):
        cutoffs = quantiles([mean(values[i] for i in ids) for ids in draws], n=40, method="inclusive")
        return dict(mean=mean(values.values()), ci95=[cutoffs[0], cutoffs[-1]])

    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        selected = [r for r in rows if r["condition"] == condition]
        out[condition] = dict(clips=len(selected), speakers=len(speakers))
        for metric in ("speaker_similarity", "cer", "rtf", "emotion_hit"):
            if metric not in selected[0] or (metric == "emotion_hit" and condition == "base"):
                continue
            values = {s["id"]: mean(r[metric] for r in selected if r["id"] == s["id"]) for s in speakers}
            out[condition][metric] = estimate(values)
            if condition != "base" and metric != "emotion_hit":
                baseline = [r for r in rows if r["condition"] == "base"]
                delta = {i: v - mean(r[metric] for r in baseline if r["id"] == i) for i, v in values.items()}
                out[condition][metric + "_delta_vs_base"] = estimate(delta)
    return out


def score(args):
    import torch
    from sooktam_rasa.evaluate import SpeakerSim

    panel_path = args.panel / "panel.json"
    settings = read_json(args.out / "settings.json")
    if hashlib.sha256(panel_path.read_bytes()).hexdigest() != settings["panel_sha256"]:
        raise SystemExit("The speaker panel differs from the one used for synthesis.")
    speakers = read_json(panel_path)["speakers"]
    with (args.out / "rows.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(speakers) * settings.get("sentences", 3) * (9 if settings["strong_guidance"] else 5)
    if len({r["clip_id"] for r in rows}) != expected or len(rows) != expected:
        raise SystemExit("Synthesis is incomplete. Finish the run before scoring.")
    cosine = torch.nn.functional.cosine_similarity
    ecapa = SpeakerSim(args.device)
    with torch.inference_mode():
        embeddings = {s["id"]: ecapa._emb(str(args.panel / s["enrollment"]["path"])) for s in speakers}
        real_similarity = {s["id"]: cosine(ecapa._emb(str(args.panel / s["ref"]["path"])),
                                            embeddings[s["id"]], dim=0).item() for s in speakers}
        for row in rows:
            row["speaker_similarity"] = cosine(ecapa._emb(str(args.out / row["path"])),
                                                embeddings[row["id"]], dim=0).item()
            row["rtf"] = float(row["rtf"])
    del ecapa, embeddings
    if args.device == "cuda":
        torch.cuda.empty_cache()
    if not args.speaker_only:
        score_content(rows, args.out, args.device)
    for row in rows:
        if not all(math.isfinite(row[key]) for key in ("speaker_similarity", "rtf", "cer") if key in row):
            raise RuntimeError(f"Non-finite metric: {row['clip_id']}")
    suffix = "_speaker" if args.speaker_only else ""
    write_csv(args.out / f"rows_scored{suffix}.csv", rows)
    metrics = summary(rows, speakers, settings["seed"])
    write_json(args.out / f"summary{suffix}.json", dict(conditions=metrics,
               real_vs_real_similarity=dict(mean=mean(real_similarity.values()), by_speaker=real_similarity)))
    print(json.dumps(metrics, indent=2))


def score_content(rows, out, device):
    import jiwer
    import librosa
    import torch
    from sooktam_rasa.evaluate import CER, WHISPER_CODE
    from sooktam_rasa.score_control import Emotion2Vec

    asr = CER(device)
    for row in rows:
        audio, _ = librosa.load(out / row["path"], sr=16000)
        lang = WHISPER_CODE[row["lang"]]
        if asr.backend == "faster":
            segments, _ = asr.m.transcribe(audio, language=lang, beam_size=5)
            hyp = "".join(segment.text for segment in segments)
        else:
            hyp = asr.m({"raw": audio, "sampling_rate": 16000},
                        generate_kwargs={"language": lang, "task": "transcribe"})["text"]
        row["asr_text"] = hyp.strip()
        row["cer"] = jiwer.cer(normalized(row["text"]), normalized(hyp))
    del asr
    if device == "cuda":
        torch.cuda.empty_cache()
    emotion = Emotion2Vec(device)
    for row in rows:
        row["emotion_prediction"] = emotion(str(out / row["path"])) or "unknown"
        row["emotion_hit"] = int(row["emotion_prediction"] == row["target"]) if row["target"] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--panel", type=Path, required=True)
    prep.add_argument("--training-manifest", type=Path, required=True)
    prep.add_argument("--per-language", type=int, default=12)
    prep.add_argument("--revision", default="main")
    prep.add_argument("--seed", type=int, default=2026)
    gen = actions.add_parser("run")
    scoring = actions.add_parser("score")
    listen = actions.add_parser("listening")
    listen.add_argument("--out", type=Path, required=True)
    for action in (gen, scoring):
        action.add_argument("--panel", type=Path, required=True)
        action.add_argument("--out", type=Path, required=True)
    gen.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    scoring.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    gen.add_argument("--sooktam-dir", type=Path, required=True)
    gen.add_argument("--adapter", type=Path, required=True)
    gen.add_argument("--duration-model", type=Path, required=True)
    gen.add_argument("--steps", type=int, default=32)
    gen.add_argument("--sentences", type=int, choices=(1, 2, 3), default=3)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--strong-guidance", action="store_true")
    scoring.add_argument("--speaker-only", action="store_true", help="Skip Whisper and emotion2vec.")
    args = parser.parse_args()
    if args.action == "prepare" and (args.per_language < 2 or args.per_language % 2):
        parser.error("--per-language must be a positive even number")
    if args.action == "run" and args.steps < 2:
        parser.error("--steps must be at least 2")
    {"prepare": prepare, "run": run, "score": score, "listening": listening}[args.action](args)


if __name__ == "__main__":
    main()
