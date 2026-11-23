"""
AI4Bharat Rasa -> cached log-mels + CLS tokens for Sooktam-2 fine-tuning.

The full dataset is ~1,145 h of 48 kHz audio (8-37 GB per language), far too
much for Colab. So we:
  1. scan only the small metadata columns of every parquet shard (cheap),
  2. choose rows (all emotional speech + a capped amount of neutral per speaker),
  3. download audio only for parquet row-groups that contain chosen rows,
  4. resample to 24 kHz, trim edge silence, compute Sooktam's exact log-mel
     (vocos, 100 bins, hop 256), and store fp16 mels (~70 MB per hour).

Dataset columns (from the HF card): filename, text, language, gender, style,
duration, wav_path, audio.  One config per language ("Hindi", "Tamil", ...).
"""
from __future__ import annotations

import io
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch

from .rasa import BASIS, LANG_FAMILY, LANG_TO_ID, NUM_BASIS, SOOKTAM_LANG, UNK_LANG_ID, style_to_basis

REPO = "datasets/ai4bharat/Rasa"
META_COLS = ["filename", "text", "language", "gender", "style", "duration"]
SR = 24000
HOP = 256


# --------------------------------------------------------------------------- #
# 1. metadata scan
# --------------------------------------------------------------------------- #
def _fs(token):
    from huggingface_hub import HfFileSystem
    return HfFileSystem(token=token)


def scan_metadata(lang: str, split: str = "train", token: str | None = None, cache_dir: str | None = None):
    """Return a pandas DataFrame of metadata + (shard, row_group, row) locations. Cached to CSV."""
    import pandas as pd
    import pyarrow.parquet as pq

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = os.path.join(cache_dir, f"meta_{lang}_{split}.csv")
        if os.path.exists(cpath):
            return pd.read_csv(cpath, keep_default_na=False)
    fs = _fs(token)
    shards = sorted(fs.glob(f"{REPO}/{lang}/{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards for {lang}/{split}. Did you accept the dataset terms and pass an HF token?")
    rows = []
    for si, path in enumerate(shards):
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            cols = [c for c in META_COLS if c in pf.schema_arrow.names]
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=cols).to_pylist()
                for ri, r in enumerate(t):
                    r.update(shard=path, row_group=rg, row=ri)
                    rows.append(r)
        print(f"[{lang}/{split}] scanned shard {si + 1}/{len(shards)}  rows so far: {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    if cache_dir:
        df.to_csv(cpath, index=False)
    return df


def inspect_styles(df, include_other_styles=False):
    """Print hours per raw `style` value and what it maps to. Check this before training."""
    g = df.groupby("style")["duration"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    print(f"{'style':<28}{'utts':>8}{'hours':>8}   -> basis")
    for style, r in g.iterrows():
        b = style_to_basis(style, include_other_styles)
        print(f"{str(style):<28}{int(r['count']):>8}{r['sum'] / 3600:>8.2f}   -> {BASIS[b] if b is not None else 'SKIP'}")
    return g


def select_rows(df, neutral_hours_per_speaker=2.0, max_hours_per_emotion_speaker=None,
                min_dur=1.0, max_dur=10.0, include_other_styles=False, seed=0):
    df = df.copy()
    df["basis"] = [style_to_basis(s, include_other_styles) for s in df["style"]]
    df = df[df["basis"].notna() & (df["duration"] >= min_dur) & (df["duration"] <= max_dur)]
    df = df[df["text"].str.strip().str.len() > 0]
    df["basis"] = df["basis"].astype(int)
    df = df.sample(frac=1.0, random_state=seed)
    keep, used = [], defaultdict(float)
    neutral = BASIS.index("shanta")
    for idx, r in df.iterrows():
        key = (r["gender"], r["basis"])
        cap = neutral_hours_per_speaker if r["basis"] == neutral else max_hours_per_emotion_speaker
        if cap is not None and used[key] + r["duration"] > cap * 3600:
            continue
        used[key] += r["duration"]
        keep.append(idx)
    sel = df.loc[keep]
    summ = sel.groupby(["gender", "basis"])["duration"].sum().unstack(fill_value=0) / 3600
    summ.columns = [BASIS[c] for c in summ.columns]
    print("Selected hours per speaker x rasa:\n", summ.round(2))
    return sel


# --------------------------------------------------------------------------- #
# 2. audio -> mel, text -> CLS tokens
# --------------------------------------------------------------------------- #
class Featurizer:
    def __init__(self, device="cpu"):
        from f5_tts.model.modules import MelSpec
        self.mel = MelSpec(n_fft=1024, hop_length=HOP, win_length=1024, n_mel_channels=100,
                           target_sample_rate=SR, mel_spec_type="vocos").to(device)
        self.device = device

    def wav_from_bytes(self, b: bytes, top_db=45):
        import librosa
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        wav, _ = librosa.effects.trim(wav, top_db=top_db)
        wav = np.concatenate([wav, np.zeros(int(0.05 * SR), dtype=np.float32)])  # +50 ms, like Sooktam's ref preprocessing
        return wav

    @torch.no_grad()
    def mel_from_wav(self, wav: np.ndarray):
        x = torch.from_numpy(wav).float()[None].to(self.device)
        return self.mel(x)[0].T.contiguous().cpu()  # (T, 100)


def normalize_text(text: str) -> str:
    """Mirror Sooktam's preprocess_ref_audio_text: text always ends with '. '"""
    text = " ".join(text.strip().split())
    if text.endswith("."):
        return text + " "
    if text and text[-1] in "।॥?!":
        return text + " "
    return text + ". "


def cls_tokens(text: str, language: str):
    from f5_tts.infer.utils_infer import tokenize_texts
    return tokenize_texts([text], tokenizer="cls", cls_language=language)[0]


def extract(sel, lang: str, out_dir: str, token: str | None, split="train", save_wav=False, device="cpu", log_every=200):
    """Download chosen rows, write mels (+ optional 24k wavs) and append to manifest.jsonl (resumable)."""
    import pyarrow.parquet as pq
    import soundfile as sf

    s_lang = SOOKTAM_LANG[lang]
    feat = Featurizer(device)
    man_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(os.path.join(out_dir, "mels", lang), exist_ok=True)
    if save_wav:
        os.makedirs(os.path.join(out_dir, "wavs", lang), exist_ok=True)
    done = set()
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f}
    fs = _fs(token)
    n_new, oov = 0, Counter()
    with open(man_path, "a", encoding="utf-8") as man:
        for (shard, rg), grp in sel.groupby(["shard", "row_group"]):
            todo = [r for _, r in grp.iterrows() if f"{lang}/{split}/{r['filename']}" not in done]
            if not todo:
                continue
            with fs.open(shard, "rb") as fh:
                audio_col = pq.ParquetFile(fh).read_row_group(int(rg), columns=["audio"]).column("audio")
            for r in todo:
                uid = f"{lang}/{split}/{r['filename']}"
                try:
                    a = audio_col[int(r["row"])].as_py()
                    wav = feat.wav_from_bytes(a["bytes"])
                    mel = feat.mel_from_wav(wav)
                    text = normalize_text(r["text"])
                    toks = cls_tokens(text, s_lang)
                except Exception as e:  # noqa: BLE001
                    print("skip", uid, repr(e)[:200])
                    continue
                stem = os.path.splitext(os.path.basename(str(r["filename"])))[0]
                mel_rel = f"mels/{lang}/{split}_{stem}.pt"
                torch.save(mel.half(), os.path.join(out_dir, mel_rel))
                wav_rel = None
                if save_wav:
                    wav_rel = f"wavs/{lang}/{split}_{stem}.wav"
                    sf.write(os.path.join(out_dir, wav_rel), wav, SR)
                import grapheme
                item = dict(id=uid, split=split, lang=s_lang, family=LANG_FAMILY[s_lang],
                            speaker=f"{s_lang}_{str(r['gender']).lower()}", style=str(r["style"]),
                            basis=int(r["basis"]), text=text, tokens=toks, n_frames=int(mel.shape[0]),
                            n_graphemes=grapheme.length(text), n_bytes=len(text.encode("utf-8")),
                            mel=mel_rel, wav=wav_rel)
                man.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_new += 1
                if n_new % log_every == 0:
                    man.flush()
                    print(f"[{lang}/{split}] extracted {n_new}", flush=True)
    print(f"[{lang}/{split}] done, {n_new} new items -> {man_path}")
    return n_new


def load_manifest(data_dir, split=None, langs=None):
    items = []
    with open(os.path.join(data_dir, "manifest.jsonl"), encoding="utf-8") as f:
        for l in f:
            it = json.loads(l)
            if split and it["split"] != split:
                continue
            if langs and it["lang"] not in langs:
                continue
            items.append(it)
    return items


def token_ids(tokens, vocab_map):
    return [vocab_map.get(t, 0) for t in tokens]


def oov_report(items, vocab_map):
    c, tot = Counter(), 0
    for it in items:
        for t in it["tokens"]:
            tot += 1
            if t not in vocab_map:
                c[t] += 1
    rate = sum(c.values()) / max(tot, 1)
    print(f"OOV token rate: {rate:.4%}  most common: {c.most_common(10)}")
    return rate


# --------------------------------------------------------------------------- #
# 3. training datasets
# --------------------------------------------------------------------------- #
class EmotionTrainSet(torch.utils.data.Dataset):
    """
    With prob p_pair:  [neutral prompt of SAME speaker] + [target utterance], loss only on the target.
                       This is exactly the inference situation (neutral reference clip + rasa label),
                       so the model learns that the label, not the prompt, decides the emotion.
    Otherwise:         single utterance with F5-style random infill span (70-100%).
    """

    def __init__(self, items, data_dir, vocab_map, p_pair=0.7, max_ref_frames=450, max_total_frames=1400,
                 span=(0.7, 1.0), seed=0):
        self.items, self.dir, self.vmap = items, data_dir, vocab_map
        self.p_pair, self.max_ref, self.max_total, self.span = p_pair, max_ref_frames, max_total_frames, span
        self.rng = random  # global RNG: PyTorch reseeds it per DataLoader worker
        neutral = BASIS.index("shanta")
        self.neutral_by_spk = defaultdict(list)
        for i, it in enumerate(items):
            if it["basis"] == neutral and it["n_frames"] <= max_ref_frames:
                self.neutral_by_spk[it["speaker"]].append(i)

    def __len__(self):
        return len(self.items)

    def _mel(self, it):
        return torch.load(os.path.join(self.dir, it["mel"])).float()

    def cost(self, i):
        return self.items[i]["n_frames"] + (self.max_ref // 2 if self.p_pair > 0 else 0)

    def __getitem__(self, i):
        tgt = self.items[i]
        mel_t, ids_t = self._mel(tgt), token_ids(tgt["tokens"], self.vmap)
        pool = [j for j in self.neutral_by_spk.get(tgt["speaker"], []) if j != i]
        if pool and self.rng.random() < self.p_pair:
            ref = self.items[self.rng.choice(pool)]
            mel_r = self._mel(ref)
            room = self.max_total - mel_r.shape[0]
            mel_t = mel_t[:room]
            mel = torch.cat([mel_r, mel_t])
            ids = token_ids(ref["tokens"], self.vmap) + ids_t
            lm = torch.zeros(mel.shape[0], dtype=torch.bool)
            lm[mel_r.shape[0]:] = True
        else:
            mel = mel_t[: self.max_total]
            ids = ids_t
            n = mel.shape[0]
            span_len = int(n * self.rng.uniform(*self.span))
            start = self.rng.randint(0, n - span_len)
            lm = torch.zeros(n, dtype=torch.bool)
            lm[start:start + span_len] = True
        return dict(mel=mel, ids=torch.tensor(ids, dtype=torch.long), loss_mask=lm, basis=tgt["basis"])


def collate_emotion(batch, emo_drop_prob=0.0):
    from torch.nn.utils.rnn import pad_sequence
    lens = torch.tensor([b["mel"].shape[0] for b in batch])
    mel = pad_sequence([b["mel"] for b in batch], batch_first=True)
    ids = pad_sequence([b["ids"] for b in batch], batch_first=True, padding_value=-1)
    lm = pad_sequence([b["loss_mask"] for b in batch], batch_first=True)
    basis = torch.tensor([b["basis"] for b in batch])
    w = torch.nn.functional.one_hot(basis, NUM_BASIS).float()
    null = torch.rand(len(batch)) < emo_drop_prob
    return dict(mel=mel, text=ids, lens=lens, loss_mask=lm, emo_weights=w, emo_null=null, basis=basis)


class FrameBudgetSampler(torch.utils.data.Sampler):
    """Emotion-balanced sampling (w ~ count^-balance_power) packed into batches of <= frames_per_batch."""

    def __init__(self, dataset, frames_per_batch=6000, batches_per_epoch=1000, balance_power=0.5, max_batch=32, seed=0):
        self.ds, self.fpb, self.n, self.max_batch = dataset, frames_per_batch, batches_per_epoch, max_batch
        cnt = Counter(it["basis"] for it in dataset.items)
        self.w = torch.tensor([cnt[it["basis"]] ** -balance_power for it in dataset.items], dtype=torch.double)
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        for _ in range(self.n):
            batch, frames = [], 0
            while len(batch) < self.max_batch:
                i = int(torch.multinomial(self.w, 1, generator=self.g))
                c = self.ds.cost(i)
                if batch and frames + c > self.fpb:
                    break
                batch.append(i)
                frames += c
            yield batch


class DurationPairSet(torch.utils.data.Dataset):
    """(reference utterance, target text) -> target frame count, same speaker, different utterance."""

    def __init__(self, items, data_dir, vocab_map, p_neutral_ref=0.7, max_ref_frames=600, seed=0, fixed=False):
        self.items, self.dir, self.vmap = items, data_dir, vocab_map
        self.p_neutral, self.max_ref, self.fixed = p_neutral_ref, max_ref_frames, fixed
        self.rng = random  # global RNG: PyTorch reseeds it per DataLoader worker
        self.by_spk, self.neutral_by_spk = defaultdict(list), defaultdict(list)
        neutral = BASIS.index("shanta")
        for i, it in enumerate(items):
            if it["n_frames"] <= max_ref_frames:
                self.by_spk[it["speaker"]].append(i)
                if it["basis"] == neutral:
                    self.neutral_by_spk[it["speaker"]].append(i)
        # fixed pairs for validation (reproducible)
        self.pairs = [self._pick(i, random.Random(seed + i)) for i in range(len(items))] if fixed else None

    def _pick(self, i, rng):
        spk = self.items[i]["speaker"]
        pool = self.neutral_by_spk[spk] if (self.neutral_by_spk[spk] and rng.random() < self.p_neutral) else self.by_spk[spk]
        pool = [j for j in pool if j != i] or [i]
        return rng.choice(pool)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tgt = self.items[i]
        j = self.pairs[i] if self.fixed else self._pick(i, self.rng)
        ref = self.items[j]
        return dict(ref_mel=torch.load(os.path.join(self.dir, ref["mel"])).float(),
                    ref_ids=torch.tensor(token_ids(ref["tokens"], self.vmap)) + 1,
                    tgt_ids=torch.tensor(token_ids(tgt["tokens"], self.vmap)) + 1,
                    tgt_frames=float(tgt["n_frames"]),
                    lang=LANG_TO_ID.get(tgt["lang"], UNK_LANG_ID), basis=tgt["basis"],
                    family=tgt["family"], lang_name=tgt["lang"],
                    # baseline formula inputs
                    ref_frames=ref["n_frames"], ref_bytes=ref["n_bytes"], tgt_bytes=tgt["n_bytes"],
                    ref_graph=ref["n_graphemes"], tgt_graph=tgt["n_graphemes"])


def collate_duration(batch):
    from torch.nn.utils.rnn import pad_sequence
    out = dict(
        ref_mel=pad_sequence([b["ref_mel"] for b in batch], batch_first=True),
        ref_mel_lens=torch.tensor([b["ref_mel"].shape[0] for b in batch]),
        ref_ids=pad_sequence([b["ref_ids"] for b in batch], batch_first=True, padding_value=0),
        tgt_ids=pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=0),
        tgt_frames=torch.tensor([b["tgt_frames"] for b in batch]),
        lang=torch.tensor([b["lang"] for b in batch]),
        emo_weights=torch.nn.functional.one_hot(torch.tensor([b["basis"] for b in batch]), NUM_BASIS).float(),
    )
    for k in ["ref_frames", "ref_bytes", "tgt_bytes", "ref_graph", "tgt_graph"]:
        out[k] = torch.tensor([float(b[k]) for b in batch])
    out["family"] = [b["family"] for b in batch]
    out["lang_name"] = [b["lang_name"] for b in batch]
    return out
