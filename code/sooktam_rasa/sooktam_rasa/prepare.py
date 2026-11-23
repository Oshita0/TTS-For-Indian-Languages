"""
Build the training cache from AI4Bharat Rasa.

    HF_TOKEN=hf_xxx python -m sooktam_rasa.prepare --languages Hindi Tamil \
        --data_dir /content/drive/MyDrive/sooktam_rasa/data --neutral_hours 2 --inspect

--inspect only prints the style table (cheap: metadata columns only) and exits.
Re-running is safe: finished utterances are skipped.
"""
from __future__ import annotations

import argparse
import os

from .data import extract, inspect_styles, scan_metadata, select_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", required=True, help="Rasa config names, e.g. Hindi Tamil Bengali")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--neutral_hours", type=float, default=2.0, help="neutral (shanta) hours per speaker, train split")
    ap.add_argument("--emotion_hours", type=float, default=None, help="cap per speaker x emotion (default: all)")
    ap.add_argument("--test_neutral_hours", type=float, default=0.3)
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=10.0)
    ap.add_argument("--include_other_styles", action="store_true", help="keep conversation/command styles as shanta")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--device", default="cpu", help="mel extraction device (cpu is fine)")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    meta_dir = os.path.join(a.data_dir, "meta")
    for lang in a.languages:
        for split in a.splits:
            df = scan_metadata(lang, split, token, cache_dir=meta_dir)
            print(f"\n=== {lang} / {split}: {len(df)} rows, {df['duration'].sum() / 3600:.1f} h ===")
            inspect_styles(df, a.include_other_styles)
            if a.inspect:
                continue
            nh = a.neutral_hours if split == "train" else a.test_neutral_hours
            sel = select_rows(df, nh, a.emotion_hours, a.min_dur, a.max_dur, a.include_other_styles)
            extract(sel, lang, a.data_dir, token, split=split, save_wav=(split == "test"), device=a.device)


if __name__ == "__main__":
    main()
