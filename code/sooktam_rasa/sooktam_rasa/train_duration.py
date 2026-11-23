"""
Train the cross-attention duration predictor on cached Rasa mels (no Sooktam forward pass needed,
so this runs in minutes even on a free T4).

    python -m sooktam_rasa.train_duration --set data_dir='"/content/drive/MyDrive/sooktam_rasa/data"'

Reports mean absolute % error per language and per family against the byte formula (original
Sooktam-2) and the grapheme formula (your fix), so the result plugs straight into the bug report.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import torch

from .data import DurationPairSet, collate_duration, load_manifest
from .duration import CrossAttnDurationPredictor, abs_pct_err, heuristic_frames
from .train_emotion import split_items

DEFAULTS = dict(
    sooktam_dir="/content/sooktam2", data_dir="/content/drive/MyDrive/sooktam_rasa/data",
    out_dir="/content/drive/MyDrive/sooktam_rasa/ckpt/duration", languages=None,
    d=256, heads=4, text_layers=3, mel_layers=2, xattn_layers=2, dropout=0.1,
    lr=3e-4, batch_size=32, max_steps=15000, val_every=500, tau_reg=0.01, seed=0, num_workers=2, val_frac=0.03,
)


def vocab_map_from(sooktam_dir):
    from f5_tts.model.utils import get_tokenizer  # exact same parsing Sooktam uses at inference
    vmap, _ = get_tokenizer(os.path.join(sooktam_dir, "vocab.txt"), "custom")
    return vmap


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = defaultdict(lambda: defaultdict(list))
    for b in loader:
        bb = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        pred, _, _ = model(bb["ref_mel"], bb["ref_mel_lens"], bb["ref_ids"], bb["tgt_ids"], bb["lang"], bb["emo_weights"])
        true = b["tgt_frames"]
        ests = {
            "learned": pred.cpu(),
            "bytes": heuristic_frames(b["ref_frames"], b["ref_bytes"], b["tgt_bytes"]),
            "grapheme": heuristic_frames(b["ref_frames"], b["ref_graph"], b["tgt_graph"]),
            "cls_tokens": heuristic_frames(b["ref_frames"], (b["ref_ids"] > 0).sum(1).float(), (b["tgt_ids"] > 0).sum(1).float()),
        }
        for name, est in ests.items():
            e = abs_pct_err(est, true)
            for i in range(len(true)):
                for key in ("ALL", b["family"][i], b["lang_name"][i]):
                    rows[key][name].append(e[i].item())
    model.train()
    return {k: {m: sum(v) / len(v) for m, v in d.items()} for k, d in rows.items()}


def print_table(res):
    methods = ["bytes", "grapheme", "cls_tokens", "learned"]
    print(f"{'group':<14}" + "".join(f"{m:>12}" for m in methods) + "   (mean abs % error, same-speaker pairs)")
    for k in sorted(res, key=lambda x: (x != "ALL", x not in ("indo-aryan", "dravidian"), x)):
        print(f"{k:<14}" + "".join(f"{res[k][m]:>11.1f}%" for m in methods))


def main(cfg):
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vmap = vocab_map_from(cfg["sooktam_dir"])
    items = load_manifest(cfg["data_dir"], langs=cfg["languages"])
    tr_items, va_items = split_items(items, cfg["val_frac"], cfg["seed"])
    ds = DurationPairSet(tr_items, cfg["data_dir"], vmap, seed=cfg["seed"])
    vds = DurationPairSet(va_items, cfg["data_dir"], vmap, seed=99, fixed=True)
    dl = torch.utils.data.DataLoader(ds, cfg["batch_size"], shuffle=True, collate_fn=collate_duration,
                                     num_workers=cfg["num_workers"], drop_last=True, persistent_workers=cfg["num_workers"] > 0)
    vdl = torch.utils.data.DataLoader(vds, 64, collate_fn=collate_duration)

    kw = dict(vocab_size=len(vmap), d=cfg["d"], heads=cfg["heads"], text_layers=cfg["text_layers"],
              mel_layers=cfg["mel_layers"], xattn_layers=cfg["xattn_layers"], dropout=cfg["dropout"])
    model = CrossAttnDurationPredictor(**kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["max_steps"], pct_start=0.05)
    print(f"duration model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    res0 = evaluate(model, vdl, device)
    print("at init (learned == cls_tokens heuristic):")
    print_table(res0)
    step, best = 0, float("inf")
    while step < cfg["max_steps"]:
        for b in dl:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            pred, _, tau = model(b["ref_mel"], b["ref_mel_lens"], b["ref_ids"], b["tgt_ids"], b["lang"], b["emo_weights"])
            loss = model.loss(pred, b["tgt_frames"], tau, cfg["tau_reg"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}  loss {loss.item():.4f}", flush=True)
            if step % cfg["val_every"] == 0 or step == cfg["max_steps"]:
                res = evaluate(model, vdl, device)
                print_table(res)
                if res["ALL"]["learned"] < best:
                    best = res["ALL"]["learned"]
                    torch.save(dict(model=model.state_dict(), model_kwargs=kw, step=step, results=res),
                               os.path.join(cfg["out_dir"], "best.pt"))
                    json.dump(res, open(os.path.join(cfg["out_dir"], "best_results.json"), "w"), indent=2)
            if step >= cfg["max_steps"]:
                break
    print("best mean abs % error:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(json.load(open(a.config)))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    main(cfg)
