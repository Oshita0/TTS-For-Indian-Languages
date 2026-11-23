"""
Does the adapter actually USE the rasa label?  (label-ablation on held-out val clips)

    python -m sooktam_rasa.diagnose --data_dir /content/data_local \
        --ckpts /content/drive/MyDrive/sooktam_rasa/ckpt/emotion/best.pt /content/drive/MyDrive/sooktam_rasa/ckpt/emotion/latest.pt

Every condition sees the SAME clips, the SAME neutral prompts and the SAME noise / timesteps, so
differences come only from what the model is told:
    base      : original Sooktam-2 (adapter off)
    correct   : adapter + true rasa label
    shuffled  : adapter + a wrong rasa label
    null      : adapter + "no emotion" embedding
Reading the result:
    base -> null      : gain from adapting to Rasa speakers/recordings (NOT emotion)
    null -> correct   : gain from the emotion label itself   <- the number that matters
    correct < shuffled: the label is used in the right direction (wrong labels hurt)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from .data import EmotionTrainSet, load_manifest
from .emotion_dit import attach_emotion, emotion_forward, lens_to_mask, load_adapter, set_lora_enabled
from .rasa import BASIS, NUM_BASIS
from .train_emotion import fixed_val_batches, split_items


@torch.no_grad()
def per_sample_loss(cfm, b, weights, null, use_emotion, seed, amp_dtype):
    tr = cfm.transformer
    torch.manual_seed(seed)
    mel, text, lens, lm = b["mel"], b["text"], b["lens"], b["loss_mask"]
    n = mel.shape[1]
    valid = lens_to_mask(lens, n)
    lm = lm & valid
    x0 = torch.randn_like(mel)
    t = torch.rand((mel.shape[0],), device=mel.device, dtype=mel.dtype)
    phi = (1 - t[:, None, None]) * x0 + t[:, None, None] * mel
    cond = torch.where(lm[..., None], torch.zeros_like(mel), mel)
    with torch.autocast(mel.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        pred = emotion_forward(tr, phi, cond, text, t, mask=valid,
                               emo_weights=weights if use_emotion else None, emo_null=null)
    err = F.mse_loss(pred.float(), (mel - x0).float(), reduction="none").mean(-1)
    return ((err * lm).sum(1) / lm.sum(1).clamp(min=1)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sooktam_dir", default="/content/sooktam2")
    ap.add_argument("--frames_per_batch", type=int, default=12000)
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    from . import infer
    f5 = infer.load_sooktam(a.sooktam_dir, device=str(device))
    cfm = f5.ema_model.float().eval()
    first = torch.load(a.ckpts[0], map_location="cpu", weights_only=False)
    attach_emotion(cfm, lora_r=first["config"].get("lora_r", 16), lora_alpha=first["config"].get("lora_alpha", 16),
                   input_proj=first["config"].get("emo_input_proj", False))

    items = load_manifest(a.data_dir)
    _, val_items = split_items(items, 0.03, 0)
    vds = EmotionTrainSet(val_items, a.data_dir, cfm.vocab_char_map, p_pair=1.0)
    batches = fixed_val_batches(vds, a.frames_per_batch, a.n_batches)
    print(f"{sum(len(b['lens']) for b in batches)} val clips in {len(batches)} fixed batches")

    results = {}
    for ck in a.ckpts:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        load_adapter(cfm.transformer, blob["adapter"])
        name = f"{os.path.basename(ck)} (step {blob.get('step')})"
        per = defaultdict(list)
        basis_all = []
        for i, b in enumerate(batches):
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            B = b["mel"].shape[0]
            w = b["emo_weights"]
            basis = b["basis"]
            shift = torch.randint(1, NUM_BASIS, (B,), generator=torch.Generator().manual_seed(i)).to(device)
            w_wrong = F.one_hot((basis + shift) % NUM_BASIS, NUM_BASIS).float()
            no = torch.zeros(B, dtype=torch.bool, device=device)
            yes = torch.ones(B, dtype=torch.bool, device=device)
            seed = 10_000 + i
            set_lora_enabled(cfm.transformer, False)
            per["base"].append(per_sample_loss(cfm, b, w, no, False, seed, amp))
            set_lora_enabled(cfm.transformer, True)
            per["null"].append(per_sample_loss(cfm, b, w, yes, True, seed, amp))
            per["correct"].append(per_sample_loss(cfm, b, w, no, True, seed, amp))
            per["shuffled"].append(per_sample_loss(cfm, b, w_wrong, no, True, seed, amp))
            basis_all.append(basis.cpu())
        basis_all = torch.cat(basis_all)
        per = {k: torch.cat(v) for k, v in per.items()}
        res = {"overall": {k: v.mean().item() for k, v in per.items()}, "by_rasa": {}}
        for bi, bn in enumerate(BASIS):
            m = basis_all == bi
            if m.any():
                res["by_rasa"][bn] = {k: v[m].mean().item() for k, v in per.items()} | {"n": int(m.sum())}
        o = res["overall"]
        res["gain_label_vs_null_pct"] = 100 * (o["null"] - o["correct"]) / o["null"]
        res["gain_label_vs_shuffled_pct"] = 100 * (o["shuffled"] - o["correct"]) / o["shuffled"]
        res["gain_domain_base_vs_null_pct"] = 100 * (o["base"] - o["null"]) / o["base"]
        # paired win rate: fraction of clips where the correct label beats null
        res["clips_correct_beats_null"] = (per["correct"] < per["null"]).float().mean().item()
        res["clips_correct_beats_shuffled"] = (per["correct"] < per["shuffled"]).float().mean().item()
        results[name] = res

        print(f"\n=== {name} ===")
        print(f"{'':<12}{'base':>9}{'null':>9}{'correct':>9}{'shuffled':>9}{'n':>6}")
        print(f"{'ALL':<12}" + "".join(f"{o[k]:>9.4f}" for k in ["base", "null", "correct", "shuffled"]) + f"{len(basis_all):>6}")
        for bn, r in res["by_rasa"].items():
            print(f"{bn:<12}" + "".join(f"{r[k]:>9.4f}" for k in ["base", "null", "correct", "shuffled"]) + f"{r['n']:>6}")
        print(f"speaker/recording adaptation (base->null): {res['gain_domain_base_vs_null_pct']:+.2f}%")
        print(f"emotion label gain        (null->correct): {res['gain_label_vs_null_pct']:+.2f}%   "
              f"clips where label helps: {res['clips_correct_beats_null']:.0%}")
        print(f"right vs wrong label   (shuffled->correct): {res['gain_label_vs_shuffled_pct']:+.2f}%   "
              f"clips where right label wins: {res['clips_correct_beats_shuffled']:.0%}")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)
        print("saved", a.out)


if __name__ == "__main__":
    main()
