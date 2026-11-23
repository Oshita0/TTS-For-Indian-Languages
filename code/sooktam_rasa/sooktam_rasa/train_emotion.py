"""
Train the rasa/emotion adapter on frozen Sooktam-2.

    python -m sooktam_rasa.train_emotion --config configs/emotion_t4.json

Resumable: re-running with the same out_dir continues from out_dir/latest.pt.
Writes: latest.pt, best.pt (adapter only, a few MB), log.csv, samples/step_xxxxx/*.wav
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time

import torch

from .data import EmotionTrainSet, FrameBudgetSampler, collate_emotion, load_manifest, oov_report
from .emotion_dit import adapter_state_dict, attach_emotion, flow_matching_loss, load_adapter, one_step_estimate
from .rasa import BASIS, TRAINED_RASAS, ZERO_SHOT_RASAS

DEFAULTS = dict(
    sooktam_dir="/content/sooktam2", ckpt=None, data_dir="/content/drive/MyDrive/sooktam_rasa/data",
    out_dir="/content/drive/MyDrive/sooktam_rasa/ckpt/emotion", languages=None,
    lora_r=16, lora_alpha=16, lr=1e-4, lr_emotion=5e-4, weight_decay=0.01, warmup=300, max_steps=8000,
    frames_per_batch=6000, grad_accum=2, p_pair=0.7, balance_power=0.5,
    audio_drop_prob=0.3, cond_drop_prob=0.2, emo_drop_prob=0.15, grad_ckpt=True, amp="auto",
    val_frac=0.03, val_batches=40, log_every=25, val_every=500, save_every=500, sample_every=1000,
    num_workers=2, seed=0, backbone_dropout=False, reset_best=False,
    # ---- v2 (make the label necessary; Rasa text is emotion-specific) ----
    emo_input_proj=False,            # also inject emotion into every input frame
    text_drop_keep_emo_prob=0.0,     # hide the words, keep prompt + label
    cls_ckpt=None, cls_weight=0.0,   # frozen mel emotion critic on the one-step clean estimate
)


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"])
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def split_items(items, val_frac, seed):
    test = [it for it in items if it["split"] == "test"]
    train = [it for it in items if it["split"] != "test"]
    if test:
        return train, test
    rng = random.Random(seed)
    rng.shuffle(train)
    n = max(50, int(len(train) * val_frac))
    return train[n:], train[:n]


@torch.no_grad()
def validate(cfm, val_batches, device, amp_dtype, null_label=False):
    """Same clips, same pairings, same noise at every check, so values are comparable across steps."""
    cfm.eval()
    tot = 0.0
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(1234)
        for b in val_batches:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            null = torch.ones_like(b["emo_null"]) if null_label else b["emo_null"]
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = flow_matching_loss(cfm, b["mel"], b["text"], b["lens"], b["loss_mask"], b["emo_weights"],
                                          null, audio_drop_prob=0.0, cond_drop_prob=0.0, grad_ckpt=False)
            tot += loss.item()
    return tot / max(len(val_batches), 1)


def fixed_val_batches(vds, frames_per_batch, n_batches, seed=1234):
    state = random.getstate()
    random.seed(seed)
    sampler = FrameBudgetSampler(vds, frames_per_batch, batches_per_epoch=n_batches, balance_power=1.0, seed=seed)
    batches = [collate_emotion([vds[i] for i in idx], 0.0) for idx in sampler]
    random.setstate(state)
    return batches


def make_samples(f5, val_items, data_dir, out_dir, step, cfg):
    """One neutral reference per language -> one sentence in every rasa. For listening during training."""
    import soundfile as sf
    from .infer import EmotionalSooktam

    tts = EmotionalSooktam(f5)
    neutral = BASIS.index("shanta")
    done = set()
    for ref in val_items:
        if ref["basis"] != neutral or ref["lang"] in done or not ref.get("wav"):
            continue
        # a NEUTRAL sentence: Rasa's emotional sentences carry their emotion in the words themselves
        tgt = next((t for t in val_items if t["lang"] == ref["lang"] and t["basis"] == neutral and t["id"] != ref["id"]), None)
        if tgt is None:
            continue
        d = os.path.join(out_dir, "samples", f"step_{step:06d}")
        for r in TRAINED_RASAS + ZERO_SHOT_RASAS:
            try:
                tts.synthesize(os.path.join(data_dir, ref["wav"]), ref["text"], tgt["text"], ref["lang"], rasa=r,
                               duration="grapheme", seed=0, out_path=os.path.join(d, f"{ref['lang']}_{r}.wav"))
            except Exception as e:  # noqa: BLE001
                print("sample failed", ref["lang"], r, repr(e)[:200])
        done.add(ref["lang"])
    f5.ema_model.train()


def main(cfg):
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    json.dump(cfg, open(os.path.join(cfg["out_dir"], "config.json"), "w"), indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .infer import load_sooktam
    f5 = load_sooktam(cfg["sooktam_dir"], cfg["ckpt"], device=str(device))
    cfm = f5.ema_model.float()
    names = attach_emotion(cfm, lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], input_proj=cfg["emo_input_proj"])
    critic = None
    if cfg["cls_ckpt"] and cfg["cls_weight"] > 0:
        from .emo_classifier import load_classifier
        critic = load_classifier(cfg["cls_ckpt"], device)
        print(f"emotion critic loaded ({cfg['cls_ckpt']}), weight {cfg['cls_weight']}")
    tr = cfm.transformer
    n_train = sum(p.numel() for p in tr.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.2f}M of {sum(p.numel() for p in cfm.parameters()) / 1e6:.0f}M")

    items = load_manifest(cfg["data_dir"], langs=cfg["languages"])
    train_items, val_items = split_items(items, cfg["val_frac"], cfg["seed"])
    print(f"train {len(train_items)}  val {len(val_items)}")
    oov_report(train_items[:2000], cfm.vocab_char_map)

    ds = EmotionTrainSet(train_items, cfg["data_dir"], cfm.vocab_char_map, p_pair=cfg["p_pair"], seed=cfg["seed"])
    vds = EmotionTrainSet(val_items, cfg["data_dir"], cfm.vocab_char_map, p_pair=1.0, seed=1)
    coll = lambda b: collate_emotion(b, cfg["emo_drop_prob"])
    sampler = FrameBudgetSampler(ds, cfg["frames_per_batch"], batches_per_epoch=10 ** 9,
                                 balance_power=cfg["balance_power"], seed=cfg["seed"])
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=coll, num_workers=cfg["num_workers"],
                                         persistent_workers=cfg["num_workers"] > 0)
    val_batches = fixed_val_batches(vds, cfg["frames_per_batch"], cfg["val_batches"])

    emo_params = [p for n, p in tr.named_parameters() if p.requires_grad and n.startswith("emotion.")]
    lora_params = [p for n, p in tr.named_parameters() if p.requires_grad and "lora_" in n]
    opt = torch.optim.AdamW([{"params": emo_params, "lr": cfg["lr_emotion"]},
                             {"params": lora_params, "lr": cfg["lr"]}], weight_decay=cfg["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]

    if cfg["amp"] == "auto":
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    else:
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[cfg["amp"]]
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    print("AMP:", amp_dtype)

    step, best = 0, float("inf")
    latest = os.path.join(cfg["out_dir"], "latest.pt")
    if os.path.exists(latest):
        blob = torch.load(latest, map_location="cpu")
        load_adapter(tr, blob["adapter"])
        opt.load_state_dict(blob["opt"])
        scaler.load_state_dict(blob["scaler"])
        step, best = blob["step"], blob.get("best", best)
        if cfg.get("reset_best"):  # old best came from a different val set; start the comparison fresh
            best = float("inf")
        print(f"resumed from step {step}")

    def save(path, extra=None):
        blob = dict(adapter=adapter_state_dict(tr), step=step, best=best, config=cfg, basis=BASIS)
        if extra:
            blob.update(extra)
        torch.save(blob, path + ".tmp")
        os.replace(path + ".tmp", path)

    log_f = open(os.path.join(cfg["out_dir"], "log.csv"), "a", newline="")
    log = csv.writer(log_f)
    # Backbone stays in eval mode (no dropout) by default: only the adapters are being fitted.
    cfm.train() if cfg["backbone_dropout"] else cfm.eval()
    it = iter(loader)
    t0, run_loss = time.time(), 0.0
    run_cls, run_cls_hit, run_cls_n = 0.0, 0, 0
    while step < cfg["max_steps"]:
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg["grad_accum"]):
            b = next(it)
            b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                fm, parts = flow_matching_loss(cfm, b["mel"], b["text"], b["lens"], b["loss_mask"], b["emo_weights"],
                                               b["emo_null"], cfg["audio_drop_prob"], cfg["cond_drop_prob"], cfg["grad_ckpt"],
                                               text_drop_keep_emo_prob=cfg["text_drop_keep_emo_prob"], return_parts=True)
                loss = fm
                if critic is not None:
                    # score the model's guess of the clean mel on the generated span; weight by t because
                    # the guess is pure blur near t=0. Only samples that were given a label count.
                    x1h = one_step_estimate(parts)
                    logits = critic(x1h, parts["loss_mask"])
                    ce = torch.nn.functional.cross_entropy(logits.float(), b["basis"], reduction="none")
                    w = parts["time"].float() * (~parts["emo_null"]).float()
                    cls_loss = (ce * w).sum() / w.sum().clamp(min=1e-6)
                    loss = fm + cfg["cls_weight"] * cls_loss
                    with torch.no_grad():
                        hit = ((logits.argmax(1) == b["basis"]).float() * (parts["time"] > 0.5).float() * (~parts["emo_null"]).float())
                        run_cls_hit += hit.sum().item()
                        run_cls_n += ((parts["time"] > 0.5) & ~parts["emo_null"]).sum().item()
                        run_cls += cls_loss.item() / cfg["grad_accum"]
            scaler.scale(loss / cfg["grad_accum"]).backward()
            run_loss += fm.item() / cfg["grad_accum"]
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % cfg["log_every"] == 0:
            dt = (time.time() - t0) / cfg["log_every"]
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0
            extra = ""
            if critic is not None:
                extra = f"  critic {run_cls / cfg['log_every']:.3f}  critic-acc(t>.5) {run_cls_hit / max(run_cls_n, 1):.2f}"
            print(f"step {step:6d}  loss {run_loss / cfg['log_every']:.4f}{extra}  gnorm {gn:.3f}  "
                  f"lr {opt.param_groups[1]['lr']:.2e}  {dt:.2f}s/step  mem {mem:.1f}G", flush=True)
            log.writerow([step, "train", run_loss / cfg["log_every"], float(gn), dt, run_cls / cfg["log_every"]])
            log_f.flush()
            t0, run_loss = time.time(), 0.0
            run_cls, run_cls_hit, run_cls_n = 0.0, 0, 0
        if step % cfg["val_every"] == 0:
            vl = validate(cfm, val_batches, device, amp_dtype)
            vn = validate(cfm, val_batches, device, amp_dtype, null_label=True)
            cfm.train() if cfg["backbone_dropout"] else cfm.eval()
            print(f"  val loss {vl:.4f}  (best {best:.4f})   no-label {vn:.4f}   "
                  f"LABEL GAIN {100 * (vn - vl) / vn:+.2f}%  <- should grow clearly above 0")
            log.writerow([step, "val", vl, vn, 100 * (vn - vl) / vn])
            log_f.flush()
            if vl < best:
                best = vl
                save(os.path.join(cfg["out_dir"], "best.pt"))
        if step % cfg["save_every"] == 0:
            save(latest, dict(opt=opt.state_dict(), scaler=scaler.state_dict()))
            if step % 1000 == 0:
                os.makedirs(os.path.join(cfg["out_dir"], "snapshots"), exist_ok=True)
                save(os.path.join(cfg["out_dir"], "snapshots", f"step_{step:06d}.pt"))
        if cfg["sample_every"] and step % cfg["sample_every"] == 0:
            make_samples(f5, val_items, cfg["data_dir"], cfg["out_dir"], step, cfg)
            cfm.train() if cfg["backbone_dropout"] else cfm.eval()
    save(latest, dict(opt=opt.state_dict(), scaler=scaler.state_dict()))
    print("finished. best val", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (JSON values)")
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
