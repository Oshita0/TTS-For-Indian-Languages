"""
Mel-spectrogram emotion classifier trained on REAL Rasa speech.

Used in v2 as a frozen "emotion critic": during TTS training it scores the model's one-step estimate of
the clean mel and penalises outputs whose rasa is not recognisable. Audio-only, so it cannot read the text.

    python -m sooktam_rasa.emo_classifier --data_dir /content/data_local --out /content/drive/MyDrive/sooktam_rasa/ckpt/emo_cls.pt
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

from .data import load_manifest
from .rasa import BASIS, NUM_BASIS

MEL_MEAN, MEL_STD = -4.5, 2.5   # rough log-mel stats of Sooktam/vocos features, for input scaling


class MelEmotionClassifier(nn.Module):
    def __init__(self, n_mels=100, d=256, n_classes=NUM_BASIS, dropout=0.2):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv1d(cin, cout, 5, stride=stride, padding=2), nn.GroupNorm(8, cout), nn.GELU())

        self.net = nn.Sequential(block(n_mels, d, 1), block(d, d, 2), block(d, d, 1), block(d, d, 2), block(d, d, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * d, n_classes)

    def forward(self, mel, mask=None):
        """mel (b, n, 100) log-mel; mask (b, n) True = valid frame. Returns logits (b, n_classes)."""
        x = ((mel - MEL_MEAN) / MEL_STD).transpose(1, 2)
        if mask is not None:
            x = x * mask[:, None, :].to(x.dtype)
        h = self.net(x)                                   # (b, d, n/4)
        if mask is None:
            m = torch.ones(h.shape[0], h.shape[2], device=h.device, dtype=h.dtype)
        else:
            m = F.max_pool1d(mask[:, None, :].float(), 4, 4, ceil_mode=True)[:, 0, : h.shape[2]].to(h.dtype)
        denom = m.sum(1, keepdim=True).clamp(min=1)
        mean = (h * m[:, None]).sum(2) / denom
        var = ((h - mean[..., None]) ** 2 * m[:, None]).sum(2) / denom
        return self.head(self.drop(torch.cat([mean, var.clamp(min=1e-5).sqrt()], 1)))


def load_classifier(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = MelEmotionClassifier(**blob.get("kwargs", {}))
    m.load_state_dict(blob["model"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class CropSet(torch.utils.data.Dataset):
    def __init__(self, items, data_dir, train=True, min_frames=150, max_frames=600):
        self.items, self.dir, self.train = items, data_dir, train
        self.min_f, self.max_f = min_frames, max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        mel = torch.load(os.path.join(self.dir, it["mel"])).float()
        if self.train:
            L = random.randint(self.min_f, self.max_f)
            if mel.shape[0] > L:
                s = random.randint(0, mel.shape[0] - L)
                mel = mel[s:s + L]
            # robustness to the blurry one-step estimates it will see inside TTS training
            if random.random() < 0.5:
                mel = mel + torch.randn_like(mel) * random.uniform(0.0, 0.6)
            if random.random() < 0.3:
                k = random.choice([3, 5])
                mel = F.avg_pool1d(mel.T[None], k, 1, k // 2)[0].T
            if random.random() < 0.5:  # spec-augment style frequency mask
                f0 = random.randint(0, 90)
                mel[:, f0:f0 + random.randint(2, 10)] = mel.mean()
        return mel, it["basis"]


def collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    mels = pad_sequence([b[0] for b in batch], batch_first=True)
    lens = torch.tensor([b[0].shape[0] for b in batch])
    mask = torch.arange(mels.shape[1])[None] < lens[:, None]
    return mels, mask, torch.tensor([b[1] for b in batch])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    conf = torch.zeros(NUM_BASIS, NUM_BASIS, dtype=torch.long)
    for mel, mask, y in loader:
        p = model(mel.to(device), mask.to(device)).argmax(1).cpu()
        correct += (p == y).sum().item()
        total += len(y)
        for a, b in zip(y, p):
            conf[a, b] += 1
    model.train()
    return correct / max(total, 1), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_manifest(a.data_dir)
    tr = [it for it in items if it["split"] != "test"]
    te = [it for it in items if it["split"] == "test"] or tr[: len(tr) // 20]
    cnt = Counter(it["basis"] for it in tr)
    w = torch.tensor([1.0 / cnt[it["basis"]] for it in tr], dtype=torch.double)
    dl = torch.utils.data.DataLoader(CropSet(tr, a.data_dir), a.batch_size, collate_fn=collate, num_workers=a.workers,
                                     sampler=torch.utils.data.WeightedRandomSampler(w, a.steps * a.batch_size))
    vdl = torch.utils.data.DataLoader(CropSet(te, a.data_dir, train=False), 32, collate_fn=collate)
    model = MelEmotionClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    print(f"classifier {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params; train {len(tr)} test {len(te)}")
    best = -1.0
    for step, (mel, mask, y) in enumerate(dl, 1):
        loss = F.cross_entropy(model(mel.to(device), mask.to(device)), y.to(device), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == a.steps:
            acc, conf = evaluate(model, vdl, device)
            print(f"step {step}  loss {loss.item():.3f}  test acc {acc:.3f}  (chance {1 / NUM_BASIS:.3f})", flush=True)
            if acc > best:
                best = acc
                torch.save(dict(model=model.state_dict(), kwargs={}, acc=acc, conf=conf.tolist(), basis=BASIS), a.out)
    blob = torch.load(a.out, weights_only=False)
    print(f"\nbest test accuracy {blob['acc']:.3f}. Confusion (rows = true, cols = predicted):")
    print(" " * 11 + "".join(f"{b[:6]:>8}" for b in BASIS))
    for i, row in enumerate(blob["conf"]):
        print(f"{BASIS[i]:<11}" + "".join(f"{v:>8}" for v in row))


if __name__ == "__main__":
    main()
