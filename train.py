"""Training loop (owner: Anikait - decoder training & tuning).

Usage:
  python train.py --model lstm        --root /content/flickr8k --work /content/work
  python train.py --model transformer --root /content/flickr8k --work /content/work
Saves best-val-loss checkpoint to <work>/<model>.pt and loss history to <work>/<model>_log.json.
"""
import argparse, json, os, time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import PAD, CaptionDataset, collate, load_everything
from models import build


def run_epoch(model, dl, loss_fn, dev, opt=None):
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for feats, caps in dl:
            feats, caps = feats.to(dev), caps.to(dev)
            logits = model(feats, caps[:, :-1])                       # teacher forcing
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            if opt:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(feats)
            n += len(feats)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    caps, split, split_name, vocab, feats, name2idx = load_everything(a.root, a.work)
    print("split:", split_name, {k: len(v) for k, v in split.items()}, "| vocab:", len(vocab))

    # Load features into RAM once (~1.6 GB float16) so dataloading is fast.
    feats = np.array(feats)
    tr = CaptionDataset(feats, name2idx, caps, split["train"], vocab)
    va = CaptionDataset(feats, name2idx, caps, split["val"], vocab)
    tr_dl = DataLoader(tr, a.bs, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(va, a.bs, collate_fn=collate)

    model = build(a.model, len(vocab)).to(dev)
    print(a.model, "parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)  # same loss for both models -> comparable
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    log, best, bad = [], float("inf"), 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        trl = run_epoch(model, tr_dl, loss_fn, dev, opt)
        val = run_epoch(model, va_dl, loss_fn, dev)
        log.append({"epoch": ep, "train_loss": trl, "val_loss": val, "secs": time.time() - t0})
        print(f"epoch {ep:2d}  train {trl:.3f}  val {val:.3f}  ({log[-1]['secs']:.0f}s)")
        if val < best:
            best, bad = val, 0
            torch.save({"model": model.state_dict(), "args": vars(a), "vocab": vocab.itos},
                       os.path.join(a.work, f"{a.model}.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break
        json.dump(log, open(os.path.join(a.work, f"{a.model}_log.json"), "w"), indent=1)
    print("best val loss:", best)


if __name__ == "__main__":
    main()
