"""Shared data utilities (owner: Anikait - data pipeline).

Handles caption loading/cleaning, the 6000/1000/1000 split, vocabulary
construction (training captions only) and the PyTorch Dataset/collate.
"""
import csv, json, os, random, re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

PAD, START, END, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<start>", "<end>", "<unk>"]
MAX_LEN = 30  # max caption tokens (excluding <start>/<end>)


def clean(caption):
    """Lowercase, strip punctuation, split into tokens."""
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", " ", caption)
    return caption.split()


def load_captions(root):
    """Return {image_name: [tokenised caption, ...]} from Kaggle captions.txt."""
    caps = {}
    with open(os.path.join(root, "captions.txt"), newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header: image,caption
        for img, cap in reader:
            toks = clean(cap)
            if toks:
                caps.setdefault(img, []).append(toks)
    return caps


def make_split(root, images, seed=42):
    """Karpathy split if dataset_flickr8k.json is in root, else a seeded
    6000/1000/1000 random split (state which one you used in the report)."""
    kp = os.path.join(root, "dataset_flickr8k.json")
    if os.path.exists(kp):
        data = json.load(open(kp))
        split = {"train": [], "val": [], "test": []}
        have = set(images)
        for im in data["images"]:
            s = "train" if im["split"] == "restval" else im["split"]
            if im["filename"] in have:
                split[s].append(im["filename"])
        return split, "karpathy"
    imgs = sorted(images)
    random.Random(seed).shuffle(imgs)
    return {"train": imgs[:6000], "val": imgs[6000:7000], "test": imgs[7000:8000]}, "random-seed%d" % seed


class Vocab:
    def __init__(self, train_caps, min_freq=5):
        counts = Counter(w for caps in train_caps for c in caps for w in c)
        words = sorted(w for w, n in counts.items() if n >= min_freq)
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        ids = [self.stoi.get(w, UNK) for w in toks[:MAX_LEN]]
        return [START] + ids + [END]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == END:
                break
            if i in (PAD, START):
                continue
            out.append(self.itos[i])
        return out


class CaptionDataset(Dataset):
    """One item per (image, caption) pair. feats is the cached (N,49,2048) array."""

    def __init__(self, feats, name2idx, caps, images, vocab):
        self.feats, self.vocab = feats, vocab
        self.pairs = [(name2idx[im], vocab.encode(c)) for im in images for c in caps[im]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idx, ids = self.pairs[i]
        return torch.from_numpy(self.feats[idx].astype(np.float32)), torch.tensor(ids)


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    maxlen = max(len(b[1]) for b in batch)
    caps = torch.full((len(batch), maxlen), PAD, dtype=torch.long)
    for i, (_, c) in enumerate(batch):
        caps[i, : len(c)] = c
    return feats, caps


def load_everything(root, work):
    """Captions, split, vocab and cached features, in one call."""
    caps = load_captions(root)
    split, split_name = make_split(root, list(caps))
    vocab = Vocab([caps[i] for i in split["train"]])
    names = json.load(open(os.path.join(work, "feat_names.json")))
    name2idx = {n: i for i, n in enumerate(names)}
    feats = np.load(os.path.join(work, "feats.npy"), mmap_mode="r")
    return caps, split, split_name, vocab, feats, name2idx
