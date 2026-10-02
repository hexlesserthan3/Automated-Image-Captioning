"""Frozen ResNet-50 spatial feature extraction (owner: Siya - encoder).

Resizes to 224x224, normalises with ImageNet mean/std, and caches the
7x7x2048 grid (before global pooling) as (N, 49, 2048) float16.

Usage: python extract_features.py --root /content/flickr8k --work /content/work
"""
import argparse, json, os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import load_captions


class Images(Dataset):
    def __init__(self, root, names):
        self.root, self.names = root, names
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, "Images", self.names[i])).convert("RGB")
        return self.tf(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    names = sorted(load_captions(a.root))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    enc = nn.Sequential(*list(net.children())[:-2]).to(dev).eval()  # drop avgpool + fc
    for p in enc.parameters():
        p.requires_grad = False

    out = np.lib.format.open_memmap(os.path.join(a.work, "feats.npy"), mode="w+",
                                    dtype=np.float16, shape=(len(names), 49, 2048))
    dl = DataLoader(Images(a.root, names), batch_size=a.bs, num_workers=2)
    i = 0
    with torch.no_grad():
        for x in dl:
            f = enc(x.to(dev))                      # (B, 2048, 7, 7)
            f = f.flatten(2).transpose(1, 2)        # (B, 49, 2048)
            out[i:i + len(f)] = f.half().cpu().numpy()
            i += len(f)
            print(f"\r{i}/{len(names)}", end="")
    out.flush()
    json.dump(names, open(os.path.join(a.work, "feat_names.json"), "w"))
    print("\nsaved", out.shape)


if __name__ == "__main__":
    main()
