"""Model 3: nearest-neighbour caption retrieval (classical baseline, no training).

For a query image, find the k most similar TRAINING images by cosine similarity
of mean-pooled frozen ResNet-50 features, pool their captions, and return the
candidate caption that agrees most with the other candidates (consensus).
Same split, same cached features and same metric code as the neural models.

Usage (tune k on val, then run test once):
  python knn_baseline.py --split val  --ks 1,3,5,10 --root $ROOT --work $WORK
  python knn_baseline.py --split test --ks 5        --root $ROOT --work $WORK
"""
import argparse, csv, json, os, time
from collections import Counter

import nltk
import numpy as np

from common import load_everything
from evaluate import metrics  # same BLEU/METEOR/CIDEr code as the neural models


def ngrams(t, n):
    return Counter(zip(*[t[i:] for i in range(n)]))


def overlap(a, b):
    """Mean of unigram and bigram F1 between two token lists."""
    s = 0.0
    for n in (1, 2):
        A, B = ngrams(a, n), ngrams(b, n)
        common = sum((A & B).values())
        if common:
            p, r = common / sum(A.values()), common / sum(B.values())
            s += 2 * p * r / (p + r)
    return s / 2


def pooled(feats, name2idx, names, chunk=500):
    """Mean-pool the 49 grid cells, L2-normalise. Chunked to save memory."""
    out = []
    for i in range(0, len(names), chunk):
        idx = [name2idx[n] for n in names[i:i + chunk]]
        out.append(np.asarray(feats[idx], dtype=np.float32).mean(1))
    x = np.concatenate(out)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--ks", default="5", help="comma-separated neighbour counts")
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)

    caps, split, split_name, vocab, feats, name2idx = load_everything(a.root, a.work)
    train, queries = split["train"], split[a.split]
    refs = {n: caps[n] for n in queries}

    t0 = time.time()
    sims = pooled(feats, name2idx, queries) @ pooled(feats, name2idx, train).T  # (Q, n_train)
    order = np.argsort(-sims, axis=1)
    t_sim = time.time() - t0

    results, rows = {}, {}
    for k in [int(x) for x in a.ks.split(",")]:
        t0, hyps = time.time(), {}
        for n, nbrs in zip(queries, order[:, :k]):
            cands = [c for j in nbrs for c in caps[train[j]]]
            best = max(range(len(cands)),
                       key=lambda i: sum(overlap(cands[i], cands[j]) for j in range(len(cands)) if j != i))
            hyps[n] = cands[best]
        secs = time.time() - t0 + t_sim
        m, per_img = metrics(hyps, refs)
        m["ms_per_image"] = 1000 * secs / len(queries)
        results[f"knn_k{k}"] = m
        rows[f"knn_k{k}"] = (hyps, per_img)
        print(f"k={k}", {key: round(float(v), 4) for key, v in m.items()})

    json.dump({"split_used": split_name, "results": {k: {a: float(b) for a, b in v.items()} for k, v in results.items()}},
              open(os.path.join(a.work, f"knn_{a.split}_metrics.json"), "w"), indent=1)

    best = max(results, key=lambda k: results[k]["CIDEr"])
    hyps, per_img = rows[best]
    with open(os.path.join(a.work, f"knn_{a.split}_captions_{best}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "cider", "generated", "reference_1"])
        for n in sorted(queries, key=lambda n: per_img[n]):
            w.writerow([n, round(float(per_img[n]), 4), " ".join(hyps[n]), " ".join(refs[n][0])])
    print("best by CIDEr:", best)


if __name__ == "__main__":
    main()
