"""Evaluation harness (owner: Manisha - evaluation).

Generates captions with greedy and beam search, computes multi-reference
BLEU-1..4, METEOR (NLTK variant), CIDEr, latency and simple repetition stats,
and writes per-image captions + CIDEr for error analysis.

Usage:
  python evaluate.py --model transformer --split val  --beams 3,5 --root ... --work ...   # tune beam width
  python evaluate.py --model transformer --split test --beams 3   --root ... --work ...   # final numbers
"""
import argparse, csv, json, os, time

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider

from common import load_everything
from decode import beam_search, greedy
from models import build


def metrics(hyps, refs):
    """hyps: {img: [tokens]}, refs: {img: [[tokens], ...]}"""
    ids = sorted(hyps)
    H = [hyps[i] for i in ids]
    R = [refs[i] for i in ids]
    sm = SmoothingFunction().method1
    out = {}
    for n in range(1, 5):
        w = tuple([1.0 / n] * n + [0.0] * (4 - n))
        out[f"BLEU-{n}"] = corpus_bleu(R, H, weights=w, smoothing_function=sm)
    out["METEOR"] = sum(meteor_score(r, h) for r, h in zip(R, H)) / len(H)
    gts = {i: [" ".join(r) for r in refs[i]] for i in ids}
    res = {i: [" ".join(hyps[i])] for i in ids}
    score, per_img = Cider().compute_score(gts, res)
    out["CIDEr"] = score
    out["avg_len"] = sum(len(h) for h in H) / len(H)
    out["distinct_captions"] = len({tuple(h) for h in H}) / len(H)
    out["repeated_word_rate"] = sum(len(set(h)) < len(h) for h in H) / len(H)
    return out, dict(zip(ids, per_img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--beams", default="3", help="comma-separated beam widths, e.g. 3,5")
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()

    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    caps, split, split_name, vocab, feats, name2idx = load_everything(a.root, a.work)
    ck = torch.load(os.path.join(a.work, f"{a.model}.pt"), map_location=dev)
    assert ck["vocab"] == vocab.itos, "vocab mismatch: same split/seed as training?"
    model = build(a.model, len(vocab)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    names = split[a.split]
    refs = {n: caps[n] for n in names}
    X = torch.from_numpy(feats[[name2idx[n] for n in names]].astype("float32"))

    results, rows = {}, {}
    settings = [("greedy", 1)] + [(f"beam{b}", int(b)) for b in a.beams.split(",")]
    for label, b in settings:
        hyps, t0 = {}, time.time()
        if b == 1:
            for i in range(0, len(names), 100):
                batch = X[i:i + 100].to(dev)
                for n, ids in zip(names[i:i + 100], greedy(model, batch)):
                    hyps[n] = vocab.decode(ids)
        else:
            for n, x in zip(names, X):
                hyps[n] = vocab.decode(beam_search(model, x[None].to(dev), beam=b))
        if dev == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t0
        m, per_img = metrics(hyps, refs)
        m["secs_total"] = secs
        m["ms_per_image"] = 1000 * secs / len(names)
        results[label] = m
        rows[label] = (hyps, per_img)
        print(label, {k: round(v, 4) for k, v in m.items()})

    tag = f"{a.model}_{a.split}"
    json.dump({"split_used": split_name, "results": results},
              open(os.path.join(a.work, f"{tag}_metrics.json"), "w"), indent=1)

    # Per-image captions + CIDEr, worst first, for the error-analysis section.
    best = max(results, key=lambda k: results[k]["CIDEr"])
    hyps, per_img = rows[best]
    with open(os.path.join(a.work, f"{tag}_captions_{best}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "cider", "generated", "reference_1"])
        for n in sorted(names, key=lambda n: per_img[n]):
            w.writerow([n, round(per_img[n], 4), " ".join(hyps[n]), " ".join(refs[n][0])])
    print("best setting by CIDEr:", best)


if __name__ == "__main__":
    main()
