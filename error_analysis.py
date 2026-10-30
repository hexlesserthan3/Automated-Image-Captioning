"""Comparison table + error analysis across lstm / transformer / knn (test split).

  python error_analysis.py table  --work $WORK         # comparison table + zero-score stats + most common captions
  python error_analysis.py sample --work $WORK --n 50  # label sheets of low-scoring captions (CIDEr < 0.2)
                                                       # never overwrites a sheet that already has labels unless --force
                                                       # use --models knn to regenerate just one model
  python error_analysis.py tally  --work $WORK         # after you fill 'category' and 'image_property' in the sheets
"""
import argparse, glob, json, os

import pandas as pd

MODELS = ["lstm", "transformer", "knn"]
CATS = ["missed_subject", "wrong_activity", "hallucinated_object", "wrong_detail", "generic", "acceptable"]
PROPS = ["clutter", "small_object", "unusual_scene", "none"]
COLS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "CIDEr", "avg_len",
        "distinct_captions", "immediate_repeat_rate", "repeated_trigram_rate"]


def caption_csv(work, m):
    fs = glob.glob(os.path.join(work, f"{m}_test_captions_*.csv"))
    return max(fs, key=os.path.getmtime) if fs else None


def table(work):
    rows = []
    for m in MODELS:
        p = os.path.join(work, f"{m}_test_metrics.json")
        if not os.path.exists(p):
            print("missing:", p)
            continue
        for label, res in json.load(open(p))["results"].items():
            r = {"model": m, "setting": label}
            r.update({c: res.get(c) for c in COLS})
            # fair latency: greedy one image at a time; beam/knn are already per image
            r["ms_per_image"] = res.get("ms_per_image_bs1") if label == "greedy" else res.get("ms_per_image")
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(work, "comparison_table.csv"), index=False)
    print(df.round(4).to_string(index=False))

    for m in MODELS:
        p = caption_csv(work, m)
        if not p:
            continue
        d = pd.read_csv(p)
        print(f"\n{m}  ({os.path.basename(p)})  n={len(d)}  mean CIDEr={d.cider.mean():.4f}  "
              f"CIDEr==0: {(d.cider == 0).sum()}  CIDEr<0.2: {(d.cider < 0.2).sum()}")
        print("most frequent generated captions:")
        print(d.generated.value_counts().head(5).to_string())


def sample(work, n, models, force):
    for m in models:
        p = caption_csv(work, m)
        if not p:
            print("missing captions csv for", m)
            continue
        d = pd.read_csv(p)
        low = d[d.cider < 0.2]
        s = low.sample(min(n, len(low)), random_state=0).copy()
        s["category"], s["image_property"], s["notes"] = "", "", ""
        out = os.path.join(work, f"{m}_label_sheet.csv")
        if os.path.exists(out) and not force:
            old = pd.read_csv(out, dtype=str).fillna("")
            if (old["category"] != "").any():
                print(f"{m}: {out} already has labels, NOT overwriting (use --force to replace it)")
                continue
        s.to_csv(out, index=False)
        print(f"{m}: {len(low)} captions with CIDEr<0.2, wrote {len(s)} to {out}")
    print("\ncategory values:      ", ", ".join(CATS))
    print("image_property values:", ", ".join(PROPS))
    print("Open each image next to its row and fill both columns by hand.")


def tally(work):
    for col, allowed in (("category", CATS), ("image_property", PROPS)):
        res = {}
        for m in MODELS:
            p = os.path.join(work, f"{m}_label_sheet.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            vals = d[col].fillna("").astype(str).str.strip()
            bad = sorted(set(vals) - set(allowed) - {""})
            if bad:
                print(f"WARNING {m}.{col}: unknown values {bad}")
            if (vals == "").any():
                print(f"WARNING {m}.{col}: {(vals == '').sum()} rows still unlabeled")
            res[m] = vals[vals != ""].value_counts()
        t = pd.DataFrame(res).reindex(allowed).fillna(0).astype(int)
        t.loc["total labeled"] = t.sum()
        print(f"\n{col}:\n{t.to_string()}")
        t.to_csv(os.path.join(work, f"error_tally_{col}.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["table", "sample", "tally"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--models", default=",".join(MODELS), help="sample mode: which models, e.g. knn")
    ap.add_argument("--force", action="store_true", help="sample mode: overwrite sheets that already have labels")
    a = ap.parse_args()
    {"table": lambda: table(a.work),
     "sample": lambda: sample(a.work, a.n, a.models.split(","), a.force),
     "tally": lambda: tally(a.work)}[a.mode]()
