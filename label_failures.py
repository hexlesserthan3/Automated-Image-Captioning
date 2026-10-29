"""Click-to-label tool for the failure sheets written by error_analysis.py sample.

In a Colab notebook:
    from label_failures import label
    label("lstm", ROOT, WORK)      # or "knn" / "transformer"
Each click is saved to <WORK>/<model>_label_sheet.csv straight away, and rerunning
the call resumes at the first unlabeled row.
"""
import ipywidgets as w
import pandas as pd
from IPython.display import display
from PIL import Image

try:  # lets widgets render in Colab
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

CATS = ["missed_subject", "wrong_activity", "hallucinated_object", "wrong_detail", "generic", "acceptable"]
PROPS = ["clutter", "small_object", "unusual_scene", "none"]


def label(model, root, work):
    path = f"{work}/{model}_label_sheet.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    todo = [i for i in df.index if df.at[i, "category"] == ""]
    st = {"n": 0, "cat": None}
    out, status = w.Output(), w.HTML()
    cat_row = w.HBox([w.Button(description=c) for c in CATS])
    prop_row = w.HBox([w.Button(description=p) for p in PROPS])

    def show():
        st["cat"], status.value = None, ""
        out.clear_output(wait=True)
        with out:
            if st["n"] >= len(todo):
                print("All done for", model)
                return
            r = df.loc[todo[st["n"]]]
            print(f"{model}: image {st['n'] + 1} of {len(todo)}")
            img = Image.open(f"{root}/Images/{r['image']}")
            img.thumbnail((500, 500))
            display(img)
            print("GENERATED:", r["generated"])
            print("REFERENCE:", r["reference_1"])
            print("CIDEr:", r["cider"])

    def on_cat(b):
        st["cat"] = b.description
        status.value = f"<b>category: {b.description}</b>. Now click an image property."

    def on_prop(b):
        if st["cat"] is None:
            status.value = "<b style='color:red'>Pick a category first</b>"
            return
        i = todo[st["n"]]
        df.at[i, "category"], df.at[i, "image_property"] = st["cat"], b.description
        df.to_csv(path, index=False)
        st["n"] += 1
        show()

    for b in cat_row.children:
        b.on_click(on_cat)
    for b in prop_row.children:
        b.on_click(on_prop)
    display(out, w.HTML("<b>1. Category</b>"), cat_row,
            w.HTML("<b>2. Image property (this click saves and loads the next image)</b>"), prop_row, status)
    show()
