"""Expected per-batch class overlap vs the mass m-POT found by grid search (no training).

    python src/analysis/overlap_table.py            # writes results/analysis/overlap_table.{csv,md}

Closed form (apot.overlap.expected_overlap) for the mean; Monte Carlo over the exact loaders
(stratified source, target drawn without replacement from the label list) for sd and quantiles.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import apps  # noqa: E402
from apot.overlap import expected_overlap, overlap_mc  # noqa: E402

DATA = os.path.join(apps.REPO, "data")


def list_labels(path, keep_alpha=None):
    rows = [l.split() for l in open(path) if l.strip()]
    if keep_alpha:
        name_of = {}
        for p, y in rows:
            name_of.setdefault(int(y), p.split("/")[-2])
        names = set(sorted(name_of.values())[:keep_alpha])
        rows = [(p, y) for p, y in rows if name_of[int(y)] in names]
    return np.array([int(y) for _, y in rows])


def digit_labels(name):
    from torchvision import datasets
    if name == "mnist":
        return datasets.MNIST(DATA, train=True).targets.numpy()
    return np.asarray(datasets.USPS(DATA, train=True).targets)


def main():
    rows = []
    for task, tgt, s in [("SVHN->MNIST", "mnist", "0.85"), ("USPS->MNIST", "mnist", "0.90"), ("MNIST->USPS", "usps", "0.80")]:
        rows.append(("Digits", task, 500, 10, digit_labels(tgt), s))
    visda = list_labels(os.path.join(DATA, "visda-2017", "validation_list.txt"))
    for m, s in [(36, "0.70"), (72, "0.75"), (144, "0.80")]:
        rows.append(("VisDA", f"syn->real m={m}", m, 12, visda, s))
    for dom in ["Art", "Clipart", "Product", "Real_World"]:
        y = list_labels(os.path.join(DATA, "office-home", f"{dom}.txt"))
        rows.append(("Office-Home", f"*->{dom}", 65, 65, y, "0.5-0.7"))
    for dom in ["Art", "Clipart", "Product", "Real_World"]:
        y = list_labels(os.path.join(DATA, "office-home", f"{dom}.txt"), keep_alpha=25)
        rows.append(("OH partial 65->25", f"*->{dom}", 65, 65, y, "0.325 (ramp end)"))

    out_dir = os.path.join(apps.RESULTS, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    table = []
    for bench, task, m, C, labels, s in rows:
        prior = np.bincount(labels, minlength=C)
        mc = overlap_mc(labels, m, C, n_batches=20000)
        table.append({"benchmark": bench, "task": task, "m": m, "E_ov_true_prior": round(expected_overlap(m, prior, C), 4),
                      "E_ov_uniform_prior": round(expected_overlap(m, np.ones(C), C), 4), "sd": round(mc.std(), 4),
                      "q05": round(np.quantile(mc, 0.05), 4), "q95": round(np.quantile(mc, 0.95), 4), "mpot_tuned_s": s})
    with open(os.path.join(out_dir, "overlap_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    cols = list(table[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in table]
    open(os.path.join(out_dir, "overlap_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
