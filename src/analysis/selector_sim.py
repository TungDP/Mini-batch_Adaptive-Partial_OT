"""Controlled mini-batch simulation of the mass rules (no network): which rule avoids
class-level misspecified matches, at what average mass, and at what cost?

Features are a fixed snapshot: C Gaussian classes in R^d.  Source batches are class-stratified,
target batches are drawn from a long-tailed, a partial, or a shifted target, exactly as the
m-POT loaders do.  Uses the same rule implementations (src/apot) as the trainers.

    python src/analysis/selector_sim.py                 # writes results/analysis/selector_sim.csv
"""
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import ot
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import apps  # noqa: E402
from apot import rules as R  # noqa: E402
from apot.overlap import batch_overlap, expected_overlap  # noqa: E402

warnings.filterwarnings("ignore")


def make_domain(rng, regime, C, d, sep):
    centers = rng.normal(scale=sep, size=(C, d))
    pert = rng.normal(size=(C, d))
    pert /= np.linalg.norm(pert, axis=1, keepdims=True)
    shift = rng.normal(size=d)
    shift /= np.linalg.norm(shift)
    if regime == "partial":
        prior = np.zeros(C)
        prior[: max(1, C * 2 // 5)] = 1.0
    else:  # long-tailed, imbalance ratio 10
        prior = 10.0 ** (-np.arange(C) / (C - 1))
    prior /= prior.sum()
    return centers, centers + pert + shift * (8.0 if regime == "shifted" else 1.0), prior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=10)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--m", type=int, default=50)
    p.add_argument("--B", type=int, default=300, help="batch pairs per regime")
    p.add_argument("--sep", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    grid = np.linspace(0.05, 1.0, 20)
    out_rows = []
    for regime in ["longtail", "partial", "shifted"]:
        centers, tcent, prior = make_domain(rng, regime, a.C, a.d, a.sep)
        src_means = np.stack([(centers[c] + rng.normal(size=(200, a.d))).mean(0) for c in range(a.C)])
        batches = []
        for _ in range(a.B):
            ys = np.repeat(np.arange(a.C), a.m // a.C)
            yt = rng.choice(a.C, len(ys), p=prior)
            Xs = centers[ys] + rng.normal(size=(len(ys), a.d))
            Xt = tcent[yt] + rng.normal(size=(len(ys), a.d))
            Craw = ot.dist(Xs, Xt)
            logits = -ot.dist(Xt, src_means)
            pt = np.exp(logits - logits.max(1, keepdims=True))
            pt /= pt.sum(1, keepdims=True)
            batches.append(dict(b=R.Batch(C=Craw / np.median(Craw), ys=ys, yt=yt, pt=pt, fs=Xs, ft=Xt),
                                same=ys[:, None] == yt[None, :], ov=batch_overlap(ys, yt, a.C)))
        ov = np.array([x["ov"] for x in batches])
        ov_u = expected_overlap(len(ys), np.ones(a.C))
        cands = {f"fixed s={s:.1f}": R.FixedRule(s) for s in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
        cands.update({
            "elbow (Alg.1)": R.ElbowRule(grid), "elbow chord": R.ElbowRule(grid, mode="chord"),
            "curvature (Alg.2)": R.CurvatureRule(grid), "cluster (Alg.3)": R.ClusterRule(a.C),
            "expected (uniform)": R.ExpectedRule(a.C), "expected (ema)": R.ExpectedRule(a.C, prior="ema", ema=0.9),
            "pseudo (hard)": R.PseudoRule(a.C), "oracle": R.OracleRule(a.C),
            f"price target={ov.mean():.2f}": R.PriceRule(0.5, target_mass=float(ov.mean()), dual_lr=0.2),
            f"fixed s=E[ov]={ov.mean():.2f}": R.FixedRule(float(ov.mean())),
        })
        print(f"\n=== {regime}: E[ov]={ov.mean():.3f} (sd {ov.std():.3f}), E[ov] uniform prior={ov_u:.3f}")
        print(f"{'rule':26s} {'mass':>6s} {'cross':>7s} {'recall':>7s} {'rho':>6s} {'ms':>7s}")
        for name, rule in cands.items():
            tot = cross = intra = 0.0
            s_list, t0 = [], time.perf_counter()
            for step, x in enumerate(batches):
                x["b"].step = step
                P, _ = rule.plan(x["b"])
                s = P.sum()
                s_list.append(s)
                tot += s
                cross += P[~x["same"]].sum()
                intra += P[x["same"]].sum()
            ms = 1000 * (time.perf_counter() - t0) / len(batches)
            s_arr = np.array(s_list)
            rho = spearmanr(s_arr, ov)[0] if s_arr.std() > 1e-9 else float("nan")
            row = {"regime": regime, "rule": name, "mass": tot / len(batches), "cross": cross / max(tot, 1e-12),
                   "recall": intra / ov.sum(), "rho": rho, "ms": ms}
            out_rows.append(row)
            print(f"{name:26s} {row['mass']:6.3f} {row['cross']:7.3f} {row['recall']:7.3f} {rho:6.2f} {ms:7.1f}")
    out = os.path.join(apps.RESULTS, "analysis", "selector_sim.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
