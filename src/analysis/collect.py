"""Collect finished runs into tables.

    python src/analysis/collect.py                    # all apps -> results/analysis/*.csv, *.md
    python src/analysis/collect.py --metric best_acc  # DA tables on best instead of last accuracy

Reads config.json + done.json of every finished run under results/{DeepDA,DeepGM,GradientFlow,
ColorTransfer}.  Writes a long CSV of all runs, one task x method table per benchmark (mean and
sd over seeds), a mass-diagnostics table (mean s, its sd, Spearman rho with the true batch
overlap, cross-class mass), and an adaptive-vs-matched-mass table for E4.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import apps  # noqa: E402


def method_label(c):
    tag = c.get("tag") or ""
    if tag.startswith("mm-"):
        return tag
    r = c["rule"]
    if r == "ot":
        return "mot"
    if r == "uot":
        return "muot"
    if r in ("fixed", "ramp"):
        return f"{'mpot' if r == 'fixed' else 'ramp'}{c['mass']:g}"
    if r == "elbow":
        return "elbow" + ("_chord" if c.get("elbow_mode") == "chord" else "")
    if r == "expected":
        return f"exp_{c['prior']}"
    if r == "price":
        return f"price_tm{c['target_mass']:g}" if c.get("target_mass", -1) >= 0 else f"price_k{c['kappa']:g}"
    return r


def plan_methods():
    """Run directory name -> method name in scripts/experiments.py.  The config alone cannot tell
    mpot from mpot_ent or oracle from oracle_nw (they differ only in eps / warm-up)."""
    sys.path.insert(0, os.path.join(apps.REPO, "scripts"))
    import experiments as E
    # ponytail: default seeds only (E1 up to 3); runs launched with other --seeds fall back to method_label
    jobs = E.e1_gonogo((1, 2, 3))
    for name in ("E2", "E3", "PDACHECK", "E5", "E6"):
        jobs += E.EXPERIMENTS[name](with_=("aot", "celeba"))
    return {os.path.basename(apps.run_dir(j.app, j.argv)): j.meta["method"] for j in jobs}


def benchmark_task(app, c):
    if app == "digits":
        return "digits", f"{c['source'][0].upper()}2{c['target'][0].upper()}"
    if app == "office":
        if c["dset"] == "visda":
            return f"visda_m{c['m']}", "S2R"
        return ("oh_pda" if c.get("n_shared") else "office-home"), f"{c['s'][0]}2{c['t'][0]}"
    if app == "dgm":
        return "dgm", c["dataset"]
    if app == "flow":
        return "flow", "flow"
    return "color", f"{os.path.splitext(c['source'])[0]}->{os.path.splitext(c['target'])[0]}"


PRIMARY = {"digits": "last_acc", "office": "last_acc", "dgm": "best_fid", "flow": "final_w2", "color": "w2_to_target"}


def infer_app(c):
    if "n_colors" in c:
        return "color"
    if "eval_every" in c:
        return "flow"
    if "latent" in c:
        return "dgm"
    if "dset" in c:
        return "office"
    return "digits"


def load(root):
    rows, plan = [], plan_methods()
    for done in glob.glob(os.path.join(root, "**", "done.json"), recursive=True):
        d = os.path.dirname(done)
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        c = json.load(open(os.path.join(d, "config.json")))
        r = json.load(open(done))
        app = infer_app(c)
        bench, task = benchmark_task(app, c)
        tag = c.get("tag") or ""
        method = plan.get(os.path.basename(d)) or method_label(c)
        rows.append(dict(app=app, bench=bench, task=task, method=method, seed=c["seed"],
                         tag="" if tag.startswith("mm-") else tag, run=os.path.basename(d), **r))
    return rows


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r if not isinstance(r[k], (list, dict))})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pivot(rows, metric, scale):
    cells = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            cells[(r["task"], r["method"])].append(r[metric] * scale)
    tasks = sorted({t for t, _ in cells})
    methods = sorted({mth for _, mth in cells})
    lines = ["| task | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    means = defaultdict(list)
    for t in tasks:
        row = []
        for mth in methods:
            v = cells.get((t, mth))
            if v:
                means[mth].append(np.mean(v))
                row.append(f"{np.mean(v):.2f} ± {np.std(v):.2f} ({len(v)})")
            else:
                row.append("–")
        lines.append(f"| {t} | " + " | ".join(row) + " |")
    if len(tasks) > 1:
        lines.append("| **avg** | " + " | ".join(f"{np.mean(means[mth]):.2f}" if len(means[mth]) == len(tasks)
                                               else "–" for mth in methods) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=apps.RESULTS)
    p.add_argument("--metric", default=None, help="override the primary metric of every app")
    p.add_argument("--tag", default="", help="only runs with this tag ('' = untagged protocol runs)")
    a = p.parse_args()
    rows = [r for r in load(a.root) if r["tag"] == a.tag]
    out = os.path.join(apps.RESULTS, "analysis")
    os.makedirs(out, exist_ok=True)
    if not rows:
        print("no finished runs")
        return
    write_csv(os.path.join(out, "runs.csv"), rows)
    md = []
    for bench in sorted({r["bench"] for r in rows}):
        sub = [r for r in rows if r["bench"] == bench]
        metric = a.metric or PRIMARY[sub[0]["app"]]
        scale = 100.0 if metric.endswith("acc") or metric.startswith("acc") else 1.0
        md += [f"## {bench} — {metric}{' (%)' if scale == 100 else ''}, mean ± sd (n seeds)", "", pivot(sub, metric, scale), ""]
        diag = defaultdict(lambda: defaultdict(list))
        for r in sub:
            for key in ("mean_s", "sd_s", "rho_s_ov", "mean_cross", "mean_plan_ms"):
                if r.get(key) is not None:
                    diag[r["method"]][key].append(r[key])
        md += [f"### {bench} — mass diagnostics (averaged over runs)", "",
               "| method | mean s | sd s | rho(s, true overlap) | cross-class mass | plan ms |", "|---|---|---|---|---|---|"]
        for mth in sorted(diag):
            dd = diag[mth]
            md.append(f"| {mth} | " + " | ".join(f"{np.mean(dd[k]):.3f}" if dd.get(k) else "–"
                                                 for k in ("mean_s", "sd_s", "rho_s_ov", "mean_cross", "mean_plan_ms")) + " |")
        md.append("")
        pairs = defaultdict(dict)  # adaptive vs its matched-mass control, same task and seed
        for r in sub:
            key = (r["task"], r["seed"])
            if r["method"].startswith("mm-"):
                pairs[key].setdefault(r["method"][3:], {})["mm"] = r.get(metric)
            else:
                pairs[key].setdefault(r["method"], {})["ad"] = r.get(metric)
        diffs = defaultdict(list)
        for key, by_m in pairs.items():
            for mth, v in by_m.items():
                if v.get("ad") is not None and v.get("mm") is not None:
                    diffs[mth].append((v["ad"] - v["mm"]) * scale)
        if diffs:
            md += [f"### {bench} — adaptive minus fixed-at-same-mean-mass (E4)", "",
                   "| method | mean diff | sd | n pairs |", "|---|---|---|---|"]
            md += [f"| {mth} | {np.mean(v):+.2f} | {np.std(v):.2f} | {len(v)} |" for mth, v in sorted(diffs.items())]
            md.append("")
    open(os.path.join(out, "tables.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
