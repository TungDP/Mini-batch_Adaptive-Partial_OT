"""The experiment plan.  Every experiment is a list of Jobs; scripts/launch.py runs them.

    E1  go/no-go      does per-batch mass beat the best fixed mass?  (VisDA m=36/72, 4 PDA tasks)
    E2  da            main DA tables: Digits, Office-Home, VisDA
    E3  pda           partial DA: Office-Home 65 -> 25, 12 tasks
    E4  matched       fixed s at each adaptive run's own mean mass (allocation vs amount)
    E5  dgm           generative model: CIFAR-10 (+ CelebA with --with celeba)
    E6  flow_color    gradient flow + color transfer (CPU)

Hyperparameters follow m-POT (Nguyen et al., ICML 2022, App. D) and its reference code;
the m-POT-tuned masses are the fixed-s baselines.  Exact LP (eps = 0) everywhere except m-UOT,
the `mpot_ent` reproduction rows and the partial-DA partial rules (PDA_EPS).
"""
import json
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import apps  # noqa: E402
from apot.overlap import expected_overlap  # noqa: E402


@dataclass
class Job:
    app: str
    argv: list
    hours: float          # rough wall time on one RTX 5080 (for planning only)
    label: str = ""
    gpu: bool = True
    meta: dict = field(default_factory=dict)


OH = ["Art", "Clipart", "Product", "Real_World"]
OH_TASKS = [(s, t) for s in OH for t in OH if s != t]
PDA_GONOGO = [("Art", "Clipart"), ("Clipart", "Real_World"), ("Product", "Art"), ("Real_World", "Product")]
DIGITS = [("svhn", "mnist", 0.85), ("usps", "mnist", 0.9), ("mnist", "usps", 0.8)]  # m-POT Table 5 masses
VISDA_S = {36: 0.70, 72: 0.75, 144: 0.80}                                          # m-POT Fig. 13 (right)
OH_S, PDA_S, DGM_S = 0.6, 0.325, 0.7  # TS-POT Office-Home; PDA ramp end point; m-KOT DGM protocol

BASE = {
    # Digits = official DeepDA/digits/sh/exp_mPOT.sh: m 500, 100 epochs, lr 4e-4, eta 0.1 / 0.1
    # (+ 10 source-only epochs inside train_digits.py).
    "digits": ["--m", "500", "--k", "1", "--n_epochs", "100", "--src_epochs", "10", "--lr", "0.0004",
               "--eta1", "0.1", "--eta2", "0.1"],
    "oh": ["--dset", "office-home", "--m", "65", "--k", "1", "--iters", "10000", "--test_interval", "500",
           "--test_10crop", "1", "--eta1", "0.01", "--eta2", "0.5", "--eta3", "1", "--bottleneck", "512"],
    "visda": ["--dset", "visda", "--k", "1", "--iters", "10000", "--test_interval", "1000", "--test_10crop", "0",
              "--eta1", "0.005", "--eta2", "1", "--eta3", "1", "--bottleneck", "512"],
    # Partial DA = the official m-POT PartialDA code (run_mOT.py + sh/exp_mPOT.sh, checked against
    # GitHub 2026-10-11): 5000 steps, eval every 500 single-crop, Resize+RandomCrop augmentation,
    # bottleneck 256, cost weights 0.03 / 7.5 / 1 for POT (m-OT and m-UOT override them below).
    "pda": ["--dset", "office-home", "--n_shared", "25", "--m", "65", "--k", "1", "--iters", "5000",
            "--test_interval", "500", "--test_10crop", "0", "--eta1", "0.03", "--eta2", "7.5", "--eta3", "1",
            "--bottleneck", "256", "--aug", "crop"],
    "cifar10": ["--dataset", "cifar10", "--m", "100", "--k", "2", "--epochs", "100", "--latent", "32",
                "--lr", "0.0005", "--fid_every", "10"],
    "celeba": ["--dataset", "celeba", "--m", "100", "--k", "2", "--epochs", "50", "--latent", "32",
               "--lr", "0.0005", "--fid_every", "5"],
    "flow": ["--n", "1000", "--m", "16", "--k", "4", "--lr", "0.001", "--T", "5"],
    "color": ["--n_colors", "3000", "--m", "100", "--k", "1", "--T", "10000"],
}
HOURS = {"digits": 0.45, "oh": 1.4, "visda": 1.3, "pda": 0.62, "cifar10": 1.0, "celeba": 4.0, "flow": 0.1, "color": 0.15}
ENT_EPS = {"digits": "0.1", "visda": "0.004"}  # official exp_mPOT.sh / paper App. D.1
# Solver of every partial / adaptive rule on PDA.  "0" = exact LP; "0.5" = the published m-POT
# PDA setting (entropic, official exp_mPOT.sh).  `mpot` stays exact and `mpot_ent` entropic
# either way, so both solver variants of the baseline are always run.
PDA_EPS = "0.5"  # set 2026-10-11: entropic beat exact by 1-3 points on A2C / C2A / P2R
UOT = {"digits": ("0.1", "1"), "oh": ("0.01", "0.5"), "visda": ("0.01", "0.3"), "pda": ("0.01", "0.06")}  # (eps, tau)
# Wall-time multipliers of the cost-curve / K-means rules (measured): 20 partial-OT solves per
# step cost ~2 s at m = 500 (digits) but are hidden behind the ResNet step at m = 65-72.
SLOW = {"digits": {"elbow": 11.0, "curv": 11.0, "cluster": 1.5},
        "pda": {"elbow": 3.5, "curv": 3.5, "cluster": 1.05}}  # 20 entropic solves (~60-80 ms each) per step
SLOW_DEFAULT = {"elbow": 1.05, "curv": 1.05, "cluster": 1.05}


def slow(proto, method):
    return SLOW.get(proto, SLOW_DEFAULT).get(method, 1.0)


def f(x):
    return f"{x:.4g}"


def methods(proto, s_tuned, m, n_classes):
    """Method name -> rule argv for one protocol.  s_tuned is m-POT's mass for this task.

    Partial DA: m-OT and m-UOT use their own official cost weights; m-POT is the official ramp
    0.01 -> 0.325 over the first half (exact LP as `mpot`, the paper's entropic eps = 0.5 as
    `mpot_ent`).  A fixed mass from step 0 collapses on PDA (the source-private classes take the
    mass while the head is untrained), so every adaptive rule gets the same warm-up shape
    (--warm_frac 0.5: its mass x min(1, t / 2500)); the `_nw` variants drop it."""
    pda = proto == "pda"
    ov_u = expected_overlap(m, [1.0] * n_classes)  # E[class overlap] under balanced target classes
    s_tm = PDA_S if pda else ov_u                  # average-mass target of the price rule
    W = ["--warm_frac", "0.5"] + (["--eps", PDA_EPS] if PDA_EPS != "0" else []) if pda else []
    E = ["--eps", PDA_EPS] if pda and PDA_EPS != "0" else []  # solver for the no-warm-up PDA variants
    warm = "0" if proto == "digits" or pda else "500"
    eps_u, tau_u = UOT[proto]
    return {
        "mot": ["--rule", "ot"] + (["--eta1", "0.001", "--eta2", "0.0001", "--eta3", "1"] if pda else []),
        "muot": ["--rule", "uot", "--eps", eps_u, "--tau", tau_u]
                + (["--eta1", "0.003", "--eta2", "0.75", "--eta3", "10"] if pda else []),
        "mpot": ["--rule", "ramp", "--mass", f(s_tuned)] if pda else ["--rule", "fixed", "--mass", f(s_tuned)],
        # the published m-POT solver setting, as a reproduction check (Office-Home's official is exact)
        "mpot_ent": (["--rule", "ramp", "--mass", f(s_tuned), "--eps", "0.5"] if pda else
                     ["--rule", "fixed", "--mass", f(s_tuned), "--eps", ENT_EPS.get(proto, "0")]),
        "mpot_fixed": ["--rule", "fixed", "--mass", f(s_tuned)] + E,
        "elbow": ["--rule", "elbow"] + W,
        "curv": ["--rule", "curv"] + W,
        "cluster": ["--rule", "cluster"] + W,
        "exp_u": ["--rule", "expected", "--prior", "uniform"] + W,
        "exp_ema": ["--rule", "expected", "--prior", "ema"] + W,
        "exp_ema_nw": ["--rule", "expected", "--prior", "ema"] + E,
        "pseudo": ["--rule", "pseudo", "--pseudo_counts", "hard", "--warmup", warm] + W,
        "price_tm": ["--rule", "price", "--kappa", "0.5", "--target_mass", f(s_tm)] + W,
        "oracle": ["--rule", "oracle"] + W,
        "oracle_nw": ["--rule", "oracle"] + E,
        "aot": ["--rule", "aot", "--aot_alpha", "0.01", "--aot_beta", "5"],
    }


def _jobs(proto, tasks, names, seeds, m=None, extra=()):
    app = "digits" if proto == "digits" else "office"
    out = []
    for task in tasks:
        if proto == "digits":
            src, tgt, s_tuned = task
            targv, n_cls, mm = ["--source", src, "--target", tgt], 10, 500
        elif proto == "visda":
            s_tuned, targv, n_cls, mm = VISDA_S[m], ["--m", str(m)], 12, m
        else:
            src, tgt = task
            s_tuned = PDA_S if proto == "pda" else OH_S
            targv, n_cls, mm = ["--s", src, "--t", tgt], 65, 65
        meths = methods(proto, s_tuned, mm, n_cls)
        for name in names:
            for seed in seeds:
                argv = BASE[proto] + targv + meths[name] + ["--seed", str(seed)] + list(extra)
                hours = HOURS[proto] * slow(proto, name)
                out.append(Job(app, argv, hours, label=f"{proto}:{'-'.join(map(str, task if proto != 'visda' else ('m', m)))}:{name}:s{seed}",
                               meta={"proto": proto, "method": name, "seed": seed}))
    return out


def e1_gonogo(seeds=(1, 2), **_):
    names_v = ["mpot", "exp_u", "exp_ema", "pseudo", "oracle"]
    names_p = ["mpot", "mpot_ent", "mpot_fixed", "oracle", "oracle_nw", "exp_ema", "pseudo"]
    jobs = []
    for seed in seeds:  # seed-major: a complete seed-1 picture arrives first
        for m in (36, 72):
            jobs += _jobs("visda", [("train", "validation")], names_v, (seed,), m=m)
        jobs += _jobs("pda", PDA_GONOGO, names_p, (seed,))
    return jobs


DA_METHODS = ["mot", "muot", "mpot", "elbow", "curv", "cluster", "exp_u", "exp_ema", "pseudo", "price_tm", "oracle"]


def e2_da(seeds=(1, 2, 3), with_=(), **_):
    names = DA_METHODS + (["aot"] if "aot" in with_ else [])
    return (_jobs("digits", DIGITS, names + ["mpot_ent"], seeds)
            + _jobs("visda", [("train", "validation")], names + ["mpot_ent"], seeds, m=72)
            + _jobs("oh", OH_TASKS, names, seeds))


def e3_pda(seeds=(1, 2, 3), with_=(), **_):
    names = ["mot", "muot", "mpot", "mpot_ent", "elbow", "curv", "cluster", "exp_ema", "pseudo", "price_tm", "oracle"]
    return _jobs("pda", OH_TASKS, names + (["aot"] if "aot" in with_ else []), seeds)


ADAPTIVE = ["elbow", "curv", "cluster", "exp_ema", "pseudo", "price_tm", "oracle", "oracle_nw", "exp_ema_nw", "muot"]


def e4_matched(seeds=(1, 2, 3), with_=(), **_):
    """Fixed s at the mean mass of every FINISHED adaptive run of E1-E3 (same task and seed).
    If the fixed control matches the adaptive run, the gain came from the amount of mass, not
    from its per-batch allocation."""
    jobs = []
    for job in e1_gonogo(seeds) + e2_da(seeds, with_) + e3_pda(seeds, with_):
        if job.meta["method"] not in ADAPTIVE:
            continue
        done = os.path.join(apps.run_dir(job.app, job.argv), "done.json")
        if not os.path.exists(done):
            continue
        summary = json.load(open(done))
        warmed = "--warm_frac" in job.argv
        # with a warm-up, match the mass after it (the control gets the same warm-up)
        mean_s = summary["mean_s_late"] if warmed else summary["mean_s"]
        jobs.append(Job(job.app, matched_argv(job, mean_s), job.hours / slow(job.meta["proto"], job.meta["method"]),
                        label=job.label.replace(job.meta["method"], f"mm-{job.meta['method']}"),
                        meta=dict(job.meta, method=f"mm-{job.meta['method']}", matched_to=job.meta["method"])))
    return jobs


def matched_argv(job, mean_s):
    """argv of the fixed-mass control of an adaptive job: same protocol, mass mean_s.  On PDA it
    gets the warm-up (a fixed mass from step 0 collapses) and the solver every PDA partial rule
    uses; _replace_rule strips --eps, which would otherwise leave the control on the exact LP."""
    pda = job.meta["proto"] == "pda"
    extra = ["--warm_frac", "0.5"] if pda and "--warm_frac" not in job.argv else []
    extra += ["--eps", PDA_EPS] if pda and PDA_EPS != "0" else []
    return _replace_rule(job.argv, ["--rule", "fixed", "--mass", f"{mean_s:.3f}", "--tag",
                                    f"mm-{job.meta['method']}"] + extra)


def _replace_rule(argv, new_rule):
    rule_flags = {"--rule", "--mass", "--eps", "--tau", "--prior", "--pseudo_counts", "--warmup", "--kappa",
                  "--target_mass", "--aot_alpha", "--aot_beta", "--elbow_mode", "--tag"}
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in rule_flags:
            skip = True
            continue
        out.append(a)
    return out + new_rule


def e5_dgm(seeds=(16, 17, 18), with_=(), **_):
    rules = {"mot": ["--rule", "ot"], "mpot": ["--rule", "fixed", "--mass", f(DGM_S)],
             "elbow": ["--rule", "elbow"], "curv": ["--rule", "curv"], "cluster": ["--rule", "cluster"],
             "price_tm": ["--rule", "price", "--kappa", "0.5", "--target_mass", f(DGM_S)],
             "price_k0.5": ["--rule", "price", "--kappa", "0.5"], "price_k1": ["--rule", "price", "--kappa", "1"]}
    jobs = []
    for ds in ["cifar10"] + (["celeba"] if "celeba" in with_ else []):
        for name, r in rules.items():
            for seed in seeds:
                jobs.append(Job("dgm", BASE[ds] + r + ["--seed", str(seed)], HOURS[ds] * slow("dgm", name),
                                label=f"{ds}:{name}:s{seed}", meta={"proto": ds, "method": name, "seed": seed}))
    return jobs


def e6_flow_color(seeds=(1, 2, 3), **_):
    rules = {"mot": ["--rule", "ot"], "mpot_0.9": ["--rule", "fixed", "--mass", "0.9"],
             "mpot_0.8": ["--rule", "fixed", "--mass", "0.8"], "elbow": ["--rule", "elbow"], "curv": ["--rule", "curv"],
             "cluster": ["--rule", "cluster"], "price_tm0.8": ["--rule", "price", "--kappa", "0.5", "--target_mass", "0.8"],
             "price_k0.5": ["--rule", "price", "--kappa", "0.5"], "price_k1": ["--rule", "price", "--kappa", "1"]}
    jobs = []
    for name, r in rules.items():
        for seed in seeds:
            jobs.append(Job("flow", BASE["flow"] + r + ["--seed", str(seed)], HOURS["flow"], f"flow:{name}:s{seed}",
                            gpu=False, meta={"proto": "flow", "method": name, "seed": seed}))
    for src, tgt in [("starry_night.jpg", "ste_victoire.jpg"), ("ocean_day.jpg", "ocean_sunset.jpg")]:
        for name, r in rules.items():
            jobs.append(Job("color", BASE["color"] + ["--source", src, "--target", tgt] + r + ["--seed", "1"],
                            HOURS["color"], f"color:{src[:-4]}:{name}", gpu=False,
                            meta={"proto": "color", "method": name, "seed": 1}))
    return jobs


def smoke(**_):
    """Two-minute end-to-end check of every trainer through the launcher (results/_smoke)."""
    out = ["--out_root", os.path.join(apps.RESULTS, "_smoke"), "--tag", "smoke"]
    jobs = [Job("digits", ["--source", "usps", "--target", "mnist", "--n_epochs", "1", "--src_epochs", "1",
                           "--rule", "pseudo"] + out, 0.02, "smoke:digits:pseudo"),
            Job("office", BASE["pda"] + ["--s", "Art", "--t", "Clipart", "--iters", "40", "--test_interval", "40",
                                         "--rule", "expected", "--prior", "ema"] + out, 0.05, "smoke:pda:exp_ema"),
            Job("office", ["--dset", "visda", "--m", "36", "--iters", "40", "--test_interval", "40", "--test_10crop", "0",
                           "--rule", "price", "--target_mass", "0.7"] + out, 0.1, "smoke:visda:price_tm"),
            Job("dgm", ["--dataset", "cifar10", "--epochs", "1", "--k", "2", "--fid_every", "1", "--fid_n", "2000",
                        "--rule", "elbow", "--elbow_mode", "chord"] + out, 0.05, "smoke:cifar10:elbow"),
            Job("flow", BASE["flow"] + ["--T", "0.3", "--rule", "curv"] + out, 0.01, "smoke:flow:curv", gpu=False),
            Job("color", ["--source", "ocean_day.jpg", "--target", "ocean_sunset.jpg", "--n_colors", "500", "--T", "200",
                          "--rule", "cluster"] + out, 0.01, "smoke:color:cluster", gpu=False)]
    for j in jobs:
        j.meta = {"proto": "smoke", "method": j.label.split(":")[-1], "seed": 1}
    return jobs


def pda_check(seeds=(1,), **_):
    """Reproduction check of m-POT on partial DA: the official ramp schedule, exact LP and the
    paper's entropic eps = 0.5, on the three pairs used to diagnose the m-KOT PDA collapse.
    Paper (Table 4): A2C 64.60, C2A 76.43, P2R 87.63."""
    pairs = [("Art", "Clipart"), ("Clipart", "Art"), ("Product", "Real_World")]
    return _jobs("pda", pairs, ["mpot"], seeds) + _jobs("pda", pairs, ["mpot_ent"], seeds)


EXPERIMENTS = {"SMOKE": smoke, "PDACHECK": pda_check, "E1": e1_gonogo, "E2": e2_da, "E3": e3_pda, "E4": e4_matched, "E5": e5_dgm, "E6": e6_flow_color}
