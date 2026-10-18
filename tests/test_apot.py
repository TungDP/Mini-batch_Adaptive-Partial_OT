"""Checks of the identities the rules rely on.  Run:  python tests/test_apot.py
(pytest-compatible as well)."""
import argparse
import os
import sys
import tempfile
import warnings

import numpy as np
import ot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from apot import overlap, rules, runlog, solvers  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def _batch(m=12, d=3, shift=1.0):
    X = rng.normal(size=(m, d))
    Y = rng.normal(size=(m, d)) + shift
    return ot.dist(X, Y)


def test_pot_curve_convex_piecewise_linear():
    C, m = _batch(), 12
    grid = np.arange(0, m + 1) / m
    c = np.array([0.0] + [float((solvers.plan_pot(C, s) * C).sum()) for s in grid[1:]])
    slopes = np.diff(c) * m
    assert np.all(np.diff(slopes) >= -1e-9)
    mid = (grid[:-1] + grid[1:]) / 2
    c_mid = np.array([float((solvers.plan_pot(C, s) * C).sum()) for s in mid])
    assert np.allclose(c_mid, (c[:-1] + c[1:]) / 2, atol=1e-9)


def test_price_equals_pot_plus_penalty():
    C = _batch()
    for lam in np.quantile(C, [0.1, 0.3, 0.6, 0.9]):
        P = solvers.plan_price(C, lam)
        s = P.sum()
        trunc = float((ot.emd(np.ones(12) / 12, np.ones(12) / 12, np.minimum(C, lam)) * np.minimum(C, lam)).sum())
        pot = float((solvers.plan_pot(C, s) * C).sum()) if s > 1e-12 else 0.0
        assert abs(trunc - (pot + lam * (1 - s))) < 1e-9


def test_aot_equals_inequality_lp():
    from scipy.optimize import linprog
    m = 10
    C = 0.3 * _batch(m) - 4.0 * rng.dirichlet(np.ones(4), m) @ rng.dirichlet(np.ones(4), m).T
    A = np.vstack([np.kron(np.eye(m), np.ones(m)), np.kron(np.ones(m), np.eye(m))])
    lp = linprog(C.ravel(), A_ub=A, b_ub=np.ones(2 * m) / m, bounds=(0, None), method="highs")
    P = solvers.plan_aot(C)
    assert abs(lp.fun - float((P * C).sum())) < 1e-9


def test_elbow_kneedle_matches_chord():
    C = _batch(20)
    args = argparse.Namespace(rule="elbow", eps=0.0, grid_lo=0.05, grid_hi=1.0, grid_n=20, elbow_mode="kneedle")
    s_knee = rules.build_rule(args).plan(rules.Batch(C=C))[1]["s_req"]
    args.elbow_mode = "chord"
    s_chord = rules.build_rule(args).plan(rules.Batch(C=C))[1]["s_req"]
    assert abs(s_knee - s_chord) <= 0.05 + 1e-9  # one grid step


def test_oracle_recovery_under_separation():
    K, n = 5, 30
    hits = 0
    for _ in range(50):
        centers = rng.normal(scale=20, size=(K, 4))
        ys = rng.integers(0, K, n)
        yt = rng.choice(K, n, p=rng.dirichlet(np.ones(K)))
        C = ot.dist(centers[ys] + rng.normal(scale=0.5, size=(n, 4)), centers[yt] + rng.normal(scale=0.5, size=(n, 4)))
        same = ys[:, None] == yt[None, :]
        lam = 0.5 * (C[same].max() + C[~same].min())
        P = solvers.plan_price(C, lam)
        hits += abs(P.sum() - overlap.batch_overlap(ys, yt, K)) < 1e-9 and P[~same].sum() < 1e-12
    assert hits == 50


def test_expected_overlap_closed_form_vs_mc():
    prior = np.array([3646, 3475, 4690, 10401, 4691, 2075, 5796, 4000, 4549, 2281, 4236, 5548], float)
    labels = np.repeat(np.arange(12), prior.astype(int))
    for m in (36, 72):
        cf = overlap.expected_overlap(m, prior)
        mc = overlap.overlap_mc(labels, m, 12, n_batches=4000).mean()
        assert abs(cf - mc) < 0.01, (m, cf, mc)
    # stratified source with a remainder (m not divisible by C)
    labels10 = rng.integers(0, 10, 50000)
    cf = overlap.expected_overlap(57, np.bincount(labels10, minlength=10))
    mc = overlap.overlap_mc(labels10, 57, 10, n_batches=4000).mean()
    assert abs(cf - mc) < 0.01, (cf, mc)


def test_rules_run_and_respect_mass():
    ns = argparse.Namespace(**{a.dest: a.default for a in rules.add_rule_args(argparse.ArgumentParser())._group_actions})
    m, K = 20, 4
    ys = np.repeat(np.arange(K), m // K)
    yt = rng.integers(0, K, m)
    pt = rng.dirichlet(np.ones(K), m)
    f = rng.normal(size=(m, 8))
    C = ot.dist(f, f + 0.3)
    b = rules.Batch(C=C, ys=ys, yt=yt, pt=pt, fs=f, ft=f + 0.3, C_signed=C - np.median(C), step=5, total_steps=100)
    for r in rules.RULES:
        ns.rule = r
        rule = rules.build_rule(ns, n_classes=K, prior_vec=np.ones(K) / K)
        P, info = rule.plan(b)
        assert P.shape == C.shape and np.all(P >= -1e-12), r
        assert P.sum(1).max() <= 1 / m + 1e-7 and P.sum(0).max() <= 1 / m + 1e-7 or r == "uot", r
        if "s_req" in info and r not in ("elbow",):
            assert abs(P.sum() - info["s_req"]) < 1e-6, (r, P.sum(), info["s_req"])
        name = runlog.run_name(rules.rule_name_fields(ns))
        assert name.startswith("rule" + r)


def test_ramp_matches_official_pda_schedule():
    """Official run_mOT.py: s = mass/T * i + 0.01 for i <= T/2, then mass/2 (mass = 0.65)."""
    T, mass = 5000, 0.65
    ns = argparse.Namespace(rule="ramp", eps=0.0, mass=0.325, ramp_s0=0.01, ramp_frac=0.5, warm_frac=0.0)
    rule = rules.build_rule(ns)
    gap = 0.0
    for i in range(0, T + 1, 50):
        official = mass / T * i + 0.01 if i <= T / 2 else mass / 2
        ours = rule.mass(rules.Batch(C=None, step=i, total_steps=T))
        gap = max(gap, abs(official - ours))
    assert gap <= 0.0101, gap


def test_warm_frac_scales_every_mass_rule():
    m, K, T = 20, 4, 1000
    ns = argparse.Namespace(**{a.dest: a.default for a in rules.add_rule_args(argparse.ArgumentParser())._group_actions})
    ys = np.repeat(np.arange(K), m // K)
    yt = rng.integers(0, K, m)
    f = rng.normal(size=(m, 8))
    C = ot.dist(f, f + 0.3)
    ns.warm_frac = 0.5
    for r in ["fixed", "oracle", "expected", "pseudo", "elbow", "curv", "cluster", "price"]:
        ns.rule = r
        if r == "price":
            ns.target_mass = 0.6
        rule = rules.build_rule(ns, n_classes=K, prior_vec=np.ones(K) / K)
        masses = []
        for step in (0, 125, 250, 500, 900):
            b = rules.Batch(C=C, ys=ys, yt=yt, pt=rng.dirichlet(np.ones(K), m), fs=f, ft=f + 0.3, step=step, total_steps=T)
            masses.append(rule.plan(b)[0].sum())
        if r != "price":  # price adapts lam by dual ascent, so its mass lags the target
            assert masses[0] <= 0.0101 + 1e-9 and masses[1] <= masses[3] + 1e-9, (r, masses)
        assert runlog.run_name(rules.rule_name_fields(ns)).endswith("_wf0.5_eps0"), r


def test_run_naming_and_skip():
    assert runlog.fmt(1) == runlog.fmt(1.0) == "1"
    assert runlog.fmt(0.90) == "0.9" and runlog.fmt(4e-4) == "0.0004"
    with tempfile.TemporaryDirectory() as d:
        r = runlog.open_run(d, [("a", 1), ("b", 0.5)], {"x": 1})
        assert r is not None and os.path.basename(r.dir) == "a1_b0.5"
        assert runlog.open_run(d, [("a", 1.0), ("b", 0.5)], {"x": 1}) is None  # held by us
        r.finish({"acc": 0.5})
        r.release()
        assert runlog.open_run(d, [("a", 1.0), ("b", 0.50)], {"x": 1}) is None  # done -> skip
        assert os.path.exists(os.path.join(r.dir, "done.json"))


def test_e4_control_solver_and_table_names():
    """E4 controls keep the adaptive run's solver; tables keep mpot / mpot_ent, oracle / oracle_nw apart."""
    root = os.path.join(os.path.dirname(__file__), "..")
    sys.path[:0] = [os.path.join(root, "scripts"), os.path.join(root, "src", "analysis")]
    import apps
    import collect
    import experiments
    names = collect.plan_methods()
    for j in experiments.e1_gonogo((1,)):
        assert names[os.path.basename(apps.run_dir(j.app, j.argv))] == j.meta["method"], j.label
        if j.meta["method"] in experiments.ADAPTIVE:
            a, ctl = apps.parse(j.app, j.argv), apps.parse(j.app, experiments.matched_argv(j, 0.3))
            assert (ctl.rule, ctl.mass, ctl.eps) == ("fixed", 0.3, a.eps), j.label


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
