"""Transport rules: how much mass each mini-batch pair moves, and the plan that moves it.

A rule maps a ``Batch`` (cost matrix plus whatever side information it needs) to a plan.
Mass-based rules pick s and solve partial OT; the other rules (balanced, KL-UOT, price, AOT)
produce the plan directly and the moved mass is read off ``P.sum()``.

    rule            mass per batch                                   reads
    ot              1 (balanced m-OT)                                -
    uot             implicit (KL marginal penalty tau)               -
    fixed           s (m-POT)                                        -
    ramp            s0 -> s linearly over a fraction of training     -
    elbow           Kneedle knee of the POT cost curve (draft Alg.1) -
    curv            max 2nd difference of the cost curve (Alg.2)     -
    cluster         1 - TV of K-means cluster histograms (Alg.3)     fs, ft
    oracle          1 - TV of true label histograms (diagnostic)     ys, yt
    pseudo          1 - TV with predicted target labels              ys, pt
    expected        E[class overlap] under a class prior             ys (+ pt for 'ema')
    price           all pairs cheaper than a price lam (TV-UOT)      -
    aot             price rule on a mixed-sign cost, lam = 0         C_signed
    ppot            EMA of the confident-target fraction (PPOT)      pt
"""
from dataclasses import dataclass

import numpy as np

from . import solvers
from .overlap import batch_overlap, expected_overlap, soft_overlap

WARM_S0 = 0.01  # start of the m-POT partial-DA ramp


@dataclass
class Batch:
    C: np.ndarray                # cost used to choose the mass and to solve the plan
    ys: np.ndarray = None        # source labels
    yt: np.ndarray = None        # target labels: oracle rule and diagnostics only
    pt: np.ndarray = None        # target class probabilities (n_t x K)
    fs: np.ndarray = None        # source features (cluster rule)
    ft: np.ndarray = None        # target features
    C_signed: np.ndarray = None  # mixed-sign cost (aot rule)
    step: int = 0
    total_steps: int = 1


class Rule:
    name = ""
    needs = ()
    loss_on_signed = False  # True: the training loss uses C_signed instead of C
    warm_frac = 0.0         # > 0: mass x min(1, t / (warm_frac T)) with floor WARM_S0 (m-POT PDA ramp shape)

    def __init__(self, eps=0.0):
        self.eps = eps

    def warm(self, b):
        if self.warm_frac <= 0:
            return 1.0
        return min(1.0, b.step / max(1.0, self.warm_frac * b.total_steps))

    def _warmed(self, s, b):
        w = self.warm(b)
        return s if w >= 1.0 else max(WARM_S0, s * w)

    def mass(self, b):
        raise NotImplementedError

    def plan(self, b):
        s = float(np.clip(self._warmed(self.mass(b), b), 0.0, 1.0))
        return solvers.plan_pot(b.C, s, self.eps), {"s_req": s}


class OTRule(Rule):
    name = "ot"

    def plan(self, b):
        return solvers.plan_ot(b.C, self.eps), {"s_req": 1.0}


class UOTRule(Rule):
    name = "uot"

    def __init__(self, eps, tau):
        super().__init__(eps)
        self.tau = tau

    def plan(self, b):
        return solvers.plan_uot(b.C, self.eps, self.tau), {}


class FixedRule(Rule):
    name = "fixed"

    def __init__(self, s, eps=0.0):
        super().__init__(eps)
        self.s = s

    def mass(self, b):
        return self.s


class RampRule(Rule):
    """Linear ramp s0 -> s over the first ramp_frac of training, then constant (the schedule
    of the m-POT partial-DA code and of WARMPOT)."""

    name = "ramp"

    def __init__(self, s, eps=0.0, s0=0.01, ramp_frac=0.5):
        super().__init__(eps)
        self.s, self.s0, self.ramp_frac = s, s0, ramp_frac

    def mass(self, b):
        t = min(1.0, b.step / max(1.0, self.ramp_frac * b.total_steps))
        return self.s0 + (self.s - self.s0) * t


class _CurveRule(Rule):
    """Shared machinery of the two cost-curve rules: K partial-OT solves per batch."""

    def __init__(self, grid, eps=0.0):
        super().__init__(eps)
        self.grid = np.asarray(grid, dtype=np.float64)

    def _curve(self, C):
        plans = [solvers.plan_pot(C, s, self.eps) for s in self.grid]
        costs = np.array([float((P * C).sum()) for P in plans])
        return plans, costs

    def _finish(self, P, s, b):
        if self.warm(b) < 1.0:
            s = self._warmed(s, b)
            return solvers.plan_pot(b.C, s, self.eps), {"s_req": s}
        return P, {"s_req": s}


class ElbowRule(_CurveRule):
    """Draft Alg. 1.  On a convex increasing curve the Kneedle knee is argmax(x_n - y_n), the
    point where the marginal cost crosses the chord slope.  mode='chord' finds the same point
    with two partial solves and one price solve instead of K solves."""

    name = "elbow"

    def __init__(self, grid, eps=0.0, mode="kneedle"):
        super().__init__(grid, eps)
        self.mode = mode

    def plan(self, b):
        g = self.grid
        if self.mode == "chord":
            c_lo = float((solvers.plan_pot(b.C, g[0], self.eps) * b.C).sum())
            c_hi = float((solvers.plan_pot(b.C, g[-1], self.eps) * b.C).sum())
            lam = (c_hi - c_lo) / (g[-1] - g[0])
            P = solvers.plan_price(b.C, lam, self.eps)
            if self.warm(b) < 1.0:
                s = self._warmed(float(P.sum()), b)
                return solvers.plan_pot(b.C, s, self.eps), {"s_req": s, "lam": lam}
            return P, {"s_req": float(P.sum()), "lam": lam}
        plans, c = self._curve(b.C)
        if c[-1] - c[0] <= 0:
            i = len(g) - 1
        else:
            xn = (g - g[0]) / (g[-1] - g[0])
            yn = (c - c[0]) / (c[-1] - c[0])
            i = int(np.argmax(xn - yn))
        return self._finish(plans[i], float(g[i]), b)


class CurvatureRule(_CurveRule):
    """Draft Alg. 2: i* = argmax of the second difference over interior grid points, and the
    rule returns s_{i*-1}, as written in the draft."""

    name = "curv"

    def plan(self, b):
        g = self.grid
        plans, c = self._curve(b.C)
        h = g[1] - g[0]
        c2 = (c[2:] - 2.0 * c[1:-1] + c[:-2]) / h**2
        i_star = int(np.argmax(c2)) + 1
        return self._finish(plans[i_star - 1], float(g[i_star - 1]), b)


class ClusterRule(Rule):
    """Draft Alg. 3: K-means on the union of both batches' features; s = sum_c min / m."""

    name = "cluster"
    needs = ("fs", "ft")

    def __init__(self, K, eps=0.0, seed=0):
        super().__init__(eps)
        self.K, self.seed = K, seed

    def mass(self, b):
        from sklearn.cluster import KMeans

        Z = np.vstack([b.fs, b.ft]).astype(np.float64)
        lab = KMeans(n_clusters=self.K, n_init=4, random_state=self.seed + b.step).fit_predict(Z)
        n = len(b.fs)
        nx = np.bincount(lab[:n], minlength=self.K)
        ny = np.bincount(lab[n:], minlength=self.K)
        return np.minimum(nx, ny).sum() / max(n, len(b.ft))


class OracleRule(Rule):
    """Per-batch class overlap from the TRUE target labels.  Diagnostic upper bound for
    per-batch adaptivity; never a method."""

    name = "oracle"
    needs = ("ys", "yt")

    def __init__(self, n_classes, eps=0.0, s_min=0.0):
        super().__init__(eps)
        self.n_classes, self.s_min = n_classes, s_min

    def mass(self, b):
        return max(self.s_min, batch_overlap(b.ys, b.yt, self.n_classes))


class PseudoRule(Rule):
    """Per-batch class overlap with predicted target labels (hard argmax or soft counts).
    During the first `warmup` steps it blends in s_prior, the expected overlap under a uniform
    class prior, because an untrained head overstates the overlap."""

    name = "pseudo"
    needs = ("ys", "pt")

    def __init__(self, n_classes, eps=0.0, counts="hard", warmup=0, s_min=0.0, s_prior=None):
        super().__init__(eps)
        self.n_classes, self.counts, self.warmup = n_classes, counts, warmup
        self.s_min, self.s_prior = s_min, s_prior

    def mass(self, b):
        if self.counts == "hard":
            s_hat = batch_overlap(b.ys, b.pt.argmax(1), self.n_classes)
        else:
            s_hat = soft_overlap(b.ys, b.pt, self.n_classes)
        if self.warmup > 0 and b.step < self.warmup:
            if self.s_prior is None:
                self.s_prior = expected_overlap(len(b.ys), np.ones(self.n_classes))
            w = b.step / self.warmup
            s_hat = (1.0 - w) * self.s_prior + w * s_hat
        return max(self.s_min, s_hat)


class ExpectedRule(Rule):
    """Tuning-free level: s = E[class overlap] for this batch size.
    prior='uniform': balanced target classes; 'ema': running mean of the predicted target
    class probabilities (estimates label shift / partial targets); 'given': a fixed vector
    (e.g. the true target prior, diagnostic only)."""

    name = "expected"
    needs = ("ys",)

    def __init__(self, n_classes, eps=0.0, prior="uniform", prior_vec=None, ema=0.99, s_min=0.0):
        super().__init__(eps)
        self.n_classes, self.prior, self.ema, self.s_min = n_classes, prior, ema, s_min
        self.p = None if prior_vec is None else np.asarray(prior_vec, dtype=np.float64)
        self._cache = {}
        if prior == "ema":
            self.needs = ("ys", "pt")

    def mass(self, b):
        m = len(b.ys)
        if self.prior == "ema":
            p_batch = b.pt[:, : self.n_classes].mean(0)
            self.p = p_batch if self.p is None else self.ema * self.p + (1 - self.ema) * p_batch
            return max(self.s_min, expected_overlap(m, self.p, self.n_classes))
        if m not in self._cache:
            p = np.ones(self.n_classes) if self.prior == "uniform" else self.p
            self._cache[m] = expected_overlap(m, p, self.n_classes)
        return max(self.s_min, self._cache[m])


class PriceRule(Rule):
    """Fix a price lam per unit of moved mass instead of the mass itself.
    lam = kappa * scale, scale = EMA of the batch's median cost, so kappa is scale-free.
    With target_mass set, log(lam / scale) is adapted by dual ascent so that the running
    average mass approaches target_mass while each batch still gets its own amount."""

    name = "price"

    def __init__(self, kappa, eps=0.0, scale_ema=0.9, target_mass=None, dual_lr=0.05):
        super().__init__(eps)
        self.kappa, self.scale_ema = kappa, scale_ema
        self.target_mass, self.dual_lr = target_mass, dual_lr
        self.scale = None
        self.log_ratio = float(np.log(kappa))

    def plan(self, b):
        med = max(float(np.median(b.C)), 1e-12)
        self.scale = med if self.scale is None else self.scale_ema * self.scale + (1 - self.scale_ema) * med
        w = self.warm(b)
        if self.target_mass is None:
            lam = float(np.exp(self.log_ratio)) * self.scale * w
            return solvers.plan_price(b.C, lam, self.eps), {"lam": lam}
        lam = float(np.exp(self.log_ratio)) * self.scale
        P = solvers.plan_price(b.C, lam, self.eps)
        target = self.target_mass if w >= 1.0 else max(WARM_S0, self.target_mass * w)
        self.log_ratio += self.dual_lr * (target - float(P.sum()))
        return P, {"lam": lam, "s_req": target}


class AOTRule(Rule):
    """Adaptive OT (Yang, Tan & Wen, arXiv 2503.05087), exact form: balanced OT on
    min(C_signed, 0) restricted to C_signed < 0.  The trainer builds
    C_signed = alpha * ||g(x) - g(z)||^2 - beta * p(x)^T q(z) and trains on <P, C_signed>."""

    name = "aot"
    needs = ("C_signed",)
    loss_on_signed = True

    def plan(self, b):
        return solvers.plan_aot(b.C_signed, self.eps), {}


class PPOTRule(Rule):
    """Mass of PPOT (Yang, Gu & Sun, AAAI 2023): EMA of the fraction of target samples whose
    top class probability is at least tau1.  Designed for universal DA."""

    name = "ppot"
    needs = ("pt",)

    def __init__(self, eps=0.0, tau1=0.9, rate=0.001, s_min=0.0):
        super().__init__(eps)
        self.tau1, self.rate, self.s_min = tau1, rate, s_min
        self.s = None

    def mass(self, b):
        frac = float((b.pt.max(1) >= self.tau1).mean())
        self.s = frac if self.s is None else (1 - self.rate) * self.s + self.rate * frac
        return max(self.s_min, self.s)


RULES = ["ot", "uot", "fixed", "ramp", "elbow", "curv", "cluster", "oracle", "pseudo",
         "expected", "price", "aot", "ppot"]
LABEL_FREE = ["ot", "uot", "fixed", "ramp", "elbow", "curv", "cluster", "price"]


def add_rule_args(parser, default_eps=0.0):
    g = parser.add_argument_group("transport rule")
    g.add_argument("--rule", type=str, default="fixed", choices=RULES)
    g.add_argument("--eps", type=float, default=default_eps, help="entropic reg. (0 = exact LP)")
    g.add_argument("--mass", type=float, default=0.75, help="s for fixed; final s for ramp")
    g.add_argument("--tau", type=float, default=1.0, help="uot marginal penalty")
    g.add_argument("--ramp_s0", type=float, default=0.01)
    g.add_argument("--ramp_frac", type=float, default=0.5)
    g.add_argument("--grid_lo", type=float, default=0.05, help="elbow/curv candidate masses")
    g.add_argument("--grid_hi", type=float, default=1.0)
    g.add_argument("--grid_n", type=int, default=20)
    g.add_argument("--elbow_mode", type=str, default="kneedle", choices=["kneedle", "chord"])
    g.add_argument("--K", type=int, default=0, help="cluster: number of clusters (0 = #classes)")
    g.add_argument("--s_min", type=float, default=0.0, help="floor for oracle/pseudo/expected/ppot")
    g.add_argument("--warm_frac", type=float, default=0.0,
                   help="mass warm-up: s x min(1, t/(warm_frac T)), floor 0.01 (0 = off; m-POT PDA ramp = 0.5)")
    g.add_argument("--pseudo_counts", type=str, default="hard", choices=["hard", "soft"])
    g.add_argument("--warmup", type=int, default=0, help="pseudo: steps blending in the prior")
    g.add_argument("--prior", type=str, default="uniform", choices=["uniform", "ema", "true"])
    g.add_argument("--prior_ema", type=float, default=0.99)
    g.add_argument("--kappa", type=float, default=0.5, help="price: lam = kappa * EMA median cost")
    g.add_argument("--target_mass", type=float, default=-1.0, help="price: dual ascent (<0 = off)")
    g.add_argument("--dual_lr", type=float, default=0.05)
    g.add_argument("--aot_alpha", type=float, default=0.01)
    g.add_argument("--aot_beta", type=float, default=5.0)
    g.add_argument("--ppot_tau", type=float, default=0.9)
    g.add_argument("--ppot_rate", type=float, default=0.001)
    return g


def build_rule(args, n_classes=None, prior_vec=None, seed=0):
    rule = _build_rule(args, n_classes, prior_vec, seed)
    if getattr(args, "warm_frac", 0.0) > 0:
        if args.rule in ("ot", "uot", "ramp", "aot"):
            raise ValueError(f"--warm_frac does not apply to rule {args.rule}")
        rule.warm_frac = args.warm_frac
    return rule


def _build_rule(args, n_classes=None, prior_vec=None, seed=0):
    r, eps = args.rule, args.eps
    if r == "ot":
        return OTRule(eps)
    if r == "uot":
        return UOTRule(eps, args.tau)
    if r == "fixed":
        return FixedRule(args.mass, eps)
    if r == "ramp":
        return RampRule(args.mass, eps, args.ramp_s0, args.ramp_frac)
    if r in ("elbow", "curv"):
        grid = np.linspace(args.grid_lo, args.grid_hi, args.grid_n)
        return ElbowRule(grid, eps, args.elbow_mode) if r == "elbow" else CurvatureRule(grid, eps)
    if r == "cluster":
        return ClusterRule(args.K or n_classes, eps, seed)
    if r == "oracle":
        return OracleRule(n_classes, eps, args.s_min)
    if r == "pseudo":
        return PseudoRule(n_classes, eps, args.pseudo_counts, args.warmup, args.s_min)
    if r == "expected":
        prior = "given" if args.prior == "true" else args.prior
        if prior == "given" and prior_vec is None:
            raise ValueError("--prior true needs the target class prior")
        return ExpectedRule(n_classes, eps, prior, prior_vec, args.prior_ema, args.s_min)
    if r == "price":
        tm = args.target_mass if args.target_mass >= 0 else None
        return PriceRule(args.kappa, eps, target_mass=tm, dual_lr=args.dual_lr)
    if r == "aot":
        return AOTRule(eps)
    if r == "ppot":
        return PPOTRule(eps, args.ppot_tau, args.ppot_rate, args.s_min)
    raise ValueError(r)


def rule_name_fields(args):
    """Run-name fields that identify the rule configuration (and nothing irrelevant)."""
    r = args.rule
    f = [("rule", r)]
    if r == "uot":
        f += [("tau", args.tau)]
    elif r == "fixed":
        f += [("s", args.mass)]
    elif r == "ramp":
        f += [("s", args.mass), ("s0", args.ramp_s0), ("rf", args.ramp_frac)]
    elif r in ("elbow", "curv"):
        f += [("g", f"{args.grid_lo:g}-{args.grid_hi:g}x{args.grid_n}")]
        if r == "elbow":
            f += [("mode", args.elbow_mode)]
    elif r == "cluster":
        f += [("K", args.K)]
    elif r == "pseudo":
        f += [("cnt", args.pseudo_counts), ("wu", args.warmup)]
    elif r == "expected":
        f += [("prior", args.prior)]
    elif r == "price":
        f += [("kap", args.kappa)]
        if args.target_mass >= 0:
            f += [("tm", args.target_mass), ("dlr", args.dual_lr)]
    elif r == "aot":
        f += [("a", args.aot_alpha), ("b", args.aot_beta)]
    elif r == "ppot":
        f += [("pt", args.ppot_tau), ("pr", args.ppot_rate)]
    if r in ("oracle", "pseudo", "expected", "ppot") and args.s_min > 0:
        f += [("smin", args.s_min)]
    if getattr(args, "warm_frac", 0.0) > 0:
        f += [("wf", args.warm_frac)]
    f += [("eps", args.eps)]
    return f
