"""Transport-plan solvers for one pair of mini-batches (NumPy / POT, CPU).

Every solver takes the cost matrix C (n_s x n_t) between a source and a target mini-batch,
puts uniform weights on both sides, and returns a plan of the same shape.  The amount of
mass a plan moves is ``P.sum()``: 1 for balanced OT, s for partial OT, data-dependent for
the price rule and for UOT.
"""
import numpy as np
import ot

NUM_ITER_MAX = 1_000_000  # exact LP; the POT default (1e5) can stop early at m = 500
FALLBACKS = {"pot": 0, "ot": 0}  # entropic solves that returned NaN and were redone exactly


def _prep(C):
    C = np.ascontiguousarray(C, dtype=np.float64)
    a = np.full(C.shape[0], 1.0 / C.shape[0])
    b = np.full(C.shape[1], 1.0 / C.shape[1])
    return C, a, b


def plan_ot(C, eps=0.0):
    """Balanced OT (m-OT). eps = 0 solves the exact LP, eps > 0 runs Sinkhorn."""
    C, a, b = _prep(C)
    if eps == 0:
        return ot.emd(a, b, C, numItermax=NUM_ITER_MAX)
    P = ot.sinkhorn(a, b, C, reg=eps)
    if not np.all(np.isfinite(P)):
        FALLBACKS["ot"] += 1
        return ot.emd(a, b, C, numItermax=NUM_ITER_MAX)
    return P


def plan_uot(C, eps, tau):
    """KL-relaxed unbalanced OT (m-UOT / JUMBOT), entropic, marginal penalty tau."""
    C, a, b = _prep(C)
    return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C, eps, tau)


def plan_pot(C, s, eps=0.0):
    """Partial OT moving total mass s in (0, 1] (m-POT)."""
    C, a, b = _prep(C)
    if s <= 0:
        return np.zeros_like(C)
    if s >= 1.0 - 1e-9:
        return plan_ot(C, eps)
    if eps == 0:
        return ot.partial.partial_wasserstein(a, b, C, m=s, numItermax=NUM_ITER_MAX)
    P = ot.partial.entropic_partial_wasserstein(a, b, C, m=s, reg=eps)
    if not np.all(np.isfinite(P)):  # underflow at small mass / small eps: use the exact LP
        FALLBACKS["pot"] += 1
        return ot.partial.partial_wasserstein(a, b, C, m=s, numItermax=NUM_ITER_MAX)
    return P


def plan_price(C, lam, eps=0.0):
    """Price rule: balanced OT on the truncated cost min(C, lam), keeping only entries with
    C < lam.  This is TV-relaxed UOT with tau = lam / 2 (equivalently, outlier-robust OT with
    a truncated cost), and it equals POT at the mass it ends up moving:
        OT(min(C, lam)) = POT_{s*}(C) + lam * (1 - s*),   s* = P.sum().
    """
    C, _, _ = _prep(C)
    P = plan_ot(np.minimum(C, lam), eps)
    return P * (C < lam)


def plan_aot(C_signed, eps=0.0):
    """Exact adaptive OT of Yang, Tan & Wen (arXiv 2503.05087): min <C, P> over P >= 0 with
    P 1 <= a, P^T 1 <= b and a mixed-sign cost.  It reduces to balanced OT on min(C, 0)
    restricted to C < 0, i.e. the price rule with lam = 0.  eps > 0 runs Sinkhorn on the
    truncated cost, an approximation of the authors' entropic solver."""
    return plan_price(C_signed, 0.0, eps)


def pot_cost_curve(C, grid, eps=0.0):
    """c(s) = <C, P_s> on a grid of masses (the curve the elbow / curvature rules read)."""
    return np.array([float((plan_pot(C, s, eps) * C).sum()) for s in grid])
