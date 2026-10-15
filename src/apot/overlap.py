"""Class overlap between a source and a target mini-batch.

The class overlap of a batch pair is  ov = sum_c min(n_c^src, n_c^tgt) / m = 1 - TV of the two
batch label histograms: the largest mass partial OT can move without crossing a class.
Under the m-POT loaders (class-stratified source, uniformly sampled target) its expectation
has a closed form, used as a tuning-free mass.
"""
import numpy as np
from scipy.stats import binom


def batch_overlap(ys, yt, n_classes):
    """Overlap from label vectors (true labels -> oracle; predicted labels -> pseudo)."""
    ns = np.bincount(ys, minlength=n_classes)[:n_classes]
    nt = np.bincount(yt, minlength=n_classes)[:n_classes]
    return float(np.minimum(ns, nt).sum()) / max(len(ys), len(yt))


def soft_overlap(ys, pt, n_classes):
    """Overlap with soft target counts n_c = sum_j p(c | x_j)."""
    ns = np.bincount(ys, minlength=n_classes)[:n_classes]
    nt = pt[:, :n_classes].sum(0)
    return float(np.minimum(ns, nt).sum()) / max(len(ys), pt.shape[0])


def expected_overlap(m, prior_t, n_src_classes=None):
    """E[ov] for a class-stratified source batch of size m and a target batch of m i.i.d.
    draws from prior_t (multinomial; exact by linearity of expectation).

    The stratified sampler gives every class floor(m / C) samples and spreads the remainder r
    over r classes chosen uniformly, so a class holds floor(m / C) + 1 with probability r / C.
    E[min(n, N)] = sum_{j=1..n} P(N >= j) for N ~ Bin(m, p_c).
    """
    prior_t = np.asarray(prior_t, dtype=np.float64)
    C = n_src_classes or len(prior_t)
    p = np.zeros(C)
    p[: len(prior_t)] = prior_t[:C]
    p = p / p.sum()
    base, r = divmod(m, C)

    def s_min(n):  # sum_c E[min(n, N_c)]
        if n == 0:
            return np.zeros(C)
        j = np.arange(1, n + 1)
        return binom.sf(j[None, :] - 1, m, p[:, None]).sum(1)

    per_class = (1 - r / C) * s_min(base) + (r / C) * s_min(base + 1)
    return float(per_class.sum() / m)


def overlap_mc(labels_t, m, n_src_classes, n_batches=20000, seed=0):
    """Monte-Carlo overlap distribution under the exact loaders (target drawn without
    replacement from a label list). Returns the per-batch overlaps."""
    rng = np.random.default_rng(seed)
    labels_t = np.asarray(labels_t)
    C = n_src_classes
    base, r = divmod(m, C)
    out = np.empty(n_batches)
    for i in range(n_batches):
        ns = np.full(C, base)
        if r:
            ns[rng.choice(C, r, replace=False)] += 1
        nt = np.bincount(labels_t[rng.choice(len(labels_t), m, replace=False)], minlength=C)[:C]
        out[i] = np.minimum(ns, nt).sum() / m
    return out
