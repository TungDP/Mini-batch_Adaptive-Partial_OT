"""Mini-batch OT gradient flow between two 2-D densities (Feydy et al. 2019 shapes), as in
m-POT App. C.8, with a pluggable (label-free) mass rule.

Particles x_i of the source flow along -grad of the mini-batch loss (1/k^2) sum_ij <P_ij, C_ij>
by explicit Euler, x <- x - lr * n * grad.  Every rule uses the same (k, m), unlike the
reference script (m-OT at m = 16 exact vs m-POT at m = 4 entropic).  The exact W2 between the
particles and the target is logged along the flow; its area under the curve measures speed.

    python src/GradientFlow/flow.py --rule fixed --mass 0.8
"""
import os
import random
import sys
import time

import matplotlib
import numpy as np
import ot
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import apps  # noqa: E402
from apot.rules import LABEL_FREE, Batch, build_rule  # noqa: E402
from apot.runlog import MassLog, open_run  # noqa: E402


def draw_samples(fname, n, rng):
    """Sample n points from the grey-level density of an image (reference implementation)."""
    A = plt.imread(fname)
    A = A[..., :3].mean(-1) if A.ndim == 3 else A
    A = 1.0 - A[::-1, :]
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))
    grid = np.stack([xg.ravel(), yg.ravel()], 1)
    dens = A.ravel() / A.sum()
    dots = grid[rng.choice(len(grid), n, p=dens)]
    dots += (0.5 / A.shape[0]) * rng.standard_normal(dots.shape)
    return torch.from_numpy(dots).float()


def w2(x, y):
    M = ot.dist(x.detach().numpy().astype(np.float64), y.numpy().astype(np.float64))
    return float(ot.emd2(np.ones(len(x)) / len(x), np.ones(len(y)) / len(y), M, numItermax=1_000_000))


def main():
    args = apps.parse("flow")
    if args.rule not in LABEL_FREE:
        raise SystemExit(f"rule {args.rule} needs labels; gradient flow supports {LABEL_FREE}")
    run = open_run(args.out_root, apps.fields("flow", args), vars(args), args.overwrite, args.print_name)
    if run is None:
        return
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    X = draw_samples(os.path.join(HERE, "data", "density_a.png"), args.n, rng)
    Y = draw_samples(os.path.join(HERE, "data", "density_b.png"), args.n, rng)
    x = X.clone().requires_grad_(True)
    rule = build_rule(args, seed=args.seed)
    masslog = MassLog(run, every=args.eval_every)
    steps = int(args.T / args.lr) + 1
    shown = [int(t / args.lr) for t in np.linspace(0, args.T, 6)]
    colors = (np.cos(10 * X[:, 0].numpy()) * np.cos(10 * X[:, 1].numpy()))
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4))
    curve = []
    for t in range(steps):
        if t % args.eval_every == 0 or t == steps - 1:
            d = w2(x, Y)
            curve.append(d)
            run.row("metrics.csv", {"step": t, "w2": d, "mean_s": float(np.mean(masslog.s[-args.k**2 * args.eval_every:]))
                                    if masslog.s else ""})
        if t in shown:
            ax = axes[shown.index(t)]
            ax.scatter(Y[:, 0], Y[:, 1], s=4, color=(0.55, 0.55, 0.95))
            ax.scatter(x.detach()[:, 0], x.detach()[:, 1], s=4, c=colors, cmap="hsv")
            ax.set_title(f"step {t}  $W_2^2$={w2(x, Y) * 100:.3f}e-2", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
        loss = 0.0
        I = [rng.choice(args.n, args.m, replace=False) for _ in range(args.k)]
        J = [rng.choice(args.n, args.m, replace=False) for _ in range(args.k)]
        for i in range(args.k):
            for j in range(args.k):
                xi, yj = x[I[i]], Y[J[j]]
                C = torch.cdist(xi, yj) ** 2
                b = Batch(C=C.detach().double().numpy(), fs=xi.detach().numpy(), ft=yj.numpy(), step=t, total_steps=steps)
                t0 = time.perf_counter()
                P, info = rule.plan(b)
                masslog.add(t, P, info, b.C, 1000 * (time.perf_counter() - t0), k=i * args.k + j)
                loss = loss + torch.sum(torch.from_numpy(P).float() * C) / args.k**2
        (g,) = torch.autograd.grad(loss, [x])
        x.data -= args.lr * args.n * g
    axes[0].set_ylabel(f"{rule.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(run.dir, "flow.png"), dpi=130)
    np.save(os.path.join(run.dir, "particles_final.npy"), x.detach().numpy())
    run.finish({"final_w2": curve[-1], "auc_w2": float(np.mean(curve)), "w2_curve": curve[:: max(1, len(curve) // 20)],
                **masslog.summary()})


if __name__ == "__main__":
    main()
