"""Mini-batch OT color transfer (m-POT App. B.3 / C.7) with a pluggable (label-free) mass rule.

Both images are compressed to a K-means palette; for T steps, k x k pairs of m random palette
colors are matched and the barycentric images are accumulated.  With partial plans the map is
the PARTIAL barycentric projection  T(x_i) = sum_j P_ij y_j / sum_j P_ij,  and a color that
is never transported keeps its own value.  (The reference script instead accumulated
m * P y and rescaled the whole image by its maximum, which darkens images under partial OT.)

Reported: W2 between output and target pixel colors (lower = closer to the target palette),
SSIM between output and source luminance (higher = structure kept), and the fraction of
pixels whose color was transported.

    python src/ColorTransfer/transfer.py --source starry_night.jpg --target ste_victoire.jpg --rule fixed --mass 0.9
"""
import os
import random
import sys
import time

import numpy as np
import ot
from skimage import color as skcolor
from skimage import img_as_ubyte, io
from skimage.metrics import structural_similarity
from sklearn.cluster import MiniBatchKMeans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import apps  # noqa: E402
from apot.rules import LABEL_FREE, Batch, build_rule  # noqa: E402
from apot.runlog import MassLog, open_run  # noqa: E402


def palette(name, n_colors, seed=0):
    """K-means palette of an image, cached under results/ColorTransfer/palettes."""
    cache = os.path.join(apps.RESULTS, "ColorTransfer", "palettes", f"{os.path.splitext(name)[0]}_{n_colors}.npz")
    img = img_as_ubyte(io.imread(os.path.join(HERE, "images", name)))[..., :3]
    if os.path.exists(cache):
        f = np.load(cache)
        return img, f["values"], f["labels"]
    km = MiniBatchKMeans(n_clusters=n_colors, n_init=4, batch_size=4096, random_state=seed).fit(img.reshape(-1, 3))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, values=km.cluster_centers_, labels=km.labels_)
    return img, km.cluster_centers_, km.labels_


def w2_colors(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = a.reshape(-1, 3)[rng.choice(a.size // 3, n, replace=False)].astype(np.float64) / 255
    b = b.reshape(-1, 3)[rng.choice(b.size // 3, n, replace=False)].astype(np.float64) / 255
    return float(ot.emd2(np.ones(n) / n, np.ones(n) / n, ot.dist(a, b), numItermax=1_000_000))


def main():
    args = apps.parse("color")
    if args.rule not in LABEL_FREE:
        raise SystemExit(f"rule {args.rule} needs labels; color transfer supports {LABEL_FREE}")
    run = open_run(args.out_root, apps.fields("color", args), vars(args), args.overwrite, args.print_name)
    if run is None:
        return
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    src_img, src_vals, src_labels = palette(args.source, args.n_colors)
    tgt_img, tgt_vals, _ = palette(args.target, args.n_colors)
    rule = build_rule(args, seed=args.seed)
    masslog = MassLog(run, every=max(1, args.T // 200))
    n, k, m = args.n_colors, args.k, args.m
    num, den = np.zeros_like(src_vals), np.zeros(n)
    for t in range(args.T):
        I = rng.choice(n, k * m, replace=False).reshape(k, m)
        J = rng.choice(n, k * m, replace=False).reshape(k, m)
        for i in range(k):
            for j in range(k):
                C = ot.dist(src_vals[I[i]], tgt_vals[J[j]])
                b = Batch(C=C, fs=src_vals[I[i]], ft=tgt_vals[J[j]], step=t, total_steps=args.T)
                t0 = time.perf_counter()
                P, info = rule.plan(b)
                masslog.add(t, P, info, C, 1000 * (time.perf_counter() - t0), k=i * k + j)
                num[I[i]] += P @ tgt_vals[J[j]]
                den[I[i]] += P.sum(1)
    moved = den > 1e-12
    mapped = src_vals.copy()
    mapped[moved] = num[moved] / den[moved, None]
    out = np.clip(mapped[src_labels].reshape(src_img.shape), 0, 255).astype(np.uint8)
    io.imsave(os.path.join(run.dir, "transfer.png"), np.concatenate([src_img, out, _fit(tgt_img, src_img)], 1),
              check_contrast=False)
    ssim = structural_similarity(skcolor.rgb2gray(src_img), skcolor.rgb2gray(out), data_range=1.0)
    run.finish({"w2_to_target": w2_colors(out, tgt_img), "w2_source_to_target": w2_colors(src_img, tgt_img),
                "ssim_luminance": float(ssim), "pixels_moved": float(moved[src_labels].mean()),
                "palette_moved": float(moved.mean()), **masslog.summary()})


def _fit(img, like):
    """Resize the target to the source height for the side-by-side figure."""
    from skimage.transform import resize
    return img_as_ubyte(resize(img, like.shape[:2]))


if __name__ == "__main__":
    main()
