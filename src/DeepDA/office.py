"""Office-Home / VisDA-2017 domain adaptation and Office-Home partial DA (65 -> 25 classes)
with a pluggable mass rule.

Protocol of the m-POT reference code: ResNet-50 (ImageNet) + bottleneck + new head, SGD
(lr 1e-3, momentum 0.9, nesterov, wd 5e-4) with the inv schedule, class-stratified source
batches, DeepJDOT cost eta1 ||g_s - g_t||^2 + eta2 CE, loss = CE_src + eta3 <P, M>.  The
target images are both the unlabelled training set and the test set (transductive).
Partial DA keeps the target images of the 25 alphabetically-first class names (the BA3US /
m-POT protocol), whatever label numbering the list files use.

    python src/DeepDA/office.py --dset office-home --s Art --t Clipart --rule expected --seed 1
    python src/DeepDA/office.py --dset visda --m 72 --rule oracle --test_10crop 0
    python src/DeepDA/office.py --dset office-home --n_shared 25 --s Art --t Clipart \
        --bottleneck 256 --eta1 0.003 --eta2 0.75 --eta3 10 --iters 5000 --test_10crop 0
"""
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import apps  # noqa: E402
from apot.overlap import expected_overlap  # noqa: E402
from apot import solvers  # noqa: E402
from apot.rules import build_rule  # noqa: E402
from apot.runlog import MassLog, open_run  # noqa: E402
from common import (BalancedBatchSampler, ImageList, ResNetFc, image_test, image_test_10crop,  # noqa: E402
                    image_test_center, image_train, image_train_crop, infinite, inv_lr_scheduler, jdot_cost,
                    transport_step)

REPO = apps.REPO


def list_path(dset, domain):
    if dset == "office-home":
        return os.path.join(REPO, "data", "office-home", f"{domain}.txt")
    return os.path.join(REPO, "data", "visda-2017", f"{domain}_list.txt")


def shared_class_ids(lines, n_shared):
    """Label ids of the n_shared alphabetically-first class names (folder names)."""
    name_of = {}
    for line in lines:
        path, y = line.split()
        name_of.setdefault(int(y), path.split("/")[-2])
    names = sorted(name_of.values())[:n_shared]
    return sorted(y for y, n in name_of.items() if n in names), names


@torch.no_grad()
def evaluate(net, loaders, ten_crop):
    net.eval()
    probs, labels = [], None
    if ten_crop:
        for loader in loaders:
            out, ys = [], []
            for x, y in loader:
                out.append(torch.softmax(net(x.cuda())[1].float(), 1).cpu())
                ys.append(y)
            probs.append(torch.cat(out))
            labels = torch.cat(ys)
        prob = sum(probs) / len(probs)
    else:
        out, ys = [], []
        for x, y in loaders:
            out.append(torch.softmax(net(x.cuda())[1].float(), 1).cpu())
            ys.append(y)
        prob, labels = torch.cat(out), torch.cat(ys)
    net.train()
    pred = prob.argmax(1)
    acc = (pred == labels).float().mean().item()
    per_class = [((pred == labels) & (labels == c)).sum().item() / max(1, (labels == c).sum().item())
                 for c in labels.unique().tolist()]
    return acc, float(np.mean(per_class))


def main():
    args = apps.parse("office")
    src_lines = [l for l in open(list_path(args.dset, args.s)) if l.strip()]
    tgt_lines = [l for l in open(list_path(args.dset, args.t)) if l.strip()]
    n_classes = len({int(l.split()[1]) for l in src_lines})
    shared_names = None
    if args.n_shared:
        keep, shared_names = shared_class_ids(tgt_lines, args.n_shared)
        tgt_lines = [l for l in tgt_lines if int(l.split()[1]) in keep]

    fields = apps.fields("office", args)
    task = dict(fields)["task"]
    config = dict(vars(args), shared_classes=shared_names)
    run = open_run(args.out_root, fields, config, args.overwrite, args.print_name)
    if run is None:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    train_tf = image_train_crop() if args.aug == "crop" else image_train()
    src_ds = ImageList(src_lines, REPO, train_tf)
    tgt_ds = ImageList(tgt_lines, REPO, train_tf)
    src_labels = np.array([y for _, y in src_ds.imgs])
    tgt_labels = np.array([y for _, y in tgt_ds.imgs])
    bs = args.k * args.m
    src_loader = DataLoader(src_ds, batch_sampler=BalancedBatchSampler(src_labels, bs),
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    tgt_loader = DataLoader(tgt_ds, batch_size=bs, shuffle=True, drop_last=True,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    test_bs = 64
    if args.test_10crop:
        test_loaders = [DataLoader(ImageList(tgt_lines, REPO, t), batch_size=test_bs, shuffle=False,
                                   num_workers=args.num_workers) for t in image_test_10crop()]
    else:
        test_tf = image_test_center() if args.aug == "crop" else image_test()
        test_loaders = DataLoader(ImageList(tgt_lines, REPO, test_tf), batch_size=test_bs, shuffle=False,
                                  num_workers=args.num_workers)

    net = ResNetFc(n_classes, args.bottleneck, args.weights).cuda()
    optimizer = torch.optim.SGD(net.get_parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))
    ce = nn.CrossEntropyLoss()

    prior = np.bincount(tgt_labels, minlength=n_classes) / len(tgt_labels)
    rule = build_rule(args, n_classes=n_classes, prior_vec=prior, seed=args.seed)
    run.log(f"{task} src={len(src_ds)} tgt={len(tgt_ds)} classes={n_classes} rule={rule.name} "
            f"E[ov](uniform)={expected_overlap(args.m, np.ones(n_classes)):.4f} "
            f"E[ov](true prior)={expected_overlap(args.m, prior, n_classes):.4f}")

    masslog = MassLog(run)
    src_iter, tgt_iter = infinite(src_loader), infinite(tgt_loader)
    best, best_step, last, last_pc = 0.0, 0, 0.0, 0.0
    t_int = time.time()
    for step in tqdm(range(args.iters), mininterval=30, disable=not sys.stdout.isatty()):
        optimizer = inv_lr_scheduler(optimizer, step, gamma=0.001, power=0.75, lr=args.lr)
        optimizer.zero_grad()
        xs_all, ys_all = next(src_iter)
        xt_all, yt_all = next(tgt_iter)
        for i, idx in enumerate(np.split(np.arange(bs), args.k)):
            xs, ys = xs_all[idx].cuda(non_blocking=True), ys_all[idx].cuda(non_blocking=True)
            xt, yt = xt_all[idx].cuda(non_blocking=True), yt_all[idx]
            with torch.autocast("cuda", enabled=bool(args.amp)):
                g_s, logits_s = net(xs)
                g_t, logits_t = net(xt)
                cls_loss = ce(logits_s.float(), ys)
            M, M_embed, agree, pred_t = jdot_cost(g_s, g_t, ys, logits_t, n_classes, args.eta1, args.eta2)
            t_loss = transport_step(rule, M, ys, yt, pred_t, g_s, g_t, step, args.iters, masslog, i,
                                    M_embed, agree, args.aot_alpha, args.aot_beta)
            scaler.scale((cls_loss + args.eta3 * t_loss) / args.k).backward()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % args.test_interval == 0 or step + 1 == args.iters:
            sec_it = (time.time() - t_int) / args.test_interval
            t_eval = time.time()
            last, last_pc = evaluate(net, test_loaders, args.test_10crop)
            t_eval = time.time() - t_eval
            if last > best:
                best, best_step = last, step + 1
            recent = masslog.s[-args.test_interval * args.k:]
            run.row("metrics.csv", {"step": step + 1, "acc": last, "acc_per_class": last_pc,
                                    "mean_s": float(np.mean(recent))})
            run.log(f"step {step + 1}  acc {last:.4f}  per-class {last_pc:.4f}  best {best:.4f}  "
                    f"mean s {np.mean(recent):.3f}  {sec_it:.3f} s/it  eval {t_eval:.0f} s")
            t_int = time.time()

    run.finish({"last_acc": last, "last_acc_per_class": last_pc, "best_acc": best, "best_step": best_step,
                "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "entropic_fallbacks": dict(solvers.FALLBACKS), **masslog.summary()})


if __name__ == "__main__":
    main()
