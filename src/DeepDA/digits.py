"""Digits domain adaptation (SVHN->MNIST, USPS->MNIST, MNIST->USPS) with a pluggable mass rule.

Protocol of m-POT (Nguyen et al., ICML 2022, App. D.1): DeepJDOT cost with eta1 = eta2 = 0.1,
Adam lr 4e-4, class-stratified source batches of m = 500, 10 epochs of source-only training
before adaptation.  Target labels are read only by the oracle rule and by the diagnostics.

    python src/DeepDA/digits.py --source svhn --target mnist --rule expected --seed 1
"""
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import apps  # noqa: E402
from apot.overlap import expected_overlap  # noqa: E402
from apot.rules import build_rule  # noqa: E402
from apot.runlog import MassLog, open_run  # noqa: E402
from common import (BalancedBatchSampler, Classifier1, SVHNGenerator, USPSGenerator,  # noqa: E402
                    infinite, jdot_cost, transport_step, weights_init_digits)

def get_datasets(name, size32, data_dir):
    if size32:
        to_rgb = transforms.Lambda(lambda x: x.convert("RGB"))
        t = [transforms.Resize(32)] + ([to_rgb] if name != "svhn" else []) + \
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    else:
        t = [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    t = transforms.Compose(t)
    if name == "mnist":
        tr, te = datasets.MNIST(data_dir, train=True, transform=t), datasets.MNIST(data_dir, train=False, transform=t)
        return tr, te, tr.targets.numpy()
    if name == "usps":
        tr, te = datasets.USPS(data_dir, train=True, transform=t), datasets.USPS(data_dir, train=False, transform=t)
        return tr, te, np.asarray(tr.targets)
    if name == "svhn":
        tr = datasets.SVHN(data_dir, split="train", transform=t)
        te = datasets.SVHN(data_dir, split="test", transform=t)
        return tr, te, np.asarray(tr.labels)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model_g, model_f, loader):
    model_g.eval()
    model_f.eval()
    correct = total = 0
    for x, y in loader:
        pred = model_f(model_g(x.cuda())).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    model_g.train()
    model_f.train()
    return correct / total


def main():
    args = apps.parse("digits")
    n_classes = apps.N_CLASSES["digits"]
    fields = apps.fields("digits", args)
    run = open_run(args.out_root, fields, vars(args), args.overwrite, args.print_name)
    if run is None:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    size32 = "svhn" in (args.source, args.target)
    src_tr, _, src_labels = get_datasets(args.source, size32, args.data_dir)
    tgt_tr, tgt_te, tgt_labels = get_datasets(args.target, size32, args.data_dir)
    bs = args.k * args.m
    src_loader = DataLoader(src_tr, batch_sampler=BalancedBatchSampler(src_labels, bs), num_workers=args.num_workers)
    tgt_loader = DataLoader(tgt_tr, batch_size=bs, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(tgt_te, batch_size=1000, shuffle=False, num_workers=args.num_workers)

    model_g = (SVHNGenerator() if size32 else USPSGenerator()).cuda().apply(weights_init_digits)
    model_f = Classifier1(nclass=n_classes).cuda().apply(weights_init_digits)
    ce = nn.CrossEntropyLoss()

    prior = np.bincount(tgt_labels, minlength=n_classes) / len(tgt_labels)
    rule = build_rule(args, n_classes=n_classes, prior_vec=prior, seed=args.seed)
    run.log(f"rule={rule.name}  E[ov](uniform)={expected_overlap(args.m, np.ones(n_classes)):.4f}  "
            f"E[ov](true prior)={expected_overlap(args.m, prior):.4f}")

    # source-only warm start (as in the reference implementation)
    opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr)
    opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr)
    for _ in range(args.src_epochs):
        for xs, ys in src_loader:
            loss = ce(model_f(model_g(xs.cuda())), ys.cuda())
            opt_g.zero_grad()
            opt_f.zero_grad()
            loss.backward()
            opt_g.step()
            opt_f.step()
    acc0 = evaluate(model_g, model_f, test_loader)
    run.log(f"source-only target acc {acc0:.4f}")
    run.row("metrics.csv", {"epoch": -1, "step": 0, "acc": acc0})

    # adaptation (fresh optimisers, as in the reference)
    opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr)
    opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr)
    masslog = MassLog(run)
    tgt_iter = infinite(tgt_loader)
    total = args.n_epochs * len(src_loader)
    step, best, last = 0, acc0, acc0
    for epoch in range(args.n_epochs):
        for xs_all, ys_all in src_loader:
            xt_all, yt_all = next(tgt_iter)
            opt_g.zero_grad()
            opt_f.zero_grad()
            for i, idx in enumerate(np.split(np.arange(bs), args.k)):
                xs, ys = xs_all[idx].cuda(), ys_all[idx].cuda()
                xt, yt = xt_all[idx].cuda(), yt_all[idx]
                g_s, g_t = model_g(xs), model_g(xt)
                logits_s, logits_t = model_f(g_s), model_f(g_t)
                M, M_embed, agree, pred_t = jdot_cost(g_s, g_t, ys, logits_t, n_classes, args.eta1, args.eta2)
                t_loss = transport_step(rule, M, ys, yt, pred_t, g_s, g_t, step, total, masslog, i,
                                        M_embed, agree, args.aot_alpha, args.aot_beta)
                ((ce(logits_s, ys) + t_loss) / args.k).backward()
            opt_g.step()
            opt_f.step()
            step += 1
        last = evaluate(model_g, model_f, test_loader)
        best = max(best, last)
        run.row("metrics.csv", {"epoch": epoch, "step": step, "acc": last})
        run.log(f"epoch {epoch}  acc {last:.4f}  best {best:.4f}  mean s {np.mean(masslog.s[-len(src_loader):]):.3f}")

    run.finish({"last_acc": last, "best_acc": best, "src_only_acc": acc0, **masslog.summary()})


if __name__ == "__main__":
    main()
