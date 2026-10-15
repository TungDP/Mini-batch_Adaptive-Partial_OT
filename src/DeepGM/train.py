"""Mini-batch OT generative model on CIFAR-10 / CelebA with a pluggable (label-free) mass rule.

Protocol of m-POT (App. B.2, D.3) and of the m-KOT runners: DCGAN-style generator and
discriminator; the cost is the squared distance between discriminator features; per step the
discriminator maximises and the generator minimises (1/k^2) sum_ij <P_ij, C_ij> over the k x k
pairs of data and noise mini-batches.  Adam lr 5e-4, betas (0.5, 0.999), latent 32, exact LP.
FID: 11 000 generated images against the test split (CIFAR-10 test set / CelebA test partition).

    python src/DeepGM/train.py --dataset cifar10 --rule fixed --mass 0.7
    python src/DeepGM/train.py --dataset cifar10 --rule price --kappa 0.5
"""
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import apps  # noqa: E402
from apot.rules import LABEL_FREE, Batch, build_rule  # noqa: E402
from apot.runlog import MassLog, open_run  # noqa: E402
from fid import FID, frechet_distance, to_uint8_01  # noqa: E402

SIZE = {"cifar10": 32, "celeba": 64}


class Discriminator(nn.Module):
    """32x32: 3 down-convs (64-128-256, tanh features);  64x64: 4 down-convs (... 512)."""

    def __init__(self, image_size, hidden=64):
        super().__init__()
        n_down = 3 if image_size == 32 else 4
        layers, c_in = [], 3
        for i in range(n_down):
            c_out = hidden * 2**i
            layers.append(nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False))
            if i > 0:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.Tanh() if i == n_down - 1 else nn.LeakyReLU(0.2, inplace=True))
            c_in = c_out
        self.main1 = nn.Sequential(*layers)
        self.main2 = nn.Sequential(nn.Conv2d(c_in, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, x):
        h = self.main1(x)
        return self.main2(h).view(x.shape[0], -1), h


class Generator(nn.Module):
    def __init__(self, image_size, latent, hidden=64):
        super().__init__()
        self.latent = latent
        n_up = 3 if image_size == 32 else 4
        c = hidden * 2**n_up // 2  # 256 for 32x32, 512 for 64x64
        layers = [nn.ConvTranspose2d(latent, c, 4, 1, 0, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)]
        for _ in range(n_up - 1):
            layers += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z.view(z.shape[0], self.latent, 1, 1))


class CelebAFolder(Dataset):
    """img_align_celeba with the official partition (0 train, 1 val, 2 test) from the CSV."""

    def __init__(self, root, partition, transform):
        rows = [l.strip().split(",") for l in open(os.path.join(root, "list_eval_partition.csv"))][1:]
        self.files = [os.path.join(root, "img_align_celeba", f) for f, p in rows if int(p) == partition]
        self.transform = transform

    def __getitem__(self, i):
        with Image.open(self.files[i]) as img:
            return self.transform(img.convert("RGB")), 0

    def __len__(self):
        return len(self.files)


def dataset(name, data_dir, train, normalize=True):
    t = [transforms.Resize(SIZE[name]), transforms.CenterCrop(SIZE[name]), transforms.ToTensor()]
    if normalize:
        t.append(transforms.Normalize((0.5,) * 3, (0.5,) * 3))
    t = transforms.Compose(t)
    if name == "cifar10":
        return datasets.CIFAR10(os.path.join(data_dir, "cifar10"), train=train, download=False, transform=t)
    return CelebAFolder(os.path.join(data_dir, "celeba"), 0 if train else 2, t)


def reference_stats(name, data_dir, fid, num_workers):
    """Inception statistics of the test split, cached next to the results."""
    path = os.path.join(apps.RESULTS, "DeepGM", f"fid_stats_{name}_test.npz")
    if os.path.exists(path):
        f = np.load(path)
        return f["mu"], f["sigma"]
    ds = dataset(name, data_dir, train=False, normalize=False)
    imgs = torch.cat([x for x, _ in DataLoader(ds, batch_size=500, num_workers=num_workers)])
    mu, sigma = fid.stats(fid.activations(imgs))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mu=mu, sigma=sigma)
    return mu, sigma


def main():
    args = apps.parse("dgm")
    if args.rule not in LABEL_FREE:
        raise SystemExit(f"rule {args.rule} needs labels; generative modelling supports {LABEL_FREE}")
    run = open_run(args.out_root, apps.fields("dgm", args), vars(args), args.overwrite, args.print_name)
    if run is None:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = "cuda"
    size, k, m = SIZE[args.dataset], args.k, args.m
    loader = DataLoader(dataset(args.dataset, args.data_dir, train=True), batch_size=k * m, shuffle=True,
                        drop_last=True, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    G = Generator(size, args.latent).to(dev)
    D = Discriminator(size).to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    rule = build_rule(args, seed=args.seed)
    fid = FID(dev)
    mu_ref, sigma_ref = reference_stats(args.dataset, args.data_dir, fid, args.num_workers)
    fixed_noise = torch.randn(64, args.latent, device=dev)
    masslog = MassLog(run, every=10)
    epochs = args.epochs * k  # as in the reference implementation
    total = epochs * len(loader)
    run.log(f"{args.dataset}: {len(loader.dataset)} images, {len(loader)} steps/epoch, {epochs} epochs, rule={rule.name}")

    def pair_loss(x_mb, z_mb, step, log_k=None):
        _, f_real = D(x_mb)
        _, f_fake = D(G(z_mb))
        f_real, f_fake = f_real.flatten(1), f_fake.flatten(1)
        C = torch.cdist(f_real, f_fake) ** 2
        b = Batch(C=C.detach().double().cpu().numpy(), fs=f_real.detach().cpu().numpy() if "fs" in rule.needs else None,
                  ft=f_fake.detach().cpu().numpy() if "ft" in rule.needs else None, step=step, total_steps=total)
        t0 = time.perf_counter()
        P, info = rule.plan(b)
        if log_k is not None:
            masslog.add(step, P, info, b.C, 1000 * (time.perf_counter() - t0), k=log_k)
        return torch.sum(torch.from_numpy(P).float().to(dev) * C)

    @torch.no_grad()
    def evaluate():
        G.eval()
        imgs = torch.cat([to_uint8_01(G(torch.randn(500, args.latent, device=dev))).cpu()
                          for _ in range((args.fid_n + 499) // 500)])[: args.fid_n]
        mu, sigma = fid.stats(fid.activations(imgs))
        save_image(G(fixed_noise), os.path.join(run.dir, "samples.png"), normalize=True, nrow=8)
        G.train()
        return frechet_distance(mu, sigma, mu_ref, sigma_ref)

    step, best, best_epoch, last = 0, float("inf"), -1, float("nan")
    for epoch in range(epochs):
        t_ep = time.time()
        for x, _ in loader:
            x = x.to(dev, non_blocking=True)
            z = torch.randn(k * m, args.latent, device=dev)
            xs, zs = x.split(m), z.split(m)
            opt_d.zero_grad()  # discriminator: maximise the transport cost
            for i in range(k):
                for j in range(k):
                    (-pair_loss(xs[i], zs[j], step) / k**2).backward()
            opt_d.step()
            opt_g.zero_grad()  # generator: minimise it
            for i in range(k):
                for j in range(k):
                    (pair_loss(xs[i], zs[j], step, log_k=i * k + j) / k**2).backward()
            opt_g.step()
            step += 1
        if (epoch + 1) % args.fid_every == 0 or epoch + 1 == epochs:
            last = evaluate()
            if last < best:
                best, best_epoch = last, epoch + 1
                torch.save(G.state_dict(), os.path.join(run.dir, "G_best.pth"))
            run.row("metrics.csv", {"epoch": epoch + 1, "step": step, "fid": last,
                                    "mean_s": float(np.mean(masslog.s[-len(loader) * k * k:]))})
            run.log(f"epoch {epoch + 1}/{epochs}  FID {last:.2f}  best {best:.2f}  "
                    f"mean s {np.mean(masslog.s[-len(loader) * k * k:]):.3f}  {time.time() - t_ep:.0f} s/epoch")
    torch.save(G.state_dict(), os.path.join(run.dir, "G_final.pth"))
    run.finish({"last_fid": last, "best_fid": best, "best_epoch": best_epoch, **masslog.summary()})


if __name__ == "__main__":
    main()
