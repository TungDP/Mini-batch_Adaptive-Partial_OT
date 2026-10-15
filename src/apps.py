"""Argument parsers and run names of every trainer, in one light module (no torch import).

Both the trainers and scripts/launch.py build run names through `run_dir(app, argv)`, so the
launcher never re-implements a name: it parses a job's argv with the trainer's own parser.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from apot.rules import add_rule_args, rule_name_fields  # noqa: E402
from apot.runlog import run_name  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(REPO, "results")
N_CLASSES = {"digits": 10, "office-home": 65, "visda": 12}


def _common(p, out_sub):
    p.add_argument("--out_root", default=os.path.join(RESULTS, *out_sub))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="", help="free label for protocol variants")
    p.add_argument("--overwrite", action="store_true", help="re-run even if done.json exists")
    p.add_argument("--print_name", action="store_true", help="print the run directory and exit")


# ------------------------------------------------------------------------------- digits
def digits_parser():
    p = argparse.ArgumentParser(description="digits DA")
    p.add_argument("--source", default="svhn", choices=["svhn", "usps", "mnist"])
    p.add_argument("--target", default="mnist", choices=["mnist", "usps"])
    p.add_argument("--data_dir", default=os.path.join(REPO, "data"))
    p.add_argument("--m", type=int, default=500)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--src_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--eta1", type=float, default=0.1)
    p.add_argument("--eta2", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    _common(p, ("DeepDA", "digits"))
    add_rule_args(p)
    return p


def digits_finalize(a):
    if a.rule == "cluster" and a.K == 0:
        a.K = N_CLASSES["digits"]
    return a


def digits_fields(a):
    return [("src", a.source), ("tgt", a.target), ("m", a.m), ("k", a.k)] + rule_name_fields(a) + \
        [("e1", a.eta1), ("e2", a.eta2), ("lr", a.lr), ("ep", a.n_epochs), ("se", a.src_epochs),
         ("seed", a.seed), ("tag", a.tag)]


# ------------------------------------------------------------------------------- office
def office_parser():
    p = argparse.ArgumentParser(description="Office-Home / VisDA DA and Office-Home partial DA")
    p.add_argument("--dset", default="office-home", choices=["office-home", "visda"])
    p.add_argument("--s", default=None, help="source domain (visda: train)")
    p.add_argument("--t", default=None, help="target domain (visda: validation)")
    p.add_argument("--n_shared", type=int, default=0, help="partial DA: keep this many target classes")
    p.add_argument("--m", type=int, default=65)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--test_interval", type=int, default=500)
    p.add_argument("--test_10crop", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta1", type=float, default=0.01)
    p.add_argument("--eta2", type=float, default=0.5)
    p.add_argument("--eta3", type=float, default=1.0, help="weight of the transport loss")
    p.add_argument("--bottleneck", type=int, default=512)
    p.add_argument("--aug", default="rrc", choices=["rrc", "crop"],
                   help="rrc: RandomResizedCrop (JUMBOT/m-POT DA code); crop: Resize+RandomCrop (m-POT/BA3US PDA code)")
    p.add_argument("--weights", default="IMAGENET1K_V1", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])
    p.add_argument("--amp", type=int, default=0, help="mixed-precision forward")
    p.add_argument("--num_workers", type=int, default=6)
    _common(p, ("DeepDA", "office"))
    add_rule_args(p)
    return p


def office_finalize(a):
    if a.dset == "visda":
        a.s, a.t = a.s or "train", a.t or "validation"
    a.s, a.t = a.s or "Art", a.t or "Clipart"
    if a.rule == "cluster" and a.K == 0:
        a.K = N_CLASSES[a.dset]
    return a


def office_fields(a):
    task = f"{a.s[0]}2{a.t[0]}" if a.dset == "office-home" else "S2R"
    return [("ds", a.dset), ("task", task), ("pda", a.n_shared or None), ("m", a.m), ("k", a.k)] + \
        rule_name_fields(a) + \
        [("e1", a.eta1), ("e2", a.eta2), ("e3", a.eta3), ("it", a.iters), ("bn", a.bottleneck),
         ("tc", a.test_10crop), ("aug", a.aug if a.aug != "rrc" else None),
         ("w", "v2" if a.weights == "IMAGENET1K_V2" else None),
         ("amp", 1 if a.amp else None), ("seed", a.seed), ("tag", a.tag)]


# ------------------------------------------------------------------------------- deep generative model
def dgm_parser():
    p = argparse.ArgumentParser(description="mini-batch OT generative model (CIFAR-10 / CelebA)")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "celeba"])
    p.add_argument("--data_dir", default=os.path.join(REPO, "data"))
    p.add_argument("--m", type=int, default=100)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100, help="multiplied by k, as in the reference")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--fid_every", type=int, default=10, help="epochs between FID evaluations")
    p.add_argument("--fid_n", type=int, default=11000, help="generated images per FID evaluation")
    p.add_argument("--num_workers", type=int, default=8)
    _common(p, ("DeepGM",))
    add_rule_args(p)
    p.set_defaults(rule="fixed", mass=0.7)
    return p


def dgm_finalize(a):
    if a.rule == "cluster" and a.K == 0:
        a.K = 10
    return a


def dgm_fields(a):
    return [("ds", a.dataset), ("m", a.m), ("k", a.k)] + rule_name_fields(a) + \
        [("lat", a.latent), ("lr", a.lr), ("ep", a.epochs), ("seed", a.seed), ("tag", a.tag)]


# ------------------------------------------------------------------------------- gradient flow
def flow_parser():
    p = argparse.ArgumentParser(description="mini-batch OT gradient flow (Feydy et al. shapes)")
    p.add_argument("--n", type=int, default=1000, help="particles per measure")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--k", type=int, default=4, help="k x k mini-batch pairs per step")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--T", type=float, default=5.0, help="flow time; steps = T / lr")
    p.add_argument("--eval_every", type=int, default=250)
    _common(p, ("GradientFlow",))
    add_rule_args(p)
    p.set_defaults(rule="fixed", mass=0.8)
    return p


def flow_finalize(a):
    if a.rule == "cluster" and a.K == 0:
        a.K = 8
    return a


def flow_fields(a):
    return [("n", a.n), ("m", a.m), ("k", a.k)] + rule_name_fields(a) + \
        [("lr", a.lr), ("T", a.T), ("seed", a.seed), ("tag", a.tag)]


# ------------------------------------------------------------------------------- color transfer
def color_parser():
    p = argparse.ArgumentParser(description="mini-batch OT color transfer")
    p.add_argument("--source", default="ocean_day.jpg", help="file in src/ColorTransfer/images")
    p.add_argument("--target", default="ocean_sunset.jpg")
    p.add_argument("--n_colors", type=int, default=3000, help="K-means palette size per image")
    p.add_argument("--m", type=int, default=100)
    p.add_argument("--k", type=int, default=1, help="k x k mini-batch pairs per step")
    p.add_argument("--T", type=int, default=10000, help="steps")
    _common(p, ("ColorTransfer",))
    add_rule_args(p)
    p.set_defaults(rule="fixed", mass=0.9)
    return p


def color_finalize(a):
    if a.rule == "cluster" and a.K == 0:
        a.K = 8
    return a


def color_fields(a):
    pair = f"{os.path.splitext(a.source)[0]}-{os.path.splitext(a.target)[0]}"
    return [("pair", pair), ("nc", a.n_colors), ("m", a.m), ("k", a.k)] + rule_name_fields(a) + \
        [("T", a.T), ("seed", a.seed), ("tag", a.tag)]


APPS = {
    "digits": (digits_parser, digits_finalize, digits_fields, "src/DeepDA/digits.py"),
    "office": (office_parser, office_finalize, office_fields, "src/DeepDA/office.py"),
    "dgm": (dgm_parser, dgm_finalize, dgm_fields, "src/DeepGM/train.py"),
    "flow": (flow_parser, flow_finalize, flow_fields, "src/GradientFlow/flow.py"),
    "color": (color_parser, color_finalize, color_fields, "src/ColorTransfer/transfer.py"),
}


def parse(app, argv=None):
    parser, finalize, _, _ = APPS[app]
    return finalize(parser().parse_args(argv))


def fields(app, args):
    return APPS[app][2](args)


def run_dir(app, argv):
    a = parse(app, argv)
    return os.path.join(a.out_root, run_name(fields(app, a)))


def script(app):
    return os.path.join(REPO, APPS[app][3])
