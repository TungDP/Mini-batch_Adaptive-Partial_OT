"""Pieces shared by the digits and Office-Home / VisDA trainers.

Sampler, image lists, transforms, networks and the LR schedule are copied from the m-POT
reference code (UT-Austin-Data-Science-Group/Mini-batch-OT, itself from JUMBOT) so that the
protocol matches the published baselines.  ``transport_step`` is the one place where a
transport rule meets the network.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from apot.rules import Batch  # noqa: E402


# ---------------------------------------------------------------- sampler (stratified source)
class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """m // C samples of every class per batch; the remainder goes to classes drawn at random."""

    def __init__(self, labels, batch_size):
        labels = np.asarray(labels)
        self.classes = sorted(set(labels.tolist()))
        n_classes = len(self.classes)
        self._n_samples, self._n_remain = divmod(batch_size, n_classes)
        if self._n_samples == 0:
            raise ValueError(f"batch_size should be bigger than the number of classes, got {batch_size}")
        self._class_iters = [InfiniteSliceIterator(np.where(labels == c)[0], c) for c in self.classes]
        self._n_batches = len(labels) // batch_size
        if self._n_batches == 0:
            raise ValueError(f"Dataset is not big enough to generate batches with size {batch_size}")

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            add_class = set(np.random.choice(self.classes, self._n_remain, replace=False))
            for it in self._class_iters:
                indices.extend(it.get(self._n_samples + (1 if it.class_ in add_class else 0)))
            np.random.shuffle(indices)
            yield indices
        for it in self._class_iters:
            it.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        self.array, self.i, self.class_ = array, 0, class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        if len_ < n:
            self.reset()
            np.random.shuffle(self.array)
            mul, rest = divmod(n, len_)
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))
        if len_ - self.i < n:
            self.reset()
        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i: self.i]


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------- image lists and transforms
class ImageList(Dataset):
    """Lines '<path> <label>'; paths in the lists start with '/data/...', resolved from root."""

    def __init__(self, lines, root=".", transform=None):
        self.imgs = []
        for line in lines:
            p, y = line.split()
            self.imgs.append((os.path.join(root, p.lstrip("/")), int(y)))
        self.transform = transform

    def __getitem__(self, i):
        path, y = self.imgs[i]
        with open(path, "rb") as f, Image.open(f) as img:
            img = img.convert("RGB")
        return (self.transform(img) if self.transform else img), y

    def __len__(self):
        return len(self.imgs)


class ResizeImage:
    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop:
    def __init__(self, size, start_x, start_y):
        self.size = (int(size), int(size))
        self.start_x, self.start_y = start_x, start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ForceFlip:
    def __call__(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)


_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([ResizeImage(resize_size), transforms.RandomResizedCrop(crop_size),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), _NORM])


def image_test(resize_size=256, crop_size=224):
    c = (resize_size - crop_size - 1) / 2
    return transforms.Compose([ResizeImage(resize_size), PlaceCrop(crop_size, c, c), transforms.ToTensor(), _NORM])


def image_train_crop(resize_size=256, crop_size=224):
    """Training augmentation of the m-POT / BA3US partial-DA code (no random resizing)."""
    return transforms.Compose([transforms.Resize((resize_size, resize_size)), transforms.RandomCrop(crop_size),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), _NORM])


def image_test_center(resize_size=256, crop_size=224):
    """Test transform of the m-POT / BA3US partial-DA code."""
    return transforms.Compose([transforms.Resize((resize_size, resize_size)), transforms.CenterCrop(crop_size),
                               transforms.ToTensor(), _NORM])


def image_test_10crop(resize_size=256, crop_size=224):
    first, center, last = 0, (resize_size - crop_size - 1) / 2, resize_size - crop_size - 1
    starts = [(first, first), (last, last), (last, first), (first, last), (center, center)]
    out = []
    for flip in (True, False):
        for sx, sy in starts:
            t = [ResizeImage(resize_size)] + ([ForceFlip()] if flip else []) + \
                [PlaceCrop(crop_size, sx, sy), transforms.ToTensor(), _NORM]
            out.append(transforms.Compose(t))
    return out


# ---------------------------------------------------------------- networks
def init_weights(m):
    name = m.__class__.__name__
    if name.find("Conv2d") != -1 or name.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif name.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ResNetFc(nn.Module):
    """ResNet-50 + bottleneck + new classifier; forward returns (features, logits)."""

    def __init__(self, class_num, bottleneck_dim=512, weights="IMAGENET1K_V1"):
        super().__init__()
        r = models.resnet50(weights=weights)
        self.feature_layers = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2,
                                            r.layer3, r.layer4, r.avgpool)
        self.bottleneck = nn.Linear(r.fc.in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(self.feature_layers(x).flatten(1))
        return x, self.fc(x)

    def get_parameters(self):
        return [{"params": self.feature_layers.parameters(), "lr_mult": 1, "decay_mult": 2},
                {"params": self.bottleneck.parameters(), "lr_mult": 10, "decay_mult": 2},
                {"params": self.fc.parameters(), "lr_mult": 10, "decay_mult": 2}]


def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    lr = lr * (1 + gamma * iter_num) ** (-power)
    for g in optimizer.param_groups:
        g["lr"] = lr * g["lr_mult"]
        g["weight_decay"] = weight_decay * g["decay_mult"]
    return optimizer


def weights_init_digits(m):
    name = m.__class__.__name__
    if name.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif name.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif name.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


class Classifier1(nn.Module):
    def __init__(self, nclass=10, in_feature=128):
        super().__init__()
        self.fc = nn.Linear(in_feature, nclass)

    def forward(self, x):
        return self.fc(x)


class SVHNGenerator(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()
        ch = [input_channel, 32, 32, 64, 64, 128, 128]
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1) for i in range(6)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(ch[i + 1]) for i in range(6)])
        self.linear1 = nn.Linear(128 * 4 * 4, 128)

    def forward(self, x):
        h = x
        for i in range(6):
            h = F.relu(self.bns[i](self.convs[i](h)))
            if i % 2 == 1:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        return torch.sigmoid(self.linear1(h.flatten(1)))


class USPSGenerator(nn.Module):
    def __init__(self, input_channel=1):
        super().__init__()
        self.c1 = nn.Conv2d(input_channel, 32, kernel_size=5, stride=1, padding=0)
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.linear1 = nn.Linear(128 * 4 * 4, 128)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm2d(32), nn.BatchNorm2d(64), nn.BatchNorm2d(128)

    def forward(self, x):
        h = F.max_pool2d(F.relu(self.bn1(self.c1(x))), kernel_size=2, stride=2)
        h = F.relu(self.bn2(self.c2(h)))
        h = F.max_pool2d(F.relu(self.bn3(self.c3(h))), kernel_size=2, stride=2)
        return torch.sigmoid(self.linear1(h.flatten(1)))


# ---------------------------------------------------------------- the transport step
def jdot_cost(g_s, g_t, ys, logits_t, n_classes, eta1, eta2):
    """DeepJDOT / JUMBOT ground cost: eta1 ||g(x_s) - g(x_t)||^2 + eta2 CE(y_s, f(g(x_t)))."""
    pred_t = F.softmax(logits_t.float(), 1)
    ys_oh = F.one_hot(ys, num_classes=n_classes).float()
    M_embed = torch.cdist(g_s.float(), g_t.float()) ** 2
    M_sce = -torch.mm(ys_oh, torch.log(pred_t + 1e-10).t())
    agree = torch.mm(ys_oh, pred_t.t())  # p(x)^T q(z) for the AOT cost
    return eta1 * M_embed + eta2 * M_sce, M_embed, agree, pred_t


def transport_step(rule, M, ys, yt, pred_t, g_s, g_t, step, total_steps, masslog, k_idx=0,
                   M_embed=None, agree=None, aot_alpha=0.0, aot_beta=0.0):
    """Choose the plan for one batch pair with `rule` and return the transport loss <P, M>
    (or <P, C_signed> for the AOT rule).  Logs the moved mass and, when target labels are
    available, the batch overlap and cross-class mass."""
    need = set(rule.needs)
    C_signed_t = None
    if "C_signed" in need:
        C_signed_t = aot_alpha * M_embed - aot_beta * agree
    b = Batch(
        C=M.detach().double().cpu().numpy(),
        ys=ys.cpu().numpy() if ys is not None else None,
        yt=yt.cpu().numpy() if ("yt" in need and yt is not None) else None,
        pt=pred_t.detach().cpu().numpy() if "pt" in need else None,
        fs=g_s.detach().float().cpu().numpy() if "fs" in need else None,
        ft=g_t.detach().float().cpu().numpy() if "ft" in need else None,
        C_signed=C_signed_t.detach().double().cpu().numpy() if C_signed_t is not None else None,
        step=step, total_steps=total_steps,
    )
    t0 = time.perf_counter()
    P, info = rule.plan(b)
    ms = 1000 * (time.perf_counter() - t0)
    if masslog is not None:
        masslog.add(step, P, info, b.C, ms, b.ys, yt.cpu().numpy() if yt is not None else None, k=k_idx)
    P_t = torch.from_numpy(P).float().to(M.device)
    cost = C_signed_t if rule.loss_on_signed else M
    return torch.sum(P_t * cost.float())
