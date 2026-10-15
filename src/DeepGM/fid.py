"""FID in memory.  Same Inception network (pytorch-fid weights) and the same uint8 rounding as
the reference pipeline, which wrote generated images to PNG and read them back."""
import numpy as np
import torch
from scipy import linalg

from inception import InceptionV3


class FID:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    @torch.no_grad()
    def activations(self, images01, batch_size=200):
        """images01: float tensor (N, 3, H, W) in [0, 1]."""
        out = []
        for i in range(0, len(images01), batch_size):
            x = images01[i: i + batch_size].to(self.device)
            out.append(self.model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out)

    @staticmethod
    def stats(act):
        return act.mean(0), np.cov(act, rowvar=False)


def to_uint8_01(x_tanh):
    """Generator output in [-1, 1] -> uint8 -> back to [0, 1] (the PNG round trip)."""
    x = ((x_tanh.clamp(-1, 1) / 2.0 + 0.5) * 255).to(torch.uint8)
    return x.float() / 255.0


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    covmean = covmean[0] if isinstance(covmean, tuple) else covmean
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = covmean[0] if isinstance(covmean, tuple) else covmean
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
