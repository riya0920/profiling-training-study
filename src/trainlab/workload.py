"""The workload under study: a small CNN over a synthetic image dataset.

The dataset carries a deliberate, tunable per-sample CPU cost (`decode_cost`).
That is not padding -- it reproduces the single most common real training
bottleneck (JPEG decode + augmentation on the CPU starving the accelerator) in a
form that is reproducible on any machine and does not require downloading
ImageNet. The ladder's first rung exists to remove exactly this stall, and with a
zero-cost dataset there would be nothing to remove and the rung would be theatre.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SyntheticImages(Dataset):
    def __init__(self, n: int = 20_000, size: int = 32, n_classes: int = 10, decode_cost: int = 400, seed: int = 0):
        self.n, self.size, self.n_classes = n, size, n_classes
        self.decode_cost = decode_cost
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        label = int(idx % self.n_classes)
        img = rng.normal(loc=label / self.n_classes, scale=1.0, size=(3, self.size, self.size)).astype("float32")

        # Simulated decode/augment cost. Real work (not sleep) so it competes for
        # the GIL and for cores exactly the way image decoding does -- a sleep
        # would be trivially hidden by any number of workers and would make the
        # dataloader rung look better than it is.
        for _ in range(self.decode_cost):
            img = np.ascontiguousarray(img[:, ::-1, :] * 1.0001)
        return torch.from_numpy(img.copy()), label


class SmallCNN(nn.Module):
    """Deliberately small. The study is about the training loop, not the model."""

    def __init__(self, n_classes: int = 10, width: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def scaled_lr(base_lr: float, base_batch: int, batch: int, rule: str = "linear") -> float:
    """LR scaling rule, fixed BEFORE any scaling run -- not tuned afterwards.

    Linear scaling (Goyal et al. 2017) for batch sizes within ~8x of the base;
    sqrt scaling is the conservative alternative when the linear rule destabilises.
    Deciding this after seeing the results would make the scaling study a
    hyperparameter search wearing a scaling study's clothes.
    """
    ratio = batch / base_batch
    if rule == "linear":
        return base_lr * ratio
    if rule == "sqrt":
        return base_lr * math.sqrt(ratio)
    raise ValueError(rule)


def warmup_steps(total_steps: int, fraction: float = 0.05) -> int:
    """Linear warmup length. Large-batch runs diverge in the first few hundred
    steps without it, and a divergence blamed on 'DDP' is usually this."""
    return max(int(total_steps * fraction), 1)
