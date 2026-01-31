from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(1, total_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def class_weights_from_counts(counts):
    # inverse frequency, normalized to sum=C
    counts = torch.tensor(counts, dtype=torch.float32)
    w = 1.0 / torch.clamp(counts, min=1.0)
    w = w * (len(counts) / w.sum())
    return w
