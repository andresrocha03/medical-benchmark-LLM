"""Shared utilities for YAGO pretraining and frozen TARTE evaluation."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42


def set_seed(seed=SEED):
    """Seed the random generators used by the evaluation pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Prefer a hardware accelerator when PyTorch can use one."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ProjectionLayer(nn.Module):
    """Projection rho(x) = Linear(ReLU(LayerNorm(x))) from the paper."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        return self.linear(F.relu(self.norm(inputs)))
