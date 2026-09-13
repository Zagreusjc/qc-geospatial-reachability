"""Phase 6 -- Oracle architectures (importable module, not a run step).

Defines the two neural architectures compared in the ablation:

  Architecture A -- Siamese: shared twin branch FC(128->256->128->64) with
    LayerNorm + ReLU; combine by absolute element-wise difference |z_u - z_v|;
    prediction head FC(64->32->1). Weight sharing makes the output symmetric by
    construction. LayerNorm normalizes per-sample rather than per-batch, so
    there is no running-statistic mismatch between training and evaluation.

  Architecture B -- MLP baseline: concatenated [z_u; z_v] (256-d) through five FC
    layers with LayerNorm + ReLU on every hidden layer (matching Architecture A's
    branch normalization scheme), ending in a single linear output neuron (no
    norm/activation on the final scalar output). Symmetry is only encouraged via
    reversed-pair training, not enforced by construction.

Used by 07_train.py and 08_evaluate.py via `build_model(arch)`.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("06_models")


class SiameseOracle(nn.Module):
    """Twin shared-weight branch + |z_u - z_v| comparison head."""

    def __init__(self, branch_dims=None, head_dims=None):
        super().__init__()
        branch_dims = branch_dims or config.SIAMESE_BRANCH_DIMS
        head_dims = head_dims or config.SIAMESE_HEAD_DIMS

        branch_layers = []
        for i in range(len(branch_dims) - 1):
            branch_layers.append(nn.Linear(branch_dims[i], branch_dims[i + 1]))
            branch_layers.append(nn.LayerNorm(branch_dims[i + 1]))
            branch_layers.append(nn.ReLU())
        self.branch = nn.Sequential(*branch_layers)

        head_layers = []
        for i in range(len(head_dims) - 1):
            head_layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
            if i < len(head_dims) - 2:
                head_layers.append(nn.ReLU())
        self.head = nn.Sequential(*head_layers)

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        h_u = self.branch(z_u)
        h_v = self.branch(z_v)
        diff = torch.abs(h_u - h_v)
        return self.head(diff).squeeze(-1)


class MLPOracle(nn.Module):
    """Baseline: concatenate both embeddings and feed through a normalized MLP.

    Symmetry is NOT enforced by construction and NOT augmented during training --
    this is intentional. The absence of any symmetry mechanism makes the
    architecture comparison against Siamese (H4) cleaner and more interpretable:
    any performance gap reflects the architectural advantage of shared-weight
    twin branches, not augmentation differences.
    """

    def __init__(self, dims=None):
        super().__init__()
        dims = dims or config.MLP_DIMS  # dims[0] == 2 * N2V_DIM (concatenated input)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_u, z_v], dim=-1)
        return self.net(x).squeeze(-1)


def build_model(arch: str) -> nn.Module:
    """Factory returning a Siamese or MLP oracle."""
    if arch == "siamese":
        return SiameseOracle()
    if arch == "mlp":
        return MLPOracle()
    raise ValueError(f"Unknown architecture: {arch!r}; expected one of {config.ARCHITECTURES}")


if __name__ == "__main__":
    for arch in config.ARCHITECTURES:
        model = build_model(arch)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("%s: %d parameters\n%s", arch, n_params, model)
