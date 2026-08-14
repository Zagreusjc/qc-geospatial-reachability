"""Phase 6 -- Oracle architectures (importable module, not a run step).

Defines the two neural architectures compared in the ablation:

  Architecture A -- Siamese: shared twin branch FC(128->256->128->64) with
    BatchNorm + ReLU; combine by absolute element-wise difference |z_u - z_v|;
    prediction head FC(64->32->1). Weight sharing makes the output symmetric by
    construction.

  Architecture B -- MLP baseline: concatenated [z_u; z_v] (256-d) through five FC
    layers to a linear scalar; symmetry only encouraged via reversed-pair training.

Used by 07_train.py and 08_evaluate.py via `build_model(arch)`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("06_models")

# NOTE: torch imports and nn.Module definitions (SiameseOracle, MLPOracle) plus a
# build_model(arch) factory will be implemented in this phase. Kept as a stub so
# the repository structure is reviewable before adding the training dependency.


def build_model(arch: str):
    """Factory returning a Siamese or MLP oracle. To be implemented."""
    raise SystemExit(
        f"[stub] 06_models.build_model('{arch}') not implemented yet."
    )


if __name__ == "__main__":
    log.info("06_models is an importable module (Siamese + MLP); nothing to run.")
