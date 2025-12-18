from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import opt_einsum as oe
import tensorly as tl
import torch
from tensorly.decomposition import tensor_train

# Ensure tensorly matches the rest of the STFNO code base
try:
    tl.set_backend("pytorch")
except Exception:
    pass


def regular_contraction(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reference einsum contraction matching scripts/contraction_test.py."""
    return oe.contract("abijk,aijk->bijk", weight, x, optimize="optimal", backend="torch")


def regular_contraction_batched(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched contraction variant (formerly regular_contraction2)."""
    cin, cout, nx, ny, nz = weight.shape
    n_total = nx * ny * nz
    w_mat = weight.reshape(cin, cout, n_total)
    x_mat = x.reshape(cin, n_total)
    result = oe.contract("abd,ad->bd", w_mat, x_mat, optimize="optimal", backend="torch")
    return result.reshape(cout, nx, ny, nz)


def to_tt(tensor: torch.Tensor, max_rank: int = 16) -> Tuple[List[torch.Tensor], Tuple[int, ...]]:
    """Decompose tensor into TT cores using tensorly.tensor_train."""
    original_shape = tensor.shape
    cores = tensor_train(tensor, rank=max_rank)
    return list(cores), tuple(original_shape)


def to_qtt(tensor: torch.Tensor, max_rank: int = 16) -> Tuple[List[torch.Tensor], Tuple[int, ...]]:
    """Quantized-TT decomposition used in scripts; requires power-of-two dims."""
    original_shape = tensor.shape
    for dim in original_shape:
        if dim & (dim - 1):
            raise ValueError(f"Dimension {dim} is not a power of two")
    binary_shape: List[int] = []
    for dim in original_shape:
        binary_shape.extend([2] * int(np.log2(dim)))
    tensor_binary = tensor.reshape(binary_shape)
    cores = tensor_train(tensor_binary, rank=max_rank)
    return list(cores), tuple(original_shape)


def tt_contraction(weight_cores: Sequence[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Apply TT cores to input matching contraction_test einsum.

    Supports x shaped either (cin, nx, ny, nz) or batched (batch, cin, nx, ny, nz).
    """
    if not weight_cores:
        raise ValueError("weight_cores must be non-empty")
    cores = list(weight_cores)
    cores[0] = cores[0].squeeze()
    cores[-1] = cores[-1].squeeze()
    if x.dim() == 4:
        equation = "ap,pbq,qir,rjs,sk,aijk->bijk"
    elif x.dim() == 5:
        # Batch dimension (l) carried through: laijk -> lbijk
        equation = "ap,pbq,qir,rjs,sk,laijk->lbijk"
    else:
        raise ValueError(f"Expected x to have 4 or 5 dims, got shape {tuple(x.shape)}")

    return oe.contract(equation, *cores, x, optimize="optimal", backend="torch")


@dataclass
class TTProfileResult:
    """Timing info for reproducible TT benchmarks."""
    regular_ms: float
    batched_ms: float
    tt_ms: float


def benchmark_tt(
    weight: torch.Tensor,
    x: torch.Tensor,
    rank: int,
    *,
    cores: Sequence[torch.Tensor] | None = None,
) -> tuple[TTProfileResult, List[torch.Tensor]]:
    """Replicates contraction_test timing; optionally reuse precomputed cores."""
    t0 = time.time(); regular_contraction(weight, x); regular_ms = (time.time() - t0) * 1e3
    t0 = time.time(); regular_contraction_batched(weight, x); batched_ms = (time.time() - t0) * 1e3
    if cores is None:
        cores, _ = to_tt(weight, max_rank=rank)
    t0 = time.time(); tt_contraction(cores, x); tt_ms = (time.time() - t0) * 1e3
    profile = TTProfileResult(regular_ms=regular_ms, batched_ms=batched_ms, tt_ms=tt_ms)
    return profile, list(cores)
