#!/usr/bin/env python3
"""Quick test for QTT contraction methods.

Creates a QTTWeight (uses tltorch FactorizedTensor under the hood),
generates random input rows and spatial indices, then runs the
operator twice on the exact same TT: on\ce with the single-einsum
fast-path enabled and once with the incremental core-by-core path.

Prints shapes and max absolute difference and allclose result.
"""
import os
import torch
from stfno.qtt import QTTWeight


def main():
    torch.manual_seed(0)

    # Small synthetic shape: (Cin, Cout, Nx, Ny, Nz)
    weight_shape = (4, 2, 8, 8, 8)
    quantize_last_ndims = 3

    # Create a QTTWeight with in-bits-out ordering to enable einsum fast-path
    w = QTTWeight(
        weight_shape=weight_shape,
        quantize_last_ndims=quantize_last_ndims,
        rank=2,
        init_std=0.02,
        base=2,
        dtype=torch.float32,
        device=None,
        tt_order='in-bits-out',
    )

    N = 20
    Cin = weight_shape[0]
    x_rows = torch.randn(N, Cin, dtype=torch.float32)

    # Random indices within spatial dims (kx, ky, kz)
    spatial_sizes = weight_shape[-3:]
    indices = torch.stack([torch.randint(0, s, (N,)) for s in spatial_sizes], dim=1)

    # Ensure we operate on the same TT: toggle the einsum flag on the same object
    w._use_single_einsum = True
    y_einsum = w.matmul(x_rows, indices)

    w._use_single_einsum = False
    y_core = w.matmul(x_rows, indices)

    print("y_einsum shape:", tuple(y_einsum.shape))
    print("y_core shape:", tuple(y_core.shape))
    diff = (y_einsum - y_core).abs()
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    print("allclose (rtol=1e-4, atol=1e-5):", torch.allclose(y_einsum, y_core, rtol=1e-4, atol=1e-5))


if __name__ == '__main__':
    main()
