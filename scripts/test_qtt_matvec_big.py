#!/usr/bin/env python3
"""Run several larger QTT/TT matvec tests comparing einsum vs core-by-core.

This script intentionally avoids reconstructing dense tensors for large cases.
It reports max/mean absolute differences and an allclose boolean per test.
"""
import os
import torch
import traceback
from stfno.qtt import QTTWeight


def run_case(name, weight_shape, quantize_last_ndims, N_rows, rank=4, dtype=torch.float32):
    print(f"\n=== Case: {name} weight_shape={weight_shape} N={N_rows} rank={rank} quantize_last_ndims={quantize_last_ndims} ===")
    try:
        torch.manual_seed(0)
        w = QTTWeight(
            weight_shape=weight_shape,
            quantize_last_ndims=quantize_last_ndims,
            rank=rank,
            init_std=0.02,
            base=2,
            dtype=dtype,
            device=None,
            tt_order='in-bits-out',
        )
        print("tt_order:", w._tt_order, "type:", type(w.tt).__name__)
        cores = w._get_tt_cores()
        if cores is not None:
            print(f"Extracted {len(cores)} TT cores. Shapes:", [tuple(c.shape) for c in cores])

        Cin = int(weight_shape[0])
        # Create inputs
        x_rows = torch.randn(N_rows, Cin, dtype=dtype)
        spatial_sizes = tuple(int(s) for s in weight_shape[-3:]) if len(weight_shape) >= 5 else (1, 1, 1)
        indices = torch.stack([torch.randint(0, s, (N_rows,)) for s in spatial_sizes], dim=1)

        # Run einsum-enabled
        w._use_single_einsum = True
        y_einsum = w.matmul(x_rows, indices)
        te = w.get_last_op_timing() or {}
        print("einsum path info: einsum_used=", te.get('einsum_used'), "timing keys=", list(te.keys()))

        # Run core-by-core
        w._use_single_einsum = False
        y_core = w.matmul(x_rows, indices)
        tc = w.get_last_op_timing() or {}
        print("core path info: einsum_used=", tc.get('einsum_used'), "timing keys=", list(tc.keys()))

        # Compare
        diff = (y_einsum - y_core).abs()
        maxd = float(diff.max())
        meand = float(diff.mean())
        allc = torch.allclose(y_einsum, y_core, rtol=1e-4, atol=1e-5)
        print("max abs diff:", maxd)
        print("mean abs diff:", meand)
        print("allclose (rtol=1e-4, atol=1e-5):", bool(allc))

    except Exception as e:
        print("Test failed with exception:")
        traceback.print_exc()


if __name__ == '__main__':
    # Optionally enable detailed timing inside matmul by uncommenting the next line
    # os.environ['STFNO_QTT_TIMING'] = '1'

    # Define a few progressively larger test cases. Adjust N rows to keep memory reasonable.
    cases = [
        ("small", (4, 2, 8, 8, 8), 3, 512, 2),
        ("medium", (16, 8, 16, 16, 16), 3, 512, 4),
        ("large", (32, 16, 32, 32, 32), 3, 256, 4),
        ("xlarge", (64, 32, 32, 32, 32), 3, 128, 4),
    ]

    for name, shape, qnd, N, rnk in cases:
        run_case(name, shape, qnd, N, rank=rnk)

    print("\nAll cases complete.")
