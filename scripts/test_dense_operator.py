#!/usr/bin/env python3
"""Tests for DenseOperator3d, QTTOperator3d, and DenseOrQTT3DOperator."""
from __future__ import annotations

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from stfno.dense_operator_3d import (
    DenseOperator3d,
    QTTOperator3d,
    DenseOrQTT3DOperator,
    _contract_qtt_dense_operator,
)


def test_dense_shapes():
    """Test DenseOperator3d produces correct output shapes."""
    print("Test 1: DenseOperator3d shapes")

    configs = [
        # (Cin, Cout, spatial_in, spatial_out)
        (2, 3, (4, 2, 2), (4, 2, 2)),
        (4, 2, (2, 2, 2), (4, 4, 4)),
        (1, 1, (8, 4, 2), (2, 2, 2)),
    ]

    for cin, cout, sp_in, sp_out in configs:
        op = DenseOperator3d(cin, cout, sp_in, sp_out)
        x = torch.randn(2, cin, *sp_in)
        y = op(x)
        expected = (2, cout, *sp_out)
        assert y.shape == expected, f"Expected {expected}, got {y.shape}"
        print(f"  Cin={cin}, Cout={cout}, {sp_in}->{sp_out}: shape={y.shape} -- PASS")

    print()


def test_qtt_shapes():
    """Test QTTOperator3d produces correct output shapes."""
    print("Test 2: QTTOperator3d shapes")

    configs = [
        # (Cin, Cout, spatial_in, spatial_out, rank)
        (2, 2, (4, 2, 2), (4, 2, 2), 4),
        (2, 4, (2, 2, 2), (2, 2, 2), 4),
        (4, 2, (4, 2, 2), (2, 4, 2), 8),
    ]

    for cin, cout, sp_in, sp_out, rank in configs:
        op = QTTOperator3d(cin, cout, sp_in, sp_out, rank=rank, quantize_last_ndims=8)
        x = torch.randn(2, cin, *sp_in)
        y = op(x)
        expected = (2, cout, *sp_out)
        assert y.shape == expected, f"Expected {expected}, got {y.shape}"
        print(f"  Cin={cin}, Cout={cout}, {sp_in}->{sp_out}, rank={rank}: shape={y.shape} -- PASS")

    print()


def test_dense_vs_qtt_equivalence():
    """Compare dense and QTT by materialising the QTT weight to dense."""
    print("Test 3: Dense vs QTT equivalence")

    torch.manual_seed(42)
    Cin, Cout = 2, 2
    sp_in = (4, 2, 2)
    sp_out = (4, 2, 2)

    # Build a dense operator
    dense_op = DenseOperator3d(Cin, Cout, sp_in, sp_out, init_std=0.01)

    # Build a QTT operator and materialise its weight to dense
    qtt_op = QTTOperator3d(Cin, Cout, sp_in, sp_out, rank=64, quantize_last_ndims=8)
    qtt_dense_w = qtt_op.qtt_weight.to_tensor()  # should be 8-D

    expected_w_shape = (Cin, *sp_in, Cout, *sp_out)
    assert qtt_dense_w.shape == expected_w_shape, (
        f"QTT materialised weight shape mismatch: {qtt_dense_w.shape} vs {expected_w_shape}"
    )
    print(f"  QTT materialised weight shape: {qtt_dense_w.shape} -- correct")

    # Overwrite the dense operator's weight with the materialised QTT weight
    with torch.no_grad():
        dense_op.weight.copy_(qtt_dense_w)

    x = torch.randn(2, Cin, *sp_in)

    y_dense = dense_op(x)
    y_qtt = qtt_op(x)

    max_err = (y_dense - y_qtt).abs().max().item()
    rel_err = max_err / (y_dense.abs().max().item() + 1e-12)

    print(f"  max abs error:  {max_err:.2e}")
    print(f"  relative error: {rel_err:.2e}")

    # With rank=64 and tiny dims (all powers of 2), the relative error
    # should be near float32 machine precision.
    assert rel_err < 1e-4, f"Relative error too large: {rel_err:.2e}"
    print(f"  Equivalence -- PASS")
    print()


def test_gradient():
    """Verify backward works for both operators."""
    print("Test 4: Gradient test")

    for name, Op, kwargs in [
        ("Dense", DenseOperator3d, dict(
            in_channels=2, out_channels=2,
            spatial_in=(4, 2, 2), spatial_out=(4, 2, 2),
        )),
        ("QTT", QTTOperator3d, dict(
            in_channels=2, out_channels=2,
            spatial_in=(4, 2, 2), spatial_out=(4, 2, 2),
            rank=4,
        )),
    ]:
        op = Op(**kwargs)
        x = torch.randn(1, 2, 4, 2, 2, requires_grad=True)
        y = op(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, f"{name}: x.grad is None"
        assert x.grad.shape == x.shape, f"{name}: x.grad shape mismatch"

        has_grad = any(p.grad is not None for p in op.parameters())
        assert has_grad, f"{name}: no parameter gradients"
        print(f"  {name}: gradients flow -- PASS")

    print()


def test_wrapper():
    """Test DenseOrQTT3DOperator dispatch for both modes."""
    print("Test 5: DenseOrQTT3DOperator wrapper")

    x = torch.randn(1, 2, 4, 2, 2)

    for fact in ('dense', 'qtt'):
        wrapper = DenseOrQTT3DOperator(
            in_channels=2, out_channels=2,
            spatial_in=(4, 2, 2), spatial_out=(4, 2, 2),
            factorization=fact, rank=4,
        )
        y = wrapper(x)
        expected = (1, 2, 4, 2, 2)
        assert y.shape == expected, f"{fact}: expected {expected}, got {y.shape}"
        print(f"  factorization='{fact}': shape={y.shape} -- PASS")

    print()


def test_non_power_of_2():
    """Test QTT with non-power-of-2 dimensions (requires padding)."""
    print("Test 6: QTT with non-power-of-2 dims")

    # Cin=3, Cout=3, spatial 3x3x3 -> 3x3x3
    # All dims are 3 (not a power of 2), so padding to 4 is required
    op = QTTOperator3d(
        in_channels=3, out_channels=3,
        spatial_in=(3, 3, 3), spatial_out=(3, 3, 3),
        rank=4, quantize_last_ndims=8,
    )
    x = torch.randn(2, 3, 3, 3, 3)
    y = op(x)
    expected = (2, 3, 3, 3, 3)
    assert y.shape == expected, f"Expected {expected}, got {y.shape}"
    print(f"  shape={y.shape} -- PASS")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Dense 8-Way Operator Tests")
    print("=" * 60)
    print()

    tests = [
        test_dense_shapes,
        test_qtt_shapes,
        test_dense_vs_qtt_equivalence,
        test_gradient,
        test_wrapper,
        test_non_power_of_2,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
