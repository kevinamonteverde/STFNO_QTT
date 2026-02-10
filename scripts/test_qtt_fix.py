#!/usr/bin/env python3
"""
Quick test to verify QTT fix for quantize_last_ndims=5
"""
import sys
import torch
sys.path.insert(0, '/global/u2/p/pepi/STFNO_QTT')

from stfno.fourier_transform_3d_layer_factorized import FactorizedSpectralConv3d

def test_qtt_quantize5():
    """Test QTT with quantize_last_ndims=5 (Cin, Cout, and spatial)"""
    print("Testing QTT with quantize_last_ndims=5...")

    # Configuration that was failing: modes=6, width=8, q=5
    in_channels = 8
    out_channels = 8
    modes = 6

    # Create layer with QTT and quantize_last_ndims=5
    layer = FactorizedSpectralConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes1=modes,
        modes2=modes,
        modes3=modes,
        factorization='qtt',
        rank=4,
        implementation='factorized',
        quantize_last_ndims=5
    )

    # Create dummy input (batch=2, channels=8, spatial=3x3x6)
    # The spatial dims match modes//2, modes//2, modes
    x = torch.randn(2, in_channels, modes//2, modes//2, modes)

    print(f"Input shape: {x.shape}")

    try:
        # Forward pass
        out = layer(x)
        print(f"Output shape: {out.shape}")
        print("✓ Test passed! QTT with quantize_last_ndims=5 works.")
        return True
    except Exception as e:
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        return False

def test_qtt_quantize3():
    """Test QTT with quantize_last_ndims=3 (spatial only) - should still work"""
    print("\nTesting QTT with quantize_last_ndims=3...")

    in_channels = 8
    out_channels = 8
    modes = 6

    layer = FactorizedSpectralConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes1=modes,
        modes2=modes,
        modes3=modes,
        factorization='qtt',
        rank=4,
        implementation='factorized',
        quantize_last_ndims=3
    )

    x = torch.randn(2, in_channels, modes//2, modes//2, modes)

    print(f"Input shape: {x.shape}")

    try:
        out = layer(x)
        print(f"Output shape: {out.shape}")
        print("✓ Test passed! QTT with quantize_last_ndims=3 still works.")
        return True
    except Exception as e:
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("QTT Fix Verification Test")
    print("=" * 60)

    test3_ok = test_qtt_quantize3()
    test5_ok = test_qtt_quantize5()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  quantize_last_ndims=3: {'PASS' if test3_ok else 'FAIL'}")
    print(f"  quantize_last_ndims=5: {'PASS' if test5_ok else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if (test3_ok and test5_ok) else 1)
