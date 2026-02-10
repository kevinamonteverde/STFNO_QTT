# In this script I will try to figure out changing the tt function to accommodate qtt
#
# REGULAR TT equation: abcde,gbh,hfi,icj,jdk,kel->afcde
# Input x:     a b c d e     (Batch, Cin, Nx, Ny, Nz)
# Core 1:      g b h         (rank_0, Cin, rank_1)
# Core 2:      h f i         (rank_1, Cout, rank_2)
# Core 3:      i c j         (rank_2, Nx, rank_3)
# Core 4:      j d k         (rank_3, Ny, rank_4)
# Core 5:      k e l         (rank_4, Nz, rank_5)
# Output:      a f c d e     (Batch, Cout, Nx, Ny, Nz)
#
# QTT equation (for 8x8x8 spatial = 2^3 x 2^3 x 2^3):
# The spatial dimensions are folded into bits:
#   Nx=8 -> 3 bits (c0, c1, c2)
#   Ny=8 -> 3 bits (d0, d1, d2)
#   Nz=8 -> 3 bits (e0, e1, e2)
#
# QTT cores (in-out-bits ordering, same as TT):
# Core 1:       g b h           (rank_0, Cin, rank_1)
# Core 2:       h f i           (rank_1, Cout, rank_2)
# Cores 3-11:   i c0 j, j c1 k, ... (bit cores for all spatial bits)
#
# Input x must be reshaped: (Batch, Cin, Nx, Ny, Nz) -> (Batch, Cin, c0,c1,c2, d0,d1,d2, e0,e1,e2)

"""
Quantized Tensor Train (QTT) contraction implementation.
Simply adapts the TT contraction to handle quantized dimensions by folding spatial dimensions 
into bits at the input tensor, then unfolding result after contraction.

NOTE ON FUNCTION: separable controls whether the weight has separate input/output channels.
Separable = True means input and output channels are the same (no separate output channel dimension).
Separable = False means distinct input/output channels.
"""


import tensorly as tl
import torch
import math
import time

# Set device - use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# todo:
#on gpu try .view() instead of .reshape() for better performance 

def _contract_tt_factorized(x, tt_weight, separable=False, verbose=False, return_timing=False):
    """TT contraction using FactorizedTensor factors (borrowed pattern from spectral_convolution).

    This is called contract_tt() in spectral_convolution.py, from the FNO repo.

    This is distinct from the single-einsum core contraction: it uses the TT factors
    stored in the FactorizedTensor and a multi-operand einsum.
    """
    timings = {}

    # === STEP 1: Build einsum equation ===
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()

    order = tl.ndim(x) #gets number of dimensions of input x

    x_syms = list(einsum_symbols[:order]) #for order 5, x_syms = ['a','b','c','d','e'] - gets first 'order' symbols from alphabet

    #a = Batch, b = Cin, c,d,e = spatial dims for input tensor x

    weight_syms = list(x_syms[1:])  # takes all symbols except the first (batch size)
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # weight_syms = ['b', 'f', 'c', 'd', 'e'] - adds output channel symbol after input channel symbol
        out_syms = list(weight_syms) #out_syms = ['b', 'f', 'c', 'd', 'e']
        out_syms[0] = x_syms[0] #out syms = ['a', 'f', 'c', 'd', 'e'] - keeps batch size symbol the same
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :]) #rank_syms = ['g','h','i','j','k','l'] - gets symbols for TT ranks starting from order+1 position
    tt_syms = []
    for i, s in enumerate(weight_syms): #builds einsum symbols for each TT core
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.is_cuda:
        torch.cuda.synchronize()
    timings['setup_ms'] = (time.perf_counter() - start) * 1000

    if verbose:
        print(f"TT equation: {eq}")

    # === STEP 2: Execute einsum contraction ===
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = tl.einsum(eq, x, *tt_weight.factors)
    if x.is_cuda:
        torch.cuda.synchronize()
    timings['einsum_ms'] = (time.perf_counter() - start) * 1000

    if return_timing:
        return result, timings
    return result


def _contract_qtt_factorized(x, qtt_weight, separable=False, verbose=True, return_timing=False):
    """QTT contraction using FactorizedTensor factors.

    This unified function handles QTT weights with any combination of quantized dimensions:
    - Spatial only (quantize_last_ndims=3)
    - Full (all dimensions including Cin/Cout, quantize_last_ndims=5)
    - Any partial quantization in between

    It automatically detects which dimensions are quantized and folds/unfolds accordingly.

    Args:
        x: Input tensor of shape (Batch, Cin, Nx, Ny, Nz)
        qtt_weight: QTTWeight object with _meta attribute
        separable: Whether to use separable convolution (not yet implemented for full QTT)
        verbose: Print debug information
        return_timing: Return detailed timing breakdown

    Returns:
        result: Output tensor of shape (Batch, Cout, Nx, Ny, Nz)
        timings (optional): Dictionary of timing information
    """
    timings = {}

    # === STEP 1: Get QTT metadata for folding ===
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()

    if hasattr(qtt_weight, '_meta'):
        meta = qtt_weight._meta
        ms = meta.ms          # bits per quantized axis
        base = meta.base      # usually 2
        quantize_axes = meta.quantize_axes  # which axes were quantized
        weight_shape = meta.orig_shape
    else:
        # Fallback: assume only spatial dims are quantized
        weight_shape = None
        spatial_shape = x.shape[2:]
        ms = [int(math.log2(d)) if d > 1 else 0 for d in spatial_shape]
        base = 2
        quantize_axes = tuple(range(2, 2 + len(spatial_shape)))  # spatial axes only

    # Save original shapes
    batch_size = x.shape[0]
    orig_cin = x.shape[1]
    spatial_shape = x.shape[2:]

    # Check if Cin is quantized (axis 0 in weight_shape)
    cin_quantized = 0 in quantize_axes
    # Check if Cout is quantized (axis 1 in weight_shape)
    cout_quantized = 1 in quantize_axes

    if x.is_cuda:
        torch.cuda.synchronize()
    timings['metadata_ms'] = (time.perf_counter() - start) * 1000

    # === STEP 2: Fold input tensor x ===
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()

    folded_x_shape = [batch_size]

    # Fold Cin if quantized
    if cin_quantized:
        cin_bits = ms[list(quantize_axes).index(0)]
        folded_x_shape.extend([base] * cin_bits)
    else:
        folded_x_shape.append(orig_cin)

    # Fold spatial dimensions if quantized
    for i, (dim_size, ax_idx) in enumerate(zip(spatial_shape, range(2, len(weight_shape) if weight_shape else 2 + len(spatial_shape)))):
        if ax_idx in quantize_axes:
            num_bits = ms[list(quantize_axes).index(ax_idx)]
            folded_x_shape.extend([base] * num_bits)
        else:
            folded_x_shape.append(dim_size)

    x_folded = x.reshape(folded_x_shape)

    if x.is_cuda:
        torch.cuda.synchronize()
    timings['fold_ms'] = (time.perf_counter() - start) * 1000

    # === STEP 3: Get TT cores ===
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()

    if hasattr(qtt_weight, '_get_tt_cores'):
        cores = qtt_weight._get_tt_cores()
    elif hasattr(qtt_weight, 'factors'):
        cores = list(qtt_weight.factors)
    else:
        raise ValueError("Cannot extract TT cores from qtt_weight")

    if x.is_cuda:
        torch.cuda.synchronize()
    timings['get_cores_ms'] = (time.perf_counter() - start) * 1000

    # === STEP 4: Build einsum equation ===
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()

    order = len(x_folded.shape)  # includes batch

    # Build einsum symbols based on quantization
    if cin_quantized or cout_quantized:
        # Full or partial channel quantization - use custom einsum building
        if separable:
            raise NotImplementedError("Separable mode not yet implemented for channel QTT folding")

        # Determine bit counts
        num_cin_bits = ms[list(quantize_axes).index(0)] if cin_quantized else 0
        num_cout_bits = ms[list(quantize_axes).index(1)] if cout_quantized else 0
        num_spatial_bits = sum(ms[list(quantize_axes).index(ax)] for ax in quantize_axes if ax >= 2)

        # Build symbols
        x_syms = list(einsum_symbols[:order])
        batch_sym = x_syms[0]

        # Input: batch, cin_bits (or cin), spatial_bits
        if cin_quantized:
            cin_bit_syms = x_syms[1:1+num_cin_bits]
            spatial_start = 1 + num_cin_bits
        else:
            cin_bit_syms = [x_syms[1]]  # single Cin dimension
            spatial_start = 2

        spatial_bit_syms = x_syms[spatial_start:]

        # Output: batch, cout_bits (or cout), spatial_bits
        if cout_quantized:
            cout_bit_syms = [einsum_symbols[order + i] for i in range(num_cout_bits)]
        else:
            cout_bit_syms = [einsum_symbols[order]]  # single Cout dimension

        out_syms = [batch_sym] + cout_bit_syms + spatial_bit_syms

        # Weight cores: cin_bits (or cin), cout_bits (or cout), spatial_bits
        rank_syms = list(einsum_symbols[order + len(cout_bit_syms):])
        weight_syms = cin_bit_syms + cout_bit_syms + spatial_bit_syms
        tt_syms = []
        for i, s in enumerate(weight_syms):
            tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    else:
        # Only spatial quantization - use original einsum building
        x_syms = list(einsum_symbols[:order])
        weight_syms = list(x_syms[1:])
        if not separable:
            weight_syms.insert(1, einsum_symbols[order])
            out_syms = list(weight_syms)
            out_syms[0] = x_syms[0]
        else:
            out_syms = list(x_syms)
        rank_syms = list(einsum_symbols[order + 1:])
        tt_syms = []
        for i, s in enumerate(weight_syms):
            tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])

    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.is_cuda:
        torch.cuda.synchronize()
    timings['setup_ms'] = (time.perf_counter() - start) * 1000

    if verbose:
        print(f"QTT equation: {eq}")
        print(f"x_folded shape: {x_folded.shape}")
        print(f"Core shapes: {[tuple(c.shape) for c in cores]}")
        if cin_quantized or cout_quantized:
            print(f"Cin quantized: {cin_quantized}, Cout quantized: {cout_quantized}")

    # === STEP 5: Execute einsum contraction ===
    if x_folded.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result_folded = tl.einsum(eq, x_folded, *cores)
    if x_folded.is_cuda:
        torch.cuda.synchronize()
    timings['einsum_ms'] = (time.perf_counter() - start) * 1000
    if verbose:
        print(f"Einsum contraction time: {timings['einsum_ms']:.3f} ms")

    # === STEP 6: Reshape result back to original dimensions ===
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()

    # Determine output Cout
    if weight_shape:
        orig_cout = weight_shape[1]
    else:
        orig_cout = result_folded.shape[1] if not separable else orig_cin

    unfolded_shape = [batch_size, orig_cout] + list(spatial_shape)
    result = result_folded.reshape(unfolded_shape)

    if x.is_cuda:
        torch.cuda.synchronize()
    timings['unfold_ms'] = (time.perf_counter() - start) * 1000

    if return_timing:
        return result, timings
    return result


# Test function to verify the QTT contraction
def test_qtt_contraction():
    """Test QTT contraction with a simple example."""
    import sys
    sys.path.insert(0, '/global/u2/p/pepi/STFNO_QTT')
    from stfno.qtt import QTTWeight

    # Create a simple QTT weight for testing
    # Weight shape: (Cin=4, Cout=4, Nx=8, Ny=8, Nz=8)
    # 8 = 2^3, so each spatial dim contributes 3 bits
    weight_shape = (16, 16, 16, 16, 16)  
    # (Cin=4=2^2, Cout=4=2^2, Nx=8=2^3, Ny=8=2^3, Nz=8=2^3)

    qtt_weight = QTTWeight(
        weight_shape=weight_shape,
        quantize_last_ndims=3,  # quantize spatial dims
        rank=4,
        init_std=0.02,
        dtype=torch.cfloat,
    ).to(device)

    # Create input tensor
    batch_size = 1
    x = torch.randn(batch_size, 16, 16, 16, 16, dtype=torch.cfloat, device=device)
    #x dimensions: (batch, cin, nx, ny, nz )

    print("=" * 60)
    print("Testing QTT Contraction")
    print("=" * 60)
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"QTT cores: {len(qtt_weight._get_tt_cores())}")
    #print(f"TT order: {qtt_weight._tt_order}")
    #print(f"Quantized axes metadata (ms): {qtt_weight._meta.ms}")

    # Test the factorized contraction
    print("\nTesting factorized QTT contraction:")
    try:
        result = _contract_qtt_factorized(x, qtt_weight, separable=False)
        print(f"Result shape: {result.shape}")
        print("SUCCESS!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Compare with reconstructed dense contraction (for testing only)
    print("\nComparing with reconstructed dense contraction:")
    try:
        dense_weight = qtt_weight.to_tensor()
        print(f"Reconstructed dense shape: {dense_weight.shape}")

        # Dense contraction (use full weight, no slicing needed)
        result_dense = torch.einsum("bixyz,ioxyz->boxyz", x, dense_weight)
        print(f"Dense result shape: {result_dense.shape}")

        # Compare results
        if result.shape == result_dense.shape:
            diff = (result - result_dense).abs().max().item()
            print(f"Max difference: {diff}")
            if diff < 1e-5:
                print("Results match! SUCCESS!")
            else:
                print("Results differ - may need debugging")
        else:
            print(f"Shape mismatch: {result.shape} vs {result_dense.shape}")
    except Exception as e:
        print(f"ERROR in dense comparison: {e}")
        import traceback
        traceback.print_exc()



def compare_tt_qtt_timing(
    weight_shape=(32,32,32,32,32),
    batch_size=4,
    rank=4,
    n_warmup=1,
    n_runs=5,
    dtype=torch.cfloat,
):
    """Compare timing between TT and QTT contractions.

    Creates two equivalent factorized tensors:
    - TT: standard tensor-train over (Cin, Cout, Nx, Ny, Nz)
    - QTT: quantized tensor-train with spatial dims folded into bits

    Both represent the same weight shape but with different factorizations.

    Args:
        weight_shape: (Cin, Cout, Nx, Ny, Nz) - must be powers of 2 for QTT
        batch_size: batch dimension for input x
        rank: TT rank for both factorizations
        n_warmup: number of warmup runs (not timed)
        n_runs: number of timed runs
        dtype: torch dtype
    """
    import sys
    sys.path.insert(0, '/global/u2/p/pepi/STFNO_QTT')
    from stfno.qtt import QTTWeight
    from tltorch.factorized_tensors.core import FactorizedTensor

    cin, cout, nx, ny, nz = weight_shape

    print("=" * 70)
    print("TT vs QTT Contraction Timing Comparison")
    print("=" * 70)
    print(f"Weight shape: {weight_shape}")
    print(f"Batch size: {batch_size}")
    print(f"Rank: {rank}")
    print(f"Warmup runs: {n_warmup}, Timed runs: {n_runs}")
    print(f"Device: {device}")
    print()

    # Create input tensor
    x = torch.randn(batch_size, cin, nx, ny, nz, dtype=dtype, device=device)
    print(f"Input x shape: {x.shape}")

    # === Create TT-factorized weight ===
    print("\n--- TT Factorization ---")
    tt_weight = FactorizedTensor.new(
        weight_shape,
        rank=rank,
        factorization='tt'
    )
    # Initialize with random values
    for factor in tt_weight.factors:
        factor.data = torch.randn_like(factor) * 0.02
    # Convert to complex if needed and move to device
    if dtype == torch.cfloat or dtype == torch.cdouble:
        tt_weight = tt_weight.to(dtype)
    tt_weight = tt_weight.to(device)

    print(f"TT cores: {len(tt_weight.factors)}")
    print(f"TT core shapes: {[tuple(f.shape) for f in tt_weight.factors]}")
    tt_params = sum(f.numel() for f in tt_weight.factors)
    print(f"TT total parameters: {tt_params:,}")

    # === Create QTT-factorized weight ===
    print("\n--- QTT Factorization ---")
    qtt_weight = QTTWeight(
        weight_shape=weight_shape,
        quantize_last_ndims=3,  # quantize spatial dims only
        rank=rank,
        init_std=0.02,
        dtype=dtype,
    ).to(device)

    qtt_cores = qtt_weight._get_tt_cores()
    print(f"QTT cores: {len(qtt_cores)}")
    print(f"QTT core shapes: {[tuple(c.shape) for c in qtt_cores]}")
    qtt_params = sum(c.numel() for c in qtt_cores)
    print(f"QTT total parameters: {qtt_params:,}")

    # === Dense weight for reference ===
    print("\n--- Dense Reference ---")
    dense_params = cin * cout * nx * ny * nz
    print(f"Dense parameters: {dense_params:,}")
    print(f"TT compression ratio: {dense_params / tt_params:.2f}x")
    print(f"QTT compression ratio: {dense_params / qtt_params:.2f}x")

    # === Warmup runs ===
    print(f"\nWarming up ({n_warmup} runs each)...")
    for _ in range(n_warmup):
        _ = _contract_tt_factorized(x, tt_weight, separable=False)
        _ = _contract_qtt_factorized(x, qtt_weight, separable=False)

    # === Timed runs: TT ===
    print(f"\nTiming TT contraction ({n_runs} runs)...")
    tt_times = []
    tt_detailed_timings = []
    for _ in range(n_runs):
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result_tt, tt_timings = _contract_tt_factorized(x, tt_weight, separable=False, return_timing=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        tt_times.append(total_time)
        tt_detailed_timings.append(tt_timings)

    # === Timed runs: QTT ===
    print(f"Timing QTT contraction ({n_runs} runs)...")
    qtt_times = []
    qtt_detailed_timings = []
    for _ in range(n_runs):
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result_qtt, qtt_timings = _contract_qtt_factorized(x, qtt_weight, separable=False, verbose=False, return_timing=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        qtt_times.append(total_time)
        qtt_detailed_timings.append(qtt_timings)

    # === Results ===
    tt_mean = sum(tt_times) / len(tt_times) * 1000  # ms
    tt_min = min(tt_times) * 1000
    tt_max = max(tt_times) * 1000

    qtt_mean = sum(qtt_times) / len(qtt_times) * 1000  # ms
    qtt_min = min(qtt_times) * 1000
    qtt_max = max(qtt_times) * 1000

    # Compute average detailed timings
    tt_detailed_avg = {}
    for key in tt_detailed_timings[0].keys():
        tt_detailed_avg[key] = sum(t[key] for t in tt_detailed_timings) / len(tt_detailed_timings)

    qtt_detailed_avg = {}
    for key in qtt_detailed_timings[0].keys():
        qtt_detailed_avg[key] = sum(t[key] for t in qtt_detailed_timings) / len(qtt_detailed_timings)

    print("\n" + "=" * 70)
    print("TIMING RESULTS (milliseconds)")
    print("=" * 70)
    print(f"TT contraction:  mean={tt_mean:.3f}ms, min={tt_min:.3f}ms, max={tt_max:.3f}ms")
    print(f"QTT contraction: mean={qtt_mean:.3f}ms, min={qtt_min:.3f}ms, max={qtt_max:.3f}ms")
    print()
    if qtt_mean < tt_mean:
        print(f"QTT is {tt_min/qtt_min:.2f}x FASTER than TT")
    else:
        print(f"TT is {qtt_min/tt_min:.2f}x FASTER than QTT")

    print(f"\nOutput shapes: TT={result_tt.shape}, QTT={result_qtt.shape}")

    # === Detailed timing breakdown ===
    print("\n" + "=" * 70)
    print("DETAILED TIMING BREAKDOWN (average over runs, milliseconds)")
    print("=" * 70)
    print("\nTT Contraction Steps:")
    for key, value in tt_detailed_avg.items():
        print(f"  {key:20s}: {value:8.4f} ms ({value/tt_mean*100:5.2f}%)")
    print(f"  {'Total':20s}: {tt_mean:8.4f} ms")

    print("\nQTT Contraction Steps:")
    for key, value in qtt_detailed_avg.items():
        print(f"  {key:20s}: {value:8.4f} ms ({value/qtt_mean*100:5.2f}%)")
    print(f"  {'Total':20s}: {qtt_mean:8.4f} ms")

    # Show which steps contribute most to the difference
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    if 'einsum_ms' in tt_detailed_avg and 'einsum_ms' in qtt_detailed_avg:
        einsum_speedup = tt_detailed_avg['einsum_ms'] / qtt_detailed_avg['einsum_ms']
        if einsum_speedup > 1:
            print(f"- Einsum: QTT is {einsum_speedup:.2f}x faster than TT")
        else:
            print(f"- Einsum: TT is {1/einsum_speedup:.2f}x faster than QTT")

    if 'fold_ms' in qtt_detailed_avg and 'unfold_ms' in qtt_detailed_avg:
        fold_overhead = qtt_detailed_avg['fold_ms'] + qtt_detailed_avg['unfold_ms']
        print(f"- QTT folding/unfolding overhead: {fold_overhead:.4f} ms ({fold_overhead/qtt_mean*100:.2f}% of total)")

    if 'setup_ms' in tt_detailed_avg and 'setup_ms' in qtt_detailed_avg:
        setup_diff = qtt_detailed_avg['setup_ms'] - tt_detailed_avg['setup_ms']
        print(f"- Setup time difference (QTT - TT): {setup_diff:.4f} ms")

    return {
        'tt_times': tt_times,
        'qtt_times': qtt_times,
        'tt_params': tt_params,
        'qtt_params': qtt_params,
        'dense_params': dense_params,
        'tt_detailed_timings': tt_detailed_timings,
        'qtt_detailed_timings': qtt_detailed_timings,
        'tt_detailed_avg': tt_detailed_avg,
        'qtt_detailed_avg': qtt_detailed_avg,
    }





def compare_tt_qtt_full_timing(
    weight_shape=(32,32,32,32,32),
    batch_size=4,
    rank=4,
    n_warmup=1,
    n_runs=5,
    dtype=torch.cfloat,
):
    """Compare timing between TT, QTT (spatial only), and QTT (full folding).

    Creates three equivalent factorized tensors:
    - TT: standard tensor-train over (Cin, Cout, Nx, Ny, Nz)
    - QTT Spatial: quantized tensor-train with only spatial dims folded
    - QTT Full: quantized tensor-train with ALL dims folded (Cin, Cout, spatial)

    Args:
        weight_shape: (Cin, Cout, Nx, Ny, Nz) - must be powers of 2 for QTT
        batch_size: batch dimension for input x
        rank: TT rank for all factorizations
        n_warmup: number of warmup runs (not timed)
        n_runs: number of timed runs
        dtype: torch dtype
    """
    import sys
    sys.path.insert(0, '/global/u2/p/pepi/STFNO_QTT')
    from stfno.qtt import QTTWeight
    from tltorch.factorized_tensors.core import FactorizedTensor

    cin, cout, nx, ny, nz = weight_shape

    print("=" * 70)
    print("TT vs QTT (Spatial) vs QTT (Full) Timing Comparison")
    print("=" * 70)
    print(f"Weight shape: {weight_shape}")
    print(f"Batch size: {batch_size}")
    print(f"Rank: {rank}")
    print(f"Warmup runs: {n_warmup}, Timed runs: {n_runs}")
    print(f"Device: {device}")
    print()

    # Create input tensor
    x = torch.randn(batch_size, cin, nx, ny, nz, dtype=dtype, device=device)
    print(f"Input x shape: {x.shape}")

    # === Create TT-factorized weight ===
    print("\n--- TT Factorization ---")
    tt_weight = FactorizedTensor.new(
        weight_shape,
        rank=rank,
        factorization='tt'
    )
    for factor in tt_weight.factors:
        factor.data = torch.randn_like(factor) * 0.02
    if dtype == torch.cfloat or dtype == torch.cdouble:
        tt_weight = tt_weight.to(dtype)
    tt_weight = tt_weight.to(device)

    print(f"TT cores: {len(tt_weight.factors)}")
    print(f"TT core shapes: {[tuple(f.shape) for f in tt_weight.factors]}")
    tt_params = sum(f.numel() for f in tt_weight.factors)
    print(f"TT total parameters: {tt_params:,}")

    # === Create QTT-spatial weight ===
    print("\n--- QTT Spatial Factorization ---")
    qtt_spatial_weight = QTTWeight(
        weight_shape=weight_shape,
        quantize_last_ndims=3,  # quantize spatial dims only
        rank=rank,
        init_std=0.02,
        dtype=dtype,
    ).to(device)

    qtt_spatial_cores = qtt_spatial_weight._get_tt_cores()
    print(f"QTT Spatial cores: {len(qtt_spatial_cores)}")
    print(f"QTT Spatial core shapes: {[tuple(c.shape) for c in qtt_spatial_cores]}")
    qtt_spatial_params = sum(c.numel() for c in qtt_spatial_cores)
    print(f"QTT Spatial total parameters: {qtt_spatial_params:,}")

    # === Create QTT-full weight (ALL dimensions quantized) ===
    print("\n--- QTT Full Factorization (ALL dimensions) ---")
    qtt_full_weight = QTTWeight(
        weight_shape=weight_shape,
        quantize_last_ndims=5,  # quantize ALL dims including Cin and Cout
        rank=rank,
        init_std=0.02,
        dtype=dtype,
    ).to(device)

    qtt_full_cores = qtt_full_weight._get_tt_cores()
    print(f"QTT Full cores: {len(qtt_full_cores)}")
    print(f"QTT Full core shapes: {[tuple(c.shape) for c in qtt_full_cores]}")
    qtt_full_params = sum(c.numel() for c in qtt_full_cores)
    print(f"QTT Full total parameters: {qtt_full_params:,}")

    # === Dense weight for reference ===
    print("\n--- Dense Reference ---")
    dense_params = cin * cout * nx * ny * nz
    print(f"Dense parameters: {dense_params:,}")
    print(f"TT compression ratio: {dense_params / tt_params:.2f}x")
    print(f"QTT Spatial compression ratio: {dense_params / qtt_spatial_params:.2f}x")
    print(f"QTT Full compression ratio: {dense_params / qtt_full_params:.2f}x")

    # === Warmup runs ===
    print(f"\nWarming up ({n_warmup} runs each)...")
    for _ in range(n_warmup):
        _ = _contract_tt_factorized(x, tt_weight, separable=False)
        _ = _contract_qtt_factorized(x, qtt_spatial_weight, separable=False, verbose=False)
        _ = _contract_qtt_factorized(x, qtt_full_weight, separable=False, verbose=False)

    # === Timed runs: TT ===
    print(f"\nTiming TT contraction ({n_runs} runs)...")
    tt_times = []
    tt_detailed_timings = []
    for _ in range(n_runs):
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result_tt, tt_timings = _contract_tt_factorized(x, tt_weight, separable=False, return_timing=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        tt_times.append(total_time)
        tt_detailed_timings.append(tt_timings)

    # === Timed runs: QTT Spatial ===
    print(f"Timing QTT Spatial contraction ({n_runs} runs)...")
    qtt_spatial_times = []
    qtt_spatial_detailed_timings = []
    for _ in range(n_runs):
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result_qtt_spatial, qtt_spatial_timings = _contract_qtt_factorized(x, qtt_spatial_weight, separable=False, verbose=False, return_timing=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        qtt_spatial_times.append(total_time)
        qtt_spatial_detailed_timings.append(qtt_spatial_timings)

    # === Timed runs: QTT Full ===
    print(f"Timing QTT Full contraction ({n_runs} runs)...")
    qtt_full_times = []
    qtt_full_detailed_timings = []
    for _ in range(n_runs):
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result_qtt_full, qtt_full_timings = _contract_qtt_factorized(x, qtt_full_weight, separable=False, verbose=False, return_timing=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        qtt_full_times.append(total_time)
        qtt_full_detailed_timings.append(qtt_full_timings)

    # === Results ===
    tt_mean = sum(tt_times) / len(tt_times) * 1000
    tt_min = min(tt_times) * 1000
    tt_max = max(tt_times) * 1000

    qtt_spatial_mean = sum(qtt_spatial_times) / len(qtt_spatial_times) * 1000
    qtt_spatial_min = min(qtt_spatial_times) * 1000
    qtt_spatial_max = max(qtt_spatial_times) * 1000

    qtt_full_mean = sum(qtt_full_times) / len(qtt_full_times) * 1000
    qtt_full_min = min(qtt_full_times) * 1000
    qtt_full_max = max(qtt_full_times) * 1000

    # Compute average detailed timings
    tt_detailed_avg = {}
    for key in tt_detailed_timings[0].keys():
        tt_detailed_avg[key] = sum(t[key] for t in tt_detailed_timings) / len(tt_detailed_timings)

    qtt_spatial_detailed_avg = {}
    for key in qtt_spatial_detailed_timings[0].keys():
        qtt_spatial_detailed_avg[key] = sum(t[key] for t in qtt_spatial_detailed_timings) / len(qtt_spatial_detailed_timings)

    qtt_full_detailed_avg = {}
    for key in qtt_full_detailed_timings[0].keys():
        qtt_full_detailed_avg[key] = sum(t[key] for t in qtt_full_detailed_timings) / len(qtt_full_detailed_timings)

    print("\n" + "=" * 70)
    print("TIMING RESULTS (milliseconds)")
    print("=" * 70)
    print(f"TT contraction:         mean={tt_mean:.3f}ms, min={tt_min:.3f}ms, max={tt_max:.3f}ms")
    print(f"QTT Spatial:            mean={qtt_spatial_mean:.3f}ms, min={qtt_spatial_min:.3f}ms, max={qtt_spatial_max:.3f}ms")
    print(f"QTT Full:               mean={qtt_full_mean:.3f}ms, min={qtt_full_min:.3f}ms, max={qtt_full_max:.3f}ms")
    print()

    # Find fastest
    min_time = min(tt_min, qtt_spatial_min, qtt_full_min)
    if tt_min == min_time:
        print("TT is FASTEST")
    elif qtt_spatial_min == min_time:
        print("QTT Spatial is FASTEST")
    else:
        print("QTT Full is FASTEST")

    print(f"  TT vs QTT Spatial: {qtt_spatial_min/tt_min:.2f}x")
    print(f"  TT vs QTT Full:    {qtt_full_min/tt_min:.2f}x")
    print(f"  QTT Spatial vs QTT Full: {qtt_full_min/qtt_spatial_min:.2f}x")

    print(f"\nOutput shapes: TT={result_tt.shape}, QTT Spatial={result_qtt_spatial.shape}, QTT Full={result_qtt_full.shape}")

    # === Detailed timing breakdown ===
    print("\n" + "=" * 70)
    print("DETAILED TIMING BREAKDOWN (average over runs, milliseconds)")
    print("=" * 70)

    print("\nTT Contraction Steps:")
    for key, value in tt_detailed_avg.items():
        print(f"  {key:20s}: {value:8.4f} ms ({value/tt_mean*100:5.2f}%)")
    print(f"  {'Total':20s}: {tt_mean:8.4f} ms")

    print("\nQTT Spatial Contraction Steps:")
    for key, value in qtt_spatial_detailed_avg.items():
        print(f"  {key:20s}: {value:8.4f} ms ({value/qtt_spatial_mean*100:5.2f}%)")
    print(f"  {'Total':20s}: {qtt_spatial_mean:8.4f} ms")

    print("\nQTT Full Contraction Steps:")
    for key, value in qtt_full_detailed_avg.items():
        print(f"  {key:20s}: {value:8.4f} ms ({value/qtt_full_mean*100:5.2f}%)")
    print(f"  {'Total':20s}: {qtt_full_mean:8.4f} ms")

    # === Key observations ===
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print(f"- Compression: TT={dense_params/tt_params:.1f}x, QTT Spatial={dense_params/qtt_spatial_params:.1f}x, QTT Full={dense_params/qtt_full_params:.1f}x")
    print(f"- QTT Full achieves {(dense_params/qtt_full_params)/(dense_params/tt_params):.2f}x better compression than TT")
    print(f"- QTT Full achieves {(dense_params/qtt_full_params)/(dense_params/qtt_spatial_params):.2f}x better compression than QTT Spatial")

    return {
        'tt_times': tt_times,
        'qtt_spatial_times': qtt_spatial_times,
        'qtt_full_times': qtt_full_times,
        'tt_params': tt_params,
        'qtt_spatial_params': qtt_spatial_params,
        'qtt_full_params': qtt_full_params,
        'dense_params': dense_params,
        'tt_detailed_avg': tt_detailed_avg,
        'qtt_spatial_detailed_avg': qtt_spatial_detailed_avg,
        'qtt_full_detailed_avg': qtt_full_detailed_avg,
    }


if __name__ == "__main__":
    # Run timing comparison instead of (or in addition to) the basic test
    #test_qtt_contraction()

    # Compare TT vs QTT with spatial folding only
    #compare_tt_qtt_timing()

    # Compare TT vs QTT Spatial vs QTT Full (all dimensions folded)
    compare_tt_qtt_full_timing()
