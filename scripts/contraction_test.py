import torch
import tensorly as tl
from tensorly.decomposition import tensor_train
import numpy as np
import time
import opt_einsum as oe  # Install with: pip install opt_einsum

# Set tensorly backend to PyTorch
tl.set_backend('pytorch')

# Timing config (match TT script)
NUM_RUNS = 30
WARMUP_RUNS = 3


def to_tt(tensor, max_rank=16):
    """
    Decompose a tensor into TT (Tensor Train) format
    
    Args:
        tensor: PyTorch tensor
        max_rank: Maximum TT rank
    
    Returns:
        cores: List of TT cores
        original_shape: Original tensor shape for reconstruction
    """
    original_shape = tensor.shape
    
    # Apply TT decomposition directly
    cores = tensor_train(tensor, rank=max_rank)
    
    return cores, original_shape


def to_qtt(tensor, max_rank=16):
    """
    Decompose a tensor into QTT (Quantized Tensor Train) format
    
    Args:
        tensor: PyTorch tensor with dimensions that are powers of 2
        max_rank: Maximum TT rank
    
    Returns:
        cores: List of TT cores
        original_shape: Original tensor shape for reconstruction
    """
    original_shape = tensor.shape
    
    # Verify all dimensions are powers of 2
    for dim in original_shape:
        assert (dim & (dim - 1)) == 0, f"Dimension {dim} is not a power of 2"
    
    # Fold to binary dimensions (2x2x...x2)
    binary_shape = []
    for dim in original_shape:
        n_bits = int(np.log2(dim))
        binary_shape.extend([2] * n_bits)
    
    # Reshape to binary format
    tensor_binary = tensor.reshape(binary_shape)
    
    # Apply TT decomposition
    cores = tensor_train(tensor_binary, rank=max_rank)
    
    return cores, original_shape




def regular_contraction(W, x):
    """
    Standard einsum contraction: W[Cin,Cout,Nx,Ny,Nz] @ x[Cin,Nx,Ny,Nz] -> [Cout,Nx,Ny,Nz]
    """
    return oe.contract('abijk,aijk->bijk', W, x,optimize='optimal',backend='torch')

def regular_contraction2(W, x):
    """
    Efficient batched matrix multiplication version
    """
    # Get original shapes
    Cin, Cout, Nx, Ny, Nz = W.shape
    
    # Reshape to matrices
    N = Nx * Ny * Nz
    W_reshaped = W.reshape(Cin, Cout, N)  # (Cin, Cout, N)
    x_reshaped = x.reshape(Cin, N)        # (Cin, N)
    
    # Method 1: Using einsum (as you wanted)
    result = oe.contract('abd,ad->bd', W_reshaped, x_reshaped,optimize='optimal',backend='torch')
    
    # Method 2: Using matrix multiplication (potentially faster)
    # result = torch.tensordot(x_reshaped, W_reshaped, dims=([0], [0]))
    
    return result.reshape(Cout, Nx, Ny, Nz)


def tt_contraction(W_cores, W_shape, x):
    """
    Contraction using TT representation of W
    
    Args:
        W_cores: TT cores of W
        W_shape: Original shape of W
        x: Input tensor
    
    Returns:
        Result of contraction
    """
    
    # Perform contraction without mutating caller cores
    cores = list(W_cores)
    cores[0] = cores[0].squeeze()
    cores[-1] = cores[-1].squeeze()
    result = oe.contract('ap,pbq,qir,rjs,sk,aijk->bijk', *cores, x, optimize='optimal', backend='torch')
    return result


def benchmark_many(fn, runs=NUM_RUNS, warmup=WARMUP_RUNS):
    """Run callable multiple times; return mean and median durations in ms."""
    # CPU only: no-op sync for interface parity
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.median(times))




def create_tensors(cin=8, cout=8, nx=16, ny=16, nz=16, device='cpu'):
    """
    Create test tensors W and x with power-of-2 dimensions
    """
    W = torch.randn(cin, cout, nx, ny, nz, device=device, dtype=torch.float32)
    x = torch.randn(cin, nx, ny, nz, device=device, dtype=torch.float32)
    return W, x


def compare_decompositions(W, tt_cores, qtt_cores):
    """
    Compare TT and QTT decompositions
    """
    print("\n" + "="*60)
    print("DECOMPOSITION COMPARISON")
    print("="*60)
    
    # Original size
    original_params = W.numel()
    print(f"Original tensor parameters: {original_params:,}")
    
    # TT compression
    tt_params = sum(c.numel() for c in tt_cores)
    tt_compression = original_params / tt_params
    print(f"\nTT decomposition:")
    print(f"  Number of cores: {len(tt_cores)}")
    print(f"  Core shapes: {[c.shape for c in tt_cores]}")
    print(f"  Total parameters: {tt_params:,}")
    print(f"  Compression ratio: {tt_compression:.2f}x")
    
    # QTT compression
    qtt_params = sum(c.numel() for c in qtt_cores)
    qtt_compression = original_params / qtt_params
    print(f"\nQTT decomposition:")
    print(f"  Number of cores: {len(qtt_cores)}")
    print(f"  Core shapes (first 5): {[c.shape for c in qtt_cores[:5]]}...")
    print(f"  Total parameters: {qtt_params:,}")
    print(f"  Compression ratio: {qtt_compression:.2f}x")
    
    print(f"\nQTT vs TT: {qtt_compression/tt_compression:.2f}x better compression")


def main():
    # Setup
    device = 'cpu'  # CPU only 
    print(f"Device: {device}\n")
    
    # Test configurations with small ranks
    test_configs = [
        # (cin, cout, nx, ny, nz, tt_rank, qtt_rank)
        (16, 16, 16, 16, 16, 4, 4),
        (32, 32, 32, 32, 32, 4, 4)
        ]
    
    for cin, cout, nx, ny, nz, tt_rank, qtt_rank in test_configs:
        print("\n" + "#"*70)
        print(f"TESTING CONFIGURATION:")
        print(f"  W: ({cin}, {cout}, {nx}, {ny}, {nz})")
        print(f"  x: ({cin}, {nx}, {ny}, {nz})")
        print(f"  TT rank: {tt_rank}, QTT rank: {qtt_rank}")
        print("#"*70)
        
        # Create tensors (seed for reproducibility)
        torch.manual_seed(0)
        W, x = create_tensors(cin, cout, nx, ny, nz, device)

        # Dense contraction (single path for comparison)
        dense_mean, dense_med = benchmark_many(lambda: regular_contraction(W, x))
        out_dense = regular_contraction(W, x)
        print("\n1. Dense contraction (einsum)...")
        print(f"   Mean: {dense_mean:.3f} ms  Median: {dense_med:.3f} ms  Output shape: {out_dense.shape}")
        
        # TT decomposition (timed for reference, not used in speedup)
        decomp_mean, decomp_med = benchmark_many(lambda: to_tt(W, max_rank=tt_rank))
        W_tt_cores, W_shape = to_tt(W, max_rank=tt_rank)
        print("\n2. TT decomposition...")
        print(f"   Mean: {decomp_mean:.3f} ms  Median: {decomp_med:.3f} ms")

        # TT contraction vs dense
        tt_mean, tt_med = benchmark_many(lambda: tt_contraction(W_tt_cores, W_shape, x))
        out_tt = tt_contraction(W_tt_cores, W_shape, x)
        rel_err = (out_tt - out_dense).abs().mean().item()
        speedup_mean = dense_mean / tt_mean if tt_mean > 0 else float('inf')
        speedup_med = dense_med / tt_med if tt_med > 0 else float('inf')
        print("\n" + "="*60)
        print("CONTRACTION PERFORMANCE")
        print("="*60)
        print(f"\nTT contraction: Mean: {tt_mean:.3f} ms  Median: {tt_med:.3f} ms  Output shape: {out_tt.shape}")
        print(f"TT speedup vs dense: {speedup_mean:.2f}x mean, {speedup_med:.2f}x median")
        print(f"Mean absolute diff vs dense: {rel_err:.3e}")
        
        
       

if __name__ == "__main__":
    
    main()