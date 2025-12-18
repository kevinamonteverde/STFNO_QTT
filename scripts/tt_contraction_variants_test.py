import time
import torch
import opt_einsum as oe
import tensorly as tl
from tensorly.decomposition import tensor_train
from tltorch.factorized_tensors.core import FactorizedTensor
import contraction_test as ct

tl.set_backend('pytorch')


#TT Original is Navjot's function
#Incremental analog is the stepwise contraction


# Benchmark configuration
NUM_RUNS = 30
WARMUP_RUNS = 3


TEST_CONFIGS = [
    # (cin, cout, nx, ny, nz, tt_rank)
    (16, 16, 16, 16, 16, 4),
    (32, 32, 32, 32, 32, 4)
]

def dense_einsum(W, x):
    return oe.contract('abijk,aijk->bijk', W, x, optimize='optimal', backend='torch')

def dense_batched(W, x):
    Cin, Cout, Nx, Ny, Nz = W.shape
    N = Nx * Ny * Nz 
    Wm = W.reshape(Cin, Cout, N)
    xm = x.reshape(Cin, N)
    out = oe.contract('abd,ad->bd', Wm, xm, optimize='optimal', backend='torch')
    return out.reshape(Cout, Nx, Ny, Nz)

def prepare_tt_cores(W_cores):

    # Squeeze first and last cores to eliminate dim 1 

    cores_contract = []
    for idx, core in enumerate(W_cores):
        if idx == 0:
            cores_contract.append(core.squeeze(0))
        elif idx == len(W_cores) - 1:
            cores_contract.append(core.squeeze(-1))
        else:
            cores_contract.append(core)
    return cores_contract

def tt_contraction_original(W_cores, W_shape, x):
    """Invoke contraction_test.tt_contraction using a shallow copy of cores."""
    cores = list(W_cores)
    return ct.tt_contraction(cores, W_shape, x)

def tt_incremental_contraction(cores, x):
    """Incremental path analogous to qtt.py matmul incremental branch.

    Logic:
      1. Contract input with first core (Cin,r1) -> (Nx,Ny,Nz,r1) after broadcasting.
      2. Sequentially contract each spatial core reducing ranks.
      3. Contract with (rCout) core to produce Cout output.

    Since we do not fold to bits (no one-hot selectors), each spatial TT core is used directly.
    This mirrors the stepwise torch.einsum loop in qtt.py when fast einsum not taken.
    Returns output (Cout,Nx,Ny,Nz)."""
    # Expect cores ordering after squeezing: (Cin,r1),(r1,Cout,r2),(r2,Nx,r3),(r3,Ny,r4),(r4,Nz)
    c_in = cores[0]        # (Cin, r1)
    c_out = cores[1]       # (r1, Cout, r2)
    c_x = cores[2]
    c_y = cores[3]
    c_z = cores[4]
    # x: (Cin,Nx,Ny,Nz)
    # Step 1: contract Cin
    tmp = torch.einsum('aijk,ap->pijk', x, c_in)         # (r1,Nx,Ny,Nz)
    # Step 2: bring Cout early similar to default path in qtt.py (combine output core before spatial loops)
    tmp = torch.einsum('pijk,pbq->bqijk', tmp, c_out)    # (Cout,r2,Nx,Ny,Nz)
    # Step 3: spatial contractions
    tmp = torch.einsum('bqijk,qir->brijk', tmp, c_x)     # (Cout,r3,Ny,Nz)
    tmp = torch.einsum('brijk,rjs->bsijk', tmp, c_y)     # (Cout,r4,Nz)
    tmp = torch.einsum('bsijk,sk->bijk', tmp, c_z)       # (Cout,Nx,Ny,Nz)
    return tmp

def tt_factorized_tltorch(weight_shape, rank, dtype, device):
    ft = FactorizedTensor.new(weight_shape, rank=rank, factorization='TT', dtype=dtype, device=device)
    # Uniform init for stability
    try:
        ft.uniform_(-0.02, 0.02)
    except Exception:
        pass
    return ft


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_many(fn, runs=NUM_RUNS, warmup=WARMUP_RUNS):
    """Run callable multiple times; return mean/median duration in ms."""
    for _ in range(warmup):
        _sync_cuda(); fn(); _sync_cuda()
    times = []
    for _ in range(runs):
        _sync_cuda(); t0 = time.perf_counter(); fn(); _sync_cuda()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(sum(times) / len(times)), float(torch.median(torch.tensor(times)).item())

def run_config(cin, cout, nx, ny, nz, rank, device):
    print('\n' + '#' * 70)
    print('TT BENCH CONFIG:')
    print(f'  W shape: ({cin},{cout},{nx},{ny},{nz})  rank={rank}')
    print('#' * 70)
    torch.manual_seed(0)
    W, x = ct.create_tensors(cin=cin, cout=cout, nx=nx, ny=ny, nz=nz, device=device)
    # Baseline dense contraction (match other script)
    out_dense = ct.regular_contraction(W, x)
    dense_mean, dense_med = benchmark_many(lambda: ct.regular_contraction(W, x))
    print(f'Average Dense einsum time: {dense_mean:.3f} ms')
    print(f'Median Dense einsum time: {dense_med:.3f} ms')

    
    """
    avg_time = 0
    for i in range(5):
        # 2. Tensorly TT decomposition (identical ordering to contraction_test)
        t0 = time.time(); W_tt_cores, W_shape = ct.to_tt(W, max_rank=rank); t_tl_decomp = (time.time() - t0) * 1e3
        avg_time += t_tl_decomp
        cores_for_measure = [core.clone() for core in W_tt_cores]
        tl_cores = prepare_tt_cores([core.clone() for core in W_tt_cores])
        #print(f'Tensorly TT decompose: {t_tl_decomp:.3f} ms  cores={[tuple(c.shape) for c in tl_cores]}')
    avg_time /= 5
    #print(f'Average Tensorly TT decomposition time: {avg_time:.3f} ms')
    """
    """
    avg_time = 0
    for i in range(5):
        # 3. Dense batched contraction (performed after decomposition in original script)
        t0 = time.time(); out_dense_b = ct.regular_contraction2(W, x); t_dense_b = (time.time() - t0) * 1e3
        avg_time += t_dense_b
        #print(f'Dense batched: {t_dense_b:.3f} ms  match={bool(torch.allclose(out_dense, out_dense_b, atol=1e-5))}')
    avg_time /= 5
    print(f'Average Dense batched contraction time: {avg_time:.3f} ms')
    """
    # TT cores once per config; clone per run because contraction squeezes in-place
    W_tt_cores, W_shape = ct.to_tt(W, max_rank=rank)
    tt_mean, tt_med = benchmark_many(lambda: tt_contraction_original([c.clone() for c in W_tt_cores], W_shape, x))
    out_tt_orig = tt_contraction_original([c.clone() for c in W_tt_cores], W_shape, x)
    rel_err_orig = (out_tt_orig - out_dense).abs().mean().item()
    speedup_mean = dense_mean / tt_mean if tt_mean > 0 else float('inf')
    speedup_med = dense_med / tt_med if tt_med > 0 else float('inf')
    print(f'Average TT contraction (original pattern) time: {tt_mean:.3f} ms')
    print(f'Median TT contraction (original pattern) time: {tt_med:.3f} ms')
    print(f'Mean absolute diff vs dense: {rel_err_orig:.3e}')
    print(f'TT speedup vs dense: {speedup_mean:.2f}x mean, {speedup_med:.2f}x median')
    """
    avg_time = 0
    for i in range(5): 
        # 5. Incremental contraction (mirrors qtt.py non-fast path structurally)
        t0 = time.time(); out_inc = tt_incremental_contraction(tl_cores, x); t_inc = (time.time() - t0) * 1e3
        avg_time += t_inc
        rel_err_inc = (out_inc - out_dense).abs().mean().item()
        #print(f'TT contraction (incremental analog): {t_inc:.3f} ms  mean|diff|={rel_err_inc:.3e}')
    avg_time /= 5
    print(f'Average TT contraction (incremental analog) time: {avg_time:.3f} ms')
    """
    """
    avg_time = 0
    for i in range(5):
        # 6. Reconstruct dense from TT cores then contract
        # Rebuild dense weight from tensorly cores
        t0 = time.time();
        W_rec = tl.tt_to_tensor(cores_for_measure)  # tensorly utility expects unsqueezed cores
        t_recon = (time.time() - t0) * 1e3
        t0 = time.time(); out_rec = dense_einsum(W_rec, x); t_rec_contract = (time.time() - t0) * 1e3
        avg_time += (t_recon + t_rec_contract)
        print(f'TT reconstruct dense: {t_recon:.3f} ms  + contract: {t_rec_contract:.3f} ms')
        print(f'Recon match dense: {bool(torch.allclose(out_dense, out_rec, atol=1e-4))}')
    avg_time /= 5
    print(f'Average TT reconstruct + contract time: {avg_time:.3f} ms')
    avg_time_recon = 0
    avg_time_contract = 0
"""
    """
    for i in range(5):
        # 7. tltorch factorized TT path (dense reconstruction only for now)
        ft = tt_factorized_tltorch((cin, cout, nx, ny, nz), rank, torch.float32, device)
        t0 = time.time(); W_ft_dense = ft.to_tensor(); t_ft_recon = (time.time() - t0) * 1e3
        t0 = time.time(); out_ft = dense_einsum(W_ft_dense, x); t_ft_contract = (time.time() - t0) * 1e3
        avg_time_recon += t_ft_recon
        avg_time_contract += t_ft_contract
        err_ft = (out_ft - out_dense).abs().mean().item()
    avg_time_recon /= 5
    avg_time_contract /= 5

    """
    #print(f'Average tltorch TT to_tensor time: {avg_time_recon:.3f} ms')
    #print(f'Average tltorch contract time: {avg_time_contract:.3f} ms')




def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f'Device: {device}')
    for cfg in TEST_CONFIGS:
        run_config(*cfg, device=device)

    

if __name__ == '__main__':
    main()