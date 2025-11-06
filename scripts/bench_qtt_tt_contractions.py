#!/usr/bin/env python3
"""
Benchmark contraction times for dense (uncompressed), QTT (reconstructed+dense),
TT (reconstructed+dense), and QTT operator core-by-core in both TT orderings.

Focus: measure the time of the actual contraction only.
- For dense baselines, we pre-gather the (N, Cout, Cin) slices and time only bmm.
- For QTT operator paths, we rely on QTTWeight's internal timing:
  - einsum_ms for single-einsum fast path
  - contract_ms for incremental core-by-core path

Outputs a CSV to Data_Logs_Tests with columns:
  cout, device, cin, modes, rank,
  dense_ms,
  qtt_reconstruct_ms, qtt_dense_ms, qtt_core_in_bits_out_ms, qtt_core_in_out_bits_ms, qtt_einsum_ms,
  tt_reconstruct_ms, tt_dense_ms

Notes (metadata) are printed before the CSV header for reproducibility.
"""
import os
import csv
import time
from typing import Tuple, List

import torch
from tltorch.factorized_tensors.core import FactorizedTensor
from stfno.qtt import QTTWeight


def repo_root_dir() -> str:
    # scripts/ -> repo root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_out_csv() -> str:
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(repo_root_dir(), 'Data_Logs_Tests')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"contraction_bench_{ts}.csv")


def choose_device() -> torch.device:
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(dev)


def sync_if_needed(t: torch.Tensor | None = None):
    try:
        if t is not None and t.is_cuda:
            torch.cuda.synchronize(t.device)
    except Exception:
        pass


def build_random_problem(cout: int, cin: int, modes: Tuple[int,int,int], N_rows: int, device: torch.device, dtype=torch.float32):
    torch.manual_seed(0)
    x_rows = torch.randn(N_rows, cin, dtype=dtype, device=device)
    sx, sy, sz = modes
    indices = torch.stack([
        torch.randint(0, sx, (N_rows,), device=device),
        torch.randint(0, sy, (N_rows,), device=device),
        torch.randint(0, sz, (N_rows,), device=device),
    ], dim=1)
    return x_rows, indices


def dense_bmm_contraction_ms(W_dense: torch.Tensor, x_rows: torch.Tensor, indices: torch.Tensor, repeats: int = 5) -> float:
    assert W_dense.dim() == 5, f"Expected (Cout,Cin,Sx,Sy,Sz), got {tuple(W_dense.shape)}"
    Cout, Cin, Sx, Sy, Sz = map(int, W_dense.shape)

    W_flat = W_dense.reshape(Cout, Cin, Sx * Sy * Sz)
    kx, ky, kz = indices[:, 0], indices[:, 1], indices[:, 2]
    lin = kx + (ky * Sx) + (kz * Sx * Sy)
    A = W_flat.index_select(dim=2, index=lin).permute(2, 0, 1).contiguous()  # (N, Cout, Cin)
    X = x_rows.unsqueeze(-1)  # (N, Cin, 1)

    # Warmup
    _ = torch.bmm(A, X)

    times = []
    for _ in range(repeats):
        sync_if_needed(A)
        t0 = time.perf_counter()
        _ = torch.bmm(A, X)
        sync_if_needed(A)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(sum(times) / len(times))


def reconstruct_dense_and_time_ms(w: QTTWeight) -> Tuple[torch.Tensor, float]:
    sync_if_needed()
    t0 = time.perf_counter()
    T = w.to_tensor()
    sync_if_needed(T)
    return T, (time.perf_counter() - t0) * 1e3


def measure_qtt_operator_contract_ms(w: QTTWeight, x_rows: torch.Tensor, indices: torch.Tensor, *, use_einsum: bool | None, repeats: int = 5) -> float:
    os.environ['STFNO_QTT_TIMING'] = '1'
    if use_einsum is not None:
        w._use_single_einsum = bool(use_einsum)
    # Warmup once
    _ = w.matmul(x_rows, indices)
    vals: List[float] = []
    for _ in range(repeats):
        _ = w.matmul(x_rows, indices)
        t = w.get_last_op_timing() or {}
        if bool(getattr(w, '_use_single_einsum', False)) and w._tt_order == 'in-bits-out':
            key = 'einsum_ms'
        else:
            key = 'contract_ms'
        vals.append(float(t.get(key, 0.0)))
    return float(sum(vals) / len(vals))


def measure_tt_reconstructed_ms(cin: int, cout: int, modes: Tuple[int,int,int], rank: int, device: torch.device, dtype=torch.float32) -> float:
    sx, sy, sz = map(int, modes)
    TT = FactorizedTensor.new((cout, cin, sx, sy, sz), rank=rank, factorization='TT', dtype=dtype, device=device)
    try:
        TT.uniform_(-0.02, 0.02)
    except Exception:
        pass
    sync_if_needed()
    t0 = time.perf_counter()
    _ = TT.to_tensor()
    sync_if_needed()
    return (time.perf_counter() - t0) * 1e3


def run_bench(
    cout_values: List[int],
    cin: int = 16,
    modes: Tuple[int,int,int] = (16,16,16),
    rank: int = 16,
    N_rows: int = 2048,
    repeats: int = 5,
    device: torch.device | None = None,
    dtype = torch.float32,
    out_csv: str | None = None,
):
    device = device or choose_device()
    sx, sy, sz = modes
    out_csv = out_csv or default_out_csv()

    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))

    meta = {
        'device': str(device), 'dtype': str(dtype), 'cin': cin,
        'modes': f"{sx}x{sy}x{sz}", 'rank': rank, 'N_rows': N_rows, 'repeats': repeats,
    }
    print('# Benchmark metadata:')
    for k, v in meta.items():
        print(f"# {k}: {v}")
    print(f"Writing results to {out_csv}")

    with open(out_csv, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            'cout','device','cin','modes','rank',
            'dense_ms',
            'qtt_reconstruct_ms','qtt_dense_ms','qtt_core_in_bits_out_ms','qtt_core_in_out_bits_ms','qtt_einsum_ms',
            'tt_reconstruct_ms','tt_dense_ms'
        ])

        for cout in cout_values:
            print(f"\n-- cout={cout} --")
            x_rows, indices = build_random_problem(cout, cin, modes, N_rows, device=device, dtype=dtype)

            W_dense = torch.randn(cout, cin, sx, sy, sz, dtype=dtype, device=device)
            dense_ms = dense_bmm_contraction_ms(W_dense, x_rows, indices, repeats=repeats)
            print(f"dense_ms={dense_ms:.3f}")

            w_qtt_in_bits_out = QTTWeight(
                weight_shape=(cin, cout, sx, sy, sz),
                quantize_last_ndims=3,
                rank=rank,
                init_std=0.02,
                base=2,
                dtype=dtype,
                device=device,
                tt_order='in-bits-out',
            )
            T_qtt, qtt_recon_ms = reconstruct_dense_and_time_ms(w_qtt_in_bits_out)
            qtt_dense_ms = dense_bmm_contraction_ms(T_qtt.permute(1,0,2,3,4).contiguous(), x_rows, indices, repeats=repeats)
            qtt_core_in_bits_out_ms = measure_qtt_operator_contract_ms(w_qtt_in_bits_out, x_rows, indices, use_einsum=False, repeats=repeats)
            qtt_einsum_ms = measure_qtt_operator_contract_ms(w_qtt_in_bits_out, x_rows, indices, use_einsum=True, repeats=repeats)
            print(f"qtt_recon_ms={qtt_recon_ms:.3f} qtt_dense_ms={qtt_dense_ms:.3f} qtt_core_in_bits_out_ms={qtt_core_in_bits_out_ms:.3f} qtt_einsum_ms={qtt_einsum_ms:.3f}")

            w_qtt_in_out_bits = QTTWeight(
                weight_shape=(cin, cout, sx, sy, sz),
                quantize_last_ndims=3,
                rank=rank,
                init_std=0.02,
                base=2,
                dtype=dtype,
                device=device,
                tt_order='in-out-bits',
            )
            qtt_core_in_out_bits_ms = measure_qtt_operator_contract_ms(w_qtt_in_out_bits, x_rows, indices, use_einsum=False, repeats=repeats)
            print(f"qtt_core_in_out_bits_ms={qtt_core_in_out_bits_ms:.3f}")

            tt_recon_ms = measure_tt_reconstructed_ms(cin, cout, modes, rank, device, dtype=dtype)
            TT = FactorizedTensor.new((cout, cin, sx, sy, sz), rank=rank, factorization='TT', dtype=dtype, device=device)
            try:
                TT.uniform_(-0.02, 0.02)
            except Exception:
                pass
            W_tt = TT.to_tensor()
            tt_dense_ms = dense_bmm_contraction_ms(W_tt, x_rows, indices, repeats=repeats)
            print(f"tt_recon_ms={tt_recon_ms:.3f} tt_dense_ms={tt_dense_ms:.3f}")

            wcsv.writerow([
                cout, str(device), cin, f"{sx}x{sy}x{sz}", rank,
                f"{dense_ms:.3f}",
                f"{qtt_recon_ms:.3f}", f"{qtt_dense_ms:.3f}", f"{qtt_core_in_bits_out_ms:.3f}", f"{qtt_core_in_out_bits_ms:.3f}", f"{qtt_einsum_ms:.3f}",
                f"{tt_recon_ms:.3f}", f"{tt_dense_ms:.3f}",
            ])

    print('Done.')


if __name__ == '__main__':
    couts = [8, 12, 16, 20]
    # couts += [64, 128, 256]
    run_bench(couts)
