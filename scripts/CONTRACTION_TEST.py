#!/usr/bin/env python3
"""Minimal contraction tester for QTTWeight.

Runs a few contraction variants (dense recon, operator-in-out, operator-in-bits, einsum)
and prints per-variant contraction timings and pairwise correctness checks.

This file is intentionally small and avoids complex parsing; use the flags to
disable variants if needed.
"""

import os
import time
import argparse

import torch
from stfno.qtt import QTTWeight


def choose_device(name):
	if name == 'cuda' and torch.cuda.is_available():
		return torch.device('cuda')
	return torch.device('cpu')


def dense_bmm_contraction_ms(W_dense, x_rows, indices, repeats=3):
	# W_dense: (Cout, Cin, Sx, Sy, Sz)
	Cout, Cin, Sx, Sy, Sz = map(int, W_dense.shape)
	W_flat = W_dense.reshape(Cout, Cin, Sx * Sy * Sz)
	kx, ky, kz = indices[:, 0], indices[:, 1], indices[:, 2]
	lin = kx + (ky * Sx) + (kz * Sx * Sy)
	A = W_flat.index_select(dim=2, index=lin).permute(2, 0, 1).contiguous()  # (N, Cout, Cin)
	X = x_rows.unsqueeze(-1)  # (N, Cin, 1)

	# Warmup
	_ = torch.bmm(A, X)
	times = []
	Y = None
	for _ in range(repeats):
		if A.is_cuda:
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		Y = torch.bmm(A, X)
		if A.is_cuda:
			torch.cuda.synchronize()
		times.append((time.perf_counter() - t0) * 1e3)
	return Y.squeeze(-1), float(sum(times) / len(times))


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--cin', type=int, default=16)
	p.add_argument('--cout', type=int, default=16)
	p.add_argument('--modes', nargs=3, type=int, default=[8, 8, 8])
	p.add_argument('--rank', type=int, default=16)
	p.add_argument('--n-rows', type=int, default=2048)
	p.add_argument('--repeats', type=int, default=5)
	p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
	p.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
	p.add_argument('--no-dense', dest='dense', action='store_false')
	p.add_argument('--no-in-out', dest='in_out', action='store_false')
	p.add_argument('--no-in-bits', dest='in_bits', action='store_false')
	p.add_argument('--no-einsum', dest='einsum', action='store_false')
	p.set_defaults(dense=True, in_out=True, in_bits=True, einsum=True)
	args = p.parse_args()

	device = choose_device(args.device)
	dtype = torch.float32 if args.dtype == 'float32' else torch.float64
	Cin = args.cin
	Cout = args.cout
	sx, sy, sz = map(int, args.modes)
	N = int(args.n_rows)

	print('Config:', 'device=', device, 'Cin=', Cin, 'Cout=', Cout, 'modes=', (sx, sy, sz), 'rank=', args.rank, flush=True)

	# random inputs
	torch.manual_seed(0)
	x_rows = torch.randn(N, Cin, dtype=dtype, device=device)
	indices = torch.stack([
		torch.randint(0, sx, (N,), device=device),
		torch.randint(0, sy, (N,), device=device),
		torch.randint(0, sz, (N,), device=device),
	], dim=1)

	os.environ['STFNO_QTT_TIMING'] = '1'

	results = {}

	if args.dense:
		w = QTTWeight(weight_shape=(Cin, Cout, sx, sy, sz), quantize_last_ndims=3, rank=args.rank, init_std=0.02, base=2, dtype=dtype, device=device, tt_order='in-bits-out')
		T = w.to_tensor()  # (Cin, Cout, sx, sy, sz)
		Y, ms = dense_bmm_contraction_ms(T.permute(1, 0, 2, 3, 4).contiguous(), x_rows, indices, repeats=args.repeats)
		results['dense_recon'] = (Y, ms)
		print('dense_recon ms:', ms, flush=True)

	if args.in_out:
		w = QTTWeight(weight_shape=(Cin, Cout, sx, sy, sz), quantize_last_ndims=3, rank=args.rank, init_std=0.02, base=2, dtype=dtype, device=device, tt_order='in-out-bits')
		w._use_single_einsum = False
		_ = w.matmul(x_rows, indices)
		times = []
		Y = None
		for _ in range(args.repeats):
			Y = w.matmul(x_rows, indices)
			t = w.get_last_op_timing() or {}
			times.append(float(t.get('contract_ms', t.get('op_total_ms', 0.0))))
		results['op_in_out'] = (Y, float(sum(times) / len(times)))
		print('op_in_out contract_ms avg:', results['op_in_out'][1], flush=True)

	if args.in_bits:
		w = QTTWeight(weight_shape=(Cin, Cout, sx, sy, sz), quantize_last_ndims=3, rank=args.rank, init_std=0.02, base=2, dtype=dtype, device=device, tt_order='in-bits-out')
		w._use_single_einsum = False
		_ = w.matmul(x_rows, indices)
		times = []
		Y = None
		for _ in range(args.repeats):
			Y = w.matmul(x_rows, indices)
			t = w.get_last_op_timing() or {}
			times.append(float(t.get('contract_ms', t.get('op_total_ms', 0.0))))
		results['op_in_bits'] = (Y, float(sum(times) / len(times)))
		print('op_in_bits contract_ms avg:', results['op_in_bits'][1], flush=True)

	if args.einsum:
		w = QTTWeight(weight_shape=(Cin, Cout, sx, sy, sz), quantize_last_ndims=3, rank=args.rank, init_std=0.02, base=2, dtype=dtype, device=device, tt_order='in-bits-out')
		w._use_single_einsum = True
		_ = w.matmul(x_rows, indices)
		times = []
		Y = None
		for _ in range(args.repeats):
			Y = w.matmul(x_rows, indices)
			t = w.get_last_op_timing() or {}
			times.append(float(t.get('einsum_ms', t.get('op_total_ms', 0.0))))
		results['einsum'] = (Y, float(sum(times) / len(times)))
		print('einsum ms avg:', results['einsum'][1], flush=True)

	# Compare outputs
	keys = list(results.keys())
	if len(keys) > 1:
		base_key = keys[0]
		base_out = results[base_key][0]
		for k in keys[1:]:
			out = results[k][0]
			diff = (base_out - out).abs()
			print('%s vs %s: max diff=%.3e mean diff=%.3e allclose=%s' % (base_key, k, float(diff.max()), float(diff.mean()), str(torch.allclose(base_out, out, rtol=1e-4, atol=1e-5))), flush=True)

	print('Done.', flush=True)


if __name__ == '__main__':
	main()

