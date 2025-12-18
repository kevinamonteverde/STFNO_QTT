# Sparsified Time-dependent PDEs FNO (STFNO) Copyright (c) 2025, The Regents of 
# the University of California, through Lawrence Berkeley National Laboratory 
# (subject to receipt of any required approvals from the U.S.Dept. of Energy).  
# All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor
from torch.cuda import Event
import time
import os
from contextlib import contextmanager

from .tt_ops import (
    benchmark_tt,
    regular_contraction,
    regular_contraction_batched,
    to_tt,
    tt_contraction,
)

from .tt_ops import (
    regular_contraction,
    regular_contraction_batched,
    to_tt,
    tt_contraction,
    benchmark_tt
)


def get_time(use_cuda: bool):
    """Create timers. Use CUDA events only when explicitly requested, else CPU timer.

    Note: CUDA event.record() can OOM when GPU is memory constrained; prefer CPU timers
    when tensors are on CPU or when CUDA memory is tight.
    """
    if use_cuda:
        start = Event(enable_timing=True)
        end = Event(enable_timing=True)
        start.record()
        return start, end
    else:
        return time.perf_counter(), None

def record_time(start, end):
    """Get elapsed time in milliseconds between timers returned by get_time."""
    if isinstance(start, Event):
        end.record()
        # Best-effort synchronize; if CUDA OOM occurs, fall back to CPU timer semantics
        try:
            torch.cuda.synchronize()
            return float(start.elapsed_time(end))
        except RuntimeError:
            return 0.0
    else:
        # CPU perf counter path
        return float((time.perf_counter() - start) * 1000.0)

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    """Dense contraction using torch.einsum for 3D spectral convolution"""
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    
    # Get the actual dimensions
    batch_size, in_channels = x.shape[:2]
    x_spatial = x.shape[2:]
    weight_spatial = weight.shape[2:]
    
    # Take only the portion of the weight that matches the input spatial dimensions
    weight_sliced = weight[:, :, :x_spatial[0], :x_spatial[1], :x_spatial[2]]
    
    # For 3D: x is (batch, in_channels, x, y, z), weight is (in_channels, out_channels, x, y, z)
    # Output should be (batch, out_channels, x, y, z)
    return torch.einsum("bixyz,ioxyz->boxyz", x, weight_sliced)



def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # x, y, z...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]
    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # in, out
        # x, y, z...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    # Use the reconstructed tensor approach for now to avoid einsum issues
    reconstructed_weight = tucker_weight.to_tensor()
    return _contract_dense(x, reconstructed_weight, separable=separable)


def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, z...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tt_factorized(x, tt_weight, separable=False):
    """TT contraction using FactorizedTensor factors (borrowed pattern from spectral_convolution).

    This is called contract_tt() in spectral_convolution.py, from the FNO repo.
    
    This is distinct from the single-einsum core contraction: it uses the TT factors
    stored in the FactorizedTensor and a multi-operand einsum.
    """
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
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

    return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Return contraction function.

    Supported paths:
      - dense (reconstructed)
      - TT cores via ParameterList (handled in compl_mul3d)
      - TT FactorizedTensor (factorized einsum path below)
    """
    if implementation == "reconstructed":
        return _contract_dense
    if implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        if isinstance(weight, nn.ParameterList):
            return _contract_dense
        if isinstance(weight, FactorizedTensor):
            name = (getattr(weight, "name", "") or "").lower()
            if name.endswith("tt"):
                return _contract_tt_factorized
            return _contract_dense
        raise ValueError(f"Unsupported weight type for factorized path: {weight.__class__.__name__}")
    raise ValueError(
        f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
    )


class FactorizedSpectralConv3d(nn.Module):
    """Factorized 3D Fourier layer with tensor compression"""
    
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, 
                 factorization='tucker', rank=0.5, implementation='reconstructed',
                 separable=False, fft_norm='forward', init_std='auto', timing: bool = True):
        super(FactorizedSpectralConv3d, self).__init__()
        
        # Timing controls
        self.timing = bool(timing)
        self.last_timing = {
            'fft_ms': 0.0,
            'contract_ms': 0.0,
            'ifft_ms': 0.0,
            'reconstruct_ms': 0.0,
            'tt_core_ms': 0.0,
            'tt_regular_ms': 0.0,
            'tt_batched_ms': 0.0,
            'tt_fact_ms': 0.0,
        }
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2  
        self.modes3 = modes3
        self.factorization = factorization
        self.rank = rank
        self.implementation = implementation
        self.separable = separable
        self.fft_norm = fft_norm
        
        # Initialize std for weight initialization
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        
        # Normalize factorization without breaking 'dense' or 'qtt'
        fact_lower = (factorization or '').lower()

        # Create factorized weights for each of the 4 corners in 3D FFT
        self.weights = []

        self._tt_rank_by_idx = {}
        self._tt_dense_refs = {}

        corner_m1 = max(1, modes1 // 2)
        corner_m2 = max(1, modes2 // 2)
        corner_m3 = modes3

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, corner_m1, corner_m2, corner_m3)
        else:
            weight_shape = (in_channels, out_channels, corner_m1, corner_m2, corner_m3)

        # Initialize 4 factorized weight tensors for 3D FFT (similar to original implementation)
        for i in range(4):
            # Dense weights
            if factorization is None or fact_lower == 'dense':
                w = nn.Parameter(torch.zeros(weight_shape, dtype=torch.cfloat))
                nn.init.normal_(w, 0, init_std)

            # TT path using tensorly decomposition (matches contraction_test single-einsum)
            elif isinstance(factorization, str) and fact_lower in ('tt', 'tensor_train'):
                dense_init = torch.randn(weight_shape, dtype=torch.float32) * float(init_std)
                rank_int = int(self.rank) if isinstance(self.rank, (int, float)) else 16
                cores, _ = to_tt(dense_init, max_rank=rank_int)
                core_params = nn.ParameterList([nn.Parameter(c.to(torch.cfloat)) for c in cores])
                setattr(core_params, "_stfno_weight_index", i)
                self._tt_rank_by_idx[i] = rank_int
                self._tt_dense_refs[i] = dense_init.to(torch.cfloat)
                w = core_params

            # TT FactorizedTensor path (tl.einsum over TT factors)
            elif isinstance(factorization, str) and fact_lower in ('tt_factorized', 'tt_ft', 'tt_tltorch', 'tt_alt'):
                fact_name = 'ComplexTT'
                w = FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank,
                    factorization=fact_name,
                    dtype=torch.cfloat,
                )
                try:
                    std = float(init_std)
                except Exception:
                    std = 0.02
                if not (std > 0) or not (std == std):
                    std = 0.02
                std = min(std, 0.05)
                try:
                    w.uniform_(-std, std)
                except Exception:
                    try:
                        w.normal_(0, std)
                    except Exception:
                        pass

            else:
                raise ValueError("Only dense, TT (cores), or TT FactorizedTensor paths are supported.")

            setattr(self, f'weight_{i}', w)
            self.weights.append(w)

        # Set up contraction function and label
        self._contract = get_contract_fun(
            self.weights[0], implementation=implementation, separable=separable
        )
        # Derive a label for logging which path is used
        if implementation == 'reconstructed':
            self._contract_label = 'dense'
        else:
            w0 = self.weights[0]
            if torch.is_tensor(w0):
                self._contract_label = 'dense'
            elif isinstance(w0, nn.ParameterList):
                self._contract_label = 'tt_ref'
            elif isinstance(w0, FactorizedTensor):
                self._contract_label = 'tt_fact'
            else:
                self._contract_label = 'unknown'
    
    #RECONTRUCT_MS IS TIMED HERE
    def compl_mul3d(self, input, weights):
        """Complex multiplication for 3D case"""
        if isinstance(weights, nn.ParameterList):
            idx = getattr(weights, "_stfno_weight_index", None)
            return self._contract_tt_reference(input, weights, idx)

        # If we are in a reconstructed path (FactorizedTensor -> dense), time the materialization.
        if self._contract is _contract_dense and not torch.is_tensor(weights):
            recon_start = time.perf_counter() if self.timing else None
            dense_weight = weights.to_tensor()
            if recon_start is not None:
                self.last_timing['reconstruct_ms'] += float((time.perf_counter() - recon_start) * 1e3)
            return _contract_dense(input, dense_weight, separable=self.separable)

        # FactorizedTensor TT direct contraction timing
        if isinstance(weights, FactorizedTensor):
            name = (getattr(weights, "name", "") or "").lower()
            if name.endswith("tt"):
                t0 = time.perf_counter() if self.timing else None
                out = self._contract(input, weights, separable=self.separable)
                if t0 is not None:
                    self.last_timing['tt_fact_ms'] += float((time.perf_counter() - t0) * 1e3)
                return out

        return self._contract(input, weights, separable=self.separable)

#THIS IS WHERE CONTRACTION HAPPENS TT 

    def _contract_tt_reference(self, x, tt_params: nn.ParameterList, idx: int | None):
        """Apply TT cores using the same contraction implementation as contraction_test.py."""
        cores = [core for core in tt_params]
        rank = self._tt_rank_by_idx.get(idx or 0, int(self.rank) if isinstance(self.rank, (int, float)) else 16)
        dense_weight = self._tt_dense_refs.get(idx or 0)
        batch, *_ = x.shape
        start = time.perf_counter() if self.timing else None
        core_start = time.perf_counter() if self.timing else None
        y = None

        #THE FOLLOWING BLOCK TRIES BATCHED FIRST THEN FALLS BACK TO NON BATCHED IF IT FAILS
        
        try:
            # Batched contraction (preferred)
            y = tt_contraction(cores, x)
        except Exception:
            # Fallback to per-sample to preserve correctness
            outputs: list[torch.Tensor] = []
            for b in range(batch):
                xb = x[b]
                outputs.append(tt_contraction(cores, xb))
            if outputs:
                y = torch.stack(outputs, dim=0)
        if y is None:
            spatial = x.shape[2:]
            return x.new_zeros((0, self.out_channels, *spatial))
        if self.timing and batch > 0:
            contract_ms = (time.perf_counter() - start) * 1e3 if start is not None else 0.0
            core_ms = (time.perf_counter() - core_start) * 1e3 if core_start is not None else 0.0
            self.last_timing['tt_core_ms'] += float(core_ms)
            # Always use measured contraction time for reporting; keep profiles as auxiliary
            self.last_timing['contract_ms'] = float(contract_ms)
            if dense_weight is not None:
                try:
                    profile, _ = benchmark_tt(dense_weight, x[0], rank, cores=cores)
                    self.last_timing['tt_regular_ms'] = float(profile.regular_ms)
                    self.last_timing['tt_batched_ms'] = float(profile.batched_ms)
                except Exception:
                    self.last_timing['tt_regular_ms'] = float(contract_ms)
                    self.last_timing['tt_batched_ms'] = float(contract_ms)
        return y
    
    def forward(self, x):
        batchsize = x.shape[0]
        use_cuda_timing = x.is_cuda and torch.cuda.is_available()

        # Reset per-forward accumulators
        self.last_timing['reconstruct_ms'] = 0.0
        self.last_timing['tt_core_ms'] = 0.0
        
        # Time FFT
        start, end = get_time(use_cuda_timing)
        # Ensure dtype conversion doesn't move tensors off device; keep on same device
        x_ft = torch.fft.rfftn(x.to(dtype=torch.float32, device=x.device), norm=self.fft_norm, dim=[-3, -2, -1])
        fft_time = record_time(start, end)

        out_fft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), 
                             x.size(-1)// 2 + 1, dtype=torch.cfloat, device=x.device)
        
        
        # Define slices for the 4 corners in 3D FFT
        slices0 = (
            slice(None),  # batch
            slice(None),  # channels
            slice(self.modes1 // 2),  # :modes1//2
            slice(self.modes2 // 2),  # :modes2//2
            slice(self.modes3),  # :modes3
        )
        slices1 = (
            slice(None),  # batch
            slice(None),  # channels
            slice(self.modes1 // 2),  # :modes1//2
            slice(-self.modes2 // 2, None),  # -modes2//2:
            slice(self.modes3),  # :modes3
        )
        slices2 = (
            slice(None),  # batch
            slice(None),  # channels
            slice(-self.modes1 // 2, None),  # -modes1//2:
            slice(self.modes2 // 2),  # :modes2//2
            slice(self.modes3),  # :modes3
        )
        slices3 = (
            slice(None),  # batch
            slice(None),  # channels
            slice(-self.modes1 // 2, None),  # -modes1//2:
            slice(-self.modes2 // 2, None),  # -modes2//2:
            slice(self.modes3),  # :modes3
        )
        

        #CONTRACTION IS TIMED HERE
        # Time contractions
        start, end = get_time(use_cuda_timing)
        out_fft[slices0] = self.compl_mul3d(x_ft[slices0], self.weights[0])
        out_fft[slices1] = self.compl_mul3d(x_ft[slices1], self.weights[1])
        out_fft[slices2] = self.compl_mul3d(x_ft[slices2], self.weights[2])
        out_fft[slices3] = self.compl_mul3d(x_ft[slices3], self.weights[3])
        contract_time = record_time(start, end)
        
        # Time IFFT
        start, end = get_time(use_cuda_timing)
        x = torch.fft.irfftn(out_fft, s=(x.size(-3), x.size(-2), x.size(-1)), 
                            dim=[-3, -2, -1], norm=self.fft_norm)
        ifft_time = record_time(start, end)

        # Save and optionally print timings similar to the legacy layer
        self.last_timing['fft_ms'] = float(fft_time)
        self.last_timing['contract_ms'] = float(contract_time)
        self.last_timing['ifft_ms'] = float(ifft_time)
        if self.timing:
            recon_ms = self.last_timing.get('reconstruct_ms', 0.0)
            tt_core_ms = self.last_timing.get('tt_core_ms', 0.0)
            tt_fact_ms = self.last_timing.get('tt_fact_ms', 0.0)
            extras = []
            if recon_ms > 0:
                extras.append(f"Recon: {recon_ms:.2f}ms")
            if tt_core_ms > 0:
                extras.append(f"TT-core: {tt_core_ms:.2f}ms")
            if tt_fact_ms > 0:
                extras.append(f"TT-fact: {tt_fact_ms:.2f}ms")
            if self.last_timing.get('tt_regular_ms', 0.0) > 0:
                extras.append(f"TT-regular: {self.last_timing['tt_regular_ms']:.2f}ms")
            if self.last_timing.get('tt_batched_ms', 0.0) > 0:
                extras.append(f"TT-batched: {self.last_timing['tt_batched_ms']:.2f}ms")
            extras_str = ", ".join(extras)
            extra_display = f" | {extras_str}" if extras_str else ""
            print(
                f"Times - FFT: {fft_time:.2f}ms, Contractions: {contract_time:.2f}ms, IFFT: {ifft_time:.2f}ms | Path: {self._contract_label}{extra_display}"
            )
        return x