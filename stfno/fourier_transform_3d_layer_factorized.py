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
from .qtt import QTTWeight
from torch.cuda import Event
import time
import os
from contextlib import contextmanager

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

# Global cache for spatial indices to share across all layer instances
_GLOBAL_SPATIAL_CACHE = {}
_MAX_CACHE_SIZE = 50  # Limit cache size to prevent memory leaks

def _cleanup_cache_if_needed():
    """Clean up global cache if it gets too large."""
    global _GLOBAL_SPATIAL_CACHE
    if len(_GLOBAL_SPATIAL_CACHE) > _MAX_CACHE_SIZE:
        # Keep only the most recent entries (simple LRU approximation)
        items = list(_GLOBAL_SPATIAL_CACHE.items())
        _GLOBAL_SPATIAL_CACHE = dict(items[-_MAX_CACHE_SIZE//2:])

def clear_spatial_cache():
    """Manually clear the global spatial indices cache."""
    global _GLOBAL_SPATIAL_CACHE
    _GLOBAL_SPATIAL_CACHE.clear()

def _get_cached_spatial_indices(sx: int, sy: int, sz: int, batch_size: int, device: torch.device) -> torch.Tensor | None:
    """Get cached spatial indices for QTT operator, creating if necessary.
    
    Returns cached indices tensor of shape (batch_size * sx * sy * sz, 3) or None if caching fails.
    """
    try:
        cache_key = (sx, sy, sz, str(device))
        S = sx * sy * sz
        
        if cache_key not in _GLOBAL_SPATIAL_CACHE:
            # Clean up cache if needed before adding new entry
            _cleanup_cache_if_needed()
            
            # Create base spatial grid once
            ix = torch.arange(sx, device=device, dtype=torch.long)
            iy = torch.arange(sy, device=device, dtype=torch.long)
            iz = torch.arange(sz, device=device, dtype=torch.long)
            gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing='ij')
            base_grid = torch.stack((gx, gy, gz), dim=-1).reshape(S, 3)  # (S, 3)
            _GLOBAL_SPATIAL_CACHE[cache_key] = base_grid
        
        # Expand for current batch size
        base_grid = _GLOBAL_SPATIAL_CACHE[cache_key]
        return base_grid.repeat(batch_size, 1)  # (batch_size * S, 3)
    
    except Exception:
        # Return None to trigger fallback to original implementation
        return None


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


def _contract_qtt_operator(x: torch.Tensor, qtt_weight: QTTWeight, separable: bool = False) -> torch.Tensor:
    """Apply QTT weight as a linear operator without reconstructing dense weights.

    x: (batch, in_channels, sx, sy, sz), complex
    Returns: (batch, out_channels, sx, sy, sz)

    Strategy: reshape to rows (B·S, Cin) and invoke a batched matvec on qtt_weight.
    Falls back to dense reconstruction if the fast path is unavailable.
    """
    B, Cin, sx, sy, sz = x.shape
    S = sx * sy * sz
    # Fine-grained timing for reshape/permute around the operator-path
    timing_on = str(os.environ.get("STFNO_QTT_TIMING", "0")).strip().lower() not in ("0", "false", "no", "off", "")
    def _sync():
        try:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
        except Exception:
            pass
    # Only collect row reshape/permute timings when single-einsum fast path is eligible
    use_einsum = bool(getattr(qtt_weight, '_use_single_einsum', False)) and getattr(qtt_weight, '_tt_order', '') == 'in-bits-out'
    pre_times = {}
    if timing_on and use_einsum:
        _sync(); t0 = time.perf_counter()
    # (B, Cin, S)
    tmp1 = x.reshape(B, Cin, S)
    if timing_on and use_einsum:
        _sync(); pre_times["pre_reshape1_ms"] = (time.perf_counter() - t0) * 1e3; t0 = time.perf_counter()
    # (B, S, Cin)
    tmp2 = tmp1.permute(0, 2, 1).contiguous()
    if timing_on and use_einsum:
        _sync(); pre_times["pre_permute_ms"] = (time.perf_counter() - t0) * 1e3; t0 = time.perf_counter()
    # (B·S, Cin)
    rows = tmp2.reshape(B * S, Cin)
    if timing_on and use_einsum:
        _sync(); pre_times["pre_reshape2_ms"] = (time.perf_counter() - t0) * 1e3
    
    # Use cached spatial indices if available
    indices = _get_cached_spatial_indices(sx, sy, sz, B, x.device)
    if indices is None:
        # Fallback to original implementation if caching fails
        ix = torch.arange(sx, device=x.device, dtype=torch.long)
        iy = torch.arange(sy, device=x.device, dtype=torch.long)
        iz = torch.arange(sz, device=x.device, dtype=torch.long)
        # Meshgrid in (x,y,z) order and flatten to (S,3)
        gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing='ij')
        grid = torch.stack((gx, gy, gz), dim=-1).reshape(S, 3)  # (S,3)
        indices = grid.repeat(B, 1)  # (B·S, 3)
    # Prefer a dedicated operator apply if provided by QTTWeight
    y_rows = None
    strict_no_fallback = (os.environ.get("STFNO_QTT_NO_DENSE_FALLBACK", "0") == "1")
    if hasattr(qtt_weight, 'matmul') and callable(getattr(qtt_weight, 'matmul')):
        try:
            y_rows = qtt_weight.matmul(rows, indices=indices)
        except Exception as e:
            if strict_no_fallback:
                raise
            y_rows = None
    if y_rows is None and hasattr(qtt_weight, 'apply_linear') and callable(getattr(qtt_weight, 'apply_linear')):
        try:
            y_rows = qtt_weight.apply_linear(rows, indices=indices)
        except Exception as e:
            if strict_no_fallback:
                raise
            y_rows = None
    if y_rows is None:
        # Fallback to dense reconstruction path for correctness unless disallowed
        if strict_no_fallback:
            raise RuntimeError("QTT operator apply failed and dense fallback is disabled by STFNO_QTT_NO_DENSE_FALLBACK=1")
        return _contract_dense(x, qtt_weight, separable=separable)
    # Post reshape/permute timings
    post_times = {}
    Cout = y_rows.shape[1]
    if timing_on and use_einsum:
        _sync(); t1 = time.perf_counter()
    tmp3 = y_rows.reshape(B, S, Cout)
    if timing_on and use_einsum:
        _sync(); post_times["post_reshape1_ms"] = (time.perf_counter() - t1) * 1e3; t1 = time.perf_counter()
    tmp4 = tmp3.permute(0, 2, 1).contiguous()
    if timing_on and use_einsum:
        _sync(); post_times["post_permute_ms"] = (time.perf_counter() - t1) * 1e3; t1 = time.perf_counter()
    y = tmp4.reshape(B, Cout, sx, sy, sz)
    if timing_on and use_einsum:
        _sync(); post_times["post_reshape2_ms"] = (time.perf_counter() - t1) * 1e3

    # Attach reshape/permute timings to the last op timing dict for visibility
    if timing_on and use_einsum and hasattr(qtt_weight, "_last_op_timing"):
        try:
            # Aggregate pre/post totals too
            pre_total = sum(float(v) for v in pre_times.values()) if pre_times else 0.0
            post_total = sum(float(v) for v in post_times.values()) if post_times else 0.0
            qtt_weight._last_op_timing.update(pre_times)
            qtt_weight._last_op_timing.update(post_times)
            qtt_weight._last_op_timing["pre_rows_ms"] = pre_total
            qtt_weight._last_op_timing["post_rows_ms"] = post_total
        except Exception:
            pass
    return y


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


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Get contraction function for factorized weights"""
    if implementation == "reconstructed":
        return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        # For QTT use an operator path to avoid dense reconstruction
        elif isinstance(weight, QTTWeight):
            # Allow a debug override via env var: STFNO_QTT_OP in {auto,operator,dense}
            mode = (os.environ.get("STFNO_QTT_OP", "auto") or "auto").lower()
            if mode not in ("auto", "operator", "dense"):
                mode = "auto"
            if mode == "operator" and not separable:
                return _contract_qtt_operator
            if mode == "dense":
                return _contract_dense
            # auto: use operator for non-separable
            return _contract_qtt_operator if not separable else _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            elif weight.name.lower().endswith("tt"):
                # For TT, use dense fallback for now (modes are small)
                return _contract_dense
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
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
        
        # Initialize caching for spatial indices and bit patterns (QTT operator optimization)
        self._spatial_indices_cache = {}  # Cache by (sx, sy, sz, device)
        self._bit_patterns_cache = {}     # Cache by (spatial_size, meta_info)
        
        # Initialize std for weight initialization
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        
        # Normalize factorization without breaking 'dense' or 'qtt'
        fact_lower = (factorization or '').lower()

        # Create factorized weights for each of the 4 corners in 3D FFT
        self.weights = []

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, modes1, modes2, modes3)
        else:
            weight_shape = (in_channels, out_channels, modes1, modes2, modes3)

        # Initialize 4 factorized weight tensors for 3D FFT (similar to original implementation)
        for i in range(4):
            # Dense weights
            if factorization is None or fact_lower == 'dense':
                w = nn.Parameter(torch.zeros(weight_shape, dtype=torch.cfloat))
                nn.init.normal_(w, 0, init_std)

            # QTT branch: use QTTWeight; detect by suffix to be robust
            elif isinstance(factorization, str) and (fact_lower == 'qtt' or fact_lower.endswith("qtt")):
                # Default to the optimized 'in-bits-out' ordering when using QTT
                qtt_order = (os.environ.get("STFNO_QTT_ORDER", "in-bits-out") or "in-bits-out").lower()
                if qtt_order not in ("in-out-bits", "in-bits-out"):
                    qtt_order = "in-bits-out"
                # Allow configuring how many trailing dims to fold via env var.
                # Default is 3 (spatial-only). Use 4 to fold (out + 3 spatial), or 5 to fold (in, out, 3 spatial).
                try:
                    fold_ndims = int(os.environ.get("STFNO_QTT_FOLD_NDIMS", "3"))
                except Exception:
                    fold_ndims = 3
                # Clamp to valid range [0, order]
                try:
                    max_ndims = len(weight_shape)
                except Exception:
                    max_ndims = 3
                if fold_ndims < 0:
                    fold_ndims = 0
                if fold_ndims > max_ndims:
                    fold_ndims = max_ndims
                w = QTTWeight(
                    weight_shape,
                    rank=self.rank,  # same semantics as TT
                    quantize_last_ndims=fold_ndims,  # controlled via STFNO_QTT_FOLD_NDIMS; default 3 (spatial only)
                    base=2,
                    dtype=torch.cfloat,
                    init_std=init_std,
                    tt_order=qtt_order,
                )

            # Other factorized forms via tltorch
            else:
                # Map common names to complex variants for tltorch
                if fact_lower in ('tt', 'complextt'):
                    fact_name = 'ComplexTT'
                elif fact_lower in ('cp', 'complexcp'):
                    fact_name = 'ComplexCP'
                elif fact_lower in ('tucker', 'complextucker'):
                    fact_name = 'ComplexTucker'
                else:
                    # Assume caller passed a valid tltorch factorization name
                    fact_name = factorization
                w = FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank,
                    factorization=fact_name,
                    dtype=torch.cfloat,
                )
                # Safer initialization to avoid NaNs in tltorch core init
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
                # Sanitize factor components if present
                for attr in ("factors", "factors_list", "cores", "weights", "components"):
                    fs = getattr(w, attr, None)
                    if fs is None:
                        continue
                    if isinstance(fs, (list, tuple)):
                        for f in fs:
                            data = getattr(f, "data", None)
                            if data is not None:
                                f.data = torch.nan_to_num(f.data, nan=0.0, posinf=1e3, neginf=-1e3)
                    else:
                        data = getattr(fs, "data", None)
                        if data is not None:
                            fs.data = torch.nan_to_num(fs.data, nan=0.0, posinf=1e3, neginf=-1e3)

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
            elif isinstance(w0, QTTWeight):
                # Mirror env override in label for clarity
                mode = (os.environ.get("STFNO_QTT_OP", "auto") or "auto").lower()
                if mode == "dense":
                    self._contract_label = 'dense'
                elif mode == "operator" and not separable:
                    order = getattr(w0, '_tt_order', 'in-out-bits')
                    self._contract_label = f'qtt_op[{order}]'
                else:
                    order = getattr(w0, '_tt_order', 'in-out-bits')
                    self._contract_label = (f'qtt_op[{order}]' if not separable else 'dense')
            elif isinstance(w0, FactorizedTensor):
                nm = (getattr(w0, 'name', '') or '').lower()
                if nm.endswith('tucker'):
                    self._contract_label = 'tucker'
                elif nm.endswith('cp'):
                    self._contract_label = 'cp'
                else:
                    self._contract_label = 'dense'
            else:
                self._contract_label = 'unknown'
    
    def compl_mul3d(self, input, weights):
        """Complex multiplication for 3D case"""
        return self._contract(input, weights, separable=self.separable)
    
    def forward(self, x):
        batchsize = x.shape[0]
        use_cuda_timing = x.is_cuda and torch.cuda.is_available()
        
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
            print(f"Times - FFT: {fft_time:.2f}ms, Contractions: {contract_time:.2f}ms, IFFT: {ifft_time:.2f}ms | Path: {self._contract_label}")
        return x