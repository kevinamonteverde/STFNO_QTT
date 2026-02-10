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


from .qtt import QTTWeight


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
# Use "greedy" instead of "optimal" to avoid exponential path-finding time
# for large einsum equations (e.g., Q5 with 19+ TT cores)
use_opt_einsum("greedy")
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


def _contract_qtt_factorized(x, qtt_weight, separable=False):
    """QTT contraction using QTTWeight.

    Folds input tensor x based on QTT metadata, performs TT contraction over
    the folded representation, then unfolds the result back to original shape.

    The input tensor's spatial dimensions are padded to powers of 2 (matching
    the QTT weight's folding) before reshaping into binary axes.

    Handles both quantize_last_ndims=3 (spatial only) and quantize_last_ndims=5
    (Cin, Cout, and spatial).

    Args:
        x: Input tensor of shape (Batch, Cin, Nx, Ny, Nz)
        qtt_weight: QTTWeight object with _meta attribute
        separable: Whether to use separable convolution (not implemented for QTT)

    Returns:
        result: Output tensor of shape (Batch, Cout, Nx, Ny, Nz)
    """
    import math

    if separable:
        raise NotImplementedError("Separable mode not implemented for QTT")

    # Get QTT metadata for folding
    meta = qtt_weight._meta
    ms = meta.ms          # bits per quantized axis
    base = meta.base      # usually 2
    weight_shape = meta.orig_shape
    pads = meta.pads      # padding added per quantized axis

    # Save original shapes
    # x shape: (Batch, Cin, Nx, Ny, Nz)
    # weight shape: (Cin, Cout, Mx, My, Mz) where Mx,My,Mz are corner modes
    batch_size = x.shape[0]
    orig_cin = x.shape[1]
    spatial_shape = list(x.shape[2:])
    orig_spatial_shape = tuple(spatial_shape)  # Save for unpadding later

    # Weight shape has: [Cin=0, Cout=1, spatial=2,3,4]
    # Input x has:      [Batch=0, Cin=1, spatial=2,3,4]
    # So weight axis 0 (Cin) maps to input axis 1
    # Weight axis 1 (Cout) does NOT exist in input
    # Weight axes 2,3,4 (spatial) map to input axes 2,3,4

    # Check if Cin is quantized (axis 0 in weight)
    cin_quantized = 0 in meta.quantize_axes
    # Check if Cout is quantized (axis 1 in weight)
    cout_quantized = 1 in meta.quantize_axes

    # Build a mapping: for each quantized weight axis, get its (qtt_index, num_bits, padded_size)
    # qtt_index is the position in ms/pads arrays
    quantize_info = {}
    for qtt_idx, weight_ax in enumerate(meta.quantize_axes):
        num_bits = ms[qtt_idx]
        padded_size = base ** num_bits
        quantize_info[weight_ax] = (qtt_idx, num_bits, padded_size)

    # Pad Cin if it's quantized (weight axis 0 -> input axis 1)
    padded_cin = orig_cin
    if cin_quantized:
        _, cin_bits, padded_cin = quantize_info[0]
        if padded_cin != orig_cin:
            # Pad the channel dimension
            pad_shape = [batch_size, padded_cin] + spatial_shape
            x_padded = x.new_zeros(pad_shape)
            x_padded[:, :orig_cin, ...] = x
            x = x_padded

    # Pad spatial dimensions if they are quantized
    # Weight axes 2,3,4 correspond to input axes 2,3,4 (both are spatial)
    # But we need to compare against the WEIGHT's spatial sizes, not input's
    # The input spatial shape x.shape[2:] may differ from weight spatial shape
    # Actually, the input x is sliced from FFT to match the weight's spatial dims
    # So x.shape[2:] should equal weight_shape[2:] (the corner modes)
    for i in range(len(spatial_shape)):
        weight_ax = 2 + i  # spatial axes in weight start at index 2
        if weight_ax in quantize_info:
            _, num_bits, padded_size = quantize_info[weight_ax]
            if padded_size != spatial_shape[i]:
                # Need to pad this spatial dimension
                new_spatial_shape = list(spatial_shape)
                new_spatial_shape[i] = padded_size
                pad_shape = [batch_size, padded_cin] + new_spatial_shape
                x_padded = x.new_zeros(pad_shape)
                # Copy original data
                slices = [slice(None), slice(None)] + [slice(0, s) for s in spatial_shape]
                x_padded[tuple(slices)] = x
                x = x_padded
                spatial_shape[i] = padded_size

    # Build folded shape for input tensor
    # Input tensor: (Batch, Cin, spatial...)
    # We fold Cin if quantized, and spatial dims if quantized
    folded_x_shape = [batch_size]

    # Fold Cin if quantized
    if cin_quantized:
        _, cin_bits, _ = quantize_info[0]
        folded_x_shape.extend([base] * cin_bits)
    else:
        folded_x_shape.append(padded_cin)

    # Fold spatial dimensions
    for i, dim_size in enumerate(spatial_shape):
        weight_ax = 2 + i  # spatial axes in weight start at index 2
        if weight_ax in quantize_info:
            _, num_bits, _ = quantize_info[weight_ax]
            folded_x_shape.extend([base] * num_bits)
        else:
            folded_x_shape.append(dim_size)

    x_folded = x.reshape(folded_x_shape)

    # Get TT cores from QTTWeight
    cores = qtt_weight._get_tt_cores()
    if not cores:
        raise ValueError("Cannot extract TT cores from qtt_weight")

    # Build contraction using opt_einsum with integer subscripts.
    # This avoids the 52-character symbol limit of torch.einsum,
    # which is exceeded when quantize_last_ndims=5 with large channels.
    import opt_einsum

    next_id = 0
    batch_id = next_id; next_id += 1

    # Input subscripts: (batch, folded_cin..., folded_spatial...)
    x_subs = [batch_id]
    # Weight mode subscripts: one per TT core, matching the weight's folded order
    weight_mode_ids = []
    # Output subscripts: (batch, folded_cout..., folded_spatial...)
    out_subs = [batch_id]

    # Go through each axis in the original weight shape
    for ax_idx in range(len(weight_shape)):
        if ax_idx in quantize_info:
            _, num_bits, _ = quantize_info[ax_idx]

            if ax_idx == 0:
                # Cin bits - from input, contracted (not in output)
                for _ in range(num_bits):
                    dim_id = next_id; next_id += 1
                    x_subs.append(dim_id)
                    weight_mode_ids.append(dim_id)
            elif ax_idx == 1:
                # Cout bits - new output dimensions
                for _ in range(num_bits):
                    dim_id = next_id; next_id += 1
                    weight_mode_ids.append(dim_id)
                    out_subs.append(dim_id)
            else:
                # Spatial bits - from input and in output
                for _ in range(num_bits):
                    dim_id = next_id; next_id += 1
                    x_subs.append(dim_id)
                    weight_mode_ids.append(dim_id)
                    out_subs.append(dim_id)
        else:
            if ax_idx == 0:
                # Cin unquantized - from input, contracted
                dim_id = next_id; next_id += 1
                x_subs.append(dim_id)
                weight_mode_ids.append(dim_id)
            elif ax_idx == 1:
                # Cout unquantized - new output dimension
                dim_id = next_id; next_id += 1
                weight_mode_ids.append(dim_id)
                out_subs.append(dim_id)
            else:
                # Spatial unquantized - from input and in output
                dim_id = next_id; next_id += 1
                x_subs.append(dim_id)
                weight_mode_ids.append(dim_id)
                out_subs.append(dim_id)

    # Assign rank subscript IDs for TT cores
    n_cores = len(weight_mode_ids)
    rank_ids = [next_id + i for i in range(n_cores + 1)]

    # Build interleaved args: (tensor, subs, tensor, subs, ..., output_subs)
    args = [x_folded, x_subs]
    for i in range(n_cores):
        core_subs = [rank_ids[i], weight_mode_ids[i], rank_ids[i + 1]]
        args.extend([cores[i], core_subs])
    args.append(out_subs)

    # Execute contraction
    result_folded = opt_einsum.contract(*args, optimize='greedy')

    # Build unfolded shape for output
    orig_cout = weight_shape[1]
    unfolded_shape = [batch_size]

    # Unfold Cout if quantized
    if cout_quantized:
        _, cout_bits, padded_cout = quantize_info[1]
        unfolded_shape.append(padded_cout)
    else:
        unfolded_shape.append(orig_cout)

    unfolded_shape.extend(spatial_shape)
    result = result_folded.reshape(unfolded_shape)

    # Unpad Cout if it was quantized and padded
    if cout_quantized:
        result = result[:, :orig_cout, ...]

    # Unpad spatial dimensions if padding was applied
    if tuple(spatial_shape) != orig_spatial_shape:
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig_spatial_shape]
        result = result[tuple(slices)]

    return result


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Return contraction function.

    Supported paths:
      - dense (reconstructed)
      - TT FactorizedTensor (factorized einsum path)
      - QTT via QTTWeight (factorized or reconstructed)
    """
    if implementation == "reconstructed":
        return _contract_dense
    if implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        if isinstance(weight, QTTWeight):
            return _contract_qtt_factorized
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
                 factorization='tucker', rank=4, implementation='reconstructed',
                 separable=False, fft_norm='forward', init_std='auto', timing: bool = True,
                 quantize_last_ndims: int = 3):
        super(FactorizedSpectralConv3d, self).__init__()
        
        # Timing controls
        self.timing = bool(timing)
        self.last_timing = {
            'fft_ms': 0.0,
            'contract_ms': 0.0,
            'ifft_ms': 0.0,
            'reconstruct_ms': 0.0,
            'tt_ms': 0.0,
            'qtt_ms': 0.0,
        }
        self.quantize_last_ndims = quantize_last_ndims
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

            # TT path using tltorch FactorizedTensor (tl.einsum over TT factors)
            elif isinstance(factorization, str) and fact_lower in ('tt', 'tensor_train'):
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

            # QTT path using QTTWeight with spatial dimension quantization
            elif isinstance(factorization, str) and fact_lower == 'qtt':
                try:
                    std = float(init_std)
                except Exception:
                    std = 0.02
                if not (std > 0) or not (std == std):
                    std = 0.02
                std = min(std, 0.05)

                rank_int = int(self.rank) if isinstance(self.rank, (int, float)) else 4
                w = QTTWeight(
                    weight_shape=weight_shape,
                    quantize_last_ndims=self.quantize_last_ndims,
                    rank=rank_int,
                    init_std=std,
                    base=2,
                    dtype=torch.cfloat,
                    tt_order='in-out-bits',  # default ordering for spectral convolution
                )
                setattr(w, "_stfno_weight_index", i)

            else:
                raise ValueError("Only dense, TT, or QTT factorization paths are supported.")

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
            elif isinstance(w0, FactorizedTensor):
                self._contract_label = 'tt'
            elif isinstance(w0, QTTWeight):
                self._contract_label = 'qtt'
            else:
                self._contract_label = 'unknown'
    
    def compl_mul3d(self, input, weights):
        """Complex multiplication for 3D case"""
        # QTT contraction path
        if isinstance(weights, QTTWeight):
            t0 = time.perf_counter() if self.timing else None
            # Use reconstructed path if implementation is 'reconstructed'
            if self.implementation == 'reconstructed':
                dense_weight = weights.to_tensor()
                if t0 is not None:
                    self.last_timing['reconstruct_ms'] += float((time.perf_counter() - t0) * 1e3)
                return _contract_dense(input, dense_weight, separable=self.separable)
            else:
                # Use factorized QTT contraction
                out = _contract_qtt_factorized(input, weights, separable=self.separable)
                if t0 is not None:
                    self.last_timing['qtt_ms'] += float((time.perf_counter() - t0) * 1e3)
                return out

        # If we are in a reconstructed path (FactorizedTensor -> dense), time the materialization.
        if self._contract is _contract_dense and not torch.is_tensor(weights):
            recon_start = time.perf_counter() if self.timing else None
            dense_weight = weights.to_tensor()
            if recon_start is not None:
                self.last_timing['reconstruct_ms'] += float((time.perf_counter() - recon_start) * 1e3)
            return _contract_dense(input, dense_weight, separable=self.separable)

        # TT FactorizedTensor direct contraction timing
        if isinstance(weights, FactorizedTensor):
            name = (getattr(weights, "name", "") or "").lower()
            if name.endswith("tt"):
                t0 = time.perf_counter() if self.timing else None
                out = self._contract(input, weights, separable=self.separable)
                if t0 is not None:
                    self.last_timing['tt_ms'] += float((time.perf_counter() - t0) * 1e3)
                return out

        return self._contract(input, weights, separable=self.separable)

    def forward(self, x):
        batchsize = x.shape[0]
        use_cuda_timing = x.is_cuda and torch.cuda.is_available()

        # Reset per-forward accumulators
        self.last_timing['reconstruct_ms'] = 0.0
        self.last_timing['tt_ms'] = 0.0
        self.last_timing['qtt_ms'] = 0.0
        
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
            tt_ms = self.last_timing.get('tt_ms', 0.0)
            qtt_ms = self.last_timing.get('qtt_ms', 0.0)
            extras = []
            if recon_ms > 0:
                extras.append(f"Recon: {recon_ms:.2f}ms")
            if tt_ms > 0:
                extras.append(f"TT: {tt_ms:.2f}ms")
            if qtt_ms > 0:
                extras.append(f"QTT: {qtt_ms:.2f}ms")
            extras_str = ", ".join(extras)
            extra_display = f" | {extras_str}" if extras_str else ""
            print(
                f"Times - FFT: {fft_time:.2f}ms, Contractions: {contract_time:.2f}ms, IFFT: {ifft_time:.2f}ms | Path: {self._contract_label}{extra_display}"
            )
        return x