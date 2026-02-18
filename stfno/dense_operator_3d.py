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

from __future__ import annotations

import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import opt_einsum

from .qtt import QTTWeight


# ---------------------------------------------------------------------------
#  Dense 8-way operator
# ---------------------------------------------------------------------------

class DenseOperator3d(nn.Module):
    """Dense 8-way real-valued operator (no FFT).

    Computes:
        y[b, o, X, Y, Z] = sum_{i,x,y,z} x[b, i, x, y, z] * W[i, x, y, z, o, X, Y, Z]

    via ``torch.einsum("bixyz,ixyzoXYZ->boXYZ", x, W)``.

    Args:
        in_channels:  Number of input channels (Cin).
        out_channels: Number of output channels (Cout).
        spatial_in:   Tuple (Mx_in, My_in, Mz_in).
        spatial_out:  Tuple (Mx_out, My_out, Mz_out).
        init_std:     Initialization std ('auto' for Xavier-style, or a float).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_in: Tuple[int, int, int],
        spatial_out: Tuple[int, int, int],
        init_std: str | float = 'auto',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_in = tuple(spatial_in)
        self.spatial_out = tuple(spatial_out)

        # Weight shape: (Cin, Mx_in, My_in, Mz_in, Cout, Mx_out, My_out, Mz_out)
        weight_shape = (
            in_channels, *self.spatial_in,
            out_channels, *self.spatial_out,
        )

        fan_in = in_channels * self.spatial_in[0] * self.spatial_in[1] * self.spatial_in[2]
        fan_out = out_channels * self.spatial_out[0] * self.spatial_out[1] * self.spatial_out[2]
        if init_std == 'auto':
            std = (2.0 / (fan_in + fan_out)) ** 0.5
        else:
            std = float(init_std)

        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.normal_(self.weight, 0.0, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Cin, Mx_in, My_in, Mz_in) real tensor.
        Returns:
            (B, Cout, Mx_out, My_out, Mz_out) real tensor.
        """
        return torch.einsum("bixyz,ixyzoXYZ->boXYZ", x, self.weight)


# ---------------------------------------------------------------------------
#  QTT contraction helper for the dense 8-way operator
# ---------------------------------------------------------------------------

def _contract_qtt_dense_operator(
    x: torch.Tensor,
    qtt_weight: QTTWeight,
) -> torch.Tensor:
    """QTT contraction for the dense 8-way operator.

    Weight logical shape: (Cin, Mx_in, My_in, Mz_in, Cout, Mx_out, My_out, Mz_out).
    Weight axes 0-3 are *contracted* with x; axes 4-7 are *free* (output).

    Args:
        x: (B, Cin, Mx_in, My_in, Mz_in) real tensor.
        qtt_weight: QTTWeight with 8-D logical shape.
    Returns:
        (B, Cout, Mx_out, My_out, Mz_out) real tensor.
    """
    meta = qtt_weight._meta
    ms = meta.ms
    base = meta.base
    weight_shape = meta.orig_shape  # 8-D
    pads = meta.pads

    # Number of weight axes that come from the input (contracted)
    N_INPUT_AXES = 4  # Cin, Mx_in, My_in, Mz_in

    batch_size = x.shape[0]
    # x axes 1..4 correspond to weight axes 0..3
    x_dims = list(x.shape[1:])  # [Cin, Mx_in, My_in, Mz_in]

    # Build quantize_info: weight_ax -> (qtt_idx, num_bits, padded_size)
    quantize_info = {}
    for qtt_idx, weight_ax in enumerate(meta.quantize_axes):
        num_bits = ms[qtt_idx]
        padded_size = base ** num_bits
        quantize_info[weight_ax] = (qtt_idx, num_bits, padded_size)

    # --- Pad input dimensions that are quantized (weight axes 0..3) ---
    for i in range(len(x_dims)):
        weight_ax = i  # weight axes 0..3 map to x dims 0..3 (after batch)
        if weight_ax in quantize_info:
            _, _, padded_size = quantize_info[weight_ax]
            if padded_size != x_dims[i]:
                new_shape = list(x.shape)
                new_shape[i + 1] = padded_size  # +1 for batch
                x_padded = x.new_zeros(new_shape)
                slices = [slice(None)] * len(new_shape)
                slices[i + 1] = slice(0, x.shape[i + 1])
                x_padded[tuple(slices)] = x
                x = x_padded
                x_dims[i] = padded_size

    # --- Fold input into binary axes ---
    folded_x_shape = [batch_size]
    for i, dim_size in enumerate(x_dims):
        weight_ax = i
        if weight_ax in quantize_info:
            _, num_bits, _ = quantize_info[weight_ax]
            folded_x_shape.extend([base] * num_bits)
        else:
            folded_x_shape.append(dim_size)

    x_folded = x.reshape(folded_x_shape)

    # --- Get TT cores ---
    cores = qtt_weight._get_tt_cores()
    if not cores:
        raise ValueError("Cannot extract TT cores from qtt_weight")

    # --- Build opt_einsum integer-subscript contraction ---
    next_id = 0
    batch_id = next_id; next_id += 1

    x_subs = [batch_id]
    weight_mode_ids = []
    out_subs = [batch_id]

    for ax_idx in range(len(weight_shape)):
        if ax_idx in quantize_info:
            _, num_bits, _ = quantize_info[ax_idx]
            for _ in range(num_bits):
                dim_id = next_id; next_id += 1
                weight_mode_ids.append(dim_id)
                if ax_idx < N_INPUT_AXES:
                    # Contracted with input
                    x_subs.append(dim_id)
                else:
                    # Free output dimension
                    out_subs.append(dim_id)
        else:
            dim_id = next_id; next_id += 1
            weight_mode_ids.append(dim_id)
            if ax_idx < N_INPUT_AXES:
                x_subs.append(dim_id)
            else:
                out_subs.append(dim_id)

    # Assign rank subscript IDs
    n_cores = len(weight_mode_ids)
    rank_ids = [next_id + i for i in range(n_cores + 1)]

    # Build interleaved opt_einsum args
    args = [x_folded, x_subs]
    for i in range(n_cores):
        core_subs = [rank_ids[i], weight_mode_ids[i], rank_ids[i + 1]]
        args.extend([cores[i], core_subs])
    args.append(out_subs)

    #TODO: if less than 52 characters, use einsum instead of opt
    result_folded = opt_einsum.contract(*args, optimize='greedy')

    # --- Unfold output ---
    # Output weight axes 4..7: Cout, Mx_out, My_out, Mz_out
    unfolded_shape = [batch_size]
    for ax_idx in range(N_INPUT_AXES, len(weight_shape)):
        if ax_idx in quantize_info:
            _, _, padded_size = quantize_info[ax_idx]
            unfolded_shape.append(padded_size)
        else:
            unfolded_shape.append(weight_shape[ax_idx])

    result = result_folded.reshape(unfolded_shape)

    # --- Unpad output dimensions ---
    slices = [slice(None)]  # batch
    needs_unpad = False
    for ax_idx in range(N_INPUT_AXES, len(weight_shape)):
        orig_size = weight_shape[ax_idx]
        if ax_idx in quantize_info:
            _, _, padded_size = quantize_info[ax_idx]
            if padded_size != orig_size:
                needs_unpad = True
                slices.append(slice(0, orig_size))
            else:
                slices.append(slice(None))
        else:
            slices.append(slice(None))

    if needs_unpad:
        result = result[tuple(slices)]

    return result


# ---------------------------------------------------------------------------
#  QTT 8-way operator
# ---------------------------------------------------------------------------

class QTTOperator3d(nn.Module):
    """QTT-factorized 8-way real-valued operator.

    Same logical operation as DenseOperator3d but with QTT-compressed weight.

    Args:
        in_channels:  Cin.
        out_channels: Cout.
        spatial_in:   (Mx_in, My_in, Mz_in).
        spatial_out:  (Mx_out, My_out, Mz_out).
        rank:         TT rank.
        quantize_last_ndims: How many trailing weight axes to quantize (0-8, default 8).
        init_std:     Initialisation scale ('auto' or float).
        base:         Quantization base (default 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_in: Tuple[int, int, int],
        spatial_out: Tuple[int, int, int],
        rank: int = 4,
        quantize_last_ndims: int = 8,
        init_std: str | float = 'auto',
        base: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_in = tuple(spatial_in)
        self.spatial_out = tuple(spatial_out)

        weight_shape = (
            in_channels, *self.spatial_in,
            out_channels, *self.spatial_out,
        )

        fan_in = in_channels * self.spatial_in[0] * self.spatial_in[1] * self.spatial_in[2]
        fan_out = out_channels * self.spatial_out[0] * self.spatial_out[1] * self.spatial_out[2]
        if init_std == 'auto':
            std = (2.0 / (fan_in + fan_out)) ** 0.5
        else:
            std = float(init_std)
        std = min(std, 0.05)

        self.qtt_weight = QTTWeight(
            weight_shape=weight_shape,
            quantize_last_ndims=quantize_last_ndims,
            rank=int(rank),
            init_std=std,
            base=base,
            dtype=torch.float32,
            tt_order='in-out-bits',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Cin, Mx_in, My_in, Mz_in) real tensor.
        Returns:
            (B, Cout, Mx_out, My_out, Mz_out) real tensor.
        """
        return _contract_qtt_dense_operator(x, self.qtt_weight)


# ---------------------------------------------------------------------------
#  Wrapper: Dense or QTT dispatcher
# ---------------------------------------------------------------------------

class DenseOrQTT3DOperator(nn.Module):
    """Wrapper that dispatches to DenseOperator3d or QTTOperator3d.

    Args:
        in_channels, out_channels: Channel dims.
        spatial_in, spatial_out:   Spatial shapes.
        factorization: 'dense' or 'qtt'.
        rank:         TT rank (QTT only).
        quantize_last_ndims: QTT param (default 8).
        init_std:     Weight init scale.
        timing:       Print forward-pass timing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_in: Tuple[int, int, int],
        spatial_out: Tuple[int, int, int],
        factorization: str = 'dense',
        rank: int = 4,
        quantize_last_ndims: int = 8,
        init_std: str | float = 'auto',
        timing: bool = False,
    ):
        super().__init__()
        self.factorization = (factorization or 'dense').lower()
        self.timing = timing
        self.last_timing: dict = {}

        if self.factorization == 'dense':
            self.op = DenseOperator3d(
                in_channels, out_channels,
                spatial_in, spatial_out,
                init_std=init_std,
            )
        elif self.factorization == 'qtt':
            self.op = QTTOperator3d(
                in_channels, out_channels,
                spatial_in, spatial_out,
                rank=rank,
                quantize_last_ndims=quantize_last_ndims,
                init_std=init_std,
            )
        else:
            raise ValueError(
                f"factorization must be 'dense' or 'qtt', got '{factorization}'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.timing:
            t0 = time.perf_counter()
        y = self.op(x)
        if self.timing:
            elapsed = (time.perf_counter() - t0) * 1e3
            self.last_timing['contract_ms'] = elapsed
            print(
                f"DenseOrQTT3DOperator [{self.factorization}]: {elapsed:.2f}ms"
            )
        return y
