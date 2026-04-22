from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn
import opt_einsum

from .qtt import QTTWeight


# ---------------------------------------------------------------------------
#  Dense 6-way operator (2D spatial, no FFT)
# ---------------------------------------------------------------------------

class DenseOperator2d(nn.Module):
    """Dense 6-way real-valued operator (no FFT).

    Computes:
        y[b, o, X, Y] = sum_{i,x,y} x[b, i, x, y] * W[i, x, y, o, X, Y]

    via ``torch.einsum("bixy,ixyoXY->boXY", x, W)``.

    At 256×256 the weight has ~256^4 × width^2 entries and is ONLY usable
    as a reference / very-small test.  Use QTTOperator2d for real runs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_in: Tuple[int, int],
        spatial_out: Tuple[int, int],
        init_std: str | float = 'auto',
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

        fan_in = in_channels * self.spatial_in[0] * self.spatial_in[1]
        fan_out = out_channels * self.spatial_out[0] * self.spatial_out[1]
        std = (2.0 / (fan_in + fan_out)) ** 0.5 if init_std == 'auto' else float(init_std)

        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.normal_(self.weight, 0.0, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, H_in, W_in)
        return torch.einsum("bixy,ixyoXY->boXY", x, self.weight)


# ---------------------------------------------------------------------------
#  QTT contraction helper for the 2D dense operator
# ---------------------------------------------------------------------------

def _contract_qtt_dense_operator_2d(
    x: torch.Tensor,
    qtt_weight: QTTWeight,
) -> torch.Tensor:
    """QTT contraction for the 6-way 2D dense operator.

    Weight logical shape: (Cin, Mx_in, My_in, Cout, Mx_out, My_out).
    Axes 0-2 are contracted with x; axes 3-5 are free (output).
    """
    meta = qtt_weight._meta
    ms = meta.ms
    base = meta.base
    weight_shape = meta.orig_shape  # 6-D
    pads = meta.pads

    N_INPUT_AXES = 3  # Cin, Mx_in, My_in

    batch_size = x.shape[0]
    x_dims = list(x.shape[1:])  # [Cin, Mx_in, My_in]

    # Build quantize_info: weight_ax -> (qtt_idx, num_bits, padded_size)
    quantize_info = {}
    for qtt_idx, weight_ax in enumerate(meta.quantize_axes):
        num_bits = ms[qtt_idx]
        padded_size = base ** num_bits
        quantize_info[weight_ax] = (qtt_idx, num_bits, padded_size)

    # Pad input dimensions that are quantized (weight axes 0..2)
    for i in range(len(x_dims)):
        weight_ax = i
        if weight_ax in quantize_info:
            _, _, padded_size = quantize_info[weight_ax]
            if padded_size != x_dims[i]:
                new_shape = list(x.shape)
                new_shape[i + 1] = padded_size
                x_padded = x.new_zeros(new_shape)
                slices = [slice(None)] * len(new_shape)
                slices[i + 1] = slice(0, x.shape[i + 1])
                x_padded[tuple(slices)] = x
                x = x_padded
                x_dims[i] = padded_size

    # Fold input into binary axes
    folded_x_shape = [batch_size]
    for i, dim_size in enumerate(x_dims):
        weight_ax = i
        if weight_ax in quantize_info:
            _, num_bits, _ = quantize_info[weight_ax]
            folded_x_shape.extend([base] * num_bits)
        else:
            folded_x_shape.append(dim_size)

    x_folded = x.reshape(folded_x_shape)

    cores = qtt_weight._get_tt_cores()
    if not cores:
        raise ValueError("Cannot extract TT cores from qtt_weight")

    # Build opt_einsum integer-subscript contraction
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
                    x_subs.append(dim_id)
                else:
                    out_subs.append(dim_id)
        else:
            dim_id = next_id; next_id += 1
            weight_mode_ids.append(dim_id)
            if ax_idx < N_INPUT_AXES:
                x_subs.append(dim_id)
            else:
                out_subs.append(dim_id)

    # Apply bit ordering permutation
    _bit_perm = getattr(qtt_weight, '_bit_perm', None)
    if _bit_perm and getattr(qtt_weight, 'bit_ordering', 'serial') != 'serial':
        _n_non_q = len(weight_shape) - len(meta.quantize_axes)
        _total_bits = len(_bit_perm)
        if _total_bits > 0 and len(weight_mode_ids) == _n_non_q + _total_bits:
            _serial_ids = list(weight_mode_ids[_n_non_q:_n_non_q + _total_bits])
            weight_mode_ids = (
                list(weight_mode_ids[:_n_non_q])
                + [_serial_ids[_bit_perm[p]] for p in range(_total_bits)]
                + list(weight_mode_ids[_n_non_q + _total_bits:])
            )

    n_cores = len(weight_mode_ids)
    rank_ids = [next_id + i for i in range(n_cores + 1)]

    args = [x_folded, x_subs]
    for i in range(n_cores):
        core_subs = [rank_ids[i], weight_mode_ids[i], rank_ids[i + 1]]
        args.extend([cores[i], core_subs])
    args.append(out_subs)

    result_folded = opt_einsum.contract(*args, optimize='greedy')

    # Unfold output (weight axes 3..5: Cout, Mx_out, My_out)
    unfolded_shape = [batch_size]
    for ax_idx in range(N_INPUT_AXES, len(weight_shape)):
        if ax_idx in quantize_info:
            _, _, padded_size = quantize_info[ax_idx]
            unfolded_shape.append(padded_size)
        else:
            unfolded_shape.append(weight_shape[ax_idx])

    result = result_folded.reshape(unfolded_shape)

    # Unpad output dimensions
    slices = [slice(None)]
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
#  QTT 6-way operator
# ---------------------------------------------------------------------------

class QTTOperator2d(nn.Module):
    """QTT-factorized 6-way real-valued 2D operator.

    Weight logical shape: (Cin, H_in, W_in, Cout, H_out, W_out).

    At H=W=256 (=2^8) with quantize_last_ndims=6 and width=32 (=2^5):
      TT chain length = 5 + 8 + 8 + 5 + 8 + 8 = 42 cores.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_in: Tuple[int, int],
        spatial_out: Tuple[int, int],
        rank: int = 4,
        quantize_last_ndims: int = 6,
        init_std: str | float = 'auto',
        base: int = 2,
        bit_ordering: str = 'serial',
        bitrev_ax_indices=None,
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

        fan_in = in_channels * self.spatial_in[0] * self.spatial_in[1]
        fan_out = out_channels * self.spatial_out[0] * self.spatial_out[1]
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
            bit_ordering=bit_ordering,
            bitrev_ax_indices=bitrev_ax_indices,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, H_in, W_in)
        return _contract_qtt_dense_operator_2d(x, self.qtt_weight)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
#  Wrapper dispatcher
# ---------------------------------------------------------------------------

class DenseOrQTT2DOperator(nn.Module):
    """Dispatches to DenseOperator2d or QTTOperator2d.

    For SWE at 256×256 always use factorization='qtt' — the dense variant
    is only practical at very small resolutions (testing/debugging).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_in: Tuple[int, int],
        spatial_out: Tuple[int, int],
        factorization: str = 'qtt',
        rank: int = 4,
        quantize_last_ndims: int = 6,
        init_std: str | float = 'auto',
        timing: bool = False,
        bit_ordering: str = 'serial',
        bitrev_ax_indices=None,
    ):
        super().__init__()
        self.factorization = (factorization or 'dense').lower()
        self.timing = timing

        if self.factorization == 'dense':
            self.op = DenseOperator2d(
                in_channels, out_channels, spatial_in, spatial_out, init_std=init_std,
            )
        elif self.factorization == 'qtt':
            self.op = QTTOperator2d(
                in_channels, out_channels, spatial_in, spatial_out,
                rank=rank,
                quantize_last_ndims=quantize_last_ndims,
                init_std=init_std,
                bit_ordering=bit_ordering,
                bitrev_ax_indices=bitrev_ax_indices,
            )
        else:
            raise ValueError(f"factorization must be 'dense' or 'qtt', got '{factorization}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.timing:
            t0 = time.perf_counter()
        y = self.op(x)
        if self.timing:
            print(f"DenseOrQTT2DOperator [{self.factorization}]: {(time.perf_counter()-t0)*1e3:.2f}ms")
        return y

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
