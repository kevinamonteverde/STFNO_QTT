from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tltorch.factorized_tensors.core import FactorizedTensor


@dataclass
class QTTMeta:
    """Metadata to undo QTT folding and padding."""
    orig_shape: Tuple[int, ...]          # original tensor shape
    quantize_axes: Tuple[int, ...]       # which axes were folded (by index in orig_shape)
    ms: List[int]                        # bit-splits per quantized axis (base^m)
    pads: List[int]                      # padding added per quantized axis
    base: int = 2                        # usually 2


def _normalize_axes(axes: Sequence[int], ndim: int) -> Tuple[int, ...]:
    """Return unique, sorted, non-negative axis indices."""
    out = []
    for a in axes:
        a = a if a >= 0 else a + ndim
        if a < 0 or a >= ndim:
            raise IndexError(f"Axis {a} out of bounds for ndim={ndim}")
        out.append(a)
    return tuple(sorted(set(out)))


def _next_pow_base(n: int, base: int) -> Tuple[int, int, int]:
    """Return (q, m, pad) so that q = base**m >= n and pad = q - n."""
    if n <= 1:
        return n, 0, 0
    m = math.ceil(math.log(n, base))
    q = base**m
    return q, m, q - n


def _fold_shape_only(
    shape: Tuple[int, ...],
    quantize_axes: Sequence[int],
    base: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Compute folded shape and per-axis (ms, pads, qsizes) without touching data.
    Returns: (folded_shape, ms, pads, qsizes)
    """
    ndim = len(shape)
    axes = _normalize_axes(quantize_axes, ndim)
    padded = list(shape)
    ms, pads, qsizes = [], [], []

    for ax in axes:
        q, m, pad = _next_pow_base(shape[ax], base)
        padded[ax] = q
        ms.append(m)
        pads.append(pad)
        qsizes.append(q)

    folded: List[int] = []
    m_by_ax = {ax: m for ax, m in zip(axes, ms)}
    for i, size in enumerate(padded):
        if i in m_by_ax and m_by_ax[i] > 0:
            folded.extend([base] * m_by_ax[i])
        else:
            folded.append(size)
    return folded, ms, pads, qsizes


def _compute_bit_perm(ms: List[int], ordering: str, bitrev_ax_indices: set) -> List[int]:
    """Compute bit permutation for QTT mode ordering.

    Returns perm such that perm[k] = j means TT core k corresponds to serial bit j.

    'serial': identity (no reordering).
    'interleaved': at each bit level b, take bit b from each axis (skip axes with fewer bits).
    'bitrev_interleaved': like interleaved but axes in bitrev_ax_indices have their bits
        traversed in reversed significance (MSB first).
    'paired_spatial': 8-axis real_op — Cin | xi0,Xo0,xi1,Xo1,... | yi0,Yo0,... | zi0,Zo0,... | Cout
    'paired_all': 8-axis real_op — Cin | xi0,Xo0,yi0,Yo0,zi0,Zo0 | xi1,... | Cout
    'ch_serial': 8-axis real_op — contracted serial (Cin,xi,yi,zi) then free serial (Cout,Xo,Yo,Zo).
        Identical to 'serial' for standard real_op layout; kept for documentation.
    'ch_io_block': 8-axis real_op — contracted serial, then free in order Xo,Yo,Zo,Cout.
    'ch_paired_spatial': 8-axis real_op — same as 'serial' after free-last fix.
    'ch_paired_all': 8-axis real_op — Cin | interleaved (xi,yi,zi) per bit level | Cout,Xo,Yo,Zo.
        Only ch_* ordering that remains genuinely distinct after the free-last fix.
    'input_interleaved': 8-axis real_op — interleaved contracted axes (Cin,xi,yi,zi) per bit level,
        then free axes (Cout,Xo,Yo,Zo) in serial order at the end.
    """
    total = sum(ms)
    if total == 0 or ordering == 'serial':
        return list(range(total))

    offsets: List[int] = []
    o = 0
    for m in ms:
        offsets.append(o)
        o += m

    if ordering in ('interleaved', 'bitrev_interleaved'):
        perm: List[int] = []
        max_bits = max(ms) if ms else 0
        for b in range(max_bits):
            for ax_idx, m in enumerate(ms):
                if b < m:
                    if ordering == 'bitrev_interleaved' and ax_idx in bitrev_ax_indices:
                        bit_idx = m - 1 - b  # reversed: coarsest (MSB) first
                    else:
                        bit_idx = b
                    perm.append(offsets[ax_idx] + bit_idx)
        return perm

    if ordering == 'paired_spatial':
        # Requires 8-axis real_op tensor: [Cin=0, xi=1, yi=2, zi=3, Cout=4, Xo=5, Yo=6, Zo=7]
        if len(ms) != 8:
            return list(range(total))
        perm = []
        for b in range(ms[0]):                          # Cin bits
            perm.append(offsets[0] + b)
        for in_ax, out_ax in [(1, 5), (2, 6), (3, 7)]:  # xi/Xo, yi/Yo, zi/Zo
            for b in range(max(ms[in_ax], ms[out_ax])):
                if b < ms[in_ax]:
                    perm.append(offsets[in_ax] + b)
                if b < ms[out_ax]:
                    perm.append(offsets[out_ax] + b)
        for b in range(ms[4]):                          # Cout bits
            perm.append(offsets[4] + b)
        return perm

    if ordering == 'paired_all':
        # Requires 8-axis real_op tensor: [Cin=0, xi=1, yi=2, zi=3, Cout=4, Xo=5, Yo=6, Zo=7]
        if len(ms) != 8:
            return list(range(total))
        perm = []
        for b in range(ms[0]):                          # Cin bits
            perm.append(offsets[0] + b)
        spatial_pairs = [(1, 5), (2, 6), (3, 7)]
        max_spatial = max(ms[ax] for pair in spatial_pairs for ax in pair)
        for b in range(max_spatial):                    # xi0,Xo0,yi0,Yo0,zi0,Zo0 | xi1,...
            for in_ax, out_ax in spatial_pairs:
                if b < ms[in_ax]:
                    perm.append(offsets[in_ax] + b)
                if b < ms[out_ax]:
                    perm.append(offsets[out_ax] + b)
        for b in range(ms[4]):                          # Cout bits
            perm.append(offsets[4] + b)
        return perm

    if ordering == 'ch_serial':
        # FIXED (free-last): contracted axes Cin|xi|yi|zi serially, then free axes Cout|Xo|Yo|Zo.
        # Note: this is identical to 'serial' for an 8-axis tensor; kept for documentation.
        # Requires 8-axis real_op tensor: [Cin=0, xi=1, yi=2, zi=3, Cout=4, Xo=5, Yo=6, Zo=7]
        if len(ms) != 8:
            return list(range(total))
        perm = []
        for ax in [0, 1, 2, 3]:   # contracted: Cin, xi, yi, zi
            for b in range(ms[ax]): perm.append(offsets[ax] + b)
        for ax in [4, 5, 6, 7]:   # free: Cout, Xo, Yo, Zo
            for b in range(ms[ax]): perm.append(offsets[ax] + b)
        return perm

    if ordering == 'ch_io_block':
        # FIXED (free-last): contracted serial (Cin|xi|yi|zi), then free in io order: Xo|Yo|Zo|Cout.
        # Keeps the "input before output" spirit but only in the free section at the end.
        # Requires 8-axis real_op tensor: [Cin=0, xi=1, yi=2, zi=3, Cout=4, Xo=5, Yo=6, Zo=7]
        if len(ms) != 8:
            return list(range(total))
        perm = []
        for ax in [0, 1, 2, 3]:   # contracted: Cin, xi, yi, zi
            for b in range(ms[ax]): perm.append(offsets[ax] + b)
        for ax in [5, 6, 7, 4]:   # free: Xo, Yo, Zo, Cout (Cout last)
            for b in range(ms[ax]): perm.append(offsets[ax] + b)
        return perm

    if ordering == 'ch_paired_spatial':
        # FIXED (free-last): contracted serial, free serial = identical to 'serial'.
        # Pairing contracted spatial dims with their free counterparts makes no sense once
        # free axes are pushed to the end; the contracted part is inherently serial per-dim.
        # Requires 8-axis real_op tensor: [Cin=0, xi=1, yi=2, zi=3, Cout=4, Xo=5, Yo=6, Zo=7]
        if len(ms) != 8:
            return list(range(total))
        perm = []
        for ax in [0, 1, 2, 3, 4, 5, 6, 7]:  # = serial
            for b in range(ms[ax]): perm.append(offsets[ax] + b)
        return perm

    if ordering == 'ch_paired_all':
        # FIXED (free-last): interleave CONTRACTED spatial dims at each bit level, then free at end.
        # Structure: Cin_bits | (xi0,yi0,zi0) | (xi1,yi1,zi1) | ... | Cout_bits | Xo_bits | Yo_bits | Zo_bits
        # This is the only ch_* ordering that remains genuinely distinct after the free-last fix.
        # Requires 8-axis real_op tensor: [Cin=0, xi=1, yi=2, zi=3, Cout=4, Xo=5, Yo=6, Zo=7]
        if len(ms) != 8:
            return list(range(total))
        perm = []
        for b in range(ms[0]):   perm.append(offsets[0] + b)   # Cin
        contracted_spatial = [1, 2, 3]                          # xi, yi, zi
        max_bits = max(ms[ax] for ax in contracted_spatial)
        for b in range(max_bits):                               # interleave contracted spatial
            for ax in contracted_spatial:
                if b < ms[ax]: perm.append(offsets[ax] + b)
        for ax in [4, 5, 6, 7]:                                 # free: Cout, Xo, Yo, Zo
            for b in range(ms[ax]): perm.append(offsets[ax] + b)
        return perm

    if ordering == 'input_interleaved':
        # Free-last variant of 'interleaved': all contracted axes (Cin, xi, yi, zi) interleaved
        # at each bit level, then all free axes (Cout, Xo, Yo, Zo) in serial order at the end.
        # Structure: (Cin0,xi0,yi0,zi0) | (Cin1,xi1,yi1,zi1) | ... | Cout_bits | Xo_bits | Yo_bits | Zo_bits
        # Requires 8-axis real_op tensor: [Cin=0, xi=1, yi=2, zi=3, Cout=4, Xo=5, Yo=6, Zo=7]
        if len(ms) != 8:
            return list(range(total))
        perm = []
        contracted = [0, 1, 2, 3]                               # Cin, xi, yi, zi
        max_bits = max(ms[ax] for ax in contracted)
        for b in range(max_bits):                               # interleave all contracted
            for ax in contracted:
                if b < ms[ax]: perm.append(offsets[ax] + b)
        for ax in [4, 5, 6, 7]:                                 # free: Cout, Xo, Yo, Zo
            for b in range(ms[ax]): perm.append(offsets[ax] + b)
        return perm

    return list(range(total))


def qtt_fold_selected(
    x: torch.Tensor,
    quantize_axes: Sequence[int],
    base: int = 2,
) -> Tuple[torch.Tensor, QTTMeta]:
    """
    Zero-pad selected axes to base**m and reshape each into m axes of length `base`.
    Leaves other axes unchanged and preserves axis order.
    """
    orig_shape = tuple(x.shape)
    ndim = len(orig_shape)
    axes = _normalize_axes(quantize_axes, ndim)

    folded_shape, ms, pads, qsizes = _fold_shape_only(orig_shape, axes, base)
    # Pad only selected axes by copying into a larger zero tensor
    if any(pads):
        padded_shape = list(orig_shape)
        for ax, q in zip(axes, qsizes):
            padded_shape[ax] = q
        xp = x.new_zeros(padded_shape)
        slices = tuple(slice(0, s) for s in orig_shape)
        xp[slices] = x
    else:
        xp = x

    xq = xp.contiguous().reshape(folded_shape)
    meta = QTTMeta(orig_shape=orig_shape, quantize_axes=axes, ms=ms, pads=pads, base=base)
    return xq, meta


def qtt_unfold_selected(xq: torch.Tensor, meta: QTTMeta) -> torch.Tensor:
    """
    Inverse of qtt_fold_selected: reshape folded tensor back to padded shape,
    then remove the zero padding on quantized axes.
    """
    # Build the padded target shape
    padded_shape = list(meta.orig_shape)
    for ax, m in zip(meta.quantize_axes, meta.ms):
        if m > 0:
            padded_shape[ax] = meta.base**m
    xp = xq.contiguous().reshape(padded_shape)
    # Remove padding
    sl = tuple(slice(0, s) for s in meta.orig_shape)
    return xp[sl]


def qtt_fold(x: torch.Tensor, base: int = 2) -> Tuple[torch.Tensor, QTTMeta]:
    """Fold all axes (per-dimension quantization)."""
    return qtt_fold_selected(x, quantize_axes=tuple(range(x.dim())), base=base)


def qtt_unfold(xq: torch.Tensor, meta: QTTMeta) -> torch.Tensor:
    """Unfold counterpart to qtt_fold."""
    return qtt_unfold_selected(xq, meta)


class QTTWeight(nn.Module):
    """
    Trainable QTT weight:
    - Quantizes only the last N dims (e.g., spatial modes of a spectral kernel),
      folding them into base-2 axes.
    - Stores a TT factorization (tltorch FactorizedTensor) over the folded shape.
    - to_tensor() reconstructs dense weights and un-quantizes (unfold + unpad).
    """
    def __init__(
        self,
        weight_shape: Tuple[int, ...],
        quantize_last_ndims: int = 0,
    rank = 4,
        init_std: float = 1.0,
        base: int = 2,
        dtype = torch.cfloat,
        device: torch.device | None = None,
        tt_order: str = 'in-out-bits',  # 'in-out-bits' (default, current) or 'in-bits-out' (optimized operator)
        bit_ordering: str = 'serial',   # 'serial', 'interleaved', 'bitrev_interleaved', 'paired_spatial', 'paired_all'
        bitrev_ax_indices = None,        # optional Sequence[int]: quantized-axis indices with reversed bits
    ):
        super().__init__()
        if quantize_last_ndims < 0 or quantize_last_ndims > len(weight_shape):
            raise ValueError("quantize_last_ndims must be in [0, len(weight_shape)]")

        self.weight_shape = tuple(weight_shape)
        self.base = base

        # Select the trailing axes to quantize (e.g., the spatial modes)
        q_axes = tuple(range(len(weight_shape) - quantize_last_ndims, len(weight_shape)))

        # Compute folded shape from sizes only (no allocation)
        folded_shape, ms, pads, _ = _fold_shape_only(self.weight_shape, q_axes, base)
        self._meta = QTTMeta(
            orig_shape=self.weight_shape,
            quantize_axes=q_axes,
            ms=ms,
            pads=pads,
            base=base,
        )

        # Bit ordering permutation for QTT dimensions
        _valid_orderings = ('serial', 'interleaved', 'bitrev_interleaved', 'paired_spatial', 'paired_all',
                            'ch_serial', 'ch_io_block', 'ch_paired_spatial', 'ch_paired_all',
                            'input_interleaved')
        self.bit_ordering = bit_ordering if bit_ordering in _valid_orderings else 'serial'
        _bitrev_set = set(bitrev_ax_indices) if bitrev_ax_indices is not None else set()
        self._bit_perm: List[int] = _compute_bit_perm(ms, self.bit_ordering, _bitrev_set)

        # Store TT ordering preference
        self._tt_order = tt_order
        if self._tt_order not in ('in-out-bits', 'in-bits-out'):
            self._tt_order = 'in-out-bits'

        # Initialize caching for dense weight reconstruction
        self._cached_dense_tensor = None
        self._cache_valid = False
        self._last_tt_hash = None
        
        # Initialize caching for bit patterns (operator optimization)
        self._bit_patterns_cache = {}  # Cache by (N, device, spatial_shape)
        # Toggle for einsum operator contraction path.
        # Default behavior: if STFNO_QTT_EINSUM is unset, enable for 'in-bits-out' ordering,
        # otherwise respect explicit env toggle (treat 0/false/no/off as disabled).
        env_einsum = os.environ.get('STFNO_QTT_EINSUM', None)
        if env_einsum is None:
            self._use_single_einsum = (self._tt_order == 'in-bits-out')
        else:
            val = str(env_einsum).strip().lower()
            self._use_single_einsum = val not in ('0', 'false', 'no', 'off', '')
        
        # Choose complex or real TT depending on dtype. torch.is_complex_dtype may not
        # exist in older/newer installs, so fall back to checking common complex dtypes.
        try:
            is_complex = torch.is_complex_dtype(dtype)
        except Exception:
            is_complex = dtype in (torch.cfloat, torch.cdouble)
        factorization = 'ComplexTT' if is_complex else 'TT'

        # Normalize rank for TT: floats <=1 treated as ratio; floats >1 -> int rank
        def _normalize_tt_rank(rnk, order: int):
            if isinstance(rnk, float):
                if rnk <= 1.0:
                    # clamp ratio to a small positive value
                    return max(1e-6, float(rnk))
                return max(1, int(round(rnk)))
            if isinstance(rnk, int):
                return max(1, rnk)
            if isinstance(rnk, (list, tuple)):
                if len(rnk) == 1:
                    return max(1, int(rnk[0]))
                # ensure per-core ranks are >=1
                return tuple(max(1, int(x)) for x in rnk)
            # default safe rank
            return 1

        # Create TT cores over the folded shape.
        # Default 'in-out-bits': TT dims are [Cin, Cout, bit1, bit2, ...]
        # Optimized 'in-bits-out': TT dims are [Cin, bit1, bit2, ..., Cout]
        if self._tt_order == 'in-bits-out':
            # Ensure the non-quantized dims are (Cin, Cout) leading axes
            # and quantized axes are the trailing ones
            expected_q_axes = tuple(range(len(self.weight_shape) - quantize_last_ndims, len(self.weight_shape)))
            if self._meta.quantize_axes != expected_q_axes or len(self.weight_shape) < 2:
                # Fallback to default ordering if unsupported configuration
                tt_dims = tuple(folded_shape)
                self._tt_order = 'in-out-bits'
            else:
                total_bits = sum(int(m) for m in (self._meta.ms or []))
                # If there are no quantized bits, treat this as a plain TT over the
                # folded_shape (which equals the original shape when no axes were
                # quantized). The in-bits-out ordering only makes sense when bits
                # are present; fall back to in-out-bits for the zero-bit case.
                if total_bits == 0:
                    tt_dims = tuple(folded_shape)
                    self._tt_order = 'in-out-bits'
                else:
                    Cin = int(self.weight_shape[0])
                    Cout = int(self.weight_shape[1]) if len(self.weight_shape) > 1 else 1
                    tt_dims = (Cin, *([self.base] * total_bits), Cout)
        else:
            tt_dims = tuple(folded_shape)

        rank_spec = _normalize_tt_rank(rank, order=len(tt_dims))
        self.tt = FactorizedTensor.new(
            tt_dims,
            rank=rank_spec,
            factorization=factorization,
            dtype=dtype,
            device=device,
        )
        # Init cores: prefer small uniform init to avoid tltorch normal_ NaNs
        try:
            init_std_val = float(init_std)
        except Exception:
            init_std_val = 0.02
        # clamp std to a small positive range
        if not (init_std_val > 0) or not (init_std_val == init_std_val):
            init_std_val = 0.02
        init_std_val = min(init_std_val, 0.05)
        # Directly initialize each TT factor to avoid tltorch's normal_() overflow bug.
        # When quantize_last_ndims=5 with rank>=8, np.prod(rank) overflows int64,
        # causing division by zero and NaN in tltorch's initialization.
        #
        # The factors in ComplexTTTensor are stored as Parameters named factor_0, factor_1, etc.
        # within a ComplexFactorList container. We need to access them directly via named_parameters.
        #
        # Rank-aware scaling: for a TT of order N with bond rank R, the reconstructed
        # tensor has E[w^2] ≈ R^(N-1) × (a²/3)^N where a is the per-core uniform range.
        # Solving for a so that std(w) = init_std:
        #   a = sqrt(3) × init_std^(1/N) / R^((N-1)/(2N))
        # This prevents overflow for large N (e.g. Dense-QTT with quantize_last_ndims=8
        # has N=42 binary modes at w=56 and R=20, giving std(w)~10^15 with the naive a=init_std^(1/N)).
        tt_order = self.tt.order if hasattr(self.tt, 'order') else len(folded_shape)
        # Extract scalar rank for normalization (use the rank parameter passed to __init__)
        if isinstance(rank, (int, float)) and float(rank) > 1.0:
            rank_scalar = max(1, int(round(float(rank))))
        elif isinstance(rank, (list, tuple)) and len(rank) > 0:
            rank_scalar = max(1, int(max(int(r) for r in rank)))
        else:
            rank_scalar = 1
        if tt_order > 0:
            std_per_core = (
                math.sqrt(3.0)
                * (init_std_val ** (1.0 / tt_order))
                / (rank_scalar ** ((tt_order - 1) / (2.0 * tt_order)))
            )
        else:
            std_per_core = init_std_val

        for name, param in self.tt.named_parameters():
            if param.data is not None:
                if param.data.is_complex():
                    # For complex tensors, initialize real and imag parts separately
                    real_part = torch.empty_like(param.data.real).uniform_(-std_per_core, std_per_core)
                    imag_part = torch.empty_like(param.data.imag).uniform_(-std_per_core, std_per_core)
                    param.data = torch.complex(real_part, imag_part)
                else:
                    param.data.uniform_(-std_per_core, std_per_core)

        # Safety net: sanitize any remaining NaN/Inf values (should not be needed now)
        for attr in ("factors", "factors_list", "cores", "weights", "components"):
            fs = getattr(self.tt, attr, None)
            if fs is None:
                continue
            if isinstance(fs, (list, tuple)):
                for f in fs:
                    data = getattr(f, "data", None)
                    if data is not None and (torch.isnan(data).any() or torch.isinf(data).any()):
                        f.data = torch.nan_to_num(f.data, nan=0.0, posinf=1e3, neginf=-1e3)
            else:
                data = getattr(fs, "data", None)
                if data is not None and (torch.isnan(data).any() or torch.isinf(data).any()):
                    fs.data = torch.nan_to_num(fs.data, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Register hook to invalidate cache when parameters change during training
        self.register_full_backward_hook(self._on_backward_hook)

    def _on_backward_hook(self, module, grad_input, grad_output):
        """Hook called after backward pass to invalidate caches."""
        self.invalidate_cache()
        return None

    @property
    def name(self) -> str:
        return "QTT"

    def to_tensor(self) -> torch.Tensor:
        """Return dense weight in the original (unquantized, unpadded) shape.
        
        Uses intelligent caching to avoid expensive reconstruction on every call.
        """
        # Check if cache is valid
        if self._cache_valid and self._cached_dense_tensor is not None:
            # Verify TT parameters haven't changed
            current_hash = self._compute_tt_hash()
            if current_hash == self._last_tt_hash:
                return self._cached_dense_tensor
            else:
                # Parameters changed, invalidate cache
                self._cache_valid = False
        
        # Reconstruct dense tensor and cache it
        folded_dense = self.tt.to_tensor()
        # If TT was built as (Cin, bits..., Cout), permute back to (Cin, Cout, bits...)
        if self._tt_order == 'in-bits-out':
            total_bits = sum(int(m) for m in (self._meta.ms or []))
            # folded_dense dims: [Cin, bit1..bitB, Cout] -> [Cin, Cout, bit1..bitB]
            if folded_dense.dim() == (2 + total_bits):
                perm = [0, total_bits + 1] + list(range(1, total_bits + 1))
                folded_dense = folded_dense.permute(perm).contiguous()
        # Apply bit ordering permutation: convert TT-ordered bits back to serial order.
        # After this, folded_dense has bits in the same order as qtt_fold_selected produced.
        #
        # folded_dense[..., v_0,...,v_{B-1}]: TT core k has value v_k = serial bit _bit_perm[k].
        # serial_folded[..., s_0,...,s_{B-1}]: s_j = value of serial bit j.
        # Relationship: serial_folded[...,s] = folded_dense[..., s_{perm[0]}, ..., s_{perm[B-1]}]
        #   where perm = _bit_perm.
        # PyTorch permute(perm_arg)[...,j] = folded_dense[..., j_{perm_arg^{-1}(0)}, ...]
        # So we need perm_arg = inverse(_bit_perm).
        if self.bit_ordering != 'serial' and self._bit_perm:
            n_non_q = len(self._meta.orig_shape) - len(self._meta.quantize_axes)
            total_bits = len(self._bit_perm)
            if total_bits > 0 and folded_dense.dim() == n_non_q + total_bits:
                inv_perm = [0] * total_bits
                for k, p in enumerate(self._bit_perm):
                    inv_perm[p] = k
                full_perm = list(range(n_non_q)) + [n_non_q + inv_perm[k] for k in range(total_bits)]
                folded_dense = folded_dense.permute(full_perm).contiguous()
        dense_tensor = qtt_unfold_selected(folded_dense, self._meta)
        
        # Update cache
        self._cached_dense_tensor = dense_tensor
        self._cache_valid = True
        self._last_tt_hash = self._compute_tt_hash()
        
        return dense_tensor 

    def _compute_tt_hash(self) -> int:
        """Compute a hash of TT parameters to detect changes."""
        try:
            # Get TT cores/factors and compute hash of their data
            cores = self._get_tt_cores()
            if cores:
                # Hash the concatenated data of all cores
                data_parts = []
                for core in cores:
                    if hasattr(core, 'data'):
                        data_parts.append(core.data.flatten())
                    else:
                        data_parts.append(core.flatten())
                if data_parts:
                    combined = torch.cat(data_parts)
                    return hash(combined.cpu().numpy().tobytes())
            # Fallback: use hash of TT object's string representation
            return hash(str(self.tt.state_dict()))
        except Exception:
            # If hashing fails, return a random value to force reconstruction
            import time
            return hash(time.time())
    
    def invalidate_cache(self):
        """Manually invalidate all caches."""
        self._cache_valid = False
        self._cached_dense_tensor = None
        self._last_tt_hash = None
        self._bit_patterns_cache.clear()
    
    def _get_cached_bit_patterns(self, indices: torch.Tensor) -> torch.Tensor | None:
        """Get cached bit patterns for spatial indices, creating if necessary."""
        try:
            N = indices.shape[0]
            # Create a cache key based on the unique spatial pattern
            # For efficiency, we hash the indices to create a compact key
            indices_key = hash(indices.cpu().numpy().tobytes())
            cache_key = (N, str(indices.device), indices_key, tuple(self._meta.ms))
            
            if cache_key not in self._bit_patterns_cache:
                # Limit cache size to prevent memory leaks
                if len(self._bit_patterns_cache) > 20:
                    # Clear old entries (simple cleanup)
                    keys_to_remove = list(self._bit_patterns_cache.keys())[:-10]
                    for k in keys_to_remove:
                        del self._bit_patterns_cache[k]
                
                # Compute bit patterns and cache them
                kx, ky, kz = indices[:, 0], indices[:, 1], indices[:, 2]
                ms = self._meta.ms
                if len(ms) == 3:
                    b1 = self._idx_to_bits(kx.to(torch.long), int(ms[0]), self._meta.base)
                    b2 = self._idx_to_bits(ky.to(torch.long), int(ms[1]), self._meta.base)
                    b3 = self._idx_to_bits(kz.to(torch.long), int(ms[2]), self._meta.base)
                    bits_all = torch.cat([b1, b2, b3], dim=-1)  # (N, total_bits)
                    self._bit_patterns_cache[cache_key] = bits_all
                else:
                    return None  # Fall back for non-standard cases
            
            return self._bit_patterns_cache[cache_key]
        
        except Exception:
            # Return None to trigger fallback to original implementation
            return None
    
    def get_cache_info(self) -> dict:
        """Get information about cache status for debugging."""
        return {
            'dense_cache_valid': self._cache_valid,
            'has_cached_tensor': self._cached_dense_tensor is not None,
            'cached_tensor_shape': self._cached_dense_tensor.shape if self._cached_dense_tensor is not None else None,
            'cached_tensor_memory_mb': (self._cached_dense_tensor.numel() * self._cached_dense_tensor.element_size() / 1024**2) if self._cached_dense_tensor is not None else 0,
            'last_tt_hash': self._last_tt_hash,
            'bit_patterns_cache_size': len(self._bit_patterns_cache)
        }

    def get_last_op_timing(self) -> dict:
        """Return the last detailed operator-path timing breakdown (if enabled).

        Enable by setting env STFNO_QTT_TIMING to a truthy value (e.g., 1/true/on).
        Keys (when single-einsum path is used):
          - cores_ms: TT cores retrieval time
          - bits_decode_ms: decoding or fetching per-row bit patterns
          - gin_prep_ms: preparing the input core (Cin->r1)
          - selectors_ms: building one-hot selectors for all bits
          - einsum_setup_ms: building equation/operands for einsum
          - einsum_ms: the einsum contraction call
          - op_total_ms: total time spent inside matmul()
        """
        return getattr(self, "_last_op_timing", {})

    # ------------------------
    # Operator-style matvec helpers and kernel
    # ------------------------
    def _get_tt_cores(self):
        """Best-effort extractor for TT cores with shape (r_{k-1}, n_k, r_k).

        Returns a list of tensors or None if unavailable.
        """
        import torch.nn as nn  # local import to avoid global cost
        # Common attribute names in tltorch
        candidate_attrs = ("cores", "factors", "factors_list", "components", "decomposition")
        for attr in candidate_attrs:
            obj = getattr(self.tt, attr, None)
            if obj is None:
                continue
            # Unwrap container types into a list
            seq = None
            if isinstance(obj, (list, tuple)):
                seq = list(obj)
            else:
                # Some factorized tensors expose an object with an attribute that holds the list
                for subname in ("cores", "factors", "factors_list", "components"):
                    sub = getattr(obj, subname, None)
                    if isinstance(sub, (list, tuple)) and sub:
                        seq = list(sub)
                        break
                # If still not a plain list/tuple, try generic iterable containers (e.g., ComplexFactorList)
                if seq is None:
                    try:
                        # Some containers have __len__ and __getitem__ but not __iter__
                        if hasattr(obj, '__len__') and hasattr(obj, '__getitem__'):
                            n = len(obj)
                            if n > 0:
                                tmp = [obj[i] for i in range(n)]
                                if tmp:
                                    seq = tmp
                    except Exception:
                        pass
            if not seq:
                continue
            # Normalize to raw tensors where possible
            norm = []
            for c in seq:
                if torch.is_tensor(c):
                    norm.append(c)
                elif isinstance(c, nn.Parameter):
                    norm.append(c.data)
                else:
                    data = getattr(c, "data", None)
                    if torch.is_tensor(data):
                        norm.append(data)
                    else:
                        # Some wrappers keep the tensor under .weight or .weights
                        w = getattr(c, "weight", None)
                        if w is None:
                            w = getattr(c, "weights", None)
                        if torch.is_tensor(w):
                            norm.append(w)
            if norm:
                return norm
        return None

    @staticmethod
    def _idx_to_bits(idx: torch.Tensor, m: int, base: int = 2) -> torch.Tensor:
        """Convert indices (N,) to base-ary digit vectors (N, m), least-significant digit first."""
        if m == 0:
            return idx.new_zeros(idx.shape + (0,))
        bits = [(idx // (base ** b)) % base for b in range(m)]
        return torch.stack(bits, dim=-1)

    def _matmul_einsum_in_bits_out(
        self,
        x_rows: torch.Tensor,                # (N, Cin)
        core_in: torch.Tensor,               # (r0, Cin, r1) or (Cin, r1)
        bit_cores: List[torch.Tensor],       # list of (rk, base, r{k+1})
        core_out: torch.Tensor,              # (rB, Cout, r_end)
        bits_all: torch.Tensor,              # (N, B) values in [0, base-1]
        *,
        timing_on: bool = False,
    ) -> torch.Tensor:
        """Single-einsum contraction for in-bits-out ordering using one-hot selectors."""
        N, Cin = x_rows.shape
        base = int(self.base)
        B = len(bit_cores)
        assert bits_all.shape == (N, B), f"bits_all must be (N,{B}), got {tuple(bits_all.shape)}"
        # Helper sync for more accurate timing when on CUDA
        def _sync():
            try:
                if x_rows.is_cuda:
                    torch.cuda.synchronize(x_rows.device)
            except Exception:
                pass

        # Prepare selectors Sk: (N, base) for each bit position
        """
        ONE HOT SELECTORS: INDEX INTO SPECIFIC SLICES OF TT CORES BASED ON BINARY 
        REPRESENTATIONS OF SPATIAL FREQ

        For a 32³ grid with modes=16³, each spatial frequency (kx, ky, kz) is encoded 
        as a 12-bit binary number (4 bits per dimension). The einsum contraction needs 
        to select the correct "slice" from each TT core based on these bits.

        Each selector Sk is a (N, base) matrix where row i has [1,0] if bit k is 0, 
        or [0,1] if bit k is 1.
        """
    
        selectors = []
        # Always time selectors to ensure breakdown prints
        _sync(); t0_sel = time.perf_counter()
        for k in range(B):
            bk = bits_all[:, k].to(torch.long)
            Sk = F.one_hot(bk, num_classes=base).to(dtype=x_rows.dtype, device=x_rows.device)
            selectors.append(Sk)
        _sync();
        self._last_op_timing = getattr(self, "_last_op_timing", {})
        self._last_op_timing["selectors_ms"] = (time.perf_counter() - t0_sel) * 1e3

        # Prepare first core Gin: (Cin, r1)
        _sync(); t0_gin = time.perf_counter()
        if core_in.dim() == 3 and core_in.shape[0] == 1:
            Gin = core_in.squeeze(0) 
        elif core_in.dim() == 2:
            Gin = core_in
        else:
            raise RuntimeError(f"Unexpected core_in shape {tuple(core_in.shape)}")
        _sync(); self._last_op_timing["gin_prep_ms"] = (time.perf_counter() - t0_gin) * 1e3

        if core_out.dim() != 3:
            raise RuntimeError(f"Unexpected core_out shape {tuple(core_out.shape)}")

        # Build einsum equation symbols
        _sync(); t0_setup = time.perf_counter()
        import string
        # Reserve symbols; avoid 'n','i','o'
        sym_pool = [c for c in (string.ascii_lowercase + string.ascii_uppercase) if c not in ('n', 'i', 'o')]
        need = (B + 2) + B + 1  # r0..rB + bits + q i.e. tt rank bits plus bit symbols plus output trailing rank q 
        if len(sym_pool) < need:
            # Fallback to incremental path if not enough symbols
            raise RuntimeError("Not enough einsum symbols for large B")
        rank_syms = sym_pool[:B + 2]          # r0..rB (we will use up to rB)
        bit_syms = sym_pool[B + 2:B + 2 + B]  # b0..b{B-1}
        q_sym = sym_pool[B + 2 + B]           # trailing rank of core_out

        eq_terms = []
        operands = []

        # X and Gin
        eq_terms.append('ni'); operands.append(x_rows)
        eq_terms.append(f"i{rank_syms[0]}"); operands.append(Gin)  # 'ir0'

        # Chain bit cores with selectors
        r_prev = rank_syms[0]
        for k in range(B):
            r_next = rank_syms[k + 1]
            b = bit_syms[k]
            Ck = bit_cores[k]
            if Ck.dim() != 3 or Ck.shape[1] != base:
                raise RuntimeError(f"bit_core[{k}] has shape {tuple(Ck.shape)}, expected (*,{base},*)")
            eq_terms.append(f"{r_prev}{b}{r_next}"); operands.append(Ck)
            eq_terms.append(f"n{b}"); operands.append(selectors[k]) # selector for bit k used here
            r_prev = r_next

        # Output core
        eq_terms.append(f"{r_prev}o{q_sym}"); operands.append(core_out)

        equation = ','.join(eq_terms) + '->no'
        _sync(); self._last_op_timing["einsum_setup_ms"] = (time.perf_counter() - t0_setup) * 1e3
        # Always measure the einsum call; use CUDA events on GPU, perf_counter on CPU
        if x_rows.is_cuda:
            try:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(x_rows.device)
                start_evt.record()
                print("Executing einsum:", equation)
                print("With operands shapes:", [tuple(op.shape) for op in operands])
                y = torch.einsum(equation, *operands)
                end_evt.record()
                torch.cuda.synchronize(x_rows.device)
                self._last_op_timing["einsum_ms"] = float(start_evt.elapsed_time(end_evt))
            except Exception:
                _sync(); t0_einsum = time.perf_counter()
                print("Executing einsum:", equation)
                print("With operands shapes:", [tuple(op.shape) for op in operands])
                y = torch.einsum(equation, *operands)
                _sync(); self._last_op_timing["einsum_ms"] = (time.perf_counter() - t0_einsum) * 1e3
        else:
            _sync(); t0_einsum = time.perf_counter()
            y = torch.einsum(equation, *operands)
            _sync(); self._last_op_timing["einsum_ms"] = (time.perf_counter() - t0_einsum) * 1e3
        # Mark that we reached the einsum contraction path
        try:
            self._last_op_timing["debug_reached_einsum"] = True
        except Exception:
            pass
        return y

    def matmul(self, x_rows: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the QTT operator to many row vectors without reconstructing the dense tensor.

        x_rows: (N, Cin) complex or real tensor
        indices: optional (N, 3) long tensor of (kx, ky, kz) local indices per row
        returns: (N, Cout) tensor
        """
        if x_rows.dim() != 2:
            raise ValueError(f"x_rows must be 2D (N, Cin), got {tuple(x_rows.shape)}")

        N, Cin = x_rows.shape
        in_dim = self.weight_shape[0]
        if Cin != in_dim:
            raise ValueError(f"Expected Cin={in_dim} but got {Cin}")

        # Set up timing
        timing_on = str(os.getenv("STFNO_QTT_TIMING", "0")).strip().lower() not in ("0", "false", "no", "off", "")
        self._last_op_timing = {}
        def _sync():
            try:
                if x_rows.is_cuda:
                    torch.cuda.synchronize(x_rows.device)
            except Exception:
                pass
        t_all0 = time.perf_counter();

        # Load TT cores
        if timing_on:
            _sync(); t0 = time.perf_counter()
        cores = self._get_tt_cores()
        if timing_on:
            _sync(); self._last_op_timing["cores_ms"] = (time.perf_counter() - t0) * 1e3
        if not cores:
            # Fallback: reconstruct dense and collapse spatial dims
            dense = self.to_tensor()
            if len(dense.shape) == 5:
                dense_mat = dense.sum(dim=(2, 3, 4))  # (Cin, Cout)
            elif len(dense.shape) == 4:
                dense_mat = dense.sum(dim=(1, 2, 3)).unsqueeze(1)  # (Cin,1)
            else:
                dense_mat = dense
            y = x_rows @ dense_mat
            if timing_on:
                _sync(); self._last_op_timing["op_total_ms"] = (time.perf_counter() - t_all0) * 1e3
            return y

        if self._tt_order == 'in-bits-out':
            core_in = cores[0]
            core_out = cores[-1]
            bit_cores = cores[1:-1]
        else:
            core_in = cores[0]
            core_out = cores[1]
            bit_cores = cores[2:]

        # Build per-row bit selections for spatial cores when present
        total_bits = sum(int(m) for m in (self._meta.ms or []))
        if timing_on:
            # Record how many bit cores we expect and have
            try:
                self._last_op_timing["debug_total_bits"] = float(total_bits)
                self._last_op_timing["debug_bit_cores_len"] = float(len(bit_cores))
            except Exception:
                pass
        if total_bits != len(bit_cores):
            total_bits = min(total_bits, len(bit_cores))

        if total_bits > 0:
            if indices is None:
                raise ValueError("indices=(kx,ky,kz) must be provided for QTT matmul with spatial bits")
            if indices.shape != (N, 3):
                raise ValueError(f"indices must have shape (N,3), got {tuple(indices.shape)}")
            if timing_on:
                _sync(); t_bits = time.perf_counter()
            bits_all = self._get_cached_bit_patterns(indices)
            if bits_all is None:
                kx, ky, kz = indices[:, 0], indices[:, 1], indices[:, 2]
                ms = self._meta.ms
                if len(ms) != 3:
                    parts = []
                    src = [indices[:, i] if i < indices.shape[1] else indices[:, -1]*0 for i in range(len(ms))]
                    for idx_axis, m in zip(src, ms):
                        parts.append(self._idx_to_bits(idx_axis.to(torch.long), int(m), self._meta.base))
                    bits_all = torch.cat(parts, dim=-1) if parts else indices.new_zeros((N, 0))
                else:
                    b1 = self._idx_to_bits(kx.to(torch.long), int(ms[0]), self._meta.base)
                    b2 = self._idx_to_bits(ky.to(torch.long), int(ms[1]), self._meta.base)
                    b3 = self._idx_to_bits(kz.to(torch.long), int(ms[2]), self._meta.base)
                    bits_all = torch.cat([b1, b2, b3], dim=-1)
            if timing_on:
                _sync(); self._last_op_timing["bits_decode_ms"] = (time.perf_counter() - t_bits) * 1e3
        else:
            bits_all = x_rows.new_zeros((N, 0), dtype=torch.long)

        # Fast path: single-einsum contraction for in-bits-out ordering
        fastpath_allowed = (self._use_single_einsum and self._tt_order == 'in-bits-out')
        if timing_on:
            # Debug visibility into branch conditions
            self._last_op_timing["debug_fastpath_allowed"] = float(1.0 if fastpath_allowed else 0.0)
            self._last_op_timing["debug_timing_on"] = float(1.0)
            try:
                self._last_op_timing["debug_tt_order"] = str(self._tt_order)
            except Exception:
                pass
        if timing_on:
            # Record whether we intended to use the einsum fast path
            self._last_op_timing["einsum_attempted"] = bool(fastpath_allowed)
        if fastpath_allowed:
            try:
                y = self._matmul_einsum_in_bits_out(
                    x_rows, core_in, bit_cores[:total_bits], core_out, bits_all, timing_on=timing_on
                )
                # Mark that the einsum fast-path was used. Always set this flag so
                # callers can detect which path was executed regardless of whether
                # detailed timing was requested.
                self._last_op_timing.setdefault("einsum_used", True)
                if timing_on:
                    _sync(); self._last_op_timing["op_total_ms"] = (time.perf_counter() - t_all0) * 1e3
                return y
            except Exception as e:
                # Optionally require einsum fast-path (for debugging/timing focus)
                require = str(os.getenv("STFNO_QTT_REQUIRE_EINSUM", "0")).strip().lower() not in ("0", "false", "no", "off", "")
                # Record that einsum was attempted but failed
                self._last_op_timing.setdefault("einsum_used", False)
                self._last_op_timing.setdefault("einsum_error", f"{e.__class__.__name__}: {e}")
                if require:
                    raise
                # else fall through to incremental path
        else:
            # Fast path not attempted due to configuration/order mismatch
            require = str(os.getenv("STFNO_QTT_REQUIRE_EINSUM", "0")).strip().lower() not in ("0", "false", "no", "off", "")
            if timing_on:
                self._last_op_timing["einsum_used"] = False
                self._last_op_timing.setdefault("einsum_error", "")
                if not self._last_op_timing["einsum_error"]:
                    reason = []
                    if not self._use_single_einsum:
                        reason.append("einsum_disabled")
                    if self._tt_order != 'in-bits-out':
                        reason.append(f"order={self._tt_order}")
                    self._last_op_timing["einsum_error"] = ";".join(reason) if reason else "condition_not_met"
            # Also mark that the einsum fast-path was not used
            self._last_op_timing.setdefault("einsum_used", False)
            if require:
                raise RuntimeError("Einsum fast path required but not used due to configuration/order")

        # Incremental contraction paths
        # 1) Contract input core: (N, Cin) x (Cin, r1) -> (N, r1)
        Gin = core_in.squeeze(0) if core_in.dim() == 3 and core_in.shape[0] == 1 else core_in
        # Measure contraction-only time for incremental path
        contract_ms_total = 0.0
        if timing_on:
            _sync(); _t0c = time.perf_counter()
        left_rank = x_rows @ Gin
        if timing_on:
            _sync(); contract_ms_total += (time.perf_counter() - _t0c) * 1e3

        if self._tt_order == 'in-bits-out':
            # Walk bits then apply output core
            r_cur = left_rank.shape[-1]
            for j in range(total_bits):
                core_j = bit_cores[j]
                r_next = core_j.shape[-1]
                bj = bits_all[:, j].to(x_rows.device)
                core_perm = core_j.permute(1, 0, 2).contiguous()  # (base, r_cur, r_next)
                if timing_on:
                    _sync(); _t0c = time.perf_counter()
                M = core_perm.index_select(0, bj).permute(1, 0, 2)
                M = M.permute(1, 0, 2).contiguous()                # (N, r_cur, r_next)
                left_rank = torch.einsum('nr,nrs->ns', left_rank, M)
                if timing_on:
                    _sync(); contract_ms_total += (time.perf_counter() - _t0c) * 1e3
                r_cur = r_next
            if core_out.shape[-1] != 1:
                W = core_out.sum(dim=-1)   # (r_last, Cout)
            else:
                W = core_out[..., 0]       # (r_last, Cout)
            if timing_on:
                _sync(); _t0c = time.perf_counter()
            y_rows = left_rank @ W
            if timing_on:
                _sync(); contract_ms_total += (time.perf_counter() - _t0c) * 1e3
            if timing_on:
                _sync(); self._last_op_timing["op_total_ms"] = (time.perf_counter() - t_all0) * 1e3
                # Mark that incremental path was used if not already marked above
                self._last_op_timing.setdefault("einsum_used", False)
                # Record contraction-only time for incremental path
                try:
                    self._last_op_timing["contract_ms"] = float(contract_ms_total)
                except Exception:
                    pass
            return y_rows

        # Default path: combine Cout early and walk bits while carrying Cout
        if timing_on:
            _sync(); _t0c = time.perf_counter()
        T = torch.tensordot(left_rank, core_out, dims=([1], [0]))  # (N, Cout, r2)
        if timing_on:
            _sync(); contract_ms_total += (time.perf_counter() - _t0c) * 1e3
        r_cur = T.shape[-1]
        for j in range(total_bits):
            core_j = bit_cores[j]
            r_next = core_j.shape[-1]
            bj = bits_all[:, j].to(x_rows.device)
            core_perm = core_j.permute(1, 0, 2).contiguous()
            if timing_on:
                _sync(); _t0c = time.perf_counter()
            M = core_perm.index_select(0, bj).permute(1, 0, 2)
            M = M.permute(1, 0, 2).contiguous()
            T = torch.einsum('ncr,nrs->ncs', T, M)
            if timing_on:
                _sync(); contract_ms_total += (time.perf_counter() - _t0c) * 1e3
            r_cur = r_next
        if timing_on:
            _sync(); _t0c = time.perf_counter()
        y_rows = T.sum(dim=-1) if r_cur != 1 else T.squeeze(-1)
        if timing_on:
            _sync(); contract_ms_total += (time.perf_counter() - _t0c) * 1e3
        if timing_on:
            _sync(); self._last_op_timing["op_total_ms"] = (time.perf_counter() - t_all0) * 1e3
            # Mark that incremental path was used if not already marked above
            self._last_op_timing.setdefault("einsum_used", False)
            # Record contraction-only time for incremental path
            try:
                self._last_op_timing["contract_ms"] = float(contract_ms_total)
            except Exception:
                pass
        return y_rows

def get_global_cache_info() -> dict:
    """Get information about global caches for debugging."""
    from . import fourier_transform_3d_layer_factorized as ft3d
    return {
        'global_spatial_cache_size': len(getattr(ft3d, '_GLOBAL_SPATIAL_CACHE', {})),
        'max_cache_size_limit': getattr(ft3d, '_MAX_CACHE_SIZE', 0)
    }
