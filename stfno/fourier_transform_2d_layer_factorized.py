import time
import torch
import torch.nn as nn
from tltorch.factorized_tensors.core import FactorizedTensor

from .qtt import QTTWeight
from .fourier_transform_3d_layer_factorized import (
    _contract_tt_factorized,
    _contract_qtt_factorized,
    get_contract_fun,
    get_time,
    record_time,
)


def _contract_dense_2d(x, weight):
    """Dense einsum contraction for 2D spectral modes.

    x: (batch, in_channels, modes1, modes2)
    weight: (in_channels, out_channels, modes1, modes2)
    """
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    x_s = x.shape[2:]
    return torch.einsum("bixy,ioxy->boxy", x, weight[:, :, :x_s[0], :x_s[1]])


class FactorizedSpectralConv2d(nn.Module):
    """2D Fourier layer with optional TT/QTT weight compression.

    Mirrors FactorizedSpectralConv3d but for 2D spatial inputs.
    Weight shape: (C_in, C_out, modes1, modes2).
    Two corner weights (top-left, bottom-left of the rfft2 output).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        factorization: str = 'dense',
        rank: int = 4,
        implementation: str = 'reconstructed',
        fft_norm: str = 'forward',
        init_std='auto',
        timing: bool = False,
        quantize_last_ndims: int = 2,
        bit_ordering: str = 'serial',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.factorization = factorization
        self.rank = rank
        self.implementation = implementation
        self.fft_norm = fft_norm
        self.timing = timing
        self.quantize_last_ndims = quantize_last_ndims
        self.bit_ordering = bit_ordering

        if init_std == 'auto':
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        fact_lower = (factorization or '').lower()
        weight_shape = (in_channels, out_channels, modes1, modes2)

        self.weights = []
        for i in range(2):  # 2 corners in rfft2
            if factorization is None or fact_lower == 'dense':
                w = nn.Parameter(torch.zeros(weight_shape, dtype=torch.cfloat))
                nn.init.normal_(w, 0, init_std)

            elif fact_lower in ('tt', 'tensor_train'):
                w = FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank,
                    factorization='ComplexTT',
                    dtype=torch.cfloat,
                )
                std = min(float(init_std) if float(init_std) > 0 else 0.02, 0.05)
                try:
                    w.uniform_(-std, std)
                except Exception:
                    pass

            elif fact_lower == 'qtt':
                std = min(float(init_std) if float(init_std) > 0 else 0.02, 0.05)
                w = QTTWeight(
                    weight_shape=weight_shape,
                    quantize_last_ndims=self.quantize_last_ndims,
                    rank=int(self.rank),
                    init_std=std,
                    base=2,
                    dtype=torch.cfloat,
                    tt_order='in-out-bits',
                    bit_ordering=self.bit_ordering,
                )
            else:
                raise ValueError(f"Unsupported factorization: {factorization}")

            setattr(self, f'weight_{i}', w)
            self.weights.append(w)

        self._contract = get_contract_fun(self.weights[0], implementation=implementation)

    def _mul(self, x, w):
        """Apply contraction weight w to Fourier-domain slice x."""
        if isinstance(w, QTTWeight):
            if self.implementation == 'reconstructed':
                return _contract_dense_2d(x, w.to_tensor())
            return _contract_qtt_factorized(x, w)
        w_tensor = w.to_tensor() if not torch.is_tensor(w) else w
        return _contract_dense_2d(x, w_tensor)

    def forward(self, x):
        # x: (batch, C_in, H, W)
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x.to(dtype=torch.float32), norm=self.fft_norm)
        # x_ft: (batch, C_in, H, W//2+1)

        out_ft = torch.zeros(
            B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # Top-left corner: positive x, positive y frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = self._mul(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights[0]
        )
        # Bottom-left corner: negative x, positive y frequencies
        out_ft[:, :, -self.modes1:, :self.modes2] = self._mul(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights[1]
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm=self.fft_norm)

    def count_parameters(self):
        total = 0
        for w in self.weights:
            if torch.is_tensor(w):
                total += w.numel()
            elif isinstance(w, (QTTWeight, FactorizedTensor)):
                total += sum(p.numel() for p in w.parameters())
        return total
