import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .fourier_transform_2d_layer import SpectralConv2d
from .fourier_transform_2d_layer_factorized import FactorizedSpectralConv2d
from .dense_operator_2d import DenseOrQTT2DOperator


class STFNO2d_SWE(nn.Module):
    """2D STFNO / ST-NO for single-variable autoregressive prediction (SWE benchmark).

    Input:  (batch, H, W, T_in)  — T_in past snapshots per spatial point
    Output: (batch, H, W, 1)     — predicted next snapshot

    operator_type='spectral':
      Uses FFT-based Fourier layers (spectral FNO). Factorization applies to the
      mode-space weight (C_in, C_out, modes1, modes2). Weight is already small;
      dense is fine here, TT/QTT provide minimal compression gain.

    operator_type='real_op':
      Uses direct 6-way spatial operator W(Cin, H_in, W_in, Cout, H_out, W_out)
      with no FFT. REQUIRES QTT factorization at 256×256 (unfactorized weight is
      ~256^4 × width^2 entries — completely intractable). This is the ST-NO-2D
      variant and the primary QTT compression contribution for the SWE benchmark.
    """

    def __init__(
        self,
        T_in: int,
        modes1: int,
        modes2: int,
        width: int,
        n_layers: int = 4,
        factorization: str = 'dense',
        rank: int = 4,
        quantize_last_ndims: int = 2,
        bit_ordering: str = 'serial',
        implementation: str = 'reconstructed',
        operator_type: str = 'spectral',
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.T_in = T_in
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.factorization = factorization
        self.operator_type = operator_type

        # Lifting: T_in channels + 2 grid coordinates → width
        self.fc0 = nn.Linear(T_in + 2, width)

        # Build Fourier / spatial operator layers
        fact_lower = (factorization or '').lower()

        if operator_type == 'spectral':
            if factorization is None or fact_lower == 'dense':
                self.convs = nn.ModuleList([
                    SpectralConv2d(width, width, modes1, modes2)
                    for _ in range(n_layers)
                ])
            else:
                self.convs = nn.ModuleList([
                    FactorizedSpectralConv2d(
                        width, width, modes1, modes2,
                        factorization=factorization,
                        rank=rank,
                        quantize_last_ndims=quantize_last_ndims,
                        bit_ordering=bit_ordering,
                        implementation=implementation,
                    )
                    for _ in range(n_layers)
                ])

        elif operator_type == 'real_op':
            H, W = spatial_size
            if fact_lower == 'dense' and H * W > 4096:
                raise ValueError(
                    f"real_op with factorization='dense' is infeasible at {H}×{W}. "
                    "Use factorization='qtt'."
                )
            self.convs = nn.ModuleList([
                DenseOrQTT2DOperator(
                    in_channels=width,
                    out_channels=width,
                    spatial_in=(H, W),
                    spatial_out=(H, W),
                    factorization=factorization,
                    rank=rank,
                    quantize_last_ndims=quantize_last_ndims,
                    bit_ordering=bit_ordering,
                )
                for _ in range(n_layers)
            ])

        else:
            raise ValueError(f"operator_type must be 'spectral' or 'real_op', got '{operator_type}'")

        # Pointwise bypass for residual (both spectral and real_op)
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        # Projection: width → 128 → 1
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, H, W, T_in)
        grid = self._get_grid(x.shape, x.device)      # (batch, H, W, 2)
        x = torch.cat([x, grid], dim=-1)              # (batch, H, W, T_in+2)
        x = self.fc0(x)                               # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)                     # (batch, width, H, W)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)                     # (batch, H, W, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                               # (batch, H, W, 1)
        return x

    def _get_grid(self, shape, device):
        batch, H, W, _ = shape
        gx = torch.linspace(0, 1, H, dtype=torch.float, device=device)
        gy = torch.linspace(0, 1, W, dtype=torch.float, device=device)
        gx = gx.view(1, H, 1, 1).expand(batch, H, W, 1)
        gy = gy.view(1, 1, W, 1).expand(batch, H, W, 1)
        return torch.cat([gx, gy], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
