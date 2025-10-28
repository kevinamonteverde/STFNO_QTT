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


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Get contraction function for factorized weights"""
    if implementation == "reconstructed":
        return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        #Allow QTTWeight by falling back to reconstructed-to-dense path
        elif isinstance(weight, QTTWeight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
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
                 separable=False, fft_norm='forward', init_std='auto'):
        super(FactorizedSpectralConv3d, self).__init__()
        
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
                w = QTTWeight(
                    weight_shape,
                    rank=self.rank,  # same semantics as TT
                    quantize_last_ndims=5,  # fold all dims: (in, out, modes1, modes2, modes3)
                    base=2,
                    dtype=torch.cfloat,
                    init_std=init_std,
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

        # Set up contraction function
        self._contract = get_contract_fun(
            self.weights[0], implementation=implementation, separable=separable
        )
    
    def compl_mul3d(self, input, weights):
        """Complex multiplication for 3D case"""
        return self._contract(input, weights, separable=self.separable)
    
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[-3, -2, -1])
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
        
        # Apply factorized convolution to each corner
        out_fft[slices0] = self.compl_mul3d(x_ft[slices0], self.weights[0])
        out_fft[slices1] = self.compl_mul3d(x_ft[slices1], self.weights[1])
        out_fft[slices2] = self.compl_mul3d(x_ft[slices2], self.weights[2])
        out_fft[slices3] = self.compl_mul3d(x_ft[slices3], self.weights[3])
        
        # Inverse FFT
        x = torch.fft.irfftn(out_fft, s=(x.size(-3), x.size(-2), x.size(-1)), 
                            dim=[-3, -2, -1], norm=self.fft_norm)
        return x