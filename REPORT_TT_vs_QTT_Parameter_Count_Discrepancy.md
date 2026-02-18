# Report: TT vs QTT Parameter Count Discrepancy Analysis

**Date:** January 27, 2026
**Author:** Analysis by Claude (requested by Kevin Monteverde)
**Subject:** Investigation of irreproducible 759x parameter count difference between TT and QTT factorizations

---

## Executive Summary

The original experiments (September 2024) reported a dramatic 759x parameter reduction when comparing QTT to TT factorization (44,362 vs 33,659,194 parameters at rank 4). This result **cannot be reproduced** because **the TT factorization was not functioning correctly** during the original experiments. The TT weights were being stored and counted as dense tensors, not as factorized TT cores. QTT was working correctly. The code has since been fixed, and current experiments correctly show TT and QTT achieving similar compression (~13,000-14,000 parameters), with QTT providing approximately 13-20% additional savings when folding all 5 dimensions.

---

## 1. Original Experimental Results (summary.csv)

The original `summary.csv` in `Data_Logs_Tests/` reported:

| Factorization | Rank | Parameter Count |
|---------------|------|-----------------|
| qtt_r4        | 4    | 44,362          |
| qtt_r8        | 8    | 108,106         |
| qtt_r12       | 12   | 208,714         |
| tt_r4         | 4    | 33,659,194      |
| tt_r8         | 8    | 67,267,850      |
| tt_r12        | 12   | 100,763,786     |

This suggested QTT achieved **759x better compression** than TT at rank 4.

---

## 2. Evidence of the Bug

### 2.1 Anomalous Linear Scaling

The TT parameter counts scale **linearly** with rank:
- r4 → r8: exactly 2.0x (67,267,850 / 33,659,194 = 2.0)
- r4 → r12: exactly 3.0x
- r4 → r16: exactly 4.0x
- r4 → r20: exactly 5.0x

**This is incorrect.** TT factorization parameters should scale approximately with rank², not linearly. The linear scaling suggests the rank was being multiplied into something that should have been constant (i.e., dense tensor parameters).

### 2.2 Parameter Counts Match Dense Tensor Sizes

The TT parameter count of 33,659,194 closely matches the expected **dense** parameter count for the model configuration used. For comparison:
- Dense model (modes=13): ~33,752,155 parameters
- TT r4 reported: 33,659,194 parameters

### 2.3 Direct Evidence from Slurm Logs

The original experiment log (`slurm_qtt_vs_tt_42617815.out`, dated September 8, 2024) contains model summaries showing the per-layer parameter breakdown:

**QTT r4 (correctly factorized):**
```
└─FactorizedSpectralConv3d: 3-2    [1, 24, 32, 32, 32]    7,744
└─FactorizedSpectralConv3d: 3-7    [1, 24, 32, 32, 32]    6,720
└─FactorizedSpectralConv3d: 3-12   [1, 8, 32, 32, 32]     5,696
└─FactorizedSpectralConv3d: 3-17   [1, 24, 32, 32, 32]    6,720
Total params: 44,362
```

**TT r4 (NOT factorized - using dense weights):**
```
└─FactorizedSpectralConv3d: 3-2    [1, 24, 32, 32, 32]    13,280,976
└─FactorizedSpectralConv3d: 3-7    [1, 24, 32, 32, 32]    7,967,664
└─FactorizedSpectralConv3d: 3-12   [1, 8, 32, 32, 32]     4,425,408
└─FactorizedSpectralConv3d: 3-17   [1, 24, 32, 32, 32]    7,967,664
Total params: 33,659,194
```

The TT layer shows **13+ million parameters per spectral convolution**, which is the dense tensor size, not factorized TT cores (which should be ~1,000-2,000 parameters).

---

## 3. Root Cause Analysis

### 3.1 Timeline

| Date | Event |
|------|-------|
| September 8, 2024 | Original QTT vs TT experiments run (slurm job 42617815) |
| October 28, 2025 | First git commit to repository |
| January 2026 | Current investigation |

**Critical finding:** The experiments were run **before the code was committed to git**. The original buggy code that produced the invalid results does not exist in the git history.

### 3.2 The Bug: TT Weights Not Properly Registered as Factorized

Based on analysis of backup files and code evolution, the issue was in how TT weights were handled in `FactorizedSpectralConv3d`:

**The Problem:**

When using tltorch's `FactorizedTensor.new()` with `factorization='ComplexTT'`, the code creates a proper TT-factorized tensor with small cores. However, the original code had one of these issues:

1. **Parameter Registration Bug:** The TT cores from `FactorizedTensor` may not have been properly registered as `nn.Parameter` objects, causing PyTorch's `model.parameters()` to not find them.

2. **Weight Reconstruction in `__init__`:** The code may have been calling `.to_tensor()` during initialization, reconstructing the factorized weights to dense and storing the dense version.

3. **Missing TT Case in Contract Function:** The backup file (`fourier_transform_3d_layer_factorized_backup.py`) shows that `get_contract_fun()` had no explicit case for TT:

```python
# From backup file, lines 104-112:
elif isinstance(weight, FactorizedTensor):
    if weight.name.lower().endswith("dense"):
        return _contract_dense
    elif weight.name.lower().endswith("tucker"):
        return _contract_tucker
    elif weight.name.lower().endswith("cp"):
        return _contract_cp
    else:
        raise ValueError(f"Got unexpected factorized weight type {weight.name}")
```

Note: There is **no case for `.endswith("tt")`**. A ComplexTT tensor would raise a ValueError, suggesting the experiments may have used an even earlier version of the code, or the error was caught and silently handled by falling back to dense.

### 3.3 Why QTT Worked But TT Didn't

QTT uses a custom `QTTWeight` class (defined in `stfno/qtt.py`) that:
1. Explicitly inherits from `nn.Module`
2. Properly registers TT cores as parameters via `nn.ParameterList`
3. Has dedicated handling in `get_contract_fun()` with `isinstance(weight, QTTWeight)`

TT used tltorch's `FactorizedTensor` which:
1. Is also an `nn.Module` and should register parameters correctly
2. But may have had initialization or integration issues in the original code
3. Did not have explicit handling in the contraction function

---

## 4. Current State (Code is Now Fixed)

The current code correctly handles TT factorization. Testing with the current codebase:

```python
# Current code produces correct TT parameter counts:
TT  rank=4, modes=6, width=8:  13,595 params  # Correct!
QTT rank=4, modes=6, width=8:  11,867 params  # Correct!
```

The current `get_contract_fun()` includes proper TT handling:
```python
elif weight.name.lower().endswith("tt"):
    # For TT, use dense fallback for now (modes are small)
    return _contract_dense
```

While this still uses dense reconstruction for the forward pass (for performance with small modes), the **parameter counting is now correct** because the TT cores are properly registered.

---

## 5. Correct Comparison: TT vs QTT

Based on current (fixed) code and the new `stfno_sweep` experiments:

### 5.1 Full Model Parameter Counts (modes=6, width=8, rank=4)

| Factorization | Parameters | Relative to Dense |
|---------------|------------|-------------------|
| Dense         | 3,323,995  | 1.00x             |
| TT            | 13,595     | 0.004x (245x compression) |
| QTT-q5        | 11,867     | 0.004x (280x compression) |

### 5.2 QTT Advantage Over TT

The **actual** advantage of QTT over TT is approximately **13-20%** additional parameter savings, not 759x. This advantage comes from:

1. **Binary folding of channel dimensions:** When `quantize_last_ndims=5`, QTT folds the channel dimensions (Cin, Cout) into base-2 axes, which can provide additional compression when channels are large (e.g., 80, 48, 24).

2. **More uniform core sizes:** QTT's binary folding creates more uniform core dimensions, which can be more efficient for certain tensor shapes.

The advantage is modest but real, and most significant when:
- Channel dimensions are large (not powers of 2)
- Using `quantize_last_ndims=5` (fold all dimensions, not just spatial)

---

## 6. Conclusions

### 6.1 Why Original Results Cannot Be Reproduced

1. **The TT factorization was broken** in the code version used for the September 2024 experiments
2. **The original code does not exist** in git history (first commit was October 2025)
3. **The bug has been fixed** in the current codebase

### 6.2 Validity of Original Results

| Metric | QTT Results | TT Results |
|--------|-------------|------------|
| Parameter Count | **Valid** (correctly factorized) | **Invalid** (dense, not factorized) |
| Training Loss | Valid | Valid (model trained correctly, just not compressed) |
| Test Loss | Valid | Valid |

The QTT experiments are valid. The TT experiments trained successfully but were **not actually using TT compression** - they were training a dense model.

### 6.3 Path Forward

1. **Discard the TT parameter counts from summary.csv** - they are invalid
2. **Use current stfno_sweep results** for valid TT vs QTT comparison
3. **Expected QTT advantage: 13-20%** over TT (not 759x)
4. **Both TT and QTT provide ~200-300x compression** over dense

---

## 7. Supporting Evidence Files

| File | Description |
|------|-------------|
| `Data_Logs_Tests/slurm_qtt_vs_tt_42617815.out` | Original experiment log showing TT with dense params |
| `Data_Logs_Tests/summary.csv` | Original (invalid) summary with 759x difference |
| `Data_Logs_Tests/stfno_sweep/combined_results.csv` | Current valid sweep results |
| `STFNO_QTT_old/STFNO_QTT/stfno/fourier_transform_3d_layer_factorized_backup.py` | Backup showing missing TT case |

---

## 8. Appendix: Verification Commands

To verify current (correct) behavior:

```bash
module load python
python3 -c "
import sys
sys.path.insert(0, 'examples/NIMROD3D')
sys.path.insert(0, '.')
from stfno.stfno_3d import FNO2d_NIMRODglobal_3D
from stfno.utilities3 import count_params

input_parameter_order = [[0,1,2,3,4,5,6,7,8,9]]
mWidth_input_parameters = [10]
nWidth_output_parameters = [3]

for fact in ['dense', 'tt', 'qtt']:
    model = FNO2d_NIMRODglobal_3D(
        modes1=6, modes2=6, modes3=6, width=8, T_in=1,
        total_vector_a_elements_i=10, T=1, total_vector_u_elements_i=3,
        number_of_layers=1, input_parameter_order=input_parameter_order,
        mWidth_input_parameters=mWidth_input_parameters,
        nWidth_output_parameters=nWidth_output_parameters,
        if_model_jit_torchCompile=False, factorization=fact, rank=4
    )
    print(f'{fact}: {count_params(model):,} params')
"
```

Expected output:
```
dense: 3,323,995 params
tt: 13,595 params
qtt: 11,867 params
```

---

*End of Report*
