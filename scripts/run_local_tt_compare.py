#!/usr/bin/env python3
import csv
import pathlib
import time
import torch
from stfno.fourier_transform_3d_layer_factorized import FactorizedSpectralConv3d

# ---- Configuration (edit here) ----
batch_size = 1
runs = 20           # measured runs
warmup = 5        # warmup runs (excluded from averages)
device_pref = "cpu"  # "auto" | "cpu" | "cuda"

# Mirror contraction_test.py configs
# Each tuple: (label, width, spatial, rank)
TEST_CONFIGS = [
    ("c16_s16_r4", 16, 16, 4),
    ("c32_s32_r4", 32, 32, 4),
    ("c48_s48_r4", 48, 48, 4),
    ("c64_s64_r4", 64, 64, 4),
    ("c32_s32_r8", 32, 32, 8),
    ("c64_s64_r8", 64, 64, 8)
]
# -----------------------------------

# Timing columns written to CSVs (forward-pass order):
# - fft_ms: forward FFT time
# - contract_ms: spectral contraction time (path-dependent)
# - ifft_ms: inverse FFT time
# - reconstruct_ms: factorized -> dense materialization time (reconstructed path)
# - tt_core_ms: TT core contraction timing (single-einsum path)
# - tt_batched_ms: TT batched profile (if available)
# - tt_fact_ms: TT FactorizedTensor contraction timing (tltorch-style factors)
# - total_ms: sum of stage timings (fft + contract + ifft + reconstruct)
# - total_wall_ms: host wall-clock around the whole forward (includes overhead/sync)

def run_once(cfg_label: str, layer_label: str, layer: FactorizedSpectralConv3d, x: torch.Tensor, rows: list[dict], run_idx: int) -> None:
    """Single forward pass with timing capture."""
    wall_start = time.perf_counter()
    with torch.no_grad():
        _ = layer(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    total_wall_ms = (time.perf_counter() - wall_start) * 1e3
    timing = getattr(layer, "last_timing", {})
    fft_ms = float(timing.get("fft_ms", 0.0))
    contract_ms = float(timing.get("contract_ms", 0.0))
    ifft_ms = float(timing.get("ifft_ms", 0.0))
    reconstruct_ms = float(timing.get("reconstruct_ms", 0.0))
    tt_core_ms = float(timing.get("tt_core_ms", 0.0))
    tt_fact_ms = float(timing.get("tt_fact_ms", 0.0))
    total_ms = fft_ms + contract_ms + ifft_ms + reconstruct_ms
    rows.append(
        {
            "run": run_idx,
            "config": cfg_label,
            "layer": layer_label,
            "fft_ms": fft_ms,
            "contract_ms": contract_ms,
            "ifft_ms": ifft_ms,
            "reconstruct_ms": reconstruct_ms,
            "tt_core_ms": tt_core_ms,
            "tt_fact_ms": tt_fact_ms,
            "total_ms": float(total_ms),
            "total_wall_ms": float(total_wall_ms),
        }
    )


def main() -> None:
    torch.manual_seed(0)
    if device_pref == "cuda":
        device = torch.device("cuda")
    elif device_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    RUNS = runs
    WARMUP = warmup

    rows: list[dict] = []

    for cfg_label, width, spatial, rank in TEST_CONFIGS:
        in_channels = width
        out_channels = width
        # For rfftn, the last dimension is N/2+1; keep modes in-bounds to avoid shape mismatches.
        modes1 = spatial
        modes2 = spatial
        modes3 = min(spatial, spatial // 2 + 1)

        x = torch.randn(batch_size, in_channels, spatial, spatial, spatial, device=device)

        # Dense path (no factorization)
        layer_dense = FactorizedSpectralConv3d(
            in_channels,
            out_channels,
            modes1,
            modes2,
            modes3,
            factorization=None,
            rank=rank,
            implementation="factorized",
            fft_norm="forward",
            timing=True,
        ).to(device)

        # TT cores path (single-einsum, ParameterList cores)
        layer_tt_core = FactorizedSpectralConv3d(
            in_channels,
            out_channels,
            modes1,
            modes2,
            modes3,
            factorization="tt",
            rank=rank,
            implementation="factorized",
            fft_norm="forward",
            timing=True,
        ).to(device)

        # TT FactorizedTensor path (tltorch-style factors, tl.einsum over factors)
        layer_tt_fact = FactorizedSpectralConv3d(
            in_channels,
            out_channels,
            modes1,
            modes2,
            modes3,
            factorization="tt_factorized",
            rank=rank,
            implementation="factorized",
            fft_norm="forward",
            timing=True,
        ).to(device)

        print(
            f"\n=== Config {cfg_label}: width={width}, spatial={spatial}, rank={rank}, "
            f"modes=({modes1},{modes2},{modes3}) ==="
        )

        for run_idx in range(RUNS + WARMUP):
            is_warmup = run_idx < WARMUP
            label = "(warmup) " if is_warmup else ""
            print(f"\nRun {run_idx} {label}- dense…")
            run_once(cfg_label, "dense", layer_dense, x, rows, run_idx)

            print(f"Run {run_idx} {label}- TT cores (single-einsum)…")
            run_once(cfg_label, "tt_core", layer_tt_core, x, rows, run_idx)

            print(f"Run {run_idx} {label}- TT factorized (factors einsum)…")
            run_once(cfg_label, "tt_fact", layer_tt_fact, x, rows, run_idx)

    # Save per-run timings and summary averages (excluding warmup)
    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True)
    fieldnames = [
        "run",
        "config",
        "layer",
        "fft_ms",
        "contract_ms",
        "ifft_ms",
        "reconstruct_ms",
        "tt_core_ms",
        "tt_fact_ms",
        "total_ms",
        "total_wall_ms",
    ]

    tag_base = f"d{device.type}_b{batch_size}_r{runs}_w{warmup}"
    runs_csv = out_dir / f"local_tt_compare_runs_{tag_base}.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Compute averages excluding warmup runs per config
    summary = []
    for cfg_label, width, spatial, rank in TEST_CONFIGS:
        for layer_name in ("dense", "tt_core", "tt_fact"):
            layer_rows = [
                r
                for r in rows
                if r["config"] == cfg_label and r["layer"] == layer_name and r["run"] >= WARMUP
            ]
            if not layer_rows:
                continue
            avg = {k: 0.0 for k in fieldnames if k not in ("run", "layer", "config")}
            for r in layer_rows:
                for k in avg:
                    avg[k] += float(r.get(k, 0.0))
            count = len(layer_rows)
            for k in avg:
                avg[k] /= count
            avg_row = {"config": cfg_label, "layer": layer_name, **avg}
            summary.append(avg_row)

    summary_fieldnames = [
        "config",
        "layer",
        "fft_ms",
        "contract_ms",
        "ifft_ms",
        "reconstruct_ms",
        "tt_core_ms",
        "tt_fact_ms",
        "total_ms",
        "total_wall_ms",
    ]
    summary_csv = out_dir / f"local_tt_compare_summary_{tag_base}.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved per-run timings to {runs_csv}")
    print(f"Saved averages (excluding {WARMUP} warmup run) to {summary_csv}")

    # Report speedup factors vs dense using contract_ms (mean timings) per config
    for cfg_label, _, _, _ in TEST_CONFIGS:
        dense_row = next((r for r in summary if r["config"] == cfg_label and r["layer"] == "dense"), None)
        if not dense_row:
            continue
        dense_contract = dense_row.get("contract_ms", 0.0)
        print(f"\nSpeedups for config {cfg_label} (contract_ms):")
        for name in ("tt_core", "tt_fact"):
            row = next((r for r in summary if r["config"] == cfg_label and r["layer"] == name), None)
            if row and dense_contract > 0:
                speed = dense_contract / row.get("contract_ms", dense_contract)
                print(f"  {name}: {speed:.2f}x")


if __name__ == "__main__":
    main()
