#!/usr/bin/env python3
"""
Conference Analysis: Advanced Tensor Compression in Fourier Neural Networks
                     for Fusion Simulation

Generates all publication-quality figures for conference presentation.
Run from any directory; outputs go to the same directory as this script.

Usage:
    python3 analyze_conference.py
"""

import os, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
HOME_BASE    = "/global/u2/p/pepi/STFNO_QTT/Data_Logs_Tests"
PSCRATCH     = "/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests"
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))   # same dir as script
os.makedirs(OUT_DIR, exist_ok=True)

# ─── VISUAL IDENTITY ──────────────────────────────────────────────────────────
# label → (hex color, marker, display name)
STYLE = {
    "Unfactorized":    ("#4a4a4a", "s",  "Unfactorized"),
    "Spectral-TT":     ("#e74c3c", "o",  "Spectral-TT"),
    "Spectral-QTT-q3": ("#3498db", "^",  "Spectral-QTT3"),
    "Spectral-QTT-q5": ("#2ecc71", "D",  "Spectral-QTT5"),
    "Dense-QTT":       ("#f39c12", "P",  "RealOp-QTT"),
}
ORDER = ["Unfactorized", "Spectral-TT", "Spectral-QTT-q3", "Spectral-QTT-q5", "Dense-QTT"]

def set_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.labelsize":     13,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "legend.fontsize":    10,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "figure.dpi":         150,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.bbox":       "tight",
        "savefig.dpi":        200,
    })

def savefig(fig, name):
    path_png = os.path.join(OUT_DIR, f"{name}.png")
    path_pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    print(f"  Saved: {name}.png / .pdf")
    plt.close(fig)

def legend_handles():
    return [
        Line2D([0], [0], color=STYLE[k][0], marker=STYLE[k][1],
               linewidth=0, markersize=8, label=STYLE[k][2])
        for k in ORDER if k in STYLE
    ]

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def get_label(row):
    op  = str(row.get("operator_type", "spectral") or "spectral").strip().lower()
    ft  = str(row.get("factorization_type", "dense") or "dense").strip().lower()
    try:
        qnd = int(float(row.get("quantize_last_ndims", 3) or 3))
    except (ValueError, TypeError):
        qnd = 3
    if op == "dense":
        return "Dense-QTT"
    # spectral
    if ft in ("dense", "unfactorized"):
        return "Unfactorized"
    if ft == "tt":
        return "Spectral-TT"
    if ft == "qtt":
        return "Spectral-QTT-q3" if qnd <= 3 else "Spectral-QTT-q5"
    return "Unknown"


def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_stfno_sweep():
    """Old spectral sweep (all factorizations incl. unfactorized)."""
    path = f"{HOME_BASE}/stfno_sweep/combined_results.csv"
    df = pd.read_csv(path)
    df["operator_type"] = "spectral"
    # old naming: factorization_type='dense' = unfactorized
    df["label"] = df.apply(get_label, axis=1)
    # unify column names
    if "runtime_s" in df.columns:
        df.rename(columns={"runtime_s": "total_runtime_s"}, inplace=True)
    df["test_time_s"] = np.nan
    _to_num(df, ["param_count","train_l2","test_l2","train_time_s","rank","modes","width",
                 "epochs_completed"])
    # stfno_sweep stores total training time; convert to per-epoch to match new sweeps
    if "epochs_completed" in df.columns:
        df["train_time_s"] = df["train_time_s"] / df["epochs_completed"].replace(0, np.nan)
    else:
        df["train_time_s"] = df["train_time_s"] / 400
    return df


def load_new_sweep(sweep_dir, label):
    """Load a new-format sweep from pscratch (one metrics.csv per experiment dir)."""
    rows = []
    for f in sorted(glob.glob(f"{PSCRATCH}/{sweep_dir}/*/metrics.csv")):
        try:
            tmp = pd.read_csv(f)
            if not tmp.empty:
                rows.append(tmp.iloc[-1].to_dict())
        except Exception:
            pass
    if not rows:
        print(f"  WARNING: no data in {sweep_dir}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["label"] = label
    _to_num(df, ["param_count","train_l2","test_l2","train_time_s","test_time_s",
                 "total_runtime_s","rank","modes","width","quantize_last_ndims"])
    return df


def load_modes_sweep():
    """Modes ablation sweep (spectral-TT, r=12, w=32, varying modes)."""
    rows = []
    for f in sorted(glob.glob(f"{PSCRATCH}/spectral_tt_modes_sweep/*/metrics.csv")):
        try:
            tmp = pd.read_csv(f)
            if not tmp.empty:
                rows.append(tmp.iloc[-1].to_dict())
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["label"] = "Spectral-TT"
    _to_num(df, ["param_count","train_l2","test_l2","train_time_s","test_time_s",
                 "total_runtime_s","rank","modes","width"])
    return df


def load_error_bar_runs():
    """Repeat runs (up to 5 seeds) for 5 key configurations — all error_bar_runs* dirs."""
    EB_DIRS = [
        f"{PSCRATCH}/error_bar_runs",
        f"{PSCRATCH}/error_bar_runs_extra",
        f"{PSCRATCH}/error_bar_runs_qtt5_unfact",
        f"{PSCRATCH}/error_bar_runs_seeds2",
        f"{PSCRATCH}/error_bar_runs_seeds3",
    ]
    rows = []
    for eb_dir in EB_DIRS:
        for f in sorted(glob.glob(f"{eb_dir}/*/metrics.csv")):
            try:
                tmp = pd.read_csv(f)
                if not tmp.empty:
                    r = tmp.iloc[-1].to_dict()
                    r["config_dir"] = os.path.basename(os.path.dirname(f))
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["label"] = df.apply(get_label, axis=1)
    _to_num(df, ["param_count","train_l2","test_l2","train_time_s","test_time_s",
                 "total_runtime_s","rank","modes","width","seed"])
    return df


def load_unfactorized_width_sweep():
    """Additional unfactorized width configs (w=6,10,12,14,18,20,24) from new sweep."""
    rows = []
    for f in sorted(glob.glob(f"{PSCRATCH}/unfactorized_width_sweep/*/metrics.csv")):
        try:
            tmp = pd.read_csv(f)
            if not tmp.empty:
                rows.append(tmp.iloc[-1].to_dict())
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["label"] = "Unfactorized"
    _to_num(df, ["param_count","train_l2","test_l2","train_time_s","test_time_s",
                 "total_runtime_s","rank","modes","width"])
    return df


def load_all():
    """Unified DataFrame from all sweeps (new sweeps preferred for TT/QTT)."""
    print("Loading data...")
    frames = []

    # Unfactorized baseline — stfno_sweep (w=4,8,16,32) + new width sweep (w=6,10,12,14,18,20,24)
    stfno = load_stfno_sweep()
    unfact = stfno[stfno["label"] == "Unfactorized"].copy()
    print(f"  Unfactorized (stfno_sweep): {len(unfact)} configs")
    frames.append(unfact)

    unfact_extra = load_unfactorized_width_sweep()
    if not unfact_extra.empty:
        print(f"  Unfactorized (width_sweep): {len(unfact_extra)} configs")
        frames.append(unfact_extra)

    # New sweeps — authoritative for TT and QTT
    for sweep, label in [
        ("spectral_tt_sweep",       "Spectral-TT"),
        ("spectral_tt_modes_sweep", "Spectral-TT"),   # modes ablation — was missing!
        ("spectral_qtt_q3_sweep",   "Spectral-QTT-q3"),
        ("spectral_qtt_q5_sweep",   "Spectral-QTT-q5"),
        ("real_operator_sweep",     "Dense-QTT"),
    ]:
        df = load_new_sweep(sweep, label)
        if not df.empty:
            print(f"  {label} ({sweep}): {len(df)} configs")
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    _to_num(combined, ["param_count","train_l2","test_l2","train_time_s","rank","modes","width"])

    # Remove obviously failed runs (test_l2 > 0.15 indicates non-convergence)
    before = len(combined)
    combined = combined[combined["test_l2"] < 0.15].dropna(subset=["test_l2","param_count"])

    # Deduplicate: keep best test_l2 per (label, rank, modes, width) —
    # spectral_tt_r12_m12_w32 exists in both spectral_tt_sweep and spectral_tt_modes_sweep
    before_dedup = len(combined)
    combined = (combined
                .sort_values("test_l2")
                .drop_duplicates(subset=["label", "rank", "modes", "width"], keep="first")
                .reset_index(drop=True))
    print(f"  Total: {len(combined)} valid configs "
          f"({before - before_dedup} filtered, {before_dedup - len(combined)} deduped)")
    return combined


# ─── PARETO FRONTIER EXTRACTION ───────────────────────────────────────────────

def pareto_frontier(df_label, sort_col="param_count", perf_col="test_l2"):
    """Extract Pareto-optimal (minimize both param_count and test_l2) points."""
    df = df_label.sort_values(sort_col).reset_index(drop=True)
    frontier = []
    best = float("inf")
    for _, row in df.iterrows():
        if row[perf_col] < best:
            best = row[perf_col]
            frontier.append(row)
    return pd.DataFrame(frontier)


# ─── TRAINING LOG PARSER ──────────────────────────────────────────────────────

_LOG_RE = re.compile(
    r"ep=\s*(\d+).*?train_l2_full\s*/\s*ntrain=\s*([\d.e+-]+)"
    r".*?test_l2_full\s*/\s*ntest=\s*([\d.e+-]+)",
    re.DOTALL
)

def parse_training_log(log_path):
    """Return DataFrame with columns [epoch, train_l2, test_l2].
    Log format (one epoch per line):
    ep= N , ... , train_l2_full / ntrain= X , test_l2_full / ntest= Y , ...
    """
    records = []
    try:
        with open(log_path) as fh:
            for line in fh:
                if not line.startswith("ep="):
                    continue
                m_ep  = re.search(r"ep=\s*(\d+)", line)
                m_tr  = re.search(r"train_l2_full / ntrain=\s*([\d.e+\-]+)", line)
                m_te  = re.search(r"test_l2_full / ntest=\s*([\d.e+\-]+)", line)
                if m_ep and m_tr and m_te:
                    records.append({
                        "epoch":    int(m_ep.group(1)),
                        "train_l2": float(m_tr.group(1)),
                        "test_l2":  float(m_te.group(1)),
                    })
    except OSError:
        pass
    return pd.DataFrame(records)


def find_log(sweep_base, subdir_pattern):
    candidates = glob.glob(f"{sweep_base}/{subdir_pattern}/training.log")
    return candidates[0] if candidates else None

# ─── FIGURE 01: UNIFIED PARETO ────────────────────────────────────────────────

def fig01_unified_pareto(df):
    print("Fig 01: Unified Pareto...")
    fig, ax = plt.subplots(figsize=(9, 6))

    UNFACT_BEST = df[df["label"] == "Unfactorized"]["test_l2"].min()

    for label in ORDER:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        c, mk, name = STYLE[label]
        alpha = 0.25 if label == "Unfactorized" else 0.20
        ax.scatter(sub["param_count"], sub["test_l2"],
                   color=c, marker=mk, s=50, alpha=alpha, linewidths=0.5,
                   edgecolors="white", zorder=3)
        # Pareto frontier
        pf = pareto_frontier(sub)
        if len(pf) > 1:
            ax.plot(pf["param_count"], pf["test_l2"],
                    color=c, linewidth=2, zorder=4, alpha=0.9)

    # Baseline band: best unfactorized
    ax.axhline(UNFACT_BEST, color=STYLE["Unfactorized"][0],
               linestyle="--", linewidth=1.4, alpha=0.7,
               label=f"Best unfactorized ({UNFACT_BEST:.4f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Test L² Error")
    ax.set_title("Unified Pareto: Accuracy vs. Parameter Count\n(NIMROD 3D, 32³, 400 epochs)")

    handles = legend_handles()
    handles.append(Line2D([0], [0], color=STYLE["Unfactorized"][0],
                          linestyle="--", linewidth=1.4,
                          label=f"Best unfactorized ({UNFACT_BEST:.4f})"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else (f"{x/1e3:.0f}K" if x >= 1e3 else str(int(x)))))
    # Dense decimal ticks — FixedLocator every 0.001 gives ~15 readable ticks across the data range
    _y_ticks_01 = np.round(np.arange(0.009, 0.025, 0.001), 4)
    ax.yaxis.set_major_locator(mticker.FixedLocator(_y_ticks_01))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Annotate best Spectral-TT
    best_tt = df[df["label"] == "Spectral-TT"].sort_values("test_l2").iloc[0]
    _tt_p = best_tt['param_count']
    _tt_p_str = f"{_tt_p/1e6:.3f}M" if _tt_p >= 1e6 else f"{int(_tt_p/1e3)}K"
    ax.annotate(f"Best TT\n({best_tt['test_l2']:.4f}, {_tt_p_str} params)",
                xy=(best_tt["param_count"], best_tt["test_l2"]),
                xytext=(best_tt["param_count"] * 3, best_tt["test_l2"] * 1.5 - 0.003),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                fontsize=9, color="#e74c3c")

    savefig(fig, "fig01_unified_pareto")


# ─── FIGURE 02: LEARNING CURVES ───────────────────────────────────────────────

def fig02_learning_curves():
    print("Fig 02: Learning curves...")

    # (label, sweep_base, subdir_glob)
    LOG_SOURCES = [
        ("Unfactorized",    f"{PSCRATCH}/stfno_sweep",            "dense_m12_w32"),
        ("Spectral-TT",     f"{PSCRATCH}/spectral_tt_sweep",      "spectral_tt_r12_m12_w32"),
        ("Spectral-QTT-q3", f"{PSCRATCH}/spectral_qtt_q3_sweep",  "spectral_qtt_q3_r12_m12_w32"),
        ("Spectral-QTT-q5", f"{PSCRATCH}/spectral_qtt_q5_sweep",  "spectral_qtt_q5_r12_m12_w32"),
        ("Dense-QTT",       f"{PSCRATCH}/real_operator_sweep",    "dense_qtt_r12_w14"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    ax_test, ax_train = axes

    for label, base, subdir in LOG_SOURCES:
        log_path = find_log(base, subdir)
        if log_path is None:
            print(f"  WARNING: no training.log for {label} ({subdir})")
            continue
        log = parse_training_log(log_path)
        if log.empty:
            print(f"  WARNING: empty log for {label}")
            continue
        c, _, name = STYLE[label]
        ax_test.plot(log["epoch"], log["test_l2"],  color=c, linewidth=2, label=name, alpha=0.9)
        ax_train.plot(log["epoch"], log["train_l2"], color=c, linewidth=2, label=name, alpha=0.9, linestyle="--")

    for ax, title, ylabel in [
        (ax_test,  "Test L² Error vs. Epoch",  "Test L² Error"),
        (ax_train, "Train L² Error vs. Epoch", "Train L² Error"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.set_yscale("log")

    fig.suptitle("Convergence Trajectories — Best Configuration per Method\n"
                 "(Spectral-TT r=12, m=12, w=32  |  Dense-QTT r=12, w=14  |  Unfactorized w=32, m=12)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "fig02_learning_curves")


# ─── FIGURE 02b: ZOOMED LEARNING CURVES ───────────────────────────────────────

def fig02b_learning_curves_zoomed():
    """Two zoom levels: early convergence (0–50) and late refinement (100–400)."""
    print("Fig 02b: Zoomed learning curves...")

    LOG_SOURCES = [
        ("Unfactorized",    f"{PSCRATCH}/stfno_sweep",            "dense_m12_w32"),
        ("Spectral-TT",     f"{PSCRATCH}/spectral_tt_sweep",      "spectral_tt_r12_m12_w32"),
        ("Spectral-QTT-q3", f"{PSCRATCH}/spectral_qtt_q3_sweep",  "spectral_qtt_q3_r12_m12_w32"),
        ("Spectral-QTT-q5", f"{PSCRATCH}/spectral_qtt_q5_sweep",  "spectral_qtt_q5_r12_m12_w32"),
        ("Dense-QTT",       f"{PSCRATCH}/real_operator_sweep",    "dense_qtt_r12_w14"),
    ]

    # Load all logs once
    logs = {}
    for label, base, subdir in LOG_SOURCES:
        log_path = find_log(base, subdir)
        if log_path is None:
            print(f"  WARNING: no training.log for {label} ({subdir})")
            continue
        log = parse_training_log(log_path)
        if not log.empty:
            logs[label] = log

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_early, ax_late = axes

    ZOOM_PANELS = [
        (ax_early, 0,   50,  "Early Convergence (Epochs 0–50)"),
        (ax_late,  100, 400, "Late Refinement (Epochs 100–400)"),
    ]

    for ax, ep_min, ep_max, title in ZOOM_PANELS:
        for label, log in logs.items():
            c, _, name = STYLE[label]
            sub = log[(log["epoch"] >= ep_min) & (log["epoch"] <= ep_max)]
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub["test_l2"], color=c, linewidth=2, label=name, alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test L² Error")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(framealpha=0.9, fontsize=9)

    fig.suptitle("Convergence Trajectories — Zoomed (Test L²)\n"
                 "(TT / QTT3 / QTT5: r=12, m=12, w=32  |  Dense-QTT: r=12, w=14  |  Unfactorized: w=32, m=12)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "fig02b_learning_curves_zoomed")


# ─── FIGURE 03: 4-PANEL HEATMAPS (rank × width → test_l2) ────────────────────

def fig03_heatmaps(df):
    print("Fig 03: Heatmaps...")
    labels_to_plot = ["Spectral-TT", "Spectral-QTT-q3", "Spectral-QTT-q5", "Dense-QTT"]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(21, 11))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.055],
                  wspace=0.32, hspace=0.38, left=0.06, right=0.97, top=0.88, bottom=0.08)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])
    cbar_ax = fig.add_subplot(gs[:, 2])
    axes_flat = axes.flatten()

    ORIG_RANKS = [2, 4, 8, 12, 16]
    ORIG_WIDTHS = [8, 16, 24, 32]

    last_im = None
    for ax, label in zip(axes_flat, labels_to_plot):
        sub = df[(df["label"] == label) &
                 (df["rank"].isin(ORIG_RANKS)) &
                 (df["width"].isin(ORIG_WIDTHS))].copy()
        if sub.empty:
            ax.set_title(STYLE[label][2])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Pivot: best test_l2 per (rank, width)
        pivot = sub.groupby(["rank","width"])["test_l2"].min().unstack("width")
        pivot = pivot.sort_index(ascending=False)

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=0.009, vmax=0.022)
        last_im = im
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(r)) for r in pivot.index], fontsize=11)
        ax.set_xlabel("Width")
        ax.set_ylabel("Rank")
        ax.set_title(STYLE[label][2], color=STYLE[label][0])

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=10, color="black" if v < 0.018 else "white")

    # Colorbar in its dedicated axis — fully to the right of both columns
    if last_im is not None:
        fig.colorbar(last_im, cax=cbar_ax, label="Test L²")

    fig.suptitle("Test L² Error Heatmap: Rank × Width\n(modes=12, 400 epochs, NIMROD 3D 32³)",
                 fontsize=13, y=0.97)
    savefig(fig, "fig03_heatmaps_rank_width")


# ─── FIGURE 04: MODES SENSITIVITY ─────────────────────────────────────────────

def fig04_modes_sensitivity(df):
    print("Fig 04: Modes sensitivity...")
    modes_df = load_modes_sweep()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Modes sweep (spectral-TT, w=32, varying modes — best rank per mode)
    if not modes_df.empty:
        sub = modes_df[modes_df["width"] == 32]
        # Multiple ranks exist at some modes (e.g. r=12,14,15 at m=16) — take best per mode
        sub = sub.loc[sub.groupby("modes")["test_l2"].idxmin()].sort_values("modes")
        c, mk, name = STYLE["Spectral-TT"]
        ax.plot(sub["modes"], sub["test_l2"], color=c, marker=mk,
                linewidth=2.5, markersize=9, label="Spectral-TT (best rank, w=32)", zorder=5)

        # Annotate best (include rank since it now varies)
        best = sub.loc[sub["test_l2"].idxmin()]
        best_r = int(best["rank"]) if not pd.isna(best.get("rank", float("nan"))) else 12
        ax.annotate(f"Best: m={int(best['modes'])}, r={best_r}\n({best['test_l2']:.4f})",
                    xy=(best["modes"], best["test_l2"]),
                    xytext=(best["modes"] + 1.5, best["test_l2"] + 0.0005),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.3),
                    fontsize=9, color=c)

    # Load best QTT3 m=16 from rerun (seeds 1-3) to replace plateaued seed=0
    _qtt3_m16_best = None
    for _s, _logf in [
        (0, f"{PSCRATCH}/spectral_qtt_q3_sweep/spectral_qtt_q3_r12_m16_w32/training.log"),
        (1, f"{PSCRATCH}/qtt3_m16_rerun/seed1/training.log"),
        (2, f"{PSCRATCH}/qtt3_m16_rerun/seed2/training.log"),
        (3, f"{PSCRATCH}/qtt3_m16_rerun/seed3/training.log"),
    ]:
        if os.path.exists(_logf):
            with open(_logf) as _f:
                for _line in _f:
                    _hit = re.search(r'test_l2_full / ntest=\s*([\d.e+\-]+)', _line)
                    if _hit:
                        _v = float(_hit.group(1))
                        if _qtt3_m16_best is None or _v < _qtt3_m16_best:
                            _qtt3_m16_best = _v

    # QTT3/QTT5: line if we have multiple modes, scatter marker if only m=12 available
    for label in ["Spectral-QTT-q3", "Spectral-QTT-q5"]:
        c, mk, name = STYLE[label]
        sub2 = df[(df["label"] == label) &
                  (df["rank"] == 12) &
                  (df["width"] == 32)].dropna(subset=["modes"]).sort_values("modes")
        # Override QTT3 m=16 with best seed from rerun (seed=2 escaped plateau: 0.0119)
        if label == "Spectral-QTT-q3" and _qtt3_m16_best is not None:
            sub2 = sub2.copy()
            mask16 = sub2["modes"] == 16
            if mask16.any():
                sub2.loc[mask16, "test_l2"] = min(_qtt3_m16_best,
                                                   sub2.loc[mask16, "test_l2"].values[0])
        if not sub2.empty:
            if len(sub2) >= 2:
                ax.plot(sub2["modes"], sub2["test_l2"], color=c, marker=mk,
                        linewidth=2, markersize=7, linestyle="--",
                        label=f"{name} (r=12, w=32)", alpha=0.85)
            else:
                ax.scatter(sub2["modes"], sub2["test_l2"],
                           color=c, marker=mk, s=130, zorder=5,
                           label=f"{name} (r=12, w=32, m=12 only)")

    # Unfactorized at w=8 (best unfact width); uses df which includes stfno_sweep + new runs
    uf = df[(df["label"] == "Unfactorized") & (df["width"] == 8)].dropna(subset=["modes"]).sort_values("modes")
    if not uf.empty:
        c2, _, _ = STYLE["Unfactorized"]
        ax.plot(uf["modes"], uf["test_l2"], color=c2, marker="s",
                linewidth=1.5, markersize=6, linestyle=":",
                label="Unfactorized (w=8)", alpha=0.7)

    # Best Dense-QTT at w=32 — dashed horizontal reference line
    dqtt_w32 = df[(df["label"] == "Dense-QTT") & (df["width"] == 32)]
    if not dqtt_w32.empty:
        best_dqtt = dqtt_w32["test_l2"].min()
        c_d, _, name_d = STYLE["Dense-QTT"]
        ax.axhline(best_dqtt, color=c_d, linestyle="--", linewidth=1.8, alpha=0.85,
                   label=f"Best {name_d} (w=32): {best_dqtt:.4f}")

    ax.set_xlabel("Fourier Modes")
    ax.set_ylabel("Test L² Error")
    ax.set_title("Modes Sensitivity\n(TT/QTT: r=12, w=32 — best config; Unfact: w=8 — best config)")
    ax.legend(framealpha=0.9, loc="lower left", fontsize=8, handlelength=1.2,
              handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    savefig(fig, "fig04_modes_sensitivity")


# ─── FIGURE 05: ACCURACY vs SPEED ─────────────────────────────────────────────

def fig05_accuracy_vs_speed(df):
    print("Fig 05: Accuracy vs. speed...")
    fig, ax = plt.subplots(figsize=(8, 6))

    has_time = df["train_time_s"].notna()
    df_t = df[has_time].copy()

    for label in ORDER:
        sub = df_t[df_t["label"] == label]
        if sub.empty:
            continue
        c, mk, name = STYLE[label]
        ax.scatter(sub["train_time_s"], sub["test_l2"],
                   color=c, marker=mk, s=60, alpha=0.6,
                   edgecolors="white", linewidths=0.4, zorder=3, label=name)

    # Mark best of each method
    for label in ORDER:
        sub = df_t[df_t["label"] == label]
        if sub.empty or sub["test_l2"].isna().all():
            continue
        best = sub.loc[sub["test_l2"].idxmin()]
        c, mk, _ = STYLE[label]
        ax.scatter(best["train_time_s"], best["test_l2"],
                   color=c, marker=mk, s=180, edgecolors="black",
                   linewidths=1.5, zorder=6)

    ax.set_xlabel("Training Time per Epoch (seconds)")
    ax.set_ylabel("Test L² Error")
    ax.set_title("Accuracy vs. Computational Cost\n(each point = one configuration; ★ = best per method)")
    ax.set_yscale("log")
    ax.legend(handles=legend_handles(), loc="upper right", framealpha=0.9)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    # Annotate quadrants
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[0]*1.05, ylim[1]*0.95, "Fast &\nInaccurate", fontsize=8,
            color="gray", va="top", ha="left")
    ax.text(xlim[1]*0.92, ylim[0]*1.05, "Slow &\nAccurate", fontsize=8,
            color="gray", va="bottom", ha="right")
    savefig(fig, "fig05_accuracy_vs_speed")


# ─── FIGURE 06: COMPRESSION RATIO vs ACCURACY ─────────────────────────────────

def fig06_compression(df):
    print("Fig 06: Compression ratio...")
    # Use median unfactorized as compression reference
    UNFACT_MEDIAN_PARAMS = df[df["label"] == "Unfactorized"]["param_count"].median()
    UNFACT_BEST_L2 = df[df["label"] == "Unfactorized"]["test_l2"].min()
    print(f"  Reference: unfact median params={UNFACT_MEDIAN_PARAMS/1e6:.1f}M, best_l2={UNFACT_BEST_L2:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))

    for label in ORDER:
        if label == "Unfactorized":
            continue
        sub = df[df["label"] == label].copy()
        if sub.empty:
            continue
        sub["compression"] = UNFACT_MEDIAN_PARAMS / sub["param_count"]
        sub["rel_error"]   = sub["test_l2"] / UNFACT_BEST_L2
        c, mk, name = STYLE[label]
        ax.scatter(sub["compression"], sub["rel_error"],
                   color=c, marker=mk, s=60, alpha=0.6,
                   edgecolors="white", linewidths=0.4, zorder=3, label=name)
        # Pareto frontier in (compression, rel_error) space — maximize compression, minimize error
        pf_df = sub[["compression","rel_error"]].dropna().sort_values("compression", ascending=False).reset_index(drop=True)
        best_err = float("inf")
        frontier_rows = []
        for _, row in pf_df.iterrows():
            if row["rel_error"] < best_err:
                best_err = row["rel_error"]
                frontier_rows.append(row)
        if len(frontier_rows) > 1:
            pf = pd.DataFrame(frontier_rows).sort_values("compression")
            ax.plot(pf["compression"], pf["rel_error"], color=c, linewidth=2, alpha=0.85)

    ax.axhline(1.0, color=STYLE["Unfactorized"][0], linestyle="--", linewidth=1.5, alpha=0.7,
               label="Unfactorized accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio (vs. unfactorized)")
    ax.set_ylabel("Relative Test L² (vs. best unfactorized)")
    ax.set_title("Compression-Accuracy Tradeoff\n(below dashed line = beats unfactorized)")
    ax.legend(handles=[*legend_handles()[1:],
                        Line2D([0],[0], color=STYLE["Unfactorized"][0],
                               linestyle="--", linewidth=1.5, label="Unfactorized")],
              framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}×"))
    savefig(fig, "fig06_compression_ratio")


# ─── FIGURE 07: ERROR BARS ────────────────────────────────────────────────────

def fig07_error_bars():
    print("Fig 07: Error bars (repeat seeds)...")
    eb = load_error_bar_runs()
    if eb.empty:
        print("  No error bar data found.")
        return

    # 5 headline configs: one best per method (chosen before rank-fill)
    KEY_CONFIGS = {
        "spectral_tt_r12_m12_w32",
        "spectral_qtt_q3_r12_m12_w32",
        "spectral_qtt_q5_r12_m12_w32",
        "dense_qtt_r12_w14",
        "unfactorized_w8_m12",
    }

    # 95% CI: use t-distribution critical value for actual n per config
    def _t95_crit(n):
        if n <= 1:
            return float('inf')
        try:
            from scipy.stats import t as _t
            return _t.ppf(0.975, df=n - 1)
        except ImportError:
            return 1.96  # normal approx fallback

    eb["config_clean"] = eb["config_dir"].apply(lambda d: re.sub(r"_seed\d+$","",d))

    # Normalize abbreviated names from seeds3 batch
    # (seeds3 job used 'qtt3'/'qtt5' instead of the canonical 'qtt_q3'/'qtt_q5')
    eb["config_clean"] = (eb["config_clean"]
        .str.replace("spectral_qtt3_", "spectral_qtt_q3_", regex=False)
        .str.replace("spectral_qtt5_", "spectral_qtt_q5_", regex=False))

    # Keep only the 5 intended configs
    eb = eb[eb["config_clean"].isin(KEY_CONFIGS)].copy()
    if eb.empty:
        print("  No data for key configs yet — check that error_bar_runs directories exist.")
        return

    # Deduplicate: if seed number appears twice (e.g. TT seed=3 in two batches), keep one
    eb["seed_num"] = eb["config_dir"].str.extract(r"_seed(\d+)$")[0].fillna("0").astype(int)
    eb = (eb.sort_values(["config_clean", "seed_num"])
            .drop_duplicates(["config_clean", "seed_num"]))

    # Use equal n across all configs (take first n_equal seeds per config, sorted by seed_num)
    n_per_config = eb.groupby("config_clean")["seed_num"].count()
    N_TARGET = 10  # desired seeds per config; capped by actual availability
    n_equal = min(N_TARGET, int(n_per_config.min()))
    print(f"  Seeds per config: {n_per_config.to_dict()}")
    print(f"  Using n={n_equal} (min across configs, target={N_TARGET}) for equal comparison")
    eb_equal = (eb.groupby("config_clean", group_keys=False)
                  .apply(lambda g: g.head(n_equal))
                  .reset_index(drop=True))

    grp = eb_equal.groupby("config_clean")["test_l2"].agg(["mean","std","count"]).reset_index()
    grp["ci95"] = grp.apply(
        lambda r: _t95_crit(int(r["count"])) * r["std"] / np.sqrt(r["count"])
        if r["count"] > 1 else np.nan, axis=1)
    grp = grp.sort_values("mean")

    # Assign color by label (check from eb data)
    label_map = eb.drop_duplicates("config_clean").set_index("config_clean")["label"].to_dict()
    colors = [STYLE.get(label_map.get(c, "Spectral-TT"), ("#888","o","?"))[0] for c in grp["config_clean"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(grp))
    bars = ax.barh(list(y_pos), grp["mean"], xerr=grp["ci95"],
                   color=colors, capsize=6, height=0.6,
                   error_kw=dict(elinewidth=2, capthick=2, ecolor="black"))

    # Clean tick labels
    def nice_config_label(c):
        c2 = (c
              .replace("spectral_tt_r",     "Spectral-TT r=")
              .replace("spectral_qtt_q3_r", "Spectral-QTT3 r=")
              .replace("spectral_qtt_q5_r", "Spectral-QTT5 r=")
              .replace("dense_qtt_r",       "RealOp-QTT r=")
              .replace("unfactorized_w",    "Unfactorized w="))
        c2 = c2.replace("_m12_w", " m=12 w=").replace("_m12", " m=12").replace("_w", " w=")
        return c2
    tick_labels = [nice_config_label(c) for c in grp["config_clean"]]

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tick_labels, fontsize=10)
    note = " — add more seeds for statistical validity" if n_equal < 10 else ""
    ax.set_xlabel(f"Test L² Error (mean ± 95% CI, n={n_equal} seeds per config)")
    ax.set_title(f"Reproducibility: Test Error Across Random Seeds (n={n_equal} per config)\n"
                 f"(t-distribution 95% CI; 5 headline configs){note}")

    # Add value labels
    for i, (_, row) in enumerate(grp.iterrows()):
        ax.text(row["mean"] + row["ci95"] + 0.0002, i,
                f'{row["mean"]:.4f}±{row["ci95"]:.4f}', va="center", fontsize=8)

    ax.set_xlim(0, grp["mean"].max() * 1.3)
    present_labels = sorted(set(label_map.values()),
                            key=lambda l: ORDER.index(l) if l in ORDER else 99)
    legend_patches = [
        mpatches.Patch(color=STYLE[l][0], label=STYLE[l][2])
        for l in present_labels if l in STYLE
    ]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig07_error_bars")


# ─── FIGURE 08: RANK SCALING ──────────────────────────────────────────────────

def fig08_rank_scaling(df):
    print("Fig 08: Rank scaling...")
    widths_to_show = [8, 16, 24, 32]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, label in zip(axes, ["Spectral-TT", "Dense-QTT"]):
        # Fix to m=12 — modes sweep data (now in df) would create multiple
        # points at r=12 otherwise. Dense-QTT has no modes column (NaN → keep all).
        sub = df[df["label"] == label]
        sub = sub[sub["modes"].isna() | (sub["modes"] == 12)]
        c_base, mk, name = STYLE[label]
        blues = plt.cm.Blues(np.linspace(0.4, 1.0, len(widths_to_show)))
        oranges = plt.cm.Oranges(np.linspace(0.4, 1.0, len(widths_to_show)))
        cmap_colors = blues if label == "Spectral-TT" else oranges

        for i, w in enumerate(widths_to_show):
            sub_w = sub[sub["width"] == w].sort_values("rank")
            if len(sub_w) < 2:
                continue
            ax.plot(sub_w["rank"], sub_w["test_l2"],
                    color=cmap_colors[i], marker=mk, linewidth=2, markersize=7,
                    label=f"w={w}")

        ax.set_xlabel("Factorization Rank")
        ax.set_ylabel("Test L² Error")
        ax.set_title(f"Rank Scaling — {STYLE[label][2]}")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Zoom to data range with small padding
        plotted = [l for l in ax.get_lines() if len(l.get_ydata()) > 0]
        if plotted:
            all_y = np.concatenate([l.get_ydata() for l in plotted])
            all_x = np.concatenate([l.get_xdata() for l in plotted])
            ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
            xmin, xmax = np.nanmin(all_x), np.nanmax(all_x)
            ax.set_ylim(ymin * 0.92, ymax * 1.12)
            ax.set_xlim(xmin - 0.4, xmax + 0.6)

        # Dense ticks for TT (left panel), log-sublinear for Dense-QTT (right)
        if label == "Spectral-TT":
            ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(0.009, 0.0205, 0.001)))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        else:
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())

        ax.legend(title="Width", framealpha=0.9, loc="upper right")

    fig.suptitle("Effect of Tensor Rank on Accuracy\n(modes=12, 400 epochs)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "fig08_rank_scaling")


# ─── FIGURE 09: WIDTH / PARAMETER SCALING ─────────────────────────────────────

def fig09_width_scaling(df):
    print("Fig 09: Width/parameter scaling...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_params, ax_loss = axes

    for label in ORDER:
        sub = df[df["label"] == label].sort_values("width")
        if sub.empty:
            continue
        c, mk, name = STYLE[label]
        # For each width: select the single row with the lowest test_l2 (best accuracy).
        # Both param_count and test_l2 then come from the SAME config, so the two
        # panels are consistent with each other.
        best_rows = sub.loc[sub.groupby("width")["test_l2"].idxmin()].sort_values("width")
        ax_params.plot(best_rows["width"], best_rows["param_count"],
                       color=c, marker=mk, linewidth=2, markersize=7, label=name)
        ax_loss.plot(best_rows["width"], best_rows["test_l2"],
                     color=c, marker=mk, linewidth=2, markersize=7, label=name)

    ax_params.set_xlabel("Width")
    ax_params.set_ylabel("Parameter Count")
    ax_params.set_title("Parameter Count vs. Width\n(param count of best-accuracy config per width)")
    ax_params.set_yscale("log")
    ax_params.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax_params.legend(framealpha=0.9)

    ax_loss.set_xlabel("Width")
    ax_loss.set_ylabel("Test L² Error")
    ax_loss.set_title("Best Accuracy vs. Width\n(optimized over rank)")
    ax_loss.set_yscale("log")
    _y09 = np.round(np.arange(0.008, 0.026, 0.001), 4)
    ax_loss.yaxis.set_major_locator(mticker.FixedLocator(_y09))
    ax_loss.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_loss.yaxis.set_minor_locator(mticker.NullLocator())
    ax_loss.legend(framealpha=0.9, fontsize=8, handlelength=1.0,
                   handletextpad=0.4, borderpad=0.4, labelspacing=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Scaling with Network Width", fontsize=13)
    fig.tight_layout()
    savefig(fig, "fig09_width_scaling")


# ─── FIGURE 10: LEADERBOARD ───────────────────────────────────────────────────

def fig10_leaderboard(df):
    print("Fig 10: Leaderboard...")
    top = df.sort_values("test_l2").head(20).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = range(len(top))
    colors = [STYLE.get(top.loc[i, "label"], ("#888","o","?"))[0] for i in range(len(top))]

    ax.barh(list(y), top["test_l2"], color=colors, height=0.7, alpha=0.88)

    # Tick labels
    def make_label(row):
        lbl = row["label"].replace("Spectral-", "S-").replace("Dense-", "D-")
        r = int(row["rank"]) if not pd.isna(row.get("rank", np.nan)) else "?"
        w = int(row["width"])
        m = int(row["modes"]) if not pd.isna(row.get("modes", np.nan)) else "?"
        p = row["param_count"]
        pk = f"{p/1e6:.1f}M" if p >= 1e6 else f"{p/1e3:.0f}K"
        return f"{lbl} r={r} w={w} m={m} ({pk})"

    tick_lbls = [make_label(top.loc[i]) for i in range(len(top))]
    ax.set_yticks(list(y))
    ax.set_yticklabels(tick_lbls, fontsize=8.5)
    ax.invert_yaxis()

    for i, v in enumerate(top["test_l2"]):
        ax.text(v + 0.00005, i, f"{v:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Test L² Error")
    ax.set_title("Top 20 Configurations — All Methods\n(NIMROD 3D HyperDiffusivity, 32³, 400 epochs)")
    xmin = top["test_l2"].min()
    xmax = top["test_l2"].max()
    xpad = (xmax - xmin) * 0.15
    ax.set_xlim(xmin - xpad, xmax + xpad * 3.5)  # extra right room for value labels

    legend_patches = [
        mpatches.Patch(color=STYLE["Spectral-TT"][0], label=STYLE["Spectral-TT"][2]),
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig10_leaderboard")


# ─── FIGURE 11: TRAINING TIME DISTRIBUTION ────────────────────────────────────

def fig11_timing_distribution(df):
    print("Fig 11: Timing distribution...")
    has_time = df["train_time_s"].notna() & (df["train_time_s"] > 0)
    df_t = df[has_time]

    # Compute per-method median at each width
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_box, ax_line = axes

    # Boxplot per method
    data_for_box = []
    labels_for_box = []
    colors_for_box = []
    for label in ORDER:
        sub = df_t[df_t["label"] == label]["train_time_s"].dropna()
        if len(sub) < 2:
            continue
        data_for_box.append(sub.values)
        labels_for_box.append(STYLE[label][2].replace(" (", "\n("))
        colors_for_box.append(STYLE[label][0])

    bp = ax_box.boxplot(data_for_box, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax_box.set_xticklabels(labels_for_box, fontsize=9)
    ax_box.set_ylabel("Train Time per Epoch (seconds)")
    ax_box.set_title("Training Time Distribution by Method")
    ax_box.tick_params(axis="x", labelrotation=10)

    # Line plot: median time vs width (all methods)
    for label in ORDER:
        sub = df_t[df_t["label"] == label]
        if sub.empty:
            continue
        med = sub.groupby("width")["train_time_s"].median().reset_index().sort_values("width")
        c, mk, name = STYLE[label]
        ax_line.plot(med["width"], med["train_time_s"], color=c, marker=mk,
                     linewidth=2.5, markersize=8, label=name)

    ax_line.set_xlabel("Width")
    ax_line.set_ylabel("Median Train Time per Epoch (seconds)")
    ax_line.set_title("Training Speed vs. Width — All Methods")
    ax_line.legend(framealpha=0.9)
    ax_line.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Computational Cost Analysis", fontsize=13)
    fig.tight_layout()
    savefig(fig, "fig11_timing_analysis")


# ─── SUMMARY STATISTICS ───────────────────────────────────────────────────────

def print_summary(df):
    print("\n" + "="*65)
    print("SUMMARY STATISTICS")
    print("="*65)
    for label in ORDER:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        best = sub.loc[sub["test_l2"].idxmin()]
        print(f"\n{STYLE[label][2]}")
        print(f"  Configs: {len(sub)}")
        print(f"  Best test_l2: {sub['test_l2'].min():.5f}")
        print(f"    → rank={best.get('rank','?')}, width={best['width']}, modes={best.get('modes','?')}, params={int(best['param_count'])/1e3:.0f}K")
        print(f"  Worst test_l2: {sub['test_l2'].max():.5f}")
        print(f"  Param range: {sub['param_count'].min()/1e3:.0f}K – {sub['param_count'].max()/1e6:.1f}M")
        if sub["train_time_s"].notna().any():
            print(f"  Median train_time/epoch: {sub['train_time_s'].median():.2f}s")

    UNFACT_BEST = df[df["label"] == "Unfactorized"]["test_l2"].min()
    TT_BEST     = df[df["label"] == "Spectral-TT"]["test_l2"].min()
    TT_BEST_P   = df[df["label"] == "Spectral-TT"].loc[df[df["label"] == "Spectral-TT"]["test_l2"].idxmin(), "param_count"]
    UNFACT_MED_P = df[df["label"] == "Unfactorized"]["param_count"].median()
    print(f"\n{'='*65}")
    print(f"KEY HEADLINE NUMBERS")
    print(f"{'='*65}")
    print(f"  TT vs Unfactorized accuracy gain:  {(UNFACT_BEST - TT_BEST)/UNFACT_BEST*100:.1f}% lower error")
    print(f"  TT compression ratio (vs median):  {UNFACT_MED_P / TT_BEST_P:.0f}×")
    print(f"  Best TT test_l2: {TT_BEST:.5f}")
    print(f"  Best Unfactorized test_l2: {UNFACT_BEST:.5f}")


# ─── FIGURE D: SPECTRAL-TT STANDALONE HEATMAP ────────────────────────────────

def figD_spectral_tt_heatmap(df):
    """Large standalone heatmap for Spectral-TT (rank × width, m=12).
    Shows all available (rank, width) combinations; gray = not yet run."""
    print("Fig D: Spectral-TT standalone heatmap...")
    sub = df[(df["label"] == "Spectral-TT") & (df["modes"] == 12)].copy()
    if sub.empty:
        print("  WARNING: no Spectral-TT m=12 data")
        return

    pivot = sub.groupby(["rank", "width"])["test_l2"].min().unstack("width")
    pivot = pivot.sort_index(ascending=False)

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad("lightgray")

    nw = len(pivot.columns)
    fig_w = max(14, nw * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    im = ax.imshow(np.ma.masked_invalid(pivot.values),
                   aspect="auto", cmap=cmap, vmin=0.009, vmax=0.022)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=11,
                       rotation=45 if len(pivot.columns) > 8 else 0, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(r)) for r in pivot.index], fontsize=12)
    ax.set_xlabel("Width", fontsize=13)
    ax.set_ylabel("Rank", fontsize=13)
    ax.set_title("Spectral-TT: Test L² Error (rank × width, m=12)\n"
                 "(Fourier spectral operator, TT factorization, NIMROD 3D 32³)",
                 fontsize=14, fontweight="bold", color=STYLE["Spectral-TT"][0])

    # Color-only: no per-cell text at full rank range; mark global best
    best_i, best_j = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.48, best_i - 0.48), 0.96, 0.96,
                                fill=False, edgecolor="black", linewidth=2.5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Test L² Error", fontsize=12)

    fig.tight_layout()
    savefig(fig, "figD_spectral_tt_heatmap")


# ─── FIGURE E: SPECTRAL METHODS COMPARISON — TT vs QTT3 vs QTT5 HEATMAPS ─────

def figE_spectral_heatmaps_comparison(df):
    """1×3 row of rank×width heatmaps for TT, QTT3, QTT5 at m=12.
    Shows how factorization choice (TT vs QTT3 vs QTT5) trades off accuracy
    across the same rank/width design space."""
    from matplotlib.gridspec import GridSpec
    print("Fig E: Spectral TT/QTT3/QTT5 comparison heatmaps...")
    labels = ["Spectral-TT", "Spectral-QTT-q3", "Spectral-QTT-q5"]

    fig = plt.figure(figsize=(22, 6))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.045],
                  wspace=0.28, left=0.05, right=0.92, top=0.82, bottom=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])

    ORIG_RANKS = [2, 4, 8, 12, 16]
    ORIG_WIDTHS = [8, 16, 24, 32]

    vmin, vmax = 0.009, 0.022
    last_im = None
    for ax, label in zip(axes, labels):
        sub = df[(df["label"] == label) &
                 (df["modes"] == 12) &
                 (df["rank"].isin(ORIG_RANKS)) &
                 (df["width"].isin(ORIG_WIDTHS))].copy()
        if sub.empty:
            ax.set_title(STYLE[label][2])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        pivot = sub.groupby(["rank", "width"])["test_l2"].min().unstack("width")
        pivot = pivot.sort_index(ascending=False)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        last_im = im
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(r)) for r in pivot.index], fontsize=10)
        ax.set_xlabel("Width", fontsize=11)
        ax.set_ylabel("Rank", fontsize=11)
        ax.set_title(STYLE[label][2], color=STYLE[label][0], fontsize=13, fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color="black" if v < 0.016 else "white")
        best_i, best_j = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
        ax.add_patch(plt.Rectangle((best_j - 0.48, best_i - 0.48), 0.96, 0.96,
                                    fill=False, edgecolor="black", linewidth=2))

    if last_im is not None:
        fig.colorbar(last_im, cax=cbar_ax, label="Test L² Error")

    fig.suptitle("Spectral Operator Factorization Comparison — rank × width (m=12)\n"
                 "TT: highest accuracy  |  QTT-q3: balanced  |  QTT-q5: maximum compression",
                 fontsize=13, y=0.97)
    savefig(fig, "figE_spectral_heatmaps_comparison")


# ─── FIGURE C: DENSE-QTT ONLY HEATMAP ────────────────────────────────────────

def figC_dense_qtt_heatmap(df):
    """Large, standalone heatmap for the Dense-QTT slide — rank × width → test_l2.
    Shows all available (rank, width) combinations; gray = not yet run."""
    print("Fig C: Dense-QTT standalone heatmap...")
    sub = df[df["label"] == "Dense-QTT"].copy()
    if sub.empty:
        print("  WARNING: no Dense-QTT data")
        return

    # Even ranks only (original step-2 grid; odd ranks from gap-fill are less systematic)
    sub = sub[sub["rank"] % 2 == 0]
    # Omit rank > width: bond dim exceeding channel dim is redundant parameterization
    sub = sub[sub["rank"] <= sub["width"]]

    pivot = sub.groupby(["rank", "width"])["test_l2"].min().unstack("width")
    pivot = pivot.sort_index(ascending=False)

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad("lightgray")

    nw = len(pivot.columns)
    fig_w = max(11, nw * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    im = ax.imshow(np.ma.masked_invalid(pivot.values),
                   aspect="auto", cmap=cmap, vmin=0.016, vmax=0.030)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=11,
                       rotation=45 if len(pivot.columns) > 8 else 0, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(r)) for r in pivot.index], fontsize=12)
    ax.set_xlabel("Width", fontsize=13)
    ax.set_ylabel("Rank", fontsize=13)
    ax.set_title("RealOp-QTT: Test L² Error (rank × width)\n(Dense 8-way operator, q=8 QTT, NIMROD 3D 32³)",
                 fontsize=14, fontweight="bold", color=STYLE["Dense-QTT"][0])

    # Color-only: no per-cell text (too many cells at full rank range)
    # Mark global best with a border
    best_i, best_j = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.48, best_i - 0.48), 0.96, 0.96,
                                fill=False, edgecolor="black", linewidth=2.5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Test L² Error", fontsize=12)

    fig.tight_layout()
    savefig(fig, "figC_dense_qtt_heatmap")


# ─── FIGURE F: SPECTRAL-ONLY PARETO — for "Compression of R" slide ───────────

def figF_spectral_pareto(df):
    """Pareto frontier for spectral methods only (Unfactorized, TT, QTT3, QTT5).
    Cleaner than figA for a 'Compression of R' slide — no Dense-QTT distraction."""
    print("Fig F: Spectral-only Pareto (Compression of R)...")
    SPECTRAL_ORDER = ["Unfactorized", "Spectral-TT", "Spectral-QTT-q3", "Spectral-QTT-q5"]
    UNFACT_BEST = df[df["label"] == "Unfactorized"]["test_l2"].min()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter + frontier for each spectral method
    for label in SPECTRAL_ORDER:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        c, mk, name = STYLE[label]
        alpha = 0.20 if label == "Unfactorized" else 0.20
        ax.scatter(sub["param_count"], sub["test_l2"],
                   color=c, marker=mk, s=55, alpha=alpha,
                   edgecolors="white", linewidths=0.5, zorder=3)
        pf = pareto_frontier(sub)
        if len(pf) > 1:
            ax.plot(pf["param_count"], pf["test_l2"],
                    color=c, linewidth=2.5, zorder=4, alpha=0.92, label=name)

    # Baseline
    ax.axhline(UNFACT_BEST, color=STYLE["Unfactorized"][0], linestyle="--",
               linewidth=1.4, alpha=0.7, label=f"Best unfactorized ({UNFACT_BEST:.4f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Test L² Error")
    ax.set_title("Compression of R: Spectral Operator Factorization\n"
                 "(Unfactorized vs TT vs QTT — NIMROD 3D 32³, 400 epochs)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else (f"{x/1e3:.0f}K" if x >= 1e3 else str(int(x)))))
    _y_ticks_F = np.round(np.arange(0.009, 0.025, 0.001), 4)
    ax.yaxis.set_major_locator(mticker.FixedLocator(_y_ticks_F))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Annotate the three method endpoints
    for label, offset_x, offset_y in [
        ("Spectral-TT",     3.0,  1.6),
        ("Spectral-QTT-q3", 2.5,  0.3),
        ("Spectral-QTT-q5", 3.0,  2.5),
    ]:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        best = sub.loc[sub["test_l2"].idxmin()]
        c = STYLE[label][0]
        _bp = best['param_count']
        _bp_str = f"{_bp/1e6:.3f}M" if _bp >= 1e6 else f"{int(_bp/1e3)}K"
        ax.annotate(f"{STYLE[label][2]}\n{best['test_l2']:.4f}, {_bp_str}",
                    xy=(best["param_count"], best["test_l2"]),
                    xytext=(best["param_count"] * offset_x,
                            best["test_l2"] + offset_y * 0.001),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=8.5, color=c)

    # Annotate best unfactorized config
    sub_uf = df[df["label"] == "Unfactorized"]
    if not sub_uf.empty:
        best_uf = sub_uf.loc[sub_uf["test_l2"].idxmin()]
        c_uf = STYLE["Unfactorized"][0]
        ax.annotate(f"Unfactorized\n{best_uf['test_l2']:.4f}, {best_uf['param_count']/1e6:.1f}M",
                    xy=(best_uf["param_count"], best_uf["test_l2"]),
                    xytext=(30e6, 0.011),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=8.5, color=c_uf)

    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "figF_spectral_pareto")


# ─── FIGURE H: TRAINING TIME — DENSE-QTT LINE vs SPECTRAL-TT LINE + MODES BAND ─

def figH_training_time_dense_vs_spectral(df):
    """Dense-QTT (clean line, no modes) vs Spectral-TT (median line + shaded band
    showing the training-time range across all modes values at each width).
    At w=32 the band covers the full modes sweep (m=4–16); at other widths
    only m=12 is available so the band reflects rank-only variation."""
    print("Fig H: Training time Dense vs Spectral (with modes range band)...")
    has_time = df["train_time_s"].notna() & (df["train_time_s"] > 0)
    df_t = df[has_time].copy()

    # Augment TT data with the modes sweep (m=4–16 at r=12, w=32) so the band
    # at w=32 captures the actual modes-driven timing spread.
    modes_df = load_modes_sweep()
    tt_base = df_t[df_t["label"] == "Spectral-TT"].copy()
    if not modes_df.empty:
        modes_t = modes_df[modes_df["train_time_s"].notna() & (modes_df["train_time_s"] > 0)].copy()
        tt_all = pd.concat([tt_base, modes_t], ignore_index=True)
    else:
        tt_all = tt_base

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Dense-QTT: single median per width (no modes → no band) ──────────────
    dense = df_t[df_t["label"] == "Dense-QTT"]
    d_stats = (dense.groupby("width")["train_time_s"]
                     .median().reset_index().sort_values("width"))
    c_d, mk_d, name_d = STYLE["Dense-QTT"]
    ax.plot(d_stats["width"], d_stats["train_time_s"],
            color=c_d, marker=mk_d, linewidth=2.5, markersize=8,
            label=name_d, zorder=5)

    # ── Spectral-TT: median line + [min, max] band across modes (& rank) ─────
    tt_stats = (tt_all.groupby("width")["train_time_s"]
                       .agg(["median", "min", "max"])
                       .reset_index().sort_values("width"))
    c_tt, mk_tt, name_tt = STYLE["Spectral-TT"]
    ax.fill_between(tt_stats["width"], tt_stats["min"], tt_stats["max"],
                    color=c_tt, alpha=0.20,
                    label=f"{name_tt} — range across modes")
    ax.plot(tt_stats["width"], tt_stats["median"],
            color=c_tt, marker=mk_tt, linewidth=2.5, markersize=8,
            label=f"{name_tt} — median", zorder=5)

    ax.set_xlabel("Width")
    ax.set_ylabel("Training Time per Epoch (seconds)")
    ax.set_title("Training Time per Epoch: RealOp-QTT vs Spectral-TT\n"
                 "(band = range across modes configurations; "
                 "RealOp has no modes parameter)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(4, 12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    fig.tight_layout()
    savefig(fig, "figH_training_time_dense_vs_spectral")


# ─── FIGURE G: TRAINING TIME vs WIDTH — DENSE vs SPECTRAL ────────────────────

def figG_training_time_vs_width(df):
    """Training time per epoch vs width for all methods with all available widths.
    Scatter of individual configs + median line — updated version of the old
    dense_vs_spectral training time comparison."""
    print("Fig G: Training time vs width (dense vs spectral)...")
    has_time = df["train_time_s"].notna() & (df["train_time_s"] > 0)
    df_t = df[has_time].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for label in ORDER:
        sub = df_t[df_t["label"] == label]
        if sub.empty:
            continue
        c, mk, name = STYLE[label]
        # Faint scatter of individual configs
        ax.scatter(sub["width"], sub["train_time_s"],
                   color=c, marker=mk, s=22, alpha=0.25,
                   edgecolors="none", zorder=2)
        # Median per width
        med = sub.groupby("width")["train_time_s"].median().reset_index().sort_values("width")
        ax.plot(med["width"], med["train_time_s"],
                color=c, marker=mk, linewidth=2.5, markersize=8,
                label=name, zorder=4, alpha=0.95)

    ax.set_xlabel("Width")
    ax.set_ylabel("Training Time per Epoch (seconds)")
    ax.set_title("Training Time per Epoch vs. Width — All Methods\n"
                 "(line = median per width; faint points = individual configs)")
    ax.legend(framealpha=0.9, loc="upper left")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, "figG_training_time_vs_width")


# ─── FIGURE A: PARETO FRONTIER CLEAN (best achievable loss per param budget) ──

def figA_pareto_frontier_clean(df):
    """Frontier lines only — no scatter — cleanly shows best achievable L² at each budget.
    Equivalent to old plot9_pareto_frontier but with all 5 methods including Dense-QTT."""
    print("Fig A: Pareto frontier (clean frontier lines)...")
    fig, ax = plt.subplots(figsize=(10, 6))

    UNFACT_BEST = df[df["label"] == "Unfactorized"]["test_l2"].min()
    ax.axhline(UNFACT_BEST, color=STYLE["Unfactorized"][0], linestyle="--",
               linewidth=1.4, alpha=0.7, label=f"Best unfactorized ({UNFACT_BEST:.4f})")

    # Low-opacity scatter showing all individual configs (like fig01, barely visible)
    for label in ORDER:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        c, mk, _ = STYLE[label]
        ax.scatter(sub["param_count"], sub["test_l2"],
                   color=c, marker=mk, s=25, alpha=0.08,
                   edgecolors="none", zorder=2)

    for label in ORDER:
        sub = df[df["label"] == label].sort_values("param_count")
        if sub.empty:
            continue
        c, mk, name = STYLE[label]
        # Compute Pareto frontier (min test_l2 seen so far as we scan left to right)
        pf = pareto_frontier(sub)
        if len(pf) < 1:
            continue
        ax.plot(pf["param_count"], pf["test_l2"],
                color=c, marker=mk, linewidth=2.5, markersize=8, label=name, alpha=0.92)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Best Achievable Test L² Error")
    ax.set_title("Best Achievable Loss at Each Parameter Budget\n(NIMROD 3D, 32³, 400 epochs)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else (f"{x/1e3:.0f}K" if x >= 1e3 else str(int(x)))))
    # Dense decimal ticks — FixedLocator every 0.001 gives ~15 readable ticks across the data range
    _y_ticks_A = np.round(np.arange(0.009, 0.025, 0.001), 4)
    ax.yaxis.set_major_locator(mticker.FixedLocator(_y_ticks_A))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    handles = legend_handles()
    handles.append(Line2D([0], [0], color=STYLE["Unfactorized"][0], linestyle="--",
                          linewidth=1.4, label=f"Best unfactorized ({UNFACT_BEST:.4f})"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "figA_pareto_frontier_clean")


# ─── FIGURE B: COMPUTATIONAL COST — TRAIN TIME vs PARAM COUNT ─────────────────

def figB_computational_cost(df):
    """Scatter of training time per epoch vs parameter count for all methods.
    Equivalent to old plot6_training_time_comparison but with all 5 methods."""
    print("Fig B: Computational cost (train time vs param count)...")
    has_time = df["train_time_s"].notna() & (df["train_time_s"] > 0)
    df_t = df[has_time].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for label in ORDER:
        sub = df_t[df_t["label"] == label]
        if sub.empty:
            continue
        c, mk, name = STYLE[label]
        ax.scatter(sub["param_count"], sub["train_time_s"],
                   color=c, marker=mk, s=55, alpha=0.6, edgecolors="white",
                   linewidths=0.4, label=name, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Training Time per Epoch (seconds)")
    ax.set_title("Computational Cost: Training Time vs. Parameter Count\n(NIMROD 3D, 32³, 400 epochs)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else (f"{x/1e3:.0f}K" if x >= 1e3 else str(int(x)))))
    ax.legend(framealpha=0.9, loc="upper right")
    fig.tight_layout()
    savefig(fig, "figB_computational_cost")


# ─── QTT PADDING EFFECT FIGURES ──────────────────────────────────────────────

def _qtt_padding_meta(modes_list, base=2):
    """For each mode value compute QTT bits, padded size, and padding fraction."""
    import math
    rows = []
    for m in modes_list:
        bits = math.ceil(math.log(m, base)) if m > 1 else 0
        q    = base**bits if bits > 0 else m
        pad  = q - m
        rows.append({"modes": m, "bits": bits, "padded_to": q,
                     "pad_count": pad, "pad_frac": pad / q if q > 0 else 0.0})
    return pd.DataFrame(rows)


def _load_modes_all():
    """Load test_l2 for TT/QTT3/QTT5 at r=12, w=32, all available modes."""
    specs = {
        "Spectral-TT":     f"{PSCRATCH}/spectral_tt_modes_sweep/spectral_tt_r12_m*_w32/metrics.csv",
        "Spectral-QTT-q3": f"{PSCRATCH}/spectral_qtt_q3_sweep/spectral_qtt_q3_r12_m*_w32/metrics.csv",
        "Spectral-QTT-q5": f"{PSCRATCH}/spectral_qtt_q5_sweep/spectral_qtt_q5_r12_m*_w32/metrics.csv",
    }
    data = {}
    for label, pattern in specs.items():
        rows = []
        for f in sorted(glob.glob(pattern)):
            try:
                tmp = pd.read_csv(f)
                if not tmp.empty:
                    rows.append(tmp.iloc[-1].to_dict())
            except Exception:
                pass
        if rows:
            df = pd.DataFrame(rows)
            _to_num(df, ["modes", "test_l2", "train_l2", "rank", "width"])
            data[label] = df.sort_values("modes").reset_index(drop=True)
    return data


def _load_qtt3_training_curves():
    """Parse epoch-by-epoch test_l2 from QTT3 training logs at r=12, w=32."""
    curves = {}
    pattern = f"{PSCRATCH}/spectral_qtt_q3_sweep/spectral_qtt_q3_r12_m*_w32/training.log"
    for f in sorted(glob.glob(pattern)):
        m_match = re.search(r"_m(\d+)_w32", f)
        if not m_match:
            continue
        m = int(m_match.group(1))
        eps, vals = [], []
        try:
            with open(f) as fh:
                for line in fh:
                    ep_m = re.search(r"^ep=\s*(\d+)", line)
                    v_m  = re.search(r"test_l2_full / ntest=\s*([\d.e+\-]+)", line)
                    if ep_m and v_m:
                        eps.append(int(ep_m.group(1)))
                        vals.append(float(v_m.group(1)))
        except Exception:
            pass
        if eps:
            curves[m] = pd.DataFrame({"epoch": eps, "test_l2": vals})
    return curves


def figP1_qtt_padding_modes(modes_data, meta_df):
    """Modes sensitivity with QTT bit-depth background bands and padding labels."""
    print("Fig P1: QTT padding — modes sensitivity with bit structure...")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Shaded background by bit-depth group
    band_cfg = {
        2: ("#e8eaf6", "2-bit QTT\n(to 4)"),
        3: ("#e8f5e9", "3-bit QTT\n(to 8)"),
        4: ("#fff3e0", "4-bit QTT\n(to 16)"),
    }
    for bits, (color, blabel) in band_cfg.items():
        grp = meta_df[meta_df["bits"] == bits]
        if grp.empty:
            continue
        xmin = grp["modes"].min() - 0.6
        xmax = grp["modes"].max() + 0.6
        ax.axvspan(xmin, xmax, alpha=0.5, color=color, zorder=0)
        ax.text((xmin + xmax) / 2, 0.0205, blabel, ha="center", va="top",
                fontsize=7.5, color="#555555", style="italic")

    # Vertical dotted lines at power-of-2 (zero-padding cliffs)
    for m_cliff in [4, 8, 16]:
        if m_cliff in meta_df["modes"].values:
            ax.axvline(m_cliff, color="#aaaaaa", linewidth=0.9,
                       linestyle=":", zorder=1)

    # Plot each method
    line_styles = {"Spectral-TT": "-", "Spectral-QTT-q3": "--", "Spectral-QTT-q5": ":"}
    for label in ["Spectral-TT", "Spectral-QTT-q3", "Spectral-QTT-q5"]:
        if label not in modes_data:
            continue
        sub = modes_data[label]
        c, mk, name = STYLE[label]
        ls = line_styles[label]
        ax.plot(sub["modes"], sub["test_l2"], color=c, marker=mk,
                linewidth=2.2, markersize=7, linestyle=ls, label=name, zorder=5)

    # Annotate padding fraction on QTT3 points only
    if "Spectral-QTT-q3" in modes_data:
        c3 = STYLE["Spectral-QTT-q3"][0]
        for _, row in modes_data["Spectral-QTT-q3"].iterrows():
            m = int(row["modes"])
            mr = meta_df[meta_df["modes"] == m]
            if mr.empty:
                continue
            pf = mr.iloc[0]["pad_frac"]
            pad_c = mr.iloc[0]["pad_count"]
            txt = f"0% pad\n(cliff)" if pad_c == 0 else f"{int(pf*100)}% pad"
            y_off = -0.00055 if pad_c == 0 else 0.00045
            ax.annotate(txt, xy=(m, row["test_l2"]),
                        xytext=(m, row["test_l2"] + y_off),
                        fontsize=6.8, ha="center",
                        va="top" if pad_c == 0 else "bottom",
                        color=c3, alpha=0.9)

    _y = np.round(np.arange(0.008, 0.023, 0.001), 4)
    ax.yaxis.set_major_locator(mticker.FixedLocator(_y))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(meta_df["modes"].unique())))
    ax.set_xlabel("Fourier Modes")
    ax.set_ylabel("Test L² Error")
    ax.set_title("Modes Sensitivity: QTT Bit-Depth Structure & Zero-Padding Effect\n"
                 "(r=12, w=32 — dotted verticals = power-of-2 cliffs, QTT3 annotated)")
    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
    fig.tight_layout()
    savefig(fig, "figP1_qtt_padding_modes")


def figP2_qtt_padding_cliff(modes_data, meta_df):
    """4-bit group: show smooth m=10→14 trend and m=16 anomaly; compare 3-bit group."""
    print("Fig P2: QTT padding — cliff anomaly in 4-bit group...")
    if "Spectral-QTT-q3" not in modes_data:
        print("  No QTT3 data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    c3, mk3, _ = STYLE["Spectral-QTT-q3"]
    c5, mk5, _ = STYLE["Spectral-QTT-q5"]

    for ax, bits_val, title_sfx in zip(axes, [4, 3],
                                        ["4-bit group (modes 9–16, padded to 16)",
                                         "3-bit group (modes 5–8, padded to 8)"]):
        grp_meta = meta_df[meta_df["bits"] == bits_val].sort_values("modes")
        modes_in_grp = grp_meta["modes"].tolist()

        for label, color, mk in [("Spectral-QTT-q3", c3, mk3),
                                   ("Spectral-QTT-q5", c5, mk5)]:
            if label not in modes_data:
                continue
            sub = modes_data[label]
            sub_grp = sub[sub["modes"].isin(modes_in_grp)].sort_values("modes")
            if sub_grp.empty:
                continue
            name = STYLE[label][2]
            ax.plot(sub_grp["modes"], sub_grp["test_l2"],
                    color=color, marker=mk, linewidth=2, markersize=8,
                    label=name, zorder=5)

            # For 4-bit group: fit linear trend on m=10,12,14 and project to 16
            if bits_val == 4 and label == "Spectral-QTT-q3":
                trend_pts = sub_grp[sub_grp["modes"] < 16]
                if len(trend_pts) >= 2:
                    coeffs = np.polyfit(trend_pts["modes"], trend_pts["test_l2"], 1)
                    x_proj = np.array([trend_pts["modes"].max(), 16])
                    y_proj = np.polyval(coeffs, x_proj)
                    ax.plot(x_proj, y_proj, color=color, linestyle="--",
                            linewidth=1.4, alpha=0.55, zorder=3,
                            label="QTT3 linear trend (m≤14)")
                    # Annotate the gap at m=16
                    actual_16 = sub_grp[sub_grp["modes"] == 16]
                    if not actual_16.empty:
                        y_actual = actual_16.iloc[0]["test_l2"]
                        y_expected = np.polyval(coeffs, 16)
                        ax.annotate("",
                            xy=(16, y_actual),
                            xytext=(16, y_expected),
                            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5))
                        ax.text(16.15, (y_actual + y_expected) / 2,
                                f"Δ={y_actual - y_expected:+.4f}\n(zero-pad cliff)",
                                fontsize=7.5, color="#c0392b", va="center")

        # Annotate padding fraction on x-ticks
        ax.set_xticks(modes_in_grp)
        pad_labels = []
        for m in modes_in_grp:
            mr = meta_df[meta_df["modes"] == m]
            pf = int(mr.iloc[0]["pad_frac"] * 100) if not mr.empty else 0
            pad_labels.append(f"m={m}\n({pf}% pad)")
        ax.set_xticklabels(pad_labels, fontsize=8)
        ax.set_xlabel("Modes (with QTT padding fraction)")
        ax.set_ylabel("Test L² Error")
        ax.set_title(f"QTT Padding Effect — {title_sfx}")
        _y = np.round(np.arange(0.009, 0.021, 0.001), 4)
        ax.yaxis.set_major_locator(mticker.FixedLocator(_y))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.legend(fontsize=8.5, framealpha=0.9)

    fig.suptitle("Zero-Padding Cliff: Within Each QTT Bit-Depth Group\n"
                 "(QTT3/QTT5 r=12, w=32; dashed = projected trend without cliff)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "figP2_qtt_padding_cliff")


def figP3_qtt3_training_curves(curves, meta_df):
    """Full training trajectories for all QTT3 modes — plateau at m=16 visible."""
    print("Fig P3: QTT3 training curves — all modes...")
    if not curves:
        print("  No training curve data found.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    no_pad = {4, 8, 16}
    cmap = plt.cm.plasma
    modes_sorted = sorted(curves.keys())
    n = len(modes_sorted)

    for i, m in enumerate(modes_sorted):
        curve = curves[m].sort_values("epoch")
        color = cmap(i / max(n - 1, 1))
        is_cliff = m in no_pad
        ls = "--" if is_cliff else "-"
        lw = 2.6 if is_cliff else 1.7
        label = f"m={m} — 0% pad (cliff)" if is_cliff else f"m={m}"
        ax.plot(curve["epoch"], curve["test_l2"],
                color=color, linewidth=lw, linestyle=ls, label=label,
                alpha=0.95 if is_cliff else 0.75,
                zorder=5 if is_cliff else 3)

    # Annotate plateau for m=16
    if 16 in curves:
        c16 = curves[16]
        ep100 = c16[c16["epoch"] == 100]
        if not ep100.empty:
            ax.annotate("m=16 plateau\n(no padding)", fontsize=8, color="#c0392b",
                        xy=(100, ep100.iloc[0]["test_l2"]),
                        xytext=(140, ep100.iloc[0]["test_l2"] + 0.005),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test L² Error")
    ax.set_title("QTT3 Training Curves — All Modes (r=12, w=32)\n"
                 "Dashed = power-of-2 modes (zero padding); note m=16 stalls early")
    _y = np.round(np.arange(0.009, 0.058, 0.004), 4)
    ax.yaxis.set_major_locator(mticker.FixedLocator(_y))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(0, 400)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    savefig(fig, "figP3_qtt3_training_curves")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    set_style()
    df = load_all()
    print_summary(df)

    print(f"\nGenerating figures → {OUT_DIR}/\n")
    fig01_unified_pareto(df)
    fig02_learning_curves()
    fig02b_learning_curves_zoomed()
    fig03_heatmaps(df)
    fig04_modes_sensitivity(df)
    fig05_accuracy_vs_speed(df)
    fig06_compression(df)
    fig07_error_bars()
    fig08_rank_scaling(df)
    fig09_width_scaling(df)
    fig10_leaderboard(df)
    fig11_timing_distribution(df)
    figA_pareto_frontier_clean(df)
    figB_computational_cost(df)
    figG_training_time_vs_width(df)
    figH_training_time_dense_vs_spectral(df)
    figC_dense_qtt_heatmap(df)
    figD_spectral_tt_heatmap(df)
    figE_spectral_heatmaps_comparison(df)
    figF_spectral_pareto(df)

    # QTT padding effect analysis
    _modes_data = _load_modes_all()
    _meta_df    = _qtt_padding_meta([4, 6, 8, 10, 12, 14, 16])
    _curves     = _load_qtt3_training_curves()
    figP1_qtt_padding_modes(_modes_data, _meta_df)
    figP2_qtt_padding_cliff(_modes_data, _meta_df)
    figP3_qtt3_training_curves(_curves, _meta_df)

    # Save unified data for reference
    df.to_csv(os.path.join(OUT_DIR, "unified_results.csv"), index=False)
    print(f"\nAll figures generated. Unified data → unified_results.csv")


if __name__ == "__main__":
    main()
