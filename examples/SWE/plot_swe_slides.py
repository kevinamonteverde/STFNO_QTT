"""Generate all SWE slide figures and save to Obsidian vault."""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE   = pathlib.Path('/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results')
FIGS   = pathlib.Path('/global/homes/p/pepi/pepi-notes/STFNO/figures')
FIGS.mkdir(parents=True, exist_ok=True)

FNO_BASELINE = 9.53e-2

def load(dirname):
    p = BASE / dirname / 'results.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

# ── colour palette ────────────────────────────────────────────────────────────
SPEC_COLOR   = '#1f77b4'   # blue — spectral family
W32_COLOR    = '#ff7f0e'   # orange — realop w=32
W64_COLOR    = '#d62728'   # red — realop w=64
FNO_COLOR    = '#7f7f7f'   # grey — FNO baseline

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Pareto (param efficiency)
# ─────────────────────────────────────────────────────────────────────────────
configs_pareto = [
    # (dirname,               label,             color,      marker, ms)
    ('baseline_spectral_dense_w32_m16', 'Spectral w=32', SPEC_COLOR, 's', 10),
    ('spectral_w64_m16',                'Spectral w=64', SPEC_COLOR, 'D', 12),
    ('realop_qtt_r8_w32_400ep',   'Realop r=8  w=32',  W32_COLOR, 'o', 7),
    ('realop_qtt_r16_w32_400ep',  'Realop r=16 w=32',  W32_COLOR, 'o', 7),
    ('realop_qtt_r24_w32_400ep',  'Realop r=24 w=32',  W32_COLOR, 'o', 7),
    ('realop_qtt_r32_w32_400ep',  'Realop r=32 w=32',  W32_COLOR, 'o', 9),
    ('realop_qtt_r16_w64',        'Realop r=16 w=64',  W64_COLOR, '^', 7),
    ('realop_qtt_r24_w64_400ep',  'Realop r=24 w=64',  W64_COLOR, '^', 7),
    ('realop_qtt_r32_w64_400ep',  'Realop r=32 w=64',  W64_COLOR, '^', 9),
]

fig, ax = plt.subplots(figsize=(7, 5))

# Collect points by family for connecting lines
spec_pts  = []
w32_pts   = []
w64_pts   = []

for dirname, label, color, marker, ms in configs_pareto:
    r = load(dirname)
    if r is None:
        print(f"  Missing: {dirname}")
        continue
    n, l = r['n_params'], r['best_test_l2']
    ax.scatter(n, l, color=color, marker=marker, s=ms**2, zorder=5,
               edgecolors='white', linewidths=0.5)
    if 'spectral' in dirname:
        spec_pts.append((n, l))
    elif 'w32' in dirname and 'spectral' not in dirname:
        w32_pts.append((n, l))
    elif 'w64' in dirname and 'spectral' not in dirname:
        w64_pts.append((n, l))

# Connect points within each family
for pts, color, ls in [
    (sorted(spec_pts),  SPEC_COLOR, '-'),
    (sorted(w32_pts),   W32_COLOR,  '--'),
    (sorted(w64_pts),   W64_COLOR,  '--'),
]:
    if len(pts) >= 2:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.5, zorder=3)

# FNO baseline
ax.axhline(FNO_BASELINE, color=FNO_COLOR, linestyle=':', linewidth=2, zorder=4)
ax.text(2e4, FNO_BASELINE * 0.91, 'FNO baseline (9.53e−2)', color=FNO_COLOR,
        fontsize=9, va='top')

# Annotate headline points — (xytext = offset in points from data point)
for dirname, note, ha, xytext in [
    ('spectral_w64_m16',         '2.00e−3\n(47× FNO)',  'right', (-15, 28)),
    ('realop_qtt_r32_w32_400ep', '3.17e−2\n(3.0× FNO)', 'right', (-15, 40)),
    ('realop_qtt_r32_w64_400ep', '2.47e−2\n(3.9× FNO)', 'left',  (40,  28)),
]:
    r = load(dirname)
    if r:
        ax.annotate(note, xy=(r['n_params'], r['best_test_l2']),
                    xytext=xytext, textcoords='offset points',
                    fontsize=8, color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                    ha=ha)

# Legend
import matplotlib.lines as mlines
leg_handles = [
    mlines.Line2D([], [], color=SPEC_COLOR, marker='s', markersize=9,
                  linestyle='-', label='Spectral (dense FNO)'),
    mlines.Line2D([], [], color=W32_COLOR, marker='o', markersize=8,
                  linestyle='--', label='Realop QTT  w=32'),
    mlines.Line2D([], [], color=W64_COLOR, marker='^', markersize=8,
                  linestyle='--', label='Realop QTT  w=64'),
    mlines.Line2D([], [], color=FNO_COLOR, linestyle=':', linewidth=2,
                  label='FNO baseline (StFT paper)'),
]
ax.legend(handles=leg_handles, fontsize=8, loc='upper right')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameter Count', fontsize=11)
ax.set_ylabel('Test L² Error', fontsize=11)
ax.set_title('SWE: Accuracy vs. Parameter Count\n(Shallow Water Equations, 256²)', fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
out = FIGS / 'swe_fig1_pareto.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Rollout comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
rollout_data = {
    'FNO\nbaseline':       {'one_step': None, 'rollout67': 9.53e-2},
    'Spectral\nw=32':      {'one_step': 3.28e-3, 'rollout67': 4.49e-2},
    'Spectral\nw=64':      {'one_step': 2.00e-3, 'rollout67': 1.85e-2},
}

labels = list(rollout_data.keys())
one_step  = [rollout_data[k]['one_step']  for k in labels]
rollout67 = [rollout_data[k]['rollout67'] for k in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4.5))

bars1 = ax.bar(x - width/2, rollout67, width,
               color=[FNO_COLOR, SPEC_COLOR, SPEC_COLOR],
               alpha=0.9, label='67-step rollout', edgecolor='white')
bars2_vals = [v if v is not None else 0 for v in one_step]
bars2 = ax.bar(x + width/2, bars2_vals, width,
               color=[FNO_COLOR, SPEC_COLOR, SPEC_COLOR],
               alpha=0.45, label='One-step', edgecolor='white',
               hatch='///')

# Value labels on bars
for bar, val in zip(bars1, rollout67):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.04,
            f'{val:.2e}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for bar, val in zip(bars2, one_step):
    if val is not None:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.04,
                f'{val:.2e}', ha='center', va='bottom', fontsize=8.5)

# Improvement annotations
for xi, k in enumerate(labels):
    r67 = rollout_data[k]['rollout67']
    improvement = FNO_BASELINE / r67
    if improvement > 1.05:
        ax.text(xi - width/2, r67 * 0.55, f'{improvement:.1f}× better\nthan FNO',
                ha='center', fontsize=7.5, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Test L² Error (log scale)', fontsize=11)
ax.set_title('SWE: One-step vs. 67-step Autoregressive Rollout\n(lower is better)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
out = FIGS / 'swe_fig2_rollout.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Rank scaling (w=32 vs w=64)
# ─────────────────────────────────────────────────────────────────────────────
rank_data = {
    'w=32': [
        (8,  load('realop_qtt_r8_w32_400ep')),
        (16, load('realop_qtt_r16_w32_400ep')),
        (24, load('realop_qtt_r24_w32_400ep')),
        (32, load('realop_qtt_r32_w32_400ep')),
    ],
    'w=64': [
        (16, load('realop_qtt_r16_w64')),
        (24, load('realop_qtt_r24_w64_400ep')),
        (32, load('realop_qtt_r32_w64_400ep')),
    ],
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left panel: test_l2 vs rank
ax = axes[0]
for family, color in [('w=32', W32_COLOR), ('w=64', W64_COLOR)]:
    pts = [(r, d['best_test_l2'], d['n_params'])
           for r, d in rank_data[family] if d is not None]
    ranks  = [p[0] for p in pts]
    errors = [p[1] for p in pts]
    params = [p[2] for p in pts]
    ax.plot(ranks, errors, 'o-', color=color, linewidth=2, markersize=8, label=family)
    for r, e, p in zip(ranks, errors, params):
        ax.annotate(f'{p//1000}K', (r, e),
                    textcoords='offset points', xytext=(4, 4), fontsize=7.5)

ax.axhline(FNO_BASELINE, color=FNO_COLOR, linestyle=':', linewidth=2)
ax.text(8.2, FNO_BASELINE * 1.04, 'FNO baseline', fontsize=8.5, color=FNO_COLOR)
ax.set_xlabel('TT Rank (r)', fontsize=11)
ax.set_ylabel('Test L² Error', fontsize=11)
ax.set_title('Realop QTT — Rank Scaling\n(SWE, 400 epochs)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right panel: param count vs test_l2 (Pareto view of realop only)
ax = axes[1]
for family, color, marker in [('w=32', W32_COLOR, 'o'), ('w=64', W64_COLOR, '^')]:
    pts = sorted([(d['n_params'], d['best_test_l2'])
                  for r, d in rank_data[family] if d is not None])
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker+'-', color=color, linewidth=2, markersize=8, label=family)

ax.axhline(FNO_BASELINE, color=FNO_COLOR, linestyle=':', linewidth=2)
ax.text(3.5e4, FNO_BASELINE * 1.06, 'FNO baseline', fontsize=8.5, color=FNO_COLOR)
ax.set_xscale('log')
ax.set_xlabel('Parameter Count', fontsize=11)
ax.set_ylabel('Test L² Error', fontsize=11)
ax.set_title('Realop QTT — Param Efficiency\n(SWE, 400 epochs)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

plt.suptitle('ST-NO-QTT: Width vs. Rank Trade-off on SWE', fontsize=12, y=1.02)
plt.tight_layout()
out = FIGS / 'swe_fig3_rank_scaling.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Learning curves (all 400ep runs + spectral references)
# ─────────────────────────────────────────────────────────────────────────────
curve_configs = [
    ('realop_qtt_r8_w32_400ep',   'Realop r=8  w=32 (29K)',   W32_COLOR,  ':',  1.5),
    ('realop_qtt_r16_w32_400ep',  'Realop r=16 w=32 (91K)',   W32_COLOR,  '--', 1.5),
    ('realop_qtt_r24_w32_400ep',  'Realop r=24 w=32 (194K)',  W32_COLOR,  '-',  1.5),
    ('realop_qtt_r32_w32_400ep',  'Realop r=32 w=32 (337K)',  W32_COLOR,  '-',  2.5),
    ('realop_qtt_r16_w64',        'Realop r=16 w=64 (112K)',  W64_COLOR,  '--', 1.5),
    ('realop_qtt_r24_w64_400ep',  'Realop r=24 w=64 (220K)',  W64_COLOR,  '-',  1.5),
    ('realop_qtt_r32_w64_400ep',  'Realop r=32 w=64 (370K)',  W64_COLOR,  '-',  2.5),
]

fig, ax = plt.subplots(figsize=(9, 5))

for dirname, label, color, ls, lw in curve_configs:
    r = load(dirname)
    if r is None:
        print(f"  Missing: {dirname}")
        continue
    epochs = [h['epoch'] for h in r['history']]
    vals   = [h['val_l2'] for h in r['history']]
    ax.semilogy(epochs, vals, color=color, linestyle=ls, linewidth=lw,
                label=label, alpha=0.9)

# Reference lines
ax.axhline(FNO_BASELINE, color=FNO_COLOR, linestyle=':', linewidth=2, zorder=1)
ax.text(5, FNO_BASELINE * 1.08, 'FNO baseline (9.53e−2)', color=FNO_COLOR, fontsize=8.5)

spec32 = load('baseline_spectral_dense_w32_m16')
spec64 = load('spectral_w64_m16')
if spec32:
    ax.axhline(spec32['best_test_l2'], color=SPEC_COLOR, linestyle='-.', linewidth=1.5)
    ax.text(5, spec32['best_test_l2'] * 0.88, 'Spectral w=32 (3.28e−3)',
            color=SPEC_COLOR, fontsize=8.5)
if spec64:
    ax.axhline(spec64['best_test_l2'], color=SPEC_COLOR, linestyle='-', linewidth=1.5)
    ax.text(5, spec64['best_test_l2'] * 0.72, 'Spectral w=64 (2.00e−3)',
            color=SPEC_COLOR, fontsize=8.5)

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Val L² Error', fontsize=11)
ax.set_title('SWE Learning Curves — Realop QTT\n(w=32 orange, w=64 red; dashed/solid = lighter/heavier)', fontsize=11)
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
out = FIGS / 'swe_fig4_learning_curves.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Summary table (text as figure — good for slides)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.axis('off')

rows = [
    ['Config',                'Params',   'One-step L²', 'vs FNO',   'Rollout L²\n(67 steps)', 'vs FNO\n(rollout)'],
    ['Spectral  w=64',        '8.4M',     '2.00e−3',     '47×',      '1.85e−2',                '5.1×'],
    ['Spectral  w=32',        '2.1M',     '3.28e−3',     '29×',      '4.49e−2',                '2.1×'],
    ['Realop r=32  w=64',     '370K',     '2.47e−2',     '3.9×',     '—',                      '—'],
    ['Realop r=24  w=64',     '220K',     '3.22e−2',     '3.0×',     '—',                      '—'],
    ['Realop r=32  w=32',     '337K',     '3.17e−2',     '3.0×',     '—',                      '—'],
    ['Realop r=16  w=32',     '91K',      '5.23e−2',     '1.8×',     '—',                      '—'],
    ['Realop r=8   w=32',     '29K',      '7.05e−2',     '1.4×',     '—',                      '—'],
    ['FNO (StFT paper)',       '~2M',      '9.53e−2',     'baseline', '9.53e−2',                'baseline'],
]

# Row colours
row_colors = []
for i, row in enumerate(rows):
    if i == 0:
        row_colors.append(['#2c3e50'] * len(row))
    elif 'Spectral  w=64' in row[0]:
        row_colors.append(['#d4e6f1'] * len(row))
    elif 'Realop r=32  w=64' in row[0]:
        row_colors.append(['#fde8d8'] * len(row))
    elif 'FNO' in row[0]:
        row_colors.append(['#e8e8e8'] * len(row))
    else:
        row_colors.append(['#f9f9f9'] * len(row))

tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               cellLoc='center', loc='center',
               cellColours=row_colors[1:],
               colColours=row_colors[0])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.7)

# Bold header
for j in range(len(rows[0])):
    tbl[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title('SWE Benchmark — All Results', fontsize=12, pad=12, fontweight='bold')
plt.tight_layout()
out = FIGS / 'swe_fig5_results_table.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

print("\nAll figures saved to:", FIGS)
