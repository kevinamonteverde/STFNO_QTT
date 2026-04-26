"""Generate learning curve figure from completed SWE real_op results."""
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BASE = pathlib.Path('/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results')
OUT_DIR = pathlib.Path('/global/homes/p/pepi/pepi-notes/STFNO')

configs = [
    ('realop_qtt_r8_w32',  'r=8 (29K)',  'C0'),
    ('realop_qtt_r16_w32', 'r=16 (91K)', 'C1'),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, metric, ylabel in zip(axes, ['train_l2', 'val_l2'],
                               ['Train L2', 'Val L2']):
    for dirname, label, color in configs:
        path = BASE / dirname / 'results.json'
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        epochs = [h['epoch'] for h in data['history']]
        vals   = [h[metric] for h in data['history']]
        ax.semilogy(epochs, vals, color=color, label=label, linewidth=2)

    # Baselines
    ax.axhline(9.53e-2, color='gray', linestyle='--', linewidth=1.2, label='FNO baseline (9.53e-2)')
    ax.axhline(3.28e-3, color='black', linestyle=':', linewidth=1.2, label='STFNO spectral (3.28e-3)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(f'SWE real_op QTT — {ylabel}')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / 'swe_realop_learning_curves_200ep.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
