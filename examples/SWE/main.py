import sys
import os
import pathlib
import argparse
import json
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stfno.stfno_2d_swe import STFNO2d_SWE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SWEDataset(Dataset):
    """Autoregressive dataset from sw2.pkl.

    Each sample is (x, y) where:
      x: (H, W, T_in) — T_in consecutive snapshots, spatial last
      y: (H, W)       — the next snapshot to predict
    """

    def __init__(self, data: torch.Tensor, T_in: int):
        # data: (N_traj, T, H, W) — already normalized, on CPU
        self.data = data
        self.N, self.T, self.H, self.W = data.shape
        self.T_in = T_in
        self.n_per_traj = self.T - T_in  # valid start indices per trajectory

    def __len__(self):
        return self.N * self.n_per_traj

    def __getitem__(self, idx):
        traj = idx // self.n_per_traj
        t0 = idx % self.n_per_traj
        x = self.data[traj, t0 : t0 + self.T_in]          # (T_in, H, W)
        y = self.data[traj, t0 + self.T_in]                # (H, W)
        x = x.permute(1, 2, 0)                             # (H, W, T_in)
        return x, y


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LpLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, pred, target):
        # pred, target: (batch, ...) — flatten all but batch
        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1)
        tgt_flat = target.reshape(B, -1)
        diff = torch.norm(pred_flat - tgt_flat, p=self.p, dim=1)
        norm = torch.norm(tgt_flat, p=self.p, dim=1).clamp(min=1e-8)
        return (diff / norm).mean()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='STFNO 2D SWE Training')

    parser.add_argument('--data_path', type=str,
                        default='/global/cfs/cdirs/m5031/data/sw2.pkl',
                        help='Path to sw2.pkl')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for checkpoints and metrics')

    # Model
    parser.add_argument('--operator_type', type=str, default='spectral',
                        choices=['spectral', 'real_op'],
                        help="'spectral': FFT-based FNO; 'real_op': direct 6-way spatial operator (ST-NO-2D, requires QTT)")
    parser.add_argument('--factorization_type', type=str, default='dense',
                        choices=['dense', 'tt', 'qtt'],
                        help='Weight factorization type (real_op requires qtt at 256x256)')
    parser.add_argument('--factorization_rank', type=int, default=4,
                        help='TT/QTT rank')
    parser.add_argument('--quantize_last_ndims', type=int, default=None,
                        help='Dims to QTT-quantize (default: 2 for spectral, 6 for real_op)')
    parser.add_argument('--bit_ordering', type=str, default='serial',
                        choices=['serial', 'interleaved'],
                        help='QTT bit ordering')
    parser.add_argument('--ft_implementation', type=str, default='reconstructed',
                        choices=['factorized', 'reconstructed'],
                        help='TT/QTT contraction implementation')
    parser.add_argument('--modes', type=int, default=16,
                        help='Fourier modes per spatial dimension (default 16; must be ≤ 128)')
    parser.add_argument('--width', type=int, default=32,
                        help='Model channel width')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of Fourier layers')
    parser.add_argument('--T_in', type=int, default=10,
                        help='Number of input time steps (conditioning window)')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    print(f"Loading {args.data_path} ...")
    t0 = time.time()
    with open(args.data_path, 'rb') as f:
        raw = pickle.load(f)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # (N, T, 1, H, W) float64 → float32, squeeze channel
    train_data = torch.tensor(raw['train'], dtype=torch.float32).squeeze(2)  # (320, 72, 256, 256)
    val_data   = torch.tensor(raw['val'],   dtype=torch.float32).squeeze(2)  # ( 40, 72, 256, 256)
    test_data  = torch.tensor(raw['test'],  dtype=torch.float32).squeeze(2)  # ( 40, 72, 256, 256)
    del raw

    # Normalize by train statistics
    train_mean = train_data.mean()
    train_std  = train_data.std().clamp(min=1e-8)
    train_data = (train_data - train_mean) / train_std
    val_data   = (val_data   - train_mean) / train_std
    test_data  = (test_data  - train_mean) / train_std

    print(f"  train: {train_data.shape}, val: {val_data.shape}, test: {test_data.shape}")
    print(f"  mean={train_mean:.4f}, std={train_std:.4f}")

    H, W = train_data.shape[-2], train_data.shape[-1]

    # Resolve quantize_last_ndims default based on operator type
    if args.quantize_last_ndims is None:
        args.quantize_last_ndims = 6 if args.operator_type == 'real_op' else 2

    print(f"quantize_last_ndims resolved to: {args.quantize_last_ndims}")

    train_loader = DataLoader(
        SWEDataset(train_data, args.T_in),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == 'cuda',
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = STFNO2d_SWE(
        T_in=args.T_in,
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        n_layers=args.n_layers,
        factorization=args.factorization_type,
        rank=args.factorization_rank,
        quantize_last_ndims=args.quantize_last_ndims,
        bit_ordering=args.bit_ordering,
        implementation=args.ft_implementation,
        operator_type=args.operator_type,
        spatial_size=(H, W),
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = LpLoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_l2 = float('inf')
    best_test_l2 = float('inf')
    metrics_history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        train_l2 = 0.0
        n_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)   # (B, H, W, T_in)
            y_batch = y_batch.to(device)   # (B, H, W)

            pred = model(x_batch).squeeze(-1)  # (B, H, W)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_l2 += loss.item() * x_batch.shape[0]
            n_train += x_batch.shape[0]

        scheduler.step()
        train_l2 /= n_train

        # ------------------------------------------------------------------
        # Validation (one-step, no autoregressive rollout, every 10 epochs)
        # ------------------------------------------------------------------
        if ep % 10 == 0 or ep == args.epochs:
            model.eval()
            val_l2, test_l2 = _evaluate(model, val_data, test_data, args.T_in, device, loss_fn)

            if val_l2 < best_val_l2:
                best_val_l2 = val_l2
                best_test_l2 = test_l2
                torch.save(
                    {'epoch': ep, 'model_state': model.state_dict(),
                     'val_l2': val_l2, 'test_l2': test_l2},
                    os.path.join(args.output_dir, 'best_model.pt'),
                )

            metrics_history.append({
                'epoch': ep,
                'train_l2': train_l2,
                'val_l2': val_l2,
                'test_l2': test_l2,
                'best_val_l2': best_val_l2,
                'best_test_l2': best_test_l2,
            })

            print(
                f"Epoch {ep:4d}/{args.epochs}  "
                f"train={train_l2:.4f}  val={val_l2:.4f}  test={test_l2:.4f}  "
                f"best_test={best_test_l2:.4f}",
                flush=True,
            )

            # Save intermediate results so a timeout doesn't lose everything
            with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
                json.dump({
                    'config': vars(args),
                    'n_params': n_params,
                    'best_val_l2': best_val_l2,
                    'best_test_l2': best_test_l2,
                    'history': metrics_history,
                }, f, indent=2)

    # ------------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------------
    results = {
        'config': vars(args),
        'n_params': n_params,
        'best_val_l2': best_val_l2,
        'best_test_l2': best_test_l2,
        'history': metrics_history,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. best_val={best_val_l2:.4f}  best_test={best_test_l2:.4f}")
    print(f"Results saved to {args.output_dir}")


def _evaluate(model, val_data, test_data, T_in, device, loss_fn):
    """One-step evaluation over all valid windows in val and test sets."""
    def _split_eval(data):
        N, T, H, W = data.shape
        total_l2 = 0.0
        n_samples = 0
        with torch.no_grad():
            for t0 in range(T - T_in):
                x = data[:, t0:t0+T_in].permute(0, 2, 3, 1).to(device)  # (N, H, W, T_in)
                y = data[:, t0+T_in].to(device)                           # (N, H, W)
                pred = model(x).squeeze(-1)                                # (N, H, W)
                total_l2 += loss_fn(pred, y).item() * N
                n_samples += N
        return total_l2 / n_samples

    val_l2  = _split_eval(val_data)
    test_l2 = _split_eval(test_data)
    return val_l2, test_l2


if __name__ == '__main__':
    main()
