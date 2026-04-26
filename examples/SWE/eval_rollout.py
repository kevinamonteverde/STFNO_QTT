"""
Autoregressive rollout evaluation — matches StFT paper's test protocol.

Usage:
    python3 eval_rollout.py \
        --checkpoint /pscratch/.../best_model.pt \
        --data_path /global/cfs/cdirs/m5031/data/sw2.pkl \
        --rollout_steps 20   # how many steps to predict autoregressively
        [--operator_type real_op] [--factorization_type qtt] [--factorization_rank 16]
        [--width 32] [--n_layers 4] [--T_in 5]

The StFT paper evaluates autoregressive rollout (predictions fed back as input).
Our training loop uses one-step prediction. This script measures the harder metric
so we can report an apples-to-apples number alongside the paper's FNO baseline.

Relative L2 formula (matches both our training loss and StFT's data_utils.py):
    L2_rel = ||pred - target||_2 / ||target||_2   (per sample, then averaged)
"""
import sys
import os
import pathlib
import argparse
import json
import pickle

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stfno.stfno_2d_swe import STFNO2d_SWE


class LpLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, pred, target):
        B = pred.shape[0]
        diff = torch.norm((pred - target).reshape(B, -1), p=self.p, dim=1)
        norm = torch.norm(target.reshape(B, -1), p=self.p, dim=1).clamp(min=1e-8)
        return (diff / norm).mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_path', type=str, default='/global/cfs/cdirs/m5031/data/sw2.pkl')
    p.add_argument('--rollout_steps', type=int, default=20,
                   help='Number of autoregressive steps to predict (default 20)')
    # Model architecture — must match the checkpoint
    p.add_argument('--operator_type', type=str, default='spectral')
    p.add_argument('--factorization_type', type=str, default='dense')
    p.add_argument('--factorization_rank', type=int, default=4)
    p.add_argument('--quantize_last_ndims', type=int, default=None)
    p.add_argument('--bit_ordering', type=str, default='serial')
    p.add_argument('--ft_implementation', type=str, default='reconstructed')
    p.add_argument('--modes', type=int, default=16)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--T_in', type=int, default=5)
    return p.parse_args()


def rollout_eval(model, data, T_in, rollout_steps, device, loss_fn):
    """
    Autoregressive rollout evaluation.

    For each trajectory and each valid start t0:
      - Seed with T_in ground-truth frames
      - Predict rollout_steps steps, each time feeding the prediction back
      - Compute relative L2 at each step, then average over steps and samples

    Returns: mean relative L2 over all (trajectory, start, step) combinations.
    """
    N, T, H, W = data.shape
    max_start = T - T_in - rollout_steps + 1
    if max_start <= 0:
        raise ValueError(
            f"T={T}, T_in={T_in}, rollout_steps={rollout_steps}: "
            f"not enough timesteps (need T >= T_in + rollout_steps)"
        )

    model.eval()
    total_l2 = 0.0
    n_terms = 0

    with torch.no_grad():
        for t0 in range(max_start):
            # Seed window: (N, T_in, H, W) — ground truth
            window = data[:, t0 : t0 + T_in].clone().to(device)  # (N, T_in, H, W)

            for step in range(rollout_steps):
                x = window.permute(0, 2, 3, 1)          # (N, H, W, T_in)
                pred = model(x).squeeze(-1)              # (N, H, W)

                gt = data[:, t0 + T_in + step].to(device)   # (N, H, W)
                total_l2 += loss_fn(pred, gt).item() * N
                n_terms += N

                # Slide window: drop oldest, append prediction
                window = torch.cat([window[:, 1:], pred.unsqueeze(1)], dim=1)

    return total_l2 / n_terms


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading {args.data_path} ...")
    with open(args.data_path, 'rb') as f:
        raw = pickle.load(f)

    test_data  = torch.tensor(raw['test'],  dtype=torch.float32).squeeze(2)
    train_data = torch.tensor(raw['train'], dtype=torch.float32).squeeze(2)
    del raw

    # Normalize by train statistics (same as training)
    train_mean = train_data.mean()
    train_std  = train_data.std().clamp(min=1e-8)
    test_data  = (test_data - train_mean) / train_std
    del train_data

    H, W = test_data.shape[-2], test_data.shape[-1]

    if args.quantize_last_ndims is None:
        args.quantize_last_ndims = 6 if args.operator_type == 'real_op' else 2

    # Build model
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

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"one-step val_l2={ckpt.get('val_l2', '?'):.4f}")

    loss_fn = LpLoss()
    T = test_data.shape[1]

    # One-step eval (same as training metric)
    model.eval()
    total_onestep = 0.0
    n_onestep = 0
    with torch.no_grad():
        for t0 in range(T - args.T_in):
            x = test_data[:, t0 : t0 + args.T_in].permute(0, 2, 3, 1).to(device)
            y = test_data[:, t0 + args.T_in].to(device)
            pred = model(x).squeeze(-1)
            total_onestep += loss_fn(pred, y).item() * test_data.shape[0]
            n_onestep += test_data.shape[0]
    onestep_l2 = total_onestep / n_onestep

    # Autoregressive rollout eval
    rollout_l2 = rollout_eval(model, test_data, args.T_in, args.rollout_steps, device, loss_fn)

    print(f"\n--- Results ---")
    print(f"One-step test L2:          {onestep_l2:.4e}")
    print(f"Rollout test L2 ({args.rollout_steps} steps): {rollout_l2:.4e}")
    print(f"StFT FNO baseline (paper): 9.53e-02  (autoregressive, unknown # steps)")

    # Save
    out_path = pathlib.Path(args.checkpoint).parent / 'rollout_eval.json'
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'rollout_steps': args.rollout_steps,
            'one_step_test_l2': onestep_l2,
            'rollout_test_l2': rollout_l2,
            'stft_fno_baseline': 9.53e-2,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
