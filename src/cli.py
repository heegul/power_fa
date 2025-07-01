"""Command‐line interface entry‐point.

Usage examples
--------------
Run single sim:
    python -m power_fa.cli run --config cfgs/base.yaml --algo fs_bruteforce

Batch (sweep seeds 0..9):
    python -m power_fa.cli batch --config cfgs/base.yaml --algo fs_bruteforce --seeds 0 9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import pickle
import os
import random

import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .config import SimulationConfig
from .parallel import run_batch
from .simulator.engine import run_once
from .simulator.scenario import generate_samples, CHANNEL_TYPES
from .simulator.metrics import sinr_linear, sum_rate_bps, sum_rate_dimensionless

SUBCOMMANDS = {"run", "batch", "generate-samples", "train_dnn", "validate_dnn"}


def _parse_args(argv: List[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="power_fa", description="D2D power & FA simulator")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    p_run = subparsers.add_parser("run", help="Run a single simulation instance")
    p_run.add_argument("--config", required=True, type=Path, help="YAML config file")
    p_run.add_argument("--algo", required=True, help="Algorithm name (class name)")
    p_run.add_argument("--hidden_size", type=int, nargs="*", default=None, help="Hidden layer sizes for ML_DNN (e.g. 200 200)")

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------
    p_batch = subparsers.add_parser("batch", help="Run multiple seeds in parallel")
    p_batch.add_argument("--config", required=True, type=Path, help="YAML config file")
    p_batch.add_argument("--algo", required=True, help="Algorithm name")
    p_batch.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive range of seeds to iterate (START END)",
        required=True,
    )
    p_batch.add_argument("--processes", type=int, default=None, help="Number of worker processes")

    # ------------------------------------------------------------------
    # generate-samples
    # ------------------------------------------------------------------
    p_gen = subparsers.add_parser("generate-samples", help="Generate ML input samples (channel gains in dB)")
    p_gen.add_argument("--n_samples", type=int, required=True, help="Number of samples to generate")
    p_gen.add_argument("--n_pairs", type=int, required=True, help="Number of device pairs")
    p_gen.add_argument("--n_fa", type=int, required=True, help="Number of frequency allocations (FAs)")
    p_gen.add_argument("--area_size_m", type=float, default=100.0, help="Area size (meters)")
    p_gen.add_argument("--channel_type", type=str, choices=list(CHANNEL_TYPES), required=True, help="Channel type (urban, suburban, rural)")
    p_gen.add_argument("--seed", type=int, default=0, help="Random seed")
    p_gen.add_argument("--out_path", type=str, required=True, help="Output .npy file path")
    p_gen.add_argument("--restrict_rx_distance", action="store_true", help="Restrict RX to be within 1%-10% of area_size_m from its TX (random angle)")

    # ------------------------------------------------------------------
    # train_dnn
    # ------------------------------------------------------------------
    p_train = subparsers.add_parser("train_dnn", help="Train ML_DNN on one scenario (same sample for train/val)")
    p_train.add_argument("--config", required=True, type=Path, help="YAML config file")
    p_train.add_argument("--hidden_size", type=int, nargs="*", default=[200, 200], help="Hidden layer sizes (e.g. 200 200)")
    p_train.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    p_train.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    p_train.add_argument("--save_path", type=str, default="trained_weights.pt", help="Path to save trained weights")
    p_train.add_argument("--model_seed", type=int, default=0, help="Random seed for model initialization")
    p_train.add_argument("--n_eval_samples", type=int, default=1, help="Number of evaluation samples after training")
    p_train.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    p_train.add_argument("--min_delta", type=float, default=1e-3, help="Early stopping min_delta improvement")
    # Mutually exclusive group for soft-fa (default) and no-soft-fa
    soft_fa_group = p_train.add_mutually_exclusive_group(required=False)
    soft_fa_group.add_argument('--soft-fa', dest='soft_fa', action='store_true', help='Use soft FA loss (differentiable) [default]')
    soft_fa_group.add_argument('--no-soft-fa', dest='soft_fa', action='store_false', help='Use hard FA loss (argmax, non-differentiable)')
    p_train.set_defaults(soft_fa=True)
    p_train.add_argument("--results_path", type=str, default="results_eval.yaml", help="Path to save structured YAML results for all samples")
    p_train.add_argument('--eval_sample_idx', type=int, help='Index of evaluation sample to visualize')
    p_train.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds to try per sample (Best-of-N)')
    p_train.add_argument('--n_train_samples', type=int, default=None, help='Number of training samples (optional)')
    p_train.add_argument('--no_shuffle', action='store_true', help='Disable shuffling of the training DataLoader (useful for memorisation studies)')
    p_train.add_argument('--train_npy', type=str, default=None, help='Path to .npy file for training samples (channel gains)')
    p_train.add_argument('--val_npy', type=str, default=None, help='Path to .npy file for validation samples (channel gains)')
    p_train.add_argument('--batch_size', type=int, default=64, help='Batch size for mini-batch training')
    p_train.add_argument('--no_batch_norm', action='store_true', help='Disable BatchNorm in DNN model')
    p_train.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device for DNN (auto, cpu, cuda, mps)')
    p_train.add_argument('--restrict_rx_distance', action='store_true', help='Restrict RX to be within 1%-10% of area_size_m from its TX (random angle)')
    p_train.add_argument('--figs_dir', type=str, default='figs', help='Directory to save figures (default: figs)')
    p_train.add_argument('--normalization', type=str, choices=['global', 'local', 'none'], default='global',
        help="Normalization scheme: 'global' (NPY mode), 'local' (per-sample), or 'none' (no normalization)")
    p_train.add_argument('--fa_gumbel_softmax', action='store_true', help='Use Gumbel-Softmax for frequency assignment (differentiable)')
    p_train.add_argument('--gumbel_temp', type=float, default=1.0, help='Temperature for Gumbel-Softmax (default: 1.0)')
    p_train.add_argument('--scale_epochs_by_samples', action='store_true', help='Multiply --epochs by n_train_samples to give each sample the full epoch budget (matching dnn_d2d_pytorch single-sample mode)')
    # Discrete power support
    p_train.add_argument('--discrete_power', action='store_true', help='Use discrete power levels instead of continuous range')
    p_train.add_argument('--power_levels', type=float, nargs='+', default=[1e-10, 0.25, 0.5, 1.0], help='Power levels for discrete power mode')

    # ------------------------------------------------------------------
    # validate_dnn (NEW)
    # ------------------------------------------------------------------
    p_validate = subparsers.add_parser("validate_dnn", help="Evaluate a trained DNN on validation set and save YAML results")
    p_validate.add_argument("--config", required=True, type=Path, help="YAML config file")
    p_validate.add_argument("--weights", required=True, type=str, help="Path to trained model weights (.pt)")
    p_validate.add_argument("--val_npy", type=str, default=None, help="Path to .npy file for validation samples (channel gains)")
    p_validate.add_argument("--n_val_samples", type=int, default=100, help="Number of validation samples to generate if no val_npy")
    p_validate.add_argument("--results_path", type=str, default="results_validation.yaml", help="Path to save YAML results")
    p_validate.add_argument("--batch_size", type=int, default=64, help="Batch size for validation (if using .npy)")
    p_validate.add_argument("--soft-fa", action="store_true", help="Use soft FA loss (for completeness, not used in eval)")
    p_validate.add_argument("--hidden_size", type=int, nargs="*", default=None, help="Hidden layer sizes (e.g. 200 200)")
    p_validate.add_argument("--no_batch_norm", action="store_true", help="Disable BatchNorm in DNN model")
    p_validate.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device for DNN (auto, cpu, cuda, mps)')
    p_validate.add_argument('--normalization', type=str, choices=['global', 'local', 'none'], default='global',
        help="Normalization scheme: 'global' (NPY mode), 'local' (per-sample), or 'none' (no normalization)")
    p_validate.add_argument('--fa_gumbel_softmax', action='store_true', help='Use Gumbel-Softmax for frequency assignment (differentiable)')
    p_validate.add_argument('--gumbel_temp', type=float, default=1.0, help='Temperature for Gumbel-Softmax (default: 1.0)')
    # Discrete power eval
    p_validate.add_argument('--discrete_power', action='store_true', help='Assume the network was trained with discrete power levels')
    p_validate.add_argument('--power_levels', type=float, nargs='*', default=[1e-10, 0.25, 0.5, 1.0], help='Discrete power levels list (must match training)')
    return parser


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    """Main entry point for the command-line interface."""
    parser = _parse_args(argv)
    args = parser.parse_args(argv)
    scenario = None
    if args.cmd not in SUBCOMMANDS:
        print(f"Invalid subcommand '{args.cmd}'. Must be one of {SUBCOMMANDS}")
        parser.print_help()
        sys.exit(1)

    if args.cmd == "run":
        cfg = SimulationConfig.from_yaml(args.config)
        cfg.seed = cfg.seed  # use seed from YAML as-is
        # If ML_DNN, pass hidden_size
        if args.algo.lower() == "ml_dnn":
            from .algorithms.ml_dnn import ML_DNN
            algo = ML_DNN(cfg, hidden_size=args.hidden_size)
            from .simulator.engine import SimulationResult
            scenario = None
            from .simulator.scenario import Scenario
            scenario = Scenario.random(cfg, restrict_rx_distance=args.restrict_rx_distance)
            decision = algo.decide(scenario)
            # Reuse run_once logic for result
            from .simulator.engine import SimulationResult
            result = SimulationResult(
                config=cfg,
                algorithm_name=args.algo,
                seed=cfg.seed,
                sum_rate_bps=None,  # Will compute below
                tx_power_dbm=decision["tx_power_dbm"],
                fa_indices=decision["fa_indices"],
                scenario=scenario,
            )
            # Compute sum-rate
            from .simulator.metrics import sinr_linear, sum_rate_bps
            from .simulator.environment import db_to_linear
            tx_power_lin = db_to_linear(result.tx_power_dbm) * 1e-3
            noise_power_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3
            g = scenario.get_channel_gain_with_fa_penalty(result.fa_indices)
            sinr = sinr_linear(
                tx_power_lin=tx_power_lin,
                fa_indices=result.fa_indices,
                channel_gain=g,
                noise_power_lin=noise_power_lin,
            )
            result.sum_rate_bps = sum_rate_bps(sinr, cfg.bandwidth_hz)
        else:
            result = run_once(cfg, args.algo)
        print(f"Sum-rate = {result.sum_rate_bps:.2e} bit/s (algo={args.algo})")

    elif args.cmd == "batch":
        cfg = SimulationConfig.from_yaml(args.config)
        start_seed, end_seed = args.seeds
        cfgs = []
        for seed in range(start_seed, end_seed + 1):
            c = SimulationConfig.from_yaml(args.config)
            c.seed = seed
            cfgs.append(c)

        results = run_batch(cfgs, args.algo, processes=args.processes)
        avg_sum_rate = sum(r.sum_rate_bps for r in results) / len(results)
        print(
            f"Average sum-rate over seeds {start_seed}..{end_seed}: {avg_sum_rate:.2e} bit/s (algo={args.algo})"
        )
    elif args.cmd == "generate-samples":
        generate_samples(
            n_samples=args.n_samples,
            n_pairs=args.n_pairs,
            n_fa=args.n_fa,
            area_size_m=args.area_size_m,
            channel_type=args.channel_type,
            seed=args.seed,
            out_path=args.out_path,
            restrict_rx_distance=args.restrict_rx_distance,
        )
    elif args.cmd == "train_dnn":
        # --- START of self-contained train_dnn command ---
        from .config import SimulationConfig
        from .algorithms.ml_dnn import train_model  # delegate to shared trainer

        cfg = SimulationConfig.from_yaml(args.config)

        # train_model handles weight saving, meta-file, early-stopping, etc.
        # We simply forward the CLI options.
        algo, losses, _ = train_model(
            cfg,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.save_path,
            patience=args.patience,
            train_npy=args.train_npy,
            n_train_samples=args.n_train_samples,
            batch_norm=(not args.no_batch_norm),
            model_seed=args.model_seed,
            device=args.device,
            normalization=args.normalization,
            soft_fa=args.soft_fa,
            shuffle_data=(not args.no_shuffle),
            discrete_power=args.discrete_power,
            power_levels=args.power_levels,
        )

        # Simple confirmation message
        if args.save_path:
            print(f"[INFO] Training finished. Weights saved to {args.save_path}")
        else:
            print("[INFO] Training finished (no save_path specified).")

        return
        # --- END of thin train_dnn wrapper ---
    elif args.cmd == "validate_dnn":
        from .config import SimulationConfig
        cfg = SimulationConfig.from_yaml(args.config)
        import torch
        import numpy as np
        from .algorithms.ml_dnn import ML_DNN, dnn_output_to_decision_torch, get_device
        from .algorithms.fs_bruteforce import fs_bruteforce
        from .simulator.scenario import Scenario, ChannelGainDataset
        from .simulator.metrics import sinr_linear, sum_rate_bps, sum_rate_dimensionless
        from .simulator.environment import db_to_linear
        import yaml
        meta_path = args.weights + ".meta.yaml"
        hidden_size = args.hidden_size
        if (hidden_size is None or hidden_size == [] or hidden_size == [None]) and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
                hidden_size = meta.get("hidden_size", None)
        use_batch_norm = (not args.no_batch_norm) and (args.batch_size > 1)
        def resolve_device(device_str):
            import torch
            if device_str == 'auto':
                from .algorithms.ml_dnn import get_device as get_dev
                return get_dev()
            return torch.device(device_str)
        device = resolve_device(args.device)
        algo = ML_DNN(cfg, hidden_size=hidden_size, batch_norm=use_batch_norm, device=device, discrete_power=args.discrete_power, power_levels=args.power_levels)
        algo.load_weights(args.weights)
        model = algo.model
        model.eval()
        input_mean = None
        input_std = None
        normalization = args.normalization
        meta_normalization = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
            meta_normalization = meta.get("normalization", None)
            if normalization == 'global':
                input_mean = meta.get("input_mean", None)
                input_std = meta.get("input_std", None)
                if input_mean is not None and input_std is not None:
                    # Print the same info as in training
                    mean_val = np.mean(input_mean)
                    std_val = np.mean(input_std)
                    print(f"[INFO] Global normalization stats (from training) - Mean: {mean_val:.2f} dB, Std: {std_val:.2f} dB")
            # If normalization not explicitly set, use meta's normalization
            if normalization is None and meta_normalization is not None:
                normalization = meta_normalization
            # Warn if mismatch
            if meta_normalization is not None and normalization != meta_normalization:
                print(f"[WARNING] Requested normalization '{normalization}' does not match model's normalization '{meta_normalization}' in meta.yaml.")
        if args.val_npy is not None:
            val_data = np.load(args.val_npy)
            n_val = val_data.shape[0]
            tx_xy_path = args.val_npy.replace('.npy', '_tx_xy.npy')
            rx_xy_path = args.val_npy.replace('.npy', '_rx_xy.npy')
            try:
                tx_xy_data = np.load(tx_xy_path)
                rx_xy_data = np.load(rx_xy_path)
                has_locations = True
            except FileNotFoundError:
                tx_xy_data = None
                rx_xy_data = None
                has_locations = False
            print("tx_xy_data shape:", None if tx_xy_data is None else tx_xy_data.shape)
            print("rx_xy_data shape:", None if rx_xy_data is None else rx_xy_data.shape)
            print("val_data shape:", val_data.shape)
            if tx_xy_data is not None:
                print("First tx_xy sample:", tx_xy_data[0])
            if rx_xy_data is not None:
                print("First rx_xy sample:", rx_xy_data[0])
            if args.n_val_samples is not None and args.n_val_samples < n_val:
                val_data = val_data[:args.n_val_samples]
                if has_locations:
                    tx_xy_data = tx_xy_data[:args.n_val_samples]
                    rx_xy_data = rx_xy_data[:args.n_val_samples]
                n_val = val_data.shape[0]
        else:
            n_val = args.n_val_samples
            val_data = []
            tx_xy_data = None
            rx_xy_data = None
            for i in range(n_val):
                cfg_val = SimulationConfig.from_yaml(args.config)
                cfg_val.seed = 20000 + i
                scenario = Scenario.random(cfg_val)
                val_data.append(scenario.channel_gains_db())
            val_data = np.stack(val_data)
        all_results = []
        print(f"Evaluating on {n_val} validation samples...")
        for i in range(n_val):
            cfg_val = SimulationConfig.from_yaml(args.config)
            channel_gains_db = val_data[i]
            if tx_xy_data is not None and rx_xy_data is not None:
                tx_xy = tx_xy_data[i]
                rx_xy = rx_xy_data[i]
            else:
                tx_xy = np.zeros((cfg_val.n_pairs, 2))
                rx_xy = np.zeros((cfg_val.n_pairs, 2))
            scenario = Scenario.from_channel_gains(cfg=cfg_val, channel_gains_db=channel_gains_db)
            # Use the original channel gains data and apply normalization
            x_raw = torch.tensor(channel_gains_db, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            # Apply normalization
            if normalization == 'global' and input_mean is not None and input_std is not None:
                input_mean_tensor = torch.tensor(input_mean, dtype=torch.float32, device=device)
                input_std_tensor = torch.tensor(input_std, dtype=torch.float32, device=device)
                x = (x_raw - input_mean_tensor) / input_std_tensor
            elif normalization == 'local':
                means = x_raw.mean(dim=1, keepdim=True)
                stds = x_raw.std(dim=1, keepdim=True) + 1e-8
                x = (x_raw - means) / stds
            elif normalization == 'none':
                x = x_raw
            else:
                x = x_raw
            with torch.no_grad():
                power_lin_ml, fa_probs_ml = model(x)
            # Convert power to dBm and pick hard FA indices
            tx_power_dbm_ml = 10 * np.log10(power_lin_ml.cpu().numpy() + 1e-12) + 30
            fa_indices_ml = np.argmax(fa_probs_ml.cpu().numpy(), axis=2)
            tx_power_lin_ml = power_lin_ml.cpu().numpy()
            noise_power_lin = db_to_linear(cfg_val.noise_power_dbm) * 1e-3
            g_ml = scenario.get_channel_gain_with_fa_penalty(fa_indices_ml.flatten())
            sinr_ml = sinr_linear(
                tx_power_lin=tx_power_lin_ml.flatten(),
                fa_indices=fa_indices_ml.flatten(),
                channel_gain=g_ml,
                noise_power_lin=noise_power_lin,
            )
            sum_rate_ml = sum_rate_bps(sinr_ml, cfg_val.bandwidth_hz)
            sum_rate_ml_dim = sum_rate_dimensionless(sinr_ml)
            fs_algo = fs_bruteforce(cfg_val)
            decision_fs = fs_algo.decide(scenario)
            tx_power_dbm_fs = decision_fs["tx_power_dbm"]
            fa_indices_fs = decision_fs["fa_indices"]
            tx_power_lin_fs = db_to_linear(tx_power_dbm_fs) * 1e-3
            g_fs = scenario.get_channel_gain_with_fa_penalty(fa_indices_fs)
            sinr_fs = sinr_linear(
                tx_power_lin=tx_power_lin_fs,
                fa_indices=fa_indices_fs,
                channel_gain=g_fs,
                noise_power_lin=noise_power_lin,
            )
            sum_rate_fs = sum_rate_bps(sinr_fs, cfg_val.bandwidth_hz)
            sum_rate_fs_dim = sum_rate_dimensionless(sinr_fs)
            ratio = sum_rate_ml_dim / sum_rate_fs_dim if sum_rate_fs_dim != 0 else 0.0
            sample_dict = {
                "index": i,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg_val.__dict__.items()},
                "tx_xy": tx_xy.tolist(),
                "rx_xy": rx_xy.tolist(),
                "channel_gains_db": channel_gains_db.tolist(),
                "dnn": {
                    "tx_power_dbm": tx_power_dbm_ml.tolist(),
                    "fa_indices": fa_indices_ml.tolist(),
                    "sum_rate": sum_rate_ml,
                },
                "fs": {
                    "tx_power_dbm": tx_power_dbm_fs.tolist(),
                    "fa_indices": fa_indices_fs.tolist(),
                    "sum_rate": sum_rate_fs,
                },
                "ratio": ratio,
            }
            all_results.append(sample_dict)
            print(f"{i:<8}{sum_rate_ml:>15.2e}{sum_rate_fs:>15.2e}{ratio:>10.3f}")
        with open(args.results_path, "w") as f:
            yaml.safe_dump({"samples": all_results}, f)
        print(f"Validation results saved to {args.results_path}")
        return
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main(sys.argv[1:]) 