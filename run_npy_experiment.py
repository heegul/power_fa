#!/usr/bin/env python3
"""
Runs a single training and validation experiment using specified NPY files.

This script is designed to be called by a batch runner. It takes training
and validation NPY files, trains a model, validates it, and produces a
YAML file with the performance results, which can then be used for
comparison and plotting.
"""

import numpy as np
import os
import subprocess
import yaml
import argparse
import tempfile
from pathlib import Path

def run_experiment(args):
    """
    Main function to run a single training and validation experiment.
    """
    print("🚀 Starting NPY-based experiment...")
    print(f"   Training data: {args.train_npy}")
    print(f"   Validation data: {args.val_npy}")
    print(f"   Config: {args.config}")
    print(f"   Hidden Size: {args.hidden_size}")
    print("="*50)

    # --- 1. Prepare Data ---
    try:
        train_data_full = np.load(args.train_npy)
        val_data_full = np.load(args.val_npy)
    except FileNotFoundError as e:
        print(f"❌ Error: NPY file not found: {e}")
        return

    # Determine number of samples to use
    n_train = min(args.n_train_samples, len(train_data_full)) if args.n_train_samples else len(train_data_full)
    n_val = min(args.n_val_samples, len(val_data_full)) if args.n_val_samples else len(val_data_full)

    print(f"Using {n_train} training samples and {n_val} validation samples.")

    # Create temporary NPY files for the selected subset of data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_train_npy = Path(tmpdir) / "train_subset.npy"
        tmp_val_npy = Path(tmpdir) / "val_subset.npy"
        
        np.save(tmp_train_npy, train_data_full[:n_train])
        np.save(tmp_val_npy, val_data_full[:n_val])

        # --- 2. Train the Model ---
        print("\n🔧 Training model...")
        hidden_size_str = ' '.join(map(str, args.hidden_size))
        tmp_weights_path = Path(tmpdir) / "model_weights.pt"
        tmp_train_results = Path(tmpdir) / "train_results.yaml"

        train_cmd = (
            f"python -m src.cli train_dnn "
            f"--config {args.config} "
            f"--epochs {args.epochs} "
            f"--lr {args.lr} "
            f"--results_path {tmp_train_results} "
            f"--device {args.device} "
            f"--patience {args.patience} "
            f"--soft-fa "
            f"--hidden_size {hidden_size_str} "
            f"--train_npy {tmp_train_npy} "
            f"--n_train_samples {n_train} "
            f"--save_path {tmp_weights_path}"
        )
        
        try:
            subprocess.run(train_cmd, shell=True, check=True, capture_output=True, text=True)
            print("✅ Training complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed!")
            print(f"   Command: {train_cmd}")
            print(f"   Stderr: {e.stderr}")
            return

        # --- 3. Validate the Model ---
        print("\n🔎 Validating model...")
        tmp_val_results = Path(tmpdir) / "val_results.yaml"

        val_cmd = (
            f"python -m src.cli validate_dnn "
            f"--config {args.config} "
            f"--weights {tmp_weights_path} "
            f"--n_val_samples {n_val} "
            f"--results_path {tmp_val_results} "
            f"--val_npy {tmp_val_npy}"
        )
        
        try:
            subprocess.run(val_cmd, shell=True, check=True, capture_output=True, text=True)
            print("✅ Validation complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Validation failed!")
            print(f"   Command: {val_cmd}")
            print(f"   Stderr: {e.stderr}")
            return

        # --- 4. Process and Save Final Results ---
        print("\n💾 Processing and saving final results...")
        try:
            with open(tmp_val_results, 'r') as f:
                val_yaml = yaml.safe_load(f)
            
            # Extract ratios and other relevant data
            ratios = [float(s['ratio']) for s in val_yaml.get('samples', [])]
            
            final_results = {
                'metadata': {
                    'train_npy': args.train_npy,
                    'val_npy': args.val_npy,
                    'config': args.config,
                    'hidden_size': [int(x) for x in args.hidden_size],
                    'n_train_samples_used': int(n_train),
                    'n_val_samples_used': int(n_val),
                    'epochs': int(args.epochs),
                    'lr': float(args.lr),
                    'device': args.device,
                    'patience': int(args.patience),
                },
                'results': {
                    'ratios': [float(r) for r in ratios],
                    'mean_ratio': float(np.mean(ratios)) if ratios else 0.0,
                    'std_ratio': float(np.std(ratios)) if ratios else 0.0,
                    'median_ratio': float(np.median(ratios)) if ratios else 0.0,
                }
            }
            
            # Save the final aggregated YAML
            os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
            with open(args.results_path, 'w') as f:
                yaml.safe_dump(final_results, f, sort_keys=False)
                
            print(f"✅ Final results saved to: {args.results_path}")

        except Exception as e:
            print(f"❌ Failed to process and save results: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a DNN training and validation experiment with NPY files.")
    
    # --- File/Config Arguments ---
    parser.add_argument('--train_npy', type=str, required=True, help='Path to the training .npy file.')
    parser.add_argument('--val_npy', type=str, required=True, help='Path to the validation .npy file.')
    parser.add_argument('--config', type=str, default='cfgs/config_fa1.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--results_path', type=str, default='results/npy_experiment_results.yaml', help='Path to save the final results YAML file.')

    # --- Model/Training Arguments ---
    parser.add_argument('--hidden_size', nargs='+', type=int, required=True, help='List of hidden layer sizes (e.g., 200 200).')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=2000, help='Patience for early stopping.')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Training device.')

    # --- Data Sampling Arguments ---
    parser.add_argument('--n_train_samples', type=int, default=None, help='Number of samples to use from the training file (default: all).')
    parser.add_argument('--n_val_samples', type=int, default=None, help='Number of samples to use from the validation file (default: all).')

    args = parser.parse_args()
    
    run_experiment(args) 