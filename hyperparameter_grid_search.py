#!/usr/bin/env python3
"""
Hyperparameter Grid Search for DNN Training
============================================

This script performs a comprehensive grid search over:
- Hidden layer architectures (different sizes and depths)
- Learning rates
- Other key hyperparameters

For single-sample DNN training mode to find optimal configurations.
"""

import os
import sys
import subprocess
import itertools
import json
import time
from pathlib import Path
import yaml
import argparse

def run_training(config_file, hidden_sizes, lr, results_path, n_eval_samples=50, 
                epochs=5000, patience=1000, device='cpu'):
    """Run a single training experiment with given hyperparameters."""
    
    # Convert hidden_sizes list to command line arguments
    hidden_args = []
    for size in hidden_sizes:
        hidden_args.append(str(size))
    
    cmd = [
        'python', '-m', 'src.cli', 'train_dnn',
        '--config', config_file,
        '--n_eval_samples', str(n_eval_samples),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--results_path', results_path,
        '--device', device,
        '--patience', str(patience),
        '--soft-fa',
        '--hidden_size'
    ] + hidden_args
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Training completed successfully")
            return True, result.stdout, result.stderr
        else:
            print(f"❌ Training failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Training timed out after 1 hour")
        return False, "", "Timeout"
    except Exception as e:
        print(f"💥 Exception during training: {e}")
        return False, "", str(e)

def extract_results(results_path):
    """Extract key metrics from the results file."""
    try:
        with open(results_path, 'r') as f:
            results = yaml.safe_load(f)
        
        # Extract average ratio and other key metrics
        ratios = [sample['ratio'] for sample in results['samples']]
        avg_ratio = sum(ratios) / len(ratios)
        
        return {
            'avg_ratio': avg_ratio,
            'min_ratio': min(ratios),
            'max_ratio': max(ratios),
            'n_samples': len(ratios),
            'ratios_above_0_9': sum(1 for r in ratios if r >= 0.9),
            'ratios_above_0_95': sum(1 for r in ratios if r >= 0.95),
            'perfect_ratios': sum(1 for r in ratios if r >= 0.999)
        }
    except Exception as e:
        print(f"Failed to extract results from {results_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='DNN Hyperparameter Grid Search')
    parser.add_argument('--config', default='cfgs/debug.yaml', help='Config file to use')
    parser.add_argument('--n_eval_samples', type=int, default=50, help='Number of evaluation samples')
    parser.add_argument('--epochs', type=int, default=5000, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=1000, help='Early stopping patience')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--output_dir', default='grid_search_results', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define hyperparameter grid
    hidden_architectures = [

        # Two-layer architectures
        [128, 64],
        [256, 128],
        [512, 256],
        [128, 128],
        [256, 256],
        
        # Three-layer architectures
        [256, 128, 64],
        [512, 256, 128],
        [256, 256, 256],
        [128, 128, 128],
        
        # Bottleneck architectures
        [256, 64, 256],
        [512, 128, 512],
        [128, 32, 128],
        
        # Four-layer architectures

        [200, 100, 100, 200],
        
        # Deep architectures
        [128, 128, 128, 128],

        
        # Wide architectures
  
        [1024, 512],
    ]
    
    learning_rates = [1e-3, 3e-3]
    
    # Results tracking
    results_summary = []
    results_file = output_dir / 'grid_search_summary.json'
    
    # Load existing results if resuming
    if args.resume and results_file.exists():
        with open(results_file, 'r') as f:
            results_summary = json.load(f)
        print(f"Resuming from {len(results_summary)} existing results")
    
    # Get completed experiments
    completed_experiments = set()
    if args.resume:
        for result in results_summary:
            key = (tuple(result['hidden_sizes']), result['learning_rate'])
            completed_experiments.add(key)
    
    total_experiments = len(hidden_architectures) * len(learning_rates)
    current_experiment = 0
    
    print(f"Starting grid search with {total_experiments} total experiments")
    print(f"Hidden architectures: {len(hidden_architectures)}")
    print(f"Learning rates: {learning_rates}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    for hidden_sizes in hidden_architectures:
        for lr in learning_rates:
            current_experiment += 1
            
            # Skip if already completed
            experiment_key = (tuple(hidden_sizes), lr)
            if experiment_key in completed_experiments:
                print(f"[{current_experiment}/{total_experiments}] Skipping completed experiment: {hidden_sizes}, lr={lr}")
                continue
            
            print(f"\n[{current_experiment}/{total_experiments}] Testing: hidden_sizes={hidden_sizes}, lr={lr}")
            
            # Generate unique results filename
            hidden_str = '_'.join(map(str, hidden_sizes))
            lr_str = f"{lr:.0e}".replace('-', 'm')
            results_filename = f"results_h{hidden_str}_lr{lr_str}.yaml"
            results_path = output_dir / results_filename
            
            # Run training
            success, stdout, stderr = run_training(
                config_file=args.config,
                hidden_sizes=hidden_sizes,
                lr=lr,
                results_path=str(results_path),
                n_eval_samples=args.n_eval_samples,
                epochs=args.epochs,
                patience=args.patience,
                device=args.device
            )
            
            # Extract and store results
            experiment_result = {
                'experiment_id': current_experiment,
                'hidden_sizes': hidden_sizes,
                'learning_rate': lr,
                'success': success,
                'results_file': results_filename,
                'timestamp': time.time()
            }
            
            if success:
                metrics = extract_results(results_path)
                if metrics:
                    experiment_result.update(metrics)
                    print(f"📊 Results: avg_ratio={metrics['avg_ratio']:.3f}, "
                          f"perfect={metrics['perfect_ratios']}/{metrics['n_samples']}")
                else:
                    experiment_result['error'] = 'Failed to extract results'
            else:
                experiment_result['error'] = stderr[:500]  # Truncate error message
            
            results_summary.append(experiment_result)
            
            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            # Print progress
            elapsed = time.time() - start_time
            avg_time_per_exp = elapsed / current_experiment
            remaining_time = avg_time_per_exp * (total_experiments - current_experiment)
            
            print(f"⏱️  Elapsed: {elapsed/60:.1f}min, "
                  f"Estimated remaining: {remaining_time/60:.1f}min")
    
    # Generate final summary report
    print(f"\n🎉 Grid search completed!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Analyze results
    successful_results = [r for r in results_summary if r.get('success', False) and 'avg_ratio' in r]
    
    if successful_results:
        # Sort by average ratio
        successful_results.sort(key=lambda x: x['avg_ratio'], reverse=True)
        
        print(f"\n📈 Top 10 configurations by average ratio:")
        print(f"{'Rank':<4} {'Hidden Sizes':<20} {'LR':<8} {'Avg Ratio':<10} {'Perfect':<8} {'Min':<8}")
        print("-" * 70)
        
        for i, result in enumerate(successful_results[:10]):
            hidden_str = str(result['hidden_sizes'])
            lr_str = f"{result['learning_rate']:.0e}"
            avg_ratio = result['avg_ratio']
            perfect = result.get('perfect_ratios', 0)
            n_samples = result.get('n_samples', 0)
            min_ratio = result.get('min_ratio', 0)
            
            print(f"{i+1:<4} {hidden_str:<20} {lr_str:<8} {avg_ratio:<10.3f} "
                  f"{perfect}/{n_samples:<8} {min_ratio:<8.3f}")
        
        # Save detailed report
        report_file = output_dir / 'best_configurations.json'
        with open(report_file, 'w') as f:
            json.dump(successful_results[:10], f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        print(f"📄 Best configurations saved to: {report_file}")
        
        # Best configuration
        best = successful_results[0]
        print(f"\n🏆 Best configuration:")
        print(f"   Hidden sizes: {best['hidden_sizes']}")
        print(f"   Learning rate: {best['learning_rate']}")
        print(f"   Average ratio: {best['avg_ratio']:.3f}")
        print(f"   Perfect ratios: {best.get('perfect_ratios', 0)}/{best.get('n_samples', 0)}")
    
    else:
        print("❌ No successful experiments found!")

if __name__ == '__main__':
    main() 