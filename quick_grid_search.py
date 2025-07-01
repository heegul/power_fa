#!/usr/bin/env python3
"""
Quick Hyperparameter Grid Search for DNN Training
=================================================

A faster version with fewer combinations for quick testing.
"""

import subprocess
import json
import time
from pathlib import Path
import yaml
import argparse

def run_training(hidden_sizes, lr, config='cfgs/debug.yaml', n_samples=20, epochs=3000):
    """Run a single training experiment."""
    
    hidden_args = [str(size) for size in hidden_sizes]
    hidden_str = '_'.join(hidden_args)
    lr_str = f"{lr:.0e}".replace('-', 'm')
    results_path = f"quick_results_h{hidden_str}_lr{lr_str}.yaml"
    
    cmd = [
        'python', '-m', 'src.cli', 'train_dnn',
        '--config', config,
        '--n_eval_samples', str(n_samples),
        '--n_train_samples', str(n_samples),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--results_path', results_path,
        '--device', 'cpu',
        '--patience', '1000',
        '--soft-fa',
        '--hidden_size'
    ] + hidden_args
    
    print(f"🚀 Testing: {hidden_sizes}, lr={lr}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0 and Path(results_path).exists():
            try:
                with open(results_path, 'r') as f:
                    results = yaml.safe_load(f)
                ratios = [sample['ratio'] for sample in results['samples']]
            except Exception as e:
                print(f"⚠️  Could not parse {results_path}: {e}")
                ratios = []
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                perfect_count = sum(1 for r in ratios if r >= 0.999)
                print(f"✅ Avg ratio: {avg_ratio:.3f}, Perfect: {perfect_count}/{len(ratios)}")
                return {
                    'hidden_sizes': hidden_sizes,
                    'learning_rate': lr,
                    'avg_ratio': avg_ratio,
                    'perfect_ratios': perfect_count,
                    'total_samples': len(ratios),
                    'min_ratio': min(ratios) if ratios else 0,
                    'max_ratio': max(ratios) if ratios else 0,
                    'success': True
                }
            else:
                print("⚠️  No ratios extracted – marking as failure")
        else:
            print(f"❌ Failed (returncode={result.returncode}): {result.stderr[:120]}")
        return {'hidden_sizes': hidden_sizes, 'learning_rate': lr, 'success': False}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout")
        return {'hidden_sizes': hidden_sizes, 'learning_rate': lr, 'success': False, 'error': 'timeout'}

def main():
    parser = argparse.ArgumentParser(description='Quick DNN Hyperparameter Search')
    parser.add_argument('--config', default='cfgs/debug.yaml', help='Config file')
    parser.add_argument('--n_samples', type=int, default=20, help='Number of evaluation samples')
    parser.add_argument('--epochs', type=int, default=3000, help='Maximum epochs')
    
    args = parser.parse_args()
    
    # Quick grid - most promising architectures based on your experiments
    architectures = [
        [128],           # Simple baseline
        [256],           # Larger simple
        [128, 64],       # Two-layer
        [256, 128],      # Larger two-layer
        [128, 128],      # Symmetric
        [256, 128, 64],  # Three-layer pyramid
        [256, 64, 256],  # Bottleneck
        [200, 100, 100, 200],  # Your successful config
        [128, 128, 128], # Deep uniform
    ]
    
    learning_rates = [1e-4, 3e-4, 1e-3]  # Most common good values
    
    results = []
    total = len(architectures) * len(learning_rates)
    current = 0
    
    print(f"🎯 Quick grid search: {total} experiments")
    start_time = time.time()
    
    for arch in architectures:
        for lr in learning_rates:
            current += 1
            print(f"\n[{current}/{total}] ", end="")
            
            result = run_training(arch, lr, args.config, args.n_samples, args.epochs)
            results.append(result)
            
            # Show progress
            elapsed = time.time() - start_time
            remaining = (elapsed / current) * (total - current)
            print(f"⏱️  {elapsed/60:.1f}min elapsed, {remaining/60:.1f}min remaining")
    
    # Analyze results
    successful = [r for r in results if r.get('success', False)]
    successful.sort(key=lambda x: x.get('avg_ratio', 0), reverse=True)
    
    print(f"\n🏆 Results Summary:")
    print(f"{'Rank':<4} {'Architecture':<20} {'LR':<8} {'Avg Ratio':<10} {'Perfect':<8}")
    print("-" * 60)
    
    for i, result in enumerate(successful[:10]):
        arch_str = str(result['hidden_sizes'])
        lr_str = f"{result['learning_rate']:.0e}"
        avg_ratio = result.get('avg_ratio', 0)
        perfect = result.get('perfect_ratios', 0)
        total_samples = result.get('total_samples', 0)
        
        print(f"{i+1:<4} {arch_str:<20} {lr_str:<8} {avg_ratio:<10.3f} {perfect}/{total_samples}")
    
    # Save results
    with open('quick_grid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if successful:
        best = successful[0]
        print(f"\n🥇 Best configuration:")
        print(f"   Architecture: {best['hidden_sizes']}")
        print(f"   Learning rate: {best['learning_rate']}")
        print(f"   Average ratio: {best.get('avg_ratio', 0):.3f}")
        
        # Generate command for best config
        hidden_args = ' '.join(map(str, best['hidden_sizes']))
        print(f"\n📋 Command to reproduce best result:")
        print(f"python -m src.cli train_dnn --config {args.config} --lr {best['learning_rate']} --hidden_size {hidden_args} --soft-fa")

if __name__ == '__main__':
    main() 