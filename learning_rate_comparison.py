#!/usr/bin/env python3
"""
Learning Rate Comparison Script
Trains DNN models with different learning rates and plots learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from pathlib import Path
from src.algorithms.ml_dnn import train_model
from src.simulator.scenario import SimulationConfig

def run_lr_comparison():
    """Run training with multiple learning rates and compare learning curves."""
    
    # Learning rates to test (logarithmic scale)
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    
    # Training 
    config_path = "cfgs/config_fa1.yaml"
    train_npy = "samples_rural_6pairs_1fa.npy"
    n_train_samples = 1000  # Small for quick comparison
    epochs = 200
    batch_size = 32
    hidden_sizes = [200, 200]
    
    # Load config
    cfg = SimulationConfig.from_yaml(config_path)
    
    # Storage for results
    results = {}
    
    print("=== Learning Rate Comparison ===")
    print(f"Config: {config_path}")
    print(f"Training data: {train_npy}")
    print(f"Samples: {n_train_samples}, Epochs: {epochs}")
    print(f"Learning rates: {learning_rates}")
    print()
    
    # Run training for each learning rate
    for lr in learning_rates:
        print(f"Training with LR = {lr:.1e}...")
        
        try:
            # Train model
            save_path = f"lr_comparison_lr{lr:.1e}.pt"
            algo, losses, meta = train_model(
                cfg=cfg,
                hidden_size=hidden_sizes,
                epochs=epochs,
                lr=lr,
                verbose=False,  # Reduce output
                save_path=save_path,
                patience=epochs,  # No early stopping
                soft_fa=True,  # Use corrected soft FA
                train_npy=train_npy,
                batch_size=batch_size,
                n_train_samples=n_train_samples,
                device="mps"
            )
            
            # Store results
            results[lr] = {
                'losses': losses,
                'final_loss': losses[-1] if losses else float('inf'),
                'convergence_epoch': len(losses),
                'meta': meta
            }
            
            print(f"  Final loss: {losses[-1]:.2e}, Converged in {len(losses)} epochs")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[lr] = {'losses': [], 'final_loss': float('inf'), 'error': str(e)}
    
    # Plot learning curves
    plot_learning_curves(results, learning_rates)
    
    # Find best learning rate
    best_lr = find_best_lr(results)
    print(f"\n=== RECOMMENDATION ===")
    print(f"Best learning rate: {best_lr:.1e}")
    
    return results

def plot_learning_curves(results, learning_rates):
    """Plot learning curves for all learning rates."""
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Color palette
    colors = sns.color_palette("husl", len(learning_rates))
    
    # Plot 1: Learning curves (log scale)
    plt.subplot(2, 2, 1)
    for i, lr in enumerate(learning_rates):
        if 'losses' in results[lr] and results[lr]['losses']:
            losses = results[lr]['losses']
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, [-loss for loss in losses], 
                    label=f'LR = {lr:.1e}', color=colors[i], linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Sum Rate (bit/s)')
    plt.title('Learning Curves - Sum Rate vs Epoch')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves (negative sum rate)
    plt.subplot(2, 2, 2)
    for i, lr in enumerate(learning_rates):
        if 'losses' in results[lr] and results[lr]['losses']:
            losses = results[lr]['losses']
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, 
                    label=f'LR = {lr:.1e}', color=colors[i], linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Negative Sum Rate (Loss)')
    plt.title('Loss Curves - Training Loss vs Epoch')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final performance comparison
    plt.subplot(2, 2, 3)
    final_losses = []
    valid_lrs = []
    for lr in learning_rates:
        if 'final_loss' in results[lr] and results[lr]['final_loss'] != float('inf'):
            final_losses.append(-results[lr]['final_loss'])  # Convert to sum rate
            valid_lrs.append(lr)
    
    if final_losses:
        bars = plt.bar(range(len(valid_lrs)), final_losses, color=colors[:len(valid_lrs)])
        plt.xlabel('Learning Rate')
        plt.ylabel('Final Sum Rate (bit/s)')
        plt.title('Final Performance Comparison')
        plt.xticks(range(len(valid_lrs)), [f'{lr:.1e}' for lr in valid_lrs], rotation=45)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Highlight best performance
        if final_losses:
            best_idx = np.argmax(final_losses)
            bars[best_idx].set_color('red')
            bars[best_idx].set_alpha(0.8)
    
    # Plot 4: Convergence speed
    plt.subplot(2, 2, 4)
    convergence_epochs = []
    conv_lrs = []
    for lr in learning_rates:
        if 'convergence_epoch' in results[lr]:
            convergence_epochs.append(results[lr]['convergence_epoch'])
            conv_lrs.append(lr)
    
    if convergence_epochs:
        plt.plot(conv_lrs, convergence_epochs, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Learning Rate')
        plt.ylabel('Epochs to Convergence')
        plt.title('Convergence Speed')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Learning curves saved to: learning_rate_comparison.png")

def find_best_lr(results):
    """Find the best learning rate based on final performance."""
    best_lr = None
    best_performance = float('-inf')
    
    for lr, result in results.items():
        if 'final_loss' in result and result['final_loss'] != float('inf'):
            # Higher sum rate is better (negative loss)
            performance = -result['final_loss']
            if performance > best_performance:
                best_performance = performance
                best_lr = lr
    
    return best_lr

def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n=== DETAILED RESULTS ===")
    print(f"{'LR':<10} {'Final Sum Rate':<15} {'Epochs':<8} {'Status':<10}")
    print("-" * 50)
    
    for lr in sorted(results.keys()):
        result = results[lr]
        if 'error' in result:
            print(f"{lr:<10.1e} {'ERROR':<15} {'-':<8} {'Failed':<10}")
        else:
            final_rate = -result['final_loss']
            epochs = result['convergence_epoch']
            status = 'Success'
            print(f"{lr:<10.1e} {final_rate:<15.2e} {epochs:<8} {status:<10}")

if __name__ == "__main__":
    # Run the comparison
    results = run_lr_comparison()
    
    # Print detailed summary
    print_summary_table(results) 