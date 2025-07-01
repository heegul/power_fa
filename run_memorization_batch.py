#!/usr/bin/env python3
"""
Batch script to run memorization experiments with various network architectures.
This script systematically explores different hidden layer configurations to study
the relationship between network capacity and memorization performance.

Enhanced with parallel processing for Mac Mini with 64GB unified memory.
"""

import subprocess
import time
import os
import sys
from datetime import datetime
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue

# Define network architectures to test
NETWORK_ARCHITECTURES = [
    # Small networks
    [32, 32],
    [64, 64],
    [100, 100],
    
    # Medium networks  
    [128, 128],
    [200, 200],
    [256, 256],
    
    # Large networks
    [400, 400],
    [512, 512],
    [1024, 1024],
    [2048, 2048],  # Added even larger network
    
    # Deep networks
    [128, 64, 128],
    [200, 100, 200],
    [256, 128, 256],
    [512, 256, 512],
    [1024, 512, 1024],  # Added even larger deep network
    
    # Very deep networks
    [128, 64, 32, 64, 128],
    [200, 100, 50, 100, 200],
    [512, 256, 128, 256, 512],
    [1024, 512, 256, 512, 1024],  # Added even larger very deep network
]

def calculate_parameters(hidden_sizes, input_size=12, output_size=6):
    """Calculate total number of parameters in the network"""
    total_params = 0
    
    # Input to first hidden layer
    total_params += input_size * hidden_sizes[0] + hidden_sizes[0]
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        total_params += hidden_sizes[i] * hidden_sizes[i+1] + hidden_sizes[i+1]
    
    # Last hidden to output
    total_params += hidden_sizes[-1] * output_size + output_size
    
    return total_params

def run_single_experiment(args_tuple):
    """Run a single memorization experiment - designed for parallel execution"""
    arch_idx, hidden_size, target_ratios, epochs, device, force_overwrite = args_tuple
    
    # Convert hidden_size list to command line arguments
    hidden_size_str = ' '.join(map(str, hidden_size))
    hidden_size_filename = '_'.join(map(str, hidden_size))
    
    # Calculate parameters for logging
    total_params = calculate_parameters(hidden_size)
    
    # Construct command
    cmd = [
        'python', 'n_sample_memorization.py',
        '--hidden_size'] + [str(h) for h in hidden_size] + [
        '--target_ratios', str(target_ratios),
        '--epochs', str(epochs),
        '--device', device
    ]
    
    results_file = f"n_sample_memorization_results_{hidden_size_filename}.yaml"
    
    # Check if results already exist
    if os.path.exists(results_file) and not force_overwrite:
        return {
            'arch_idx': arch_idx,
            'hidden_size': hidden_size,
            'success': True,
            'duration': 0,
            'message': f"Skipped - results file {results_file} already exists",
            'total_params': total_params
        }
    
    # Run the experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'arch_idx': arch_idx,
            'hidden_size': hidden_size,
            'success': True,
            'duration': duration,
            'message': f"SUCCESS: Completed in {duration:.1f}s, saved to {results_file}",
            'total_params': total_params
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        error_msg = f"FAILED after {duration:.1f}s: {e.returncode}"
        if e.stderr:
            error_msg += f" - {e.stderr[:200]}"
        
        return {
            'arch_idx': arch_idx,
            'hidden_size': hidden_size,
            'success': False,
            'duration': duration,
            'message': error_msg,
            'total_params': total_params
        }
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'arch_idx': arch_idx,
            'hidden_size': hidden_size,
            'success': False,
            'duration': duration,
            'message': f"EXCEPTION after {duration:.1f}s: {str(e)}",
            'total_params': total_params
        }

def estimate_memory_usage(hidden_size, target_ratios=200):
    """Estimate memory usage for an experiment (rough approximation)"""
    total_params = calculate_parameters(hidden_size)
    # Rough estimate: parameters * 4 bytes + overhead + data
    # This is a conservative estimate for Apple Silicon
    base_memory_mb = (total_params * 4) / (1024 * 1024)  # Model parameters
    data_memory_mb = target_ratios * 0.1  # Data overhead
    overhead_mb = 500  # Python + framework overhead
    
    return base_memory_mb + data_memory_mb + overhead_mb

def determine_optimal_workers(architectures_to_run, available_memory_gb=64):
    """Determine optimal number of parallel workers based on memory and architecture"""
    # Conservative estimate for Mac Mini with 64GB
    available_memory_mb = available_memory_gb * 1024 * 0.8  # Use 80% of available memory
    
    # Estimate memory usage for largest architecture
    max_memory_per_experiment = max(
        estimate_memory_usage(arch[1]) for arch in architectures_to_run
    )
    
    # Calculate based on memory
    memory_based_workers = int(available_memory_mb / max_memory_per_experiment)
    
    # Calculate based on CPU cores (Apple Silicon typically has 8-10 cores)
    cpu_cores = mp.cpu_count()
    cpu_based_workers = max(1, cpu_cores - 1)  # Leave one core free
    
    # Use the minimum to be safe, but at least 2 workers
    optimal_workers = max(2, min(memory_based_workers, cpu_based_workers, 6))
    
    return optimal_workers, max_memory_per_experiment

def main():
    parser = argparse.ArgumentParser(description='Batch runner for memorization experiments with parallel processing')
    parser.add_argument('--target_ratios', type=int, default=200, 
                       help='Target number of ratio measurements per N value')
    parser.add_argument('--epochs', type=int, default=1000, 
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'],
                       help='Training device')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be run without actually running')
    parser.add_argument('--start_from', type=int, default=0,
                       help='Start from architecture index (for resuming)')
    parser.add_argument('--architectures', nargs='*', type=int,
                       help='Run only specific architecture indices (e.g., --architectures 0 2 5)')
    parser.add_argument('--list_architectures', action='store_true',
                       help='List all available architectures and exit')
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of parallel workers (0 = auto-detect based on memory)')
    parser.add_argument('--force_overwrite', action='store_true',
                       help='Overwrite existing result files without asking')
    parser.add_argument('--memory_gb', type=int, default=64,
                       help='Available memory in GB for worker calculation (default: 64)')
    args = parser.parse_args()
    
    if args.list_architectures:
        print("Available network architectures:")
        print("="*60)
        for i, arch in enumerate(NETWORK_ARCHITECTURES):
            params = calculate_parameters(arch)
            memory_est = estimate_memory_usage(arch, args.target_ratios)
            print(f"{i:2d}: {arch} ({params:,} parameters, ~{memory_est:.0f}MB)")
        return
    
    # Determine which architectures to run
    if args.architectures:
        architectures_to_run = [(i, NETWORK_ARCHITECTURES[i]) for i in args.architectures 
                               if 0 <= i < len(NETWORK_ARCHITECTURES)]
        if len(architectures_to_run) != len(args.architectures):
            print("⚠️  Some architecture indices are invalid!")
            return
    else:
        architectures_to_run = [(i, arch) for i, arch in enumerate(NETWORK_ARCHITECTURES) 
                               if i >= args.start_from]
    
    # Determine optimal number of workers
    if args.workers == 0:
        optimal_workers, max_memory = determine_optimal_workers(architectures_to_run, args.memory_gb)
    else:
        optimal_workers = args.workers
        max_memory = max(estimate_memory_usage(arch[1]) for arch in architectures_to_run)
    
    print(f"🚀 PARALLEL MEMORIZATION BATCH EXPERIMENT")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target ratios per N: {args.target_ratios}")
    print(f"🔄 Epochs per experiment: {args.epochs}")
    print(f"💻 Device: {args.device}")
    print(f"🏗️  Architectures to run: {len(architectures_to_run)}")
    print(f"⚡ Parallel workers: {optimal_workers}")
    print(f"💾 Estimated max memory per experiment: {max_memory:.0f}MB")
    print(f"🔄 Force overwrite: {args.force_overwrite}")
    
    if args.dry_run:
        print("🧪 DRY RUN MODE: No experiments will actually run")
    
    print("\nArchitectures to be tested:")
    for i, arch in architectures_to_run:
        params = calculate_parameters(arch)
        memory_est = estimate_memory_usage(arch, args.target_ratios)
        print(f"  {i:2d}: {arch} ({params:,} parameters, ~{memory_est:.0f}MB)")
    
    if args.dry_run:
        print(f"\n🧪 DRY RUN: Would run {len(architectures_to_run)} experiments with {optimal_workers} parallel workers")
        return
    
    # Prepare experiment arguments
    experiment_args = [
        (arch_idx, hidden_size, args.target_ratios, args.epochs, args.device, args.force_overwrite)
        for arch_idx, hidden_size in architectures_to_run
    ]
    
    # Run experiments in parallel
    successful = 0
    failed = 0
    total_time = 0
    failed_experiments = []
    completed_experiments = []
    
    start_time = time.time()
    
    print(f"\n🚀 Starting {len(experiment_args)} experiments with {optimal_workers} parallel workers...")
    print("="*80)
    
    try:
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all experiments
            future_to_args = {
                executor.submit(run_single_experiment, args): args 
                for args in experiment_args
            }
            
            # Process completed experiments as they finish
            for future in as_completed(future_to_args):
                result = future.result()
                completed_experiments.append(result)
                
                # Update counters
                if result['success']:
                    successful += 1
                    status_icon = "✅"
                else:
                    failed += 1
                    failed_experiments.append((result['arch_idx'], result['hidden_size']))
                    status_icon = "❌"
                
                total_time += result['duration']
                
                # Print progress
                progress = len(completed_experiments)
                remaining = len(experiment_args) - progress
                
                print(f"{status_icon} [{progress:2d}/{len(experiment_args)}] Arch {result['arch_idx']:2d} "
                      f"{result['hidden_size']} ({result['total_params']:,} params): {result['message']}")
                
                # Show estimated time remaining
                if progress > 0 and remaining > 0:
                    avg_time = total_time / progress
                    estimated_remaining = avg_time * remaining / optimal_workers  # Account for parallelism
                    print(f"    ⏱️  Estimated time remaining: {estimated_remaining/60:.1f} minutes")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Batch run interrupted by user!")
    
    end_time = time.time()
    total_wall_time = end_time - start_time
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 PARALLEL BATCH EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful experiments: {successful}")
    print(f"❌ Failed experiments: {failed}")
    print(f"⏱️  Total CPU time: {total_time/60:.1f} minutes")
    print(f"⏱️  Total wall time: {total_wall_time/60:.1f} minutes")
    print(f"⚡ Speedup factor: {total_time/total_wall_time:.1f}x")
    print(f"👥 Parallel workers used: {optimal_workers}")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments:")
        for arch_idx, hidden_size in failed_experiments:
            print(f"  {arch_idx}: {hidden_size}")
        print(f"\nTo retry failed experiments:")
        failed_indices = [str(arch_idx) for arch_idx, _ in failed_experiments]
        print(f"python run_memorization_batch.py --architectures {' '.join(failed_indices)} --force_overwrite")
    
    if successful > 0:
        print(f"\n📈 To compare results:")
        print(f"python compare_network_capacities.py")
    
    print(f"\n🎉 Parallel batch experiment completed!")

if __name__ == '__main__':
    main() 