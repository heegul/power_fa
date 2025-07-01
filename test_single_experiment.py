#!/usr/bin/env python3
"""
Test Single DNN Experiment
==========================

Quick test to verify the grid search setup works correctly.
"""

import subprocess
import sys
import yaml

def test_single_experiment():
    """Test a single DNN training experiment."""
    
    print("🧪 Testing single DNN experiment...")
    print("Configuration: [128, 64], lr=3e-4, 5 samples, 1000 epochs")
    
    cmd = [
        'python', '-m', 'src.cli', 'train_dnn',
        '--config', 'cfgs/debug.yaml',
        '--n_eval_samples', '5',
        '--epochs', '1000',
        '--lr', '3e-4',
        '--results_path', 'test_single_result.yaml',
        '--device', 'cpu',
        '--patience', '500',
        '--soft-fa',
        '--hidden_size', '128', '64'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            
            # Extract results
            with open('test_single_result.yaml', 'r') as f:
                results = yaml.safe_load(f)
            
            ratios = [sample['ratio'] for sample in results['samples']]
            avg_ratio = sum(ratios) / len(ratios)
            perfect_count = sum(1 for r in ratios if r >= 0.999)
            
            print(f"📊 Results:")
            print(f"   Average ratio: {avg_ratio:.3f}")
            print(f"   Perfect ratios: {perfect_count}/{len(ratios)}")
            print(f"   Min ratio: {min(ratios):.3f}")
            print(f"   Max ratio: {max(ratios):.3f}")
            
            if avg_ratio > 0.8:
                print("🎉 Test PASSED - Good performance!")
                return True
            else:
                print("⚠️  Test WARNING - Low performance, but training works")
                return True
                
        else:
            print(f"❌ Training failed!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"💥 Exception: {e}")
        return False

if __name__ == '__main__':
    success = test_single_experiment()
    
    if success:
        print("\n✅ Grid search setup is ready!")
        print("You can now run:")
        print("  ./run_grid_search.sh --quick")
        print("  python quick_grid_search.py")
        sys.exit(0)
    else:
        print("\n❌ Setup test failed. Please check your configuration.")
        sys.exit(1) 