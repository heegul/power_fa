# DNN Hyperparameter Grid Search Tools

This directory contains tools for systematically finding the best DNN architecture and hyperparameters for single-sample training mode.

## 🎯 Overview

The grid search tools test different combinations of:
- **Hidden layer architectures** (various sizes and depths)
- **Learning rates** (from 1e-4 to 1e-2)
- **Training configurations** (epochs, patience, etc.)

## 📁 Files

### Main Scripts
- **`hyperparameter_grid_search.py`** - Comprehensive grid search with 125+ experiments
- **`quick_grid_search.py`** - Fast grid search with 27 carefully selected experiments
- **`run_grid_search.sh`** - Shell script wrapper for easy execution

### Configuration
- Uses existing config files (e.g., `cfgs/debug.yaml`, `cfgs/config_fa1.yaml`)
- Supports single-sample training mode with `--soft-fa`

## 🚀 Quick Start

### Option 1: Shell Script (Recommended)
```bash
# Quick search with defaults (20 samples, 3000 epochs)
./run_grid_search.sh --quick

# Full comprehensive search
./run_grid_search.sh --full --samples 50

# Custom configuration
./run_grid_search.sh --quick --config cfgs/config_fa1.yaml --samples 30
```

### Option 2: Direct Python Execution
```bash
# Quick search
python quick_grid_search.py --config cfgs/debug.yaml --n_samples 20

# Full search
python hyperparameter_grid_search.py --config cfgs/debug.yaml --n_eval_samples 50
```

## 🔧 Command Line Options

### Shell Script Options
```bash
./run_grid_search.sh [OPTIONS]

Options:
  --quick          Run quick grid search (default, ~27 experiments)
  --full           Run full grid search (~125 experiments)
  --config FILE    Config file to use (default: cfgs/debug.yaml)
  --device DEVICE  Device to use (default: cpu)
  --samples N      Number of evaluation samples (default: 20)
  --epochs N       Maximum epochs (default: 3000)
  --help, -h       Show help message
```

### Python Script Options
```bash
# Quick search
python quick_grid_search.py --help

# Full search  
python hyperparameter_grid_search.py --help
```

## 🏗️ Architecture Search Space

### Quick Search (27 experiments)
Tests the most promising architectures:
```python
architectures = [
    [128],                    # Simple baseline
    [256],                    # Larger simple
    [128, 64],               # Two-layer
    [256, 128],              # Larger two-layer
    [128, 128],              # Symmetric
    [256, 128, 64],          # Three-layer pyramid
    [256, 64, 256],          # Bottleneck
    [200, 100, 100, 200],    # Your successful config
    [128, 128, 128],         # Deep uniform
]

learning_rates = [1e-4, 3e-4, 1e-3]
```

### Full Search (125+ experiments)
Comprehensive search including:
- Simple: `[64]`, `[128]`, `[256]`, `[512]`, `[1024]`
- Two-layer: `[128, 64]`, `[256, 128]`, `[512, 256]`, etc.
- Three-layer: `[256, 128, 64]`, `[512, 256, 128]`, etc.
- Bottleneck: `[256, 64, 256]`, `[512, 128, 512]`, etc.
- Deep: `[128, 128, 128, 128]`, `[256, 256, 256, 256]`
- Wide: `[1024]`, `[512, 512]`, `[1024, 512]`

Learning rates: `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`

## 📊 Results Analysis

### Output Files
- **`quick_grid_results.json`** - Quick search results
- **`grid_search_summary.json`** - Full search results  
- **`best_configurations.json`** - Top 10 configurations
- **Individual YAML files** - Detailed results for each experiment

### Key Metrics
- **Average Ratio** - DNN performance vs Full Search baseline
- **Perfect Ratios** - Number of samples with ratio ≥ 0.999
- **Min/Max Ratio** - Range of performance
- **Success Rate** - Percentage of successful experiments

### Example Output
```
🏆 Results Summary:
Rank Architecture         LR       Avg Ratio  Perfect  
---- -------------------- -------- ---------- -------- 
1    [200, 100, 100, 200] 3e-04    0.987      18/20
2    [256, 128, 64]       1e-03    0.982      16/20
3    [128, 128]           3e-04    0.978      15/20
```

## ⚡ Performance Tips

### Quick Testing
```bash
# Fast test with fewer samples
./run_grid_search.sh --quick --samples 10 --epochs 2000
```

### Production Search
```bash
# Thorough search for final results
./run_grid_search.sh --full --samples 50 --epochs 5000
```

### Resume Interrupted Search
```bash
# Full search supports resuming
python hyperparameter_grid_search.py --resume --output_dir existing_results_dir
```

## 🎯 Recommended Workflow

1. **Quick Search First**
   ```bash
   ./run_grid_search.sh --quick --samples 20
   ```

2. **Analyze Top Candidates**
   - Check `quick_grid_results.json`
   - Identify promising architectures

3. **Focused Search**
   - Manually test top 3-5 architectures with more samples
   - Use different configs (FA1 vs FA2, etc.)

4. **Final Validation**
   ```bash
   # Test best config with full evaluation
   python -m src.cli train_dnn --config cfgs/config_fa1.yaml \
     --lr 3e-4 --hidden_size 200 100 100 200 --soft-fa \
     --n_eval_samples 100 --epochs 10000
   ```

## 🔍 Troubleshooting

### Common Issues
- **Timeout errors**: Reduce `--epochs` or increase timeout in scripts
- **Memory issues**: Use smaller architectures or `--device cpu`
- **Config not found**: Check path to config file

### Debug Mode
```bash
# Run single experiment manually
python -m src.cli train_dnn --config cfgs/debug.yaml \
  --lr 3e-4 --hidden_size 128 64 --soft-fa \
  --n_eval_samples 5 --epochs 1000 --device cpu
```

## 📈 Expected Results

Based on your experiments, good configurations typically achieve:
- **Average ratio**: 0.95+ (95% of full search performance)
- **Perfect ratios**: 15+ out of 20 samples
- **Convergence**: Within 1000-3000 epochs

## 🎉 Next Steps

After finding the best architecture:
1. **Validate on different scenarios** (restricted vs unrestricted)
2. **Test with different FA configurations** (FA1 vs FA2)
3. **Scale up evaluation** (100+ samples for final validation)
4. **Document the winning configuration** for production use

---

*Happy hyperparameter hunting! 🔍✨* 