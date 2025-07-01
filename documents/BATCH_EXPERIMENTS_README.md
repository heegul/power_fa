# Parallel Memorization Batch Experiments

This system provides automated parallel batch processing for memorization experiments across different network architectures, optimized for Mac Mini with 64GB unified memory and Apple Silicon.

## Key Features

- **Parallel Processing**: Automatically runs multiple experiments simultaneously
- **No Interactive Prompts**: Fully automated execution for unattended runs
- **Memory-Aware**: Auto-detects optimal worker count based on available memory
- **Apple Silicon Optimized**: Leverages MPS (Metal Performance Shaders) acceleration
- **Resume Capability**: Can restart from specific architecture indices
- **Comprehensive Logging**: Real-time progress with time estimation

## Quick Start

### 1. Run Small Test (Parallel)
```bash
# Quick test with 3 small networks running in parallel
./run_batch.sh quick --force_overwrite
```

### 2. Run All Architectures (Parallel)
```bash
# All 13 architectures with auto-detected parallel workers
./run_batch.sh all --force_overwrite
```

### 3. Custom Parallel Configuration
```bash
# Specify exact worker count and architectures
./run_batch.sh custom --architectures 0 2 4 --workers 4 --force_overwrite
```

## Parallel Processing Details

### Automatic Worker Detection
The system automatically determines optimal parallel workers based on:
- **Memory**: Estimates memory usage per experiment (~520MB each)
- **CPU Cores**: Uses CPU count minus 1 (leaves one core free)
- **Conservative Limits**: Maximum 6 workers for stability

### Memory Estimation
For Mac Mini with 64GB unified memory:
- Uses 80% of available memory (51.2GB)
- Each experiment: ~520MB estimated usage
- Theoretical max: ~98 parallel workers
- Practical limit: 6 workers (CPU + stability considerations)

### Performance Benefits
- **Speedup**: Typically 3-6x faster than sequential execution
- **Efficiency**: Better resource utilization on multi-core Apple Silicon
- **Scalability**: Automatically adapts to available resources

## Available Presets

| Preset    | Architectures | Description | Parallel Benefit |
|-----------|---------------|-------------|------------------|
| `quick`   | 0,1,2         | Small networks (32x32 to 100x100) | 3x speedup |
| `medium`  | 3,4,5         | Medium networks (128x128 to 256x256) | 3x speedup |
| `large`   | 6,7,8         | Large networks (400x400, 512x512, 1024x1024) | 3x speedup |
| `xlarge`  | 9             | Extra large network (2048x2048) | 1x |
| `deep`    | 10,11,12,13,14| Multi-layer networks (3-4 layers, up to 1024 units) | 5x speedup |
| `verydeep`| 15,16,17,18   | Very deep networks (5 layers, up to 1024 units) | 4x speedup |
| `all`     | 0-18          | All 19 architectures | 6x speedup |
| `custom`  | User-defined  | Specify exact indices | Variable |

## Command Options

### Shell Script (`run_batch.sh`)
```bash
./run_batch.sh [PRESET] [OPTIONS]

OPTIONS:
  --target_ratios N    Target ratios per N (default: 200)
  --epochs N           Training epochs (default: 1000)
  --device DEVICE      Training device: cpu/cuda/mps (default: mps)
  --workers N          Parallel workers (default: auto-detect)
  --force_overwrite    Overwrite existing results without asking
  --dry_run           Show what would run without executing
```

### Python Script (`run_memorization_batch.py`)
```bash
python run_memorization_batch.py [OPTIONS]

OPTIONS:
  --architectures N [N ...]    Specific architecture indices
  --workers N                  Number of parallel workers (0=auto)
  --memory_gb N               Available memory for calculation (default: 64)
  --force_overwrite           Skip existing files without prompting
  --start_from N              Resume from architecture index N
```

## Architecture Table

| Index | Architecture | Parameters | Est. Memory | Complexity |
|-------|-------------|------------|-------------|------------|
| 0 | [32, 32] | 1,670 | ~520MB | Small |
| 1 | [64, 64] | 5,382 | ~520MB | Small |
| 2 | [100, 100] | 12,006 | ~520MB | Small |
| 3 | [128, 128] | 18,950 | ~520MB | Medium |
| 4 | [200, 200] | 44,006 | ~520MB | Medium |
| 5 | [256, 256] | 70,662 | ~520MB | Medium |
| 6 | [400, 400] | 168,006 | ~521MB | Large |
| 7 | [512, 512] | 272,390 | ~521MB | Large |
| 8 | [1024, 1024] | 1,050,626 | ~525MB | Large |
| 9 | [2048, 2048] | 4,198,402 | ~537MB | XLarge |
| 10 | [128, 64, 128] | 19,014 | ~520MB | Deep |
| 11 | [200, 100, 200] | 44,106 | ~520MB | Deep |
| 12 | [256, 128, 256] | 70,790 | ~520MB | Deep |
| 13 | [512, 256, 512] | 273,154 | ~521MB | Deep |
| 14 | [1024, 512, 1024] | 1,051,650 | ~525MB | Deep |
| 15 | [128, 64, 32, 64, 128] | 23,206 | ~520MB | Very Deep |
| 16 | [200, 100, 50, 100, 200] | 54,256 | ~520MB | Very Deep |
| 17 | [512, 256, 128, 256, 512] | 274,498 | ~521MB | Very Deep |
| 18 | [1024, 512, 256, 512, 1024] | 1,054,338 | ~525MB | Very Deep |

## Usage Examples

### Basic Parallel Execution
```bash
# No prompts - runs immediately with parallel processing
./run_batch.sh medium --force_overwrite

# Custom worker count
./run_batch.sh all --workers 3 --force_overwrite

# Specific architectures with parallel processing
./run_batch.sh custom --architectures 0 2 4 6 --workers 4
```

### Advanced Parallel Configuration
```bash
# High-performance run with maximum workers
python run_memorization_batch.py --architectures 0 1 2 3 4 5 --workers 6 --force_overwrite

# Memory-constrained run
python run_memorization_batch.py --architectures 6 7 --workers 2 --memory_gb 32

# Resume interrupted parallel batch
python run_memorization_batch.py --start_from 5 --workers 4 --force_overwrite
```

### Dry Run Testing
```bash
# Test parallel configuration without running
./run_batch.sh all --dry_run --workers 8

# See memory estimates and worker calculation
python run_memorization_batch.py --list_architectures
```

## Expected Results and Performance

### Timing Estimates (Mac Mini M2, 64GB)
- **Sequential**: ~13 hours for all architectures (1000 epochs each)
- **Parallel (4 workers)**: ~3-4 hours for all architectures
- **Parallel (6 workers)**: ~2.5-3 hours for all architectures

### Memory Usage
- **Per Experiment**: ~520MB (conservative estimate)
- **4 Workers**: ~2.1GB total
- **6 Workers**: ~3.1GB total
- **Available**: 64GB (plenty of headroom)

### Performance Characteristics
Different architectures show varying memorization capabilities:
- **Small networks** (32x32): May struggle with complex patterns
- **Medium networks** (200x200): Good balance of capacity and efficiency  
- **Large networks** (512x512): High capacity but may overfit
- **Deep networks**: Better feature learning but slower convergence

## Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Reduce worker count
./run_batch.sh all --workers 2

# Or run smaller batches
./run_batch.sh medium --force_overwrite
```

**Interrupted Experiments**
```bash
# Resume from where it stopped (architecture index 5)
python run_memorization_batch.py --start_from 5 --force_overwrite
```

**Device Issues**
```bash
# Force CPU if MPS has issues
./run_batch.sh quick --device cpu --force_overwrite
```

### Performance Optimization

**For Maximum Speed**
```bash
# Use all available workers with MPS
./run_batch.sh all --workers 6 --device mps --force_overwrite
```

**For Stability**
```bash
# Conservative settings
./run_batch.sh all --workers 2 --device mps --force_overwrite
```

## Analysis and Visualization

After running parallel experiments:

```bash
# Compare all results with parallel processing benefits
python compare_network_capacities.py

# The comparison tool automatically detects all result files
# and provides comprehensive analysis across architectures
```

## Research Applications

This parallel batch system enables efficient investigation of:

1. **Network Capacity Effects**: How parameter count affects memorization
2. **Architecture Comparison**: Shallow vs. deep network performance  
3. **Scaling Laws**: Performance trends across network sizes
4. **Efficiency Analysis**: Parameter efficiency vs. absolute performance
5. **Convergence Patterns**: Training dynamics across architectures

The parallel processing significantly reduces experiment time, enabling more comprehensive studies and faster iteration on research questions.

## System Requirements

- **Mac Mini M2 or later** (Apple Silicon recommended)
- **64GB unified memory** (32GB minimum)
- **macOS with MPS support**
- **Python 3.8+** with PyTorch MPS support
- **~10GB free disk space** for all results

## Best Practices

1. **Start Small**: Use `quick` preset to test setup
2. **Use Force Overwrite**: Add `--force_overwrite` for unattended runs
3. **Monitor Resources**: Check Activity Monitor during large batches
4. **Save Results**: Results are automatically saved with architecture info
5. **Regular Analysis**: Run comparisons after each batch completion 