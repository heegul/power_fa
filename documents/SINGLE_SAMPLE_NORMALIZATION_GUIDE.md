# Single Sample Mode Normalization Guide

## Overview

This guide walks you through exactly how normalization works in our single sample mode training, showing the code, data flow, and key insights.

## The Complete Flow

### Step 1: Generate Raw Channel Gains

```python
# In train_model() function (src/algorithms/ml_dnn.py:418-420)
scenario = Scenario.random(cfg, restrict_rx_distance=getattr(cfg, 'restrict_rx_distance', False))
```

**Output**: 6×6 matrix of channel gains in dB
```
Raw channel gains (dB):
[[ -64.13,  -87.74, -102.33,  -86.36,  -99.52,  -95.58]
 [ -87.87,  -59.44, -101.04,  -76.85, -101.61,  -92.69]
 [-103.86, -101.70,  -62.21,  -98.68,  -93.31,  -89.86]
 [ -91.46,  -71.68,  -98.40,  -67.44, -100.29,  -87.88]
 [ -99.13, -101.51,  -95.89,  -98.16,  -66.27,  -95.18]
 [ -99.56,  -95.25,  -82.85,  -90.69,  -94.65,  -69.59]]
```

### Step 2: Flatten for DNN Input

```python
# In train_model() function (src/algorithms/ml_dnn.py:423-424)
x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
sample_flat = x_raw.unsqueeze(0)  # Shape: (1, n_pairs*n_pairs)
```

**Output**: Flattened vector of 36 elements
```
Shape: torch.Size([1, 36])
Values: [-64.13, -87.74, -102.33, ..., -69.59]
Range: -103.86 to -59.44 dB
```

### Step 3: Compute Per-Sample Normalization Statistics

```python
# In train_model() function (src/algorithms/ml_dnn.py:425-426)
input_mean = sample_flat.mean(dim=1, keepdim=True)  # Per-sample mean
input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8  # Per-sample std
```

**Output**: Normalization statistics
```
Input mean: -89.19 dB
Input std: 13.14 dB
```

### Step 4: Apply Normalization

```python
# In train_model() function (src/algorithms/ml_dnn.py:430)
x_normalized = (sample_flat - input_mean) / input_std
```

**Output**: Normalized input for DNN
```
Normalized mean: 0.000000 (perfect!)
Normalized std: 1.000000 (perfect!)
Normalized range: [-1.12, 2.26]
```

### Step 5: Store Original Gains for Loss Calculation

```python
# In train_model() function (src/algorithms/ml_dnn.py:436)
original_gains = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).unsqueeze(0)
```

**Critical Point**: We keep the original unnormalized gains separate!

### Step 6: Training Loop

```python
# In train_model() function (src/algorithms/ml_dnn.py:440-450)
for ep in range(epochs):
    model.train()
    output = model(x_normalized)[0]  # DNN receives NORMALIZED input
    tx_power_dbm, fa_indices, fa_probs = dnn_output_to_decision_torch(output, cfg, device=device)
    
    # CRITICAL: Use ORIGINAL gains for loss calculation
    if soft_fa:
        loss = negative_sum_rate_loss_torch_soft_from_matrix(
            tx_power_dbm.unsqueeze(0), fa_probs.unsqueeze(0), None, cfg, original_gains)
    else:
        loss = negative_sum_rate_loss_torch_from_matrix(
            tx_power_dbm.unsqueeze(0), fa_probs.unsqueeze(0), None, cfg, original_gains)
```

**Key Insight**: 
- DNN input: `x_normalized` (mean=0, std=1)
- Loss calculation: `original_gains` (actual dB values)

### Step 7: Evaluation (Same Normalization)

```python
# In CLI evaluation (src/cli.py:275-280)
x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
sample_flat = x_raw.unsqueeze(0)  # Shape: (1, n_pairs*n_pairs)
input_mean = sample_flat.mean(dim=1, keepdim=True)  # Per-sample mean
input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8  # Per-sample std
x_normalized = (sample_flat - input_mean) / input_std
x = x_normalized
```

**Critical**: Evaluation uses the **exact same normalization** as training!

## Code Locations

### Training Normalization
```python
# File: src/algorithms/ml_dnn.py
# Lines: 418-436 (single sample mode)

# Key code:
x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
sample_flat = x_raw.unsqueeze(0)  # Shape: (1, n_pairs*n_pairs)
input_mean = sample_flat.mean(dim=1, keepdim=True)  # Per-sample mean
input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8  # Per-sample std
x_normalized = (sample_flat - input_mean) / input_std
```

### Evaluation Normalization
```python
# File: src/cli.py
# Lines: 275-280 (train_dnn command)

# Key code:
x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
sample_flat = x_raw.unsqueeze(0)  # Shape: (1, n_pairs*n_pairs)
input_mean = sample_flat.mean(dim=1, keepdim=True)  # Per-sample mean
input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8  # Per-sample std
x_normalized = (sample_flat - input_mean) / input_std
```

### Loss Calculation
```python
# File: src/algorithms/ml_dnn.py
# Lines: 445-450 (training) and 212-253 (loss function)

# Key code:
# CRITICAL: Use original unnormalized gains for loss calculation
loss = negative_sum_rate_loss_torch_from_matrix(
    tx_power_dbm.unsqueeze(0), fa_probs.unsqueeze(0), None, cfg, original_gains)
```

## Why This Approach Works

### 1. **Stable DNN Training**
- **Input**: Normalized gains (mean=0, std=1)
- **Benefit**: Consistent input range for neural network
- **Result**: Stable gradients and faster convergence

### 2. **Accurate Performance Evaluation**
- **Loss Calculation**: Original gains (actual dB values)
- **Benefit**: Accurate SINR computation using real channel conditions
- **Result**: Meaningful loss values and correct optimization

### 3. **Training/Evaluation Consistency**
- **Training**: Per-sample normalization
- **Evaluation**: Same per-sample normalization
- **Benefit**: No distribution shift between training and evaluation
- **Result**: Reliable performance assessment

## Comparison with Reference Implementation

### Our Implementation
```python
# Per-sample normalization
sample_flat = x_raw.unsqueeze(0)  # PyTorch tensor
input_mean = sample_flat.mean(dim=1, keepdim=True)
input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8
x_normalized = (sample_flat - input_mean) / input_std
```

### Reference Implementation
```python
# Per-sample normalization (from compare_implementations.py)
sample_flat = sample.flatten().reshape(1, -1)  # NumPy array
mean = sample_flat.mean(axis=1, keepdims=True)
std = sample_flat.std(axis=1, keepdims=True) + 1e-8
sample_norm = (sample_flat - mean) / std
```

**Result**: Nearly identical (small differences due to PyTorch vs NumPy)
- Mean difference: 0.000005 dB
- Std difference: 0.18 dB

## Key Insights

### 🔑 Critical Separation
```
DNN Input:        Normalized gains (mean=0, std=1)
                  ↓
Loss Calculation: Original gains (actual dB values)
```

This separation allows:
- **Stable training**: Normalized inputs prevent gradient issues
- **Accurate evaluation**: Original gains ensure correct SINR computation
- **Consistent behavior**: Same normalization in training and evaluation

### ⚠️ Common Pitfalls

1. **Using normalized gains for loss**: Would give incorrect SINR values
2. **Different normalization in evaluation**: Would cause distribution shift
3. **Global normalization in single-sample mode**: Would break per-sample consistency

### ✅ Best Practices

1. **Always use original gains for loss/SINR calculation**
2. **Apply same normalization in training and evaluation**
3. **Use per-sample normalization for single-sample mode**
4. **Use global normalization for batch mode**

## Performance Impact

Based on our experiments:

### Normal Scenarios
- **Average ratio**: 0.87-0.95 (excellent performance)
- **Many perfect ratios**: 1.000 (optimal solutions found)

### Restricted RX Distance Scenarios  
- **Average ratio**: 0.68-0.78 (good performance)
- **Improvement over previous**: 10-15% better than scenario-type normalization

## Summary

Single sample mode normalization works by:

1. **Flattening** the 6×6 channel gains matrix to a 36-element vector
2. **Computing** per-sample mean and standard deviation
3. **Normalizing** the input: `(x - mean) / std`
4. **Training** the DNN with normalized input (mean=0, std=1)
5. **Computing loss** using original unnormalized gains
6. **Evaluating** with the same per-sample normalization

This approach ensures stable DNN training while maintaining accurate performance evaluation, leading to the strong results we observe in our experiments. 