# Single Sample Mode Validation: Code Walkthrough

## The Answer: YES, We Use the Exact Same Sample

**Short Answer**: Yes, in single sample mode, we use the **exact same scenario** for both training and validation.

## Code Evidence

### 1. The Critical Line

```python
# File: src/algorithms/ml_dnn.py, Line 418
# Single sample training mode
scenario = Scenario.random(cfg, restrict_rx_distance=getattr(cfg, 'restrict_rx_distance', False))
val_scenario = scenario  # Use same scenario for validation
```

**This line literally assigns the same object to both training and validation!**

### 2. Object Identity Proof

```python
# From our demonstration:
Training scenario ID: 4379837392
Validation scenario ID: 4379837392
Are they the same object? True  # Same memory address!
```

### 3. Channel Gains Are Identical

```python
# Training channel gains:
[[ -64.13,  -87.74, -102.33, ...]]

# Validation channel gains:  
[[ -64.13,  -87.74, -102.33, ...]]  # Exactly the same!

Are gains identical? True
Max difference: 0.0
```

### 4. Normalization Uses Same Data

```python
# File: src/algorithms/ml_dnn.py, Lines 423-430
# Training normalization
x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
sample_flat = x_raw.unsqueeze(0)
input_mean = sample_flat.mean(dim=1, keepdim=True)
input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8
x_normalized = (sample_flat - input_mean) / input_std

# Validation normalization (using SAME scenario and SAME normalization stats)
val_x_raw = torch.tensor(val_scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
val_x_normalized = (val_x_raw.unsqueeze(0) - input_mean) / input_std
```

Since `val_scenario = scenario`, both `x_raw` and `val_x_raw` are identical!

### 5. Training Loop Evidence

```python
# File: src/algorithms/ml_dnn.py, Lines 440-470
for ep in range(epochs):
    # TRAINING
    model.train()
    output = model(x_normalized)[0]  # Uses training scenario
    # ... compute loss using original_gains (from training scenario)
    
    # VALIDATION  
    model.eval()
    with torch.no_grad():
        val_output = model(val_x_normalized)[0]  # Uses validation scenario (same as training!)
        # ... compute loss using original_gains (from training scenario)
```

## Why Do We See Different Train/Val Losses?

Even though we use the same scenario, you might see slightly different training and validation losses:

```
Epoch 1112/10000: Train sum-rate = 1.39e+08 bit/s, Val sum-rate = 1.39e+08 bit/s
```

**Reason**: The small differences come from:
1. **BatchNorm behavior**: 
   - `model.train()`: Uses batch statistics
   - `model.eval()`: Uses running statistics
2. **Dropout** (if enabled): Different random masks in train vs eval mode

## Is This a Problem?

### ⚠️ Potential Issues:
1. **No real validation**: We're testing on training data
2. **Overfitting risk**: Model can memorize the single scenario  
3. **No generalization test**: Can't assess performance on unseen data

### ✅ But It's Actually Reasonable For:
1. **Single scenario optimization**: We want to find the best solution for this specific scenario
2. **Convergence detection**: Early stopping based on when the model stops improving
3. **Our use case**: We're optimizing for specific channel conditions

## What Should We Do?

If we wanted **true validation**, we should:

```python
# Generate training scenario
scenario = Scenario.random(cfg, restrict_rx_distance=restrict_rx)

# Generate DIFFERENT validation scenario  
cfg.seed = cfg.seed + 1000  # Different seed!
val_scenario = Scenario.random(cfg, restrict_rx_distance=restrict_rx)

# Use training scenario's normalization for both
train_flat = torch.tensor(scenario.channel_gains_db()).flatten().unsqueeze(0)
input_mean = train_flat.mean(dim=1, keepdim=True)
input_std = train_flat.std(dim=1, keepdim=True) + 1e-8

# Apply same normalization to both
train_norm = (train_flat - input_mean) / input_std
val_flat = torch.tensor(val_scenario.channel_gains_db()).flatten().unsqueeze(0)  
val_norm = (val_flat - input_mean) / input_std  # Same normalization!
```

## Summary

**Yes, we use exactly the same sample for training and validation in single sample mode.**

This is evident from:
- ✅ Same object reference: `val_scenario = scenario`
- ✅ Identical channel gains: `np.array_equal(train_gains, val_gains) = True`
- ✅ Same normalization: Both use statistics from the same scenario
- ✅ Same loss calculation: Both use `original_gains` from the same scenario

The current approach is reasonable for single-scenario optimization but doesn't provide true generalization assessment. 