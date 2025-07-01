# Input Normalization in Power FA: Single Sample vs NPY Mode

## Overview

The Power FA DNN training uses **different normalization strategies** depending on the training mode:

1. **NPY Mode (Batch Mode)**: Uses global dataset statistics for normalization
2. **Single Sample Mode**: Uses per-sample statistics for normalization

This document elaborates on both approaches with code examples and mathematical details.

---

## 1. NPY Mode (Batch Mode) Normalization

### **Strategy: Global Dataset Normalization**

When training with `.npy` files (multiple samples), the system computes **global normalization statistics** across the entire training dataset.

### **Code Implementation:**
```python
# NPY Mode normalization (lines 368-374 in train_model)
if train_npy is not None:
    dataset = ChannelGainDataset(train_npy)
    # Compute global normalization statistics from ALL training data
    all_x = dataset.data.reshape(len(dataset), -1)  # Flatten: [N_samples, n_pairs*n_pairs]
    input_mean = all_x.mean(axis=0)                 # Mean across samples: [n_pairs*n_pairs]
    input_std = all_x.std(axis=0)                   # Std across samples: [n_pairs*n_pairs] 
    input_std[input_std == 0] = 1.0                 # Prevent division by zero
```

### **Mathematical Formulation:**
For a dataset with N samples, each with channel gains `G_i ∈ ℝ^(n_pairs × n_pairs)`:

```
Flattened data: X = [flatten(G_1), flatten(G_2), ..., flatten(G_N)]
                  ∈ ℝ^(N × n_pairs²)

Global mean: μ = (1/N) Σᵢ₌₁ᴺ flatten(G_i)  ∈ ℝ^(n_pairs²)
Global std:  σ = sqrt((1/N) Σᵢ₌₁ᴺ (flatten(G_i) - μ)²)  ∈ ℝ^(n_pairs²)

Normalized input: X_norm = (X - μ) / σ
```

### **Training Loop Application:**
```python
# During training (lines 388-392)
for batch_idx, batch in enumerate(train_loader):
    x = batch.view(batch.size(0), -1).to(device)  # [batch_size, n_pairs*n_pairs]
    # Apply global normalization to each batch
    x_normalized = (x - torch.tensor(input_mean, device=device)) / \
                   torch.tensor(input_std, device=device)
    
    # Forward pass with normalized input
    output = model(x_normalized)
```

### **Example Output:**
```
[INFO] Global normalization stats - Mean: -98.00 dB, Std: 10.16 dB
```

---

## 2. Single Sample Mode Normalization

### **Strategy: Per-Sample Normalization**

When training on a single randomly generated scenario, normalization statistics are computed **from that specific sample**.

### **Code Implementation:**
```python
# Single sample mode normalization (lines 413-420)
else:  # Single sample training mode
    scenario = Scenario.random(cfg)
    x_raw = torch.tensor(scenario.channel_gains_db(), device=device).flatten()
    sample_flat = x_raw.unsqueeze(0)           # Shape: (1, n_pairs*n_pairs)
    
    # Per-sample normalization statistics
    input_mean = sample_flat.mean(dim=1, keepdim=True)  # Shape: (1, 1) - scalar
    input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8  # Shape: (1, 1) - scalar
    
    # Apply normalization
    x_normalized = (sample_flat - input_mean) / input_std
```

### **Mathematical Formulation:**
For a single sample with channel gains `G ∈ ℝ^(n_pairs × n_pairs)`:

```
Flattened sample: x = flatten(G) ∈ ℝ^(n_pairs²)

Sample mean: μ_sample = (1/n_pairs²) Σⱼ₌₁^(n_pairs²) x_j  (scalar)
Sample std:  σ_sample = sqrt((1/n_pairs²) Σⱼ₌₁^(n_pairs²) (x_j - μ_sample)²)  (scalar)

Normalized input: x_norm = (x - μ_sample) / σ_sample
```

### **Example Output:**
```
[INFO] Single-sample normalization - Mean: -86.00 dB, Std: 19.18 dB
```

---

## 3. Key Differences

| Aspect | NPY Mode | Single Sample Mode |
|--------|----------|-------------------|
| **Statistics Source** | Entire training dataset | Individual sample |
| **Mean/Std Shape** | `[n_pairs²]` (per-feature) | `[1]` (scalar) |
| **Stability** | More stable (large N) | Sample-dependent |
| **Generalization** | Better across scenarios | Specialized to scenario |
| **Computation** | One-time pre-calculation | Per-sample calculation |

---

## 4. Validation/Inference Normalization

### **Strategy: Use Training Statistics**

During validation/inference, the model applies the **same normalization statistics** used during training.

### **Code Implementation:**
```python
# Validation normalization (lines 470-476 in cli.py)
if input_mean is not None and input_std is not None:
    # Use saved training normalization statistics
    input_mean_tensor = torch.tensor(input_mean, device=device)
    input_std_tensor = torch.tensor(input_std, device=device)
    x = (x_raw - input_mean_tensor) / input_std_tensor
else:
    # No normalization (fallback)
    x = x_raw
```

### **Statistics Storage:**
Training statistics are saved in `.meta.yaml` files:
```yaml
# Example: trained_weights.pt.meta.yaml
input_mean: [-98.123, -97.456, ...]  # NPY mode: per-feature
input_std: [10.234, 9.876, ...]      # NPY mode: per-feature

# OR for single sample mode:
input_mean: [-86.0]  # Single sample: scalar
input_std: [19.18]   # Single sample: scalar
```

---

## 5. Dual Normalization Strategy: Input vs Loss

### **Critical Design Principle:**

The system uses a **dual normalization approach**:

1. **Input Normalization**: Applied to DNN inputs for stable training
2. **Original Data for Loss**: Loss functions use **unnormalized** channel gains for physical accuracy

### **Code Example:**
```python
# Training loop (lines 388-402)
# 1. Normalize input for neural network
x_normalized = (x - input_mean) / input_std
output = model(x_normalized)

# 2. Use ORIGINAL unnormalized data for loss calculation  
original_batch = original_data[batch_start:batch_end].to(device)
loss = negative_sum_rate_loss_torch_soft_from_matrix(
    tx_power_dbm, fa_probs, None, cfg, original_batch)  # ← Original data!
```

### **Why This Matters:**

- **Input normalization**: Ensures neural network receives well-conditioned inputs (mean ≈ 0, std ≈ 1)
- **Original data for loss**: Ensures SINR calculations use correct physical channel gains in dB scale
- **Prevents double normalization**: Avoids corrupting the physical meaning of the optimization objective

---

## 6. Practical Examples

### **NPY Mode Example:**
```bash
# Training with 1000 samples
python -m src.cli train_dnn --train_npy samples_urban_6pairs_1fa.npy --n_train_samples 1000

# Output:
# [INFO] Global normalization stats - Mean: -98.00 dB, Std: 10.16 dB
# Training on 1000 samples, batch size 64, 16 batches per epoch
```

### **Single Sample Mode Example:**
```bash  
# Training on single random scenario
python -m src.cli train_dnn --config cfgs/debug.yaml --epochs 100

# Output:
# [INFO] Single-sample normalization - Mean: -86.00 dB, Std: 19.18 dB
# Epoch 1/100: Train sum-rate = 2.34e+08 bit/s
```

---

## 7. Recommendations

### **When to Use NPY Mode:**
- **Production training**: Better generalization across diverse scenarios
- **Batch training**: Efficient GPU utilization with multiple samples
- **Stable statistics**: More robust normalization statistics

### **When to Use Single Sample Mode:**
- **Quick prototyping**: Fast testing of model architecture
- **Scenario-specific optimization**: When targeting specific channel conditions
- **Debugging**: Easier to analyze model behavior on known scenarios

### **Best Practices:**
1. **Always save normalization statistics** in `.meta.yaml` files
2. **Use consistent normalization** between training and inference
3. **Monitor normalization statistics** to detect data distribution shifts
4. **Prefer NPY mode** for production models requiring robustness 