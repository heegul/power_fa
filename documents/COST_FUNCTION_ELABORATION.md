# Cost Function Elaboration for FA > 1 Case

## Overview

This document provides a detailed elaboration of the cost function used in the D2D power and frequency allocation problem when the number of frequency allocations (FA) is greater than 1. It also separates the custom loss function used during training from the validation function used for performance evaluation.

## Mathematical Foundation

### Problem Formulation

For a D2D communication system with `n_pairs` device pairs and `n_fa` frequency allocations, we need to optimize:

1. **Power Allocation**: `P = [p₁, p₂, ..., pₙ]` where `pᵢ ∈ [Pₘᵢₙ, Pₘₐₓ]` (in dBm)
2. **Frequency Allocation**: `F = [f₁, f₂, ..., fₙ]` where `fᵢ ∈ {0, 1, ..., n_fa-1}`

### Objective Function

The goal is to maximize the system sum rate:

```
maximize: R_total = B × Σᵢ log₂(1 + SINRᵢ)
```

Where:
- `B` is the bandwidth (Hz)
- `SINRᵢ` is the Signal-to-Interference-plus-Noise Ratio for pair `i`

## SINR Calculation for FA > 1

### Key Principle: Orthogonal Frequency Model

Devices on different frequency allocations are **orthogonal** and do not interfere with each other. Only devices on the **same** frequency allocation cause interference.

### SINR Formula

For receiver `i` on frequency allocation `fᵢ`:

```
SINRᵢ = (Pᵢ × Gᵢᵢ) / (Iᵢ + N)
```

Where:
- `Pᵢ`: Transmit power of pair `i` (linear scale, Watts)
- `Gᵢᵢ`: Channel gain from transmitter `i` to receiver `i` (with FA penalty applied)
- `Iᵢ`: Interference from other transmitters on the same FA
- `N`: Noise power (linear scale, Watts)

### Interference Calculation

```
Iᵢ = Σⱼ≠ᵢ,fⱼ=fᵢ (Pⱼ × Gⱼᵢ)
```

Only transmitters `j` where `fⱼ = fᵢ` (same FA) contribute to interference at receiver `i`.

### FA Penalty Application

Channel gains are modified by FA penalty:

```
G'ᵢⱼ = Gᵢⱼ - (fa_penalty_db × fⱼ)
```

Where `fa_penalty_db` is the penalty in dB applied to higher frequency allocations.

## Cost Function Implementation

### 1. Training Loss Function (Custom Loss)

The training loss function is designed to be differentiable and handle both hard and soft FA assignments.

#### Hard FA Assignment Loss

```python
def negative_sum_rate_loss_torch_from_matrix(tx_power_dbm, fa_probs, scenario, cfg, channel_gains_db):
    """
    Hard FA assignment using argmax of probabilities.
    Used when discrete FA decisions are needed.
    """
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    
    # Convert power from dBm to linear Watts
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # [B, n_pairs]
    
    # Get FA indices from probabilities (hard assignment)
    if n_fa == 1:
        fa_indices = torch.zeros((B, n_pairs), dtype=torch.long, device=device)
    else:
        fa_indices = torch.argmax(fa_probs, dim=2)  # [B, n_pairs]
    
    # Apply FA penalty to channel gains
    channel_gain_db = channel_gains_db.clone()
    fa_indices_np = fa_indices.detach().cpu().numpy()
    channel_gain_db -= cfg.fa_penalty_db * fa_indices_np[:, None, :]
    
    # Convert to linear scale
    g = torch.tensor(10 ** (channel_gain_db / 10), dtype=torch.float32, device=device)
    
    # Vectorized SINR computation
    fa_i = fa_indices.unsqueeze(2)  # [B, n_pairs, 1]
    fa_j = fa_indices.unsqueeze(1)  # [B, 1, n_pairs]
    same_fa = (fa_i == fa_j)        # [B, n_pairs, n_pairs]
    
    # Calculate interference matrix
    tx_power_lin_exp = tx_power_lin.unsqueeze(1)  # [B, 1, n_pairs]
    interf_matrix = same_fa * tx_power_lin_exp * g  # [B, n_pairs, n_pairs]
    
    # Total interference and desired signal
    total_interf = interf_matrix.sum(dim=2)  # [B, n_pairs]
    self_interf = tx_power_lin * g.diagonal(dim1=1, dim2=2)  # [B, n_pairs]
    interference = total_interf - self_interf  # [B, n_pairs]
    desired = self_interf  # [B, n_pairs]
    
    # SINR calculation
    noise_power_lin = torch.tensor(10 ** (cfg.noise_power_dbm / 10) * 1e-3, device=device)
    sinr = desired / (interference + noise_power_lin)  # [B, n_pairs]
    
    # Sum rate calculation
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + sinr), dim=1)  # [B]
    
    return -sum_rate.mean()  # Negative for minimization
```

#### Soft FA Assignment Loss

```python
def negative_sum_rate_loss_torch_soft_from_matrix(tx_power_dbm, fa_probs, scenario, cfg, channel_gains_db):
    """
    Soft FA assignment using expected interference based on FA probabilities.
    More differentiable but computationally intensive.
    """
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    
    # Convert power from dBm to linear Watts
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # [B, n_pairs]
    
    # Channel gains (no FA penalty in soft version - handled in expectation)
    g = 10 ** (channel_gains_db / 10)  # [B, n_pairs, n_pairs]
    
    noise_power_lin = 10 ** (cfg.noise_power_dbm / 10) * 1e-3
    
    sinr = []  # Will be [B, n_pairs]
    
    for j in range(n_pairs):
        # Expected desired signal: sum over FA, weighted by probabilities
        desired = torch.sum(fa_probs[:, j, :] * 
                          (tx_power_lin[:, j].unsqueeze(1) * g[:, j, j].unsqueeze(1)), dim=1)
        
        # Expected interference: sum over FA, weighted by probabilities
        interference = torch.zeros(B, device=device)
        for k in range(n_fa):
            # Interference from all i != j to j on FA k
            interf_k = torch.zeros(B, device=device)
            for i in range(n_pairs):
                if i != j:
                    # Apply FA penalty for FA k
                    g_penalty = g[:, i, j] * (10 ** (-cfg.fa_penalty_db * k / 10))
                    interf_k += fa_probs[:, i, k] * tx_power_lin[:, i] * g_penalty
            interference += fa_probs[:, j, k] * interf_k
        
        sinr_j = desired / (interference + noise_power_lin)
        sinr.append(sinr_j)
    
    sinr = torch.stack(sinr, dim=1)  # [B, n_pairs]
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + sinr), dim=1)  # [B]
    
    return -sum_rate.mean()  # Negative for minimization
```

### 2. Validation Function (Performance Evaluation)

The validation function provides a clean, interpretable evaluation of the model's performance using the actual allocation decisions.

```python
def evaluate_sum_rate(tx_power_dbm, fa_indices, channel_gains_db, cfg):
    """
    Clean validation function for performance evaluation.
    Uses actual discrete FA assignments and provides interpretable metrics.
    """
    n_pairs = cfg.n_pairs
    
    # Convert power to linear scale
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # dBm to Watts
    
    # Apply FA penalty to channel gains
    channel_gain_db = channel_gains_db.copy()
    for rx in range(n_pairs):
        penalty_db = cfg.fa_penalty_db * fa_indices[rx]
        channel_gain_db[:, rx] -= penalty_db
    
    # Convert to linear scale
    channel_gain_lin = 10 ** (channel_gain_db / 10)
    
    # Calculate SINR using the metrics module
    from src.simulator.metrics import sinr_linear, sum_rate_bps
    
    sinr = sinr_linear(
        tx_power_lin=tx_power_lin,
        fa_indices=fa_indices,
        channel_gain=channel_gain_lin,
        noise_power_lin=10 ** (cfg.noise_power_dbm / 10) * 1e-3,
    )
    
    # Calculate sum rate
    sum_rate = sum_rate_bps(sinr, cfg.bandwidth_hz)
    
    return {
        'sum_rate': sum_rate,
        'sinr': sinr,
        'tx_power_dbm': tx_power_dbm,
        'fa_indices': fa_indices,
        'individual_rates': cfg.bandwidth_hz * np.log2(1.0 + sinr)
    }

def validate_model_performance(model, scenario, cfg, device='cpu'):
    """
    Comprehensive validation function that evaluates model performance
    and provides detailed metrics.
    """
    model.eval()
    
    # Get model predictions
    x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
    x = x_raw.unsqueeze(0)
    
    with torch.no_grad():
        output = model(x)[0]
        tx_power_dbm, fa_indices = dnn_output_to_decision(output, cfg)
    
    # Evaluate performance
    results = evaluate_sum_rate(tx_power_dbm, fa_indices, scenario.channel_gains_db(), cfg)
    
    # Additional validation metrics
    results.update({
        'power_efficiency': np.mean(10 ** (tx_power_dbm / 10)),  # Average linear power
        'fa_distribution': np.bincount(fa_indices, minlength=cfg.n_fa),
        'min_sinr_db': 10 * np.log10(np.min(results['sinr'])),
        'max_sinr_db': 10 * np.log10(np.max(results['sinr'])),
        'fairness_index': len(results['sinr']) ** 2 * np.sum(results['sinr']) / 
                         (len(results['sinr']) * np.sum(results['sinr'] ** 2))
    })
    
    return results
```

## Key Differences Between Training and Validation

### Training Loss Function
- **Purpose**: Optimize model parameters through backpropagation
- **Characteristics**:
  - Fully differentiable
  - Handles soft FA assignments via probabilities
  - Vectorized for batch processing
  - Uses negative sum rate for minimization
  - May use approximations for computational efficiency

### Validation Function
- **Purpose**: Evaluate model performance with interpretable metrics
- **Characteristics**:
  - Uses discrete FA assignments (argmax)
  - Provides detailed performance breakdown
  - Includes additional metrics (fairness, efficiency, etc.)
  - Uses exact SINR calculations
  - Returns positive sum rate values
  - Provides per-link analysis

## Usage Examples

### Training with Custom Loss
```python
# During training
for epoch in range(epochs):
    model.train()
    output = model(x_normalized)
    tx_power_dbm, fa_indices, fa_probs = dnn_output_to_decision_torch(output, cfg, device)
    
    # Use custom loss function
    if soft_fa:
        loss = negative_sum_rate_loss_torch_soft_from_matrix(
            tx_power_dbm, fa_probs, None, cfg, original_gains)
    else:
        loss = negative_sum_rate_loss_torch_from_matrix(
            tx_power_dbm, fa_probs, None, cfg, original_gains)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Validation with Performance Evaluation
```python
# During validation
model.eval()
with torch.no_grad():
    # Get comprehensive performance metrics
    validation_results = validate_model_performance(model, scenario, cfg, device)
    
    print(f"Sum Rate: {validation_results['sum_rate']:.2e} bit/s")
    print(f"Min SINR: {validation_results['min_sinr_db']:.2f} dB")
    print(f"Max SINR: {validation_results['max_sinr_db']:.2f} dB")
    print(f"Fairness Index: {validation_results['fairness_index']:.3f}")
    print(f"FA Distribution: {validation_results['fa_distribution']}")
```

## Configuration Parameters

The cost function behavior is controlled by several configuration parameters:

- `n_pairs`: Number of D2D pairs
- `n_fa`: Number of frequency allocations
- `fa_penalty_db`: Penalty applied to higher frequency allocations (dB)
- `tx_power_min_dbm`: Minimum transmit power (dBm)
- `tx_power_max_dbm`: Maximum transmit power (dBm)
- `noise_power_dbm`: Receiver noise power (dBm)
- `bandwidth_hz`: System bandwidth (Hz)

## Computational Complexity

### Hard FA Assignment
- **Time Complexity**: O(B × n_pairs²) for batch size B
- **Space Complexity**: O(B × n_pairs²)
- **Advantages**: Faster, more memory efficient
- **Disadvantages**: Less smooth gradients

### Soft FA Assignment
- **Time Complexity**: O(B × n_pairs² × n_fa)
- **Space Complexity**: O(B × n_pairs² × n_fa)
- **Advantages**: Smoother gradients, better for training
- **Disadvantages**: Computationally intensive for large n_fa

## Best Practices

1. **Use soft FA assignment during early training** for better gradient flow
2. **Switch to hard FA assignment for final training** for discrete decisions
3. **Always use validation function for performance reporting** to ensure interpretability
4. **Monitor both training loss and validation metrics** to detect overfitting
5. **Consider FA penalty values carefully** as they significantly impact the solution space 