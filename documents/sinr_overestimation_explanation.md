# 🔍 Why `sinr_linear()` Systematically Overestimates SINR

## 📊 **Summary**

The `sinr_linear()` function in `src.simulator.metrics` systematically **overestimates SINR by ~5.1 dB on average**, leading to **29.6% overestimation of sum-rate performance**. This causes the "All Power & FA Comb. Avg" calculation in `sample_visualization.py` to show **374.82 Mbps** instead of the correct **344.07 Mbps** (8.9% difference).

## 🎯 **Root Cause: Incorrect Interference Calculation**

### 🔴 **INCORRECT Method (`sinr_linear()` function):**

```python
def sinr_linear(tx_power_lin, fa_indices, channel_gain, noise_power_lin):
    for i in range(n_pairs):
        same_fa = fa_indices == fa_indices[i]  # ❌ WRONG: Only same FA interfere
        interference = np.sum(tx_power_lin[same_fa] * channel_gain[same_fa, i]) - (
            tx_power_lin[i] * channel_gain[i, i]
        )
        sinr[i] = (tx_power_lin[i] * channel_gain[i, i]) / (interference + noise_power_lin)
```

**Problem**: Only considers interference from devices on the **same frequency assignment (FA)**.

### 🟢 **CORRECT Method (MATLAB-style):**

```python
def sinr_correct(tx_power_lin, channel_gain, noise_power_lin):
    for i in range(n_pairs):
        # ✅ CORRECT: ALL transmitters interfere
        interference = np.sum(tx_power_lin * channel_gain[:, i]) - (
            tx_power_lin[i] * channel_gain[i, i]
        )
        sinr[i] = (tx_power_lin[i] * channel_gain[i, i]) / (interference + noise_power_lin)
```

**Correct**: Considers interference from **ALL transmitters**, regardless of FA assignment.

## 🧠 **Conceptual Misunderstanding**

### **The Key Issue:**
`sinr_linear()` assumes that **FA penalty = orthogonal frequencies**, but this is **WRONG** in this system:

1. **FA penalty is applied to channel gains** (reducing signal strength)
2. **All devices still interfere with each other** (they're not on orthogonal frequencies)
3. **FA assignment affects signal quality, not interference filtering**

### **Physical Reality:**
- All transmitters operate in the same spectrum
- FA assignment applies a penalty to reduce effective channel gain
- **Every transmitter interferes with every receiver**

## 📈 **Quantitative Impact**

From our detailed analysis:

| Metric | `sinr_linear()` (Incorrect) | MATLAB (Correct) | Overestimation |
|--------|----------------------------|------------------|----------------|
| **Average SINR** | +5.1 dB higher | Baseline | 5.1 dB |
| **Sum-rate** | 43.83 | 33.82 | +29.6% |
| **All Power & FA Comb. Avg** | 374.82 Mbps | 344.07 Mbps | +8.9% |

### **Per-Device SINR Overestimation:**
- **RX 0**: 15.96× higher SINR (+12.0 dB)
- **RX 1**: 1.03× higher SINR (+0.1 dB)  
- **RX 2**: 18.04× higher SINR (+12.6 dB)
- **RX 3**: 1.12× higher SINR (+0.5 dB)
- **RX 4**: 1.87× higher SINR (+2.7 dB)
- **RX 5**: 1.74× higher SINR (+2.4 dB)

## 🔧 **The Fix**

Replace all `sinr_linear()` calls with direct SINR calculation:

```python
# CORRECT SINR calculation
def compute_sinr_correct(tx_power_lin, channel_gain, noise_power_lin):
    n_pairs = len(tx_power_lin)
    sinr = np.zeros(n_pairs)
    
    for i in range(n_pairs):
        signal = tx_power_lin[i] * channel_gain[i, i]
        interference = np.sum(tx_power_lin * channel_gain[:, i]) - signal
        sinr[i] = signal / (interference + noise_power_lin)
    
    return sinr
```

## 🎯 **Impact on Results**

This explains why:
1. **`sample_visualization.py`** shows higher "All Power & FA Comb. Avg" (374.82 Mbps)
2. **MATLAB results** show lower, correct values (344.07 Mbps)
3. **`matlab_equivalent_fixed.py`** matches MATLAB exactly (344.06 Mbps)

## ✅ **Validation Status**

- ✅ **MATLAB**: 344.07 Mbps (ground truth)
- ✅ **`matlab_equivalent_fixed.py`**: 344.06 Mbps (validated Python equivalent)
- ❌ **`sample_visualization.py`**: 374.82 Mbps (incorrect due to `sinr_linear()`)

## 🛠️ **Recommendations**

1. **Use `matlab_equivalent_fixed.py`** for accurate MATLAB-equivalent results
2. **Fix `sample_visualization.py`** to use correct SINR calculation
3. **Update any analysis** that relied on `sample_visualization.py` values
4. **Use MATLAB results (344.07 Mbps)** as the correct baseline

The systematic overestimation by `sinr_linear()` affects all performance baselines and comparisons, making it crucial to use the correct SINR calculation method. 