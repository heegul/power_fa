"""
Validation functions for evaluating DNN performance in D2D power and frequency allocation.

This module provides clean, interpretable validation functions that are separate from
the training loss functions. These functions are designed for performance evaluation
and provide detailed metrics for analysis.
"""

import numpy as np
import torch
from typing import Dict, Any, Union
from pathlib import Path

from ..config import SimulationConfig
from ..simulator.metrics import sinr_linear, sum_rate_bps, sum_rate_dimensionless
from .ml_dnn import dnn_output_to_decision


def evaluate_sum_rate(
    tx_power_dbm: np.ndarray,
    fa_indices: np.ndarray,
    channel_gains_db: np.ndarray,
    cfg: SimulationConfig
) -> Dict[str, Any]:
    """
    Clean validation function for performance evaluation.
    Uses actual discrete FA assignments and provides interpretable metrics.
    
    Parameters
    ----------
    tx_power_dbm : np.ndarray
        Transmit power in dBm for each pair, shape (n_pairs,)
    fa_indices : np.ndarray
        Frequency allocation indices for each pair, shape (n_pairs,)
    channel_gains_db : np.ndarray
        Channel gain matrix in dB, shape (n_pairs, n_pairs)
    cfg : SimulationConfig
        Configuration object containing system parameters
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - sum_rate: Total system sum rate (bit/s)
        - sum_rate_dimensionless: Total system sum rate (dimensionless)
        - sinr: SINR values for each pair (linear scale)
        - tx_power_dbm: Input power values
        - fa_indices: Input FA assignments
        - individual_rates: Per-link rates (bit/s)
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
    sinr = sinr_linear(
        tx_power_lin=tx_power_lin,
        fa_indices=fa_indices,
        channel_gain=channel_gain_lin,
        noise_power_lin=10 ** (cfg.noise_power_dbm / 10) * 1e-3,
    )
    
    # Calculate sum rate
    sum_rate = sum_rate_bps(sinr, cfg.bandwidth_hz)
    sum_rate_dim = sum_rate_dimensionless(sinr)
    
    return {
        'sum_rate': sum_rate,
        'sum_rate_dimensionless': sum_rate_dim,
        'sinr': sinr,
        'tx_power_dbm': tx_power_dbm,
        'fa_indices': fa_indices,
        'individual_rates': cfg.bandwidth_hz * np.log2(1.0 + sinr)
    }


def validate_model_performance(
    model: torch.nn.Module,
    scenario,
    cfg: SimulationConfig,
    device: str = 'cpu',
    input_mean: float = None,
    input_std: float = None
) -> Dict[str, Any]:
    """
    Comprehensive validation function that evaluates model performance
    and provides detailed metrics.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained DNN model
    scenario : Scenario
        Scenario object containing channel gains
    cfg : SimulationConfig
        Configuration object
    device : str
        Device to run inference on
    input_mean : float, optional
        Mean for input normalization (if None, no normalization)
    input_std : float, optional
        Standard deviation for input normalization (if None, no normalization)
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive performance metrics including:
        - All metrics from evaluate_sum_rate
        - power_efficiency: Average linear power consumption
        - fa_distribution: Distribution of FA assignments
        - min_sinr_db, max_sinr_db: SINR range in dB
        - fairness_index: Jain's fairness index
    """
    model.eval()
    
    # Get model predictions
    x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
    x = x_raw.unsqueeze(0)
    
    # Apply normalization if provided
    if input_mean is not None and input_std is not None:
        x = (x - input_mean) / input_std
    
    with torch.no_grad():
        output = model(x)[0]
        tx_power_dbm, fa_indices = dnn_output_to_decision(output, cfg)
    
    # Evaluate performance using the clean validation function
    results = evaluate_sum_rate(tx_power_dbm, fa_indices, scenario.channel_gains_db(), cfg)
    
    # Additional validation metrics
    results.update({
        'power_efficiency': np.mean(10 ** (tx_power_dbm / 10)),  # Average linear power (mW)
        'fa_distribution': np.bincount(fa_indices, minlength=cfg.n_fa),
        'min_sinr_db': 10 * np.log10(np.min(results['sinr'])) if np.min(results['sinr']) > 0 else -np.inf,
        'max_sinr_db': 10 * np.log10(np.max(results['sinr'])),
        'fairness_index': calculate_fairness_index(results['sinr']),
        'power_range_dbm': np.max(tx_power_dbm) - np.min(tx_power_dbm),
        'avg_power_dbm': np.mean(tx_power_dbm),
        'std_power_dbm': np.std(tx_power_dbm)
    })
    
    return results


def calculate_fairness_index(values: np.ndarray) -> float:
    """
    Calculate Jain's fairness index for a set of values.
    
    Parameters
    ----------
    values : np.ndarray
        Array of values (e.g., SINR, rates)
        
    Returns
    -------
    float
        Fairness index between 0 and 1 (1 = perfectly fair)
    """
    if len(values) == 0:
        return 0.0
    
    sum_values = np.sum(values)
    sum_squares = np.sum(values ** 2)
    
    if sum_squares == 0:
        return 1.0  # All values are zero, considered fair
    
    return (sum_values ** 2) / (len(values) * sum_squares)


def compare_with_baseline(
    model_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    baseline_name: str = "Baseline"
) -> Dict[str, Any]:
    """
    Compare model performance with a baseline algorithm.
    
    Parameters
    ----------
    model_results : Dict[str, Any]
        Results from validate_model_performance
    baseline_results : Dict[str, Any]
        Results from baseline algorithm (same format)
    baseline_name : str
        Name of the baseline algorithm
        
    Returns
    -------
    Dict[str, Any]
        Comparison metrics including ratios and improvements
    """
    comparison = {
        'baseline_name': baseline_name,
        'sum_rate_ratio': model_results['sum_rate'] / baseline_results['sum_rate'],
        'sum_rate_improvement_percent': 
            (model_results['sum_rate'] - baseline_results['sum_rate']) / baseline_results['sum_rate'] * 100,
        'fairness_improvement': 
            model_results['fairness_index'] - baseline_results['fairness_index'],
        'power_efficiency_ratio': 
            model_results['power_efficiency'] / baseline_results['power_efficiency'],
        'min_sinr_improvement_db': 
            model_results['min_sinr_db'] - baseline_results['min_sinr_db']
    }
    
    # Add detailed comparison
    comparison['model_better_sum_rate'] = model_results['sum_rate'] > baseline_results['sum_rate']
    comparison['model_better_fairness'] = model_results['fairness_index'] > baseline_results['fairness_index']
    comparison['model_more_efficient'] = model_results['power_efficiency'] < baseline_results['power_efficiency']
    
    return comparison


def validate_multiple_scenarios(
    model: torch.nn.Module,
    scenarios: list,
    cfg: SimulationConfig,
    device: str = 'cpu',
    input_mean: float = None,
    input_std: float = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate model performance across multiple scenarios.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained DNN model
    scenarios : list
        List of scenario objects
    cfg : SimulationConfig
        Configuration object
    device : str
        Device to run inference on
    input_mean : float, optional
        Mean for input normalization
    input_std : float, optional
        Standard deviation for input normalization
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    Dict[str, Any]
        Aggregated performance metrics across all scenarios
    """
    all_results = []
    
    for i, scenario in enumerate(scenarios):
        if verbose and (i + 1) % 10 == 0:
            print(f"Validating scenario {i + 1}/{len(scenarios)}")
        
        results = validate_model_performance(
            model, scenario, cfg, device, input_mean, input_std
        )
        all_results.append(results)
    
    # Aggregate results
    sum_rates = [r['sum_rate'] for r in all_results]
    fairness_indices = [r['fairness_index'] for r in all_results]
    power_efficiencies = [r['power_efficiency'] for r in all_results]
    min_sinrs_db = [r['min_sinr_db'] for r in all_results if r['min_sinr_db'] != -np.inf]
    max_sinrs_db = [r['max_sinr_db'] for r in all_results]
    
    aggregated = {
        'n_scenarios': len(scenarios),
        'sum_rate_mean': np.mean(sum_rates),
        'sum_rate_std': np.std(sum_rates),
        'sum_rate_min': np.min(sum_rates),
        'sum_rate_max': np.max(sum_rates),
        'fairness_mean': np.mean(fairness_indices),
        'fairness_std': np.std(fairness_indices),
        'power_efficiency_mean': np.mean(power_efficiencies),
        'power_efficiency_std': np.std(power_efficiencies),
        'min_sinr_db_mean': np.mean(min_sinrs_db) if min_sinrs_db else -np.inf,
        'max_sinr_db_mean': np.mean(max_sinrs_db),
        'all_results': all_results
    }
    
    if verbose:
        print(f"\nValidation Summary ({len(scenarios)} scenarios):")
        print(f"  Sum Rate: {aggregated['sum_rate_mean']:.2e} ± {aggregated['sum_rate_std']:.2e} bit/s")
        print(f"  Fairness: {aggregated['fairness_mean']:.3f} ± {aggregated['fairness_std']:.3f}")
        print(f"  Power Efficiency: {aggregated['power_efficiency_mean']:.2f} ± {aggregated['power_efficiency_std']:.2f} mW")
    
    return aggregated


def save_validation_results(
    results: Dict[str, Any],
    save_path: Union[str, Path],
    include_detailed: bool = True
) -> None:
    """
    Save validation results to a file.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Validation results dictionary
    save_path : Union[str, Path]
        Path to save the results
    include_detailed : bool
        Whether to include detailed per-scenario results
    """
    import yaml
    
    # Prepare results for saving
    save_data = results.copy()
    
    # Convert numpy arrays to lists for YAML serialization
    for key, value in save_data.items():
        if isinstance(value, np.ndarray):
            save_data[key] = value.tolist()
        elif isinstance(value, np.integer):
            save_data[key] = int(value)
        elif isinstance(value, np.floating):
            save_data[key] = float(value)
    
    # Optionally exclude detailed results to reduce file size
    if not include_detailed and 'all_results' in save_data:
        del save_data['all_results']
    
    # Save to file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(save_data, f, default_flow_style=False, indent=2)
    
    print(f"Validation results saved to {save_path}") 