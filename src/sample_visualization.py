"""Visualize generated samples and channel gains."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import yaml
import pickle
from itertools import product
from joblib import Parallel, delayed

from .config import SimulationConfig
from .simulator.scenario import Scenario
from .simulator.engine import run_once
from .algorithms import get_algorithm
from .simulator.environment import db_to_linear
from .simulator.metrics import sinr_linear, sum_rate_bps


def visualize_sample(sample_idx: int = 0, npy_path: str = "samples_test_2x4.npy", show: bool = True, save_path: str = None, area_size_m: float = None):
    """Visualize a specific sample from the saved numpy file.
    
    Args:
        sample_idx: Index of the sample to visualize
        npy_path: Path to the numpy file with samples
        show: Whether to display the plot
        save_path: Path to save the figure (optional)
        area_size_m: Area size in meters (optional, overrides default)
    """
    # Load samples
    samples = np.load(npy_path)
    print(f"Loaded samples with shape: {samples.shape}")
    
    if sample_idx >= samples.shape[0]:
        raise ValueError(f"Sample index {sample_idx} out of range (0-{samples.shape[0]-1})")
    
    # Get parameters from filename (assuming format like samples_urban_4pairs_2fa_100m.npy)
    parts = npy_path.split('_')
    channel_type = "urban"  # Default
    inferred_area_size = 100.0
    for part in parts:
        if part in ["urban", "suburban", "rural"]:
            channel_type = part
        if part.endswith("m.npy"):
            try:
                inferred_area_size = float(part.replace("m.npy", ""))
            except Exception:
                pass
        elif part.endswith("m"):
            try:
                inferred_area_size = float(part.replace("m", ""))
            except Exception:
                pass
    # Use argument if provided, else inferred, else default
    area_size = area_size_m if area_size_m is not None else inferred_area_size
    
    n_pairs = samples.shape[1]
    
    # Manually create a scenario for visualization
    cfg = SimulationConfig(
        area_size_m=area_size,
        n_pairs=n_pairs,
        n_fa=2,  # Doesn't matter for visualization
        pathloss_exp=3.7 if channel_type == "urban" else 3.0,  # Approximate
        seed=sample_idx,  # Use sample_idx as seed
    )
    
    scenario = Scenario.random(cfg)
    channel_gains_db = samples[sample_idx]  # Get the specific sample
    
    # Create a figure with 2 subplots
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 0.1])
    
    # 1. Plot device locations
    ax1 = plt.subplot(gs[0])
    tx_xy, rx_xy = scenario.as_tuple()
    
    # Plot transmitters
    ax1.scatter(tx_xy[:, 0], tx_xy[:, 1], c='red', s=100, marker='^', label='Transmitters')
    
    # Plot receivers
    ax1.scatter(rx_xy[:, 0], rx_xy[:, 1], c='blue', s=100, marker='o', label='Receivers')
    
    # Connect pairs with lines
    for i in range(n_pairs):
        ax1.plot([tx_xy[i, 0], rx_xy[i, 0]], 
                 [tx_xy[i, 1], rx_xy[i, 1]], 'k-', alpha=0.6, linewidth=1)
        
        # Label pairs
        ax1.annotate(f"{i+1}", (tx_xy[i, 0], tx_xy[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=12)
        ax1.annotate(f"{i+1}", (rx_xy[i, 0], rx_xy[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title(f'Device Locations in {channel_type.capitalize()} Area')
    ax1.set_xlim(0, area_size)
    ax1.set_ylim(0, area_size)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # 2. Plot channel gains as heatmap
    ax2 = plt.subplot(gs[1])
    # Use mean and std for normalization (dB values can be negative)
    mean = np.mean(channel_gains_db)
    std = np.std(channel_gains_db)
    vmin = mean - 2 * std
    vmax = mean + 2 * std
    im = ax2.imshow(channel_gains_db, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Channel Gains (dB)')
    
    # Add pair indices
    for i in range(n_pairs):
        for j in range(n_pairs):
            text = ax2.text(j, i, f"{channel_gains_db[i, j]:.1f}",
                           ha="center", va="center", color="w", fontsize=9)
    
    ax2.set_xticks(np.arange(n_pairs))
    ax2.set_yticks(np.arange(n_pairs))
    ax2.set_xticklabels([f"Rx {i+1}" for i in range(n_pairs)])
    ax2.set_yticklabels([f"Tx {i+1}" for i in range(n_pairs)])
    ax2.set_xlabel('Receiver')
    ax2.set_ylabel('Transmitter')
    
    # Add colorbar
    cax = plt.subplot(gs[2])
    plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Plot distance distribution for this sample
    plot_distance_distribution(tx_xy, rx_xy, show=show, save_path=save_path)
    if show:
        plt.show()
    else:
        plt.close()


def visualize_fs_result(config_path: str, algo_name: str = "fs_bruteforce", save_path: str = None, show: bool = True):
    """Run FS and visualize the power and FA assignment for each Tx.
    Args:
        config_path: Path to YAML config file
        algo_name: Algorithm name (default: fs_bruteforce)
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    cfg = SimulationConfig.from_yaml(config_path)
    result = run_once(cfg, algo_name)
    tx_power_dbm = result.tx_power_dbm
    fa_indices = result.fa_indices
    n_pairs = cfg.n_pairs

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        np.arange(n_pairs),
        tx_power_dbm,
        c=fa_indices,
        cmap="tab10",
        s=120,
        edgecolor="k",
        label="Tx Power (dBm)"
    )
    for i in range(n_pairs):
        ax.annotate(f"FA {fa_indices[i]}", (i, tx_power_dbm[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    ax.set_xlabel("Transmitter Index")
    ax.set_ylabel("Power (dBm)")
    ax.set_title(f"FS Power & FA Assignment (algo={algo_name})")
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(cfg.n_fa))
    cbar.set_label("FA Index")
    cbar.set_ticks(range(cfg.n_fa))
    ax.set_xticks(np.arange(n_pairs))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def visualize_fs_result_locations(config_path: str, algo_name: str = "fs_bruteforce", save_path: str = None, show: bool = True):
    """Run FS and visualize device locations with arrows: width=power, color=FA."""
    cfg = SimulationConfig.from_yaml(config_path)
    result = run_once(cfg, algo_name)
    tx_power_dbm = result.tx_power_dbm
    fa_indices = result.fa_indices
    scenario = result.scenario
    n_pairs = cfg.n_pairs
    area_size = cfg.area_size_m

    tx_xy, rx_xy = scenario.as_tuple()
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # dBm to Watt

    # Color map for FAs
    import matplotlib.colors as mcolors
    fa_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    color_list = [fa_colors[fa % len(fa_colors)] for fa in fa_indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n_pairs):
        # Clamp negative/small power to minimum width
        width = max(2, 2 + 8 * (max(tx_power_lin[i], 0) / max(cfg.tx_power_max_dbm, 1)))
        ax.annotate(
            '',
            xy=rx_xy[i],
            xytext=tx_xy[i],
            arrowprops=dict(
                arrowstyle="->",
                color=color_list[i],
                lw=width,  # width by power, clamped
                alpha=0.8
            ),
        )
        # Mark Tx and Rx
        ax.scatter(*tx_xy[i], c=color_list[i], marker='^', s=120, edgecolor='k', label=f"Tx {i+1}" if i==0 else None)
        ax.scatter(*rx_xy[i], c=color_list[i], marker='o', s=120, edgecolor='k', label=f"Rx {i+1}" if i==0 else None)
        # Annotate pair index
        ax.annotate(f"{i+1}", tx_xy[i], textcoords="offset points", xytext=(5,5), fontsize=10, color='k')
        ax.annotate(f"{i+1}", rx_xy[i], textcoords="offset points", xytext=(5,5), fontsize=10, color='k')
        # Annotate Tx power (dBm)
        ax.annotate(f"{tx_power_dbm[i]:.2f} dBm", tx_xy[i], textcoords="offset points", xytext=(5,20), fontsize=10, color=color_list[i], fontweight='bold')

    # Legend for FAs
    handles = [plt.Line2D([0], [0], color=fa_colors[fa % len(fa_colors)], lw=3, label=f"FA {fa}") for fa in range(cfg.n_fa)]
    ax.legend(handles=handles, title="Frequency Allocation", loc='upper right')
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"FS Device Locations, Power (arrow width), FA (color)")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_allocations_side_by_side(scenario, dnn_decision, fs_decision, cfg):
    # Use scenario.as_tuple() to get device locations
    import matplotlib.colors as mcolors
    tx_xy, rx_xy = scenario.as_tuple()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['DNN Allocation', 'FS Allocation']
    decisions = [dnn_decision, fs_decision]
    n_fa = cfg.n_fa
    # Use a consistent color map for FA assignments
    fa_colors = [f"C{i % 10}" for i in range(n_fa)]
    for ax, title, decision in zip(axes, titles, decisions):
        ax.set_title(title)
        ax.set_xlim(0, cfg.area_size_m)
        ax.set_ylim(0, cfg.area_size_m)
        # Plot device pairs as arrows
        for i in range(len(tx_xy)):
            tx, rx = tx_xy[i], rx_xy[i]
            power = decision['tx_power_dbm'][i]
            fa = decision['fa_indices'][i]
            color = fa_colors[fa % len(fa_colors)]
            # Clamp negative/small power to minimum width
            width = max(2, 2 + 8 * (max(power, 0) / max(cfg.tx_power_max_dbm, 1)))
            ax.annotate(
                '',
                xy=rx,
                xytext=tx,
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=width,  # width by power, clamped
                    alpha=0.8
                ),
            )
            # Annotate power at midpoint
            mid_x = (tx[0] + rx[0]) / 2
            mid_y = (tx[1] + rx[1]) / 2
            ax.text(mid_x, mid_y, f"{power:.1f} dBm", color=color, fontsize=8)
            # Mark Tx and Rx
            ax.scatter(*tx, c=color, marker='^', s=80, edgecolor='k', zorder=3)
            ax.scatter(*rx, c=color, marker='o', s=80, edgecolor='k', zorder=3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle='--', alpha=0.7)
    # Add a legend for FA colors
    handles = [plt.Line2D([0], [0], color=fa_colors[fa], lw=3, label=f"FA {fa}") for fa in range(n_fa)]
    axes[1].legend(handles=handles, title="Frequency Allocation", loc='upper right')
    plt.tight_layout()
    plt.show()


def compute_sum_rate_for_assignment(tx_power_dbm_vec, fa_indices, channel_gains_db, fa_penalty_db, db_to_linear, sinr_linear, sum_rate_bps, bandwidth_hz, noise_power_lin):
    n_pairs = len(tx_power_dbm_vec)
    channel_gain_db = channel_gains_db.copy()
    for rx in range(n_pairs):
        penalty_db = fa_penalty_db * fa_indices[rx]
        channel_gain_db[:, rx] -= penalty_db
    channel_gain = db_to_linear(channel_gain_db)
    dummy_fa_indices = np.zeros(n_pairs, dtype=int)
    tx_power_lin_vec = db_to_linear(np.array(tx_power_dbm_vec)) * 1e-3
    sinr = sinr_linear(
        tx_power_lin=tx_power_lin_vec,
        fa_indices=dummy_fa_indices,
        channel_gain=channel_gain,
        noise_power_lin=noise_power_lin,
    )
    sum_rate = sum_rate_bps(sinr, bandwidth_hz)
    return sum_rate


def plot_sum_rate_vs_sample(samples, save_path=None, figs_dir=".", no_show=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from itertools import product
    from src.simulator.environment import db_to_linear
    from src.simulator.metrics import sinr_linear, sum_rate_bps
    sample_indices = [s['index'] for s in samples]
    dnn_sum_rates = [s['dnn']['sum_rate'] for s in samples]
    fs_sum_rates = [s['fs']['sum_rate'] for s in samples]
    all_max_sum_rates = []
    all_power_fa_avg_sum_rates = []
    for s in samples:
        cfg = s['config']
        n_pairs = int(cfg['n_pairs'])
        n_fa = int(cfg['n_fa'])
        tx_power_min_dbm = float(cfg['tx_power_min_dbm'])
        tx_power_max_dbm = float(cfg['tx_power_max_dbm'])
        # --- All Max Power (Uniform FA Avg) ---
        # Use power levels: {min, max-6, max-3, max} dBm, matching FS and config
        power_levels = np.array([
            tx_power_min_dbm,
            tx_power_max_dbm - 6,
            tx_power_max_dbm - 3,
            tx_power_max_dbm
        ])
        tx_power_dbm = [tx_power_max_dbm] * n_pairs
        channel_gains_db = np.array(s['channel_gains_db'])
        tx_power_lin = db_to_linear(np.array(tx_power_dbm)) * 1e-3
        noise_power_lin = db_to_linear(float(cfg['noise_power_dbm'])) * 1e-3
        fa_penalty_db = float(cfg['fa_penalty_db'])
        # Enumerate all possible FA assignments (cartesian product)
        fa_assignments = list(product(range(n_fa), repeat=n_pairs))
        sum_rates = []
        for fa_indices in fa_assignments:
            channel_gain_db = channel_gains_db.copy()
            for rx in range(n_pairs):
                penalty_db = fa_penalty_db * fa_indices[rx]
                channel_gain_db[:, rx] -= penalty_db
            channel_gain = db_to_linear(channel_gain_db)
            sinr = sinr_linear(
                tx_power_lin=tx_power_lin,
                fa_indices=np.array(fa_indices),
                channel_gain=channel_gain,
                noise_power_lin=noise_power_lin,
            )
            sum_rate = sum_rate_bps(sinr, float(cfg['bandwidth_hz']))
            sum_rates.append(sum_rate)
        avg_sum_rate = np.mean(sum_rates)
        all_max_sum_rates.append(avg_sum_rate)
        # --- Average over all power and FA combinations (quantized power) ---
        if n_pairs <= 6:
            # Use power levels: {min, max-6, max-3, max} dBm, matching FS and config
            power_assignments = list(product(power_levels, repeat=n_pairs))
            fa_assignments = list(product(range(n_fa), repeat=n_pairs))
            # Prepare arguments for parallel processing
            args_list = [
                (tx_power_dbm_vec, fa_indices, channel_gains_db, fa_penalty_db, db_to_linear, sinr_linear, sum_rate_bps, float(cfg['bandwidth_hz']), noise_power_lin)
                for tx_power_dbm_vec in power_assignments
                for fa_indices in fa_assignments
            ]
            # Parallel computation
            sum_rates_all = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                delayed(compute_sum_rate_for_assignment)(*args)
                for args in args_list
            )
            avg_sum_rate_all = np.mean(sum_rates_all)
            all_power_fa_avg_sum_rates.append(avg_sum_rate_all)
        else:
            all_power_fa_avg_sum_rates.append(np.nan)
    plt.plot(sample_indices, dnn_sum_rates, label='DNN')
    plt.plot(sample_indices, fs_sum_rates, label='FS')
    plt.plot(sample_indices, all_max_sum_rates, '--', label='All Max Power (Uniform FA Avg)')
    if not all(np.isnan(all_power_fa_avg_sum_rates)):
        plt.plot(sample_indices, all_power_fa_avg_sum_rates, ':', label='All Power & FA Comb. Avg')
    # Add average lines
    dnn_avg = np.mean(dnn_sum_rates)
    fs_avg = np.mean(fs_sum_rates)
    all_max_avg = np.mean(all_max_sum_rates)
    if not all(np.isnan(all_power_fa_avg_sum_rates)):
        all_power_fa_avg = np.nanmean(all_power_fa_avg_sum_rates)
    else:
        all_power_fa_avg = np.nan
    plt.axhline(dnn_avg, color='C0', linestyle=':', linewidth=2, label=f'DNN Avg: {dnn_avg/1e6:.2f} Mbps')
    plt.axhline(fs_avg, color='C1', linestyle=':', linewidth=2, label=f'FS Avg: {fs_avg/1e6:.2f} Mbps')
    plt.axhline(all_max_avg, color='C2', linestyle=':', linewidth=2, label=f'All Max Power Avg: {all_max_avg/1e6:.2f} Mbps')
    if not np.isnan(all_power_fa_avg):
        plt.axhline(all_power_fa_avg, color='C3', linestyle=':', linewidth=2, label=f'All Power & FA Comb. Avg: {all_power_fa_avg/1e6:.2f} Mbps')
    # Annotate average values at the right edge
    x_right = sample_indices[-1] + 0.1
    plt.text(x_right, dnn_avg, f'{dnn_avg/1e6:.2f} Mbps', color='C0', va='center', fontsize=10, fontweight='bold')
    plt.text(x_right, fs_avg, f'{fs_avg/1e6:.2f} Mbps', color='C1', va='center', fontsize=10, fontweight='bold')
    plt.text(x_right, all_max_avg, f'{all_max_avg/1e6:.2f} Mbps', color='C2', va='center', fontsize=10, fontweight='bold')
    if not np.isnan(all_power_fa_avg):
        plt.text(x_right, all_power_fa_avg, f'{all_power_fa_avg/1e6:.2f} Mbps', color='C3', va='center', fontsize=10, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Sum Rate (bit/s)')
    plt.legend()
    plt.title('Sum Rate vs. Sample')
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        save_path_final = save_path
    else:
        os.makedirs(figs_dir, exist_ok=True)
        save_path_final = os.path.join(figs_dir, 'sum_rate_vs_sample.png')
    
    if save_path or figs_dir != ".":
        plt.savefig(save_path_final, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path_final}")
    
    if not no_show:
        plt.show()
    else:
        plt.close()


def plot_distance_distribution(tx_xy, rx_xy, show=True, save_path=None):
    """Plot histogram of TX-RX distances for all pairs."""
    distances = np.linalg.norm(tx_xy - rx_xy, axis=1)
    plt.figure(figsize=(5, 3))
    plt.hist(distances, bins=10, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel('TX-RX Distance (meters)')
    plt.ylabel('Count')
    plt.title('Distribution of TX-RX Distances')
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path.replace('.png', '_dist_hist.png'), dpi=300, bbox_inches='tight')
        print(f"Distance histogram saved to {save_path.replace('.png', '_dist_hist.png')}")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated samples or FS results")
    parser.add_argument("--npy_path", type=str, help="Path to .npy file with samples")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample to visualize")
    parser.add_argument("--save_path", type=str, help="Path to save the figure", default=None)
    parser.add_argument("--figs_dir", type=str, help="Directory to save figures (default: current directory)", default=".")
    parser.add_argument("--no_show", action="store_true", help="Don't display the plot (just save)")
    parser.add_argument("--area_size_m", type=float, help="Area size in meters (optional, overrides default/inferred)", default=None)
    parser.add_argument("--fs_result_yaml", type=str, help="YAML config to run FS and visualize power/FA", default=None)
    parser.add_argument("--fs_result_loc_yaml", type=str, help="YAML config to run FS and plot device locations with power/FA", default=None)
    parser.add_argument("--compare_allocations", action="store_true", help="Compare DNN and FS allocations side by side (uses same scenario for both)")
    parser.add_argument("--weights", type=str, help="Path to trained DNN weights", default="trained_weights.pt")
    parser.add_argument("--save_scenario", type=str, help="Path to save scenario (pickle)", default=None)
    parser.add_argument("--load_scenario", type=str, help="Path to load scenario (pickle)", default=None)
    parser.add_argument("--results_yaml", type=str, help="Path to YAML file with stored results", default=None)
    parser.add_argument("--plot_type", type=str, choices=["allocations", "power_vs_sample", "sum_rate_vs_sample", "ratio_vs_sample", "power_distribution"], default="allocations", help="Type of plot: allocations (default), power_vs_sample, sum_rate_vs_sample, or ratio_vs_sample")
    args = parser.parse_args()

    if args.compare_allocations:
        import torch
        from src.algorithms.ml_dnn import ML_DNN, dnn_output_to_decision
        from src.algorithms.fs_bruteforce import fs_bruteforce
        from src.sample_visualization import plot_allocations_side_by_side
        import sys
        # If --load_scenario is provided, use it exclusively
        if args.load_scenario:
            if args.fs_result_loc_yaml or args.fs_result_yaml:
                print("Warning: --fs_result_loc_yaml and --fs_result_yaml are ignored when --load_scenario is used.")
            scenario = pickle.load(open(args.load_scenario, "rb"))
            print(f"Loaded scenario from {args.load_scenario}")
            cfg = scenario.cfg
        else:
            # Fallback: require a config file
            config_path = args.fs_result_loc_yaml or args.fs_result_yaml
            if config_path is None:
                print("Error: --compare_allocations requires --load_scenario or --fs_result_loc_yaml/--fs_result_yaml to specify the config file.")
                sys.exit(1)
            from src.config import SimulationConfig
            from src.simulator.scenario import Scenario
            cfg = SimulationConfig.from_yaml(config_path)
            scenario = Scenario.random(cfg)
            if args.save_scenario:
                with open(args.save_scenario, "wb") as f:
                    pickle.dump(scenario, f)
                print(f"Saved scenario to {args.save_scenario}")
        # DNN allocation
        dnn_algo = ML_DNN(cfg)
        dnn_algo.load_weights(args.weights)
        x = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32).flatten().unsqueeze(0)
        with torch.no_grad():
            output = dnn_algo.model(x)[0]
        tx_power_dbm_dnn, fa_indices_dnn = dnn_output_to_decision(output, cfg)
        dnn_decision = {"tx_power_dbm": tx_power_dbm_dnn, "fa_indices": fa_indices_dnn}
        # FS allocation
        fs_algo = fs_bruteforce(cfg)
        fs_decision = fs_algo.decide(scenario)
        plot_allocations_side_by_side(scenario, dnn_decision, fs_decision, cfg)
        sys.exit(0)
    elif args.fs_result_loc_yaml:
        visualize_fs_result_locations(
            config_path=args.fs_result_loc_yaml,
            save_path=args.save_path,
            show=not args.no_show
        )
    elif args.fs_result_yaml:
        visualize_fs_result(
            config_path=args.fs_result_yaml,
            save_path=args.save_path,
            show=not args.no_show
        )
    elif args.results_yaml:
        # Visualize precomputed results from YAML
        with open(args.results_yaml, "r") as f:
            results_data = yaml.safe_load(f)
        samples_list = results_data.get("samples", [])
        if args.plot_type == "allocations":
            idx = args.sample_idx if args.sample_idx is not None else 0
            sample = next((s for s in samples_list if s.get("index") == idx), None)
            if sample is None:
                print(f"Sample index {idx} not found in {args.results_yaml}")
                sys.exit(1)
            cfg_dict = sample["config"]
            cfg = SimulationConfig(**cfg_dict)
            tx_xy = np.array(sample["tx_xy"])
            rx_xy = np.array(sample["rx_xy"])
            # Reconstruct scenario
            scenario = Scenario(cfg=cfg, tx_xy=tx_xy, rx_xy=rx_xy)
            dnn_decision = {
                "tx_power_dbm": np.array(sample["dnn"]["tx_power_dbm"]),
                "fa_indices": np.array(sample["dnn"]["fa_indices"]),
            }
            fs_decision = {
                "tx_power_dbm": np.array(sample["fs"]["tx_power_dbm"]),
                "fa_indices": np.array(sample["fs"]["fa_indices"]),
            }
            print(f"Visualizing sample {idx}: DNN sum-rate={sample['dnn']['sum_rate']:.2e}, FS sum-rate={sample['fs']['sum_rate']:.2e}, ratio={sample['ratio']:.3f}")
            plot_allocations_side_by_side(scenario, dnn_decision, fs_decision, cfg)
            sys.exit(0)
        elif args.plot_type == "power_vs_sample":
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            # Only plot the first pair's tx_power_dbm for DNN and FS
            n_samples = len(samples_list)
            dnn_powers = np.array([s["dnn"]["tx_power_dbm"][0] for s in samples_list])  # shape: (n_samples,)
            fs_powers = np.array([s["fs"]["tx_power_dbm"][0] for s in samples_list])    # shape: (n_samples,)
            sample_indices = np.array([s["index"] for s in samples_list])
            plt.figure(figsize=(10, 6))
            plt.plot(sample_indices, dnn_powers, 'o-', color='C0', label='DNN Tx 1')
            plt.plot(sample_indices, fs_powers, 's--', color='C1', label='FS Tx 1')
            # Plot average lines
            dnn_avg = np.mean(dnn_powers)
            fs_avg = np.mean(fs_powers)
            plt.axhline(dnn_avg, color='C0', linestyle=':', linewidth=2, label=f'DNN Avg: {dnn_avg:.2f} dBm')
            plt.axhline(fs_avg, color='C1', linestyle=':', linewidth=2, label=f'FS Avg: {fs_avg:.2f} dBm')
            # Annotate average values
            plt.text(sample_indices[-1]+0.1, dnn_avg, f'{dnn_avg:.2f}', color='C0', va='center', fontsize=10, fontweight='bold')
            plt.text(sample_indices[-1]+0.1, fs_avg, f'{fs_avg:.2f}', color='C1', va='center', fontsize=10, fontweight='bold')
            plt.xlabel('Sample Index')
            plt.ylabel('Tx Power (dBm)')
            plt.title('Transmit Power of First Pair per Sample (DNN vs FS)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure if requested
            if args.save_path:
                save_path = args.save_path
            else:
                os.makedirs(args.figs_dir, exist_ok=True)
                save_path = os.path.join(args.figs_dir, 'power_vs_sample.png')
            
            if args.save_path or args.figs_dir != ".":
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {save_path}")
            
            if not args.no_show:
                plt.show()
            else:
                plt.close()
            sys.exit(0)
        elif args.plot_type == "sum_rate_vs_sample":
            plot_sum_rate_vs_sample(samples_list, save_path=args.save_path, figs_dir=args.figs_dir, no_show=args.no_show)
            sys.exit(0)
        elif args.plot_type == "ratio_vs_sample":
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            # Extract ratios
            n_samples = len(samples_list)
            ratios = np.array([s["ratio"] for s in samples_list])
            sample_indices = np.array([s["index"] for s in samples_list])
            plt.figure(figsize=(10, 6))
            plt.plot(sample_indices, ratios, 'o-', color='C2', label='ML/FS Sum-Rate Ratio')
            # Plot average line
            ratio_avg = np.mean(ratios)
            plt.axhline(ratio_avg, color='C2', linestyle=':', linewidth=2, label=f'Avg Ratio: {ratio_avg:.3f}')
            # Annotate average value
            plt.text(sample_indices[-1]+0.1, ratio_avg, f'{ratio_avg:.3f}', color='C2', va='center', fontsize=10, fontweight='bold')
            plt.xlabel('Sample Index')
            plt.ylabel('ML/FS Sum-Rate Ratio')
            plt.title('ML Model / FS Sum-Rate Ratio per Sample')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure if requested
            if args.save_path:
                save_path = args.save_path
            else:
                os.makedirs(args.figs_dir, exist_ok=True)
                save_path = os.path.join(args.figs_dir, 'ratio_vs_sample.png')
            
            if args.save_path or args.figs_dir != ".":
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {save_path}")
            
            if not args.no_show:
                plt.show()
            else:
                plt.close()
            sys.exit(0)
        elif args.plot_type == "power_distribution":
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            # Collect all tx_power_dbm values for DNN and FS (all pairs, all samples)
            dnn_powers = np.concatenate([np.array(s["dnn"]["tx_power_dbm"]) for s in samples_list])
            fs_powers = np.concatenate([np.array(s["fs"]["tx_power_dbm"]) for s in samples_list])
            plt.figure(figsize=(10, 6))
            plt.hist(dnn_powers, bins=30, alpha=0.6, color='C0', label='DNN', density=True)
            plt.hist(fs_powers, bins=30, alpha=0.6, color='C1', label='FS', density=True)
            plt.xlabel('Transmit Power (dBm)')
            plt.ylabel('Density')
            plt.title('Distribution of Transmit Power (DNN vs FS, all pairs, all samples)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure if requested
            if args.save_path:
                save_path = args.save_path
            else:
                os.makedirs(args.figs_dir, exist_ok=True)
                save_path = os.path.join(args.figs_dir, 'power_distribution.png')
            
            if args.save_path or args.figs_dir != ".":
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {save_path}")
            
            if not args.no_show:
                plt.show()
            else:
                plt.close()
            sys.exit(0)
    elif args.npy_path:
        visualize_sample(
            sample_idx=args.sample_idx,
            npy_path=args.npy_path,
            show=not args.no_show,
            save_path=args.save_path,
            area_size_m=args.area_size_m
        )
    else:
        print("Please provide either --npy_path, --fs_result_yaml, or --fs_result_loc_yaml.") 