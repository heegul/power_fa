#!/usr/bin/env python3
"""
Visual Flow Diagram for Single Sample Mode Normalization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_normalization_flow_diagram():
    """Create a visual flow diagram showing the normalization process"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Single Sample Mode Normalization Flow', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Colors
    input_color = '#E8F4FD'
    process_color = '#FFF2CC'
    output_color = '#E1F5FE'
    critical_color = '#FFEBEE'
    
    # Step 1: Raw Channel Gains
    box1 = FancyBboxPatch((0.5, 9.5), 2, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=input_color, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(box1)
    ax.text(1.5, 10.1, 'Raw Channel Gains\n(6×6 matrix)\n-103.86 to -59.44 dB', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow 1
    ax.arrow(2.5, 10.1, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(2.9, 10.3, 'flatten()', ha='center', fontsize=8)
    
    # Step 2: Flattened Vector
    box2 = FancyBboxPatch((3.5, 9.5), 2, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=process_color, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(box2)
    ax.text(4.5, 10.1, 'Flattened Vector\n(36 elements)\nShape: [1, 36]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow 2
    ax.arrow(5.5, 10.1, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(5.9, 10.3, 'compute stats', ha='center', fontsize=8)
    
    # Step 3: Normalization Stats
    box3 = FancyBboxPatch((6.5, 9.5), 2.5, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=process_color, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(box3)
    ax.text(7.75, 10.1, 'Normalization Stats\nMean: -89.19 dB\nStd: 13.14 dB', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow 3 (down)
    ax.arrow(4.5, 9.5, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(4.8, 9.0, 'normalize', ha='left', fontsize=8)
    
    # Step 4: Normalized Input
    box4 = FancyBboxPatch((3.5, 7.5), 2, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=output_color, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(box4)
    ax.text(4.5, 8.1, 'Normalized Input\nMean: 0.0\nStd: 1.0\nRange: [-1.12, 2.26]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow 4
    ax.arrow(4.5, 7.5, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(4.8, 7.0, 'DNN forward', ha='left', fontsize=8)
    
    # Step 5: DNN Processing
    box5 = FancyBboxPatch((3.5, 5.5), 2, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=process_color, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(box5)
    ax.text(4.5, 6.1, 'DNN Processing\nHidden Layers\n+ Activation', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow 5
    ax.arrow(4.5, 5.5, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(4.8, 5.0, 'output', ha='left', fontsize=8)
    
    # Step 6: DNN Output
    box6 = FancyBboxPatch((3.5, 3.5), 2, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=output_color, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(box6)
    ax.text(4.5, 4.1, 'DNN Output\nPower Logits\nFA Logits', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow 6
    ax.arrow(4.5, 3.5, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(4.8, 3.0, 'convert', ha='left', fontsize=8)
    
    # Step 7: Power/FA Decisions
    box7 = FancyBboxPatch((3.5, 1.5), 2, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=output_color, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(box7)
    ax.text(4.5, 2.1, 'Power/FA Decisions\nTX Power (dBm)\nFA Indices', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # CRITICAL PATH: Original gains for loss
    # Arrow from raw gains to loss
    ax.arrow(1.5, 9.5, 0, -6.5, head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2, linestyle='--')
    ax.text(0.8, 6.5, 'ORIGINAL\nGAINS\n(for loss)', ha='center', fontsize=8, 
            color='red', fontweight='bold', rotation=90)
    
    # Step 8: Loss Calculation (CRITICAL)
    box8 = FancyBboxPatch((0.5, 1.5), 2, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=critical_color, 
                          edgecolor='red', linewidth=2)
    ax.add_patch(box8)
    ax.text(1.5, 2.1, 'SINR & Loss Calc\nUsing ORIGINAL\nChannel Gains!', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='red')
    
    # Arrow from decisions to loss
    ax.arrow(3.5, 2.1, -0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Key insights box
    key_box = FancyBboxPatch((6.5, 1.5), 3, 6, 
                             boxstyle="round,pad=0.2", 
                             facecolor='#F5F5F5', 
                             edgecolor='black', linewidth=1)
    ax.add_patch(key_box)
    
    ax.text(8, 7, '🔑 KEY INSIGHTS', ha='center', fontsize=12, fontweight='bold')
    
    insights = [
        "1. DNN Input: Normalized gains",
        "   • Mean = 0, Std = 1",
        "   • Stable training",
        "",
        "2. Loss Calculation: Original gains",
        "   • Actual dB values",
        "   • Accurate SINR",
        "",
        "3. Evaluation: Same normalization",
        "   • Consistent with training",
        "   • Same mean/std values",
        "",
        "4. Critical Separation:",
        "   • Normalized → DNN",
        "   • Original → Loss/SINR"
    ]
    
    for i, insight in enumerate(insights):
        color = 'red' if 'Original' in insight or 'Critical' in insight else 'black'
        weight = 'bold' if insight.startswith(('1.', '2.', '3.', '4.')) else 'normal'
        ax.text(6.7, 6.5 - i*0.3, insight, ha='left', fontsize=9, 
                color=color, fontweight=weight)
    
    # Add formula
    ax.text(8, 0.8, 'Normalization Formula:', ha='center', fontsize=10, fontweight='bold')
    ax.text(8, 0.4, 'x_norm = (x_raw - mean) / std', ha='center', fontsize=10, 
            fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    plt.tight_layout()
    plt.savefig('figs/normalization_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Normalization flow diagram saved to figs/normalization_flow_diagram.png")

def create_comparison_diagram():
    """Create a comparison diagram showing different normalization strategies"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    
    # Strategy 1: Per-sample normalization
    ax1.set_title('Per-Sample Normalization\n(Single Sample Mode)', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Sample 1
    box1 = FancyBboxPatch((1, 7), 8, 1.5, boxstyle="round,pad=0.1", 
                          facecolor='#E8F4FD', edgecolor='black')
    ax1.add_patch(box1)
    ax1.text(5, 7.75, 'Sample 1: Mean=-89.19, Std=13.14 → Norm=[0, 1]', 
             ha='center', va='center', fontsize=10)
    
    # Sample 2
    box2 = FancyBboxPatch((1, 5), 8, 1.5, boxstyle="round,pad=0.1", 
                          facecolor='#E8F4FD', edgecolor='black')
    ax1.add_patch(box2)
    ax1.text(5, 5.75, 'Sample 2: Mean=-88.12, Std=16.17 → Norm=[0, 1]', 
             ha='center', va='center', fontsize=10)
    
    # Sample 3
    box3 = FancyBboxPatch((1, 3), 8, 1.5, boxstyle="round,pad=0.1", 
                          facecolor='#E8F4FD', edgecolor='black')
    ax1.add_patch(box3)
    ax1.text(5, 3.75, 'Sample 3: Mean=-89.28, Std=15.42 → Norm=[0, 1]', 
             ha='center', va='center', fontsize=10)
    
    ax1.text(5, 1.5, '✅ Each sample normalized individually\n✅ Consistent DNN input range\n⚠️ Different scenarios, same normalization', 
             ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
    
    # Strategy 2: Global normalization
    ax2.set_title('Global Normalization\n(Batch Mode)', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Global stats
    global_box = FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1", 
                                facecolor='#FFF2CC', edgecolor='black')
    ax2.add_patch(global_box)
    ax2.text(5, 8.5, 'Global: Mean=-86.39, Std=15.49', 
             ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Sample 1
    box1 = FancyBboxPatch((1, 6.5), 8, 1, boxstyle="round,pad=0.1", 
                          facecolor='#E1F5FE', edgecolor='black')
    ax2.add_patch(box1)
    ax2.text(5, 7, 'Sample 1 → Norm=[-1.13, 1.74]', 
             ha='center', va='center', fontsize=10)
    
    # Sample 2
    box2 = FancyBboxPatch((1, 5), 8, 1, boxstyle="round,pad=0.1", 
                          facecolor='#E1F5FE', edgecolor='black')
    ax2.add_patch(box2)
    ax2.text(5, 5.5, 'Sample 2 → Norm=[-1.19, 2.09]', 
             ha='center', va='center', fontsize=10)
    
    # Sample 3
    box3 = FancyBboxPatch((1, 3.5), 8, 1, boxstyle="round,pad=0.1", 
                          facecolor='#E1F5FE', edgecolor='black')
    ax2.add_patch(box3)
    ax2.text(5, 4, 'Sample 3 → Norm=[-1.09, 2.75]', 
             ha='center', va='center', fontsize=10)
    
    ax2.text(5, 1.5, '✅ Consistent normalization across samples\n✅ Better generalization\n✅ Suitable for batch training', 
             ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
    
    # Strategy 3: Type-specific normalization
    ax3.set_title('Type-Specific Normalization\n(Our Previous Approach)', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # Restricted stats
    restricted_box = FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1", 
                                    facecolor='#FFEBEE', edgecolor='black')
    ax3.add_patch(restricted_box)
    ax3.text(5, 8.5, 'Restricted: Mean=-87.14, Std=15.98', 
             ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Normal stats
    normal_box = FancyBboxPatch((1, 6.5), 8, 1, boxstyle="round,pad=0.1", 
                                facecolor='#E8F5E8', edgecolor='black')
    ax3.add_patch(normal_box)
    ax3.text(5, 7, 'Normal: Mean=-92.86, Std=9.75', 
             ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Sample results
    ax3.text(5, 5, 'Restricted Sample → Norm=[-1.05, 1.73]\nNormal Sample → Norm=[different range]', 
             ha='center', va='center', fontsize=10)
    
    ax3.text(5, 1.5, '✅ Accounts for scenario differences\n⚠️ Requires knowing scenario type\n⚠️ More complex implementation', 
             ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figs/normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Normalization comparison diagram saved to figs/normalization_comparison.png")

if __name__ == "__main__":
    import os
    os.makedirs('figs', exist_ok=True)
    
    create_normalization_flow_diagram()
    create_comparison_diagram()
    
    print("\n🎯 SUMMARY:")
    print("   - Flow diagram shows the complete normalization process")
    print("   - Comparison diagram shows different normalization strategies")
    print("   - Key insight: Normalized input for DNN, original gains for loss") 