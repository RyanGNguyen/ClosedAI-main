#!/usr/bin/env python3
"""
Script to read awq_optimal_scales.bin and analyze the AWQ activation scales.
These scales are applied during inference to balance quantization error.
"""

import struct
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def read_optimal_scales(filename):
    """
    Read the binary file containing AWQ optimal scales.
    
    File format:
    - Header: L (int), C (int)
    - Data: L*C*7 floats
    
    Layout (L*C*7):
    - ln1 scales: (L, C) - for ln1 -> qkv projection
    - atty scales: (L, C) - for atty -> attproj projection  
    - ln2 scales: (L, C) - for ln2 -> fc projection
    - fch_gelu scales: (L, 4*C) - for fch_gelu -> fcproj projection
    
    Returns:
        L, C, scales_dict with keys 'ln1', 'atty', 'ln2', 'fch_gelu'
    """
    with open(filename, 'rb') as f:
        # Read header
        L = struct.unpack('i', f.read(4))[0]
        C = struct.unpack('i', f.read(4))[0]
        
        # Read all scales
        total_size = L * C * 7
        all_scales = np.frombuffer(f.read(total_size * 4), dtype=np.float32)
        
        # Parse into individual scale groups
        scales = {}
        offset = 0
        
        scales['ln1'] = all_scales[offset:offset + L*C].reshape(L, C)
        offset += L * C
        
        scales['atty'] = all_scales[offset:offset + L*C].reshape(L, C)
        offset += L * C
        
        scales['ln2'] = all_scales[offset:offset + L*C].reshape(L, C)
        offset += L * C
        
        scales['fch_gelu'] = all_scales[offset:offset + L*4*C].reshape(L, 4*C)
        offset += L * 4 * C
        
        return L, C, scales

def analyze_scale_group(scales, name):
    """Compute and print statistics for a scale group."""
    print(f"\n{name}:")
    print(f"  Shape: {scales.shape}")
    print(f"  Range: [{np.min(scales):.6f}, {np.max(scales):.6f}]")
    print(f"  Mean: {np.mean(scales):.6f}")
    print(f"  Median: {np.median(scales):.6f}")
    print(f"  Std: {np.std(scales):.6f}")
    
    # Check for problematic values
    num_extreme_low = np.sum(scales < 0.1)
    num_extreme_high = np.sum(scales > 10.0)
    num_near_one = np.sum((scales >= 0.9) & (scales <= 1.1))
    
    print(f"  Values < 0.1: {num_extreme_low} ({100*num_extreme_low/scales.size:.2f}%)")
    print(f"  Values > 10.0: {num_extreme_high} ({100*num_extreme_high/scales.size:.2f}%)")
    print(f"  Values near 1.0 (0.9-1.1): {num_near_one} ({100*num_near_one/scales.size:.2f}%)")
    
    # Percentiles
    p1 = np.percentile(scales, 1)
    p5 = np.percentile(scales, 5)
    p25 = np.percentile(scales, 25)
    p75 = np.percentile(scales, 75)
    p95 = np.percentile(scales, 95)
    p99 = np.percentile(scales, 99)
    
    print(f"  Percentiles:")
    print(f"    1%: {p1:.6f}, 5%: {p5:.6f}, 25%: {p25:.6f}")
    print(f"    75%: {p75:.6f}, 95%: {p95:.6f}, 99%: {p99:.6f}")

def plot_scale_distributions(scales_dict, L, C):
    """Create visualizations for all scale groups."""
    sns.set_style("whitegrid")
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AWQ Optimal Activation Scales Distribution', fontsize=16)
    
    scale_info = [
        ('ln1', 'LN1 → QKV Scales', axes[0, 0], 'steelblue'),
        ('atty', 'ATTY → ATTPROJ Scales', axes[0, 1], 'coral'),
        ('ln2', 'LN2 → FC Scales', axes[1, 0], 'seagreen'),
        ('fch_gelu', 'FCH_GELU → FCPROJ Scales', axes[1, 1], 'mediumpurple'),
    ]
    
    for key, title, ax, color in scale_info:
        scale_data = scales_dict[key].flatten()
        
        # Create histogram
        sns.histplot(scale_data, bins=100, kde=True, ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel('Scale Value')
        ax.set_ylabel('Count')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Scale=1.0')
        ax.legend()
        
        # Add statistics text
        stats_text = (f'Mean: {np.mean(scale_data):.4f}\n'
                     f'Median: {np.median(scale_data):.4f}\n'
                     f'Std: {np.std(scale_data):.4f}\n'
                     f'Min: {np.min(scale_data):.4f}\n'
                     f'Max: {np.max(scale_data):.4f}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_scale_heatmaps(scales_dict, L, C):
    """Create heatmaps showing scale values per layer and channel."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AWQ Scales Heatmap (Per Layer & Channel)', fontsize=16)
    
    scale_info = [
        ('ln1', 'LN1 → QKV', axes[0, 0]),
        ('atty', 'ATTY → ATTPROJ', axes[0, 1]),
        ('ln2', 'LN2 → FC', axes[1, 0]),
        ('fch_gelu', 'FCH_GELU → FCPROJ', axes[1, 1]),
    ]
    
    for key, title, ax in scale_info:
        scale_data = scales_dict[key]
        
        # Create heatmap
        im = ax.imshow(scale_data, aspect='auto', cmap='RdYlBu_r', 
                      vmin=0.5, vmax=1.5, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Layer Index')
        ax.set_yticks(range(L))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Scale Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig

def check_scale_issues(scales_dict):
    """Check for potential issues with scale values."""
    print("\n" + "="*70)
    print("POTENTIAL ISSUES CHECK")
    print("="*70)
    
    issues_found = False
    
    for key, scales in scales_dict.items():
        print(f"\n{key}:")
        
        # Check for zeros or near-zeros (would cause division issues)
        near_zero = np.sum(scales < 0.01)
        if near_zero > 0:
            print(f"  ⚠️  WARNING: {near_zero} values < 0.01 (near-zero, division risk!)")
            issues_found = True
        else:
            print(f"  ✓ No near-zero values")
        
        # Check for very large values (could cause overflow)
        very_large = np.sum(scales > 100.0)
        if very_large > 0:
            print(f"  ⚠️  WARNING: {very_large} values > 100.0 (very large!)")
            issues_found = True
        else:
            print(f"  ✓ No extremely large values")
        
        # Check for NaN or Inf
        nan_count = np.sum(np.isnan(scales))
        inf_count = np.sum(np.isinf(scales))
        if nan_count > 0 or inf_count > 0:
            print(f"  ❌ CRITICAL: {nan_count} NaN values, {inf_count} Inf values!")
            issues_found = True
        else:
            print(f"  ✓ No NaN/Inf values")
        
        # Check variance (scales should vary across channels)
        if np.std(scales) < 0.001:
            print(f"  ⚠️  WARNING: Very low variance ({np.std(scales):.6f}), scales may be uniform")
            issues_found = True
        else:
            print(f"  ✓ Good variance in scale values")
    
    if not issues_found:
        print("\n✅ All scale groups look healthy!")
    else:
        print("\n⚠️  Some issues detected - review warnings above")

def main():
    filename = 'awq_optimal_scales.bin'
    
    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        print("Please run awq_grid_search.cu first to generate this file.")
        return
    
    print("="*70)
    print("AWQ OPTIMAL SCALES ANALYSIS")
    print("="*70)
    print(f"\nReading {filename}...")
    
    L, C, scales_dict = read_optimal_scales(filename)
    
    print(f"\nModel Configuration:")
    print(f"  Number of layers (L): {L}")
    print(f"  Channels (C): {C}")
    print(f"  Total scale values: {L * C * 7:,}")
    
    print("\n" + "="*70)
    print("SCALE STATISTICS BY GROUP")
    print("="*70)
    
    # Analyze each scale group
    analyze_scale_group(scales_dict['ln1'], "LN1 → QKV Scales")
    analyze_scale_group(scales_dict['atty'], "ATTY → ATTPROJ Scales")
    analyze_scale_group(scales_dict['ln2'], "LN2 → FC Scales")
    analyze_scale_group(scales_dict['fch_gelu'], "FCH_GELU → FCPROJ Scales")
    
    # Check for issues
    check_scale_issues(scales_dict)
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    print("Creating distribution plots...")
    fig1 = plot_scale_distributions(scales_dict, L, C)
    
    print("Creating heatmaps...")
    fig2 = plot_scale_heatmaps(scales_dict, L, C)
    
    print("\n✅ Visualization complete!")
    print("Close the figure windows to exit...")
    plt.show()

if __name__ == "__main__":
    main()
