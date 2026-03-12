#!/usr/bin/env python3
"""
Script to read awq_channel_avgs.bin and compute top 1% salient channel indices.
Prints verification information about the data.
"""

import struct
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def read_channel_averages(filename):
    """Read the binary file containing channel averages."""
    with open(filename, 'rb') as f:
        # Read header
        L = struct.unpack('i', f.read(4))[0]
        C = struct.unpack('i', f.read(4))[0]
        total_tokens = struct.unpack('Q', f.read(8))[0]  # size_t = unsigned long long (8 bytes)
        
        # Read averages
        total_channels = L * C * 7  # L*C + L*C + L*C + L*4*C
        averages = np.frombuffer(f.read(total_channels * 4), dtype=np.float32)
        
        return L, C, total_tokens, averages

def compute_salient_indices(averages, L, C):
    """Compute top 1% salient channel indices per layer per tensor type."""
    k_c = int(np.ceil(0.01 * C))      # top 1% of C
    k_4c = int(np.ceil(0.01 * 4 * C)) # top 1% of 4*C
    
    # Layout: [L*C ln1] [L*C atty] [L*C ln2] [L*4*C fch_gelu]
    offset_ln1 = 0
    offset_atty = L * C
    offset_ln2 = 2 * L * C
    offset_fch_gelu = 3 * L * C
    
    # Allocate arrays for indices
    salient_ln1 = np.zeros((L, k_c), dtype=np.uint32)
    salient_atty = np.zeros((L, k_c), dtype=np.uint32)
    salient_ln2 = np.zeros((L, k_c), dtype=np.uint32)
    salient_fch_gelu = np.zeros((L, k_4c), dtype=np.uint32)
    
    for l in range(L):
        # ln1: top k_c channels from layer l
        segment = averages[offset_ln1 + l*C : offset_ln1 + (l+1)*C]
        salient_ln1[l] = np.argpartition(segment, -k_c)[-k_c:]
        salient_ln1[l] = salient_ln1[l][np.argsort(segment[salient_ln1[l]])[::-1]]
        
        # atty: top k_c channels from layer l
        segment = averages[offset_atty + l*C : offset_atty + (l+1)*C]
        salient_atty[l] = np.argpartition(segment, -k_c)[-k_c:]
        salient_atty[l] = salient_atty[l][np.argsort(segment[salient_atty[l]])[::-1]]
        
        # ln2: top k_c channels from layer l
        segment = averages[offset_ln2 + l*C : offset_ln2 + (l+1)*C]
        salient_ln2[l] = np.argpartition(segment, -k_c)[-k_c:]
        salient_ln2[l] = salient_ln2[l][np.argsort(segment[salient_ln2[l]])[::-1]]
        
        # fch_gelu: top k_4c channels from layer l
        segment = averages[offset_fch_gelu + l*4*C : offset_fch_gelu + (l+1)*4*C]
        salient_fch_gelu[l] = np.argpartition(segment, -k_4c)[-k_4c:]
        salient_fch_gelu[l] = salient_fch_gelu[l][np.argsort(segment[salient_fch_gelu[l]])[::-1]]
    
    return salient_ln1, salient_atty, salient_ln2, salient_fch_gelu, k_c, k_4c

def graph_layer(ax, segment, layer_num, color):
    """
    Create a histogram for a single layer with statistics.
    
    Args:
        ax: matplotlib axis object
        segment: data array for this layer
        layer_num: layer number for title
        color: color for the histogram
    """
    # Create histogram with KDE
    sns.histplot(segment, bins=50, kde=True, ax=ax, color=color)
    
    # Compute statistics
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    min_val = np.min(segment)
    max_val = np.max(segment)
    median_val = np.median(segment)
    q1 = np.percentile(segment, 25)
    q3 = np.percentile(segment, 75)
    iqr = q3 - q1
    
    # Add statistics as text box
    stats_text = (f'Mean: {mean_val:.4f}\n'
                  f'Std: {std_val:.4f}\n'
                  f'Min: {min_val:.4f}\n'
                  f'Max: {max_val:.4f}\n'
                  f'Median: {median_val:.4f}\n'
                  f'IQR: {iqr:.4f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=8)
    
    # Set labels
    ax.set_title(f'Layer {layer_num}')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Count')

def graph_tensor(averages, L, offset, channels_per_layer, tensor_name, color):
    """
    Create a figure with histograms for all layers of a specific tensor type.
    
    Args:
        averages: full array of channel averages
        L: number of layers
        offset: starting offset in averages array
        channels_per_layer: number of channels per layer (C or 4*C)
        tensor_name: name for the figure title
        color: color for histograms
    
    Returns:
        matplotlib figure object
    """
    # Determine subplot layout (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(L)))
    n_rows = int(np.ceil(L / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    fig.suptitle(f'{tensor_name} - Channel Magnitude Distribution by Layer', 
                 fontsize=16, y=0.995)
    axes = axes.flatten() if L > 1 else [axes]
    
    # Create histogram for each layer
    for l in range(L):
        segment = averages[offset + l*channels_per_layer : offset + (l+1)*channels_per_layer]
        graph_layer(axes[l], segment, l, color)
    
    # Hide unused subplots
    for l in range(L, len(axes)):
        axes[l].set_visible(False)
    
    fig.tight_layout()
    return fig

def plot_layer_histograms(averages, L, C):
    """
    Create histograms by layer for all 4 tensor types using Seaborn.
    Creates 4 separate figures, one for each tensor type.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Layout: [L*C ln1] [L*C atty] [L*C ln2] [L*4*C fch_gelu]
    offset_ln1 = 0
    offset_atty = L * C
    offset_ln2 = 2 * L * C
    offset_fch_gelu = 3 * L * C
    
    # Create figures for each tensor type
    fig1 = graph_tensor(averages, L, offset_ln1, C, 'LayerNorm 1 (ln1)', 'steelblue')
    fig2 = graph_tensor(averages, L, offset_atty, C, 'Attention Output (atty)', 'coral')
    fig3 = graph_tensor(averages, L, offset_ln2, C, 'LayerNorm 2 (ln2)', 'seagreen')
    fig4 = graph_tensor(averages, L, offset_fch_gelu, 4*C, 'Feed-Forward GELU (fch_gelu)', 'mediumpurple')
    
    print("\n" + "="*70)
    print("HISTOGRAMS GENERATED")
    print("="*70)
    print(f"Created 4 figures with histograms for all {L} layers")
    print("Close the figure windows to continue...")

def main():
    # Read the binary file
    filename = 'awq_channel_avgs.bin'
    
    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        return
    
    print(f"Reading {filename}...")
    L, C, total_tokens, averages = read_channel_averages(filename)
    
    print("\n" + "="*70)
    print("FILE HEADER INFORMATION")
    print("="*70)
    print(f"Number of layers (L):     {L}")
    print(f"Channels per layer (C):   {C}")
    print(f"Total tokens processed:   {total_tokens}")
    print(f"Total channels stored:    {len(averages)} (expected: {L*C*7})")
    
    # Generate and display histograms
    plot_layer_histograms(averages, L, C)
    plt.show()

if __name__ == "__main__":
    main()
