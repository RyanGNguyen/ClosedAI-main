import matplotlib.pyplot as plt
import numpy as np

# Data extracted from nsys profiling results (in milliseconds)
# Converting from nanoseconds to milliseconds for readability
# Longer sequence length run

# Flash Attention with Tensor Cores (TF32)
flash_tc = {
    'flash_attn_kernel': 121.111,   # 121,111,012 ns
    'permute_kernel': 18.133,       # 18,132,513 ns
    'unpermute_kernel': 4.047,      # 4,046,570 ns
}

# Flash Attention without Tensor Cores
flash_no_tc = {
    'flash_attn_kernel': 502.000,   # 502,000,397 ns
    'permute_kernel': 18.154,       # 18,153,695 ns
    'unpermute_kernel': 4.044,      # 4,044,075 ns
}

# Classic Attention (multiple kernels)
classic = {
    'pv_matmul_kernel': 476.017,     # 476,017,277 ns
    'qk_matmul_kernel': 131.321,     # 131,321,346 ns (matmul_forward_kernel)
    'softmax_kernel': 18.062,        # 18,062,017 ns
    'permute_kernel': 18.313,        # 18,313,214 ns
    'unpermute_kernel': 4.087,       # 4,086,922 ns
}

# Use a nice color palette
# Palette: Warm sunset / coral theme
palette = {
    'coral': '#FF6B6B',
    'orange': '#FFA07A', 
    'gold': '#FFD93D',
    'teal': '#4ECDC4',
    'navy': '#2C3E50',
    'purple': '#9B59B6',
    'blue': '#3498DB',
    'mint': '#1ABC9C',
    'slate': '#34495E',
    'pink': '#E91E63'
}

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#FAFAFA')

# ============== Plot 1: Stacked Bar Chart ==============
ax1 = axes[0]
ax1.set_facecolor('#FAFAFA')

# Prepare data for stacked bar
implementations = ['Flash Attn\n(With Tensor Core)', 'Flash Attn\n(Without Tensor Core)', 'Naive Attn\n(With Tensor Core)']

# Stack components
flash_tc_stack = [flash_tc['flash_attn_kernel'], flash_tc['permute_kernel'], flash_tc['unpermute_kernel']]
flash_no_tc_stack = [flash_no_tc['flash_attn_kernel'], flash_no_tc['permute_kernel'], flash_no_tc['unpermute_kernel']]
classic_stack = [
    classic['qk_matmul_kernel'] + classic['softmax_kernel'] + classic['pv_matmul_kernel'],  # Core attention
    classic['permute_kernel'], 
    classic['unpermute_kernel']
]

x = np.arange(len(implementations))
width = 0.6

# Colors - warm palette
colors = [palette['teal'], palette['coral'], palette['gold']]
labels = ['Attention Foward', 'Permute', 'Unpermute']

# Stack the bars
bottom = np.zeros(3)
data = np.array([
    [flash_tc_stack[0], flash_no_tc_stack[0], classic_stack[0]],  # Core attention
    [flash_tc_stack[1], flash_no_tc_stack[1], classic_stack[1]],  # Permute
    [flash_tc_stack[2], flash_no_tc_stack[2], classic_stack[2]],  # Unpermute
])

for i, (row, color, label) in enumerate(zip(data, colors, labels)):
    bars = ax1.bar(x, row, width, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=1.5)
    bottom += row

# Add total time labels on top
totals = [sum(flash_tc_stack), sum(flash_no_tc_stack), sum(classic_stack)]
for i, total in enumerate(totals):
    ax1.annotate(f'{total:.1f} ms', xy=(i, total), ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Total Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Attention Kernel Total Latency Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(implementations, fontsize=11)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.set_ylim(0, max(totals) * 1.15)
ax1.grid(axis='y', alpha=0.4, linestyle='--', color='gray')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ============== Plot 2: Core Attention Breakdown ==============
ax2 = axes[1]
ax2.set_facecolor('#FAFAFA')

# For classic attention, break down the core attention components
classic_breakdown = {
    'Q@K Matmul': classic['qk_matmul_kernel'],
    'Softmax': classic['softmax_kernel'],
    'P@V Matmul': classic['pv_matmul_kernel'],
}

# Grouped bar chart for core attention only
x2 = np.arange(3)
width2 = 0.25

# Flash TC - single fused kernel
bars1 = ax2.bar(x2[0] - width2, flash_tc['flash_attn_kernel'], width2, 
                label='Flash Attention (With Tensor Core)', color=palette['teal'], edgecolor='white', linewidth=1.5)
# Flash No TC - single fused kernel  
bars2 = ax2.bar(x2[0], flash_no_tc['flash_attn_kernel'], width2,
                label='Flash Attention (Without Tensor Core)', color=palette['coral'], edgecolor='white', linewidth=1.5)

# Classic - show breakdown
classic_colors = [palette['navy'], palette['purple'], palette['orange']]
bottom = 0
for i, (name, val) in enumerate(classic_breakdown.items()):
    ax2.bar(x2[0] + width2, val, width2, bottom=bottom, 
            label=f'Naive Attention: {name}', color=classic_colors[i], edgecolor='white', linewidth=1.5)
    bottom += val

# Add value labels
ax2.annotate(f'{flash_tc["flash_attn_kernel"]:.1f}', 
             xy=(x2[0] - width2, flash_tc['flash_attn_kernel']), 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.annotate(f'{flash_no_tc["flash_attn_kernel"]:.1f}', 
             xy=(x2[0], flash_no_tc['flash_attn_kernel']), 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.annotate(f'{sum(classic_breakdown.values()):.1f}', 
             xy=(x2[0] + width2, sum(classic_breakdown.values())), 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax2.set_title('Core Attention Kernel Breakdown', fontsize=13, fontweight='bold')
ax2.set_xticks([x2[0]])
ax2.set_xticklabels(['Core Attention Kernels'], fontsize=11)
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax2.grid(axis='y', alpha=0.4, linestyle='--', color='gray')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('./attention_latency_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('./attention_latency_comparison.pdf', bbox_inches='tight')
print("Plots saved!")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY: Attention Kernel Performance Comparison")
print("="*60)

print(f"\n{'Implementation':<25} {'Total (ms)':<12} {'Speedup vs Classic':<20}")
print("-"*60)
classic_total = sum(classic.values())
flash_tc_total = sum(flash_tc.values())
flash_no_tc_total = sum(flash_no_tc.values())

print(f"{'Naive Attention (With Tensor Core)':<25} {classic_total:<12.2f} {'1.00x (baseline)':<20}")
print(f"{'Flash Attn (Without Tensor Core)':<25} {flash_no_tc_total:<12.2f} {classic_total/flash_no_tc_total:<20.2f}x")
print(f"{'Flash Attn (With Tensor Core)':<25} {flash_tc_total:<12.2f} {classic_total/flash_tc_total:<20.2f}x")

print(f"\n{'Core Attention Only:':<25}")
print("-"*60)
classic_core = classic['pv_matmul_kernel'] + classic['qk_matmul_kernel'] + classic['softmax_kernel']
print(f"{'Classic (Q@K + Softmax + P@V)':<25} {classic_core:<12.2f} {'1.00x (baseline)':<20}")
print(f"{'Flash (No TC)':<25} {flash_no_tc['flash_attn_kernel']:<12.2f} {classic_core/flash_no_tc['flash_attn_kernel']:<20.2f}x")
print(f"{'Flash (TF32 TC)':<25} {flash_tc['flash_attn_kernel']:<12.2f} {classic_core/flash_tc['flash_attn_kernel']:<20.2f}x")

print(f"\nTensor Core Speedup: {flash_no_tc['flash_attn_kernel']/flash_tc['flash_attn_kernel']:.2f}x faster")