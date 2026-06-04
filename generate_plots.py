#!/usr/bin/env python3
"""
Generate presentation plots from agentic outcomes.

This script must be run AFTER run_agentic.py has completed and generated
outputs/agentic_outcomes.csv. It produces four publication-quality plots
summarizing the three-agent optimization results.

Outputs:
- outputs/figures/price_trajectory.png
- outputs/figures/parameter_convergence.png
- outputs/figures/revenue_progression.png
- outputs/figures/utilization_price_scatter.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure matplotlib for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Load data
outcomes = pd.read_csv('outputs/agentic_outcomes.csv')

# Create output directory
Path('outputs/figures').mkdir(parents=True, exist_ok=True)

# Define regime colors
regime_colors = {
    'surge': '#E74C3C',     # Red
    'neutral': '#3498DB',   # Blue
    'discount': '#2ECC71'   # Green
}

print("=" * 70)
print("GENERATING PRESENTATION PLOTS")
print("=" * 70)

# ==============================================================================
# PLOT 1: Price Trajectory
# ==============================================================================
print("\n[1/4] Generating Price Trajectory plot...")

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Plot points colored by regime
for regime, color in regime_colors.items():
    mask = outcomes['regime'] == regime
    if mask.any():
        ax.scatter(
            outcomes.loc[mask, 'step'],
            outcomes.loc[mask, 'p_new'],
            c=color,
            label=regime.capitalize(),
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

# Connect points with line
ax.plot(outcomes['step'], outcomes['p_new'], 'k-', alpha=0.3, linewidth=1)

# Baseline
ax.axhline(y=15.0, color='gray', linestyle='--', linewidth=2, label='₹15 Baseline')

ax.set_xlabel('Optimization Step')
ax.set_ylabel('Price (₹/kWh)')
ax.set_title('Dynamic Price Decisions Across 34 Steps')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/price_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved to outputs/figures/price_trajectory.png")

# ==============================================================================
# PLOT 2: Parameter Convergence
# ==============================================================================
print("\n[2/4] Generating Parameter Convergence plot...")

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Plot parameter trajectories
ax.plot(outcomes['step'], outcomes['epsilon'], 'o-', 
        color='#9B59B6', linewidth=2, markersize=4, 
        label=f'ε: 0.250→{outcomes["epsilon"].iloc[-1]:.3f}')
ax.plot(outcomes['step'], outcomes['alpha'], 's-', 
        color='#E67E22', linewidth=2, markersize=4,
        label=f'α: 2.500→{outcomes["alpha"].iloc[-1]:.3f}')
ax.plot(outcomes['step'], outcomes['beta'], '^-', 
        color='#16A085', linewidth=2, markersize=4,
        label=f'β: 2.500→{outcomes["beta"].iloc[-1]:.3f}')

# Initial value reference lines
ax.axhline(y=0.25, color='#9B59B6', linestyle='--', linewidth=1, alpha=0.3)
ax.axhline(y=2.5, color='#E67E22', linestyle='--', linewidth=1, alpha=0.3)
ax.axhline(y=2.5, color='#16A085', linestyle='--', linewidth=1, alpha=0.3)

ax.set_xlabel('Optimization Step')
ax.set_ylabel('Parameter Value')
ax.set_title('Monitoring Agent Parameter Convergence')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/parameter_convergence.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved to outputs/figures/parameter_convergence.png")

# ==============================================================================
# PLOT 3: Revenue Gain Progression
# ==============================================================================
print("\n[3/4] Generating Revenue Gain Progression plot...")

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Plot revenue gain line
ax.plot(outcomes['step'], outcomes['revenue_gain_pct'], 
        'o-', color='#34495E', linewidth=2, markersize=4)

# Fill areas
positive_mask = outcomes['revenue_gain_pct'] >= 0
ax.fill_between(outcomes['step'], 0, outcomes['revenue_gain_pct'],
                where=positive_mask, color='#2ECC71', alpha=0.3, label='Gain')
ax.fill_between(outcomes['step'], 0, outcomes['revenue_gain_pct'],
                where=~positive_mask, color='#E74C3C', alpha=0.3, label='Loss')

# Zero line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Annotate at peak (step 22)
peak_step = 22
peak_gain = outcomes.loc[outcomes['step'] == peak_step, 'revenue_gain_pct'].values[0]
ax.annotate('+3.13% avg gain',
           xy=(peak_step, peak_gain),
           xytext=(10, 15), textcoords='offset points',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Optimization Step')
ax.set_ylabel('Revenue Gain vs Baseline (%)')
ax.set_title('Revenue Gain % Over Optimization Steps')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/revenue_progression.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved to outputs/figures/revenue_progression.png")

# ==============================================================================
# PLOT 4: Utilisation vs Price Scatter
# ==============================================================================
print("\n[4/4] Generating Utilisation vs Price Scatter plot...")

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Scatter points colored by regime
for regime, color in regime_colors.items():
    mask = outcomes['regime'] == regime
    if mask.any():
        ax.scatter(
            outcomes.loc[mask, 'u_pred'] * 100,
            outcomes.loc[mask, 'p_new'],
            c=color,
            label=regime.capitalize(),
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

# Regime boundary lines
ax.axvline(x=30, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axvline(x=80, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# Baseline price line
ax.axhline(y=15.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='₹15 Baseline')

# Annotate regime zones
ax.text(15, ax.get_ylim()[1] * 0.95, 'Discount\nZone', 
        ha='center', va='top', fontsize=9, style='italic', alpha=0.6)
ax.text(55, ax.get_ylim()[1] * 0.95, 'Neutral\nZone',
        ha='center', va='top', fontsize=9, style='italic', alpha=0.6)
ax.text(90, ax.get_ylim()[1] * 0.95, 'Surge\nZone',
        ha='center', va='top', fontsize=9, style='italic', alpha=0.6)

ax.set_xlabel('Predicted Utilisation (%)')
ax.set_ylabel('Price (₹/kWh)')
ax.set_title('Predicted Utilisation vs Dynamic Price by Regime')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/utilization_price_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved to outputs/figures/utilization_price_scatter.png")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 70)
print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 70)
print("\nOutput files:")
print("  1. outputs/figures/price_trajectory.png")
print("  2. outputs/figures/parameter_convergence.png")
print("  3. outputs/figures/revenue_progression.png")
print("  4. outputs/figures/utilization_price_scatter.png")
print("\nPlots ready for presentation and publication.")
