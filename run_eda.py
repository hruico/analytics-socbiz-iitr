"""
Exploratory Data Analysis (EDA) for EV Charging Data
Generates visualizations and insights for OP26 deliverables.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directories
OUTPUT_DIR = Path("outputs/eda")
FIGURES_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EV CHARGING DATA - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Load unified analytical base
print("\n[1/9] Loading data...")
df = pd.read_csv("data/processed/unified_analytical_base.csv")
print(f"✓ Loaded {len(df)} records")
print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df['time_step'].min()} to {df['time_step'].max()} steps")

# Basic statistics
print("\n[2/9] Computing summary statistics...")
summary_stats = df.describe()
summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")
print(f"✓ Summary statistics saved")

print("\nKey Metrics:")
print(f"  Sessions: {df['acn_sessions_count'].sum():.0f} total, {df['acn_sessions_count'].mean():.1f} avg/hour")
print(f"  Energy: {df['acn_total_kwh'].sum():.0f} kWh total, {df['acn_total_kwh'].mean():.1f} kWh/hour")
print(f"  Utilization: {df['urban_mean_utilization'].mean():.1%} mean, [{df['urban_mean_utilization'].min():.1%} - {df['urban_mean_utilization'].max():.1%}] range")
print(f"  Revenue: ₹{df['acn_base_revenue'].sum():.0f} total, ₹{df['acn_base_revenue'].mean():.1f} avg/hour")

# 1. TEMPORAL PATTERNS - Intraday
print("\n[3/9] Analyzing temporal patterns...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Hourly session volume
hourly_sessions = df.groupby('hour_of_day')['acn_sessions_count'].agg(['mean', 'std'])
axes[0, 0].bar(hourly_sessions.index, hourly_sessions['mean'], 
               yerr=hourly_sessions['std'], capsize=3, alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Average Sessions')
axes[0, 0].set_title('Intraday Session Volume Pattern')
axes[0, 0].grid(axis='y', alpha=0.3)

# Hourly utilization
hourly_util = df.groupby('hour_of_day')['urban_mean_utilization'].agg(['mean', 'std'])
axes[0, 1].plot(hourly_util.index, hourly_util['mean'], marker='o', linewidth=2, color='darkgreen')
axes[0, 1].fill_between(hourly_util.index, 
                         hourly_util['mean'] - hourly_util['std'],
                         hourly_util['mean'] + hourly_util['std'],
                         alpha=0.2, color='darkgreen')
axes[0, 1].axhline(y=0.80, color='red', linestyle='--', label='Surge threshold (80%)')
axes[0, 1].axhline(y=0.30, color='orange', linestyle='--', label='Discount threshold (30%)')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Mean Utilization')
axes[0, 1].set_title('Intraday Utilization Pattern')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Weekday vs Weekend
weekday_hourly = df[df['is_weekend']==0].groupby('hour_of_day')['acn_sessions_count'].mean()
weekend_hourly = df[df['is_weekend']==1].groupby('hour_of_day')['acn_sessions_count'].mean()
axes[1, 0].plot(weekday_hourly.index, weekday_hourly.values, marker='o', label='Weekday', linewidth=2)
axes[1, 0].plot(weekend_hourly.index, weekend_hourly.values, marker='s', label='Weekend', linewidth=2)
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Average Sessions')
axes[1, 0].set_title('Weekday vs Weekend Comparison')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Day of week pattern
daily_sessions = df.groupby('day_of_week')['acn_sessions_count'].mean()
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1, 1].bar(range(7), daily_sessions.values, color=['steelblue']*5 + ['coral']*2, alpha=0.7)
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(day_names)
axes[1, 1].set_xlabel('Day of Week')
axes[1, 1].set_ylabel('Average Sessions')
axes[1, 1].set_title('Weekly Pattern')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "temporal_patterns.png", dpi=300, bbox_inches='tight')
print(f"✓ Temporal patterns chart saved")
plt.close()

# 2. PEAK HOUR ANALYSIS
print("\n[4/9] Analyzing peak hours...")

# Identify peak vs off-peak
peak_sessions = df[df['is_peak_hour']==1]['acn_sessions_count'].mean()
offpeak_sessions = df[df['is_peak_hour']==0]['acn_sessions_count'].mean()
peak_hours = sorted(df[df['is_peak_hour']==1]['hour_of_day'].unique())

print(f"  Peak hours: {peak_hours}")
print(f"  Peak sessions: {peak_sessions:.1f} avg")
print(f"  Off-peak sessions: {offpeak_sessions:.1f} avg")
print(f"  Peak uplift: {(peak_sessions/offpeak_sessions - 1)*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Peak vs off-peak comparison
peak_data = df.groupby('is_peak_hour').agg({
    'acn_sessions_count': 'mean',
    'urban_mean_utilization': 'mean',
    'acn_base_revenue': 'mean'
})
x = ['Off-Peak', 'Peak']
width = 0.25

axes[0].bar([i - width for i in range(2)], peak_data['acn_sessions_count'], 
            width, label='Sessions', alpha=0.8)
axes[0].bar([i for i in range(2)], peak_data['urban_mean_utilization'] * 10, 
            width, label='Utilization (×10)', alpha=0.8)
axes[0].bar([i + width for i in range(2)], peak_data['acn_base_revenue'] / 10, 
            width, label='Revenue (÷10)', alpha=0.8)
axes[0].set_xticks(range(2))
axes[0].set_xticklabels(x)
axes[0].set_ylabel('Normalized Values')
axes[0].set_title('Peak vs Off-Peak Comparison')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Peak hour heatmap
pivot = df.pivot_table(values='acn_sessions_count', 
                       index='hour_of_day', 
                       columns='day_of_week', 
                       aggfunc='mean')
sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.1f', ax=axes[1], cbar_kws={'label': 'Avg Sessions'})
axes[1].set_xlabel('Day of Week (0=Mon, 6=Sun)')
axes[1].set_ylabel('Hour of Day')
axes[1].set_title('Session Volume Heatmap')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "peak_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Peak hour analysis saved")
plt.close()

# 3. UTILIZATION DISTRIBUTION
print("\n[5/9] Analyzing utilization distribution...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Distribution
axes[0].hist(df['urban_mean_utilization'], bins=30, color='teal', alpha=0.7, edgecolor='black')
axes[0].axvline(df['urban_mean_utilization'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["urban_mean_utilization"].mean():.1%}')
axes[0].axvline(0.30, color='orange', linestyle=':', linewidth=2, label='Discount threshold')
axes[0].axvline(0.80, color='darkred', linestyle=':', linewidth=2, label='Surge threshold')
axes[0].set_xlabel('Utilization Rate')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Utilization Distribution')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Regime classification
util_bins = pd.cut(df['urban_mean_utilization'], bins=[0, 0.30, 0.80, 1.0], 
                   labels=['Discount (<30%)', 'Neutral (30-80%)', 'Surge (>80%)'])
regime_counts = util_bins.value_counts()
colors = ['orange', 'steelblue', 'darkred']
axes[1].pie(regime_counts, labels=regime_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
axes[1].set_title('Utilization Regime Distribution')

# Volatility by hour
hourly_volatility = df.groupby('hour_of_day')['urban_mean_utilization'].std()
axes[2].bar(hourly_volatility.index, hourly_volatility.values, color='purple', alpha=0.7)
axes[2].set_xlabel('Hour of Day')
axes[2].set_ylabel('Std Dev of Utilization')
axes[2].set_title('Utilization Volatility by Hour')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "utilization_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Utilization analysis saved")
plt.close()

# Regime breakdown stats
print(f"\n  Utilization Regimes:")
for regime, count in regime_counts.items():
    print(f"    {regime}: {count} timesteps ({count/len(df)*100:.1f}%)")

# 4. REVENUE ANALYSIS
print("\n[6/9] Analyzing revenue patterns...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Revenue per hour
hourly_revenue = df.groupby('hour_of_day')['acn_base_revenue'].mean()
axes[0, 0].plot(hourly_revenue.index, hourly_revenue.values, marker='o', 
                linewidth=2, color='green')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Average Revenue (₹)')
axes[0, 0].set_title('Baseline Revenue by Hour')
axes[0, 0].grid(axis='y', alpha=0.3)

# Revenue vs Sessions correlation
axes[0, 1].scatter(df['acn_sessions_count'], df['acn_base_revenue'], alpha=0.5)
axes[0, 1].set_xlabel('Session Count')
axes[0, 1].set_ylabel('Revenue (₹)')
axes[0, 1].set_title(f'Revenue vs Sessions (R²={np.corrcoef(df["acn_sessions_count"], df["acn_base_revenue"])[0,1]**2:.3f})')
axes[0, 1].grid(alpha=0.3)

# Revenue per session
revenue_per_session = df['acn_base_revenue'] / df['acn_sessions_count'].replace(0, np.nan)
axes[1, 0].hist(revenue_per_session.dropna(), bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(revenue_per_session.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: ₹{revenue_per_session.mean():.2f}')
axes[1, 0].set_xlabel('Revenue per Session (₹)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Revenue per Session Distribution')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Cumulative revenue
df_sorted = df.sort_values('time_step')
cumulative_revenue = df_sorted['acn_base_revenue'].cumsum()
axes[1, 1].plot(df_sorted['time_step'], cumulative_revenue, linewidth=2, color='darkblue')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Cumulative Revenue (₹)')
axes[1, 1].set_title(f'Cumulative Revenue Over Time (Total: ₹{cumulative_revenue.iloc[-1]:.0f})')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "revenue_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Revenue analysis saved")
plt.close()

# 5. ENERGY CONSUMPTION
print("\n[7/9] Analyzing energy consumption...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Energy per session
energy_per_session = df['acn_total_kwh'] / df['acn_sessions_count'].replace(0, np.nan)
axes[0].hist(energy_per_session.dropna(), bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[0].axvline(energy_per_session.mean(), color='darkred', linestyle='--', 
                linewidth=2, label=f'Mean: {energy_per_session.mean():.2f} kWh')
axes[0].set_xlabel('Energy per Session (kWh)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Energy Consumption per Session')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Total energy by hour
hourly_energy = df.groupby('hour_of_day')['acn_total_kwh'].sum()
axes[1].bar(hourly_energy.index, hourly_energy.values, color='darkorange', alpha=0.7)
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Total Energy (kWh)')
axes[1].set_title('Total Energy Consumption by Hour')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "energy_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Energy analysis saved")
plt.close()

# 6. CORRELATION MATRIX
print("\n[8/9] Computing correlation matrix...")

fig, ax = plt.subplots(figsize=(10, 8))

# Select numeric columns for correlation
corr_cols = ['acn_sessions_count', 'acn_total_kwh', 'acn_avg_kwh_per_session',
             'acn_base_revenue', 'urban_mean_utilization', 'hour_of_day', 
             'day_of_week', 'is_weekend', 'is_peak_hour']
corr_matrix = df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300, bbox_inches='tight')
print(f"✓ Correlation matrix saved")
plt.close()

# 7. KEY INSIGHTS SUMMARY
print("\n[9/9] Generating insights summary...")

insights = []

# Peak hours
insights.append(f"Peak Hours: {peak_hours}")
insights.append(f"Peak hours account for {(df['is_peak_hour'].sum()/len(df)*100):.1f}% of timesteps but {(df[df['is_peak_hour']==1]['acn_sessions_count'].sum()/df['acn_sessions_count'].sum()*100):.1f}% of sessions")

# Utilization
insights.append(f"\nUtilization Statistics:")
insights.append(f"  Mean: {df['urban_mean_utilization'].mean():.1%}")
insights.append(f"  Std: {df['urban_mean_utilization'].std():.1%}")
insights.append(f"  Range: [{df['urban_mean_utilization'].min():.1%}, {df['urban_mean_utilization'].max():.1%}]")
insights.append(f"  Regime distribution: Discount={regime_counts.get('Discount (<30%)', 0)}, Neutral={regime_counts.get('Neutral (30-80%)', 0)}, Surge={regime_counts.get('Surge (>80%)', 0)}")

# Revenue
total_revenue = df['acn_base_revenue'].sum()
insights.append(f"\nRevenue Metrics:")
insights.append(f"  Total baseline revenue: ₹{total_revenue:.0f}")
insights.append(f"  Average revenue per hour: ₹{df['acn_base_revenue'].mean():.2f}")
insights.append(f"  Average revenue per session: ₹{revenue_per_session.mean():.2f}")

# Temporal patterns
peak_hour_util = df[df['is_peak_hour']==1]['urban_mean_utilization'].mean()
offpeak_hour_util = df[df['is_peak_hour']==0]['urban_mean_utilization'].mean()
insights.append(f"\nTemporal Patterns:")
insights.append(f"  Peak hour utilization: {peak_hour_util:.1%}")
insights.append(f"  Off-peak utilization: {offpeak_hour_util:.1%}")
insights.append(f"  Weekday avg sessions: {df[df['is_weekend']==0]['acn_sessions_count'].mean():.1f}")
insights.append(f"  Weekend avg sessions: {df[df['is_weekend']==1]['acn_sessions_count'].mean():.1f}")

# Pricing implications
insights.append(f"\nPricing Implications:")
insights.append(f"  {regime_counts.get('Surge (>80%)', 0)} timesteps ({regime_counts.get('Surge (>80%)', 0)/len(df)*100:.1f}%) qualify for surge pricing")
insights.append(f"  {regime_counts.get('Discount (<30%)', 0)} timesteps ({regime_counts.get('Discount (<30%)', 0)/len(df)*100:.1f}%) qualify for discount pricing")
insights.append(f"  Neutral pricing applies to {regime_counts.get('Neutral (30-80%)', 0)/len(df)*100:.1f}% of timesteps")

insights_text = '\n'.join(insights)
print(insights_text)

# Save insights
with open(OUTPUT_DIR / "key_insights.txt", 'w') as f:
    f.write("EV CHARGING DATA - KEY INSIGHTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(insights_text)

print(f"\n✓ Key insights saved to {OUTPUT_DIR / 'key_insights.txt'}")

# Generate EDA summary CSV
eda_summary = pd.DataFrame({
    'Metric': [
        'Total Records',
        'Total Sessions',
        'Total Energy (kWh)',
        'Total Baseline Revenue (₹)',
        'Mean Utilization',
        'Peak Hour Count',
        'Surge Timesteps',
        'Discount Timesteps',
        'Neutral Timesteps',
        'Peak Hours',
        'Mean Sessions per Hour',
        'Mean Revenue per Hour (₹)',
        'Mean Energy per Session (kWh)'
    ],
    'Value': [
        len(df),
        df['acn_sessions_count'].sum(),
        df['acn_total_kwh'].sum(),
        df['acn_base_revenue'].sum(),
        f"{df['urban_mean_utilization'].mean():.1%}",
        len(peak_hours),
        regime_counts.get('Surge (>80%)', 0),
        regime_counts.get('Discount (<30%)', 0),
        regime_counts.get('Neutral (30-80%)', 0),
        str(peak_hours),
        f"{df['acn_sessions_count'].mean():.2f}",
        f"{df['acn_base_revenue'].mean():.2f}",
        f"{energy_per_session.mean():.2f}"
    ]
})

eda_summary.to_csv(OUTPUT_DIR / "eda_summary.csv", index=False)
print(f"✓ EDA summary saved to {OUTPUT_DIR / 'eda_summary.csv'}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  📊 {FIGURES_DIR / 'temporal_patterns.png'}")
print(f"  📊 {FIGURES_DIR / 'peak_analysis.png'}")
print(f"  📊 {FIGURES_DIR / 'utilization_analysis.png'}")
print(f"  📊 {FIGURES_DIR / 'revenue_analysis.png'}")
print(f"  📊 {FIGURES_DIR / 'energy_analysis.png'}")
print(f"  📊 {FIGURES_DIR / 'correlation_matrix.png'}")
print(f"  📄 {OUTPUT_DIR / 'summary_statistics.csv'}")
print(f"  📄 {OUTPUT_DIR / 'eda_summary.csv'}")
print(f"  📄 {OUTPUT_DIR / 'key_insights.txt'}")
print("\nNext step: Review visualizations and create presentation slides")
