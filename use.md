# 🚀 How to Run the EV Tariff Optimization System

## Quick Start (Copy & Paste)

```bash
# 1. Activate virtual environment
source .venv/bin/activate

export GROQ_API_KEY='your-api-key'

# 2. Rebuild data with peak-zone utilization
python rebuild_data.py

# 3. (Optional) Run EDA for insights
python run_eda.py

# 4. (Optional) Test all fixes
python test_fixes.py

# 5. Run optimization
python run_agentic.py

python generate_plots.py

# 6. Evaluate metrics
python evaluate_metrics.py
```

---

## Step-by-Step Guide

### Step 1: Activate Virtual Environment
```bash
source .venv/bin/activate
```
**Windows**: `.venv\Scripts\activate`

### Step 2: Rebuild Data Pipeline
```bash
python rebuild_data.py
```

**What it does**:
- Loads ACN-Data (14,999 sessions from JSON)
- Loads UrbanEV (24,798 charging piles from CSVs)
- Computes **peak-zone utilization** (max across zones)
- Creates `data/processed/unified_analytical_base.csv`

**Expected output**:
```
✓ ACN: 14,999 sessions loaded
✓ UrbanEV: 24,798 piles processed
✓ Utilization range: 74.28% - 100.00%
✓ Surge timesteps: 145 (86.3%)
```

### Step 3: Run EDA (Optional)
```bash
python run_eda.py
```

**What it does**:
- Section A: ACN analysis (workplace charging patterns)
- Section B: UrbanEV analysis (urban charging patterns)
- Section C: Cross-dataset comparison
- Generates 6 visualizations in `outputs/figures/`

**Outputs**:
- `outputs/eda/key_insights.txt`
- `outputs/figures/*.png` (6 charts)

### Step 4: Test All Fixes (Optional)
```bash
python test_fixes.py
```

**What it tests**:
1. Real data loading
2. Demand agent UrbanEV-only features
3. Separate ACN/UrbanEV loaders
4. Utilization range & surge pricing
5. Temporal structure
6. Reward normalization
7. Separate EDA sections
8. Evaluation metrics module
9. Separate peak hour logging
10. Regime distribution check

**Expected**: `Passed: 10/10`

### Step 5: Run Optimization
```bash
python run_agentic.py
```

**What it does**:
- Loads real unified data (168 rows)
- Trains Demand Agent (XGBoost on UrbanEV features)
- Runs 34-step optimization loop
- Learns surge multiplier (α) from data
- Logs reward decomposition every 10 steps

**Expected output**:
```
Steps executed: 34
Mean revenue gain: 0.52%
Mean reward: 0.17  ← Not -17.32!
Final parameters:
  ε: 0.250
  α: 2.652  ← Learned!
  β: 2.500
```

**Output file**: `outputs/agentic_outcomes.csv`

### Step 6: Evaluate Metrics
```bash
python evaluate_metrics.py
```

**What it computes**:
1. Revenue Gain % (ACN data)
2. Charger Utilization (UrbanEV data)
3. Off-Peak Uplift (UrbanEV data)
4. Waiting Time Reduction (UrbanEV data)
5. Customer Response Rate (ACN + elasticity)
6. Pricing Efficiency (ACN data)

**Output file**: `outputs/evaluation_metrics.csv`

---

## Understanding the Results

### 1. Data Pipeline (`rebuild_data.py`)
- **Utilization 74-100%**: Peak-zone utilization exposes congested zones
- **86% surge timesteps**: Enables meaningful surge pricing
- **168 rows**: Weekly pattern (7 days × 24 hours)

### 2. EDA Insights (`run_eda.py`)
- **ACN peaks**: Hours 0-1 (overnight), 14-17 (afternoon)
- **UrbanEV peaks**: Different from workplace (urban commute patterns)
- **Pricing implications**: Separate strategies for different contexts

### 3. Optimization (`run_agentic.py`)
- **Reward ~0**: Normalized, stable (was -17.32)
- **α learning**: Surge multiplier increases with experience
- **All surge regime**: 100% of test timesteps trigger surge pricing

### 4. Metrics (`evaluate_metrics.py`)
- **Wait time -57.5%**: Significant improvement
- **Utilization +8%**: Better capacity usage
- **Response rate**: 0.04% session shift from elasticity

---

## Common Issues

### Issue 1: Virtual environment not activated
```bash
# You'll see: ModuleNotFoundError
# Fix: Activate virtual environment
source .venv/bin/activate
```

### Issue 2: Data file not found
```bash
# You'll see: unified_analytical_base.csv not found
# Fix: Run data pipeline first
python rebuild_data.py
```

### Issue 3: Test failures
```bash
# Run diagnostics
python test_fixes.py

# Check specific test output for guidance
```

---

## Output Files Summary

| File | What it contains |
|------|------------------|
| `data/processed/unified_analytical_base.csv` | Real ACN + UrbanEV merged data |
| `outputs/agentic_outcomes.csv` | Step-by-step optimization results |
| `outputs/evaluation_metrics.csv` | All 6 metrics evaluated |
| `outputs/eda/key_insights.txt` | Dataset-specific insights |
| `outputs/figures/*.png` | 6 visualizations |

---

## Next Steps After Running

1. **Review insights**: Check `outputs/eda/key_insights.txt`
2. **Analyze outcomes**: Open `outputs/agentic_outcomes.csv` in Excel/pandas
3. **Check metrics**: Review `outputs/evaluation_metrics.csv`
4. **Visualize**: View `outputs/figures/*.png` charts

---

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB+ recommended
- **Disk**: 500MB for data and outputs
- **Time**: ~30 seconds total for all steps

---

## Tips for Best Results

1. **Always run `rebuild_data.py` first** after any data changes
2. **Check test results** (`test_fixes.py`) before optimization
3. **Review EDA insights** to understand data patterns
4. **Monitor reward decomposition** in optimization logs
5. **Verify metrics** are computed on correct datasets

---

## Advanced: Custom Configuration

Edit `src/config.py` to customize:

```python
SystemConfig(
    baseline_tariff_per_kwh=15.0,      # Change baseline price
    pricing_bounds=(10.0, 22.0),       # Adjust price range
    theta_init=(0.25, 2.5, 2.5),       # Initial [ε, α, β]
    max_iterations=40                   # More steps for learning
)
```

---

## Support

**Stuck?** Check:
1. This guide (you're here!)
2. `README.md` for architecture details
3. `python test_fixes.py` for diagnostics
4. Error messages in terminal output

---

**Status**: ✅ All systems operational  
**Last Updated**: 2024  
**Test Coverage**: 10/10 passing
