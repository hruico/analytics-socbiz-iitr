# Agentic EV Tariff Optimization System

**A fully autonomous multi-agent system for optimizing EV charging tariffs using LLM-powered reasoning.**

## Overview

This system uses three autonomous agents to optimize EV charging prices through continuous learning:

- **Demand Agent** (XGBoost): Predicts utilization, queue length, and congestion probability
- **Pricing Agent** (LLM-powered): Determines optimal tariffs through contextual reasoning
- **Monitoring Agent** (LLM-powered): Evaluates outcomes and adjusts system parameters

The system runs locally using Ollama (100% free) and achieves multi-objective optimization balancing revenue, utilization distribution, and queue management.

## Quick Start

### 1. Install Ollama (Free Local LLM)

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

### 2. Pull the Model

```bash
# Mistral (recommended for reasoning)
ollama pull mistral

# Or use llama3.2 (lighter, faster)
ollama pull llama3.2:3b
```

### 3. Install Dependencies

```bash
source .venv/bin/activate
pip install langchain-ollama langchain-core
```

### 4. Run the System

```bash
python run_agentic.py
```

## System Architecture

```
Input Data (EV charging sessions)
         ↓
   Demand Agent (XGBoost)
   → Learns from historical data
   → Predicts: utilization, queue, congestion
         ↓
   Pricing Agent (LLM-powered)
   → Reasons about demand context
   → Decides optimal tariff
   → Provides natural language rationale
         ↓
   Metrics Engine
   → Computes revenue gain, reward
         ↓
   Monitoring Agent (LLM-powered)
   → Evaluates pricing outcomes
   → Proposes parameter adjustments (Δε, Δα, Δβ)
   → Explains reasoning
         ↓
   Parameter Update with Learning Rate Decay
   → θ = [ε, α, β] updated
         ↓
   Convergence Check (4 criteria)
   → Revenue variance, parameter stability,
     utilization health, queue reduction
         ↓
   Loop until convergence or max iterations
```

## What Makes It Agentic?

### Traditional Approach (Hardcoded)
```python
if utilization > 0.8:
    price = baseline + (utilization - 0.8) * factor
```

### This System (LLM-Powered)
```python
prompt = f"""
Current utilization: {util}%, queue: {queue}, time: {hour}
Analyze context and determine optimal price.
Consider surge pricing when congestion high.
"""
response = llm.invoke(prompt)  # LLM reasons autonomously
price = response['p_new']
rationale = response['rationale']
```

**Key Differences:**
- ✅ LLM analyzes full context (not just rules)
- ✅ Natural language explanations for every decision
- ✅ Adapts reasoning based on situation
- ✅ No hardcoded formulas

## Features

### Multi-Objective Optimization
Balances three objectives simultaneously:
1. **Revenue Maximization**: Increase charging revenue through dynamic pricing
2. **Utilization Distribution**: Smooth demand across peak and off-peak hours
3. **Queue Management**: Minimize customer waiting times

### Convergence-Driven Execution
System automatically detects convergence across 4 criteria:
1. Revenue variance < threshold
2. Parameter changes < threshold
3. Utilization within healthy bounds
4. Queue reduction achieved

### Robust Fallbacks
If LLM temporarily unavailable:
- System continues with deterministic fallbacks
- Fallback usage tracked and reported
- LLM success rates displayed in results

### 100% Free
- Uses Ollama (local LLM)
- No API costs
- No cloud dependencies

## Expected Output

```
======================================================================
FULLY AGENTIC EV Tariff Optimization System
======================================================================
✓ Ollama is installed and running

[1/7] Generating synthetic data...
✓ 200 hours of data created

[2/7] Loading configuration...
✓ Config loaded: baseline=₹15.0, θ=(1.5, 2.5, 2.5)

[3/7] Splitting dataset...
✓ Train: 160 rows, Test: 40 rows

[4/7] Initializing agentic system...
✓ Three agents initialized:
  • Demand Agent (XGBoost) - learns from data
  • Pricing Agent (LLM+fallback) - decides tariffs
  • Monitoring Agent (LLM+fallback) - adjusts parameters

[5/7] Training Demand Agent...
✓ Demand predictions ready

[6/7] Preparing test environment...
✓ Test set loaded

[7/7] Running optimization loop...
----------------------------------------------------------------------

======================================================================
OPTIMIZATION COMPLETE
======================================================================

Steps executed: 40
Mean revenue gain: +2.34%
Mean reward: 12.56

Final parameters:
  ε (elasticity): 1.478
  α (surge): 2.650
  β (discount): 2.500

Regime distribution:
  neutral: 28 steps (70.0%)
  surge: 8 steps (20.0%)
  discount: 4 steps (10.0%)

Agent performance:
  Pricing LLM success rate: 87.5%      ← LLM making decisions
  Monitoring LLM success rate: 92.5%   ← LLM proposing updates
  Overall LLM success rate: 90.0%      ← Fully agentic!

✓ Results saved: outputs/agentic_outcomes.csv
======================================================================
```

## Configuration

Edit settings in `run_agentic.py`:

```python
config = SystemConfig(
    llm_provider="ollama",
    llm_model="mistral:latest",         # Model: mistral (best reasoning) or llama3.2:3b (faster)
    baseline_tariff_per_kwh=15.0,      # ₹15/kWh baseline (Indian market)
    pricing_bounds=(10.0, 22.0),        # Tariff range
    theta_init=(1.5, 2.5, 2.5),        # Initial [ε, α, β] parameters
    random_seed=42,                     # For reproducibility
    max_iterations=50,                  # Safety limit
    convergence_window=20               # Convergence detection window
)
```

## Project Structure

```
.
├── run_agentic.py              # Main entry point
├── README.md                   # This file
├── CHANGELOG.md                # Project evolution log
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test configuration
│
├── src/
│   ├── agents/                 # Three autonomous agents
│   │   ├── demand.py           # XGBoost demand predictor
│   │   ├── pricing.py          # LLM-powered pricing agent
│   │   └── monitoring.py       # LLM-powered monitoring agent
│   │
│   ├── preprocessing/          # Data pipeline
│   │   ├── acn_parser.py       # ACN-Data JSON→CSV parser
│   │   ├── urbanev_parser.py   # UrbanEV CSV parser
│   │   └── dataset_fusion.py   # Dataset alignment & clustering
│   │
│   ├── utils/                  # Utilities
│   │   ├── llm_provider.py     # Ollama LLM wrapper (LangChain)
│   │   ├── metrics.py          # Metrics computation engine
│   │   └── convergence.py      # Convergence checker
│   │
│   ├── orchestrator.py         # Main optimization loop
│   ├── config.py               # System configuration (Pydantic)
│   └── data_loader.py          # Data loading utilities
│
├── tests/                      # Unit tests (29 passing)
│   ├── test_acn_parser.py      # ACN parser tests (16 tests)
│   └── test_urbanev_parser.py  # UrbanEV parser tests (13 tests)
│
├── data/
│   ├── raw/                    # Raw datasets (gitignored)
│   └── processed/              # Processed datasets
│       └── synthetic_unified.csv
│
└── outputs/                    # Results (gitignored)
    └── agentic_outcomes.csv    # Optimization results
```

## Verification

### Check LLM Success Rates
High rates (>80%) indicate LLM is making decisions:

```
Pricing LLM success rate: 87.5%      ← Good!
Monitoring LLM success rate: 92.5%   ← Good!
```

Low rates indicate fallback usage (Ollama may not be running).

### Check Rationale Messages
View decision reasoning:

```bash
tail -50 outputs/agentic_outcomes.csv | grep -v FALLBACK
```

Should see natural language explanations without `[FALLBACK]` tags.

### Test Ollama Directly
```bash
ollama run mistral "Explain surge pricing in 1 sentence"
```

## Troubleshooting

### "Ollama not found"
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Mistral (recommended for reasoning)
ollama pull mistral

# Or pull llama3.2 (lighter)
ollama pull llama3.2:3b

# Verify
ollama list
```

### "LLM success rate: 0%"
Ollama isn't running. The system will use fallbacks (still works, but not fully agentic).

### "Import error: langchain_ollama"
```bash
pip install langchain-ollama langchain-core
```

### Slow performance
- Try faster model: `ollama pull llama3.2:1b`
- Or reduce iterations: Set `max_iterations=20` in config

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected: **29 tests passing**
- 16 tests for ACN parser (100% coverage)
- 13 tests for UrbanEV parser (100% coverage)

## Performance

- **Training**: ~3 seconds (XGBoost)
- **Per optimization step**: ~2-3 seconds (includes LLM reasoning)
- **Total**: ~2 minutes for 40 steps
- **Memory**: ~4GB (Ollama model)

## Technical Details

### Data Schema
The system expects hourly records with:
- Session counts and kWh delivered (from ACN-Data)
- Utilization and queue metrics (from UrbanEV)
- Temporal features (hour, day, weekend indicator)
- Spatial features (station cluster IDs)

### Agents

**Demand Agent (XGBoost)**
- Input: 9 features (sessions, kWh, temporal, spatial)
- Output: 3 targets (utilization, queue, congestion probability)
- Architecture: MultiOutputRegressor with XGBoost base

**Pricing Agent (LLM)**
- Input: Demand predictions + context
- Process: LLM reasoning via LangChain
- Output: Optimal price + regime + rationale
- Fallback: Deterministic formula if LLM fails

**Monitoring Agent (LLM)**
- Input: Pricing outcomes + recent history
- Process: LLM analysis via LangChain
- Output: Parameter adjustments (Δε, Δα, Δβ) + reasoning
- Fallback: Rule-based updates if LLM fails

### Metrics
- **Revenue Gain**: (revenue_new - revenue_baseline) / revenue_baseline × 100%
- **Reward**: w₁×revenue_gain + w₂×utilization_improvement - w₃×queue_penalty
- **Demand Shift**: -ε × (p_new - baseline) / baseline (price elasticity)

## Cost

**$0.00** - Everything runs locally with Ollama.

## License

This is a prototype implementation for the OP26 Analytics project.

## Support

For issues or questions, check:
- `CHANGELOG.md` for project history
- Source code comments for implementation details
- Test files for usage examples

---

**Status**: ✅ Fully Operational  
**Agentic**: ✅ LLM-Powered Agents  
**Cost**: ✅ 100% Free
