# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

**OP'26 Analytics — Society of Business, IIT Roorkee**

An Agentic AI system that autonomously predicts EV charging demand, sets dynamic tariffs in real time using LLM reasoning, and continuously refines pricing parameters through a monitored feedback loop.

---

## Results

| Metric | Value |
|---|---|
| Revenue Gain % | **+4.24%** over static ₹15/kWh baseline |
| Pricing Efficiency | ₹15.00 → **₹15.64/kWh** |
| Demand Agent R² | **0.9967** |
| Demand Agent RMSE | **0.0032** |
| LLM Success Rate | **100%** (134/134 calls) |
| Absolute Revenue Gain | **₹230,751,440** across test set |

---

## Architecture

Three autonomous agents in a closed-loop pipeline:

```
UrbanEV + ACN Data
        │
        ▼
┌─────────────────────┐
│   Demand Agent      │  XGBoost (13 features) → u_pred, q_pred, congestion_prob
│   (XGBoost)         │
└─────────┬───────────┘
          │ demand signal
          ▼
┌─────────────────────┐
│   Pricing Agent     │  LLM classifies regime → deterministic price formula
│   (Groq LLM)        │  Surge: p = baseline × (1 + α × excess)
└─────────┬───────────┘  Discount: p = baseline − deficit × β
          │ price + outcome
          ▼
┌─────────────────────┐
│   Monitoring Agent  │  LLM proposes Δε, Δα, Δβ with written rationale
│   (Groq LLM)        │  Parameters updated via learning-rate-decayed gradient
└─────────────────────┘
```

**Why Agentic AI, not a plain ML model:**
The Pricing and Monitoring agents use a live LLM (Llama 3.3 70B) at every step. The LLM writes natural-language reasoning for each pricing decision — fully auditable and explainable. Plain ML optimises a fixed objective silently. This system reasons, adapts, and explains.

---

## Datasets

| Dataset | Source | Coverage |
|---|---|---|
| ACN-Data | [ev.caltech.edu](https://ev.caltech.edu/dataset.html) | 14,848 sessions, Caltech/JPL, Apr–Dec 2018 |
| UrbanEV (ST-EVCDP) | [GitHub](https://github.com/IntelligentSystemsLab/ST-EVCDP) | 247 districts, 8,640 timestamps, Shenzhen, Jun–Jul 2022 |

Place raw files in `data/raw/` before running. The UrbanEV CSVs (`occupancy.csv`, `volume.csv`, `price.csv`, `duration.csv`, `time.csv`, `information.csv`) and the ACN Excel file (`acndata_sessions.json.xlsx`) are required.

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API keys to .env
cp .env.example .env   # then edit .env with your keys
```

`.env` format:
```
GROQ_API_KEY_1=gsk_your_first_key_here
GROQ_API_KEY_2=gsk_your_second_key_here
GROQ_API_KEY_3=gsk_your_third_key_here
GROQ_API_KEY_4=gsk_your_fourth_key_here
```

The system automatically rotates to the next key when one hits its daily quota.
Get free keys at [console.groq.com](https://console.groq.com/keys).

---

## Running the Pipeline

```bash
# Step 1 — Build unified analytical base from raw data
python rebuild_data.py

# Step 2 — Exploratory Data Analysis + figures
python run_eda.py

# Step 3 — Run agentic optimization (uses LLM)
python run_agentic.py

# Step 4 — Generate agentic plots
python generate_plots.py

# Step 5 — Compute final evaluation metrics
python evaluate_metrics.py
```

Outputs land in `outputs/` — CSV results, EDA summaries, and figures.

---

## Project Structure

```
├── data/
│   └── raw/                    # Raw ACN + UrbanEV datasets (not tracked in git)
├── outputs/
│   ├── agentic_outcomes.csv    # Step-by-step optimization results
│   ├── evaluation_metrics.csv  # Final 6 evaluation metrics
│   ├── eda/                    # EDA summaries and key insights
│   └── figures/                # All visualizations
├── src/
│   ├── agents/
│   │   ├── demand.py           # XGBoost demand prediction agent
│   │   ├── pricing.py          # LLM-driven tariff pricing agent
│   │   └── monitoring.py       # LLM-driven parameter update agent
│   ├── preprocessing/
│   │   └── real_data_pipeline.py  # ACN + UrbanEV → unified analytical base
│   ├── utils/
│   │   ├── llm_provider.py     # Groq API wrapper with key rotation
│   │   ├── metrics.py          # Revenue + reward computation engine
│   │   └── convergence.py      # Convergence checker
│   ├── config.py               # SystemConfig (Pydantic)
│   ├── data_loader.py          # Dataset loading + train/test split
│   └── orchestrator.py         # Three-agent coordination loop
├── rebuild_data.py             # Data pipeline entry point
├── run_agentic.py              # Main optimization entry point
├── run_eda.py                  # EDA entry point
├── generate_plots.py           # Visualization entry point
├── evaluate_metrics.py         # Metrics evaluation entry point
├── requirements.txt
└── .env                        # API keys (not tracked in git)
```

---

## Evaluation Metrics

| Metric | Source | Value |
|---|---|---|
| Revenue Gain % | ACN session-count model | **+4.24%** |
| Charger Utilization Change | UrbanEV elasticity simulation | −1.07% |
| Off-Peak Uplift | UrbanEV (discount zones) | −54.1%* |
| Avg Wait Time Reduction | UrbanEV queue proxy | −21.5%* |
| Customer Response Rate | ACN + ε = −0.291 | 1.11% |
| Pricing Efficiency Improvement | ACN kWh model | **+4.24%** |

*Negative off-peak uplift and wait time values reflect that surge pricing in neutral/high-demand windows increases overall prices, shifting some sessions — this is the correct directional response to price increases, consistent with ε = −0.291 elasticity. Discount-zone volume uplift was insufficient to offset the effect.

---

## Dependencies

Key packages: `xgboost`, `scikit-learn`, `langchain-groq`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pydantic`, `python-dotenv`

Full list: `requirements.txt`
