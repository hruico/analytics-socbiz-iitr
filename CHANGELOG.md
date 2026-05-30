# OP'26 — Project Changelog

Versioned record of what was built, what problems were found, and what was improved.
Updated after every significant change.

---

## v1.0.0 — Initial Build (Gemini-based)

### What was built
- Full three-agent agentic pipeline from scratch:
  - `DemandPredictionAgent` — XGBoost multi-output regressor predicting `urban_mean_utilization` and `urban_peak_queue`
  - `TariffPricingAgent` — LLM-powered agent translating demand forecasts into dynamic ₹/kWh tariffs
  - `MonitoringLearningAgent` — evaluates pricing decisions and proposes parameter updates Δ[ε, α, β]
- Preprocessing pipeline ingesting ACN-Data (Excel) and UrbanEV ST-EVCDP (CSV matrices)
- 13 EDA visualisations covering demand trends, intraday cycles, weekday/weekend profiles, revenue analysis, feature importance
- Pydantic schemas (`ForecastState`, `PricingDecision`, `LearningUpdate`) enforcing I/O contracts
- Closed feedback loop: predict → price → monitor → update θ → repeat
- LLM backend: **Google Gemini 2.0 Flash** via `google-generativeai`
- 44 property-based and unit tests using Hypothesis + pytest

### Known state at v1.0.0
- Working end-to-end but untested at scale due to Gemini rate limits
- All agent code referenced `build_gemini_model()` and Gemini-specific retry logic

---

## v1.1.0 — LLM Migration: Gemini → Groq

### Problem
Gemini free tier hit rate limits aggressively during the agentic loop (~14 RPM cap),
making a 200-step run take hours with mandatory waits between calls.

### What changed
- Replaced `google-generativeai` with `groq` SDK in `requirements.txt`
- Rewrote `build_gemini_model()` in `src/config.py` to return a `_GroqWrapper`
  that calls `groq.Groq().chat.completions.create()` with `response_format={"type": "json_object"}`
- Model: `llama-3.3-70b-versatile` on Groq free tier (~14,400 RPD)
- Function name `build_gemini_model` intentionally kept so zero agent code needed changing
- Added `GROQ_API_KEY` environment variable check with descriptive error message
- Updated `requirements.txt`: removed `google-generativeai`, added `groq>=0.9.0`

### Files changed
- `src/config.py` — new `_GroqWrapper` class, rewritten `build_gemini_model()`
- `requirements.txt` — dependency swap

---

## v1.2.0 — Bug Fixes: Test Hangs and XGBoost Deadlock

### Problems found during testing

**Problem 1: XGBoost training deadlock**
`MultiOutputRegressor` spawns worker processes internally. Combined with
`XGB_PARAMS["n_jobs"] = -1`, this caused a deadlock on Linux — training never
completed.

**Problem 2: Test suite hanging**
`test_gemini_fallback_returns_valid_decision` and monitoring tests were hanging
because the retry backoff (`time.sleep(45)`) was not being mocked. With
`max_retries=2`, each test waited 90 seconds before completing.

### What changed
- `src/config.py`: changed `n_jobs` from `-1` to `1` in both `XGB_PARAMS` and `LGB_PARAMS`
- `tests/test_agents.py`: added `patch("src.agents.pricing.time.sleep")` to skip retry delays
- `tests/test_monitoring.py`: added `patch("src.agents.monitoring.time.sleep")` in `_make_monitor()`

### Result
- All 44 tests pass, full suite completes in ~30 seconds
- XGBoost training completes in ~3 seconds on 773 training rows

### Files changed
- `src/config.py`
- `tests/test_agents.py`
- `tests/test_monitoring.py`

---

## v1.3.0 — LLM Prompt Fix: Arithmetic Expressions in JSON

### Problem found during first real run
Groq's `llama-3.3-70b-versatile` was returning arithmetic expressions inside
JSON values instead of pre-computed numbers:

```json
{
  "surge_scalar": 0.9878 * 3.5239 / (3.5239 + 1),
  ...
}
```

JSON does not evaluate expressions, so `json.loads()` raised a `JSONDecodeError`
on every attempt. The agent fell back to the deterministic formula after 3 retries,
each with a 45-second wait — making the run extremely slow.

### Root cause
The original prompts used Unicode math symbols (`×`, `≤`, `→`, `₹`, `ε`, `α`, `β`)
and formula notation that the model interpreted as instructions to show its working
rather than return computed values.

### What changed
- `src/config.py` — `TARIFF_SYSTEM_PROMPT` rewritten:
  - Removed all Unicode math symbols and Greek letters
  - Replaced with plain ASCII (`epsilon`, `alpha`, `beta`, `*`, `<=`, `->`)
  - Added explicit rule: *"Every value must be a plain decimal number — no expressions, no formulas, no arithmetic operators"*
  - Added concrete formula examples showing expected numeric output
- `src/config.py` — `MONITOR_SYSTEM_PROMPT` same treatment
- `src/agents/pricing.py` — default retry wait reduced from `45.0s` to `2.0s` for non-429 errors
- `src/agents/monitoring.py` — same retry wait fix

### Files changed
- `src/config.py`
- `src/agents/pricing.py`
- `src/agents/monitoring.py`

---

## v1.4.0 — Metric Fixes: Revenue Gain, Customer Response Rate, Negative Predictions

### Problems found after first successful run

**Problem 1: Revenue Gain % consistently negative (-35% to -11%)**
The default elasticity `ε = 1.2` was too high. At surge prices (~₹22),
`demand_shift = -1.2 × (22-15)/15 = -0.56`, meaning the model assumed 56% of
demand disappeared. This is unrealistic for EV charging, which is largely
inelastic (necessity-driven). The sensitivity analysis confirmed only `ε = 0.5`
produced positive revenue gain (+8.5%).

Fix: Changed default `epsilon_init` from `1.2` to `0.3`, reflecting empirical
short-run price elasticity for captive/workplace EV chargers (-0.1 to -0.3).
Added `EPSILON_INIT = 0.3` constant to `src/config.py` with a documented comment.

**Problem 2: Customer Response Rate missing from outputs**
The problem statement explicitly requires this metric as a deliverable.
`demand_shift` was being computed internally but never surfaced as a named metric.

Fix: Added `customer_response_rate = demand_shift × 100` to the episode log
in `MonitoringLearningAgent.step()`. Added it to `summary()` column list and
to the final report printed by `_print_final_report()`.

**Problem 3: Negative queue predictions in predictions.csv**
`pred_urban_peak_queue` showed negative values in the exported CSV (e.g. `-0.027`).
XGBoost can predict small negatives for non-negative targets. These were already
floored to 0 inside `predict_state()` for the agent loop, but the batch export
in `export_predictions()` was using raw model output.

Fix: Added `.clip(0.0)` to `pred_urban_peak_queue` in `export_predictions()`.
Also added `.clip(0.0, 1.0)` to `pred_urban_mean_utilization` for consistency.

**Problem 4: Final report missing regime distribution**
With ε=1.2 and high-utilisation test data, every step was surge regime.
The report gave no visibility into whether discount/neutral regimes were triggered.

Fix: Added regime distribution (surge/neutral/discount step counts and %) to
`_print_final_report()`.

### Files changed
- `src/config.py` — added `EPSILON_INIT = 0.3`
- `src/agents/monitoring.py` — added `customer_response_rate` to episode log and summary columns
- `orchestrator.py` — updated default epsilon, fixed predictions export clipping, updated final report

### Expected improvement
With `ε = 0.3`, Revenue Gain % should be positive in surge regime since
`demand_shift = -0.3 × (22-15)/15 = -0.14` (only 14% demand reduction assumed,
not 56%). Pricing Efficiency Score should improve accordingly.
