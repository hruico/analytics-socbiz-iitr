# Technical Design Document: Agentic EV Tariff Optimization System Rebuild

## Overview

This document specifies the technical design for rebuilding the EV Charging Dynamic Tariff Optimization System with proper agentic architecture. The system uses three specialized agents (Demand, Pricing, Monitoring) in a continuous learning loop to autonomously optimize EV charging tariffs across multiple business objectives.

### System Goals

- **Multi-objective optimization**: Balance revenue maximization, charger utilization distribution, and queue management
- **Autonomous learning**: Agents self-improve through continuous feedback without hardcoded iteration limits
- **High reliability**: >80% agent decision quality with comprehensive validation and fallback mechanisms
- **Geographic flexibility**: Configurable for any market through geography-aware pricing bounds and baseline tariffs

### Key Architectural Principles

1. **Convergence-driven execution**: System runs until stable equilibrium is achieved across all metrics
2. **Data-grounded design**: All algorithms derived from actual unified_analytical_base.csv schema
3. **Fail-safe operation**: Multiple validation layers with graceful degradation
4. **Observable behavior**: Comprehensive structured logging at every decision point
5. **Reproducible results**: Deterministic with configurable random seed control

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        System Orchestrator                          │
│  • Convergence monitoring (50-step windows)                         │
│  • Episode management                                               │
│  • Configuration validation                                         │
│  • Checkpointing (every 100 steps)                                  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent Pipeline (per step)                      │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │   Demand     │ ──▶ │   Pricing    │ ──▶ │  Monitoring  │       │
│  │   Agent      │     │   Agent      │     │   Agent      │       │
│  │  (XGBoost)   │     │    (LLM)     │     │    (LLM)     │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
│        │                     │                     │               │
│        │                     │                     ▼               │
│        │                     │          ┌──────────────────┐       │
│        │                     │          │ Parameter Update │       │
│        │                     │          │   Arbitration    │       │
│        │                     │          └──────────────────┘       │
│        ▼                     ▼                     │               │
│   predictions.csv     agentic_outcomes.csv        │               │
│                                                    │               │
└────────────────────────────────────────────────────┼───────────────┘
                                                     │
                                                     ▼
                                          θ = [ε, α, β] update
                                            (learning rate decay)
```

### Three-Agent Architecture

**Demand_Agent (XGBoost MultiOutputRegressor)**
- **Input**: Session counts, kWh delivered, temporal features (hour, day, weekend)
- **Output**: Joint predictions for utilization and queue length
- **Training**: Chronological 80/20 split, no shuffling to preserve temporal dependencies
- **Validation**: RMSE, MAE, R² on held-out test set

**Pricing_Agent (LLM-based with LangGraph)**
- **Input**: Demand predictions, current theta parameters, temporal context
- **Output**: Structured JSON with price, regime, scalars, reasoning
- **Reasoning**: Two-step graph (analyze context → compute price → validate)
- **Fallback**: Deterministic formula if LLM fails validation

**Monitoring_Agent (LLM-based with structured output)**
- **Input**: Realized outcomes (revenue, utilization, queue), pricing decision
- **Output**: Parameter adjustments [Δε, Δα, Δβ], reward, reflection
- **Arbitration**: Agent proposal → deterministic fallback → no change
- **Learning**: Proposes directionally correct parameter updates

### Data Flow

```
unified_analytical_base.csv
         │
         ▼
    [Train/Test Split]
    (80/20 chronological)
         │
         ├─────▶ Train Set ─────▶ XGBoost Training ─────▶ Demand_Agent
         │
         └─────▶ Test Set
                    │
                    ▼
            [Optimization Loop]
                    │
                    ├─────▶ Demand Predictions (u_pred, q_pred)
                    │                │
                    │                ▼
                    ├─────▶ Pricing Decisions (p_new, regime)
                    │                │
                    │                ▼
                    ├─────▶ Metrics Computation (revenue, demand_shift, reward)
                    │                │
                    │                ▼
                    └─────▶ Parameter Updates (θ_new)
                                     │
                                     ▼
                         [Convergence Check]
                         (revenue, params, utilization, queue)
                                     │
                                     ▼
                           [Export Artifacts]
                    (agentic_outcomes.csv, predictions.csv, EDA)
```

### LLM Provider Abstraction

```python
# Configuration-driven provider selection
LLM_PROVIDER: Literal["openai", "anthropic", "ollama"]

# Provider specifications
PROVIDERS = {
    "openai": {
        "model": "gpt-4o",
        "context_window": 128_000,
        "api_env": "OPENAI_API_KEY"
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "context_window": 200_000,
        "api_env": "ANTHROPIC_API_KEY"
    },
    "ollama": {
        "model": "configurable",
        "context_window": "model-dependent",
        "api_env": None  # Local
    }
}
```

### Convergence-Based Orchestration

The system terminates when all four convergence conditions are met simultaneously for 50 consecutive steps:

1. **Revenue stability**: `var(revenue_gain_pct[last_50]) < 1.0%`
2. **Parameter stability**: `max(|Δε|, |Δα|, |Δβ|) < 0.01` over 50 steps
3. **Utilization health**: `std(utilization[last_50]) < 0.15 AND max(utilization[last_50]) < 0.80`
4. **Queue reduction**: `mean(queue[last_50]) < 0.80 * baseline_mean_queue`

Hard limit: 1000 maximum iterations to prevent infinite loops.

## Components and Interfaces

### 1. SystemConfig

Centralized configuration object managing all system parameters.


**Schema**:
```python
class SystemConfig(BaseModel):
    # LLM Provider
    llm_provider: Literal["openai", "anthropic", "ollama"]
    llm_model: str  # Provider-specific model name
    
    # Geography-Aware Pricing
    baseline_tariff_per_kwh: float = Field(gt=0.0)  # e.g., 15.0 for India
    pricing_bounds: tuple[float, float] = Field(...)  # e.g., (10.0, 22.0)
    
    # Initial Parameters
    theta_init: tuple[float, float, float] = (1.5, 2.5, 2.5)  # [ε, α, β]
    
    # Convergence Criteria
    revenue_variance_threshold: float = 1.0  # percentage points
    parameter_delta_threshold: float = 0.01
    utilization_std_threshold: float = 0.15
    max_utilization_threshold: float = 0.80
    queue_reduction_target: float = 0.20  # 20% below baseline
    convergence_window: int = 50  # consecutive steps
    max_iterations: int = 1000
    
    # Learning Rate Schedule
    learning_rate_init: float = 0.1
    learning_rate_decay: float = 0.001
    
    # Reward Weights
    reward_weights: tuple[float, float, float] = (1.0, 0.5, 0.3)  # [w1, w2, w3]
    
    # Reproducibility
    random_seed: int = 42
    train_ratio: float = 0.80
    
    # Agent Retry Policy
    max_agent_retries: int = 3
    retry_backoff_seconds: float = 2.0
    
    @field_validator("pricing_bounds")
    @classmethod
    def validate_pricing_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        if len(v) != 2:
            raise ValueError("pricing_bounds must be 2-element tuple")
        if v[0] <= 0:
            raise ValueError("pricing_bounds[0] must be positive")
        if v[1] <= v[0]:
            raise ValueError("pricing_bounds[1] must be greater than pricing_bounds[0]")
        return v
    
    @field_validator("baseline_tariff_per_kwh")
    @classmethod
    def validate_baseline_in_bounds(cls, v: float, info: ValidationInfo) -> float:
        bounds = info.data.get("pricing_bounds")
        if bounds and not (bounds[0] <= v <= bounds[1]):
            raise ValueError(f"baseline_tariff {v} must be within pricing_bounds {bounds}")
        return v
    
    @field_validator("theta_init")
    @classmethod
    def validate_theta(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        eps, alpha, beta = v
        if not (0.1 <= eps <= 5.0):
            raise ValueError(f"epsilon {eps} must be in [0.1, 5.0]")
        if not (1.0 <= alpha <= 10.0):
            raise ValueError(f"alpha {alpha} must be in [1.0, 10.0]")
        if not (1.0 <= beta <= 10.0):
            raise ValueError(f"beta {beta} must be in [1.0, 10.0]")
        return v
```

**Interface**:
- `ConfigParser.parse(path: str) -> SystemConfig`: Load and validate configuration
- `ConfigParser.serialize(config: SystemConfig, path: str) -> None`: Write configuration with pretty formatting
- `ConfigParser.validate(config: SystemConfig) -> ValidationResult`: Comprehensive validation

### 2. Demand_Agent (XGBoost Predictor)

**Purpose**: Predict hourly utilization and queue length from charging session features.

**Input Features** (from unified_analytical_base.csv):
```python
DEMAND_FEATURES = [
    "acn_sessions_count",      # Number of charging sessions
    "acn_total_kwh",           # Total energy delivered
    "acn_avg_kwh_per_session", # Average energy per session
    "hour_of_day",             # 0-23
    "day_of_week",             # 0-6
    "is_weekend",              # 0 or 1
    "is_peak_hour"             # Engineered: hour in [7,8,9,17,18,19]
]
```

**Target Variables**:
- `urban_mean_utilization` (range: [0, 1])
- `urban_peak_queue` (range: [0, ∞))

**Model Architecture**:
```python
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(
    estimator=XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=config.random_seed,
        tree_method="hist"
    )
)
```

**Training Protocol**:
1. Load unified_analytical_base.csv
2. Engineer is_peak_hour feature
3. Sort by time_step (preserve chronological order)
4. Split at 80% boundary (no shuffling)
5. Train on first 80%, validate on last 20%
6. Report RMSE, MAE, R² for both targets
7. Extract and log top 5 feature importances

**Post-processing**:
- Clip `u_pred` to [0, 1]
- Clip `q_pred` to [0, ∞)

**Interface**:
```python
class DemandAgent:
    def train(self, df: pd.DataFrame) -> TrainingMetrics
    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> EvaluationMetrics
    def get_feature_importance(self) -> pd.DataFrame
```

### 3. Pricing_Agent (LLM-based with LangGraph)

**Purpose**: Determine optimal per-kWh tariff based on demand forecasts and elasticity parameters.

**Architecture**: LangGraph state machine with three nodes:

```
┌─────────────────┐
│ analyse_state   │  LLM reasons about demand context
│                 │  (utilization, queue, time, elasticity)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ compute_price   │  LLM proposes price with justification
│                 │  (structured JSON output)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ validate        │  Python enforces business logic
│                 │  (regime consistency, bounds)
└────────┬────────┘
         │
         ▼
   PricingDecision
```

**Input State**:
```python
class ForecastState(BaseModel):
    timestamp: str
    u_pred: float  # Predicted utilization [0, 1]
    q_pred: float  # Predicted queue length
    u_actual: float  # Actual utilization (for metrics)
    q_actual: float  # Actual queue length
    kwh_delivered: float  # Energy to be delivered
    hour_of_day: int  # 0-23
    is_weekend: int  # 0 or 1
```

**Output Decision**:
```python
class PricingDecision(BaseModel):
    p_new: float  # Optimized tariff [pricing_bounds[0], pricing_bounds[1]]
    regime: Literal["surge", "neutral", "discount"]
    surge_scalar: float  # [0, 1] if surge, else 0
    discount_scalar: float  # [0, 1] if discount, else 0
    elasticity_used: float  # Current epsilon value
    rationale: str  # LLM's reasoning (1-2 sentences)
```

**Regime Rules**:
- **Surge**: `u_pred > 0.80` → price in `(baseline, upper_bound]`
- **Discount**: `u_pred < 0.30` → price in `[lower_bound, baseline)`
- **Neutral**: `u_pred ∈ [0.30, 0.80]` → price in `[lower_bound, upper_bound]`

**Validation Logic**:
1. **Format validation**: JSON schema compliance
2. **Bounds validation**: `pricing_bounds[0] ≤ p_new ≤ pricing_bounds[1]`
3. **Regime consistency**: 
   - Surge regime requires `p_new > baseline`
   - Discount regime requires `p_new < baseline`
   - Regime must match utilization thresholds

**Fallback Formula** (if LLM fails all retries):
```python
if u_pred > 0.80:
    surge_scalar = (u_pred - 0.80) / 0.20  # normalized to [0, 1]
    p_new = baseline + surge_scalar * alpha * (upper_bound - baseline) / 10.0
elif u_pred < 0.30:
    discount_scalar = (0.30 - u_pred) / 0.30
    p_new = baseline - discount_scalar * beta * (baseline - lower_bound) / 10.0
else:
    p_new = baseline + (u_pred - 0.55) * 2.0

p_new = clip(p_new, lower_bound, upper_bound)
```

**LLM Prompt Strategy**:
- **Analysis prompt**: Request strategic reasoning, no formulas
- **Pricing prompt**: Request decision with context, allow deviation from formula if justified
- **System message**: Clear business rules and constraints

**Interface**:
```python
class PricingAgent:
    def __init__(self, config: SystemConfig, theta_init: np.ndarray)
    def compute_tariff(self, state: ForecastState) -> PricingDecision
    def apply_update(self, delta: np.ndarray) -> None  # Update theta
    @property
    def theta(self) -> np.ndarray  # Current [ε, α, β]
```

### 4. Monitoring_Agent (LLM-based)

**Purpose**: Evaluate pricing outcomes and propose parameter adjustments.

**Input Context**:
```python
class MonitoringInput(BaseModel):
    step: int
    pricing_decision: PricingDecision
    state: ForecastState
    revenue_baseline: float
    revenue_new: float
    demand_shift: float  # Computed by system
    utilization_baseline: float
    utilization_new: float  # After elasticity adjustment
    queue_baseline: float
    queue_new: float
    current_theta: np.ndarray  # [ε, α, β]
    recent_history: list[StepMetrics]  # Last 3-5 steps
```

**Output Proposal**:
```python
class LearningUpdate(BaseModel):
    delta_epsilon: float  # Proposed Δε
    delta_alpha: float  # Proposed Δα
    delta_beta: float  # Proposed Δβ
    reward: float  # Scalar evaluation metric
    revenue_gain_pct: float
    charger_utilisation: float
    avg_wait_reduction: float
    pricing_efficiency: float  # Revenue per kWh
    demand_shift: float
    reflection: str  # LLM's learning insight (2-3 sentences)
```

**Validation Logic**:
1. **Format validation**: JSON schema compliance
2. **Business logic validation**:
   - If high utilization persists (3+ steps above 0.80), `delta_alpha > 0`
   - If low utilization persists (3+ steps below 0.30), `delta_beta > 0`
   - If revenue declining (3+ steps negative gain), `delta_epsilon < 0`
   - All deltas must satisfy: `|Δε| ≤ 0.05`, `|Δα| ≤ 0.10`, `|Δβ| ≤ 0.10`

**Deterministic Fallback Rules**:
```python
# Applied if agent validation fails
if revenue_gain_pct < 0 for 3 consecutive steps:
    delta_epsilon = -0.02
if u_actual > 0.80 during surge for 3 consecutive steps:
    delta_alpha = +0.10
if u_actual < 0.30 during discount for 3 consecutive steps:
    delta_beta = +0.10
else:
    no_change
```

**Parameter Update Arbitration**:
```python
def apply_parameter_update(agent_proposal: LearningUpdate, 
                           validation_passed: bool,
                           fallback_delta: np.ndarray,
                           learning_rate: float) -> np.ndarray:
    if validation_passed:
        delta = np.array([
            agent_proposal.delta_epsilon,
            agent_proposal.delta_alpha,
            agent_proposal.delta_beta
        ])
        source = "agent"
    elif fallback_delta is not None:
        delta = fallback_delta
        source = "fallback"
    else:
        delta = np.zeros(3)
        source = "no_change"
    
    # Apply learning rate decay
    eta = learning_rate / (1 + config.learning_rate_decay * step)
    delta_scaled = eta * delta
    
    logger.info(f"Parameter update: source={source}, delta={delta_scaled}, eta={eta}")
    return delta_scaled
```

**Interface**:
```python
class MonitoringAgent:
    def __init__(self, config: SystemConfig)
    def evaluate_and_propose(self, context: MonitoringInput) -> LearningUpdate
    def compute_fallback_update(self, history: list[StepMetrics]) -> np.ndarray
```

### 5. MetricsComputation Engine

**Purpose**: Deterministically compute all quantitative metrics from outcomes.

**Core Formulas**:

```python
# Demand shift (price elasticity effect)
demand_shift = -epsilon * (p_new - baseline) / baseline

# Revenue calculations
revenue_baseline = baseline * kwh_delivered
adjusted_demand_factor = max(0.05, 1.0 + demand_shift)  # Floor at 5%
revenue_new = p_new * kwh_delivered * adjusted_demand_factor
revenue_gain_pct = (revenue_new - revenue_baseline) / revenue_baseline * 100

# Utilization adjustment
utilization_new = clip(u_actual + demand_shift * 0.1, 0.0, 1.0)
utilization_improvement = (utilization_new - u_actual) * 100

# Queue impact
avg_wait_reduction = -demand_shift * q_actual

# Efficiency
pricing_efficiency = revenue_new / kwh_delivered

# Customer response
customer_response_rate = demand_shift * 100

# Queue penalty
queue_penalty = max(0, q_actual - baseline_mean_queue) * 100

# Reward function
w1, w2, w3 = config.reward_weights
reward = w1 * revenue_gain_pct + w2 * utilization_improvement - w3 * queue_penalty
```

**Interface**:
```python
class MetricsEngine:
    def compute_step_metrics(
        self,
        decision: PricingDecision,
        state: ForecastState,
        baseline_tariff: float,
        baseline_mean_queue: float
    ) -> StepMetrics
```

### 6. ConvergenceChecker

**Purpose**: Monitor convergence across multiple objectives.

**State Tracking**:
```python
class ConvergenceState:
    revenue_history: deque[float]  # Last 50 revenue_gain_pct values
    theta_history: deque[np.ndarray]  # Last 50 theta vectors
    utilization_history: deque[float]  # Last 50 utilization values
    queue_history: deque[float]  # Last 50 queue values
    consecutive_convergence_steps: int
    convergence_met: bool
```

**Convergence Checks**:
```python
def check_convergence(state: ConvergenceState, config: SystemConfig) -> ConvergenceResult:
    if len(state.revenue_history) < config.convergence_window:
        return ConvergenceResult(met=False, reason="insufficient_data")
    
    # Check 1: Revenue stability
    revenue_var = np.var(state.revenue_history)
    revenue_stable = revenue_var < config.revenue_variance_threshold
    
    # Check 2: Parameter stability
    theta_deltas = [np.abs(state.theta_history[i] - state.theta_history[i-1]) 
                    for i in range(1, len(state.theta_history))]
    max_theta_delta = np.max(theta_deltas)
    params_stable = max_theta_delta < config.parameter_delta_threshold
    
    # Check 3: Utilization health
    util_std = np.std(state.utilization_history)
    util_max = np.max(state.utilization_history)
    util_healthy = (util_std < config.utilization_std_threshold and 
                    util_max < config.max_utilization_threshold)
    
    # Check 4: Queue reduction
    queue_mean = np.mean(state.queue_history)
    baseline_mean_queue = compute_baseline_queue_mean()
    queue_reduced = queue_mean < (1 - config.queue_reduction_target) * baseline_mean_queue
    
    all_met = revenue_stable and params_stable and util_healthy and queue_reduced
    
    if all_met:
        state.consecutive_convergence_steps += 1
    else:
        state.consecutive_convergence_steps = 0
    
    converged = state.consecutive_convergence_steps >= config.convergence_window
    
    return ConvergenceResult(
        met=converged,
        revenue_stable=revenue_stable,
        params_stable=params_stable,
        util_healthy=util_healthy,
        queue_reduced=queue_reduced,
        consecutive_steps=state.consecutive_convergence_steps
    )
```

**Interface**:
```python
class ConvergenceChecker:
    def update(self, metrics: StepMetrics, theta: np.ndarray) -> None
    def check(self) -> ConvergenceResult
    def get_status_summary(self) -> dict
```

### 7. LLMProviderWrapper

**Purpose**: Abstract LLM provider differences with unified retry logic.


**Provider Factory**:
```python
class LLMProviderWrapper:
    @staticmethod
    def create(config: SystemConfig) -> BaseChatModel:
        if config.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set")
            return ChatOpenAI(
                model=config.llm_model or "gpt-4o",
                api_key=api_key,
                temperature=0.2,
                timeout=30.0
            )
        elif config.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not set")
            return ChatAnthropic(
                model=config.llm_model or "claude-3-5-sonnet-20241022",
                api_key=api_key,
                temperature=0.2,
                timeout=30.0
            )
        elif config.llm_provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.llm_model or "llama3.1",
                temperature=0.2,
                timeout=60.0
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
```

**Retry Logic**:
```python
def invoke_with_retry(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    max_retries: int = 3,
    backoff_seconds: float = 2.0
) -> str:
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            error_type = classify_error(exc)  # transient vs permanent
            logger.warning(
                f"LLM invocation attempt {attempt + 1}/{max_retries} failed: "
                f"type={error_type}, error={exc}"
            )
            if error_type == "permanent" or attempt == max_retries - 1:
                raise
            time.sleep(backoff_seconds * (2 ** attempt))  # Exponential backoff
    raise RuntimeError("Max retries exceeded")

def classify_error(exc: Exception) -> Literal["transient", "permanent"]:
    # Rate limits, timeouts → transient
    # Invalid API key, model not found → permanent
    error_str = str(exc).lower()
    if any(x in error_str for x in ["rate limit", "timeout", "503", "429"]):
        return "transient"
    if any(x in error_str for x in ["invalid", "unauthorized", "404"]):
        return "permanent"
    return "transient"  # Default to retryable
```

### 8. EDA Generator

**Purpose**: Produce comprehensive exploratory data analysis visualizations and statistics.

**Deliverables**:

1. **Temporal Demand Pattern Analysis**
   - Hourly utilization heatmap (day of week vs hour)
   - Daily utilization time series with trend line
   - Peak hour identification histogram

2. **Utilization Distribution Analysis**
   - Distribution histogram with quartiles
   - Box plot by day of week
   - Surge/neutral/discount regime frequency

3. **Feature Importance Ranking**
   - Bar chart from XGBoost training
   - Top 5 features with SHAP values (if available)

4. **Correlation Analysis**
   - Heatmap: revenue vs utilization vs queue
   - Scatter: utilization vs queue length
   - Scatter: sessions count vs revenue

5. **Queue Analysis**
   - Queue length distribution
   - Queue vs time-of-day scatter
   - Congestion event identification (queue > 75th percentile)

**Output Format**:
- All plots saved as PNG (300 DPI) in `outputs/eda/`
- Summary statistics exported to `outputs/eda/eda_summary.json`
- Filename convention: `{metric}_{analysis_type}.png`

**Interface**:
```python
class EDAGenerator:
    def __init__(self, df: pd.DataFrame, demand_agent: DemandAgent)
    def generate_temporal_analysis(self) -> None
    def generate_utilization_distribution(self) -> None
    def generate_feature_importance(self) -> None
    def generate_correlation_analysis(self) -> None
    def generate_queue_analysis(self) -> None
    def generate_summary_statistics(self) -> dict
    def run_full_analysis(self) -> None
```

## Data Models

### Core Data Structures

**unified_analytical_base.csv Schema**:
```
hourly_timestamp          : str       # ISO 8601 format
acn_sessions_count        : int       # Number of sessions (ACN-Data)
acn_total_kwh             : float     # Total energy delivered (ACN-Data)
acn_base_revenue          : float     # Revenue at baseline tariff (ACN-Data)
acn_avg_kwh_per_session   : float     # Average energy per session (ACN-Data)
acn_revenue_per_session   : float     # Average revenue per session (ACN-Data)
acn_energy_cost_per_kwh   : float     # Baseline cost (ACN-Data)
time_step                 : int       # Sequential step for chronological ordering
urban_mean_utilization    : float     # Utilization [0,1] (UrbanEV)
urban_peak_queue          : float     # Queue length (UrbanEV)
urban_total_volume        : float     # Total charging volume (UrbanEV)
hour_of_day               : int       # 0-23
day_of_week               : int       # 0-6 (Monday=0)
is_weekend                : int       # 0 or 1
```

**Engineered Features**:
```python
is_peak_hour = hour_of_day in [7, 8, 9, 17, 18, 19]
revenue_per_kwh = acn_base_revenue / acn_total_kwh  # when acn_total_kwh > 0
```

**StepMetrics Output**:
```python
class StepMetrics(BaseModel):
    step: int
    timestamp: str
    regime: str  # surge | neutral | discount
    u_pred: float
    q_pred: float
    u_actual: float
    q_actual: float
    p_new: float
    kwh_delivered: float
    revenue_baseline: float
    revenue_new: float
    revenue_gain_pct: float
    demand_shift: float
    charger_utilisation: float
    avg_wait_reduction: float
    pricing_efficiency: float
    customer_response_rate: float
    reward: float
    epsilon: float
    alpha: float
    beta: float
    agent_success: bool  # True if LLM validation passed
    fallback_used: bool
```

**agentic_outcomes.csv Columns**:
```
step, timestamp, regime, u_pred, q_pred, u_actual, q_actual, 
p_new, kwh_delivered, revenue_baseline, revenue_new, revenue_gain_pct,
demand_shift, charger_utilisation, avg_wait_reduction, pricing_efficiency,
customer_response_rate, reward, epsilon, alpha, beta,
agent_success, fallback_used
```

**predictions.csv Columns**:
```
timestamp, u_pred, q_pred, u_actual, q_actual, 
u_pred_error, q_pred_error, hour_of_day, is_weekend
```

**benchmark_comparison.csv Schema**:
```python
class BenchmarkResult(BaseModel):
    baseline_name: str  # e.g., "fixed_baseline", "time_of_day", "deterministic_formula"
    mean_revenue_gain_pct: float
    std_revenue_gain_pct: float
    mean_utilization: float
    std_utilization: float
    mean_queue_length: float
    std_queue_length: float
    mean_reward: float
    t_statistic: float  # Paired t-test vs agentic system
    p_value: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Before defining the properties, I need to analyze each acceptance criterion for testability.


### Property 1: Convergence Termination Correctness

*For any* sequence of optimization metrics (revenue, theta, utilization, queue) over at least 50 steps, the system should terminate when and only when all four convergence conditions are met simultaneously for 50 consecutive steps within the maximum iteration limit.

**Validates: Requirements 1.1**

### Property 2: Revenue Stability Computation

*For any* sequence of revenue_gain_pct values of length N ≥ 50, the variance calculation should correctly identify stability when var(last_50_values) < 1.0%.

**Validates: Requirements 1.2**

### Property 3: Parameter Stability Tracking

*For any* sequence of theta vectors over 50 steps, the maximum delta computation should correctly identify all parameter changes and determine if max(|Δε|, |Δα|, |Δβ|) < 0.01.

**Validates: Requirements 1.3**

### Property 4: Utilization Health Check

*For any* sequence of utilization values over 50 steps, the system should correctly compute std(utilization) and max(utilization) and determine if the dual condition (std < 0.15 AND max < 0.80) is met.

**Validates: Requirements 1.4**

### Property 5: Queue Reduction Verification

*For any* sequence of queue length values over 50 steps and any baseline queue mean, the system should correctly determine if mean(queue) < 0.80 * baseline_mean_queue.

**Validates: Requirements 1.5**

### Property 6: LLM Provider Configuration Mapping

*For any* valid LLM provider configuration (openai | anthropic | ollama), changing the configuration should instantiate the corresponding provider without code modification.

**Validates: Requirements 2.6**

### Property 7: Agent Decision Validation

*For any* agent decision (pricing or monitoring), validation must occur before acceptance, checking both format compliance and business logic correctness.

**Validates: Requirements 3.1**

### Property 8: Pricing Output Schema Compliance

*For any* Pricing_Agent output, it should conform to the PricingDecision schema with all required fields (p_new, regime, surge_scalar, discount_scalar, elasticity_used, rationale).

**Validates: Requirements 3.2**

### Property 9: Pricing Decision Bounds and Regime Consistency

*For any* pricing decision and utilization value, validation should pass if and only if: (1) p_new is within pricing_bounds, and (2) regime matches demand conditions (surge when u > 0.80, discount when u < 0.30, neutral when u ∈ [0.30, 0.80]), and (3) if surge then p_new > baseline, if discount then p_new < baseline.

**Validates: Requirements 3.3, 15.5**

### Property 10: Monitoring Parameter Update Validation

*For any* monitoring agent proposal and recent history, validation should pass if and only if parameter adjustments are directionally correct: increase alpha when high utilization persists, increase beta when low utilization persists, adjust epsilon based on revenue direction.

**Validates: Requirements 3.4**

### Property 11: Agent Retry with Exponential Backoff

*For any* agent invocation that fails with a transient error, the system should retry up to max_retries times with exponential backoff (wait_time = backoff_seconds * 2^attempt).

**Validates: Requirements 3.6, 6.5**

### Property 12: Fallback on Validation Failure

*For any* agent decision that fails all retries or fails business logic validation, the system should log detailed error information and invoke deterministic fallback logic.

**Validates: Requirements 3.7**

### Property 13: Parameter Update Precedence

*For any* monitoring outcome with agent proposal and fallback delta, the system should apply updates according to precedence: (1) agent proposal if validation passes, (2) deterministic fallback if agent fails, (3) no change if neither applies.

**Validates: Requirements 4.1**

### Property 14: Valid Agent Proposal Application

*For any* valid Monitoring_Agent parameter proposal [Δε, Δα, Δβ] at step t, the system should apply it with learning rate decay: eta = eta_0 / (1 + decay * t), resulting in theta_new = theta_old + eta * delta.

**Validates: Requirements 4.2, 4.8**

### Property 15: Fallback on Agent Validation Failure

*For any* Monitoring_Agent proposal that fails validation, the system should compute deterministic fallback adjustment and log the conflict with old values, proposed delta, and fallback delta.

**Validates: Requirements 4.3**

### Property 16: Parameter Update Logging Structure

*For any* parameter update (agent or fallback), the system should log: source (agent | fallback | no_change), old_theta, proposed_delta, final_applied_delta, and learning_rate_used.

**Validates: Requirements 4.7, 14.3**

### Property 17: Error Classification

*For any* LLM API error, the system should correctly classify it as transient (rate limit, timeout, 503, 429) or permanent (invalid key, unauthorized, 404) based on error content.

**Validates: Requirements 6.4**

### Property 18: Direct Feature Usage

*For any* row in unified_analytical_base.csv, urban_mean_utilization and urban_peak_queue should be used directly without transformation in the feature matrix.

**Validates: Requirements 7.2, 7.3**

### Property 19: Peak Hour Engineering

*For any* hour_of_day value in [0, 23], is_peak_hour should equal True if and only if hour_of_day ∈ {7, 8, 9, 17, 18, 19}.

**Validates: Requirements 7.4**

### Property 20: Revenue Per kWh Calculation

*For any* row where acn_total_kwh > 0, revenue_per_kwh should equal acn_base_revenue / acn_total_kwh.

**Validates: Requirements 7.5**

### Property 21: Missing Value Handling Strategy

*For any* dataframe with missing values, the system should forward-fill utilization and queue columns, and drop rows with missing revenue data.

**Validates: Requirements 7.6**

### Property 22: Chronological Train-Test Split

*For any* sorted dataset by time_step, the train-test split should occur at exactly the 80% boundary with train set containing first 80% of rows and test set containing last 20%, with no shuffling.

**Validates: Requirements 7.7, 8.2**

### Property 23: Prediction Clipping

*For any* XGBoost predictions, u_pred should be clipped to [0, 1] and q_pred should be clipped to [0, ∞).

**Validates: Requirements 8.6**

### Property 24: Revenue Gain Percentage Formula

*For any* revenue_new, revenue_baseline values where revenue_baseline > 0, revenue_gain_pct should equal (revenue_new - revenue_baseline) / revenue_baseline * 100.

**Validates: Requirements 9.1**

### Property 25: Demand Shift Elasticity Formula

*For any* epsilon, price_new, price_baseline where price_baseline > 0, demand_shift should equal -epsilon * (price_new - price_baseline) / price_baseline.

**Validates: Requirements 9.2**

### Property 26: Utilization Elasticity Adjustment

*For any* u_actual and demand_shift, charger_utilization should equal clip(u_actual + demand_shift * 0.1, 0.0, 1.0).

**Validates: Requirements 9.3**

### Property 27: Pricing Efficiency Calculation

*For any* revenue_new and kwh_delivered where kwh_delivered > 0, pricing_efficiency should equal revenue_new / kwh_delivered.

**Validates: Requirements 9.4**

### Property 28: Customer Response Rate

*For any* demand_shift, customer_response_rate should equal demand_shift * 100.

**Validates: Requirements 9.5**

### Property 29: Average Wait Reduction

*For any* demand_shift and queue_actual, avg_wait_reduction should equal -demand_shift * queue_actual.

**Validates: Requirements 9.6**

### Property 30: Reward Function Computation

*For any* revenue_gain_pct, utilization_improvement, queue_penalty, and weights [w1, w2, w3], reward should equal w1 * revenue_gain_pct + w2 * utilization_improvement - w3 * queue_penalty.

**Validates: Requirements 9.7**

### Property 31: Revenue Uplift Aggregation

*For any* sequence of revenue_gain_pct values across all optimization steps, revenue uplift should equal mean(revenue_gain_pct).

**Validates: Requirements 10.3**

### Property 32: Off-Peak Utilization Uplift

*For any* sequence of optimization steps with utilization and hour_of_day, off-peak uplift should be computed as percentage increase in mean utilization during hours not in {7, 8, 9, 17, 18, 19}.

**Validates: Requirements 10.4**

### Property 33: Congestion Reduction Metric

*For any* sequence of queue lengths and baseline mean queue, congestion reduction should equal (baseline_mean_queue - mean(queue_lengths)) / baseline_mean_queue * 100.

**Validates: Requirements 10.5**

### Property 34: Output File Versioning

*For any* output file generated by the system, the filename should include a timestamp and configuration hash for versioning.

**Validates: Requirements 11.5**

### Property 35: Configuration Validation Rules

*For any* SystemConfig object, validation should pass if and only if: epsilon ∈ [0.1, 5.0], alpha ∈ [1.0, 10.0], beta ∈ [1.0, 10.0], pricing_bounds[0] > 0, pricing_bounds[1] > pricing_bounds[0], and baseline_tariff_per_kwh ∈ [pricing_bounds[0], pricing_bounds[1]].

**Validates: Requirements 12.7, 13.5, 13.6, 13.7**

### Property 36: Invalid Configuration Rejection

*For any* invalid SystemConfig, the validation should fail with a clear error message identifying the specific constraint violation.

**Validates: Requirements 12.8, 13.2**

### Property 37: Configuration Round-Trip Preservation

*For any* valid SystemConfig object, parsing it to JSON, then pretty-printing to string, then parsing back should produce an equivalent object.

**Validates: Requirements 13.1, 13.4**

### Property 38: Optimization Step Logging

*For any* optimization step, the system should log: step_number, timestamp, regime, price, revenue_gain_pct, reward, demand_shift.

**Validates: Requirements 1.7, 14.1**

### Property 39: LLM API Call Logging

*For any* LLM API invocation, the system should log: provider, model, tokens_used, latency_ms, success_status.

**Validates: Requirements 14.2**

### Property 40: Convergence Check Logging

*For any* convergence check, the system should log: metric_name, current_value, threshold, pass_fail_status.

**Validates: Requirements 14.4**

### Property 41: Graceful Agent Failure Handling

*For any* agent failure during an optimization step, the system should complete the step using deterministic fallback and continue to the next step without terminating.

**Validates: Requirements 15.1**

### Property 42: Theta Checkpointing Frequency

*For any* optimization run, theta parameters should be checkpointed at steps that are multiples of 100 (step % 100 == 0).

**Validates: Requirements 15.4**

### Property 43: Graceful Degradation Event Logging

*For any* graceful degradation event, the system should log: failure_type, fallback_used, state_preserved.

**Validates: Requirements 15.6**

### Property 44: EDA Visualization Output Format

*For any* EDA visualization generated, it should be saved as a PNG file with 300 DPI resolution in the outputs/eda/ directory.

**Validates: Requirements 17.6**

## Error Handling

### Error Classification Hierarchy

```
SystemError
├── ConfigurationError
│   ├── InvalidTheta
│   ├── InvalidPricingBounds
│   └── InvalidConvergenceThresholds
├── DataError
│   ├── MissingColumns
│   ├── InvalidSchema
│   └── InsufficientData
├── AgentError
│   ├── FormatValidationError
│   ├── BusinessLogicValidationError
│   └── LLMAPIError
│       ├── TransientError (retryable)
│       └── PermanentError (not retryable)
└── ConvergenceError
    ├── MaxIterationsExceeded
    └── UnstableMetrics
```

### Error Handling Strategies


**1. Configuration Errors**
- **Detection**: At startup during config validation
- **Response**: Reject config with specific error message, terminate before optimization
- **Logging**: ERROR level with constraint violation details

**2. Data Loading Errors**
- **Missing columns**: Check schema immediately after load, raise DataError with missing column list
- **Invalid values**: Log WARNING and apply imputation strategy (forward-fill or drop)
- **Insufficient data**: If test set < 10 rows, raise InsufficientData error

**3. Agent Format Validation Errors**
- **JSON parsing failure**: 
  - Log raw LLM response at DEBUG level
  - Retry up to max_retries with exponential backoff
  - After all retries fail, log ERROR and invoke deterministic fallback
- **Schema validation failure**:
  - Log validation errors with attempted structure
  - Retry with modified prompt emphasizing schema
  - Fallback after max_retries

**4. Agent Business Logic Validation Errors**
- **Pricing regime mismatch** (surge price but neutral regime):
  - Log WARNING with detected inconsistency
  - Do NOT retry (business logic error suggests reasoning failure)
  - Invoke deterministic fallback immediately
- **Parameter update direction error** (negative alpha when high utilization):
  - Log WARNING with directional conflict
  - Compute fallback delta using deterministic rules
  - Log precedence: "fallback overrides agent due to business logic violation"

**5. LLM API Errors**
- **Transient** (rate limit, timeout, 503, 429):
  - Log WARNING with HTTP status and error message
  - Wait with exponential backoff: 2^attempt * backoff_seconds
  - Retry up to max_retries
  - If quota exceeded: pause optimization, wait for quota reset, resume
- **Permanent** (invalid key, 401, 404):
  - Log ERROR with full context
  - Do not retry
  - Terminate optimization (cannot proceed without valid LLM access)

**6. Convergence Errors**
- **Max iterations exceeded**:
  - Log WARNING with convergence status summary
  - Export partial results with "convergence_not_met" flag in metadata
  - Generate evaluation report noting incomplete convergence
- **Unstable metrics** (oscillating without converging):
  - Log INFO with metric history
  - Continue until max_iterations (not an error, just monitoring)

### Validation Pipeline

```python
def validate_and_apply_decision(
    decision: dict,
    state: ForecastState,
    config: SystemConfig
) -> tuple[PricingDecision, bool]:
    """
    Three-stage validation with detailed error logging.
    Returns: (decision_or_fallback, agent_success_flag)
    """
    # Stage 1: Format validation (schema)
    try:
        parsed = PricingDecision(**decision)
    except ValidationError as exc:
        logger.error(f"Format validation failed: {exc}")
        logger.debug(f"Raw decision: {decision}")
        return deterministic_pricing_fallback(state, config), False
    
    # Stage 2: Bounds validation
    if not (config.pricing_bounds[0] <= parsed.p_new <= config.pricing_bounds[1]):
        logger.warning(
            f"Price {parsed.p_new} outside bounds {config.pricing_bounds}"
        )
        return deterministic_pricing_fallback(state, config), False
    
    # Stage 3: Business logic validation
    regime_valid = validate_regime_consistency(parsed, state)
    if not regime_valid:
        logger.warning(
            f"Regime inconsistency: regime={parsed.regime}, "
            f"u_pred={state.u_pred}, p_new={parsed.p_new}"
        )
        return deterministic_pricing_fallback(state, config), False
    
    # All validation passed
    logger.info(f"Agent decision validated: {parsed.regime} @ Rs{parsed.p_new:.2f}/kWh")
    return parsed, True

def validate_regime_consistency(decision: PricingDecision, state: ForecastState) -> bool:
    """Business logic: regime must match utilization thresholds and price direction."""
    if decision.regime == "surge":
        return (state.u_pred > 0.80 and decision.p_new > P_BASE)
    elif decision.regime == "discount":
        return (state.u_pred < 0.30 and decision.p_new < P_BASE)
    elif decision.regime == "neutral":
        return (0.30 <= state.u_pred <= 0.80)
    return False
```

## Testing Strategy

### Dual Testing Approach

The system requires both **unit testing** and **property-based testing** for comprehensive coverage:

**Unit Tests**: 
- Verify specific examples and integration points
- Test edge cases and error conditions
- Validate configuration loading and schema compliance
- Check logging output format

**Property Tests**:
- Verify universal properties across all inputs
- Comprehensive input coverage through randomization
- Validate correctness of mathematical formulas
- Test round-trip properties (parse/serialize, encode/decode)

### Property-Based Testing Configuration

**Library Selection**: 
- Python: `hypothesis` (recommended for rich strategy composition)
- Minimum 100 iterations per test (configurable via `@settings(max_examples=100)`)

**Test Organization**:
```python
# tests/test_properties.py
from hypothesis import given, strategies as st, settings
import pytest

@settings(max_examples=100)
@given(
    epsilon=st.floats(min_value=0.1, max_value=5.0),
    price_new=st.floats(min_value=10.0, max_value=22.0),
    price_baseline=st.floats(min_value=10.0, max_value=22.0).filter(lambda x: x > 0)
)
def test_demand_shift_formula(epsilon, price_new, price_baseline):
    """
    Feature: agentic-ev-tariff-optimization-rebuild
    Property 25: For any epsilon, price_new, price_baseline where price_baseline > 0,
    demand_shift should equal -epsilon * (price_new - price_baseline) / price_baseline
    """
    expected = -epsilon * (price_new - price_baseline) / price_baseline
    actual = compute_demand_shift(epsilon, price_new, price_baseline)
    assert abs(actual - expected) < 1e-9, \
        f"demand_shift mismatch: expected={expected}, actual={actual}"
```

**Property Test Tagging**:
Every property test must include a docstring with:
```
Feature: agentic-ev-tariff-optimization-rebuild
Property {number}: {property_text_from_design_doc}
```

### Unit Testing Strategy

**Unit tests should focus on**:
1. **Configuration examples**: Specific valid/invalid configs
2. **Edge cases**: Boundary conditions (utilization = 0.80, queue = 0)
3. **Integration points**: Agent pipeline composition
4. **Error paths**: Specific error types trigger correct handlers
5. **Output validation**: CSV files have correct columns and format

**Unit test balance**: 
- Avoid writing many unit tests for formula validation (property tests handle this)
- Focus unit tests on concrete scenarios that demonstrate correct behavior
- Use unit tests for deterministic sequences that reveal bugs

**Example unit tests**:
```python
def test_config_validation_rejects_negative_bounds():
    """Specific example: negative pricing bound should fail validation"""
    config = SystemConfig(
        pricing_bounds=(-5.0, 10.0),  # Invalid
        baseline_tariff_per_kwh=5.0
    )
    with pytest.raises(ValidationError, match="must be positive"):
        config.validate()

def test_surge_regime_assigned_above_threshold():
    """Specific example: utilization=0.85 should trigger surge regime"""
    state = ForecastState(u_pred=0.85, ...)
    decision = pricing_agent.compute_tariff(state)
    assert decision.regime == "surge"
    assert decision.p_new > 15.0

def test_agent_failure_triggers_fallback_and_continues():
    """Edge case: agent failure should not stop optimization"""
    # Mock LLM to return invalid JSON
    with patch_llm_to_fail():
        metrics = run_single_step(orchestrator, step=10)
        assert metrics.fallback_used == True
        # Step 11 should still execute
        metrics_next = run_single_step(orchestrator, step=11)
        assert metrics_next is not None
```

### Coverage Targets

- **Line coverage**: >85% (measured by pytest-cov)
- **Property coverage**: All 44 correctness properties implemented as tests
- **Edge case coverage**: Documented boundary conditions tested in unit tests
- **Integration coverage**: End-to-end pipeline tested with synthetic data

### Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run property tests only
pytest tests/test_properties.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific property test
pytest tests/test_properties.py::test_demand_shift_formula -v
```

### Test Data Strategy

**Unit tests**: 
- Use fixtures with small, hand-crafted datasets (5-10 rows)
- Focus on edge cases and specific scenarios

**Property tests**:
- Use hypothesis strategies to generate random inputs
- Define valid ranges for each parameter
- Use `st.data()` for complex dependent generation

**Integration tests**:
- Use a synthetic unified_analytical_base.csv (100 rows, known patterns)
- Validate full pipeline produces expected output structure

## Appendix A: Regime Threshold Justification

The system uses unified regime thresholds across all pricing decisions:

- **Surge threshold: 0.80** (utilization > 80%)
  - Rationale: >80% utilization indicates genuine congestion
  - Above this level, queue length typically increases non-linearly
  - Industry standard: charging stations operate optimally at 70-85% utilization

- **Discount threshold: 0.30** (utilization < 30%)
  - Rationale: <30% utilization indicates excess capacity
  - Opportunity to attract price-sensitive demand without impacting service
  - Off-peak hours typically see 15-40% utilization in urban EV charging

- **Neutral regime: [0.30, 0.80]**
  - Balanced demand, no strong signal for intervention
  - Price adjustments should be modest to avoid demand volatility

## Appendix B: Formula Derivations

### Demand Shift (Price Elasticity)

```
demand_shift = -ε * (P_new - P_baseline) / P_baseline
```

**Derivation**: Price elasticity of demand is defined as:
```
ε = (ΔQ / Q) / (ΔP / P)
```

Rearranging for quantity change:
```
ΔQ / Q = -ε * (ΔP / P)  [negative because demand decreases when price increases]
```

Where:
- `ΔQ / Q` is the proportional demand change (demand_shift)
- `ΔP / P` is the proportional price change `(P_new - P_baseline) / P_baseline`

### Revenue Calculation with Elasticity

```
adjusted_demand_factor = max(0.05, 1 + demand_shift)
revenue_new = P_new * kWh_delivered * adjusted_demand_factor
```

**Rationale**: 
- Base demand: `kWh_delivered` at baseline price
- Elasticity adjustment: demand changes by `demand_shift` proportion
- Floor at 5%: Even with extreme price increases, some inelastic demand remains (emergency charging)
- Revenue: price × adjusted quantity

### Reward Function

```
reward = w1 * revenue_gain_pct + w2 * utilization_improvement - w3 * queue_penalty
```

**Component justification**:
1. **Revenue gain** (w1 = 1.0): Primary business objective
2. **Utilization improvement** (w2 = 0.5): Operational efficiency, secondary priority
3. **Queue penalty** (w3 = 0.3): Customer experience, tertiary priority

**Weight rationale**:
- Revenue weighted highest to maintain business viability
- Utilization improvement valued at 50% of revenue to encourage efficient capacity usage
- Queue penalty at 30% to avoid excessive wait times while not overriding revenue

## Appendix C: Dataset Schema Mapping

### ACN-Data (Caltech/JPL) Contribution

Source columns from ACN-Data sessions:
```
connectionTime, disconnectTime, kWhDelivered, sessionID
```

Aggregated to hourly in unified_analytical_base.csv:
```
acn_sessions_count      ← COUNT(sessionID) per hour
acn_total_kwh           ← SUM(kWhDelivered) per hour
acn_avg_kwh_per_session ← AVG(kWhDelivered) per hour
acn_base_revenue        ← SUM(kWhDelivered * baseline_tariff) per hour
acn_revenue_per_session ← AVG(kWhDelivered * baseline_tariff) per hour
acn_energy_cost_per_kwh ← baseline_tariff (constant)
```

### UrbanEV (Shenzhen) Contribution

Source columns from UrbanEV dataset:
```
start_time, end_time, charging_volume, waiting_time, station_utilization
```

Aggregated to hourly in unified_analytical_base.csv:
```
urban_mean_utilization ← AVG(station_utilization) per hour [0, 1]
urban_peak_queue       ← MAX(waiting_time / avg_session_duration) per hour
urban_total_volume     ← SUM(charging_volume) per hour
```

### Merge Strategy

- **Temporal alignment**: Both datasets normalized to hourly UTC timestamps
- **Missing value strategy**: Forward-fill for utilization/queue, drop for revenue
- **Treatment**: Single unified network (not separate markets)

## Appendix D: Configuration Example

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4o",
  "baseline_tariff_per_kwh": 15.0,
  "pricing_bounds": [10.0, 22.0],
  "theta_init": [1.5, 2.5, 2.5],
  "revenue_variance_threshold": 1.0,
  "parameter_delta_threshold": 0.01,
  "utilization_std_threshold": 0.15,
  "max_utilization_threshold": 0.80,
  "queue_reduction_target": 0.20,
  "convergence_window": 50,
  "max_iterations": 1000,
  "learning_rate_init": 0.1,
  "learning_rate_decay": 0.001,
  "reward_weights": [1.0, 0.5, 0.3],
  "random_seed": 42,
  "train_ratio": 0.80,
  "max_agent_retries": 3,
  "retry_backoff_seconds": 2.0
}
```

**For different geographies**, adjust:
- `baseline_tariff_per_kwh`: Local market baseline (e.g., $0.30/kWh for US, ₹15/kWh for India)
- `pricing_bounds`: Local regulatory limits and competitive positioning
- `theta_init`: May tune epsilon lower for more inelastic markets

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Status**: Ready for Implementation
