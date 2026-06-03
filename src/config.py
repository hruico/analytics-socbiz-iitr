"""System configuration with validation."""
from typing import Literal, Tuple
from pydantic import BaseModel, Field, field_validator
import json


class SystemConfig(BaseModel):
    """Configuration for the EV Tariff Optimization System."""
    
    # LLM Provider
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "gpt-4o"
    
    # Geography-Aware Pricing
    baseline_tariff_per_kwh: float = Field(default=15.0, gt=0.0)
    pricing_bounds: Tuple[float, float] = Field(default=(10.0, 22.0))
    
    # Initial Parameters [epsilon, alpha, beta]
    theta_init: Tuple[float, float, float] = (1.5, 2.5, 2.5)
    
    # Convergence Criteria
    revenue_variance_threshold: float = 1.0
    parameter_delta_threshold: float = 0.01
    utilization_std_threshold: float = 0.15
    max_utilization_threshold: float = 0.80
    queue_reduction_target: float = 0.20
    convergence_window: int = 50
    max_iterations: int = 1000
    
    # Learning Rate Schedule
    learning_rate_init: float = 0.1
    learning_rate_decay: float = 0.001
    
    # Reward Weights [w1, w2, w3]
    reward_weights: Tuple[float, float, float] = (1.0, 0.5, 0.3)
    
    # Reproducibility
    random_seed: int = 42
    train_ratio: float = 0.80
    
    # Agent Retry Policy
    max_agent_retries: int = 3
    retry_backoff_seconds: float = 2.0
    
    # LLM Cost Controls
    llm_cost_budget_usd: float = 10.0
    max_llm_calls_per_step: int = 5
    llm_token_budget: int = 100000
    
    @field_validator("pricing_bounds")
    @classmethod
    def validate_pricing_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if len(v) != 2:
            raise ValueError("pricing_bounds must be 2-element tuple")
        if v[0] <= 0:
            raise ValueError("pricing_bounds[0] must be positive")
        if v[1] <= v[0]:
            raise ValueError("pricing_bounds[1] must be greater than pricing_bounds[0]")
        return v
    
    @field_validator("theta_init")
    @classmethod
    def validate_theta(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        eps, alpha, beta = v
        if not (0.1 <= eps <= 5.0):
            raise ValueError(f"epsilon {eps} must be in [0.1, 5.0]")
        if not (1.0 <= alpha <= 10.0):
            raise ValueError(f"alpha {alpha} must be in [1.0, 10.0]")
        if not (1.0 <= beta <= 10.0):
            raise ValueError(f"beta {beta} must be in [1.0, 10.0]")
        return v
    
    def model_post_init(self, __context):
        """Validate baseline is within bounds."""
        if not (self.pricing_bounds[0] <= self.baseline_tariff_per_kwh <= self.pricing_bounds[1]):
            raise ValueError(
                f"baseline_tariff {self.baseline_tariff_per_kwh} must be within "
                f"pricing_bounds {self.pricing_bounds}"
            )


class ConfigParser:
    """Parser for loading and saving configurations."""
    
    @staticmethod
    def parse(path: str) -> SystemConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return SystemConfig(**data)
    
    @staticmethod
    def serialize(config: SystemConfig, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(config.model_dump(), f, indent=2)
