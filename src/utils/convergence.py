"""Convergence checker for multi-objective optimization."""
from collections import deque
import numpy as np
from typing import Dict


class ConvergenceChecker:
    """Monitors convergence across revenue, parameters, utilization, and queue."""
    
    def __init__(self, config):
        self.window = config.convergence_window
        self.revenue_variance_threshold = config.revenue_variance_threshold
        self.parameter_delta_threshold = config.parameter_delta_threshold
        self.utilization_std_threshold = config.utilization_std_threshold
        self.max_utilization_threshold = config.max_utilization_threshold
        self.queue_reduction_target = config.queue_reduction_target
        
        self.revenue_history = deque(maxlen=self.window)
        self.theta_history = deque(maxlen=self.window)
        self.util_history = deque(maxlen=self.window)
        self.queue_history = deque(maxlen=self.window)
        
        self.consecutive_steps = 0
        self.baseline_queue_mean = None
    
    def set_baseline_queue(self, baseline_mean: float):
        """Set baseline queue mean for convergence criterion."""
        self.baseline_queue_mean = baseline_mean
    
    def update(self, revenue_gain_pct: float, theta: np.ndarray, utilization: float, queue: float):
        """Update history with latest metrics."""
        self.revenue_history.append(revenue_gain_pct)
        self.theta_history.append(theta.copy())
        self.util_history.append(utilization)
        self.queue_history.append(queue)
    
    def check(self) -> Dict:
        """Check if convergence criteria are met."""
        if len(self.revenue_history) < self.window:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        # Check 1: Revenue stability
        revenue_var = np.var(self.revenue_history)
        revenue_stable = revenue_var < self.revenue_variance_threshold
        
        # Check 2: Parameter stability
        theta_deltas = [
            np.abs(self.theta_history[i] - self.theta_history[i-1])
            for i in range(1, len(self.theta_history))
        ]
        max_delta = np.max(theta_deltas) if theta_deltas else 0
        params_stable = max_delta < self.parameter_delta_threshold
        
        # Check 3: Utilization health
        util_std = np.std(self.util_history)
        util_max = np.max(self.util_history)
        util_healthy = util_std < self.utilization_std_threshold and util_max < self.max_utilization_threshold
        
        # Check 4: Queue reduction
        queue_mean = np.mean(self.queue_history)
        queue_reduced = True
        if self.baseline_queue_mean is not None:
            target_queue = (1 - self.queue_reduction_target) * self.baseline_queue_mean
            queue_reduced = queue_mean < target_queue
        
        # All conditions met?
        all_met = revenue_stable and params_stable and util_healthy and queue_reduced
        
        if all_met:
            self.consecutive_steps += 1
        else:
            self.consecutive_steps = 0
        
        converged = self.consecutive_steps >= self.window
        
        return {
            'converged': converged,
            'revenue_stable': revenue_stable,
            'params_stable': params_stable,
            'util_healthy': util_healthy,
            'queue_reduced': queue_reduced,
            'consecutive_steps': self.consecutive_steps,
            'revenue_var': revenue_var,
            'max_param_delta': max_delta,
            'util_std': util_std
        }
