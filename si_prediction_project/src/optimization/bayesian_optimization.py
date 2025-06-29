"""
Bayesian Optimization for Blast Furnace Control

This module implements Bayesian Optimization using Gaussian Processes
for efficient parameter optimization in blast furnace operations.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """
    Bayesian Optimization for Blast Furnace Parameter Optimization
    
    Uses Gaussian Processes to model the objective function and
    acquisition functions to efficiently explore the parameter space.
    """
    
    def __init__(self, predictor_model, config: Dict[str, Any] = None):
        self.predictor_model = predictor_model
        self.config = config or {}
        
        # Default configuration
        default_config = {
            'n_initial_points': 10,
            'n_optimization_steps': 50,
            'acquisition_function': 'EI',  # EI, PI, UCB
            'xi': 0.01,  # Exploration parameter for EI
            'kappa': 2.576,  # Exploration parameter for UCB
            'target_si': 0.45,
            'alpha': 1e-6,  # GP noise parameter
            'normalize_y': True,
            'random_state': 42
        }
        
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Parameter bounds
        self.parameter_bounds = {
            'ore_feed_rate': (800, 1200),
            'coke_rate': (400, 600),
            'limestone_rate': (100, 200),
            'hot_blast_temp': (1000, 1300),
            'cold_blast_volume': (2000, 4000),
            'hot_blast_pressure': (2.5, 4.0),
            'blast_humidity': (10, 30),
            'oxygen_enrichment': (0, 5),
            'fuel_injection_rate': (100, 200),
            'hearth_temp': (1400, 1600),
            'stack_temp': (400, 600),
            'furnace_pressure': (1.5, 2.5),
            'gas_flow_rate': (3000, 5000),
            'burden_distribution': (0.3, 0.7),
            'tap_hole_temp': (1450, 1550),
            'slag_basicity': (1.0, 1.4),
            'iron_flow_rate': (200, 400),
            'thermal_state': (0.8, 1.2),
            'permeability_index': (0.5, 1.5)
        }
        
        self.param_names = list(self.parameter_bounds.keys())
        self.bounds_array = np.array(list(self.parameter_bounds.values()))
        
        # Initialize GP model
        self.gp_model = None
        self.X_observed = []
        self.y_observed = []
        self.optimization_history = []
        
        # Set random seed
        np.random.seed(self.config['random_state'])
    
    def _initialize_gp(self):
        """Initialize Gaussian Process model"""
        # Define kernel
        kernel = (
            C(1.0, (1e-3, 1e3)) * 
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
        )
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config['alpha'],
            normalize_y=self.config['normalize_y'],
            n_restarts_optimizer=10,
            random_state=self.config['random_state']
        )
    
    def _objective_function(self, params_array: np.ndarray) -> float:
        """
        Objective function to minimize
        
        Args:
            params_array: Array of parameter values
            
        Returns:
            Objective value (negative because we minimize)
        """
        # Convert array to parameter dictionary
        params = dict(zip(self.param_names, params_array))
        
        try:
            # Predict SI content
            predicted_si = self._predict_si(params)
            
            # Calculate primary objective (SI error)
            si_error = abs(predicted_si - self.config['target_si'])
            
            # Add constraint penalties
            constraint_penalty = self._calculate_constraint_penalty(params)
            
            # Add stability penalty
            stability_penalty = self._calculate_stability_penalty(params)
            
            # Total objective (minimize)
            objective = si_error + constraint_penalty + stability_penalty
            
            return objective
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1e6  # Large penalty for errors
    
    def _predict_si(self, params: Dict[str, float]) -> float:
        """Predict SI content using the trained model"""
        try:
            # Use your existing predictor model here
            # For now, using a placeholder
            return 0.4 + np.random.normal(0, 0.05)
        except Exception as e:
            logger.error(f"Error predicting SI: {e}")
            return 0.5  # Default value
    
    def _calculate_constraint_penalty(self, params: Dict[str, float]) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0.0
        
        # Parameter bounds constraints
        for param_name, value in params.items():
            bounds = self.parameter_bounds[param_name]
            if value < bounds[0]:
                penalty += (bounds[0] - value) ** 2
            elif value > bounds[1]:
                penalty += (value - bounds[1]) ** 2
        
        # Operational constraints
        # Temperature difference constraint
        temp_diff = abs(params['hot_blast_temp'] - params['hearth_temp'])
        if temp_diff > 500:
            penalty += (temp_diff - 500) * 0.001
        
        # Coke-to-ore ratio constraint
        coke_ore_ratio = params['coke_rate'] / params['ore_feed_rate']
        if coke_ore_ratio < 0.3 or coke_ore_ratio > 0.6:
            penalty += abs(coke_ore_ratio - 0.45) * 10
        
        return penalty
    
    def _calculate_stability_penalty(self, params: Dict[str, float]) -> float:
        """Calculate penalty for parameters far from stable operating regions"""
        penalty = 0.0
        
        # Prefer parameters in middle ranges
        for param_name, value in params.items():
            bounds = self.parameter_bounds[param_name]
            mid_point = (bounds[0] + bounds[1]) / 2
            range_size = bounds[1] - bounds[0]
            
            # Normalized distance from center
            distance = abs(value - mid_point) / (range_size / 2)
            
            # Quadratic penalty for extreme values
            if distance > 0.8:
                penalty += (distance - 0.8) ** 2
        
        return penalty * 0.1
    
    def _acquisition_function(self, params_array: np.ndarray) -> float:
        """
        Acquisition function for Bayesian optimization
        
        Args:
            params_array: Array of parameter values
            
        Returns:
            Acquisition value (higher is better for selection)
        """
        if len(self.X_observed) == 0:
            return 0.0
        
        # Reshape for GP prediction
        X_test = params_array.reshape(1, -1)
        
        # Get GP predictions
        mu, sigma = self.gp_model.predict(X_test, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        # Handle numerical issues
        if sigma < 1e-10:
            sigma = 1e-10
        
        # Current best observed value
        f_best = np.min(self.y_observed)
        
        if self.config['acquisition_function'] == 'EI':
            # Expected Improvement
            xi = self.config['xi']
            improvement = f_best - mu - xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei
        
        elif self.config['acquisition_function'] == 'PI':
            # Probability of Improvement
            xi = self.config['xi']
            Z = (f_best - mu - xi) / sigma
            pi = norm.cdf(Z)
            return pi
        
        elif self.config['acquisition_function'] == 'UCB':
            # Upper Confidence Bound
            kappa = self.config['kappa']
            ucb = -(mu - kappa * sigma)  # Negative because we minimize
            return ucb
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.config['acquisition_function']}")
    
    def _generate_initial_points(self, n_points: int) -> np.ndarray:
        """Generate initial points for exploration"""
        # Latin Hypercube Sampling
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=len(self.param_names), seed=self.config['random_state'])
        unit_samples = sampler.random(n_points)
        
        # Scale to parameter bounds
        samples = np.zeros_like(unit_samples)
        for i, (min_val, max_val) in enumerate(self.bounds_array):
            samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)
        
        return samples
    
    def _optimize_acquisition(self) -> np.ndarray:
        """Optimize acquisition function to find next point"""
        # Multiple random restarts
        n_restarts = 10
        best_x = None
        best_acquisition = -np.inf
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(
                self.bounds_array[:, 0], 
                self.bounds_array[:, 1]
            )
            
            # Optimize acquisition function
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x),  # Negative because minimize
                    x0,
                    method='L-BFGS-B',
                    bounds=self.bounds_array,
                    options={'maxiter': 1000}
                )
                
                if result.success and -result.fun > best_acquisition:
                    best_acquisition = -result.fun
                    best_x = result.x
            except:
                continue
        
        # Fallback to random point if optimization fails
        if best_x is None:
            best_x = np.random.uniform(
                self.bounds_array[:, 0], 
                self.bounds_array[:, 1]
            )
        
        return best_x
    
    def optimize(self, current_params: Optional[Dict[str, float]] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run Bayesian optimization
        
        Args:
            current_params: Current parameter values (optional)
            verbose: Whether to print progress
            
        Returns:
            Optimization results
        """
        logger.info("Starting Bayesian Optimization...")
        
        # Initialize GP model
        self._initialize_gp()
        
        # Clear previous observations
        self.X_observed = []
        self.y_observed = []
        
        # Generate initial points
        if current_params:
            # Include current parameters in initial points
            current_array = np.array([current_params.get(name, 0) for name in self.param_names])
            initial_points = self._generate_initial_points(self.config['n_initial_points'] - 1)
            initial_points = np.vstack([current_array.reshape(1, -1), initial_points])
        else:
            initial_points = self._generate_initial_points(self.config['n_initial_points'])
        
        # Evaluate initial points
        for i, point in enumerate(initial_points):
            objective_value = self._objective_function(point)
            self.X_observed.append(point)
            self.y_observed.append(objective_value)
            
            if verbose:
                logger.info(f"Initial point {i+1}/{len(initial_points)}: {objective_value:.6f}")
        
        # Convert to numpy arrays
        self.X_observed = np.array(self.X_observed)
        self.y_observed = np.array(self.y_observed)
        
        # Fit initial GP
        self.gp_model.fit(self.X_observed, self.y_observed)
        
        # Bayesian optimization loop
        for iteration in range(self.config['n_optimization_steps']):
            # Find next point using acquisition function
            next_point = self._optimize_acquisition()
            
            # Evaluate objective at next point
            objective_value = self._objective_function(next_point)
            
            # Update observations
            self.X_observed = np.vstack([self.X_observed, next_point.reshape(1, -1)])
            self.y_observed = np.append(self.y_observed, objective_value)
            
            # Update GP model
            self.gp_model.fit(self.X_observed, self.y_observed)
            
            if verbose:
                logger.info(f"Iteration {iteration+1}/{self.config['n_optimization_steps']}: {objective_value:.6f}")
        
        # Find best solution
        best_idx = np.argmin(self.y_observed)
        best_params_array = self.X_observed[best_idx]
        best_objective = self.y_observed[best_idx]
        
        # Convert to parameter dictionary
        best_params = dict(zip(self.param_names, best_params_array))
        
        # Predict SI for best solution
        predicted_si = self._predict_si(best_params)
        
        # Calculate uncertainty at best point
        mu, sigma = self.gp_model.predict(best_params_array.reshape(1, -1), return_std=True)
        
        # Store results
        optimization_result = {
            'optimized_parameters': best_params,
            'objective_value': best_objective,
            'predicted_si': predicted_si,
            'target_si': self.config['target_si'],
            'si_error': abs(predicted_si - self.config['target_si']),
            'uncertainty': sigma[0],
            'method': 'Bayesian Optimization',
            'acquisition_function': self.config['acquisition_function'],
            'n_evaluations': len(self.y_observed),
            'convergence_curve': self.y_observed.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Bayesian optimization completed. Best objective: {best_objective:.6f}")
        logger.info(f"Predicted SI: {predicted_si:.4f}, Target SI: {self.config['target_si']:.4f}")
        
        return optimization_result
    
    def constrained_optimize(self, constraints: List[Dict[str, Any]],
                           current_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Constrained Bayesian optimization with explicit constraints
        
        Args:
            constraints: List of constraint dictionaries
            current_params: Current parameter values
            
        Returns:
            Constrained optimization results
        """
        # Modified objective function with constraint handling
        def constrained_objective(params_array):
            params = dict(zip(self.param_names, params_array))
            
            # Base objective
            base_objective = self._objective_function(params_array)
            
            # Constraint penalties
            constraint_violation = 0.0
            for constraint in constraints:
                if constraint['type'] == 'ineq':
                    # Inequality constraint: g(x) >= 0
                    violation = -constraint['fun'](params)
                    if violation > 0:
                        constraint_violation += violation ** 2 * constraint.get('weight', 1.0)
                elif constraint['type'] == 'eq':
                    # Equality constraint: h(x) = 0
                    violation = abs(constraint['fun'](params))
                    constraint_violation += violation ** 2 * constraint.get('weight', 1.0)
            
            return base_objective + constraint_violation * 100  # Large penalty weight
        
        # Temporarily replace objective function
        original_objective = self._objective_function
        self._objective_function = constrained_objective
        
        try:
            # Run optimization with modified objective
            result = self.optimize(current_params, verbose=True)
            result['constraints'] = constraints
            result['method'] = 'Constrained Bayesian Optimization'
            
            return result
        finally:
            # Restore original objective function
            self._objective_function = original_objective
    
    def multi_objective_optimize(self, objectives: List[str],
                               weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Multi-objective Bayesian optimization using scalarization
        
        Args:
            objectives: List of objective names
            weights: Weights for objectives (if None, equal weights)
            
        Returns:
            Multi-objective optimization results
        """
        if weights is None:
            weights = [1.0] * len(objectives)
        
        # Define multi-objective function
        def multi_objective_function(params_array):
            params = dict(zip(self.param_names, params_array))
            predicted_si = self._predict_si(params)
            
            objective_values = []
            
            for objective in objectives:
                if objective == 'si_accuracy':
                    value = abs(predicted_si - self.config['target_si'])
                elif objective == 'stability':
                    value = self._calculate_stability_penalty(params)
                elif objective == 'efficiency':
                    value = params['coke_rate'] / 1000.0  # Normalized
                elif objective == 'safety':
                    value = self._calculate_constraint_penalty(params)
                else:
                    value = 0.0
                
                objective_values.append(value)
            
            # Weighted sum scalarization
            return sum(w * obj for w, obj in zip(weights, objective_values))
        
        # Temporarily replace objective function
        original_objective = self._objective_function
        self._objective_function = multi_objective_function
        
        try:
            result = self.optimize(verbose=True)
            result['objectives'] = objectives
            result['weights'] = weights
            result['method'] = 'Multi-Objective Bayesian Optimization'
            
            return result
        finally:
            # Restore original objective function
            self._objective_function = original_objective
    
    def adaptive_optimize(self, current_params: Dict[str, float],
                         performance_history: List[Dict]) -> Dict[str, Any]:
        """
        Adaptive Bayesian optimization that adjusts parameters based on history
        
        Args:
            current_params: Current parameter values
            performance_history: Historical performance data
            
        Returns:
            Adaptive optimization results
        """
        # Analyze performance history
        if len(performance_history) > 5:
            recent_errors = [h.get('si_error', 0.1) for h in performance_history[-5:]]
            avg_error = np.mean(recent_errors)
            
            # Adapt acquisition function parameters
            if avg_error > 0.05:  # High error, increase exploration
                self.config['xi'] = min(0.1, self.config['xi'] * 1.5)
                self.config['kappa'] = min(5.0, self.config['kappa'] * 1.2)
            else:  # Good performance, focus on exploitation
                self.config['xi'] = max(0.001, self.config['xi'] * 0.8)
                self.config['kappa'] = max(1.0, self.config['kappa'] * 0.9)
        
        # Run optimization with adapted parameters
        result = self.optimize(current_params, verbose=False)
        result['adaptation_info'] = {
            'adapted_xi': self.config['xi'],
            'adapted_kappa': self.config['kappa']
        }
        
        return result
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Analyze parameter importance based on GP model
        
        Returns:
            Dictionary with importance scores for each parameter
        """
        if self.gp_model is None or len(self.X_observed) == 0:
            return {}
        
        # Get kernel hyperparameters (length scales)
        kernel = self.gp_model.kernel_
        
        # Extract length scales (assuming RBF kernel)
        try:
            if hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
                length_scales = kernel.k2.length_scale
            elif hasattr(kernel, 'length_scale'):
                length_scales = kernel.length_scale
            else:
                return {}
            
            # Convert to importance (inverse of length scale)
            if np.isscalar(length_scales):
                length_scales = np.array([length_scales] * len(self.param_names))
            
            importances = 1.0 / (length_scales + 1e-10)
            
            # Normalize to sum to 1
            importances = importances / np.sum(importances)
            
            return dict(zip(self.param_names, importances))
        
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return {}
    
    def predict_with_uncertainty(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict objective value with uncertainty
        
        Args:
            params: Parameter values
            
        Returns:
            Tuple of (mean, std) predictions
        """
        if self.gp_model is None:
            return 0.0, 1.0
        
        params_array = np.array([params.get(name, 0) for name in self.param_names])
        mu, sigma = self.gp_model.predict(params_array.reshape(1, -1), return_std=True)
        
        return mu[0], sigma[0]
    
    def save_optimization_history(self, filepath: str):
        """Save optimization history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.optimization_history, f, indent=2, default=str)
    
    def load_optimization_history(self, filepath: str):
        """Load optimization history from file"""
        with open(filepath, 'r') as f:
            self.optimization_history = json.load(f) 