"""
Optimization Manager for Blast Furnace Control

This module provides a unified interface for managing multiple optimization
algorithms (RL, GA, BO) and selecting the best approach for different scenarios.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .reinforcement_learning import RLOptimizer
from .genetic_algorithm import GAOptimizer  
from .bayesian_optimization import BayesianOptimizer

logger = logging.getLogger(__name__)


class OptimizationManager:
    """
    Unified Optimization Manager for Blast Furnace Control
    
    Manages multiple optimization algorithms and provides intelligent
    algorithm selection based on problem characteristics and performance history.
    """
    
    def __init__(self, predictor_model, config: Dict[str, Any] = None):
        self.predictor_model = predictor_model
        self.config = config or {}
        
        # Default configuration
        default_config = {
            'default_algorithm': 'auto',  # auto, rl, ga, bo
            'target_si': 0.45,
            'si_tolerance': 0.02,
            'max_runtime_minutes': 30,
            'parallel_optimization': True,
            'ensemble_mode': False,
            'adaptive_selection': True,
            'performance_history_limit': 100
        }
        
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize optimizers
        self.optimizers = {}
        self._initialize_optimizers()
        
        # Performance tracking
        self.optimization_history = []
        self.algorithm_performance = {
            'rl': {'success_rate': 0.0, 'avg_error': 0.1, 'avg_runtime': 0.0, 'count': 0},
            'ga': {'success_rate': 0.0, 'avg_error': 0.1, 'avg_runtime': 0.0, 'count': 0},
            'bo': {'success_rate': 0.0, 'avg_error': 0.1, 'avg_runtime': 0.0, 'count': 0}
        }
        
    def _initialize_optimizers(self):
        """Initialize all optimization algorithms"""
        try:
            # Reinforcement Learning Optimizer
            rl_config = {
                'algorithm': 'PPO',
                'target_si': self.config['target_si'],
                'training_timesteps': 50000  # Reduced for faster initialization
            }
            self.optimizers['rl'] = RLOptimizer(self.predictor_model, rl_config)
            
            # Genetic Algorithm Optimizer
            ga_config = {
                'population_size': 50,
                'generations': 30,
                'target_si': self.config['target_si']
            }
            self.optimizers['ga'] = GAOptimizer(self.predictor_model, ga_config)
            
            # Bayesian Optimization
            bo_config = {
                'n_initial_points': 8,
                'n_optimization_steps': 25,
                'target_si': self.config['target_si']
            }
            self.optimizers['bo'] = BayesianOptimizer(self.predictor_model, bo_config)
            
            logger.info("All optimizers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing optimizers: {e}")
            raise
    
    def select_best_algorithm(self, current_params: Dict[str, float],
                            context: Optional[Dict[str, Any]] = None) -> str:
        """
        Intelligently select the best optimization algorithm
        
        Args:
            current_params: Current parameter values
            context: Additional context for selection
            
        Returns:
            Selected algorithm name ('rl', 'ga', 'bo')
        """
        if not self.config['adaptive_selection']:
            return self.config['default_algorithm']
        
        context = context or {}
        
        # Analyze problem characteristics
        problem_features = self._analyze_problem_characteristics(current_params, context)
        
        # Score algorithms based on features and performance history
        algorithm_scores = {}
        
        for algo in ['rl', 'ga', 'bo']:
            score = self._calculate_algorithm_score(algo, problem_features)
            algorithm_scores[algo] = score
        
        # Select best algorithm
        best_algorithm = max(algorithm_scores, key=algorithm_scores.get)
        
        logger.info(f"Algorithm selection scores: {algorithm_scores}")
        logger.info(f"Selected algorithm: {best_algorithm}")
        
        return best_algorithm
    
    def _analyze_problem_characteristics(self, current_params: Dict[str, float],
                                       context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze problem characteristics for algorithm selection"""
        features = {}
        
        # Parameter complexity (number of parameters to optimize)
        features['parameter_complexity'] = len(current_params) / 19.0  # Normalized
        
        # Historical performance variability
        if len(self.optimization_history) > 5:
            recent_errors = [h.get('si_error', 0.1) for h in self.optimization_history[-5:]]
            features['error_variability'] = np.std(recent_errors)
        else:
            features['error_variability'] = 0.05  # Default
        
        # Time constraint (normalized)
        max_time = context.get('max_time_minutes', self.config['max_runtime_minutes'])
        features['time_constraint'] = min(1.0, max_time / 60.0)  # Normalized to hours
        
        # Problem difficulty (distance from target)
        if 'current_si' in context:
            current_si = context['current_si']
            target_si = self.config['target_si']
            features['problem_difficulty'] = abs(current_si - target_si) / 0.5  # Normalized
        else:
            features['problem_difficulty'] = 0.5  # Default
        
        # Exploration vs exploitation need
        if len(self.optimization_history) > 10:
            recent_improvements = []
            for i in range(1, min(6, len(self.optimization_history))):
                current_error = self.optimization_history[-i].get('si_error', 0.1)
                prev_error = self.optimization_history[-i-1].get('si_error', 0.1)
                improvement = prev_error - current_error
                recent_improvements.append(improvement)
            
            avg_improvement = np.mean(recent_improvements)
            features['exploration_need'] = 1.0 if avg_improvement < 0.001 else 0.0
        else:
            features['exploration_need'] = 1.0  # High exploration for new problems
        
        return features
    
    def _calculate_algorithm_score(self, algorithm: str, features: Dict[str, float]) -> float:
        """Calculate score for an algorithm based on problem features"""
        base_score = 0.5
        perf = self.algorithm_performance[algorithm]
        
        # Performance-based component
        if perf['count'] > 0:
            success_component = perf['success_rate'] * 0.4
            error_component = (1.0 - min(1.0, perf['avg_error'] / 0.1)) * 0.3
            runtime_component = (1.0 - min(1.0, perf['avg_runtime'] / 300)) * 0.1  # 5 min baseline
            performance_score = success_component + error_component + runtime_component
        else:
            performance_score = 0.5  # Neutral for untested algorithms
        
        # Feature-based component
        feature_score = 0.5
        
        if algorithm == 'rl':
            # RL is good for: complex problems, long-term optimization, adaptive control
            feature_score += features['parameter_complexity'] * 0.3
            feature_score += (1.0 - features['time_constraint']) * 0.2  # Prefers more time
            feature_score += features['exploration_need'] * 0.2
            
        elif algorithm == 'ga':
            # GA is good for: multi-modal problems, global optimization, parallel execution
            feature_score += features['problem_difficulty'] * 0.3
            feature_score += features['exploration_need'] * 0.3
            feature_score += features['parameter_complexity'] * 0.2
            
        elif algorithm == 'bo':
            # BO is good for: expensive evaluations, small parameter spaces, quick convergence
            feature_score += (1.0 - features['parameter_complexity']) * 0.3
            feature_score += features['time_constraint'] * 0.3
            feature_score += (1.0 - features['exploration_need']) * 0.2
        
        # Combine scores
        total_score = 0.6 * performance_score + 0.4 * feature_score
        
        return max(0.0, min(1.0, total_score))
    
    def optimize(self, current_params: Dict[str, float],
                algorithm: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run optimization using selected or specified algorithm
        
        Args:
            current_params: Current parameter values
            algorithm: Specific algorithm to use (None for auto-selection)
            context: Additional context for optimization
            
        Returns:
            Optimization results
        """
        start_time = datetime.now()
        context = context or {}
        
        # Select algorithm if not specified
        if algorithm is None or algorithm == 'auto':
            algorithm = self.select_best_algorithm(current_params, context)
        
        logger.info(f"Running optimization with algorithm: {algorithm}")
        
        try:
            # Run optimization
            if algorithm == 'rl':
                result = self._run_rl_optimization(current_params, context)
            elif algorithm == 'ga':
                result = self._run_ga_optimization(current_params, context)
            elif algorithm == 'bo':
                result = self._run_bo_optimization(current_params, context)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Calculate runtime
            runtime = (datetime.now() - start_time).total_seconds()
            result['runtime_seconds'] = runtime
            result['algorithm_used'] = algorithm
            
            # Update performance tracking
            self._update_algorithm_performance(algorithm, result, runtime)
            
            # Store in history
            self.optimization_history.append(result)
            
            # Limit history size
            if len(self.optimization_history) > self.config['performance_history_limit']:
                self.optimization_history = self.optimization_history[-self.config['performance_history_limit']:]
            
            logger.info(f"Optimization completed in {runtime:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed with {algorithm}: {e}")
            
            # Try fallback algorithm
            fallback_algorithm = self._get_fallback_algorithm(algorithm)
            if fallback_algorithm and fallback_algorithm != algorithm:
                logger.info(f"Trying fallback algorithm: {fallback_algorithm}")
                return self.optimize(current_params, fallback_algorithm, context)
            else:
                raise
    
    def _run_rl_optimization(self, current_params: Dict[str, float],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Run RL optimization"""
        rl_optimizer = self.optimizers['rl']
        
        # Train if not already trained (for demo purposes, use pre-trained)
        if rl_optimizer.model is None:
            logger.info("Training RL model (this may take a while)...")
            rl_optimizer.train()
        
        # Run optimization
        result = rl_optimizer.optimize_parameters(current_params, n_steps=10)
        return result
    
    def _run_ga_optimization(self, current_params: Dict[str, float],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Run GA optimization"""
        ga_optimizer = self.optimizers['ga']
        
        # Adjust parameters based on context
        if context.get('max_time_minutes', 30) < 10:
            ga_optimizer.config['generations'] = 20
            ga_optimizer.config['population_size'] = 30
        
        result = ga_optimizer.optimize(current_params, verbose=False)
        return result
    
    def _run_bo_optimization(self, current_params: Dict[str, float],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        bo_optimizer = self.optimizers['bo']
        
        # Adjust parameters based on context
        if context.get('max_time_minutes', 30) < 5:
            bo_optimizer.config['n_optimization_steps'] = 15
            bo_optimizer.config['n_initial_points'] = 5
        
        result = bo_optimizer.optimize(current_params, verbose=False)
        return result
    
    def _get_fallback_algorithm(self, failed_algorithm: str) -> Optional[str]:
        """Get fallback algorithm when primary fails"""
        fallback_map = {
            'rl': 'bo',  # BO is faster fallback for RL
            'ga': 'bo',  # BO is more reliable fallback for GA
            'bo': 'ga'   # GA is robust fallback for BO
        }
        return fallback_map.get(failed_algorithm)
    
    def _update_algorithm_performance(self, algorithm: str, result: Dict[str, Any], runtime: float):
        """Update performance metrics for an algorithm"""
        perf = self.algorithm_performance[algorithm]
        
        # Success criteria
        si_error = result.get('si_error', 0.1)
        is_success = si_error < self.config['si_tolerance']
        
        # Update metrics
        perf['count'] += 1
        perf['success_rate'] = ((perf['success_rate'] * (perf['count'] - 1)) + (1.0 if is_success else 0.0)) / perf['count']
        perf['avg_error'] = ((perf['avg_error'] * (perf['count'] - 1)) + si_error) / perf['count']
        perf['avg_runtime'] = ((perf['avg_runtime'] * (perf['count'] - 1)) + runtime) / perf['count']
    
    def ensemble_optimize(self, current_params: Dict[str, float],
                         algorithms: Optional[List[str]] = None,
                         voting_method: str = 'weighted') -> Dict[str, Any]:
        """
        Run ensemble optimization using multiple algorithms
        
        Args:
            current_params: Current parameter values
            algorithms: List of algorithms to use (None for all)
            voting_method: How to combine results ('weighted', 'average', 'best')
            
        Returns:
            Ensemble optimization results
        """
        if algorithms is None:
            algorithms = ['rl', 'ga', 'bo']
        
        logger.info(f"Running ensemble optimization with: {algorithms}")
        
        # Run optimizations in parallel if enabled
        if self.config['parallel_optimization']:
            results = self._run_parallel_optimization(current_params, algorithms)
        else:
            results = self._run_sequential_optimization(current_params, algorithms)
        
        # Combine results
        ensemble_result = self._combine_optimization_results(results, voting_method)
        ensemble_result['method'] = f'Ensemble ({voting_method})'
        ensemble_result['individual_results'] = results
        
        return ensemble_result
    
    def _run_parallel_optimization(self, current_params: Dict[str, float],
                                 algorithms: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run optimizations in parallel"""
        results = {}
        
        def run_algorithm(algo):
            try:
                return algo, self.optimize(current_params, algo)
            except Exception as e:
                logger.error(f"Algorithm {algo} failed: {e}")
                return algo, None
        
        with ThreadPoolExecutor(max_workers=len(algorithms)) as executor:
            futures = [executor.submit(run_algorithm, algo) for algo in algorithms]
            
            for future in futures:
                algo, result = future.result()
                if result is not None:
                    results[algo] = result
        
        return results
    
    def _run_sequential_optimization(self, current_params: Dict[str, float],
                                   algorithms: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run optimizations sequentially"""
        results = {}
        
        for algo in algorithms:
            try:
                result = self.optimize(current_params, algo)
                results[algo] = result
            except Exception as e:
                logger.error(f"Algorithm {algo} failed: {e}")
                continue
        
        return results
    
    def _combine_optimization_results(self, results: Dict[str, Dict[str, Any]],
                                    voting_method: str) -> Dict[str, Any]:
        """Combine results from multiple algorithms"""
        if not results:
            raise ValueError("No successful optimization results to combine")
        
        if voting_method == 'best':
            # Select best result based on SI error
            best_algo = min(results.keys(), key=lambda k: results[k].get('si_error', 1.0))
            best_result = results[best_algo].copy()
            best_result['selected_algorithm'] = best_algo
            return best_result
        
        elif voting_method == 'weighted':
            # Weighted average based on algorithm performance
            weights = {}
            total_weight = 0.0
            
            for algo in results.keys():
                perf = self.algorithm_performance[algo]
                # Weight based on success rate and inverse error
                weight = perf['success_rate'] * (1.0 / (perf['avg_error'] + 0.01))
                weights[algo] = weight
                total_weight += weight
            
            # Normalize weights
            for algo in weights:
                weights[algo] /= total_weight
        
        elif voting_method == 'average':
            # Equal weights
            weights = {algo: 1.0/len(results) for algo in results.keys()}
        
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")
        
        # Combine parameters using weighted average
        combined_params = {}
        param_names = list(results[list(results.keys())[0]]['optimized_parameters'].keys())
        
        for param_name in param_names:
            weighted_sum = 0.0
            for algo, result in results.items():
                param_value = result['optimized_parameters'][param_name]
                weighted_sum += weights[algo] * param_value
            combined_params[param_name] = weighted_sum
        
        # Calculate combined metrics
        combined_si_error = sum(weights[algo] * results[algo].get('si_error', 0.1) 
                              for algo in results.keys())
        
        combined_predicted_si = sum(weights[algo] * results[algo].get('predicted_si', 0.45)
                                  for algo in results.keys())
        
        return {
            'optimized_parameters': combined_params,
            'predicted_si': combined_predicted_si,
            'si_error': combined_si_error,
            'target_si': self.config['target_si'],
            'weights': weights,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_optimization_recommendations(self, current_params: Dict[str, float],
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimization recommendations without running full optimization
        
        Args:
            current_params: Current parameter values
            context: Additional context
            
        Returns:
            Optimization recommendations
        """
        context = context or {}
        
        # Analyze current situation
        problem_features = self._analyze_problem_characteristics(current_params, context)
        recommended_algorithm = self.select_best_algorithm(current_params, context)
        
        # Get algorithm-specific insights
        insights = {}
        
        for algo in ['rl', 'ga', 'bo']:
            optimizer = self.optimizers[algo]
            perf = self.algorithm_performance[algo]
            
            insights[algo] = {
                'recommended': algo == recommended_algorithm,
                'expected_runtime': perf['avg_runtime'],
                'expected_error': perf['avg_error'],
                'success_probability': perf['success_rate'],
                'confidence': 'high' if perf['count'] > 10 else 'medium' if perf['count'] > 3 else 'low'
            }
        
        return {
            'recommended_algorithm': recommended_algorithm,
            'problem_features': problem_features,
            'algorithm_insights': insights,
            'optimization_history_size': len(self.optimization_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_state(self, filepath: str):
        """Save optimization manager state"""
        state = {
            'config': self.config,
            'algorithm_performance': self.algorithm_performance,
            'optimization_history': self.optimization_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load optimization manager state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config.update(state.get('config', {}))
        self.algorithm_performance = state.get('algorithm_performance', self.algorithm_performance)
        self.optimization_history = state.get('optimization_history', []) 