"""
Genetic Algorithm Optimizer for Blast Furnace Control

This module implements a Genetic Algorithm (GA) for optimizing blast furnace
parameters to achieve target silicon content while maintaining operational constraints.
"""

import numpy as np
import random
from deap import base, creator, tools, algorithms
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class GAOptimizer:
    """
    Genetic Algorithm Optimizer for Blast Furnace Parameter Optimization
    
    Uses evolutionary algorithms to find optimal parameter combinations
    that minimize the error between predicted and target SI content.
    """
    
    def __init__(self, predictor_model, config: Dict[str, Any] = None):
        self.predictor_model = predictor_model
        self.config = config or {}
        
        # Default configuration
        default_config = {
            'population_size': 100,
            'generations': 50,
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'tournament_size': 3,
            'elite_size': 5,
            'target_si': 0.45,
            'si_tolerance': 0.05,
            'constraint_penalty': 10.0
        }
        
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Parameter bounds and constraints
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
        self.optimization_history = []
        
        # Setup DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP framework for genetic algorithm"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Setup toolbox
        self.toolbox = base.Toolbox()
        
        # Gene generation function
        def create_gene(param_idx):
            bounds = list(self.parameter_bounds.values())[param_idx]
            return random.uniform(bounds[0], bounds[1])
        
        # Individual and population creation
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config['tournament_size'])
    
    def _create_individual(self):
        """Create a random individual (parameter set)"""
        individual = []
        for param_name in self.param_names:
            bounds = self.parameter_bounds[param_name]
            value = random.uniform(bounds[0], bounds[1])
            individual.append(value)
        return creator.Individual(individual)
    
    def _evaluate_individual(self, individual):
        """
        Evaluate fitness of an individual
        
        Args:
            individual: List of parameter values
            
        Returns:
            Tuple containing fitness value (lower is better)
        """
        try:
            # Convert individual to parameter dictionary
            params = dict(zip(self.param_names, individual))
            
            # Predict SI content
            predicted_si = self._predict_si(params)
            
            # Calculate fitness components
            si_error = abs(predicted_si - self.config['target_si'])
            
            # Constraint violations
            constraint_penalty = self._calculate_constraint_penalty(params)
            
            # Stability penalty (prefer parameters closer to typical ranges)
            stability_penalty = self._calculate_stability_penalty(params)
            
            # Total fitness (minimize)
            fitness = si_error + constraint_penalty + stability_penalty
            
            return (fitness,)
            
        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            return (float('inf'),)  # Return worst possible fitness
    
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
        
        # Hard constraints (parameter bounds)
        for param_name, value in params.items():
            bounds = self.parameter_bounds[param_name]
            if value < bounds[0] or value > bounds[1]:
                penalty += self.config['constraint_penalty']
        
        # Operational constraints
        # Example: Hot blast temperature should not be too different from hearth temp
        temp_diff = abs(params['hot_blast_temp'] - params['hearth_temp'])
        if temp_diff > 500:  # Example constraint
            penalty += (temp_diff - 500) * 0.01
        
        # Coke rate vs ore feed rate ratio
        coke_ore_ratio = params['coke_rate'] / params['ore_feed_rate']
        if coke_ore_ratio < 0.3 or coke_ore_ratio > 0.6:
            penalty += abs(coke_ore_ratio - 0.45) * 10
        
        return penalty
    
    def _calculate_stability_penalty(self, params: Dict[str, float]) -> float:
        """Calculate penalty for parameters far from stable operating range"""
        penalty = 0.0
        
        # Define preferred operating ranges (middle 60% of bounds)
        for param_name, value in params.items():
            bounds = self.parameter_bounds[param_name]
            range_size = bounds[1] - bounds[0]
            
            # Preferred range is middle 60%
            preferred_min = bounds[0] + 0.2 * range_size
            preferred_max = bounds[1] - 0.2 * range_size
            
            if value < preferred_min:
                penalty += (preferred_min - value) / range_size
            elif value > preferred_max:
                penalty += (value - preferred_max) / range_size
        
        return penalty * 0.1  # Small weight for stability
    
    def _crossover(self, ind1, ind2):
        """Custom crossover operator"""
        # Blend crossover (BLX-α)
        alpha = 0.5
        
        for i in range(len(ind1)):
            if random.random() < self.config['crossover_rate']:
                # Get parameter bounds
                bounds = list(self.parameter_bounds.values())[i]
                
                # Calculate blend range
                d = abs(ind1[i] - ind2[i])
                min_val = min(ind1[i], ind2[i]) - alpha * d
                max_val = max(ind1[i], ind2[i]) + alpha * d
                
                # Clip to parameter bounds
                min_val = max(min_val, bounds[0])
                max_val = min(max_val, bounds[1])
                
                # Generate new values
                ind1[i] = random.uniform(min_val, max_val)
                ind2[i] = random.uniform(min_val, max_val)
        
        return ind1, ind2
    
    def _mutate(self, individual):
        """Custom mutation operator"""
        for i in range(len(individual)):
            if random.random() < self.config['mutation_rate']:
                bounds = list(self.parameter_bounds.values())[i]
                
                # Gaussian mutation with adaptive step size
                sigma = (bounds[1] - bounds[0]) * 0.1  # 10% of range
                individual[i] += random.gauss(0, sigma)
                
                # Clip to bounds
                individual[i] = max(bounds[0], min(bounds[1], individual[i]))
        
        return (individual,)
    
    def optimize(self, current_params: Optional[Dict[str, float]] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization
        
        Args:
            current_params: Current parameter values (optional)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting Genetic Algorithm optimization...")
        
        # Initialize population
        population = self.toolbox.population(n=self.config['population_size'])
        
        # If current parameters provided, seed some individuals
        if current_params:
            # Convert current params to individual format
            current_individual = [current_params.get(name, 0) for name in self.param_names]
            # Replace some random individuals with variations of current params
            for i in range(min(10, len(population))):
                for j, value in enumerate(current_individual):
                    # Add small random variation
                    bounds = list(self.parameter_bounds.values())[j]
                    variation = random.gauss(0, (bounds[1] - bounds[0]) * 0.05)
                    population[i][j] = max(bounds[0], min(bounds[1], value + variation))
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of fame (best individuals)
        hall_of_fame = tools.HallOfFame(self.config['elite_size'])
        
        # Evolution parameters
        cxpb = self.config['crossover_rate']
        mutpb = self.config['mutation_rate']
        ngen = self.config['generations']
        
        # Run evolution
        population, logbook = algorithms.eaSimple(
            population, self.toolbox, cxpb, mutpb, ngen,
            stats=stats, halloffame=hall_of_fame, verbose=verbose
        )
        
        # Extract best solution
        best_individual = hall_of_fame[0]
        best_params = dict(zip(self.param_names, best_individual))
        best_fitness = best_individual.fitness.values[0]
        
        # Predict SI for best solution
        predicted_si = self._predict_si(best_params)
        
        # Store optimization history
        optimization_result = {
            'optimized_parameters': best_params,
            'fitness': best_fitness,
            'predicted_si': predicted_si,
            'target_si': self.config['target_si'],
            'si_error': abs(predicted_si - self.config['target_si']),
            'method': 'Genetic Algorithm',
            'generations': ngen,
            'population_size': self.config['population_size'],
            'evolution_stats': logbook,
            'hall_of_fame': [dict(zip(self.param_names, ind)) for ind in hall_of_fame],
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"GA optimization completed. Best fitness: {best_fitness:.6f}")
        logger.info(f"Predicted SI: {predicted_si:.4f}, Target SI: {self.config['target_si']:.4f}")
        
        return optimization_result
    
    def multi_objective_optimize(self, objectives: List[str], 
                               weights: List[float] = None) -> Dict[str, Any]:
        """
        Multi-objective optimization using NSGA-II
        
        Args:
            objectives: List of objectives to optimize ['si_accuracy', 'stability', 'efficiency']
            weights: Weights for objectives (if None, equal weights)
            
        Returns:
            Pareto front solutions
        """
        if weights is None:
            weights = [-1.0] * len(objectives)  # Minimize all objectives
        
        # Create multi-objective fitness
        creator.create("FitnessMulti", base.Fitness, weights=tuple(weights))
        creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)
        
        # Setup multi-objective toolbox
        mo_toolbox = base.Toolbox()
        mo_toolbox.register("individual", self._create_individual)
        mo_toolbox.register("population", tools.initRepeat, list, mo_toolbox.individual)
        mo_toolbox.register("evaluate", lambda ind: self._evaluate_multi_objective(ind, objectives))
        mo_toolbox.register("mate", self._crossover)
        mo_toolbox.register("mutate", self._mutate)
        mo_toolbox.register("select", tools.selNSGA2)
        
        # Initialize population
        population = mo_toolbox.population(n=self.config['population_size'])
        
        # Run NSGA-II
        algorithms.eaMuPlusLambda(
            population, mo_toolbox, 
            mu=self.config['population_size'],
            lambda_=self.config['population_size'],
            cxpb=self.config['crossover_rate'],
            mutpb=self.config['mutation_rate'],
            ngen=self.config['generations'],
            verbose=True
        )
        
        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        # Convert to results format
        pareto_solutions = []
        for individual in pareto_front:
            params = dict(zip(self.param_names, individual))
            pareto_solutions.append({
                'parameters': params,
                'objectives': individual.fitness.values,
                'predicted_si': self._predict_si(params)
            })
        
        return {
            'pareto_front': pareto_solutions,
            'method': 'Multi-Objective GA (NSGA-II)',
            'objectives': objectives,
            'population_size': self.config['population_size'],
            'generations': self.config['generations']
        }
    
    def _evaluate_multi_objective(self, individual, objectives):
        """Evaluate individual for multiple objectives"""
        params = dict(zip(self.param_names, individual))
        predicted_si = self._predict_si(params)
        
        objective_values = []
        
        for objective in objectives:
            if objective == 'si_accuracy':
                # Minimize SI prediction error
                value = abs(predicted_si - self.config['target_si'])
            elif objective == 'stability':
                # Minimize parameter variation from stable ranges
                value = self._calculate_stability_penalty(params)
            elif objective == 'efficiency':
                # Minimize resource consumption (example: coke rate)
                value = params['coke_rate'] / 1000.0  # Normalized
            elif objective == 'safety':
                # Minimize constraint violations
                value = self._calculate_constraint_penalty(params)
            else:
                value = 0.0
            
            objective_values.append(value)
        
        return tuple(objective_values)
    
    def adaptive_optimize(self, current_params: Dict[str, float],
                         performance_history: List[Dict]) -> Dict[str, Any]:
        """
        Adaptive optimization that adjusts strategy based on performance history
        
        Args:
            current_params: Current parameter values
            performance_history: Historical performance data
            
        Returns:
            Adaptive optimization results
        """
        # Analyze performance trends
        if len(performance_history) > 10:
            recent_errors = [h.get('si_error', 0.1) for h in performance_history[-10:]]
            avg_error = np.mean(recent_errors)
            error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
            
            # Adapt parameters based on performance
            if avg_error > 0.05:  # High error, increase exploration
                self.config['mutation_rate'] = min(0.3, self.config['mutation_rate'] * 1.2)
                self.config['population_size'] = min(200, int(self.config['population_size'] * 1.1))
            elif error_trend > 0:  # Error increasing, change strategy
                self.config['crossover_rate'] = max(0.5, self.config['crossover_rate'] * 0.9)
            else:  # Good performance, fine-tune
                self.config['mutation_rate'] = max(0.1, self.config['mutation_rate'] * 0.9)
        
        # Run optimization with adapted parameters
        result = self.optimize(current_params, verbose=False)
        result['adaptation_info'] = {
            'adapted_mutation_rate': self.config['mutation_rate'],
            'adapted_population_size': self.config['population_size'],
            'adapted_crossover_rate': self.config['crossover_rate']
        }
        
        return result
    
    def get_parameter_sensitivity(self, base_params: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze parameter sensitivity using local search around base parameters
        
        Args:
            base_params: Base parameter set
            
        Returns:
            Dictionary with sensitivity scores for each parameter
        """
        sensitivities = {}
        base_si = self._predict_si(base_params)
        
        for param_name in self.param_names:
            bounds = self.parameter_bounds[param_name]
            current_value = base_params[param_name]
            
            # Small perturbation (5% of range)
            perturbation = (bounds[1] - bounds[0]) * 0.05
            
            # Test positive perturbation
            test_params = base_params.copy()
            test_params[param_name] = min(bounds[1], current_value + perturbation)
            si_plus = self._predict_si(test_params)
            
            # Test negative perturbation
            test_params[param_name] = max(bounds[0], current_value - perturbation)
            si_minus = self._predict_si(test_params)
            
            # Calculate sensitivity (change in SI per unit change in parameter)
            sensitivity = abs(si_plus - si_minus) / (2 * perturbation)
            sensitivities[param_name] = sensitivity
        
        return sensitivities
    
    def save_optimization_history(self, filepath: str):
        """Save optimization history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.optimization_history, f, indent=2, default=str)
    
    def load_optimization_history(self, filepath: str):
        """Load optimization history from file"""
        with open(filepath, 'r') as f:
            self.optimization_history = json.load(f) 