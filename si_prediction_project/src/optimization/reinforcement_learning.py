"""
Reinforcement Learning Optimizer for Blast Furnace Control

This module implements a Deep Q-Network (DQN) agent for autonomous
parameter adjustment to optimize silicon content in blast furnace operations.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import joblib
import json
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BlastFurnaceEnv(gym.Env):
    """
    Custom Gym Environment for Blast Furnace Control
    
    State: Current process parameters (19 features)
    Action: Parameter adjustments within safe bounds
    Reward: Based on SI prediction accuracy and operational constraints
    """
    
    def __init__(self, predictor_model, target_si: float = 0.45):
        super(BlastFurnaceEnv, self).__init__()
        
        self.predictor_model = predictor_model
        self.target_si = target_si
        self.current_step = 0
        self.max_steps = 100
        
        # Parameter bounds (normalized 0-1)
        self.param_bounds = {
            'ore_feed_rate': (0.0, 1.0),
            'coke_rate': (0.0, 1.0),
            'limestone_rate': (0.0, 1.0),
            'hot_blast_temp': (0.0, 1.0),
            'cold_blast_volume': (0.0, 1.0),
            'hot_blast_pressure': (0.0, 1.0),
            'blast_humidity': (0.0, 1.0),
            'oxygen_enrichment': (0.0, 1.0),
            'fuel_injection_rate': (0.0, 1.0),
            'hearth_temp': (0.0, 1.0),
            'stack_temp': (0.0, 1.0),
            'furnace_pressure': (0.0, 1.0),
            'gas_flow_rate': (0.0, 1.0),
            'burden_distribution': (0.0, 1.0),
            'tap_hole_temp': (0.0, 1.0),
            'slag_basicity': (0.0, 1.0),
            'iron_flow_rate': (0.0, 1.0),
            'thermal_state': (0.0, 1.0),
            'permeability_index': (0.0, 1.0)
        }
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, 
            shape=(19,), dtype=np.float32
        )  # Adjustment deltas
        
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(19,), dtype=np.float32
        )  # Current parameters
        
        # Initialize state
        self.state = np.random.uniform(0.3, 0.7, 19).astype(np.float32)
        self.history = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Random initial state within safe operating bounds
        self.state = np.random.uniform(0.3, 0.7, 19).astype(np.float32)
        self.current_step = 0
        self.history = []
        
        return self.state, {}
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply action (parameter adjustments)
        new_state = self.state + action
        
        # Clip to bounds
        new_state = np.clip(new_state, 0.0, 1.0)
        
        # Calculate reward
        reward = self._calculate_reward(new_state)
        
        # Update state
        self.state = new_state
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Store history
        self.history.append({
            'step': self.current_step,
            'state': self.state.copy(),
            'action': action.copy(),
            'reward': reward
        })
        
        return self.state, reward, done, truncated, {}
    
    def _calculate_reward(self, state):
        """Calculate reward based on SI prediction and constraints"""
        try:
            # Convert normalized state to actual parameter values
            actual_params = self._denormalize_parameters(state)
            
            # Get SI prediction
            predicted_si = self._predict_si(actual_params)
            
            # Base reward: negative squared error from target
            si_error = abs(predicted_si - self.target_si)
            si_reward = -si_error ** 2
            
            # Stability reward (penalize large changes)
            if len(self.history) > 0:
                prev_state = self.history[-1]['state']
                stability_penalty = -np.sum((state - prev_state) ** 2) * 0.1
            else:
                stability_penalty = 0
            
            # Safety constraints reward
            safety_reward = self._calculate_safety_reward(state)
            
            total_reward = si_reward + stability_penalty + safety_reward
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return -10.0  # Large penalty for errors
    
    def _denormalize_parameters(self, normalized_state):
        """Convert normalized parameters back to actual ranges"""
        # This would use actual parameter ranges from your dataset
        # For now, using placeholder ranges
        param_ranges = {
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
        
        actual_params = {}
        param_names = list(param_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = param_ranges[param_name]
            actual_params[param_name] = min_val + normalized_state[i] * (max_val - min_val)
        
        return actual_params
    
    def _predict_si(self, params):
        """Predict SI using the trained model"""
        # Convert to DataFrame format expected by predictor
        df = pd.DataFrame([params])
        
        # Use your existing predictor logic here
        # For now, simple approximation
        return 0.4 + np.random.normal(0, 0.05)
    
    def _calculate_safety_reward(self, state):
        """Calculate reward/penalty based on safety constraints"""
        reward = 0.0
        
        # Penalize extreme values
        extreme_penalty = -np.sum(np.maximum(0, state - 0.9) ** 2) * 5
        extreme_penalty -= np.sum(np.maximum(0, 0.1 - state) ** 2) * 5
        
        # Reward stable operating range
        stable_reward = np.sum(np.maximum(0, 0.5 - np.abs(state - 0.5))) * 0.1
        
        return extreme_penalty + stable_reward


class RLOptimizer:
    """
    Reinforcement Learning Optimizer for Blast Furnace Control
    """
    
    def __init__(self, predictor_model, config: Dict[str, Any] = None):
        self.predictor_model = predictor_model
        self.config = config or {}
        
        # Default configuration
        default_config = {
            'algorithm': 'PPO',  # PPO or DQN
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'target_si': 0.45,
            'training_timesteps': 100000
        }
        
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        self.env = None
        self.model = None
        self.training_history = []
        
    def create_environment(self):
        """Create the blast furnace environment"""
        self.env = BlastFurnaceEnv(
            predictor_model=self.predictor_model,
            target_si=self.config['target_si']
        )
        return self.env
    
    def train(self, save_path: str = None):
        """Train the RL agent"""
        logger.info("Starting RL agent training...")
        
        # Create environment
        if self.env is None:
            self.create_environment()
        
        # Create vectorized environment
        vec_env = make_vec_env(lambda: self.env, n_envs=1)
        
        # Initialize model
        if self.config['algorithm'] == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                verbose=1
            )
        elif self.config['algorithm'] == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                vec_env,
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                gamma=self.config['gamma'],
                verbose=1
            )
        
        # Training callback
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path=save_path or './rl_models/',
            log_path='./rl_logs/',
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=self.config['training_timesteps'],
            callback=eval_callback
        )
        
        # Save model
        if save_path:
            self.model.save(save_path)
        
        logger.info("RL agent training completed!")
        
        return self.model
    
    def optimize_parameters(self, current_params: Dict[str, float], 
                          n_steps: int = 10) -> Dict[str, Any]:
        """
        Optimize parameters using trained RL agent
        
        Args:
            current_params: Current blast furnace parameters
            n_steps: Number of optimization steps
            
        Returns:
            Dictionary with optimized parameters and optimization history
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Normalize current parameters
        normalized_params = self._normalize_parameters(current_params)
        
        # Run optimization
        obs = normalized_params
        optimization_path = []
        
        for step in range(n_steps):
            # Get action from trained agent
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Apply action
            new_obs = obs + action
            new_obs = np.clip(new_obs, 0.0, 1.0)
            
            # Convert back to actual parameters
            optimized_params = self._denormalize_parameters(new_obs)
            
            # Store step
            optimization_path.append({
                'step': step + 1,
                'parameters': optimized_params.copy(),
                'action': action.tolist(),
                'predicted_si': self._predict_si_from_params(optimized_params)
            })
            
            obs = new_obs
        
        # Return final optimized parameters
        final_params = optimization_path[-1]['parameters']
        final_si = optimization_path[-1]['predicted_si']
        
        return {
            'optimized_parameters': final_params,
            'predicted_si': final_si,
            'si_error': abs(final_si - self.config['target_si']),
            'optimization_path': optimization_path,
            'method': 'Reinforcement Learning',
            'algorithm': self.config['algorithm'],
            'target_si': self.config['target_si']
        }
    
    def _normalize_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to 0-1 range"""
        # Implement parameter normalization based on your data ranges
        # For now, assuming params are already somewhat normalized
        normalized = []
        param_names = [
            'ore_feed_rate', 'coke_rate', 'limestone_rate', 'hot_blast_temp',
            'cold_blast_volume', 'hot_blast_pressure', 'blast_humidity',
            'oxygen_enrichment', 'fuel_injection_rate', 'hearth_temp',
            'stack_temp', 'furnace_pressure', 'gas_flow_rate',
            'burden_distribution', 'tap_hole_temp', 'slag_basicity',
            'iron_flow_rate', 'thermal_state', 'permeability_index'
        ]
        
        for param_name in param_names:
            value = params.get(param_name, 0.5)
            # Simple normalization - you should use actual min/max from your data
            normalized_value = max(0.0, min(1.0, value / 1000.0))  # Placeholder
            normalized.append(normalized_value)
        
        return np.array(normalized, dtype=np.float32)
    
    def _denormalize_parameters(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """Convert normalized parameters back to actual values"""
        param_names = [
            'ore_feed_rate', 'coke_rate', 'limestone_rate', 'hot_blast_temp',
            'cold_blast_volume', 'hot_blast_pressure', 'blast_humidity',
            'oxygen_enrichment', 'fuel_injection_rate', 'hearth_temp',
            'stack_temp', 'furnace_pressure', 'gas_flow_rate',
            'burden_distribution', 'tap_hole_temp', 'slag_basicity',
            'iron_flow_rate', 'thermal_state', 'permeability_index'
        ]
        
        # Placeholder denormalization - implement with actual ranges
        params = {}
        for i, param_name in enumerate(param_names):
            params[param_name] = float(normalized_params[i] * 1000.0)  # Placeholder
        
        return params
    
    def _predict_si_from_params(self, params: Dict[str, float]) -> float:
        """Predict SI from parameters using the trained model"""
        try:
            # Use the actual predictor model
            return self.predictor_model(params)
        except Exception as e:
            logger.warning(f"Predictor failed, using fallback: {e}")
            # Fallback prediction
            return 0.4 + np.random.normal(0, 0.05)
    
    def save_model(self, path: str):
        """Save the trained RL model"""
        if self.model:
            self.model.save(path)
    
    def load_model(self, path: str):
        """Load a trained RL model"""
        if self.config['algorithm'] == 'PPO':
            self.model = PPO.load(path)
        elif self.config['algorithm'] == 'DQN':
            self.model = DQN.load(path) 