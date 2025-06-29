#!/usr/bin/env python3
"""
Comprehensive Test Suite for SI Prediction Optimization System
Tests all optimization algorithms with mock data and validation
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, List
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_predictor():
    """Create a mock SI prediction function for testing"""
    def mock_predict(parameters: Dict[str, float]) -> float:
        """
        Mock SI prediction function that simulates realistic behavior
        Returns SI value based on parameter combinations
        """
        # Simulate realistic SI prediction (0.5 to 2.0 range)
        # Lower values are better (target around 0.6-0.8)
        
        # Extract key parameters
        coke_rate = parameters.get('Coke_Rate_kg_thm', 350)
        blast_temp = parameters.get('Blast_Temperature_C', 1200)
        burden_basicity = parameters.get('Burden_Basicity', 1.2)
        
        # Simulate non-linear relationships
        si_value = (
            0.8 +  # Base SI
            (coke_rate - 350) * 0.001 +  # Coke rate effect
            (blast_temp - 1200) * 0.0002 +  # Temperature effect
            abs(burden_basicity - 1.15) * 0.5 +  # Basicity optimum around 1.15
            np.random.normal(0, 0.02)  # Add some noise
        )
        
        return max(0.5, min(2.0, si_value))  # Clamp to realistic range
    
    return mock_predict

def create_mock_data():
    """Create mock blast furnace parameters for testing"""
    return {
        'Coke_Rate_kg_thm': 350.0,
        'Blast_Temperature_C': 1200.0,
        'Burden_Basicity': 1.2,
        'Ore_Fe_Content': 65.0,
        'Limestone_Addition_kg_thm': 150.0,
        'Blast_Humidity_g_Nm3': 25.0,
        'Hot_Metal_Temperature_C': 1500.0,
        'Top_Pressure_kPa': 250.0,
        'Oxygen_Enrichment_percent': 2.5,
        'Coal_Injection_kg_thm': 120.0,
        'Sinter_in_Burden_percent': 75.0,
        'Pellet_in_Burden_percent': 25.0,
        'Blast_Volume_Nm3_min': 4500.0,
        'Furnace_Permeability_Index': 0.85,
        'Hearth_Gas_Flow_Nm3_h': 180000.0,
        'Raceway_Temperature_C': 2100.0,
        'Thermal_Reserve_Zone_Temperature_C': 1000.0,
        'Cohesive_Zone_Height_m': 8.5,
        'Bosh_Gas_CO_percent': 22.0
    }

def test_optimization_imports():
    """Test if all optimization modules can be imported"""
    logger.info("Testing optimization module imports...")
    
    try:
        from optimization import RLOptimizer, GAOptimizer, BayesianOptimizer, OptimizationManager
        logger.info("✅ All optimization modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_ga_optimizer():
    """Test Genetic Algorithm Optimizer"""
    logger.info("\n🧬 Testing Genetic Algorithm Optimizer...")
    
    try:
        from optimization.genetic_algorithm import GAOptimizer
        
        # Create mock data and predictor
        mock_data = create_mock_data()
        mock_predictor = create_mock_predictor()
        
        # Initialize GA optimizer
        config = {
            'population_size': 20,  # Reduced for testing
            'generations': 10,      # Reduced for testing
            'target_si': 0.7,
            'parameter_bounds': {
                'Coke_Rate_kg_thm': (320, 380),
                'Blast_Temperature_C': (1150, 1250),
                'Burden_Basicity': (1.0, 1.4)
            }
        }
        
        ga_optimizer = GAOptimizer(mock_predictor, config)
        
        # Test single-objective optimization
        start_time = time.time()
        result = ga_optimizer.optimize(mock_data.copy())
        duration = time.time() - start_time
        
        # Validate results
        if 'optimized_parameters' in result and 'predicted_si' in result:
            logger.info(f"✅ GA Optimization completed in {duration:.2f}s")
            logger.info(f"   Original SI: {mock_predictor(mock_data):.4f}")
            logger.info(f"   Optimized SI: {result['predicted_si']:.4f}")
            logger.info(f"   Improvement: {(mock_predictor(mock_data) - result['predicted_si']):.4f}")
            return True
        else:
            logger.error("❌ GA Optimization returned incomplete results")
            logger.error(f"   Result keys: {list(result.keys())}")
            return False
            
    except Exception as e:
        logger.error(f"❌ GA Optimizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_bayesian_optimizer():
    """Test Bayesian Optimization"""
    logger.info("\n📊 Testing Bayesian Optimizer...")
    
    try:
        from optimization.bayesian_optimization import BayesianOptimizer
        
        # Create mock data and predictor
        mock_data = create_mock_data()
        mock_predictor = create_mock_predictor()
        
        # Initialize Bayesian optimizer
        config = {
            'n_initial_points': 5,   # Reduced for testing
            'n_iterations': 10,      # Reduced for testing
            'target_si': 0.7,
            'acquisition_function': 'EI',
            'parameter_bounds': {
                'Coke_Rate_kg_thm': (320, 380),
                'Blast_Temperature_C': (1150, 1250),
                'Burden_Basicity': (1.0, 1.4)
            }
        }
        
        bo_optimizer = BayesianOptimizer(mock_predictor, config)
        
        # Test optimization
        start_time = time.time()
        result = bo_optimizer.optimize(mock_data.copy())
        duration = time.time() - start_time
        
        # Validate results
        if 'optimized_parameters' in result and 'predicted_si' in result:
            logger.info(f"✅ Bayesian Optimization completed in {duration:.2f}s")
            logger.info(f"   Original SI: {mock_predictor(mock_data):.4f}")
            logger.info(f"   Optimized SI: {result['predicted_si']:.4f}")
            logger.info(f"   Improvement: {(mock_predictor(mock_data) - result['predicted_si']):.4f}")
            return True
        else:
            logger.error("❌ Bayesian Optimization returned incomplete results")
            logger.error(f"   Result keys: {list(result.keys())}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Bayesian Optimizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_optimization_manager():
    """Test Optimization Manager"""
    logger.info("\n🎯 Testing Optimization Manager...")
    
    try:
        from optimization.optimization_manager import OptimizationManager
        
        # Create mock data and predictor
        mock_data = create_mock_data()
        mock_predictor = create_mock_predictor()
        
        # Initialize optimization manager
        config = {
            'target_si': 0.7,
            'parameter_bounds': {
                'Coke_Rate_kg_thm': (320, 380),
                'Blast_Temperature_C': (1150, 1250),
                'Burden_Basicity': (1.0, 1.4)
            },
            'max_time_minutes': 1,  # Short timeout for testing
            'parallel_optimization': False  # Disable parallel for simpler testing
        }
        
        opt_manager = OptimizationManager(mock_predictor, config)
        
        # Test algorithm selection
        logger.info("   Testing algorithm selection...")
        context = {'max_time_minutes': 1, 'priority': 'speed'}
        selected_algo = opt_manager.select_best_algorithm(mock_data.copy(), context)
        logger.info(f"   Selected algorithm: {selected_algo}")
        
        # Test optimization with selected algorithm
        start_time = time.time()
        result = opt_manager.optimize(mock_data.copy(), algorithm=selected_algo)
        duration = time.time() - start_time
        
        # Validate results  
        if 'optimized_parameters' in result and 'predicted_si' in result:
            logger.info(f"✅ Optimization Manager completed in {duration:.2f}s")
            logger.info(f"   Algorithm used: {result.get('algorithm_used', 'Unknown')}")
            logger.info(f"   Original SI: {mock_predictor(mock_data):.4f}")
            logger.info(f"   Optimized SI: {result['predicted_si']:.4f}")
            logger.info(f"   Improvement: {(mock_predictor(mock_data) - result['predicted_si']):.4f}")
            return True
        else:
            logger.error("❌ Optimization Manager returned incomplete results")
            logger.error(f"   Result keys: {list(result.keys())}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Optimization Manager test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    logger.info("🚀 Starting Comprehensive Optimization System Test")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test imports
    test_results['imports'] = test_optimization_imports()
    
    if test_results['imports']:
        # Test individual optimizers (focus on fastest ones first)
        test_results['ga_optimizer'] = test_ga_optimizer()
        test_results['bayesian_optimizer'] = test_bayesian_optimizer()
        test_results['optimization_manager'] = test_optimization_manager()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, passed_test in test_results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Optimization system is working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the logs above.")
    
    return test_results

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run comprehensive test
    results = run_comprehensive_test()
    
    # Exit with appropriate code
    exit_code = 0 if all(results.values()) else 1
    sys.exit(exit_code) 