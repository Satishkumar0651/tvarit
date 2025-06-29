#!/usr/bin/env python3
"""
Debug script to check API import status and variables
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_optimization_imports():
    """Check if optimization modules can be imported"""
    print("🔍 Checking optimization module imports...")
    
    try:
        from optimization.optimization_manager import OptimizationManager
        print("✅ OptimizationManager imported successfully")
    except ImportError as e:
        print(f"❌ OptimizationManager import failed: {e}")
        return False
    
    try:
        from optimization.reinforcement_learning import RLOptimizer
        print("✅ RLOptimizer imported successfully")
    except ImportError as e:
        print(f"❌ RLOptimizer import failed: {e}")
        return False
    
    try:
        from optimization.genetic_algorithm import GAOptimizer
        print("✅ GAOptimizer imported successfully")
    except ImportError as e:
        print(f"❌ GAOptimizer import failed: {e}")
        return False
    
    try:
        from optimization.bayesian_optimization import BayesianOptimizer
        print("✅ BayesianOptimizer imported successfully")
    except ImportError as e:
        print(f"❌ BayesianOptimizer import failed: {e}")
        return False
    
    return True

def check_api_imports():
    """Check if API module can be imported"""
    print("\n🔍 Checking API module imports...")
    
    try:
        # Import the API module to see what happens
        import si_prediction_api
        print("✅ si_prediction_api imported successfully")
        print(f"   OPTIMIZATION_AVAILABLE: {getattr(si_prediction_api, 'OPTIMIZATION_AVAILABLE', 'NOT_FOUND')}")
        print(f"   optimization_manager: {getattr(si_prediction_api, 'optimization_manager', 'NOT_FOUND')}")
        return True
    except Exception as e:
        print(f"❌ si_prediction_api import failed: {e}")
        return False

def main():
    print("🚀 API Debug Check")
    print("=" * 50)
    
    # Check current directory and path
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path includes: {[p for p in sys.path if 'tvarit' in p or 'si_prediction' in p]}")
    
    # Check if src directory exists
    src_path = os.path.join(os.getcwd(), 'src')
    print(f"src directory exists: {os.path.exists(src_path)}")
    
    # Check if optimization directory exists
    opt_path = os.path.join(src_path, 'optimization')
    print(f"src/optimization directory exists: {os.path.exists(opt_path)}")
    
    if os.path.exists(opt_path):
        print(f"Optimization modules: {os.listdir(opt_path)}")
    
    # Check imports
    optimization_imports_ok = check_optimization_imports()
    api_imports_ok = check_api_imports()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print(f"Optimization imports: {'✅ OK' if optimization_imports_ok else '❌ FAILED'}")
    print(f"API imports: {'✅ OK' if api_imports_ok else '❌ FAILED'}")
    
    if optimization_imports_ok and api_imports_ok:
        print("🎉 All imports successful - optimization should be available!")
    else:
        print("⚠️ Import issues detected - optimization may not be available")

if __name__ == "__main__":
    main() 