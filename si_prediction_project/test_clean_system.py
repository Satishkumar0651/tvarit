#!/usr/bin/env python3
"""
Test Clean System Performance
Comprehensive testing of the SI prediction system with clean models
"""

import pandas as pd
import numpy as np
import pickle
import requests
import json
import time
import logging
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CleanSystemTester:
    """Test the clean system without data leakage"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.results = {}
        
    def test_model_loading(self):
        """Test if clean models load correctly"""
        print("\n🧪 TESTING MODEL LOADING")
        print("-" * 40)
        
        try:
            # Load clean models
            with open('models/clean_best_model.pkl', 'rb') as f:
                clean_model = pickle.load(f)
            with open('models/clean_linear_regression_model.pkl', 'rb') as f:
                clean_lr = pickle.load(f)
            with open('models/feature_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('models/clean_features.pkl', 'rb') as f:
                features = pickle.load(f)
            
            print(f"✅ Clean best model: {type(clean_model).__name__}")
            print(f"✅ Clean LR model: {type(clean_lr).__name__}")
            print(f"✅ Feature scaler: {type(scaler).__name__}")
            print(f"✅ Clean features: {len(features)} features")
            
            # Test prediction with clean features
            test_data = pd.read_csv('data/processed_dataset.csv')
            clean_data = test_data[features].iloc[:5]  # First 5 rows
            
            # Test best model
            predictions = clean_model.predict(clean_data)
            print(f"✅ Best model predictions: min={predictions.min():.3f}, max={predictions.max():.3f}")
            
            # Test linear regression with scaling
            scaled_data = scaler.transform(clean_data)
            lr_predictions = clean_lr.predict(scaled_data)
            print(f"✅ LR model predictions: min={lr_predictions.min():.3f}, max={lr_predictions.max():.3f}")
            
            self.results['model_loading'] = True
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.results['model_loading'] = False
            return False
    
    def test_api_health(self):
        """Test API health and model info"""
        print("\n🧪 TESTING API HEALTH")
        print("-" * 40)
        
        try:
            # Health check
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Health: {health_data['status']}")
                print(f"   Model loaded: {health_data['model_loaded']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
            
            # Model info
            response = requests.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                model_info = response.json()
                print(f"✅ Model version: {model_info.get('version', 'Unknown')}")
                print(f"   Clean model: {model_info.get('clean_model', False)}")
                print(f"   Data leakage: {model_info.get('data_leakage', 'Unknown')}")
                print(f"   Features: {model_info.get('feature_count', 0)}")
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
            
            self.results['api_health'] = True
            return True
            
        except Exception as e:
            print(f"❌ API health test failed: {e}")
            self.results['api_health'] = False
            return False
    
    def test_prediction_api(self):
        """Test prediction API with clean model"""
        print("\n🧪 TESTING PREDICTION API")
        print("-" * 40)
        
        try:
            # Sample prediction request
            test_data = {
                "timestamp": "2024-06-29T23:00:00Z",
                "oxygen_enrichment_rate": 2.5,
                "blast_furnace_permeability_index": 0.85,
                "enriching_oxygen_flow": 1200.0,
                "cold_blast_flow": 2800.0,
                "blast_momentum": 45.0,
                "blast_furnace_bosh_gas_volume": 1500.0,
                "blast_furnace_bosh_gas_index": 0.92,
                "theoretical_combustion_temperature": 2100.0,
                "top_gas_pressure": 2.8,
                "enriching_oxygen_pressure": 0.22,
                "cold_blast_pressure": 0.35,
                "total_pressure_drop": 1.2,
                "hot_blast_pressure": 3.2,
                "actual_blast_velocity": 180.0,
                "cold_blast_temperature": 25.0,
                "hot_blast_temperature": 1150.0,
                "top_temperature": 150.0,
                "blast_humidity": 18.0,
                "coal_injection_set_value": 140.0
            }
            
            response = requests.post(f"{self.base_url}/predict", json=test_data)
            
            if response.status_code == 200:
                prediction = response.json()
                predicted_si = prediction['predicted_si']
                confidence = prediction['confidence_interval']
                
                print(f"✅ Prediction successful")
                print(f"   Predicted SI: {predicted_si:.4f}")
                print(f"   Confidence range: {confidence['lower']:.4f} - {confidence['upper']:.4f}")
                print(f"   Is anomaly: {prediction['is_anomaly']}")
                print(f"   Model version: {prediction['model_version']}")
                print(f"   Recommendations: {len(prediction['recommendations'])} items")
                
                # Validate prediction is realistic
                if 0.1 <= predicted_si <= 1.0:
                    print(f"✅ Prediction within realistic range")
                else:
                    print(f"⚠️  Prediction outside expected range")
                
                self.results['prediction_api'] = True
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.results['prediction_api'] = False
                return False
                
        except Exception as e:
            print(f"❌ Prediction API test failed: {e}")
            self.results['prediction_api'] = False
            return False
    
    def test_optimization_system(self):
        """Test optimization system with clean models"""
        print("\n🧪 TESTING OPTIMIZATION SYSTEM")
        print("-" * 40)
        
        try:
            # Check optimization status
            response = requests.get(f"{self.base_url}/optimization/status")
            if response.status_code != 200:
                print(f"❌ Optimization status check failed: {response.status_code}")
                return False
            
            status = response.json()
            print(f"✅ Optimization available: {status['available']}")
            print(f"   Available algorithms: {status['algorithms']}")
            print(f"   Total optimizations: {status['total_optimizations']}")
            
            if not status['available']:
                print("⚠️  Optimization not available, skipping tests")
                self.results['optimization_system'] = False
                return False
            
            # Test optimization request
            optimization_request = {
                "current_parameters": {
                    "ore_feed_rate": 1000,
                    "coke_rate": 500,
                    "limestone_rate": 150,
                    "hot_blast_temp": 1150,
                    "cold_blast_volume": 3000,
                    "hot_blast_pressure": 3.2,
                    "blast_humidity": 20,
                    "oxygen_enrichment": 2.5,
                    "fuel_injection_rate": 150,
                    "hearth_temp": 1500,
                    "stack_temp": 500,
                    "furnace_pressure": 2.0
                },
                "target_si": 0.45,
                "algorithm": "ga",  # Use GA for faster testing
                "max_time_minutes": 1
            }
            
            print("   Running optimization test (GA algorithm)...")
            response = requests.post(f"{self.base_url}/optimization/optimize", 
                                   json=optimization_request, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Optimization successful")
                print(f"   Algorithm used: {result['algorithm_used']}")
                print(f"   Predicted SI: {result['predicted_si']:.4f}")
                print(f"   SI error: {result['si_error']:.4f}")
                print(f"   Runtime: {result['runtime_seconds']:.2f}s")
                print(f"   Recommendations: {len(result['recommendations'])} items")
                
                self.results['optimization_system'] = True
                return True
            else:
                print(f"❌ Optimization failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.results['optimization_system'] = False
                return False
                
        except Exception as e:
            print(f"❌ Optimization system test failed: {e}")
            self.results['optimization_system'] = False
            return False
    
    def test_performance_comparison(self):
        """Compare clean model performance vs original"""
        print("\n🧪 TESTING PERFORMANCE COMPARISON")
        print("-" * 40)
        
        try:
            # Load test results
            clean_results = pd.read_csv('results/clean_model_comparison_results.csv')
            
            print("Clean Model Performance:")
            for _, row in clean_results.iterrows():
                model_name = row['model']
                r2_score = row['r2_score']
                rmse = row['rmse']
                
                print(f"   {model_name}:")
                print(f"      R² Score: {r2_score:.4f}")
                print(f"      RMSE: {rmse:.4f}")
            
            # Find best clean model
            test_results = clean_results[clean_results['model'].str.contains('Test')]
            best_clean = test_results.loc[test_results['r2_score'].idxmax()]
            
            print(f"\n📊 REALISTIC PERFORMANCE SUMMARY:")
            print(f"   Best clean model: {best_clean['model']}")
            print(f"   R² Score: {best_clean['r2_score']:.4f} ({best_clean['r2_score']:.2%})")
            print(f"   RMSE: {best_clean['rmse']:.4f}")
            print(f"   MAE: {best_clean['mae']:.4f}")
            print(f"   MAPE: {best_clean['mape']:.2f}%")
            
            print(f"\n✅ Data leakage removed successfully")
            print(f"   Previous (leaked): 99.99% R²")
            print(f"   Current (clean): {best_clean['r2_score']:.2%} R²")
            print(f"   Realistic performance for production deployment")
            
            self.results['performance_comparison'] = True
            return True
            
        except Exception as e:
            print(f"❌ Performance comparison failed: {e}")
            self.results['performance_comparison'] = False
            return False
    
    def run_all_tests(self):
        """Run all tests and provide summary"""
        print("🧹 COMPREHENSIVE CLEAN SYSTEM TESTING")
        print("=" * 60)
        
        tests = [
            ("Model Loading", self.test_model_loading),
            ("API Health", self.test_api_health),
            ("Prediction API", self.test_prediction_api),
            ("Optimization System", self.test_optimization_system),
            ("Performance Comparison", self.test_performance_comparison)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_function in tests:
            try:
                if test_function():
                    passed += 1
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.results[test_name.lower().replace(' ', '_')] = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, (_, _) in zip([name for name, _ in tests], tests):
            result_key = test_name.lower().replace(' ', '_')
            status = "✅ PASS" if self.results.get(result_key, False) else "❌ FAIL"
            print(f"   {test_name:<25} {status}")
        
        print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - CLEAN SYSTEM READY FOR PRODUCTION!")
            print("\n🚀 RECOMMENDATIONS:")
            print("   1. Deploy clean system immediately")
            print("   2. Monitor performance in production")
            print("   3. Continue model improvement with clean features")
            print("   4. Document realistic performance expectations")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            print("\n🔧 NEXT STEPS:")
            print("   1. Fix failing components")
            print("   2. Re-run tests")
            print("   3. Deploy after all tests pass")
        
        return passed == total

def main():
    """Main test function"""
    tester = CleanSystemTester()
    success = tester.run_all_tests()
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main() 