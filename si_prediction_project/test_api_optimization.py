#!/usr/bin/env python3
"""
API Optimization Endpoints Test Script
Tests the FastAPI optimization endpoints with different algorithms
"""

import requests
import json
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def wait_for_api():
    """Wait for API to be ready"""
    logger.info("Waiting for API to start...")
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
        logger.info(f"   Attempt {attempt + 1}/{max_attempts}...")
    
    logger.error("❌ API failed to start")
    return False

def test_optimization_status():
    """Test optimization status endpoint"""
    logger.info("\n🔍 Testing Optimization Status Endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/optimization/status", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Optimization Status - SUCCESS")
            logger.info(f"   Available algorithms: {data.get('available_algorithms', [])}")
            logger.info(f"   System ready: {data.get('system_ready', False)}")
            return True
        else:
            logger.error(f"❌ Status check failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        return False

def test_optimization_recommendations():
    """Test optimization recommendations endpoint"""
    logger.info("\n💡 Testing Optimization Recommendations...")
    
    # Sample blast furnace parameters
    test_params = {
        "ore_feed_rate": 1000.0,
        "coke_rate": 500.0,
        "limestone_rate": 150.0,
        "hot_blast_temp": 1200.0,
        "cold_blast_volume": 3000.0,
        "hot_blast_pressure": 3.0,
        "blast_humidity": 20.0,
        "oxygen_enrichment": 2.0,
        "fuel_injection_rate": 150.0,
        "hearth_temp": 1500.0,
        "stack_temp": 500.0,
        "furnace_pressure": 2.0,
        "gas_flow_rate": 4000.0,
        "burden_distribution": 0.5,
        "tap_hole_temp": 1500.0,
        "slag_basicity": 1.2,
        "iron_flow_rate": 300.0,
        "thermal_state": 1.0,
        "permeability_index": 1.0
    }
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/optimization/recommendations",
            params=test_params,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Recommendations - SUCCESS")
            logger.info(f"   Recommended algorithm: {data.get('recommended_algorithm', 'Unknown')}")
            logger.info(f"   Confidence: {data.get('algorithm_confidence', 0):.2f}")
            return True
        else:
            logger.error(f"❌ Recommendations failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Recommendations error: {e}")
        return False

def test_optimization_algorithm(algorithm: str):
    """Test optimization with specific algorithm"""
    logger.info(f"\n🧠 Testing {algorithm.upper()} Optimization...")
    
    # Sample optimization request
    optimization_request = {
        "current_parameters": {
            "ore_feed_rate": 1000.0,
            "coke_rate": 500.0,
            "limestone_rate": 150.0,
            "hot_blast_temp": 1200.0,
            "cold_blast_volume": 3000.0,
            "hot_blast_pressure": 3.0,
            "blast_humidity": 20.0,
            "oxygen_enrichment": 2.0,
            "fuel_injection_rate": 150.0,
            "hearth_temp": 1500.0,
            "stack_temp": 500.0,
            "furnace_pressure": 2.0,
            "gas_flow_rate": 4000.0,
            "burden_distribution": 0.5,
            "tap_hole_temp": 1500.0,
            "slag_basicity": 1.2,
            "iron_flow_rate": 300.0,
            "thermal_state": 1.0,
            "permeability_index": 1.0
        },
        "target_si": 0.45,
        "algorithm": algorithm,
        "max_time_minutes": 2
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/optimization/optimize",
            json=optimization_request,
            timeout=TIMEOUT * 3  # Longer timeout for optimization
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ {algorithm.upper()} Optimization - SUCCESS")
            logger.info(f"   Runtime: {end_time - start_time:.2f}s")
            logger.info(f"   Algorithm used: {data.get('algorithm_used', 'Unknown')}")
            
            if 'predicted_si' in data:
                logger.info(f"   Predicted SI: {data['predicted_si']:.4f}")
                logger.info(f"   Target SI: {data.get('target_si', 'N/A')}")
                
                if 'si_error' in data:
                    logger.info(f"   SI Error: {data['si_error']:.4f}")
            
            if 'recommendations' in data:
                recs = data['recommendations']
                logger.info(f"   Implementation suggestions: {len(recs)} provided")
            
            return True
        else:
            logger.error(f"❌ {algorithm.upper()} Optimization failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {algorithm.upper()} Optimization error: {e}")
        return False

def test_optimization_history():
    """Test optimization history endpoint"""
    logger.info("\n📊 Testing Optimization History...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/optimization/history", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Optimization History - SUCCESS")
            logger.info(f"   History entries: {len(data.get('history', []))}")
            
            if data.get('history'):
                latest = data['history'][-1]
                logger.info(f"   Latest optimization: {latest.get('algorithm_used', 'Unknown')} algorithm")
            
            return True
        else:
            logger.error(f"❌ History check failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ History check error: {e}")
        return False

def main():
    """Run comprehensive API optimization tests"""
    logger.info("🚀 Starting API Optimization Endpoint Tests")
    logger.info("=" * 60)
    
    # Wait for API to be ready
    if not wait_for_api():
        return
    
    # Test results tracking
    tests = []
    
    # Run tests
    tests.append(("Status Check", test_optimization_status()))
    tests.append(("Recommendations", test_optimization_recommendations()))
    tests.append(("GA Optimization", test_optimization_algorithm("ga")))
    tests.append(("Bayesian Optimization", test_optimization_algorithm("bo")))
    tests.append(("Auto Selection", test_optimization_algorithm("auto")))
    tests.append(("History Check", test_optimization_history()))
    
    # RL might take too long, so make it optional
    logger.info("\n🤖 Testing RL Optimization (this may take longer)...")
    tests.append(("RL Optimization", test_optimization_algorithm("rl")))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall Result: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 ALL TESTS PASSED! Optimization API is working correctly.")
    else:
        logger.warning(f"⚠️  {len(tests) - passed} tests failed. Please check the logs above.")

if __name__ == "__main__":
    main() 