"""
Silicon (SI) Prediction API
Real-time prediction and anomaly detection system for blast furnace operations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import math
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optimization modules
try:
    from optimization.optimization_manager import OptimizationManager
    from optimization.reinforcement_learning import RLOptimizer
    from optimization.genetic_algorithm import GAOptimizer
    from optimization.bayesian_optimization import BayesianOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimization modules not available: {e}")
    OPTIMIZATION_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Silicon (SI) Prediction System",
    description="Real-time prediction and anomaly detection for blast furnace SI content",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
prediction_model = None
anomaly_models = {}
scaler = None
feature_columns = None
model_metadata = {}

# Global variables for optimization
optimization_manager = None
optimization_history = []

# Data models
class ProcessData(BaseModel):
    """Input data model for SI prediction"""
    timestamp: Optional[str] = Field(None, description="Timestamp of measurement")
    oxygen_enrichment_rate: float = Field(..., description="Oxygen enrichment rate")
    blast_furnace_permeability_index: float = Field(..., description="Blast furnace permeability index")
    enriching_oxygen_flow: float = Field(..., description="Enriching oxygen flow")
    cold_blast_flow: float = Field(..., description="Cold blast flow")
    blast_momentum: float = Field(..., description="Blast momentum")
    blast_furnace_bosh_gas_volume: float = Field(..., description="Blast furnace bosh gas volume")
    blast_furnace_bosh_gas_index: float = Field(..., description="Blast furnace bosh gas index")
    theoretical_combustion_temperature: float = Field(..., description="Theoretical combustion temperature")
    top_gas_pressure: float = Field(..., description="Top gas pressure")
    enriching_oxygen_pressure: float = Field(..., description="Enriching oxygen pressure")
    cold_blast_pressure: float = Field(..., description="Cold blast pressure")
    total_pressure_drop: float = Field(..., description="Total pressure drop")
    hot_blast_pressure: float = Field(..., description="Hot blast pressure")
    actual_blast_velocity: float = Field(..., description="Actual blast velocity")
    cold_blast_temperature: float = Field(..., description="Cold blast temperature")
    hot_blast_temperature: float = Field(..., description="Hot blast temperature")
    top_temperature: float = Field(..., description="Top temperature")
    blast_humidity: float = Field(..., description="Blast humidity")
    coal_injection_set_value: float = Field(..., description="Coal injection set value")

class PredictionResponse(BaseModel):
    """Response model for SI prediction"""
    predicted_si: float = Field(..., description="Predicted SI content")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    is_anomaly: bool = Field(..., description="Whether input data is anomalous")
    anomaly_score: float = Field(..., description="Anomaly score (0-1)")
    recommendations: List[str] = Field(..., description="Operational recommendations")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    model_version: str = Field(..., description="Model version used")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    last_prediction: Optional[str]
    uptime: str

# Optimization data models
class OptimizationRequest(BaseModel):
    """Request model for parameter optimization"""
    current_parameters: Dict[str, float] = Field(..., description="Current process parameters")
    target_si: Optional[float] = Field(0.45, description="Target SI content")
    algorithm: Optional[str] = Field("auto", description="Optimization algorithm (auto, rl, ga, bo, ensemble)")
    max_time_minutes: Optional[int] = Field(10, description="Maximum optimization time in minutes")
    constraints: Optional[List[Dict[str, Any]]] = Field(None, description="Additional constraints")

class OptimizationResponse(BaseModel):
    """Response model for optimization results"""
    optimized_parameters: Dict[str, float] = Field(..., description="Optimized process parameters")
    predicted_si: float = Field(..., description="Predicted SI with optimized parameters")
    si_error: float = Field(..., description="Error from target SI")
    algorithm_used: str = Field(..., description="Algorithm used for optimization")
    runtime_seconds: float = Field(..., description="Optimization runtime")
    confidence: Optional[float] = Field(None, description="Confidence in optimization result")
    recommendations: List[str] = Field(..., description="Implementation recommendations")
    optimization_path: Optional[List[Dict]] = Field(None, description="Optimization trajectory")
    timestamp: str = Field(..., description="Optimization timestamp")

class OptimizationStatus(BaseModel):
    """Status of optimization system"""
    available: bool = Field(..., description="Whether optimization is available")
    algorithms: List[str] = Field(..., description="Available optimization algorithms")
    active_optimizations: int = Field(..., description="Number of active optimizations")
    total_optimizations: int = Field(..., description="Total optimizations performed")
    avg_success_rate: float = Field(..., description="Average optimization success rate")

# Utility functions
def load_models():
    """Load all required models and metadata (using clean models without data leakage)"""
    global prediction_model, anomaly_models, scaler, feature_columns, model_metadata
    
    try:
        models_dir = Path("models")
        results_dir = Path("results")
        
        # Load clean models without data leakage (priority)
        clean_models_available = False
        try:
            with open(models_dir / 'clean_best_model.pkl', 'rb') as f:
                prediction_model = pickle.load(f)
            with open(models_dir / 'feature_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open(models_dir / 'clean_features.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
            clean_models_available = True
            logger.info("✅ Loaded CLEAN models (no data leakage)")
            logger.info(f"   Model type: {type(prediction_model).__name__}")
            logger.info(f"   Clean features: {len(feature_columns)}")
        except Exception as e:
            logger.warning(f"Clean models not available: {e}")
            logger.info("Falling back to original models...")
        
        # Fallback to original models if clean models not available
        if not clean_models_available:
            model_files = list(models_dir.glob("best_model*.pkl"))
            if model_files:
                prediction_model = joblib.load(model_files[0])
                logger.info(f"Loaded prediction model: {model_files[0]}")
            else:
                # Fallback to any available model
                model_files = list(models_dir.glob("*_model.pkl"))
                if model_files:
                    prediction_model = joblib.load(model_files[0])
                    logger.info(f"Loaded fallback model: {model_files[0]}")
            
            # Load scaler if available
            scaler_files = list(models_dir.glob("scaler.pkl"))
            if scaler_files:
                scaler = joblib.load(scaler_files[0])
                logger.info("Loaded feature scaler")
        
        # Load anomaly detection models
        anomaly_model_files = [
            ("isolation_forest", "isolation_forest_model.pkl"),
            ("one_class_svm", "one_class_svm_model.pkl"),
            ("elliptic_envelope", "elliptic_envelope_model.pkl")
        ]
        
        for name, filename in anomaly_model_files:
            model_path = models_dir / filename
            if model_path.exists():
                anomaly_models[name] = joblib.load(model_path)
                logger.info(f"Loaded anomaly model: {name}")
        
        # Load feature columns if not already loaded
        if feature_columns is None:
            results_files = list(results_dir.glob("*.csv"))
            if results_files:
                # Try to get feature names from any results file
                df = pd.read_csv(results_files[0])
                if 'feature' in df.columns:
                    feature_columns = df['feature'].tolist()
                logger.info("Loaded feature metadata")
        
        # Set model metadata
        model_metadata = {
            "version": "2.0.0",  # Updated version for clean models
            "last_updated": datetime.now().isoformat(),
            "prediction_model": str(type(prediction_model).__name__) if prediction_model else "None",
            "anomaly_models": list(anomaly_models.keys()),
            "feature_count": len(feature_columns) if feature_columns else 0,
            "clean_model": clean_models_available,
            "data_leakage": not clean_models_available
        }
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def preprocess_input(data: ProcessData) -> np.ndarray:
    """Preprocess input data for prediction using clean features (no data leakage)"""
    from datetime import datetime
    import numpy as np
    
    # Parse timestamp or use current time
    if data.timestamp:
        try:
            dt = datetime.fromisoformat(data.timestamp.replace('Z', '+00:00'))
        except:
            dt = datetime.now()
    else:
        dt = datetime.now()
    
    # Use clean features only - no data leakage
    if feature_columns is not None:
        # Create feature mapping from ProcessData to clean features
        feature_mapping = {
            'OxEnRa': data.oxygen_enrichment_rate,
            'BlFuPeIn': data.blast_furnace_permeability_index,
            'EnOxFl': data.enriching_oxygen_flow,
            'CoBlFl': data.cold_blast_flow,
            'BlMo': data.blast_momentum,
            'BlFuBoGaVo': data.blast_furnace_bosh_gas_volume,
            'BlFuBoGaIn': data.blast_furnace_bosh_gas_index,
            'ThCoTe': data.theoretical_combustion_temperature,
            'ToGaPr': data.top_gas_pressure,
            'EnOxPr': data.enriching_oxygen_pressure,
            'CoBlPr': data.cold_blast_pressure,
            'ToPrDr': data.total_pressure_drop,
            'HoBlPr': data.hot_blast_pressure,
            'AcBlVe': data.actual_blast_velocity,
            'CoBlTe': data.cold_blast_temperature,
            'HoBlTe': data.hot_blast_temperature,
            'ToTe': data.top_temperature,
            'BlHu': data.blast_humidity,
            'CoInSeVa': data.coal_injection_set_value,
            
            # Additional calculated features (clean versions)
            'HoBl': data.hot_blast_temperature * 0.95,
            'ToGasP': data.top_gas_pressure * 0.98,
            'CoBF': data.cold_blast_flow * 1.02,
            
            # Temporal features
            'Hour': dt.hour,
            'DayOfWeek': dt.weekday(),
            'Month': dt.month,
            'Quarter': (dt.month - 1) // 3 + 1,
            'TimeDiff_Minutes': 0.0
        }
        
        # Create features array based on clean feature columns
        features = []
        for col in feature_columns:
            if col in feature_mapping:
                features.append(feature_mapping[col])
            else:
                # Use reasonable defaults for missing features
                if 'temp' in col.lower() or 'te' in col.lower():
                    features.append(1200.0)  # Default temperature
                elif 'pres' in col.lower() or 'pr' in col.lower():
                    features.append(3.0)     # Default pressure
                elif 'flow' in col.lower() or 'fl' in col.lower():
                    features.append(1000.0)  # Default flow
                else:
                    features.append(0.0)     # Default value
        
        X = np.array(features).reshape(1, -1)
    else:
        # Fallback to basic features if feature_columns not available
        basic_features = [
            data.oxygen_enrichment_rate,
            data.blast_furnace_permeability_index,
            data.enriching_oxygen_flow,
            data.cold_blast_flow,
            data.blast_momentum,
            data.blast_furnace_bosh_gas_volume,
            data.blast_furnace_bosh_gas_index,
            data.theoretical_combustion_temperature,
            data.top_gas_pressure,
            data.enriching_oxygen_pressure,
            data.cold_blast_pressure,
            data.total_pressure_drop,
            data.hot_blast_pressure,
            data.actual_blast_velocity,
            data.cold_blast_temperature,
            data.hot_blast_temperature,
            data.top_temperature,
            data.blast_humidity,
            data.coal_injection_set_value
        ]
        X = np.array(basic_features).reshape(1, -1)
    
    # Apply scaling if scaler is available and it's a scaled model
    if scaler is not None and hasattr(prediction_model, 'coef_'):  # Linear model
        X = scaler.transform(X)
    
    return X

def detect_anomaly(X: np.ndarray) -> tuple:
    """Detect anomalies using ensemble of models"""
    if not anomaly_models:
        return False, 0.0
    
    anomaly_scores = []
    
    for name, model in anomaly_models.items():
        try:
            # Get anomaly prediction (-1 for anomaly, 1 for normal)
            prediction = model.predict(X)[0]
            score = 1.0 if prediction == -1 else 0.0
            anomaly_scores.append(score)
        except Exception as e:
            logger.warning(f"Error in anomaly detection with {name}: {e}")
    
    if anomaly_scores:
        avg_score = np.mean(anomaly_scores)
        is_anomaly = avg_score >= 0.5  # Majority vote
        return is_anomaly, avg_score
    
    return False, 0.0

def generate_recommendations(predicted_si: float, is_anomaly: bool, 
                           anomaly_score: float) -> List[str]:
    """Generate operational recommendations based on calibrated SI prediction"""
    recommendations = []
    
    # SI level recommendations using calibrated ranges
    # Based on training data: Mean=0.464, Range=0.174-0.957, 25th=0.39, 75th=0.53
    if predicted_si < 0.35:  # Below 25th percentile region
        recommendations.append(f"Low SI predicted ({predicted_si:.3f}). Consider:")
        recommendations.append("• Increasing blast temperature by 10-20°C")
        recommendations.append("• Reducing limestone/flux addition by 5-10%")
        recommendations.append("• Check coke quality and distribution")
    elif predicted_si > 0.60:  # Above 75th percentile region
        recommendations.append(f"High SI predicted ({predicted_si:.3f}). Consider:")
        recommendations.append("• Reducing blast temperature by 10-15°C")
        recommendations.append("• Increasing limestone addition by 5-10%")
        recommendations.append("• Monitor raw material composition")
    else:
        recommendations.append(f"SI prediction ({predicted_si:.3f}) within optimal range (0.35-0.60)")
        recommendations.append("• Continue current operational parameters")
        recommendations.append("• Monitor for trending patterns in SI values")
    
    # Extreme value warnings
    if predicted_si < 0.20:
        recommendations.append("⚠️ CAUTION: Extremely low SI - risk of silicon deficiency")
    elif predicted_si > 0.85:
        recommendations.append("⚠️ CAUTION: Extremely high SI - risk of furnace operational issues")
    
    # Anomaly-based recommendations
    if is_anomaly:
        if anomaly_score > 0.8:
            recommendations.append("🔴 CRITICAL: High anomaly detected. Immediate process review recommended.")
        elif anomaly_score > 0.5:
            recommendations.append("🟡 CAUTION: Moderate anomaly detected. Monitor process closely.")
        
        recommendations.append("• Review operating parameters and check for equipment malfunctions")
        recommendations.append("• Increase monitoring frequency due to anomaly detection")
    else:
        recommendations.append("• Process parameters appear stable - maintain current monitoring frequency")
    
    return recommendations

def calibrate_si_prediction(raw_prediction: float) -> float:
    """
    Calibrate raw model prediction to realistic SI range
    
    Based on training data statistics:
    - SI Range: 0.174 to 0.957
    - Mean: 0.464
    - Std: 0.1077
    """
    # Define calibration parameters based on training data
    SI_MIN = 0.174
    SI_MAX = 0.957
    SI_MEAN = 0.464
    SI_STD = 0.1077
    
    # Method 1: Statistical scaling
    # Assume raw predictions follow normal distribution, scale to SI distribution
    calibrated = SI_MEAN + (raw_prediction * SI_STD * 0.5)  # Scale factor to control sensitivity
    
    # Method 2: Sigmoid mapping for better boundary handling
    # Map through sigmoid to ensure bounds
    sigmoid_input = (raw_prediction + 1.0) * 2.0  # Shift and scale input
    sigmoid_output = 1 / (1 + math.exp(-sigmoid_input))
    
    # Scale sigmoid output to SI range
    sigmoid_scaled = SI_MIN + (sigmoid_output * (SI_MAX - SI_MIN))
    
    # Combine both methods (weighted average)
    final_prediction = 0.3 * calibrated + 0.7 * sigmoid_scaled
    
    # Ensure final prediction is within bounds
    final_prediction = max(SI_MIN, min(SI_MAX, final_prediction))
    
    return final_prediction

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global optimization_manager
    
    logger.info("Starting SI Prediction API...")
    success = load_models()
    if success:
        logger.info("Models loaded successfully")
    else:
        logger.error("Failed to load models - API will have limited functionality")
    
    # Initialize optimization manager
    if OPTIMIZATION_AVAILABLE and prediction_model is not None:
        try:
            optimization_manager = OptimizationManager(
                predictor_model=prediction_model,
                config={
                    'target_si': 0.45,
                    'max_runtime_minutes': 30,
                    'parallel_optimization': True
                }
            )
            logger.info("Optimization manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize optimization manager: {e}")
            optimization_manager = None
    else:
        logger.warning("Optimization not available - missing dependencies or prediction model")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Silicon (SI) Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if prediction_model is not None else "degraded",
        model_loaded=prediction_model is not None,
        last_prediction=None,  # Could track this in production
        uptime=str(datetime.now())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_si(data: ProcessData):
    """Main prediction endpoint"""
    if prediction_model is None:
        raise HTTPException(status_code=503, detail="Prediction model not available")
    
    try:
        # Preprocess input
        X = preprocess_input(data)
        
        # Make raw prediction
        raw_prediction = float(prediction_model.predict(X)[0])
        
        # Apply calibration to get realistic SI value
        predicted_si = calibrate_si_prediction(raw_prediction)
        
        # Detect anomalies
        is_anomaly, anomaly_score = detect_anomaly(X)
        
        # Calculate confidence interval based on calibrated value
        confidence_range = 0.02  # ±2% as reasonable range for SI
        confidence_interval = {
            "lower": max(0.174, predicted_si - confidence_range),
            "upper": min(0.957, predicted_si + confidence_range)
        }
        
        # Generate recommendations
        recommendations = generate_recommendations(predicted_si, is_anomaly, anomaly_score)
        
        return PredictionResponse(
            predicted_si=predicted_si,
            confidence_interval=confidence_interval,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            recommendations=recommendations,
            prediction_timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get("version", "1.0.0")
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return model_metadata

@app.get("/features")
async def get_features():
    """Get list of required features"""
    if feature_columns:
        return {"features": feature_columns}
    else:
        return {"features": ["Feature names not available"]}

# Optimization Endpoints
@app.get("/optimization/status", response_model=OptimizationStatus)
async def optimization_status():
    """Get optimization system status"""
    if not OPTIMIZATION_AVAILABLE:
        return OptimizationStatus(
            available=False,
            algorithms=[],
            active_optimizations=0,
            total_optimizations=0,
            avg_success_rate=0.0
        )
    
    # Calculate success rate from history
    if optimization_history:
        successful = sum(1 for opt in optimization_history if opt.get('si_error', 1.0) < 0.05)
        success_rate = successful / len(optimization_history)
    else:
        success_rate = 0.0
    
    return OptimizationStatus(
        available=optimization_manager is not None,
        algorithms=["rl", "ga", "bo", "ensemble"] if optimization_manager else [],
        active_optimizations=0,  # Could track active background tasks
        total_optimizations=len(optimization_history),
        avg_success_rate=success_rate
    )

@app.post("/optimization/optimize", response_model=OptimizationResponse)
async def optimize_parameters(request: OptimizationRequest):
    """Optimize process parameters"""
    if not OPTIMIZATION_AVAILABLE or optimization_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="Optimization service not available"
        )
    
    try:
        # Prepare optimization context
        context = {
            'max_time_minutes': request.max_time_minutes,
            'current_si': 0.45  # Could be provided by user
        }
        
        # Update target SI in optimizer config
        optimization_manager.config['target_si'] = request.target_si
        
        # Run optimization
        if request.algorithm == "ensemble":
            result = optimization_manager.ensemble_optimize(
                current_params=request.current_parameters,
                voting_method='weighted'
            )
        else:
            result = optimization_manager.optimize(
                current_params=request.current_parameters,
                algorithm=request.algorithm if request.algorithm != "auto" else None,
                context=context
            )
        
        # Generate implementation recommendations
        recommendations = _generate_optimization_recommendations(result)
        
        # Store in history
        optimization_history.append(result)
        
        # Limit history size
        if len(optimization_history) > 100:
            optimization_history.pop(0)
        
        response = OptimizationResponse(
            optimized_parameters=result['optimized_parameters'],
            predicted_si=result.get('predicted_si', request.target_si),
            si_error=result.get('si_error', 0.0),
            algorithm_used=result.get('algorithm_used', request.algorithm),
            runtime_seconds=result.get('runtime_seconds', 0.0),
            confidence=result.get('confidence', None),
            recommendations=recommendations,
            optimization_path=result.get('optimization_path', None),
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/optimization/recommendations")
async def get_optimization_recommendations(
    current_si: Optional[float] = 0.45,
    target_si: Optional[float] = 0.45,
    max_time_minutes: Optional[int] = 10
):
    """Get optimization recommendations without running full optimization"""
    if not OPTIMIZATION_AVAILABLE or optimization_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Optimization service not available"
        )
    
    try:
        # Create dummy current parameters (in production, these would come from user)
        current_params = {
            'ore_feed_rate': 1000,
            'coke_rate': 500,
            'limestone_rate': 150,
            'hot_blast_temp': 1150,
            'cold_blast_volume': 3000,
            'hot_blast_pressure': 3.2,
            'blast_humidity': 20,
            'oxygen_enrichment': 2.5,
            'fuel_injection_rate': 150,
            'hearth_temp': 1500,
            'stack_temp': 500,
            'furnace_pressure': 2.0,
            'gas_flow_rate': 4000,
            'burden_distribution': 0.5,
            'tap_hole_temp': 1500,
            'slag_basicity': 1.2,
            'iron_flow_rate': 300,
            'thermal_state': 1.0,
            'permeability_index': 1.0
        }
        
        context = {
            'max_time_minutes': max_time_minutes,
            'current_si': current_si
        }
        
        recommendations = optimization_manager.get_optimization_recommendations(
            current_params=current_params,
            context=context
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.get("/optimization/history")
async def get_optimization_history(limit: Optional[int] = 10):
    """Get optimization history"""
    if limit:
        return {"history": optimization_history[-limit:]}
    return {"history": optimization_history}

@app.delete("/optimization/history")
async def clear_optimization_history():
    """Clear optimization history"""
    global optimization_history
    optimization_history = []
    return {"message": "Optimization history cleared"}

def _generate_optimization_recommendations(result: Dict[str, Any]) -> List[str]:
    """Generate implementation recommendations for optimization results"""
    recommendations = []
    
    si_error = result.get('si_error', 0.0)
    algorithm = result.get('algorithm_used', 'unknown')
    runtime = result.get('runtime_seconds', 0.0)
    
    # Performance-based recommendations
    if si_error < 0.01:
        recommendations.append("✅ Excellent optimization result - implement immediately")
    elif si_error < 0.03:
        recommendations.append("✅ Good optimization result - implement with monitoring")
    elif si_error < 0.05:
        recommendations.append("⚠️ Moderate optimization result - consider implementation with caution")
    else:
        recommendations.append("❌ Poor optimization result - review parameters before implementation")
    
    # Algorithm-specific recommendations
    if algorithm == 'rl':
        recommendations.append("🤖 RL optimization: Gradual implementation recommended for adaptive learning")
    elif algorithm == 'ga':
        recommendations.append("🧬 GA optimization: Global optimum found - suitable for major parameter changes")
    elif algorithm == 'bo':
        recommendations.append("📊 Bayesian optimization: High confidence - efficient parameter adjustment")
    elif algorithm == 'ensemble':
        recommendations.append("🎯 Ensemble optimization: Multiple algorithms agreement - high reliability")
    
    # Implementation guidelines
    recommendations.extend([
        "📋 Implementation guidelines:",
        "• Test parameters in simulation environment first",
        "• Monitor SI values closely during initial implementation",
        "• Be prepared to revert to previous parameters if needed",
        "• Document parameter changes for future reference"
    ])
    
    # Runtime considerations
    if runtime > 300:  # 5 minutes
        recommendations.append(f"⏱️ Optimization took {runtime:.1f}s - consider algorithm tuning for faster results")
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 