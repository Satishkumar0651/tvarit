# SI Prediction API - Technical Specification

## 🔧 API Overview

**Base URL**: `http://localhost:8000`  
**API Version**: `1.0.0`  
**Framework**: FastAPI  
**Documentation**: `http://localhost:8000/docs` (Interactive Swagger UI)

---

## 📡 Endpoints

### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "last_prediction": null,
  "uptime": "2025-06-29T22:50:28.808478"
}
```

### 2. Root Information
```http
GET /
```

**Response**:
```json
{
  "message": "Silicon (SI) Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### 3. SI Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body**:
```json
{
  "timestamp": "2025-06-29T10:30:00Z",  // Optional
  "oxygen_enrichment_rate": 4.5,
  "blast_furnace_permeability_index": 0.85,
  "enriching_oxygen_flow": 1200.0,
  "cold_blast_flow": 3500.0,
  "blast_momentum": 2.8,
  "blast_furnace_bosh_gas_volume": 4500.0,
  "blast_furnace_bosh_gas_index": 0.92,
  "theoretical_combustion_temperature": 2150.0,
  "top_gas_pressure": 1.8,
  "enriching_oxygen_pressure": 2.5,
  "cold_blast_pressure": 4.2,
  "total_pressure_drop": 1.5,
  "hot_blast_pressure": 4.5,
  "actual_blast_velocity": 280.0,
  "cold_blast_temperature": 25.0,
  "hot_blast_temperature": 1100.0,
  "top_temperature": 150.0,
  "blast_humidity": 12.0,
  "coal_injection_set_value": 140.0
}
```

**Response**:
```json
{
  "predicted_si": 0.636,
  "confidence_interval": {
    "lower": 0.616,
    "upper": 0.656
  },
  "is_anomaly": false,
  "anomaly_score": 0.0,
  "recommendations": [
    "High SI predicted (0.636). Consider:",
    "• Reducing blast temperature by 10-15°C",
    "• Increasing limestone addition by 5-10%",
    "• Monitor raw material composition",
    "• Process parameters appear stable - maintain current monitoring frequency"
  ],
  "prediction_timestamp": "2025-06-29T22:52:10.422972",
  "model_version": "1.0.0"
}
```

### 4. Model Information
```http
GET /model/info
```

**Response**:
```json
{
  "model_type": "LinearRegression",
  "version": "1.0.0",
  "features_count": 38,
  "training_date": "2025-06-29",
  "performance": {
    "r2_score": 0.9999,
    "rmse": 0.000356,
    "mape": 0.064
  }
}
```

### 5. Feature List
```http
GET /features
```

**Response**:
```json
{
  "features": [
    "oxygen_enrichment_rate",
    "blast_furnace_permeability_index",
    // ... all 38 features
  ]
}
```

---

## 📊 Data Models

### ProcessData (Input)
| Field | Type | Required | Description | Range |
|-------|------|----------|-------------|-------|
| `timestamp` | string | No | ISO 8601 timestamp | - |
| `oxygen_enrichment_rate` | float | Yes | Oxygen enrichment rate | 1.0-10.0 |
| `blast_furnace_permeability_index` | float | Yes | Permeability index | 0.1-2.0 |
| `enriching_oxygen_flow` | float | Yes | Oxygen flow rate | 500-2000 |
| `cold_blast_flow` | float | Yes | Cold blast flow | 2000-5000 |
| `blast_momentum` | float | Yes | Blast momentum | 1.0-5.0 |
| `blast_furnace_bosh_gas_volume` | float | Yes | Bosh gas volume | 3000-6000 |
| `blast_furnace_bosh_gas_index` | float | Yes | Bosh gas index | 0.5-1.5 |
| `theoretical_combustion_temperature` | float | Yes | Combustion temp | 1800-2500 |
| `top_gas_pressure` | float | Yes | Top gas pressure | 1.0-3.0 |
| `enriching_oxygen_pressure` | float | Yes | Oxygen pressure | 1.5-4.0 |
| `cold_blast_pressure` | float | Yes | Cold blast pressure | 3.0-6.0 |
| `total_pressure_drop` | float | Yes | Pressure drop | 0.5-3.0 |
| `hot_blast_pressure` | float | Yes | Hot blast pressure | 3.5-6.0 |
| `actual_blast_velocity` | float | Yes | Blast velocity | 150-400 |
| `cold_blast_temperature` | float | Yes | Cold blast temp | 15-35 |
| `hot_blast_temperature` | float | Yes | Hot blast temp | 900-1300 |
| `top_temperature` | float | Yes | Top temperature | 100-200 |
| `blast_humidity` | float | Yes | Blast humidity | 5-20 |
| `coal_injection_set_value` | float | Yes | Coal injection | 80-200 |

### PredictionResponse (Output)
| Field | Type | Description |
|-------|------|-------------|
| `predicted_si` | float | Predicted SI value (0.174-0.957) |
| `confidence_interval` | object | Lower and upper bounds |
| `is_anomaly` | boolean | Anomaly detection result |
| `anomaly_score` | float | Anomaly score (0.0-1.0) |
| `recommendations` | array | Operational recommendations |
| `prediction_timestamp` | string | ISO 8601 timestamp |
| `model_version` | string | Model version used |

---

## 🔒 Authentication & Security

### Current Implementation
- **No Authentication**: Open API for development/testing
- **Input Validation**: Pydantic models with type checking
- **Rate Limiting**: Not implemented (recommended for production)

### Production Recommendations
- **API Key Authentication**: Header-based authentication
- **Rate Limiting**: 100 requests/minute per client
- **HTTPS**: SSL/TLS encryption required
- **Input Sanitization**: Already implemented via Pydantic

---

## ⚡ Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Response Time** | <100ms | Typical: 30-70ms |
| **Throughput** | 200+ req/sec | Single instance |
| **Memory Usage** | ~150MB | Including models |
| **CPU Usage** | <5% | Per prediction |
| **Model Size** | <50MB | All models combined |

---

## 🐛 Error Handling

### HTTP Status Codes
| Code | Meaning | Description |
|------|---------|-------------|
| `200` | Success | Prediction completed successfully |
| `422` | Validation Error | Invalid input parameters |
| `500` | Server Error | Internal processing error |
| `503` | Service Unavailable | Model not loaded |

### Error Response Format
```json
{
  "detail": "Validation error description",
  "type": "validation_error",
  "errors": [
    {
      "field": "oxygen_enrichment_rate",
      "message": "Field required",
      "type": "missing"
    }
  ]
}
```

---

## 🧪 Testing Examples

### cURL Examples
```bash
# Health Check
curl -X GET "http://localhost:8000/health"

# Prediction Request
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "oxygen_enrichment_rate": 4.5,
    "blast_furnace_permeability_index": 0.85,
    "enriching_oxygen_flow": 1200,
    "cold_blast_flow": 3500,
    "blast_momentum": 2.8,
    "blast_furnace_bosh_gas_volume": 4500,
    "blast_furnace_bosh_gas_index": 0.92,
    "theoretical_combustion_temperature": 2150,
    "top_gas_pressure": 1.8,
    "enriching_oxygen_pressure": 2.5,
    "cold_blast_pressure": 4.2,
    "total_pressure_drop": 1.5,
    "hot_blast_pressure": 4.5,
    "actual_blast_velocity": 280,
    "cold_blast_temperature": 25,
    "hot_blast_temperature": 1100,
    "top_temperature": 150,
    "blast_humidity": 12,
    "coal_injection_set_value": 140
  }'
```

### Python Example
```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample data
data = {
    "oxygen_enrichment_rate": 4.5,
    "blast_furnace_permeability_index": 0.85,
    "enriching_oxygen_flow": 1200,
    "cold_blast_flow": 3500,
    "blast_momentum": 2.8,
    "blast_furnace_bosh_gas_volume": 4500,
    "blast_furnace_bosh_gas_index": 0.92,
    "theoretical_combustion_temperature": 2150,
    "top_gas_pressure": 1.8,
    "enriching_oxygen_pressure": 2.5,
    "cold_blast_pressure": 4.2,
    "total_pressure_drop": 1.5,
    "hot_blast_pressure": 4.5,
    "actual_blast_velocity": 280,
    "cold_blast_temperature": 25,
    "hot_blast_temperature": 1100,
    "top_temperature": 150,
    "blast_humidity": 12,
    "coal_injection_set_value": 140
}

# Make request
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Predicted SI: {result['predicted_si']}")
    print(f"Recommendations: {result['recommendations']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

---

## 🔄 Integration Guidelines

### Real-time Integration
1. **Polling**: Check `/health` endpoint before predictions
2. **Error Handling**: Implement retry logic with exponential backoff
3. **Caching**: Cache predictions for 1-5 minutes to reduce load
4. **Monitoring**: Track response times and error rates

### Batch Processing
- **Not Currently Supported**: API designed for real-time single predictions
- **Workaround**: Send multiple individual requests
- **Future Enhancement**: Batch endpoint planned for v2.0

### Data Pipeline Integration
```python
# Example integration pattern
def get_si_prediction(process_data):
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=process_data,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Implement fallback logic
        logger.error(f"SI prediction failed: {e}")
        return None
```

---

## 📈 Monitoring & Observability

### Recommended Metrics
- **Request Rate**: Requests per second
- **Response Time**: P50, P95, P99 percentiles
- **Error Rate**: 4xx and 5xx responses
- **Model Performance**: Prediction accuracy over time

### Health Monitoring
```bash
# Continuous health check
while true; do
  curl -s http://localhost:8000/health | jq '.status'
  sleep 30
done
```

---

## 🚀 Deployment Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=./models/best_model.pkl
CALIBRATION_ENABLED=true

# Performance Tuning
MAX_WORKERS=4
TIMEOUT_SECONDS=30
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "src.si_prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📚 Additional Resources

- **Interactive API Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Source Code**: See `src/si_prediction_api.py`
- **Model Details**: See `README.md` for methodology and performance

---

**API Version**: 1.0.0  
**Last Updated**: June 29, 2025  
**Maintainer**: SI Prediction Team 