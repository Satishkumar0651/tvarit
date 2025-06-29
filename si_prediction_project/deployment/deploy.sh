#!/bin/bash

# Silicon (SI) Prediction System - Deployment Script
# This script automates the deployment of the SI prediction system

set -e  # Exit on any error

# Configuration
PROJECT_NAME="si-prediction-system"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log "Docker and Docker Compose are installed"
}

# Function to check if required files exist
check_files() {
    local required_files=(
        "../requirements.txt"
        "../src/si_prediction_api.py"
        "../models/"
        "../results/"
        "../data/"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -e "$file" ]]; then
            error "Required file/directory not found: $file"
            exit 1
        fi
    done
    
    log "All required files and directories found"
}

# Function to create environment file
create_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log "Creating environment file..."
        cat > "$ENV_FILE" << EOF
# SI Prediction System Environment Variables

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
LOG_LEVEL=INFO
MODEL_VERSION=1.0.0

# Database Configuration
POSTGRES_DB=si_prediction
POSTGRES_USER=si_user
POSTGRES_PASSWORD=si_password_2024
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123

# SSL Configuration (if using HTTPS)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Model Retraining Configuration
RETRAIN_SCHEDULE=0 2 * * 0
RETRAIN_ENABLED=true
MIN_SAMPLES_FOR_RETRAIN=1000

# Alert Configuration
ALERT_EMAIL=admin@company.com
ALERT_THRESHOLD_ANOMALY=0.8
ALERT_THRESHOLD_ACCURACY=0.8
EOF
        log "Environment file created: $ENV_FILE"
    else
        log "Environment file already exists: $ENV_FILE"
    fi
}

# Function to create necessary directories
create_directories() {
    local directories=(
        "logs"
        "ssl"
        "grafana/provisioning/dashboards"
        "grafana/provisioning/datasources"
        "prometheus"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log "Created necessary directories"
}

# Function to create Nginx configuration
create_nginx_config() {
    if [[ ! -f "nginx.conf" ]]; then
        log "Creating Nginx configuration..."
        cat > "nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream si_api {
        server si-prediction-api:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    server {
        listen 80;
        server_name localhost;
        
        # API routes
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://si_api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS headers
            add_header 'Access-Control-Allow-Origin' '*' always;
            add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range' always;
        }
        
        # Health check
        location /health {
            proxy_pass http://si_api/health;
            access_log off;
        }
        
        # Static files or dashboard
        location / {
            return 200 'SI Prediction System is running. API available at /api/';
            add_header Content-Type text/plain;
        }
    }
    
    # SSL configuration (uncomment if using HTTPS)
    # server {
    #     listen 443 ssl;
    #     server_name localhost;
    #     
    #     ssl_certificate /etc/nginx/ssl/cert.pem;
    #     ssl_certificate_key /etc/nginx/ssl/key.pem;
    #     
    #     location / {
    #         proxy_pass http://si_api;
    #         proxy_set_header Host $host;
    #         proxy_set_header X-Real-IP $remote_addr;
    #         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    #         proxy_set_header X-Forwarded-Proto $scheme;
    #     }
    # }
}
EOF
        log "Nginx configuration created"
    fi
}

# Function to create Prometheus configuration
create_prometheus_config() {
    if [[ ! -f "prometheus.yml" ]]; then
        log "Creating Prometheus configuration..."
        cat > "prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'si-prediction-api'
    static_configs:
      - targets: ['si-prediction-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
EOF
        log "Prometheus configuration created"
    fi
}

# Function to create database initialization script
create_db_init() {
    if [[ ! -f "init.sql" ]]; then
        log "Creating database initialization script..."
        cat > "init.sql" << 'EOF'
-- SI Prediction System Database Schema

-- Table for storing predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    input_data JSONB NOT NULL,
    predicted_si FLOAT NOT NULL,
    confidence_lower FLOAT,
    confidence_upper FLOAT,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_score FLOAT DEFAULT 0.0,
    actual_si FLOAT,
    model_version VARCHAR(50),
    processing_time_ms INTEGER
);

-- Table for storing anomalies
CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    prediction_id INTEGER REFERENCES predictions(id),
    anomaly_type VARCHAR(100),
    severity VARCHAR(20),
    description TEXT,
    root_causes JSONB,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Table for storing model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    data_period_start TIMESTAMP WITH TIME ZONE,
    data_period_end TIMESTAMP WITH TIME ZONE
);

-- Table for storing system alerts
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    alert_type VARCHAR(100),
    severity VARCHAR(20),
    message TEXT,
    metadata JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_is_anomaly ON predictions(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp);
CREATE INDEX IF NOT EXISTS idx_anomalies_resolved ON anomalies(resolved);
CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);

-- Create a view for recent predictions with anomalies
CREATE OR REPLACE VIEW recent_predictions_with_anomalies AS
SELECT 
    p.id,
    p.timestamp,
    p.predicted_si,
    p.is_anomaly,
    p.anomaly_score,
    p.actual_si,
    p.model_version,
    a.anomaly_type,
    a.severity,
    a.description
FROM predictions p
LEFT JOIN anomalies a ON p.id = a.prediction_id
WHERE p.timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY p.timestamp DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO si_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO si_user;
EOF
        log "Database initialization script created"
    fi
}

# Function to build and start services
deploy_services() {
    log "Building and starting services..."
    
    # Build the application
    docker-compose build --no-cache
    
    # Start services
    docker-compose up -d
    
    log "Services started successfully"
}

# Function to wait for services to be healthy
wait_for_services() {
    log "Waiting for services to be healthy..."
    
    local services=("si-prediction-api" "postgres" "redis")
    local max_attempts=30
    local attempt=1
    
    for service in "${services[@]}"; do
        log "Waiting for $service..."
        while [[ $attempt -le $max_attempts ]]; do
            if docker-compose ps | grep -q "$service.*healthy\|$service.*Up"; then
                log "$service is healthy"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                error "$service failed to become healthy"
                return 1
            fi
            
            sleep 5
            ((attempt++))
        done
        attempt=1
    done
    
    log "All services are healthy"
}

# Function to run basic tests
run_tests() {
    log "Running basic health checks..."
    
    # Test API health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "API health check passed"
    else
        error "API health check failed"
        return 1
    fi
    
    # Test database connection
    if docker-compose exec -T postgres pg_isready -U si_user -d si_prediction > /dev/null 2>&1; then
        log "Database connection test passed"
    else
        error "Database connection test failed"
        return 1
    fi
    
    # Test Redis connection
    if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
        log "Redis connection test passed"
    else
        error "Redis connection test failed"
        return 1
    fi
    
    log "All health checks passed"
}

# Function to display deployment information
show_deployment_info() {
    log "Deployment completed successfully!"
    echo
    info "Service URLs:"
    info "  SI Prediction API: http://localhost:8000"
    info "  API Documentation: http://localhost:8000/docs"
    info "  Grafana Dashboard: http://localhost:3000 (admin/admin123)"
    info "  Prometheus: http://localhost:9090"
    echo
    info "Useful commands:"
    info "  View logs: docker-compose logs -f"
    info "  Stop services: docker-compose down"
    info "  Restart services: docker-compose restart"
    info "  Update services: docker-compose pull && docker-compose up -d"
    echo
    info "For production deployment, consider:"
    info "  1. Change default passwords in $ENV_FILE"
    info "  2. Set up SSL certificates in ssl/ directory"
    info "  3. Configure backup strategies for data volumes"
    info "  4. Set up monitoring alerts"
    info "  5. Configure log rotation"
}

# Function to cleanup on failure
cleanup_on_failure() {
    error "Deployment failed. Cleaning up..."
    docker-compose down --remove-orphans
    exit 1
}

# Main deployment function
main() {
    log "Starting SI Prediction System deployment..."
    
    # Set up trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    # Run deployment steps
    check_docker
    check_files
    create_env_file
    create_directories
    create_nginx_config
    create_prometheus_config
    create_db_init
    deploy_services
    wait_for_services
    run_tests
    show_deployment_info
    
    log "Deployment completed successfully!"
}

# Command line argument handling
case "${1:-}" in
    "help"|"--help"|"-h")
        echo "SI Prediction System Deployment Script"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  deploy    Deploy the complete system (default)"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  logs      Show service logs"
        echo "  status    Show service status"
        echo "  clean     Clean up all resources"
        echo "  help      Show this help message"
        exit 0
        ;;
    "stop")
        log "Stopping services..."
        docker-compose down
        ;;
    "restart")
        log "Restarting services..."
        docker-compose restart
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        warning "This will remove all data and containers. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            log "Cleaning up all resources..."
            docker-compose down --volumes --remove-orphans
            docker system prune -f
            log "Cleanup completed"
        else
            log "Cleanup cancelled"
        fi
        ;;
    "deploy"|"")
        main
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac 