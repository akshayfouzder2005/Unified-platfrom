# 🚀 Ocean-Bio Platform Deployment Guide

## Complete Deployment Instructions for Production

This guide provides step-by-step instructions for deploying the Ocean-Bio Platform in various environments.

---

## 📋 Prerequisites

### System Requirements
- **Docker** 24.0+ and **Docker Compose** v2
- **Python** 3.11+ (for local development)
- **PostgreSQL** 14+ with PostGIS extension (if not using Docker)
- **Node.js** 18+ (for frontend development)
- **Git** for version control

### Hardware Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection

---

## 🐳 Option A: Docker Deployment (Recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/akshayfouzder2005/Unified-platfrom.git
cd Unified-platfrom
```

### Step 2: Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables (IMPORTANT!)
nano .env  # or use your preferred editor
```

**Required Environment Variables:**
```env
# Database Configuration
POSTGRES_USER=oceanbio
POSTGRES_PASSWORD=your-secure-password-here
POSTGRES_DB=oceanbio

# Security (CRITICAL - Change these!)
SECRET_KEY=your-super-secret-key-minimum-32-characters-long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
APP_NAME=Ocean-Bio Platform
ENVIRONMENT=production
```

### Step 3: Deploy with Docker Compose
```bash
# Build and start all services
docker compose up --build -d

# Check service status
docker compose ps

# View logs
docker compose logs -f api
```

### Step 4: Initialize Database
```bash
# Run database migrations
docker compose exec api alembic upgrade head

# Optional: Load sample data
docker compose exec api python scripts/init_phase1.py
```

### Step 5: Verify Deployment
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

---

## 💻 Option B: Local Development Deployment

### Step 1: Clone and Setup
```bash
git clone https://github.com/akshayfouzder2005/Unified-platfrom.git
cd Unified-platfrom
```

### Step 2: Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Database Setup (PostgreSQL)
```bash
# Install PostgreSQL with PostGIS
sudo apt-get install postgresql postgresql-contrib postgis  # Ubuntu/Debian
# OR use Docker for database only:
docker run --name oceanbio-db -e POSTGRES_PASSWORD=oceanbio123 -d -p 5432:5432 postgis/postgis:16-3.4
```

### Step 4: Environment Configuration
```bash
# Create environment file
cp .env.example .env

# Edit with your database credentials
DATABASE_URL=postgresql://username:password@localhost:5432/oceanbio
SECRET_KEY=your-super-secret-key-change-this
```

### Step 5: Database Migration
```bash
# Run migrations
alembic upgrade head

# Optional: Initialize with sample data
python scripts/init_phase1.py
```

### Step 6: Start API Server
```bash
# Development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server (with Gunicorn)
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## ☁️ Option C: Cloud Deployment

### AWS Deployment with ECS

1. **Prepare Docker Image**
```bash
# Build production image
docker build -t oceanbio-platform:latest ./backend

# Tag for AWS ECR
docker tag oceanbio-platform:latest <account-id>.dkr.ecr.<region>.amazonaws.com/oceanbio:latest

# Push to ECR
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/oceanbio:latest
```

2. **Database Setup**
- Create AWS RDS PostgreSQL instance with PostGIS
- Configure security groups and VPC
- Update environment variables with RDS endpoint

3. **ECS Service Configuration**
- Create ECS cluster
- Define task definition with environment variables
- Configure load balancer and auto-scaling

### Google Cloud Platform (GCP)

1. **Cloud SQL Setup**
```bash
# Create PostgreSQL instance
gcloud sql instances create oceanbio-db --database-version=POSTGRES_14 --region=us-central1

# Create database
gcloud sql databases create oceanbio --instance=oceanbio-db
```

2. **Cloud Run Deployment**
```bash
# Build and deploy
gcloud run deploy oceanbio-api --source ./backend --region us-central1 --allow-unauthenticated
```

---

## 🔧 Production Configuration

### Environment Variables
```env
# Production Database
DATABASE_URL=postgresql://user:password@prod-db-host:5432/oceanbio

# Security (CRITICAL)
SECRET_KEY=<64-character-random-string>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Optional: Redis for Caching
REDIS_URL=redis://redis-host:6379

# Optional: External APIs
NCBI_API_KEY=your-ncbi-api-key
GBIF_API_KEY=your-gbif-api-key
```

### SSL/TLS Configuration
```yaml
# docker-compose.yml with Nginx SSL
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - api
```

### Monitoring and Logging
```yaml
# Add to docker-compose.yml
  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
```

---

## 🧪 Testing Deployment

### Health Checks
```bash
# API Health
curl http://localhost:8000/api/health

# Database Connection
curl http://localhost:8000/api/taxonomy/species?limit=1

# Authentication
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test"}'
```

### Load Testing
```bash
# Install Artillery
npm install -g artillery

# Run load tests
artillery run load-test.yml
```

---

## 📊 Performance Optimization

### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_species_scientific_name ON species(scientific_name);
CREATE INDEX CONCURRENTLY idx_catch_records_date ON catch_records(catch_date);
```

### API Optimization
```python
# Enable Redis caching
REDIS_URL=redis://localhost:6379

# Configure connection pooling
DATABASE_URL=postgresql://user:pass@host:5432/db?pool_size=20&max_overflow=30
```

---

## 🔒 Security Checklist

- [ ] **Change default passwords** in `.env`
- [ ] **Generate strong SECRET_KEY** (64+ characters)
- [ ] **Enable HTTPS/SSL** in production
- [ ] **Configure firewall** (only ports 80, 443, 22)
- [ ] **Regular security updates** for dependencies
- [ ] **Database connection encryption**
- [ ] **API rate limiting** enabled
- [ ] **CORS configuration** for production domains
- [ ] **Log monitoring** and alerting
- [ ] **Backup strategy** implemented

---

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Errors**
```bash
# Check database status
docker compose logs db
# Verify credentials in .env
```

2. **Import Errors**
```bash
# Missing dependencies (expected warnings)
⚠️ GISManager import failed: No module named 'geopandas'
⚠️ Prophet not available. Time-series forecasting will use ARIMA only.

# These are normal - services gracefully degrade
```

3. **Port Conflicts**
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Use different external port
```

4. **Memory Issues**
```bash
# Increase Docker memory limit
docker system prune -a
# Or add memory limits in docker-compose.yml
```

---

## 📱 API Endpoints

### Core Endpoints
- `GET /api/health` - System health check
- `POST /auth/login` - User authentication
- `GET /docs` - Interactive API documentation

### Data Management
- `GET /api/taxonomy/species` - Species data
- `POST /api/ingestion/taxonomy/upload-csv` - Data upload
- `GET /api/taxonomy/stats` - Statistics

### Advanced Features (Phase 2)
- `POST /api/geospatial/query/location` - Spatial queries
- `POST /api/predictive/forecast/generate` - Predictive modeling
- `POST /api/genomics/sequences/classify` - Genomic analysis

---

## 📞 Support

### Getting Help
- **Documentation**: `/docs` endpoint
- **GitHub Issues**: [Create Issue](https://github.com/akshayfouzder2005/Unified-platfrom/issues)
- **Logs**: `docker compose logs` for troubleshooting

### Maintenance
- **Updates**: `git pull && docker compose up --build -d`
- **Backups**: Regular database dumps and volume backups
- **Monitoring**: Check health endpoints regularly

---

## 🎉 Success!

Once deployed, your Ocean-Bio Platform will provide:
- **🔬 Advanced Marine Research Tools**
- **📊 Real-time Analytics Dashboard**
- **🧬 Genomic Analysis Pipeline**
- **🗺️ Geospatial Data Processing**
- **📈 Predictive Modeling Capabilities**
- **🎯 117 API Endpoints** ready for marine research

**Welcome to the future of marine data management!** 🌊🚀