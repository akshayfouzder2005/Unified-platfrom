# 🌊 Ocean-Bio Phase 2 Deployment Guide

## Advanced Marine Data Platform - Complete Deployment Documentation

**Version**: 2.0.0  
**Last Updated**: September 2024  
**Target Environment**: Production-ready deployment of all Phase 2 features  

---

## 📋 **Overview**

Ocean-Bio Phase 2 introduces three major advanced capabilities:
- 🗺️ **Geospatial Analysis** - GIS integration, spatial analysis, and interactive mapping
- 📈 **Predictive Modeling** - Stock assessment, forecasting, and ML-driven analytics  
- 🧬 **Advanced eDNA Pipeline** - Genomic analysis, phylogenetics, and biodiversity metrics

This guide provides complete deployment instructions, system requirements, and operational procedures.

---

## 🏗️ **System Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    Ocean-Bio Phase 2                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Geospatial    │   Predictive    │    Genomics & eDNA      │
│   Analysis      │   Modeling      │    Pipeline             │
├─────────────────┼─────────────────┼─────────────────────────┤
│ • GIS Service   │ • Stock Models  │ • Sequence Processing   │
│ • Spatial Query │ • Forecasting   │ • Taxonomic Classifier  │
│ • Mapping       │ • Trend Analysis│ • Diversity Analysis    │
│ • PostGIS       │ • ML Models     │ • Phylogenetics        │
└─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 🔧 **System Requirements**

### **Hardware Requirements**

**Minimum Requirements:**
- **CPU**: 8 cores (Intel i7/AMD Ryzen 7 equivalent)
- **RAM**: 32GB DDR4
- **Storage**: 500GB SSD + 2TB HDD for data
- **Network**: 1Gbps connection

**Recommended for Production:**
- **CPU**: 16+ cores (Intel Xeon/AMD EPYC)
- **RAM**: 64GB+ DDR4/DDR5
- **Storage**: 1TB NVMe SSD + 5TB+ enterprise HDD
- **Network**: 10Gbps connection
- **GPU**: NVIDIA GPU with CUDA support (for ML acceleration)

### **Software Requirements**

**Operating System:**
- Ubuntu 20.04+ LTS (recommended)
- CentOS 8+ / RHEL 8+
- Windows Server 2019+ (with WSL2)
- macOS 12+ (development only)

**Core Dependencies:**
- **Python**: 3.9-3.11
- **Node.js**: 18+ LTS
- **PostgreSQL**: 14+ with PostGIS 3.2+
- **Redis**: 7.0+
- **Docker**: 24.0+ with Docker Compose
- **Git**: 2.30+

**External Tools:**
- **BLAST+**: 2.13+ (for genomic analysis)
- **MUSCLE**: 5.1+ (for sequence alignment)
- **RAxML**: 8.2+ (for phylogenetics)
- **GDAL/OGR**: 3.5+ (for geospatial data)

---

## 🚀 **Quick Deployment**

### **1. Clone and Setup**

```bash
# Clone the repository
git clone https://github.com/your-org/ocean-bio-platform.git
cd ocean-bio-platform/Ocean-bio

# Make deployment scripts executable
chmod +x scripts/*.sh

# Run quick setup (installs all dependencies)
./scripts/quick_deploy.sh
```

### **2. Environment Configuration**

```bash
# Copy and configure environment files
cp .env.example .env.production
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Edit configuration files
nano .env.production
```

### **3. Database Setup**

```bash
# Initialize PostgreSQL with PostGIS
./scripts/setup_database.sh

# Run migrations
./scripts/migrate_database.sh
```

### **4. Start Services**

```bash
# Start all services with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Or start individually
./scripts/start_backend.sh
./scripts/start_frontend.sh
```

---

## 📦 **Detailed Deployment Steps**

### **Step 1: Infrastructure Preparation**

#### **Database Setup (PostgreSQL + PostGIS)**

```bash
# Install PostgreSQL and PostGIS
sudo apt update
sudo apt install postgresql-14 postgresql-client-14 postgresql-contrib-14
sudo apt install postgresql-14-postgis-3 postgresql-14-postgis-3-scripts

# Enable PostGIS extension
sudo -u postgres psql -c "CREATE DATABASE oceanbio_db;"
sudo -u postgres psql -d oceanbio_db -c "CREATE EXTENSION postgis;"
sudo -u postgres psql -d oceanbio_db -c "CREATE EXTENSION postgis_topology;"
sudo -u postgres psql -d oceanbio_db -c "CREATE EXTENSION postgis_raster;"

# Create application user
sudo -u postgres psql -c "CREATE USER oceanbio_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE oceanbio_db TO oceanbio_user;"
```

#### **Redis Setup**

```bash
# Install Redis
sudo apt install redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf
# Set: maxmemory 4gb, maxmemory-policy allkeys-lru

# Start and enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### **External Tools Installation**

```bash
# Install BLAST+
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.13.0+-x64-linux.tar.gz
tar -xzf ncbi-blast-2.13.0+-x64-linux.tar.gz
sudo mv ncbi-blast-2.13.0+ /opt/blast
echo 'export PATH=/opt/blast/bin:$PATH' >> ~/.bashrc

# Install MUSCLE
wget https://github.com/rcedgar/muscle/releases/download/v5.1/muscle5.1.linux_intel64
chmod +x muscle5.1.linux_intel64
sudo mv muscle5.1.linux_intel64 /usr/local/bin/muscle

# Install GDAL
sudo apt install gdal-bin libgdal-dev
```

### **Step 2: Application Deployment**

#### **Backend Setup**

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt

# Run database migrations
alembic upgrade head

# Create initial data
python scripts/initialize_data.py
```

#### **Frontend Setup**

```bash
cd frontend

# Install Node.js dependencies
npm install

# Build production assets
npm run build

# Configure web server (Nginx)
sudo cp nginx.conf /etc/nginx/sites-available/oceanbio
sudo ln -s /etc/nginx/sites-available/oceanbio /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### **Step 3: Service Configuration**

#### **Systemd Services**

Create service files for automatic startup:

```bash
# Backend service
sudo tee /etc/systemd/system/oceanbio-backend.service << EOF
[Unit]
Description=Ocean-Bio Backend API
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/Ocean-bio/backend
Environment=PATH=/path/to/Ocean-bio/backend/venv/bin
ExecStart=/path/to/Ocean-bio/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable oceanbio-backend
sudo systemctl start oceanbio-backend
```

### **Step 4: SSL/TLS Configuration**

```bash
# Install Certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com

# Auto-renewal setup
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

---

## 🔐 **Security Configuration**

### **Database Security**

```sql
-- Create read-only user for analytics
CREATE USER oceanbio_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE oceanbio_db TO oceanbio_readonly;
GRANT USAGE ON SCHEMA public TO oceanbio_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO oceanbio_readonly;

-- Set row-level security
ALTER TABLE biodiversity_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY data_access_policy ON biodiversity_data FOR SELECT USING (true);
```

### **API Security**

```bash
# Generate secure JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Configure rate limiting in .env
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=3600
```

### **Firewall Configuration**

```bash
# Configure UFW
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL from internal network only
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis from internal network only
```

---

## 📊 **Monitoring and Logging**

### **Application Monitoring**

```bash
# Install Prometheus and Grafana
docker run -d --name prometheus -p 9090:9090 prom/prometheus
docker run -d --name grafana -p 3000:3000 grafana/grafana

# Configure metrics endpoint
# Add to backend/main.py:
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

### **Log Management**

```bash
# Configure log rotation
sudo tee /etc/logrotate.d/oceanbio << EOF
/var/log/oceanbio/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload oceanbio-backend
    endscript
}
EOF
```

---

## 🧪 **Health Checks and Validation**

### **System Health Checks**

```bash
# Run comprehensive health check
./scripts/health_check.sh

# Individual service checks
curl http://localhost:8000/health
curl http://localhost:8000/api/v2/health

# Database connectivity
psql -h localhost -U oceanbio_user -d oceanbio_db -c "SELECT version();"

# Redis connectivity
redis-cli ping
```

### **Performance Testing**

```bash
# Run load tests
cd backend/tests
python load_test.py --concurrent-users 100 --duration 300

# Run benchmark suite
./scripts/benchmark.sh
```

---

## 🔄 **Backup and Recovery**

### **Database Backup**

```bash
#!/bin/bash
# Daily backup script: /usr/local/bin/backup_oceanbio.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/oceanbio"
DB_NAME="oceanbio_db"

# Create backup directory
mkdir -p $BACKUP_DIR

# PostgreSQL backup with PostGIS data
pg_dump -h localhost -U oceanbio_user -F c -b -v -f "$BACKUP_DIR/oceanbio_${DATE}.backup" $DB_NAME

# Compress and encrypt backup
gzip "$BACKUP_DIR/oceanbio_${DATE}.backup"
gpg --cipher-algo AES256 --compress-algo 1 --symmetric --output "$BACKUP_DIR/oceanbio_${DATE}.backup.gz.gpg" "$BACKUP_DIR/oceanbio_${DATE}.backup.gz"

# Remove unencrypted backup
rm "$BACKUP_DIR/oceanbio_${DATE}.backup.gz"

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.backup.gz.gpg" -mtime +30 -delete

# Add to crontab: 0 2 * * * /usr/local/bin/backup_oceanbio.sh
```

### **Application Data Backup**

```bash
# Backup uploaded files and generated reports
rsync -av /var/lib/oceanbio/uploads/ /var/backups/oceanbio/uploads_$(date +%Y%m%d)/
rsync -av /var/lib/oceanbio/reports/ /var/backups/oceanbio/reports_$(date +%Y%m%d)/
```

---

## 🚀 **Scaling and Performance**

### **Horizontal Scaling**

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  backend:
    image: oceanbio/backend:latest
    deploy:
      replicas: 4
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/oceanbio_db
      - REDIS_URL=redis://redis:6379
    
  nginx-lb:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
```

### **Database Optimization**

```sql
-- Create indexes for Phase 2 features
CREATE INDEX CONCURRENTLY idx_biodiversity_geom ON biodiversity_data USING GIST(geometry);
CREATE INDEX CONCURRENTLY idx_water_quality_location ON water_quality_data(latitude, longitude);
CREATE INDEX CONCURRENTLY idx_genomic_species ON genomic_data(species_name);
CREATE INDEX CONCURRENTLY idx_temporal_data ON all_data(collection_date);

-- Optimize PostgreSQL configuration
-- Add to postgresql.conf:
shared_buffers = 8GB
effective_cache_size = 24GB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **PostGIS Extension Error** | Ensure PostGIS is installed: `sudo apt install postgresql-14-postgis-3` |
| **BLAST Database Missing** | Download: `update_blastdb.pl --decompress nt nr` |
| **Memory Issues** | Increase RAM or enable swap: `sudo fallocate -l 8G /swapfile` |
| **Slow Queries** | Check indexes: `EXPLAIN ANALYZE SELECT ...` |
| **API Timeouts** | Increase timeout: `TIMEOUT_SECONDS=300` in .env |

### **Debug Mode**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export OCEANBIO_DEBUG=true

# Run with debug output
python -m uvicorn main:app --reload --log-level debug
```

---

## 📈 **Performance Metrics**

### **Expected Performance (Production)**

| Component | Metric | Target |
|-----------|--------|--------|
| **API Response** | Average latency | <200ms |
| **Geospatial Query** | Complex spatial query | <2s |
| **Genomic BLAST** | 1000 sequences | <30s |
| **Predictive Model** | Forecast generation | <5s |
| **Database** | Concurrent connections | 100+ |
| **System** | CPU utilization | <70% |

---

## 🔄 **Maintenance Procedures**

### **Weekly Maintenance**

```bash
#!/bin/bash
# Weekly maintenance script

# Update system packages
sudo apt update && sudo apt upgrade -y

# Vacuum and analyze database
psql -U oceanbio_user -d oceanbio_db -c "VACUUM ANALYZE;"

# Clear old log files
find /var/log/oceanbio -name "*.log" -mtime +7 -delete

# Restart services
sudo systemctl restart oceanbio-backend
sudo systemctl restart nginx
```

### **Monthly Maintenance**

```bash
# Update BLAST databases
update_blastdb.pl --decompress nt nr

# Rebuild database statistics
psql -U oceanbio_user -d oceanbio_db -c "ANALYZE;"

# Update SSL certificates
sudo certbot renew

# Generate performance report
python scripts/generate_performance_report.py
```

---

## 🆘 **Support and Documentation**

### **Technical Support**

- **Documentation**: [https://docs.ocean-bio.org](https://docs.ocean-bio.org)
- **API Reference**: [https://api.ocean-bio.org/docs](https://api.ocean-bio.org/docs)
- **GitHub Issues**: [https://github.com/your-org/ocean-bio/issues](https://github.com/your-org/ocean-bio/issues)
- **Email**: support@ocean-bio.org

### **Training Resources**

- **User Manual**: `docs/USER_MANUAL.md`
- **API Tutorial**: `docs/API_TUTORIAL.md`  
- **Video Tutorials**: `docs/TRAINING_VIDEOS.md`
- **Sample Data**: `data/samples/`

---

## ✅ **Deployment Checklist**

Before going live, ensure:

- [ ] All system requirements met
- [ ] Database properly configured with PostGIS
- [ ] SSL certificates installed and valid
- [ ] Backup procedures tested
- [ ] Monitoring systems active
- [ ] Security configurations applied
- [ ] Performance benchmarks passed
- [ ] Health checks passing
- [ ] Documentation updated
- [ ] Team trained on operations

---

**🎉 Congratulations! Ocean-Bio Phase 2 is now ready for production deployment.**

For additional support or custom deployment assistance, contact the Ocean-Bio development team.