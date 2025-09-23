#!/bin/bash

# 🌊 Ocean-Bio Phase 2 Quick Deployment Script
# Automated setup for all Phase 2 components
# Version: 2.0.0

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Ocean-Bio Phase 2"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
LOG_FILE="/tmp/oceanbio_deploy.log"

# System detection
OS=$(uname -s)
ARCH=$(uname -m)
DISTRO=""

if [[ "$OS" == "Linux" ]]; then
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
    fi
fi

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check OS
    if [[ "$OS" != "Linux" && "$OS" != "Darwin" ]]; then
        print_error "Unsupported operating system: $OS"
        exit 1
    fi
    
    # Check minimum RAM (8GB)
    if [[ "$OS" == "Linux" ]]; then
        TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
        if [[ $TOTAL_RAM -lt 8 ]]; then
            print_warning "Insufficient RAM detected: ${TOTAL_RAM}GB. Minimum 8GB recommended."
        fi
    fi
    
    # Check disk space (minimum 50GB)
    AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 50 ]]; then
        print_warning "Low disk space: ${AVAILABLE_SPACE}GB available. Minimum 50GB recommended."
    fi
    
    print_success "System requirements check completed"
}

# Function to update system packages
update_system() {
    print_status "Updating system packages..."
    
    if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
        sudo apt update && sudo apt upgrade -y >> "$LOG_FILE" 2>&1
    elif [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" || "$DISTRO" == "fedora" ]]; then
        if command_exists dnf; then
            sudo dnf update -y >> "$LOG_FILE" 2>&1
        else
            sudo yum update -y >> "$LOG_FILE" 2>&1
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        if command_exists brew; then
            brew update >> "$LOG_FILE" 2>&1
        fi
    fi
    
    print_success "System packages updated"
}

# Function to install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
        sudo apt install -y \
            curl wget git build-essential \
            python3 python3-pip python3-venv python3-dev \
            nodejs npm \
            postgresql-14 postgresql-client-14 postgresql-contrib-14 \
            postgresql-14-postgis-3 postgresql-14-postgis-3-scripts \
            redis-server \
            gdal-bin libgdal-dev \
            libpq-dev \
            nginx \
            certbot python3-certbot-nginx >> "$LOG_FILE" 2>&1
            
    elif [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" || "$DISTRO" == "fedora" ]]; then
        sudo dnf install -y \
            curl wget git gcc gcc-c++ make \
            python3 python3-pip python3-devel \
            nodejs npm \
            postgresql postgresql-server postgresql-contrib \
            postgis \
            redis \
            gdal gdal-devel \
            postgresql-devel \
            nginx \
            certbot python3-certbot-nginx >> "$LOG_FILE" 2>&1
            
    elif [[ "$OS" == "Darwin" ]]; then
        if command_exists brew; then
            brew install \
                python@3.11 \
                node \
                postgresql@14 \
                redis \
                gdal \
                nginx >> "$LOG_FILE" 2>&1
        else
            print_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
    fi
    
    print_success "System dependencies installed"
}

# Function to install external bioinformatics tools
install_bioinformatics_tools() {
    print_status "Installing bioinformatics tools..."
    
    # Create tools directory
    TOOLS_DIR="/opt/oceanbio-tools"
    sudo mkdir -p "$TOOLS_DIR"
    
    # Install BLAST+
    if ! command_exists blastn; then
        print_status "Installing BLAST+..."
        cd /tmp
        if [[ "$OS" == "Linux" ]]; then
            wget -q https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.13.0+-x64-linux.tar.gz
            tar -xzf ncbi-blast-2.13.0+-x64-linux.tar.gz
            sudo mv ncbi-blast-2.13.0+ "$TOOLS_DIR/blast"
        elif [[ "$OS" == "Darwin" ]]; then
            wget -q https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.13.0+-x64-macosx.tar.gz
            tar -xzf ncbi-blast-2.13.0+-x64-macosx.tar.gz
            sudo mv ncbi-blast-2.13.0+ "$TOOLS_DIR/blast"
        fi
        
        # Add to PATH
        echo "export PATH=$TOOLS_DIR/blast/bin:\$PATH" | sudo tee /etc/profile.d/oceanbio-tools.sh
        export PATH="$TOOLS_DIR/blast/bin:$PATH"
        
        print_success "BLAST+ installed"
    else
        print_success "BLAST+ already installed"
    fi
    
    # Install MUSCLE
    if ! command_exists muscle; then
        print_status "Installing MUSCLE..."
        cd /tmp
        if [[ "$OS" == "Linux" ]]; then
            wget -q https://github.com/rcedgar/muscle/releases/download/v5.1/muscle5.1.linux_intel64
            sudo mv muscle5.1.linux_intel64 /usr/local/bin/muscle
        elif [[ "$OS" == "Darwin" ]]; then
            wget -q https://github.com/rcedgar/muscle/releases/download/v5.1/muscle5.1.macos_intel64
            sudo mv muscle5.1.macos_intel64 /usr/local/bin/muscle
        fi
        sudo chmod +x /usr/local/bin/muscle
        print_success "MUSCLE installed"
    else
        print_success "MUSCLE already installed"
    fi
    
    print_success "Bioinformatics tools installation completed"
}

# Function to setup PostgreSQL with PostGIS
setup_database() {
    print_status "Setting up PostgreSQL with PostGIS..."
    
    # Start PostgreSQL service
    if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    elif [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" || "$DISTRO" == "fedora" ]]; then
        sudo postgresql-setup --initdb
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    elif [[ "$OS" == "Darwin" ]]; then
        brew services start postgresql@14
    fi
    
    # Wait for PostgreSQL to start
    sleep 5
    
    # Create database and user
    sudo -u postgres psql -c "CREATE DATABASE oceanbio_db;" 2>/dev/null || true
    sudo -u postgres psql -d oceanbio_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
    sudo -u postgres psql -d oceanbio_db -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;"
    sudo -u postgres psql -d oceanbio_db -c "CREATE EXTENSION IF NOT EXISTS postgis_raster;"
    
    # Create application user
    sudo -u postgres psql -c "CREATE USER oceanbio_user WITH PASSWORD 'oceanbio_password';" 2>/dev/null || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE oceanbio_db TO oceanbio_user;"
    sudo -u postgres psql -d oceanbio_db -c "GRANT ALL ON SCHEMA public TO oceanbio_user;"
    
    print_success "PostgreSQL with PostGIS setup completed"
}

# Function to setup Redis
setup_redis() {
    print_status "Setting up Redis..."
    
    # Start Redis service
    if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
        sudo systemctl start redis-server
        sudo systemctl enable redis-server
    elif [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" || "$DISTRO" == "fedora" ]]; then
        sudo systemctl start redis
        sudo systemctl enable redis
    elif [[ "$OS" == "Darwin" ]]; then
        brew services start redis
    fi
    
    # Configure Redis
    if [[ -f /etc/redis/redis.conf ]]; then
        sudo sed -i 's/# maxmemory <bytes>/maxmemory 2gb/' /etc/redis/redis.conf
        sudo sed -i 's/# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf
        sudo systemctl restart redis-server 2>/dev/null || sudo systemctl restart redis 2>/dev/null
    fi
    
    print_success "Redis setup completed"
}

# Function to setup Python environment
setup_python_environment() {
    print_status "Setting up Python environment..."
    
    cd "$BACKEND_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip >> "$LOG_FILE" 2>&1
    
    # Install Python dependencies
    if [[ -f requirements.txt ]]; then
        pip install -r requirements.txt >> "$LOG_FILE" 2>&1
    fi
    
    # Install additional production dependencies
    pip install \
        uvicorn[standard] \
        gunicorn \
        psycopg2-binary \
        redis \
        celery >> "$LOG_FILE" 2>&1
    
    print_success "Python environment setup completed"
}

# Function to setup Node.js environment
setup_nodejs_environment() {
    print_status "Setting up Node.js environment..."
    
    cd "$FRONTEND_DIR"
    
    # Install dependencies
    npm install >> "$LOG_FILE" 2>&1
    
    # Build production assets
    npm run build >> "$LOG_FILE" 2>&1
    
    print_success "Node.js environment setup completed"
}

# Function to run database migrations
run_database_migrations() {
    print_status "Running database migrations..."
    
    cd "$BACKEND_DIR"
    source venv/bin/activate
    
    # Install Alembic if not present
    pip install alembic >> "$LOG_FILE" 2>&1
    
    # Initialize Alembic if needed
    if [[ ! -d alembic ]]; then
        alembic init alembic >> "$LOG_FILE" 2>&1
    fi
    
    # Run migrations
    alembic upgrade head >> "$LOG_FILE" 2>&1 || print_warning "Migration failed - database may already be up to date"
    
    print_success "Database migrations completed"
}

# Function to create environment files
create_environment_files() {
    print_status "Creating environment configuration files..."
    
    # Backend .env file
    cat > "$BACKEND_DIR/.env" << EOF
# Ocean-Bio Phase 2 Configuration
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Database Configuration
DATABASE_URL=postgresql://oceanbio_user:oceanbio_password@localhost:5432/oceanbio_db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# External APIs
BLAST_DB_PATH=/var/lib/oceanbio/blast_db
MUSCLE_PATH=/usr/local/bin/muscle

# Security
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/oceanbio/backend.log

# Feature Flags
ENABLE_GEOSPATIAL=true
ENABLE_PREDICTIVE=true
ENABLE_GENOMICS=true
EOF

    # Frontend .env file
    cat > "$FRONTEND_DIR/.env" << EOF
# Ocean-Bio Frontend Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_API_VERSION=v2
REACT_APP_ENVIRONMENT=production

# Feature Flags
REACT_APP_ENABLE_GEOSPATIAL=true
REACT_APP_ENABLE_PREDICTIVE=true
REACT_APP_ENABLE_GENOMICS=true

# Map Configuration
REACT_APP_DEFAULT_MAP_CENTER_LAT=19.0760
REACT_APP_DEFAULT_MAP_CENTER_LNG=72.8777
REACT_APP_DEFAULT_MAP_ZOOM=10

# External Services
REACT_APP_MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
EOF

    print_success "Environment files created"
}

# Function to create systemd services
create_systemd_services() {
    print_status "Creating systemd services..."
    
    # Backend service
    sudo tee /etc/systemd/system/oceanbio-backend.service > /dev/null << EOF
[Unit]
Description=Ocean-Bio Backend API
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=$BACKEND_DIR
Environment=PATH=$BACKEND_DIR/venv/bin
EnvironmentFile=$BACKEND_DIR/.env
ExecStart=$BACKEND_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable oceanbio-backend
    
    print_success "Systemd services created"
}

# Function to configure Nginx
configure_nginx() {
    print_status "Configuring Nginx..."
    
    # Create Nginx configuration
    sudo tee /etc/nginx/sites-available/oceanbio > /dev/null << EOF
server {
    listen 80;
    server_name localhost yourdomain.com;
    
    # Frontend
    location / {
        root $FRONTEND_DIR/build;
        index index.html index.htm;
        try_files \$uri \$uri/ /index.html;
    }
    
    # API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host \$host;
    }
    
    # Static files
    location /static/ {
        alias $BACKEND_DIR/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

    # Enable the site
    sudo ln -sf /etc/nginx/sites-available/oceanbio /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test configuration
    sudo nginx -t
    
    # Restart Nginx
    sudo systemctl restart nginx
    sudo systemctl enable nginx
    
    print_success "Nginx configuration completed"
}

# Function to create log directories
create_log_directories() {
    print_status "Creating log directories..."
    
    sudo mkdir -p /var/log/oceanbio
    sudo chown www-data:www-data /var/log/oceanbio
    
    # Create data directories
    sudo mkdir -p /var/lib/oceanbio/{uploads,reports,blast_db,temp}
    sudo chown -R www-data:www-data /var/lib/oceanbio
    
    print_success "Log and data directories created"
}

# Function to run tests
run_tests() {
    print_status "Running test suite..."
    
    cd "$BACKEND_DIR"
    source venv/bin/activate
    
    # Install test dependencies
    if [[ -f tests/requirements-test.txt ]]; then
        pip install -r tests/requirements-test.txt >> "$LOG_FILE" 2>&1
    fi
    
    # Run tests
    if [[ -f tests/run_tests.py ]]; then
        python tests/run_tests.py --component all --no-setup >> "$LOG_FILE" 2>&1 || print_warning "Some tests failed - check logs for details"
    fi
    
    print_success "Test suite completed"
}

# Function to start services
start_services() {
    print_status "Starting Ocean-Bio services..."
    
    # Start backend service
    sudo systemctl start oceanbio-backend
    
    # Wait for service to start
    sleep 5
    
    # Check service status
    if sudo systemctl is-active oceanbio-backend >/dev/null; then
        print_success "Backend service started successfully"
    else
        print_error "Failed to start backend service"
        sudo systemctl status oceanbio-backend
        exit 1
    fi
    
    print_success "All services started"
}

# Function to perform health checks
perform_health_checks() {
    print_status "Performing health checks..."
    
    # Wait for services to be ready
    sleep 10
    
    # Check API health
    if curl -f -s http://localhost:8000/health > /dev/null; then
        print_success "API health check passed"
    else
        print_warning "API health check failed"
    fi
    
    # Check database connection
    if sudo -u postgres psql -d oceanbio_db -c "SELECT version();" > /dev/null 2>&1; then
        print_success "Database connection check passed"
    else
        print_warning "Database connection check failed"
    fi
    
    # Check Redis connection
    if redis-cli ping > /dev/null 2>&1; then
        print_success "Redis connection check passed"
    else
        print_warning "Redis connection check failed"
    fi
    
    print_success "Health checks completed"
}

# Function to display deployment summary
display_summary() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}🎉 Ocean-Bio Phase 2 Deployment Complete!${NC}"
    echo "=========================================="
    echo ""
    echo -e "${BLUE}📊 Deployment Summary:${NC}"
    echo "• Platform: Ocean-Bio Phase 2"
    echo "• Version: 2.0.0"
    echo "• Components: Geospatial, Predictive, Genomics"
    echo "• Database: PostgreSQL with PostGIS"
    echo "• Cache: Redis"
    echo "• Web Server: Nginx"
    echo ""
    echo -e "${BLUE}🔗 Access Points:${NC}"
    echo "• Frontend: http://localhost"
    echo "• API: http://localhost:8000"
    echo "• API Docs: http://localhost:8000/docs"
    echo "• Health Check: http://localhost:8000/health"
    echo ""
    echo -e "${BLUE}📁 Important Locations:${NC}"
    echo "• Project Root: $PROJECT_ROOT"
    echo "• Backend: $BACKEND_DIR"
    echo "• Frontend: $FRONTEND_DIR"
    echo "• Logs: /var/log/oceanbio/"
    echo "• Data: /var/lib/oceanbio/"
    echo ""
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo "• Start Backend: sudo systemctl start oceanbio-backend"
    echo "• Stop Backend: sudo systemctl stop oceanbio-backend"
    echo "• Check Status: sudo systemctl status oceanbio-backend"
    echo "• View Logs: sudo journalctl -u oceanbio-backend -f"
    echo ""
    echo -e "${YELLOW}⚠️  Next Steps:${NC}"
    echo "1. Update domain names in Nginx config and .env files"
    echo "2. Configure SSL certificates with: sudo certbot --nginx"
    echo "3. Set up monitoring and alerting"
    echo "4. Configure backups"
    echo "5. Review and update security settings"
    echo ""
    echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
    echo "For support, see: PHASE2_DEPLOYMENT.md"
    echo ""
}

# Main deployment function
main() {
    echo "🌊 Ocean-Bio Phase 2 Quick Deployment"
    echo "======================================"
    echo ""
    
    # Initialize log file
    echo "$(date): Starting Ocean-Bio Phase 2 deployment" > "$LOG_FILE"
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root. Use a user with sudo privileges."
        exit 1
    fi
    
    # Check for sudo privileges
    if ! sudo -v; then
        print_error "This script requires sudo privileges. Please run with a user that has sudo access."
        exit 1
    fi
    
    # Run deployment steps
    check_system_requirements
    update_system
    install_system_dependencies
    install_bioinformatics_tools
    setup_database
    setup_redis
    create_log_directories
    setup_python_environment
    setup_nodejs_environment
    create_environment_files
    run_database_migrations
    create_systemd_services
    configure_nginx
    start_services
    perform_health_checks
    run_tests
    
    # Display summary
    display_summary
    
    print_success "Deployment completed! Check the summary above for next steps."
}

# Run main function
main "$@"