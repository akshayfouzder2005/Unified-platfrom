#!/bin/bash

# 🌊 Ocean-Bio Phase 2 Database Setup Script
# Comprehensive PostgreSQL + PostGIS database initialization
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
DB_NAME="oceanbio_db"
DB_USER="oceanbio_user"
DB_PASS="oceanbio_password"
DB_HOST="localhost"
DB_PORT="5432"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_FILE="/tmp/oceanbio_db_setup.log"

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

# Function to check PostgreSQL installation
check_postgresql() {
    print_status "Checking PostgreSQL installation..."
    
    if ! command_exists psql; then
        print_error "PostgreSQL is not installed. Please install PostgreSQL first."
        print_status "Ubuntu/Debian: sudo apt install postgresql postgresql-contrib"
        print_status "CentOS/RHEL: sudo dnf install postgresql postgresql-server postgresql-contrib"
        print_status "macOS: brew install postgresql"
        exit 1
    fi
    
    # Check if PostgreSQL service is running
    if systemctl is-active --quiet postgresql 2>/dev/null; then
        print_success "PostgreSQL service is running"
    elif systemctl is-active --quiet postgresql@* 2>/dev/null; then
        print_success "PostgreSQL service is running"
    else
        print_status "Starting PostgreSQL service..."
        sudo systemctl start postgresql 2>/dev/null || brew services start postgresql 2>/dev/null || {
            print_warning "Could not start PostgreSQL service automatically"
            print_status "Please start PostgreSQL manually and run this script again"
        }
    fi
}

# Function to check PostGIS installation
check_postgis() {
    print_status "Checking PostGIS installation..."
    
    # Check if PostGIS packages are available
    if dpkg -l | grep -q postgis 2>/dev/null; then
        print_success "PostGIS packages found"
    elif rpm -q postgis >/dev/null 2>&1; then
        print_success "PostGIS packages found"
    else
        print_error "PostGIS is not installed. Please install PostGIS first."
        print_status "Ubuntu/Debian: sudo apt install postgresql-14-postgis-3 postgresql-14-postgis-3-scripts"
        print_status "CentOS/RHEL: sudo dnf install postgis"
        print_status "macOS: brew install postgis"
        exit 1
    fi
}

# Function to create database and user
create_database_and_user() {
    print_status "Creating database and user..."
    
    # Create database
    if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        print_warning "Database '$DB_NAME' already exists"
    else
        sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;" >> "$LOG_FILE" 2>&1
        print_success "Database '$DB_NAME' created"
    fi
    
    # Create user
    if sudo -u postgres psql -t -c "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
        print_warning "User '$DB_USER' already exists"
    else
        sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';" >> "$LOG_FILE" 2>&1
        print_success "User '$DB_USER' created"
    fi
    
    # Grant privileges
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" >> "$LOG_FILE" 2>&1
    sudo -u postgres psql -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;" >> "$LOG_FILE" 2>&1
    sudo -u postgres psql -d "$DB_NAME" -c "GRANT CREATE ON SCHEMA public TO $DB_USER;" >> "$LOG_FILE" 2>&1
    
    print_success "Database privileges granted"
}

# Function to install PostGIS extensions
install_postgis_extensions() {
    print_status "Installing PostGIS extensions..."
    
    # Install core PostGIS extension
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS postgis;" >> "$LOG_FILE" 2>&1
    print_success "PostGIS extension installed"
    
    # Install PostGIS topology extension
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;" >> "$LOG_FILE" 2>&1
    print_success "PostGIS topology extension installed"
    
    # Install PostGIS raster extension
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS postgis_raster;" >> "$LOG_FILE" 2>&1 || {
        print_warning "PostGIS raster extension not available - skipping"
    }
    
    # Install additional useful extensions
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;" >> "$LOG_FILE" 2>&1
    print_success "UUID extension installed"
    
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;" >> "$LOG_FILE" 2>&1
    print_success "PgCrypto extension installed"
    
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS btree_gist;" >> "$LOG_FILE" 2>&1
    print_success "BTree GiST extension installed"
}

# Function to create database schema
create_schema() {
    print_status "Creating database schema..."
    
    # Connect as the application user and create tables
    cat << 'EOF' > /tmp/oceanbio_schema.sql
-- Ocean-Bio Phase 2 Database Schema
-- Version: 2.0.0

-- Enable PostGIS if not already enabled
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create enum types
CREATE TYPE data_quality_level AS ENUM ('research_grade', 'needs_id', 'casual');
CREATE TYPE analysis_status AS ENUM ('pending', 'running', 'completed', 'failed');
CREATE TYPE user_role AS ENUM ('viewer', 'researcher', 'analyst', 'admin');

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role DEFAULT 'researcher',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_public BOOLEAN DEFAULT false
);

-- Sites table for sampling locations
CREATE TABLE IF NOT EXISTS sites (
    id SERIAL PRIMARY KEY,
    site_code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    location GEOMETRY(POINT, 4326) NOT NULL,
    depth_min FLOAT,
    depth_max FLOAT,
    habitat_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

-- Biodiversity data table
CREATE TABLE IF NOT EXISTS biodiversity_data (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    site_id INTEGER REFERENCES sites(id),
    project_id INTEGER REFERENCES projects(id),
    collection_date TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    species_name VARCHAR(255) NOT NULL,
    scientific_name VARCHAR(255),
    abundance INTEGER,
    biomass FLOAT,
    collection_method VARCHAR(100),
    quality_level data_quality_level DEFAULT 'needs_id',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

-- Water quality data table
CREATE TABLE IF NOT EXISTS water_quality_data (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    site_id INTEGER REFERENCES sites(id),
    project_id INTEGER REFERENCES projects(id),
    measurement_date TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    depth FLOAT,
    temperature FLOAT,
    salinity FLOAT,
    ph FLOAT,
    dissolved_oxygen FLOAT,
    turbidity FLOAT,
    chlorophyll_a FLOAT,
    nitrate FLOAT,
    phosphate FLOAT,
    quality_level data_quality_level DEFAULT 'research_grade',
    instrument VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

-- Fisheries data table
CREATE TABLE IF NOT EXISTS fisheries_data (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    site_id INTEGER REFERENCES sites(id),
    project_id INTEGER REFERENCES projects(id),
    fishing_date TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    vessel_name VARCHAR(255),
    gear_type VARCHAR(100),
    fishing_effort FLOAT,
    species_name VARCHAR(255) NOT NULL,
    catch_weight FLOAT,
    catch_count INTEGER,
    length_mean FLOAT,
    length_std FLOAT,
    quality_level data_quality_level DEFAULT 'research_grade',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

-- Climate data table
CREATE TABLE IF NOT EXISTS climate_data (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    site_id INTEGER REFERENCES sites(id),
    project_id INTEGER REFERENCES projects(id),
    observation_date TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    air_temperature FLOAT,
    sea_surface_temperature FLOAT,
    wind_speed FLOAT,
    wind_direction FLOAT,
    wave_height FLOAT,
    precipitation FLOAT,
    atmospheric_pressure FLOAT,
    humidity FLOAT,
    cloud_cover FLOAT,
    data_source VARCHAR(100),
    quality_level data_quality_level DEFAULT 'research_grade',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

-- Oceanographic data table
CREATE TABLE IF NOT EXISTS oceanographic_data (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    site_id INTEGER REFERENCES sites(id),
    project_id INTEGER REFERENCES projects(id),
    measurement_date TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    depth FLOAT,
    current_speed FLOAT,
    current_direction FLOAT,
    tide_level FLOAT,
    wave_period FLOAT,
    wave_direction FLOAT,
    upwelling_index FLOAT,
    mixed_layer_depth FLOAT,
    thermocline_depth FLOAT,
    data_source VARCHAR(100),
    quality_level data_quality_level DEFAULT 'research_grade',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

-- Genomic sequences table (Phase 2)
CREATE TABLE IF NOT EXISTS genomic_sequences (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    site_id INTEGER REFERENCES sites(id),
    project_id INTEGER REFERENCES projects(id),
    sequence_id VARCHAR(255) UNIQUE NOT NULL,
    sequence_data TEXT NOT NULL,
    sequence_length INTEGER,
    gc_content FLOAT,
    quality_scores TEXT,
    collection_date TIMESTAMP WITH TIME ZONE,
    location GEOMETRY(POINT, 4326),
    sample_type VARCHAR(100),
    extraction_method VARCHAR(100),
    sequencing_platform VARCHAR(100),
    gene_target VARCHAR(100),
    primer_forward VARCHAR(255),
    primer_reverse VARCHAR(255),
    quality_level data_quality_level DEFAULT 'needs_id',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

-- Taxonomic classifications table (Phase 2)
CREATE TABLE IF NOT EXISTS taxonomic_classifications (
    id SERIAL PRIMARY KEY,
    sequence_id INTEGER REFERENCES genomic_sequences(id),
    classification_method VARCHAR(50) NOT NULL,
    database_used VARCHAR(100),
    kingdom VARCHAR(100),
    phylum VARCHAR(100),
    class VARCHAR(100),
    order_name VARCHAR(100),
    family VARCHAR(100),
    genus VARCHAR(100),
    species VARCHAR(255),
    confidence_score FLOAT,
    identity_percent FLOAT,
    coverage_percent FLOAT,
    e_value FLOAT,
    bit_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analysis results table (Phase 2)
CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    analysis_type VARCHAR(100) NOT NULL,
    analysis_name VARCHAR(255),
    project_id INTEGER REFERENCES projects(id),
    parameters JSONB,
    results JSONB,
    status analysis_status DEFAULT 'pending',
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Spatial analysis cache table (Phase 2)
CREATE TABLE IF NOT EXISTS spatial_analysis_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,
    parameters_hash VARCHAR(64) NOT NULL,
    result_geometry GEOMETRY,
    result_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE
);

-- Stock assessments table (Phase 2)
CREATE TABLE IF NOT EXISTS stock_assessments (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    species_name VARCHAR(255) NOT NULL,
    assessment_type VARCHAR(100) NOT NULL,
    time_period_start DATE,
    time_period_end DATE,
    region GEOMETRY(POLYGON, 4326),
    parameters JSONB,
    results JSONB,
    reference_points JSONB,
    recommendations JSONB,
    status analysis_status DEFAULT 'pending',
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Forecasts table (Phase 2)
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    forecast_name VARCHAR(255),
    data_source VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    parameters JSONB,
    forecast_values JSONB,
    confidence_intervals JSONB,
    model_performance JSONB,
    status analysis_status DEFAULT 'pending',
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Phylogenetic trees table (Phase 2)
CREATE TABLE IF NOT EXISTS phylogenetic_trees (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    tree_name VARCHAR(255),
    newick_format TEXT NOT NULL,
    method VARCHAR(100) NOT NULL,
    sequence_count INTEGER,
    bootstrap_values JSONB,
    tree_statistics JSONB,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- File uploads table
CREATE TABLE IF NOT EXISTS file_uploads (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    original_filename VARCHAR(255) NOT NULL,
    stored_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    file_hash VARCHAR(64),
    upload_status VARCHAR(50) DEFAULT 'pending',
    processing_status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_biodiversity_location ON biodiversity_data USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_biodiversity_species ON biodiversity_data(species_name);
CREATE INDEX IF NOT EXISTS idx_biodiversity_date ON biodiversity_data(collection_date);
CREATE INDEX IF NOT EXISTS idx_biodiversity_site ON biodiversity_data(site_id);

CREATE INDEX IF NOT EXISTS idx_water_quality_location ON water_quality_data USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_water_quality_date ON water_quality_data(measurement_date);
CREATE INDEX IF NOT EXISTS idx_water_quality_site ON water_quality_data(site_id);

CREATE INDEX IF NOT EXISTS idx_fisheries_location ON fisheries_data USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_fisheries_species ON fisheries_data(species_name);
CREATE INDEX IF NOT EXISTS idx_fisheries_date ON fisheries_data(fishing_date);

CREATE INDEX IF NOT EXISTS idx_climate_location ON climate_data USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_climate_date ON climate_data(observation_date);

CREATE INDEX IF NOT EXISTS idx_oceanographic_location ON oceanographic_data USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_oceanographic_date ON oceanographic_data(measurement_date);

CREATE INDEX IF NOT EXISTS idx_genomic_sequences_location ON genomic_sequences USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_genomic_sequences_date ON genomic_sequences(collection_date);
CREATE INDEX IF NOT EXISTS idx_genomic_sequences_sample_type ON genomic_sequences(sample_type);

CREATE INDEX IF NOT EXISTS idx_taxonomic_classifications_species ON taxonomic_classifications(species);
CREATE INDEX IF NOT EXISTS idx_taxonomic_classifications_method ON taxonomic_classifications(classification_method);
CREATE INDEX IF NOT EXISTS idx_taxonomic_classifications_confidence ON taxonomic_classifications(confidence_score);

CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_results_status ON analysis_results(status);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created ON analysis_results(created_at);

CREATE INDEX IF NOT EXISTS idx_spatial_cache_key ON spatial_analysis_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_spatial_cache_type ON spatial_analysis_cache(analysis_type);
CREATE INDEX IF NOT EXISTS idx_spatial_cache_expires ON spatial_analysis_cache(expires_at);

CREATE INDEX IF NOT EXISTS idx_stock_assessments_species ON stock_assessments(species_name);
CREATE INDEX IF NOT EXISTS idx_stock_assessments_status ON stock_assessments(status);

CREATE INDEX IF NOT EXISTS idx_forecasts_source ON forecasts(data_source);
CREATE INDEX IF NOT EXISTS idx_forecasts_status ON forecasts(status);

CREATE INDEX IF NOT EXISTS idx_sites_location ON sites USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_sites_code ON sites(site_code);

-- Create views for common queries
CREATE OR REPLACE VIEW biodiversity_summary AS
SELECT 
    s.name as site_name,
    s.location,
    COUNT(DISTINCT bd.species_name) as species_count,
    SUM(bd.abundance) as total_abundance,
    AVG(bd.abundance) as avg_abundance,
    MIN(bd.collection_date) as first_observation,
    MAX(bd.collection_date) as last_observation
FROM sites s
LEFT JOIN biodiversity_data bd ON s.id = bd.site_id
GROUP BY s.id, s.name, s.location;

CREATE OR REPLACE VIEW water_quality_summary AS
SELECT 
    s.name as site_name,
    s.location,
    COUNT(*) as measurement_count,
    AVG(wq.temperature) as avg_temperature,
    AVG(wq.salinity) as avg_salinity,
    AVG(wq.ph) as avg_ph,
    AVG(wq.dissolved_oxygen) as avg_dissolved_oxygen,
    MIN(wq.measurement_date) as first_measurement,
    MAX(wq.measurement_date) as last_measurement
FROM sites s
LEFT JOIN water_quality_data wq ON s.id = wq.site_id
GROUP BY s.id, s.name, s.location;

-- Insert default admin user (password: 'admin123')
INSERT INTO users (username, email, password_hash, full_name, role) 
VALUES ('admin', 'admin@oceanbio.org', crypt('admin123', gen_salt('bf')), 'System Administrator', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Insert sample project
INSERT INTO projects (name, description, created_by) 
VALUES ('Maharashtra Coast Marine Survey', 'Comprehensive marine biodiversity and water quality assessment along Maharashtra coastline', 1)
ON CONFLICT DO NOTHING;

-- Insert sample sites
INSERT INTO sites (site_code, name, description, location, depth_min, depth_max, habitat_type) VALUES
('MH001', 'Mumbai Harbor', 'Main commercial harbor area', ST_SetSRID(ST_MakePoint(72.8777, 19.0760), 4326), 5, 25, 'harbor'),
('MH002', 'Elephanta Island', 'Protected marine area near Elephanta caves', ST_SetSRID(ST_MakePoint(72.9314, 18.9633), 4326), 10, 40, 'island'),
('MH003', 'Alibaug Coast', 'Coastal waters near Alibaug beach', ST_SetSRID(ST_MakePoint(72.8717, 18.6414), 4326), 2, 15, 'coastal'),
('MH004', 'Murud Beach', 'Sandy beach ecosystem', ST_SetSRID(ST_MakePoint(72.9644, 18.3275), 4326), 1, 20, 'beach'),
('MH005', 'Ratnagiri Waters', 'Deep water sampling site', ST_SetSRID(ST_MakePoint(73.3119, 16.9902), 4326), 20, 100, 'pelagic')
ON CONFLICT (site_code) DO NOTHING;

COMMIT;
EOF

    # Execute schema creation
    PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f /tmp/oceanbio_schema.sql >> "$LOG_FILE" 2>&1
    
    # Clean up temporary file
    rm -f /tmp/oceanbio_schema.sql
    
    print_success "Database schema created successfully"
}

# Function to insert sample data
insert_sample_data() {
    print_status "Inserting sample data..."
    
    cat << 'EOF' > /tmp/sample_data.sql
-- Insert sample biodiversity data
INSERT INTO biodiversity_data (site_id, project_id, collection_date, location, species_name, scientific_name, abundance, collection_method, quality_level, created_by) VALUES
(1, 1, '2024-01-15 08:30:00+00', ST_SetSRID(ST_MakePoint(72.8777, 19.0760), 4326), 'Yellowfin Tuna', 'Thunnus albacares', 25, 'trawl', 'research_grade', 1),
(1, 1, '2024-01-15 08:30:00+00', ST_SetSRID(ST_MakePoint(72.8777, 19.0760), 4326), 'Indian Oil Sardine', 'Sardinella longiceps', 150, 'net', 'research_grade', 1),
(2, 1, '2024-01-16 09:15:00+00', ST_SetSRID(ST_MakePoint(72.9314, 18.9633), 4326), 'Indian Mackerel', 'Rastrelliger kanagurta', 89, 'trawl', 'research_grade', 1),
(3, 1, '2024-01-17 07:45:00+00', ST_SetSRID(ST_MakePoint(72.8717, 18.6414), 4326), 'Pomfret', 'Pampus argenteus', 45, 'hook_line', 'research_grade', 1),
(4, 1, '2024-01-18 08:00:00+00', ST_SetSRID(ST_MakePoint(72.9644, 18.3275), 4326), 'Bombay Duck', 'Harpadon nehereus', 78, 'trawl', 'research_grade', 1);

-- Insert sample water quality data
INSERT INTO water_quality_data (site_id, project_id, measurement_date, location, depth, temperature, salinity, ph, dissolved_oxygen, turbidity, quality_level, created_by) VALUES
(1, 1, '2024-01-15 10:00:00+00', ST_SetSRID(ST_MakePoint(72.8777, 19.0760), 4326), 5.0, 26.5, 35.2, 8.1, 7.2, 2.5, 'research_grade', 1),
(2, 1, '2024-01-16 10:30:00+00', ST_SetSRID(ST_MakePoint(72.9314, 18.9633), 4326), 10.0, 26.8, 35.0, 8.0, 7.0, 1.8, 'research_grade', 1),
(3, 1, '2024-01-17 09:00:00+00', ST_SetSRID(ST_MakePoint(72.8717, 18.6414), 4326), 3.0, 27.2, 34.8, 8.2, 6.8, 3.1, 'research_grade', 1),
(4, 1, '2024-01-18 09:30:00+00', ST_SetSRID(ST_MakePoint(72.9644, 18.3275), 4326), 2.0, 27.5, 34.6, 8.3, 6.5, 4.2, 'research_grade', 1),
(5, 1, '2024-01-19 11:00:00+00', ST_SetSRID(ST_MakePoint(73.3119, 16.9902), 4326), 50.0, 25.8, 35.5, 8.0, 7.5, 1.2, 'research_grade', 1);

-- Insert sample fisheries data
INSERT INTO fisheries_data (site_id, project_id, fishing_date, location, vessel_name, gear_type, fishing_effort, species_name, catch_weight, catch_count, quality_level, created_by) VALUES
(1, 1, '2024-01-20 06:00:00+00', ST_SetSRID(ST_MakePoint(72.8777, 19.0760), 4326), 'Mumbai Fisher 1', 'trawl_net', 4.5, 'Yellowfin Tuna', 125.5, 5, 'research_grade', 1),
(2, 1, '2024-01-21 05:30:00+00', ST_SetSRID(ST_MakePoint(72.9314, 18.9633), 4326), 'Elephanta Pride', 'gill_net', 6.0, 'Indian Mackerel', 89.3, 35, 'research_grade', 1),
(3, 1, '2024-01-22 06:15:00+00', ST_SetSRID(ST_MakePoint(72.8717, 18.6414), 4326), 'Alibaug Express', 'hook_line', 8.0, 'Pomfret', 67.8, 15, 'research_grade', 1);

-- Insert sample genomic sequences
INSERT INTO genomic_sequences (site_id, project_id, sequence_id, sequence_data, sequence_length, gc_content, collection_date, location, sample_type, gene_target, quality_level, created_by) VALUES
(1, 1, 'MH001_001', 'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG', 100, 50.0, '2024-01-15 08:30:00+00', ST_SetSRID(ST_MakePoint(72.8777, 19.0760), 4326), 'eDNA', 'COI', 'research_grade', 1),
(2, 1, 'MH002_001', 'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA', 100, 50.0, '2024-01-16 09:15:00+00', ST_SetSRID(ST_MakePoint(72.9314, 18.9633), 4326), 'eDNA', 'COI', 'research_grade', 1),
(3, 1, 'MH003_001', 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT', 100, 0.0, '2024-01-17 07:45:00+00', ST_SetSRID(ST_MakePoint(72.8717, 18.6414), 4326), 'tissue', '16S', 'needs_id', 1);

-- Insert sample taxonomic classifications
INSERT INTO taxonomic_classifications (sequence_id, classification_method, database_used, kingdom, phylum, class, order_name, family, genus, species, confidence_score, identity_percent) VALUES
(1, 'blast', 'ncbi_nt', 'Eukaryota', 'Chordata', 'Actinopterygii', 'Perciformes', 'Scombridae', 'Thunnus', 'Thunnus albacares', 0.98, 98.5),
(2, 'blast', 'ncbi_nt', 'Eukaryota', 'Chordata', 'Actinopterygii', 'Clupeiformes', 'Clupeidae', 'Sardinella', 'Sardinella longiceps', 0.95, 95.2),
(3, 'blast', 'ncbi_nt', 'Eukaryota', 'Chordata', 'Actinopterygii', 'Perciformes', 'Scombridae', 'Rastrelliger', 'Rastrelliger kanagurta', 0.92, 92.8);

COMMIT;
EOF

    # Execute sample data insertion
    PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f /tmp/sample_data.sql >> "$LOG_FILE" 2>&1
    
    # Clean up temporary file
    rm -f /tmp/sample_data.sql
    
    print_success "Sample data inserted successfully"
}

# Function to create database configuration file
create_db_config() {
    print_status "Creating database configuration file..."
    
    cat << EOF > "$(dirname "$SCRIPT_DIR")/backend/.env.database"
# Ocean-Bio Phase 2 Database Configuration
# Generated on $(date)

# Database Connection
DATABASE_HOST=$DB_HOST
DATABASE_PORT=$DB_PORT
DATABASE_NAME=$DB_NAME
DATABASE_USER=$DB_USER
DATABASE_PASSWORD=$DB_PASS

# Connection URL for SQLAlchemy
DATABASE_URL=postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME

# PostGIS Configuration
POSTGIS_VERSION=3.2
ENABLE_SPATIAL_FEATURES=true

# Connection Pool Settings
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Query Settings
DATABASE_QUERY_TIMEOUT=300
DATABASE_STATEMENT_TIMEOUT=600

# SSL Settings (for production)
DATABASE_SSL_MODE=prefer
DATABASE_SSL_CERT=
DATABASE_SSL_KEY=
DATABASE_SSL_ROOT_CERT=
EOF

    print_success "Database configuration file created at backend/.env.database"
}

# Function to optimize PostgreSQL configuration
optimize_postgresql() {
    print_status "Optimizing PostgreSQL configuration..."
    
    # Get PostgreSQL configuration file path
    PG_CONFIG_FILE=$(sudo -u postgres psql -t -P format=unaligned -c 'SHOW config_file;')
    
    if [[ -f "$PG_CONFIG_FILE" ]]; then
        print_status "PostgreSQL config file: $PG_CONFIG_FILE"
        
        # Create backup of original config
        sudo cp "$PG_CONFIG_FILE" "$PG_CONFIG_FILE.oceanbio.backup.$(date +%Y%m%d_%H%M%S)"
        
        # Add optimizations for Ocean-Bio
        cat << 'EOF' | sudo tee -a "$PG_CONFIG_FILE" > /dev/null

# Ocean-Bio Phase 2 Optimizations
# Added on $(date)

# Memory Settings
shared_buffers = 256MB              # 25% of RAM (adjust based on available memory)
effective_cache_size = 1GB          # 75% of RAM (adjust based on available memory)
work_mem = 16MB                     # Per-operation memory
maintenance_work_mem = 256MB        # Maintenance operations

# Write-Ahead Logging (WAL)
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_level = replica

# Query Planning
random_page_cost = 1.1              # Assumes SSD storage
effective_io_concurrency = 200      # SSD concurrent I/O

# Connection Settings
max_connections = 200               # Adjust based on expected load

# Logging
log_statement = 'mod'               # Log all modifications
log_duration = on                   # Log query duration
log_min_duration_statement = 1000  # Log queries taking > 1 second

# PostGIS Optimizations
shared_preload_libraries = 'postgis'
EOF

        print_success "PostgreSQL configuration optimized"
        print_warning "PostgreSQL restart required for changes to take effect"
        print_status "Run: sudo systemctl restart postgresql"
    else
        print_warning "Could not locate PostgreSQL configuration file"
    fi
}

# Function to test database connection
test_database_connection() {
    print_status "Testing database connection..."
    
    # Test connection as application user
    if PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" >> "$LOG_FILE" 2>&1; then
        print_success "Database connection test successful"
    else
        print_error "Database connection test failed"
        exit 1
    fi
    
    # Test PostGIS functionality
    if PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT PostGIS_version();" >> "$LOG_FILE" 2>&1; then
        print_success "PostGIS functionality test successful"
    else
        print_error "PostGIS functionality test failed"
        exit 1
    fi
    
    # Test spatial query
    if PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT ST_AsText(ST_MakePoint(72.8777, 19.0760));" >> "$LOG_FILE" 2>&1; then
        print_success "Spatial query test successful"
    else
        print_error "Spatial query test failed"
        exit 1
    fi
}

# Function to display database information
display_database_info() {
    print_status "Retrieving database information..."
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}🗄️  Ocean-Bio Database Setup Complete!${NC}"
    echo "=========================================="
    echo ""
    
    # Get database size
    DB_SIZE=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null | xargs)
    
    # Get table count
    TABLE_COUNT=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | xargs)
    
    # Get PostGIS version
    POSTGIS_VERSION=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT PostGIS_version();" 2>/dev/null | xargs)
    
    echo -e "${BLUE}📊 Database Information:${NC}"
    echo "• Database Name: $DB_NAME"
    echo "• Database User: $DB_USER"
    echo "• Database Host: $DB_HOST:$DB_PORT"
    echo "• Database Size: $DB_SIZE"
    echo "• Table Count: $TABLE_COUNT"
    echo "• PostGIS Version: $POSTGIS_VERSION"
    echo ""
    
    echo -e "${BLUE}🔗 Connection Details:${NC}"
    echo "• Connection URL: postgresql://$DB_USER:***@$DB_HOST:$DB_PORT/$DB_NAME"
    echo "• Configuration File: backend/.env.database"
    echo ""
    
    echo -e "${BLUE}📋 Available Tables:${NC}"
    PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "\\dt" 2>/dev/null | grep -E "^ [a-z_]+" | awk '{print "• " $3}' || echo "• Error retrieving table list"
    echo ""
    
    echo -e "${BLUE}🧪 Sample Data:${NC}"
    SAMPLE_COUNT=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM biodiversity_data;" 2>/dev/null | xargs)
    echo "• Biodiversity Records: $SAMPLE_COUNT"
    
    SITE_COUNT=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM sites;" 2>/dev/null | xargs)
    echo "• Sampling Sites: $SITE_COUNT"
    
    SEQUENCE_COUNT=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM genomic_sequences;" 2>/dev/null | xargs)
    echo "• Genomic Sequences: $SEQUENCE_COUNT"
    echo ""
    
    echo -e "${YELLOW}⚠️  Next Steps:${NC}"
    echo "1. Update backend/.env with database connection details"
    echo "2. Run database migrations: alembic upgrade head"
    echo "3. Restart PostgreSQL if configuration was optimized"
    echo "4. Test API connectivity to database"
    echo "5. Import your marine data using the data upload API"
    echo ""
    
    echo -e "${GREEN}✅ Database is ready for Ocean-Bio Phase 2!${NC}"
    echo ""
}

# Main setup function
main() {
    echo "🌊 Ocean-Bio Phase 2 Database Setup"
    echo "====================================="
    echo ""
    
    # Initialize log file
    echo "$(date): Starting Ocean-Bio database setup" > "$LOG_FILE"
    
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
    
    # Run setup steps
    check_postgresql
    check_postgis
    create_database_and_user
    install_postgis_extensions
    create_schema
    insert_sample_data
    create_db_config
    optimize_postgresql
    test_database_connection
    display_database_info
    
    print_success "Database setup completed successfully!"
    print_status "Check the log file for details: $LOG_FILE"
}

# Run main function
main "$@"