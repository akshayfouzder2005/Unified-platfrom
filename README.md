# 🌊 Ocean-Bio: AI-Driven Unified Data Platform

**A robust, cloud-ready web platform for Oceanographic, Fisheries, and Molecular Biodiversity Insights**

![Progress](https://img.shields.io/badge/Progress-83%25-green)
![Backend](https://img.shields.io/badge/Backend-Complete-brightgreen)
![Database](https://img.shields.io/badge/Database-Complete-brightgreen)
![Frontend](https://img.shields.io/badge/Frontend-In%20Progress-yellow)
![Auth](https://img.shields.io/badge/Authentication-Pending-red)

## 🎯 Project Overview

This platform provides a unified solution for managing and analyzing:
- **Oceanographic Data**: Environmental measurements and monitoring
- **Taxonomic Information**: Species classification and biodiversity
- **eDNA Analysis**: Environmental DNA sequencing and detection
- **Otolith Morphology**: Fish identification through ear stone analysis

## 📊 Current Progress (83% Complete)

### ✅ **Completed Features**

#### 🗄️ **Database Architecture**
- **Comprehensive Models**: 15+ SQLAlchemy models covering all data domains
- **Taxonomic Hierarchy**: Complete taxonomic classification system
- **Oceanographic Monitoring**: Stations, parameters, measurements with quality control
- **eDNA Pipeline**: Sample → Extraction → PCR → Sequencing → Analysis
- **Otolith Analysis**: Specimen management with morphometric measurements
- **Migration System**: Alembic configuration for database versioning

#### 🚀 **Advanced API Backend** 
- **Full CRUD Operations**: Create, Read, Update, Delete for all entities
- **RESTful Design**: 25+ endpoints with proper HTTP methods
- **Advanced Search**: Filter, pagination, and complex queries
- **Data Validation**: Pydantic schemas with comprehensive validation
- **Error Handling**: Custom exception handling with detailed responses
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

#### 📥 **Data Ingestion System**
- **CSV Upload**: File upload with validation and processing
- **Batch Processing**: Pandas-based data transformation
- **Error Reporting**: Detailed validation and processing feedback
- **Template Generation**: Downloadable CSV templates
- **Quality Control**: Data validation and duplicate detection

#### 🔧 **Error Handling & Validation**
- **Custom Exceptions**: Specialized error classes for different scenarios
- **Comprehensive Validation**: Field-level and business logic validation
- **Detailed Error Responses**: User-friendly error messages
- **Database Error Management**: SQLAlchemy exception handling

### 🔄 **In Progress**

#### 🎨 **Interactive Frontend** (50% Complete)
- **React Components**: Modern component-based architecture
- **Responsive Design**: Tailwind CSS styling
- **API Integration**: Client-side API communication
- **Data Visualization**: Chart.js integration for analytics

### ⏳ **Pending**

#### 🔐 **Authentication & Security** (0% Complete)
- User management system
- JWT token authentication
- Role-based access control
- API security middleware

## 🏗️ **Architecture**

### **Backend Stack**
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Migrations**: Alembic
- **Validation**: Pydantic
- **Container**: Docker & Docker Compose

### **Frontend Stack**
- **Framework**: React 18 (CDN-based)
- **Styling**: Tailwind CSS
- **Charts**: Chart.js
- **Icons**: Font Awesome
- **Build**: Babel Standalone

### **Data Models**

```
📊 Database Schema (15 Tables)
├── Taxonomy
│   ├── taxonomic_ranks
│   ├── taxonomic_units
│   ├── taxonomic_synonyms
│   └── taxonomic_references
├── Oceanographic
│   ├── oceanographic_stations
│   ├── oceanographic_parameters
│   ├── oceanographic_measurements
│   ├── oceanographic_datasets
│   └── oceanographic_alerts
├── eDNA
│   ├── edna_samples
│   ├── edna_extractions
│   ├── pcr_reactions
│   ├── dna_sequences
│   ├── taxonomic_assignments
│   ├── edna_detections
│   └── edna_studies
└── Otolith
    ├── otolith_specimens
    ├── otolith_measurements
    ├── otolith_images
    ├── otolith_references
    ├── otolith_classifications
    └── otolith_studies
```

## 🚀 **Quick Start**

### Prerequisites
- Python 3.11+
- PostgreSQL (optional - using Docker)
- Docker & Docker Compose (recommended)

### Option A: Docker Setup (Recommended)

1. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

2. **Start Services**
   ```bash
   docker compose up --build
   ```

3. **Access Application**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Frontend: Open `client/public/index.html`

### Option B: Local Development

1. **Backend Setup**
   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   pip install -r requirements.txt
   ```

2. **Run Migrations**
   ```bash
   alembic upgrade head
   ```

3. **Start API**
   ```bash
   uvicorn app.main:app --reload
   ```

## 📚 **API Documentation**

### **Key Endpoints**

#### Taxonomy Management
- `GET /api/taxonomy/species` - Search and filter species
- `POST /api/taxonomy/species` - Add new species
- `PUT /api/taxonomy/species/{id}` - Update species
- `DELETE /api/taxonomy/species/{id}` - Delete species
- `GET /api/taxonomy/taxonomy-tree` - Get taxonomic hierarchy

#### Data Ingestion
- `POST /api/ingestion/taxonomy/upload-csv` - Upload taxonomy CSV
- `GET /api/ingestion/taxonomy/validate-csv` - Validate CSV format
- `GET /api/ingestion/ingestion/status` - Get ingestion status

#### Health & Monitoring
- `GET /api/health` - API health check
- `GET /api/taxonomy/stats` - Taxonomy statistics

## 📁 **Project Structure**

```
Ocean-bio/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── core/              # Configuration, database, exceptions
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic validation schemas
│   │   ├── crud/              # Database operations
│   │   ├── modules/           # API route modules
│   │   │   ├── taxonomy/      # Taxonomy management
│   │   │   ├── edna/          # eDNA analysis
│   │   │   ├── otolith/       # Otolith morphology
│   │   │   └── ingestion/     # Data upload & processing
│   │   └── main.py            # FastAPI application
│   ├── alembic/               # Database migrations
│   ├── requirements.txt
│   └── Dockerfile
├── client/                     # React frontend
│   └── public/
│       └── index.html         # Single-page application
├── docs/                       # Documentation
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🧪 **Testing**

```bash
# Backend tests
cd backend
pytest

# API testing
curl http://localhost:8000/api/health
```

## 🔮 **Next Steps**

### **Phase 1: Complete Frontend** (Priority 1)
- [ ] Finish React component implementation
- [ ] Add data visualization dashboards
- [ ] Implement file upload interface

### **Phase 2: Authentication System** (Priority 2)
- [ ] JWT authentication
- [ ] User registration/login
- [ ] Role-based permissions
- [ ] API security middleware

### **Phase 3: Advanced Features** (Priority 3)
- [ ] Real-time data streaming
- [ ] Machine learning integration
- [ ] Advanced visualization tools
- [ ] Export/reporting system

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- FastAPI for the excellent web framework
- SQLAlchemy for robust ORM capabilities
- React for modern frontend development
- Chart.js for data visualization
- Tailwind CSS for responsive design

---

**🌊 Built for marine biodiversity research and oceanographic data management**

![Ocean](https://img.shields.io/badge/Ocean-Conservation-blue)
![Biodiversity](https://img.shields.io/badge/Biodiversity-Research-green)
![Data](https://img.shields.io/badge/Data-Science-orange)
