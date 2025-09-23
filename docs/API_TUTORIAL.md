# 🔌 Ocean-Bio Phase 2 API Tutorial

## Complete Integration Guide for Developers

**Version**: 2.0.0  
**Last Updated**: September 2024  
**Target Audience**: Developers, data scientists, and system integrators

---

## 📋 **Table of Contents**

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Geospatial API](#geospatial-api)
4. [Predictive Modeling API](#predictive-modeling-api)
5. [Genomics API](#genomics-api)
6. [Data Management API](#data-management-api)
7. [Real-time Integration](#real-time-integration)
8. [Best Practices](#best-practices)
9. [Code Examples](#code-examples)
10. [Error Handling](#error-handling)

---

## 🚀 **Getting Started**

### **Base URL and Versioning**

```
Base URL: https://your-domain.com/api/v2
Documentation: https://your-domain.com/docs
```

### **Content Types**

All API endpoints accept and return JSON unless otherwise specified.

```http
Content-Type: application/json
Accept: application/json
```

### **HTTP Methods**

| Method | Usage | Description |
|--------|-------|-------------|
| `GET` | Retrieve data | Read operations |
| `POST` | Create resources | Submit new data |
| `PUT` | Update resources | Full resource updates |
| `PATCH` | Partial updates | Modify specific fields |
| `DELETE` | Remove resources | Delete operations |

### **Rate Limiting**

```
Rate Limits:
- Standard users: 1000 requests/hour
- Premium users: 5000 requests/hour
- Batch operations: 100 requests/hour

Headers returned:
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1632150000
```

---

## 🔐 **Authentication**

### **JWT Token Authentication**

#### **Login and Get Token**

```bash
curl -X POST "https://your-domain.com/api/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "your_username",
       "password": "your_password"
     }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_info": {
    "user_id": 123,
    "username": "your_username",
    "role": "researcher"
  }
}
```

#### **Using the Token**

Include the token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     "https://your-domain.com/api/v2/biodiversity/"
```

#### **Refresh Token**

```bash
curl -X POST "https://your-domain.com/api/auth/refresh" \
     -H "Content-Type: application/json" \
     -d '{
       "refresh_token": "your_refresh_token"
     }'
```

### **API Key Authentication (Alternative)**

For server-to-server communication:

```bash
curl -H "X-API-Key: your_api_key" \
     "https://your-domain.com/api/v2/biodiversity/"
```

---

## 🗺️ **Geospatial API**

### **Spatial Queries**

#### **Location-based Query**

Get all data within a radius of a specific point:

```bash
POST /api/v2/geospatial/query/location
```

**Request:**
```json
{
  "latitude": 19.0760,
  "longitude": 72.8777,
  "radius_km": 10,
  "data_types": ["biodiversity", "water_quality"],
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-09-01T00:00:00Z",
  "limit": 1000,
  "include_geometry": true
}
```

**Response:**
```json
{
  "query_id": "query_12345",
  "total_results": 1847,
  "returned_results": 1000,
  "data": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [72.8777, 19.0760]
        },
        "properties": {
          "id": "bio_001",
          "data_type": "biodiversity",
          "species": "Thunnus albacares",
          "abundance": 25,
          "collection_date": "2024-01-15T08:00:00Z",
          "method": "trawl"
        }
      }
    ]
  },
  "metadata": {
    "query_time_ms": 245,
    "center_point": [72.8777, 19.0760],
    "radius_km": 10,
    "data_types_found": ["biodiversity", "water_quality"]
  }
}
```

#### **Bounding Box Query**

Get data within a rectangular area:

```bash
POST /api/v2/geospatial/query/bbox
```

**Request:**
```json
{
  "min_lat": 18.0,
  "max_lat": 20.0,
  "min_lon": 72.0,
  "max_lon": 74.0,
  "data_types": ["fisheries"],
  "temporal_filter": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "species_filter": ["Pomfret", "Mackerel"],
  "aggregation": "monthly"
}
```

#### **Custom Polygon Query**

Query within a custom polygon:

```bash
POST /api/v2/geospatial/query/polygon
```

**Request:**
```json
{
  "polygon": {
    "type": "Polygon",
    "coordinates": [[
      [72.0, 18.0],
      [74.0, 18.0], 
      [74.0, 20.0],
      [72.0, 20.0],
      [72.0, 18.0]
    ]]
  },
  "data_types": ["biodiversity"],
  "aggregation_level": "species"
}
```

### **Spatial Analysis**

#### **Hotspot Analysis**

Identify statistical clusters of high/low values:

```bash
POST /api/v2/geospatial/analysis/hotspots
```

**Request:**
```json
{
  "data_layer": "biodiversity",
  "analysis_field": "abundance",
  "confidence_level": 0.95,
  "distance_band": 1000,
  "analysis_extent": {
    "min_lat": 18.0,
    "max_lat": 20.0,
    "min_lon": 72.0,
    "max_lon": 74.0
  },
  "null_hypothesis": "randomness"
}
```

**Response:**
```json
{
  "analysis_id": "hotspot_67890",
  "status": "completed",
  "results": {
    "hotspots": [
      {
        "geometry": {
          "type": "Point", 
          "coordinates": [72.95, 19.25]
        },
        "properties": {
          "gi_z_score": 3.45,
          "p_value": 0.0006,
          "confidence_level": "99%",
          "cluster_type": "hot"
        }
      }
    ],
    "coldspots": [
      {
        "geometry": {
          "type": "Point",
          "coordinates": [72.15, 18.75]
        },
        "properties": {
          "gi_z_score": -2.87,
          "p_value": 0.0041,
          "confidence_level": "95%",
          "cluster_type": "cold"
        }
      }
    ],
    "statistics": {
      "total_features": 1250,
      "significant_hot": 23,
      "significant_cold": 18,
      "not_significant": 1209
    }
  }
}
```

#### **Cluster Analysis**

Group similar data points:

```bash
POST /api/v2/geospatial/analysis/clusters
```

**Request:**
```json
{
  "data_points": [
    {"lat": 19.0760, "lon": 72.8777, "value": 25},
    {"lat": 19.0850, "lon": 72.8650, "value": 30}
  ],
  "method": "dbscan",
  "parameters": {
    "eps": 0.1,
    "min_samples": 5
  },
  "distance_metric": "haversine"
}
```

#### **Interpolation**

Create continuous surfaces from point data:

```bash
POST /api/v2/geospatial/analysis/interpolation
```

**Request:**
```json
{
  "points": [
    {"lat": 19.0760, "lon": 72.8777, "value": 25.5},
    {"lat": 19.0850, "lon": 72.8650, "value": 26.2}
  ],
  "method": "kriging",
  "output_format": "raster",
  "grid_resolution": 0.01,
  "extent": {
    "min_lat": 18.5, "max_lat": 19.5,
    "min_lon": 72.3, "max_lon": 73.3
  }
}
```

### **Mapping Services**

#### **Generate Map**

Create interactive maps:

```bash
POST /api/v2/geospatial/maps/generate
```

**Request:**
```json
{
  "map_type": "biodiversity_hotspots",
  "region": {
    "center": {"lat": 19.0760, "lon": 72.8777},
    "zoom": 10
  },
  "layers": [
    {
      "type": "biodiversity",
      "style": {
        "color_scheme": "viridis",
        "marker_size": "abundance",
        "transparency": 0.7
      }
    },
    {
      "type": "bathymetry", 
      "style": {
        "contour_lines": true,
        "color_ramp": "depth"
      }
    }
  ],
  "export_format": "interactive"
}
```

**Response:**
```json
{
  "map_id": "map_abc123",
  "map_url": "https://your-domain.com/maps/map_abc123",
  "embed_code": "<iframe src='...' width='800' height='600'></iframe>",
  "metadata": {
    "creation_date": "2024-09-23T10:30:00Z",
    "layer_count": 2,
    "feature_count": 1847,
    "bounds": {
      "north": 19.5, "south": 18.5,
      "east": 73.3, "west": 72.3
    }
  }
}
```

---

## 📈 **Predictive Modeling API**

### **Stock Assessment**

#### **Available Species**

Get list of species with assessment data:

```bash
GET /api/v2/predictive/stock/species
```

**Response:**
```json
{
  "species": [
    {
      "name": "Pomfret",
      "scientific_name": "Pampus argenteus",
      "stock_id": "pomfret_arabian_sea",
      "assessment_history": {
        "last_assessment": "2024-01-15",
        "assessment_count": 12,
        "data_availability": "1995-2024"
      },
      "current_status": "unknown"
    }
  ]
}
```

#### **Run Stock Assessment**

Perform stock assessment analysis:

```bash
POST /api/v2/predictive/stock/assess
```

**Request:**
```json
{
  "species": "Pomfret",
  "assessment_type": "surplus_production",
  "time_period": {
    "start_year": 2020,
    "end_year": 2024
  },
  "data_sources": {
    "catch_data": true,
    "effort_data": true,
    "survey_data": false
  },
  "biological_parameters": {
    "natural_mortality": 0.2,
    "max_age": 15,
    "length_weight_a": 0.01,
    "length_weight_b": 3.0,
    "maturity_length": 25.0
  },
  "model_options": {
    "bootstrap_iterations": 1000,
    "confidence_intervals": [0.8, 0.95]
  }
}
```

**Response:**
```json
{
  "assessment_id": "assess_xyz789",
  "status": "completed",
  "completion_time": "2024-09-23T11:15:30Z",
  "results": {
    "reference_points": {
      "MSY": 1250.5,
      "BMSY": 8940.2, 
      "FMSY": 0.14,
      "B0": 17880.4
    },
    "current_estimates": {
      "current_biomass": 5364.1,
      "current_fishing_mortality": 0.31,
      "biomass_ratio": 0.60,
      "f_ratio": 2.21
    },
    "stock_status": {
      "overfished": false,
      "overfishing": true,
      "status_category": "overfishing_occurring"
    },
    "confidence_intervals": {
      "current_biomass": {
        "lower_80": 4891.7,
        "upper_80": 5836.5,
        "lower_95": 4419.3,
        "upper_95": 6308.9
      }
    },
    "recommendations": [
      {
        "category": "immediate",
        "description": "Reduce fishing mortality by 30%",
        "priority": "high",
        "timeframe": "1_year"
      }
    ]
  }
}
```

### **Forecasting**

#### **Create Forecast**

Generate predictions for marine time series:

```bash
POST /api/v2/predictive/forecasting/create
```

**Request:**
```json
{
  "data_source": {
    "type": "species_abundance",
    "species": "Thunnus albacares",
    "region": "arabian_sea",
    "time_series": "monthly"
  },
  "forecast_horizon": 12,
  "model_type": "prophet",
  "model_parameters": {
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": true,
    "weekly_seasonality": false,
    "changepoint_prior_scale": 0.05
  },
  "external_variables": [
    "sea_surface_temperature",
    "chlorophyll_concentration"
  ],
  "confidence_intervals": [0.80, 0.95],
  "validation": {
    "method": "time_series_cv",
    "n_splits": 5
  }
}
```

**Response:**
```json
{
  "forecast_id": "forecast_def456",
  "status": "completed",
  "model_performance": {
    "mae": 12.4,
    "rmse": 18.7,
    "mape": 15.2,
    "r2_score": 0.78
  },
  "forecast_values": [
    {
      "date": "2024-10-01",
      "predicted_value": 45.2,
      "lower_80": 38.1,
      "upper_80": 52.3,
      "lower_95": 34.6,
      "upper_95": 55.8
    }
  ],
  "components": {
    "trend": [42.1, 42.3, 42.5],
    "seasonal": [3.1, 2.9, 2.7],
    "residual": [0.1, -0.2, 0.3]
  }
}
```

#### **Batch Forecasting**

Create forecasts for multiple series:

```bash
POST /api/v2/predictive/forecasting/batch
```

**Request:**
```json
{
  "batch_name": "Multi-species Forecast 2024",
  "series_list": [
    {
      "id": "pomfret_abundance", 
      "species": "Pomfret",
      "region": "maharashtra_coast"
    },
    {
      "id": "mackerel_abundance",
      "species": "Mackerel", 
      "region": "maharashtra_coast"
    }
  ],
  "common_parameters": {
    "forecast_horizon": 6,
    "model_type": "arima",
    "confidence_interval": 0.95
  }
}
```

### **Trend Analysis**

#### **Linear Trend Detection**

Identify monotonic trends in time series:

```bash
POST /api/v2/predictive/trends/linear
```

**Request:**
```json
{
  "time_series_data": [
    {"date": "2020-01", "value": 45.2},
    {"date": "2020-02", "value": 47.1},
    {"date": "2020-03", "value": 46.8}
  ],
  "trend_tests": ["mann_kendall", "linear_regression"],
  "significance_level": 0.05,
  "seasonal_adjustment": true
}
```

**Response:**
```json
{
  "analysis_id": "trend_ghi789", 
  "trend_detected": true,
  "trend_direction": "increasing",
  "trend_strength": 0.67,
  "statistical_tests": {
    "mann_kendall": {
      "tau": 0.342,
      "p_value": 0.003,
      "significant": true
    },
    "linear_regression": {
      "slope": 0.185,
      "r_squared": 0.45,
      "p_value": 0.001
    }
  },
  "trend_rate": "0.185 units per month",
  "confidence_interval": {
    "lower": 0.067,
    "upper": 0.303
  }
}
```

#### **Changepoint Detection**

Find significant shifts in data patterns:

```bash
POST /api/v2/predictive/trends/changepoints
```

**Request:**
```json
{
  "time_series_data": [
    {"date": "2020-01-01", "value": 45.2},
    {"date": "2020-02-01", "value": 47.1}
  ],
  "method": "pelt",
  "min_segment_length": 6,
  "penalty_value": "auto"
}
```

---

## 🧬 **Genomics API**

### **Sequence Processing**

#### **Upload and Process Sequences**

Process FASTA files with quality control:

```bash
POST /api/v2/genomics/sequences/process
Content-Type: multipart/form-data
```

**Request (Form Data):**
```
file: [FASTA file upload]
min_length: 100
max_length: 2000
max_ambiguous: 0.05
min_complexity: 0.3
remove_duplicates: true
```

**Response:**
```json
{
  "processing_id": "proc_seq123",
  "status": "completed",
  "summary": {
    "total_sequences": 2547,
    "passed_filters": 2234,
    "failed_length": 198,
    "failed_ambiguous": 89,
    "failed_complexity": 26,
    "duplicates_removed": 156
  },
  "quality_metrics": {
    "average_length": 487.3,
    "median_length": 465,
    "gc_content_mean": 42.8,
    "gc_content_std": 8.4
  },
  "processed_sequences": {
    "download_url": "https://your-domain.com/api/downloads/proc_seq123.fasta",
    "expiry": "2024-09-30T00:00:00Z"
  }
}
```

#### **Sequence Validation**

Validate individual sequences:

```bash
POST /api/v2/genomics/sequences/validate
```

**Request:**
```json
{
  "sequences": [
    {
      "id": "seq_1",
      "sequence": "ATCGATCGATCGATCGATCGATCG"
    },
    {
      "id": "seq_2", 
      "sequence": "GCTAGCTAGCTAGCTAGCTAGCTA"
    }
  ],
  "validation_rules": {
    "min_length": 20,
    "max_ambiguous_ratio": 0.1,
    "allowed_bases": ["A", "T", "C", "G", "N"]
  }
}
```

### **Taxonomic Classification**

#### **BLAST Classification**

Classify sequences using BLAST search:

```bash
POST /api/v2/genomics/taxonomy/blast
```

**Request:**
```json
{
  "sequences": [
    {
      "id": "marine_seq_1",
      "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    }
  ],
  "database": "ncbi_nt",
  "search_parameters": {
    "e_value": 1e-5,
    "word_size": 11,
    "max_target_seqs": 10
  },
  "classification_criteria": {
    "identity_threshold": 95.0,
    "coverage_threshold": 80.0,
    "max_e_value": 1e-10
  },
  "output_format": "detailed"
}
```

**Response:**
```json
{
  "classification_id": "blast_jkl012",
  "status": "completed",
  "results": [
    {
      "query_id": "marine_seq_1",
      "classifications": [
        {
          "rank": 1,
          "accession": "NR_123456",
          "description": "Thunnus albacares COI gene",
          "identity": 98.5,
          "coverage": 95.2,
          "e_value": 1.2e-150,
          "bit_score": 558,
          "taxonomy": {
            "kingdom": "Eukaryota",
            "phylum": "Chordata", 
            "class": "Actinopterygii",
            "order": "Perciformes",
            "family": "Scombridae",
            "genus": "Thunnus",
            "species": "Thunnus albacares"
          }
        }
      ],
      "best_match": {
        "species": "Thunnus albacares",
        "confidence_score": 0.98,
        "classification_method": "top_hit"
      }
    }
  ],
  "summary": {
    "total_sequences": 1,
    "classified": 1,
    "unclassified": 0,
    "processing_time_seconds": 45.3
  }
}
```

#### **Kraken2 Classification**

Fast k-mer based classification:

```bash
POST /api/v2/genomics/taxonomy/kraken2
```

**Request:**
```json
{
  "sequences": [
    {
      "id": "seq_001",
      "sequence": "ATCGATCGATCGATCGATCGATCG"
    }
  ],
  "database": "minikraken2_v2",
  "confidence_threshold": 0.1,
  "output_unclassified": false
}
```

### **Diversity Analysis**

#### **Alpha Diversity**

Calculate within-sample diversity:

```bash
POST /api/v2/genomics/diversity/alpha
```

**Request:**
```json
{
  "community_data": [
    {
      "sample_id": "site_A",
      "species_counts": {
        "Thunnus albacares": 25,
        "Sardinella longiceps": 150,
        "Rastrelliger kanagurta": 89
      }
    },
    {
      "sample_id": "site_B", 
      "species_counts": {
        "Thunnus albacares": 12,
        "Katsuwonus pelamis": 67,
        "Auxis thazard": 34
      }
    }
  ],
  "metrics": ["shannon", "simpson", "chao1", "ace"],
  "rarefaction": {
    "enabled": true,
    "max_depth": 200,
    "step_size": 10
  }
}
```

**Response:**
```json
{
  "analysis_id": "alpha_div_mno345",
  "results": {
    "site_A": {
      "observed_species": 3,
      "shannon": 1.847,
      "simpson": 0.723,
      "chao1": 3.0,
      "ace": 3.2,
      "rarefaction_curve": {
        "sample_sizes": [10, 20, 30, 40, 50],
        "species_counts": [1.8, 2.4, 2.7, 2.9, 3.0],
        "confidence_intervals": {
          "lower": [1.2, 1.9, 2.3, 2.5, 2.7],
          "upper": [2.4, 2.9, 3.1, 3.3, 3.3]
        }
      }
    }
  }
}
```

#### **Beta Diversity**

Compare diversity between samples:

```bash
POST /api/v2/genomics/diversity/beta
```

**Request:**
```json
{
  "community_matrix": {
    "samples": ["site_A", "site_B", "site_C"],
    "species": ["Species_1", "Species_2", "Species_3"],
    "abundance_matrix": [
      [25, 15, 8],
      [12, 22, 5],
      [8, 18, 12]
    ]
  },
  "distance_metric": "bray_curtis",
  "ordination_method": "pcoa",
  "permanova_test": true
}
```

### **Phylogenetic Analysis**

#### **Multiple Sequence Alignment**

Align sequences for phylogenetic analysis:

```bash
POST /api/v2/genomics/phylogenetics/align
```

**Request:**
```json
{
  "sequences": [
    {"id": "seq1", "sequence": "ATCGATCGATCGATCG"},
    {"id": "seq2", "sequence": "ATCGAGATCGATCGATCG"},
    {"id": "seq3", "sequence": "ATCGATGATCGATCG"}
  ],
  "alignment_method": "muscle",
  "parameters": {
    "max_iterations": 16,
    "gap_open_penalty": -2.9,
    "gap_extend_penalty": 0
  }
}
```

#### **Phylogenetic Tree Construction**

Build evolutionary trees:

```bash
POST /api/v2/genomics/phylogenetics/tree
```

**Request:**
```json
{
  "aligned_sequences": [
    {"id": "seq1", "sequence": "ATCGATCGATCGATCG--"},
    {"id": "seq2", "sequence": "ATCGAGATCGATCGATCG"},
    {"id": "seq3", "sequence": "ATCGAT-GATCGATCG--"}
  ],
  "tree_method": "neighbor_joining",
  "substitution_model": "jukes_cantor", 
  "bootstrap_replicates": 100,
  "outgroup": "seq3"
}
```

**Response:**
```json
{
  "tree_id": "tree_pqr678",
  "status": "completed",
  "newick_tree": "((seq1:0.12,seq2:0.08):0.06,seq3:0.15);",
  "tree_statistics": {
    "total_branch_length": 0.41,
    "internal_nodes": 1,
    "tips": 3,
    "bootstrap_support": {
      "average": 87.5,
      "minimum": 72,
      "maximum": 96
    }
  },
  "visualization_url": "https://your-domain.com/trees/tree_pqr678"
}
```

---

## 💾 **Data Management API**

### **Data Upload**

#### **Bulk Data Import**

Upload large datasets:

```bash
POST /api/v2/data/upload/bulk
Content-Type: multipart/form-data
```

**Request (Form Data):**
```
data_type: biodiversity
file: [CSV file upload]
metadata: {
  "collection_method": "trawl",
  "project": "Maharashtra Coast Survey 2024",
  "quality_level": "research_grade"
}
```

#### **Real-time Data Streaming**

Stream live data:

```bash
POST /api/v2/data/stream
```

**Request:**
```json
{
  "stream_id": "live_sensor_001",
  "data_points": [
    {
      "timestamp": "2024-09-23T12:00:00Z",
      "location": {"lat": 19.0760, "lon": 72.8777},
      "measurements": {
        "temperature": 26.5,
        "salinity": 35.2,
        "dissolved_oxygen": 7.1
      }
    }
  ],
  "metadata": {
    "sensor_id": "WQ_001",
    "sampling_frequency": "1_minute"
  }
}
```

### **Data Export**

#### **Query and Export**

Export data in various formats:

```bash
POST /api/v2/data/export
```

**Request:**
```json
{
  "query": {
    "data_types": ["biodiversity", "water_quality"],
    "spatial_filter": {
      "type": "bbox",
      "coordinates": [72.0, 18.0, 74.0, 20.0]
    },
    "temporal_filter": {
      "start": "2024-01-01",
      "end": "2024-09-01"
    }
  },
  "export_format": "csv",
  "include_metadata": true,
  "compression": "gzip"
}
```

**Response:**
```json
{
  "export_id": "export_stu901",
  "status": "processing",
  "estimated_completion": "2024-09-23T12:15:00Z",
  "download_url": null,
  "file_size_estimate": "25.4 MB",
  "record_count": 15847
}
```

---

## ⚡ **Real-time Integration**

### **WebSocket Connections**

Connect to real-time data streams:

```javascript
// JavaScript WebSocket example
const ws = new WebSocket('wss://your-domain.com/ws/v2/realtime');

ws.onopen = function() {
    // Subscribe to specific data streams
    ws.send(JSON.stringify({
        'action': 'subscribe',
        'streams': ['water_quality_updates', 'biodiversity_alerts'],
        'filters': {
            'region': 'maharashtra_coast',
            'alert_level': 'medium'
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

### **Webhooks**

Configure webhooks for event notifications:

```bash
POST /api/v2/webhooks/register
```

**Request:**
```json
{
  "webhook_url": "https://your-app.com/webhooks/oceanbio",
  "events": [
    "analysis_completed",
    "data_upload_finished", 
    "alert_triggered"
  ],
  "filters": {
    "user_id": 123,
    "data_types": ["biodiversity"]
  },
  "secret_key": "your_webhook_secret"
}
```

---

## ✅ **Best Practices**

### **Performance Optimization**

#### **Pagination**

Always use pagination for large datasets:

```bash
GET /api/v2/biodiversity/?page=1&page_size=100
```

#### **Field Selection**

Request only needed fields:

```bash
GET /api/v2/biodiversity/?fields=id,species,abundance,location
```

#### **Caching**

Utilize caching headers:

```bash
# Cache results for 1 hour
Cache-Control: public, max-age=3600
```

### **Error Handling**

#### **Retry Logic**

Implement exponential backoff:

```python
import time
import requests

def api_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [429, 502, 503, 504]:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                break
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)
    
    return None
```

#### **Status Monitoring**

Check API status before making requests:

```bash
GET /api/v2/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "services": {
    "database": "operational",
    "redis": "operational", 
    "external_apis": "operational"
  },
  "response_time_ms": 45
}
```

### **Security**

#### **Token Management**

- Store tokens securely
- Refresh tokens before expiry
- Use HTTPS for all requests
- Implement proper CORS policies

#### **Rate Limiting**

- Monitor rate limit headers
- Implement client-side throttling
- Use batch operations when possible

---

## 💻 **Code Examples**

### **Python SDK**

```python
import requests
import json
from typing import Dict, List, Optional

class OceanBioAPI:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def spatial_query(self, lat: float, lon: float, radius_km: float, 
                     data_types: List[str]) -> Dict:
        """Perform location-based spatial query"""
        endpoint = f"{self.base_url}/api/v2/geospatial/query/location"
        payload = {
            "latitude": lat,
            "longitude": lon, 
            "radius_km": radius_km,
            "data_types": data_types
        }
        
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    
    def classify_sequences(self, sequences: List[Dict], method: str = "blast") -> Dict:
        """Classify DNA sequences taxonomically"""
        endpoint = f"{self.base_url}/api/v2/genomics/taxonomy/{method}"
        payload = {"sequences": sequences}
        
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    
    def stock_assessment(self, species: str, assessment_type: str, 
                        start_year: int, end_year: int) -> Dict:
        """Run stock assessment analysis"""
        endpoint = f"{self.base_url}/api/v2/predictive/stock/assess"
        payload = {
            "species": species,
            "assessment_type": assessment_type,
            "time_period": {
                "start_year": start_year,
                "end_year": end_year
            }
        }
        
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()

# Usage example
api = OceanBioAPI('https://your-domain.com', 'your_api_key')

# Spatial query
results = api.spatial_query(
    lat=19.0760, 
    lon=72.8777, 
    radius_km=10,
    data_types=['biodiversity', 'water_quality']
)

# Sequence classification  
sequences = [
    {"id": "seq1", "sequence": "ATCGATCGATCGATCG"},
    {"id": "seq2", "sequence": "GCTAGCTAGCTAGCTA"}
]
classification = api.classify_sequences(sequences, method="blast")

# Stock assessment
assessment = api.stock_assessment(
    species="Pomfret",
    assessment_type="surplus_production", 
    start_year=2020,
    end_year=2024
)
```

### **JavaScript/Node.js**

```javascript
class OceanBioAPI {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async spatialQuery(lat, lon, radiusKm, dataTypes) {
        const endpoint = `${this.baseUrl}/api/v2/geospatial/query/location`;
        const payload = {
            latitude: lat,
            longitude: lon,
            radius_km: radiusKm,
            data_types: dataTypes
        };
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async createForecast(dataSource, horizonMonths, modelType = 'arima') {
        const endpoint = `${this.baseUrl}/api/v2/predictive/forecasting/create`;
        const payload = {
            data_source: dataSource,
            forecast_horizon: horizonMonths,
            model_type: modelType,
            confidence_intervals: [0.80, 0.95]
        };
        
        const response = await fetch(endpoint, {
            method: 'POST', 
            headers: this.headers,
            body: JSON.stringify(payload)
        });
        
        return await response.json();
    }
    
    async uploadSequences(file, options = {}) {
        const endpoint = `${this.baseUrl}/api/v2/genomics/sequences/process`;
        const formData = new FormData();
        
        formData.append('file', file);
        Object.entries(options).forEach(([key, value]) => {
            formData.append(key, value);
        });
        
        const headers = { ...this.headers };
        delete headers['Content-Type']; // Let browser set multipart boundary
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: headers,
            body: formData
        });
        
        return await response.json();
    }
}

// Usage
const api = new OceanBioAPI('https://your-domain.com', 'your_api_key');

// Spatial query with async/await
try {
    const spatialResults = await api.spatialQuery(
        19.0760, 72.8777, 10, 
        ['biodiversity', 'water_quality']
    );
    console.log('Spatial query results:', spatialResults);
} catch (error) {
    console.error('Spatial query failed:', error);
}

// Create forecast
api.createForecast({
    type: 'species_abundance',
    species: 'Thunnus albacares',
    region: 'arabian_sea'
}, 12, 'prophet')
.then(forecast => console.log('Forecast created:', forecast))
.catch(error => console.error('Forecast failed:', error));
```

### **R Package**

```r
# R wrapper for Ocean-Bio API
library(httr)
library(jsonlite)

OceanBioAPI <- R6::R6Class("OceanBioAPI",
  public = list(
    base_url = NULL,
    api_key = NULL,
    
    initialize = function(base_url, api_key) {
      self$base_url = gsub("/$", "", base_url)
      self$api_key = api_key
    },
    
    spatial_query = function(lat, lon, radius_km, data_types) {
      endpoint <- paste0(self$base_url, "/api/v2/geospatial/query/location")
      
      payload <- list(
        latitude = lat,
        longitude = lon,
        radius_km = radius_km,
        data_types = data_types
      )
      
      response <- POST(
        endpoint,
        add_headers(Authorization = paste("Bearer", self$api_key)),
        body = payload,
        encode = "json"
      )
      
      stop_for_status(response)
      return(content(response, as = "parsed"))
    },
    
    diversity_analysis = function(community_data, metrics = c("shannon", "simpson")) {
      endpoint <- paste0(self$base_url, "/api/v2/genomics/diversity/alpha")
      
      payload <- list(
        community_data = community_data,
        metrics = metrics
      )
      
      response <- POST(
        endpoint,
        add_headers(Authorization = paste("Bearer", self$api_key)),
        body = payload,
        encode = "json"
      )
      
      stop_for_status(response)
      return(content(response, as = "parsed"))
    }
  )
)

# Usage
api <- OceanBioAPI$new("https://your-domain.com", "your_api_key")

# Spatial query
results <- api$spatial_query(19.0760, 72.8777, 10, c("biodiversity"))

# Diversity analysis
community_data <- list(
  list(
    sample_id = "site_A",
    species_counts = list(
      "Thunnus albacares" = 25,
      "Sardinella longiceps" = 150
    )
  )
)

diversity <- api$diversity_analysis(community_data, c("shannon", "simpson", "chao1"))
```

---

## ⚠️ **Error Handling**

### **HTTP Status Codes**

| Code | Meaning | Action |
|------|---------|--------|
| **200** | Success | Process response |
| **201** | Created | Resource created successfully |
| **400** | Bad Request | Check request format |
| **401** | Unauthorized | Refresh authentication |
| **403** | Forbidden | Check permissions |
| **404** | Not Found | Verify endpoint URL |
| **422** | Validation Error | Fix request parameters |
| **429** | Rate Limited | Implement backoff |
| **500** | Server Error | Retry later |
| **503** | Service Unavailable | Check system status |

### **Error Response Format**

```json
{
  "error": {
    "code": "INVALID_COORDINATES",
    "message": "Latitude must be between -90 and 90 degrees",
    "details": {
      "field": "latitude",
      "provided_value": 95.0,
      "valid_range": "[-90, 90]"
    },
    "request_id": "req_abc123",
    "timestamp": "2024-09-23T12:00:00Z"
  }
}
```

### **Common Error Scenarios**

#### **Authentication Errors**

```json
{
  "error": {
    "code": "INVALID_TOKEN",
    "message": "JWT token has expired",
    "details": {
      "expired_at": "2024-09-23T10:00:00Z",
      "current_time": "2024-09-23T12:00:00Z"
    }
  }
}
```

**Solution**: Refresh the access token using refresh token.

#### **Validation Errors**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "errors": [
        {
          "field": "radius_km",
          "message": "Must be between 0.1 and 1000"
        },
        {
          "field": "data_types",
          "message": "Must contain at least one data type"
        }
      ]
    }
  }
}
```

#### **Rate Limiting**

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED", 
    "message": "API rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": "1 hour",
      "retry_after": 1800
    }
  }
}
```

**Solution**: Wait for the specified `retry_after` seconds before making new requests.

---

## 📞 **Support and Resources**

### **Documentation**
- **API Reference**: https://your-domain.com/docs
- **User Manual**: https://docs.ocean-bio.org/user-manual
- **Code Examples**: https://github.com/ocean-bio/examples

### **Support Channels**
- **Technical Support**: api-support@ocean-bio.org
- **GitHub Issues**: https://github.com/ocean-bio/platform/issues
- **Community Forum**: https://community.ocean-bio.org

### **SDKs and Libraries**
- **Python**: `pip install oceanbio-sdk`
- **JavaScript**: `npm install oceanbio-js`
- **R**: `devtools::install_github("ocean-bio/oceanbio-r")`

### **Postman Collection**
Import our Postman collection for easy API testing:
```
https://your-domain.com/api/v2/postman-collection.json
```

---

**© 2024 Ocean-Bio Development Team. All rights reserved.**