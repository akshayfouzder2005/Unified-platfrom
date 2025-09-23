"""
🗺️ GIS Manager - PostGIS Integration & Spatial Database Operations

Advanced geospatial database management with PostGIS integration.
Handles spatial queries, geometric operations, and coordinate transformations.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from geoalchemy2 import Geometry
from sqlalchemy import func
from sqlalchemy import text, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely import wkt, wkb
import json

logger = logging.getLogger(__name__)

class GISManager:
    """
    🗺️ Advanced GIS Manager with PostGIS Integration
    
    Provides comprehensive spatial database operations:
    - Spatial queries and geometric operations
    - Coordinate transformations and projections
    - Spatial indexing and performance optimization
    - Integration with all 5 data model types
    """
    
    def __init__(self, database_url: str):
        """Initialize GIS Manager with PostGIS database connection"""
        self.database_url = database_url
        self.engine = None
        self.supported_crs = {
            'WGS84': 4326,           # Global GPS coordinates
            'WEB_MERCATOR': 3857,    # Web mapping standard
            'UTM_43N': 32643,        # India region UTM
            'INDIAN_1975': 24378     # Indian geodetic datum
        }
        self._initialize_postgis()
    
    def _initialize_postgis(self) -> bool:
        """Initialize PostGIS extension and spatial capabilities"""
        try:
            self.engine = create_engine(self.database_url)
            
            # Only attempt PostGIS initialization for PostgreSQL
            if self.engine.url.get_backend_name() == "postgresql":
                with self.engine.connect() as conn:
                    # Enable PostGIS extensions
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology;"))
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;"))
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;"))
                    conn.commit()
                    
                logger.info("🗺️ PostGIS extensions initialized successfully")
            else:
                logger.info(f"🗺️ Skipping PostGIS initialization for {self.engine.url.get_backend_name()} backend")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ PostGIS initialization failed (this is normal for SQLite): {e}")
            return False
    
    def create_spatial_tables(self) -> bool:
        """Create spatial tables for Phase 2 geospatial data"""
        try:
            with self.engine.connect() as conn:
                
                # Spatial layers table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spatial_layers (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        description TEXT,
                        layer_type VARCHAR(50) NOT NULL,
                        data_model_type VARCHAR(50) NOT NULL,
                        geometry_type VARCHAR(50) NOT NULL,
                        crs INTEGER DEFAULT 4326,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB,
                        is_active BOOLEAN DEFAULT true
                    );
                """))
                
                # Geographic regions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS geographic_regions (
                        id SERIAL PRIMARY KEY,
                        region_name VARCHAR(200) NOT NULL,
                        region_type VARCHAR(100) NOT NULL, -- marine_protected_area, fishing_zone, etc.
                        geometry GEOMETRY(POLYGON, 4326) NOT NULL,
                        area_km2 REAL,
                        perimeter_km REAL,
                        data_model_types TEXT[], -- which data types apply to this region
                        properties JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                
                # Spatial points for samples/specimens/stations
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spatial_points (
                        id SERIAL PRIMARY KEY,
                        point_name VARCHAR(200),
                        data_model_type VARCHAR(50) NOT NULL,
                        record_id INTEGER NOT NULL, -- reference to actual data record
                        location GEOMETRY(POINT, 4326) NOT NULL,
                        elevation_m REAL,
                        depth_m REAL,
                        collection_date TIMESTAMP,
                        properties JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                
                # Spatial tracks for vessels/migrations/etc.
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spatial_tracks (
                        id SERIAL PRIMARY KEY,
                        track_name VARCHAR(200),
                        data_model_type VARCHAR(50) NOT NULL,
                        track_type VARCHAR(100) NOT NULL, -- vessel_route, migration_path, etc.
                        geometry GEOMETRY(LINESTRING, 4326) NOT NULL,
                        length_km REAL,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        properties JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                
                # Create spatial indexes
                spatial_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_geographic_regions_geometry ON geographic_regions USING GIST(geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_spatial_points_location ON spatial_points USING GIST(location);",
                    "CREATE INDEX IF NOT EXISTS idx_spatial_tracks_geometry ON spatial_tracks USING GIST(geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_spatial_points_data_model ON spatial_points(data_model_type);",
                    "CREATE INDEX IF NOT EXISTS idx_spatial_tracks_data_model ON spatial_tracks(data_model_type);"
                ]
                
                for index_sql in spatial_indexes:
                    conn.execute(text(index_sql))
                
                conn.commit()
                logger.info("🗺️ Spatial tables created successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Spatial table creation failed: {e}")
            return False
    
    def add_spatial_point(self, 
                         data_model_type: str,
                         record_id: int,
                         latitude: float,
                         longitude: float,
                         point_name: str = None,
                         elevation_m: float = None,
                         depth_m: float = None,
                         collection_date: datetime = None,
                         properties: dict = None) -> Optional[int]:
        """Add a spatial point to the database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO spatial_points 
                    (point_name, data_model_type, record_id, location, elevation_m, depth_m, collection_date, properties)
                    VALUES 
                    (:point_name, :data_model_type, :record_id, ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326), 
                     :elevation_m, :depth_m, :collection_date, :properties)
                    RETURNING id;
                """), {
                    'point_name': point_name,
                    'data_model_type': data_model_type,
                    'record_id': record_id,
                    'latitude': latitude,
                    'longitude': longitude,
                    'elevation_m': elevation_m,
                    'depth_m': depth_m,
                    'collection_date': collection_date,
                    'properties': json.dumps(properties) if properties else None
                })
                conn.commit()
                
                spatial_id = result.fetchone()[0]
                logger.info(f"🗺️ Spatial point added: {spatial_id} for {data_model_type}")
                return spatial_id
                
        except Exception as e:
            logger.error(f"❌ Failed to add spatial point: {e}")
            return None
    
    def find_points_within_radius(self,
                                 center_lat: float,
                                 center_lon: float,
                                 radius_km: float,
                                 data_model_type: str = None) -> List[Dict[str, Any]]:
        """Find spatial points within specified radius"""
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT 
                        id, point_name, data_model_type, record_id,
                        ST_X(location) as longitude,
                        ST_Y(location) as latitude,
                        ST_Distance(location, ST_SetSRID(ST_MakePoint(:center_lon, :center_lat), 4326)::geography) / 1000 as distance_km,
                        elevation_m, depth_m, collection_date, properties
                    FROM spatial_points
                    WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint(:center_lon, :center_lat), 4326)::geography, :radius_m)
                """
                
                params = {
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'radius_m': radius_km * 1000  # Convert km to meters
                }
                
                if data_model_type:
                    query += " AND data_model_type = :data_model_type"
                    params['data_model_type'] = data_model_type
                
                query += " ORDER BY distance_km"
                
                result = conn.execute(text(query), params)
                points = [dict(row._mapping) for row in result]
                
                logger.info(f"🗺️ Found {len(points)} points within {radius_km}km")
                return points
                
        except Exception as e:
            logger.error(f"❌ Radius search failed: {e}")
            return []
    
    def find_points_in_polygon(self,
                              polygon_wkt: str,
                              data_model_type: str = None) -> List[Dict[str, Any]]:
        """Find spatial points within a polygon"""
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT 
                        id, point_name, data_model_type, record_id,
                        ST_X(location) as longitude,
                        ST_Y(location) as latitude,
                        elevation_m, depth_m, collection_date, properties
                    FROM spatial_points
                    WHERE ST_Within(location, ST_GeomFromText(:polygon_wkt, 4326))
                """
                
                params = {'polygon_wkt': polygon_wkt}
                
                if data_model_type:
                    query += " AND data_model_type = :data_model_type"
                    params['data_model_type'] = data_model_type
                
                result = conn.execute(text(query), params)
                points = [dict(row._mapping) for row in result]
                
                logger.info(f"🗺️ Found {len(points)} points in polygon")
                return points
                
        except Exception as e:
            logger.error(f"❌ Polygon search failed: {e}")
            return []
    
    def calculate_point_density(self,
                               bounds: Dict[str, float],
                               grid_size_km: float = 10.0,
                               data_model_type: str = None) -> Dict[str, Any]:
        """Calculate point density within specified bounds"""
        try:
            with self.engine.connect() as conn:
                # Create grid and count points per grid cell
                query = """
                    WITH grid AS (
                        SELECT 
                            i, j,
                            ST_MakeEnvelope(
                                :min_lon + (i * :grid_size_deg),
                                :min_lat + (j * :grid_size_deg),
                                :min_lon + ((i + 1) * :grid_size_deg),
                                :min_lat + ((j + 1) * :grid_size_deg),
                                4326
                            ) as cell_geom
                        FROM generate_series(0, :cols - 1) i
                        CROSS JOIN generate_series(0, :rows - 1) j
                    ),
                    density AS (
                        SELECT 
                            g.i, g.j,
                            ST_XMin(g.cell_geom) as min_lon,
                            ST_YMin(g.cell_geom) as min_lat,
                            ST_XMax(g.cell_geom) as max_lon,
                            ST_YMax(g.cell_geom) as max_lat,
                            COUNT(sp.id) as point_count
                        FROM grid g
                        LEFT JOIN spatial_points sp ON ST_Within(sp.location, g.cell_geom)
                """
                
                if data_model_type:
                    query += " AND sp.data_model_type = :data_model_type"
                
                query += """
                        GROUP BY g.i, g.j, g.cell_geom
                        HAVING COUNT(sp.id) > 0
                    )
                    SELECT * FROM density ORDER BY point_count DESC;
                """
                
                # Calculate grid parameters
                grid_size_deg = grid_size_km / 111.0  # Rough conversion km to degrees
                cols = int((bounds['max_lon'] - bounds['min_lon']) / grid_size_deg) + 1
                rows = int((bounds['max_lat'] - bounds['min_lat']) / grid_size_deg) + 1
                
                params = {
                    'min_lon': bounds['min_lon'],
                    'min_lat': bounds['min_lat'],
                    'grid_size_deg': grid_size_deg,
                    'cols': cols,
                    'rows': rows
                }
                
                if data_model_type:
                    params['data_model_type'] = data_model_type
                
                result = conn.execute(text(query), params)
                density_data = [dict(row._mapping) for row in result]
                
                return {
                    'grid_size_km': grid_size_km,
                    'total_cells': len(density_data),
                    'density_data': density_data,
                    'bounds': bounds
                }
                
        except Exception as e:
            logger.error(f"❌ Density calculation failed: {e}")
            return {}
    
    def create_geographic_region(self,
                               region_name: str,
                               region_type: str,
                               geometry_wkt: str,
                               data_model_types: List[str],
                               properties: dict = None) -> Optional[int]:
        """Create a new geographic region"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO geographic_regions 
                    (region_name, region_type, geometry, area_km2, perimeter_km, data_model_types, properties)
                    VALUES 
                    (:region_name, :region_type, ST_GeomFromText(:geometry_wkt, 4326),
                     ST_Area(ST_GeomFromText(:geometry_wkt, 4326)::geography) / 1000000,
                     ST_Perimeter(ST_GeomFromText(:geometry_wkt, 4326)::geography) / 1000,
                     :data_model_types, :properties)
                    RETURNING id;
                """), {
                    'region_name': region_name,
                    'region_type': region_type,
                    'geometry_wkt': geometry_wkt,
                    'data_model_types': data_model_types,
                    'properties': json.dumps(properties) if properties else None
                })
                conn.commit()
                
                region_id = result.fetchone()[0]
                logger.info(f"🗺️ Geographic region created: {region_id}")
                return region_id
                
        except Exception as e:
            logger.error(f"❌ Failed to create geographic region: {e}")
            return None
    
    def get_data_model_statistics(self, data_model_type: str) -> Dict[str, Any]:
        """Get spatial statistics for a specific data model type"""
        try:
            with self.engine.connect() as conn:
                stats_query = text("""
                    SELECT 
                        COUNT(*) as total_points,
                        ST_XMin(ST_Extent(location)) as min_longitude,
                        ST_YMin(ST_Extent(location)) as min_latitude,
                        ST_XMax(ST_Extent(location)) as max_longitude,
                        ST_YMax(ST_Extent(location)) as max_latitude,
                        AVG(ST_X(location)) as center_longitude,
                        AVG(ST_Y(location)) as center_latitude
                    FROM spatial_points 
                    WHERE data_model_type = :data_model_type
                """)
                
                result = conn.execute(stats_query, {'data_model_type': data_model_type})
                stats = dict(result.fetchone()._mapping)
                
                # Get region coverage
                regions_query = text("""
                    SELECT DISTINCT gr.region_name, gr.region_type
                    FROM geographic_regions gr
                    WHERE :data_model_type = ANY(gr.data_model_types)
                """)
                
                regions_result = conn.execute(regions_query, {'data_model_type': data_model_type})
                regions = [dict(row._mapping) for row in regions_result]
                
                stats['covered_regions'] = regions
                stats['region_count'] = len(regions)
                
                logger.info(f"🗺️ Retrieved statistics for {data_model_type}")
                return stats
                
        except Exception as e:
            logger.error(f"❌ Statistics retrieval failed: {e}")
            return {}
    
    def transform_coordinates(self,
                            points: List[Tuple[float, float]],
                            from_crs: int,
                            to_crs: int) -> List[Tuple[float, float]]:
        """Transform coordinates between different CRS"""
        try:
            with self.engine.connect() as conn:
                transformed_points = []
                
                for lon, lat in points:
                    result = conn.execute(text("""
                        SELECT 
                            ST_X(ST_Transform(ST_SetSRID(ST_MakePoint(:lon, :lat), :from_crs), :to_crs)) as x,
                            ST_Y(ST_Transform(ST_SetSRID(ST_MakePoint(:lon, :lat), :from_crs), :to_crs)) as y
                    """), {
                        'lon': lon,
                        'lat': lat,
                        'from_crs': from_crs,
                        'to_crs': to_crs
                    })
                    
                    transformed = result.fetchone()
                    transformed_points.append((transformed.x, transformed.y))
                
                logger.info(f"🗺️ Transformed {len(points)} coordinates from {from_crs} to {to_crs}")
                return transformed_points
                
        except Exception as e:
            logger.error(f"❌ Coordinate transformation failed: {e}")
            return points
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get GIS system status and statistics"""
        try:
            with self.engine.connect() as conn:
                # Check PostGIS version
                postgis_version = conn.execute(text("SELECT PostGIS_Version()")).fetchone()[0]
                
                # Get table counts
                counts_query = text("""
                    SELECT 
                        (SELECT COUNT(*) FROM spatial_points) as spatial_points,
                        (SELECT COUNT(*) FROM geographic_regions) as geographic_regions,
                        (SELECT COUNT(*) FROM spatial_tracks) as spatial_tracks,
                        (SELECT COUNT(*) FROM spatial_layers) as spatial_layers
                """)
                
                counts = dict(conn.execute(counts_query).fetchone()._mapping)
                
                # Get data model distribution
                distribution_query = text("""
                    SELECT data_model_type, COUNT(*) as point_count
                    FROM spatial_points 
                    GROUP BY data_model_type
                    ORDER BY point_count DESC
                """)
                
                distribution_result = conn.execute(distribution_query)
                distribution = [dict(row._mapping) for row in distribution_result]
                
                return {
                    'postgis_version': postgis_version,
                    'table_counts': counts,
                    'data_model_distribution': distribution,
                    'supported_crs': self.supported_crs,
                    'status': 'operational',
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")
            return {'status': 'error', 'error': str(e)}

# Global GIS Manager instance
gis_manager = None

def get_gis_manager() -> GISManager:
    """Get or create GIS Manager singleton"""
    global gis_manager
    if gis_manager is None:
        # This would be initialized with actual database URL in production
        database_url = "postgresql://user:password@localhost/oceanbio"
        gis_manager = GISManager(database_url)
    return gis_manager