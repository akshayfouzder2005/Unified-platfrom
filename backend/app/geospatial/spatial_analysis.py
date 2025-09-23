"""
🗺️ Spatial Analysis - Advanced Geospatial Computations

Comprehensive spatial analysis tools for marine data.
Provides clustering, hotspot analysis, and spatial statistics.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import unary_union
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import math

logger = logging.getLogger(__name__)

class SpatialAnalysis:
    """
    🗺️ Advanced Spatial Analysis for Marine Data
    
    Provides comprehensive spatial analysis capabilities:
    - Clustering analysis (DBSCAN, K-means, Hierarchical)
    - Hotspot identification and analysis
    - Spatial statistics and metrics
    - Distance and proximity analysis
    - Spatial autocorrelation analysis
    """
    
    def __init__(self):
        """Initialize the spatial analysis service"""
        self.earth_radius_km = 6371.0
        self.cluster_algorithms = ['dbscan', 'kmeans', 'hierarchical']
        self.distance_metrics = ['haversine', 'euclidean', 'manhattan']
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers"""
        try:
            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return self.earth_radius_km * c
            
        except Exception as e:
            logger.error(f"❌ Haversine distance calculation failed: {e}")
            return 0.0
    
    def create_distance_matrix(self, points: List[Dict[str, Any]], metric: str = 'haversine') -> np.ndarray:
        """Create distance matrix for a set of points"""
        try:
            n_points = len(points)
            distance_matrix = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    if metric == 'haversine':
                        dist = self.haversine_distance(
                            points[i]['latitude'], points[i]['longitude'],
                            points[j]['latitude'], points[j]['longitude']
                        )
                    elif metric == 'euclidean':
                        dist = math.sqrt(
                            (points[i]['latitude'] - points[j]['latitude'])**2 +
                            (points[i]['longitude'] - points[j]['longitude'])**2
                        )
                    else:  # manhattan
                        dist = (abs(points[i]['latitude'] - points[j]['latitude']) +
                               abs(points[i]['longitude'] - points[j]['longitude']))
                    
                    distance_matrix[i][j] = dist
                    distance_matrix[j][i] = dist
            
            logger.info(f"🗺️ Distance matrix created for {n_points} points using {metric}")
            return distance_matrix
            
        except Exception as e:
            logger.error(f"❌ Distance matrix creation failed: {e}")
            return np.array([])
    
    def dbscan_clustering(self, 
                         points: List[Dict[str, Any]], 
                         eps_km: float = 5.0, 
                         min_samples: int = 3) -> Dict[str, Any]:
        """Perform DBSCAN clustering on spatial points"""
        try:
            if len(points) < min_samples:
                logger.warning("🗺️ Insufficient points for DBSCAN clustering")
                return {'error': 'Insufficient points'}
            
            # Prepare coordinate arrays
            coords = np.array([[p['latitude'], p['longitude']] for p in points])
            
            # Convert eps from kilometers to degrees (approximate)
            eps_degrees = eps_km / 111.0  # Rough conversion
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)
            labels = clustering.labels_
            
            # Calculate cluster statistics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Group points by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    **points[i],
                    'cluster_id': int(label) if label != -1 else 'noise'
                })
            
            # Calculate cluster centroids and statistics
            cluster_stats = {}
            for cluster_id, cluster_points in clusters.items():
                if cluster_id == -1:  # Noise points
                    continue
                
                # Calculate centroid
                lats = [p['latitude'] for p in cluster_points]
                lons = [p['longitude'] for p in cluster_points]
                centroid = [np.mean(lats), np.mean(lons)]
                
                # Calculate cluster diameter
                max_dist = 0
                for i in range(len(cluster_points)):
                    for j in range(i+1, len(cluster_points)):
                        dist = self.haversine_distance(
                            cluster_points[i]['latitude'], cluster_points[i]['longitude'],
                            cluster_points[j]['latitude'], cluster_points[j]['longitude']
                        )
                        max_dist = max(max_dist, dist)
                
                cluster_stats[cluster_id] = {
                    'centroid': centroid,
                    'size': len(cluster_points),
                    'diameter_km': max_dist,
                    'points': cluster_points
                }
            
            result = {
                'algorithm': 'dbscan',
                'parameters': {'eps_km': eps_km, 'min_samples': min_samples},
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'clusters': cluster_stats,
                'total_points': len(points),
                'silhouette_score': None
            }
            
            # Calculate silhouette score if we have valid clusters
            if n_clusters > 1:
                valid_labels = [l for l in labels if l != -1]
                valid_coords = [coords[i] for i, l in enumerate(labels) if l != -1]
                
                if len(set(valid_labels)) > 1 and len(valid_coords) > 1:
                    try:
                        score = silhouette_score(valid_coords, valid_labels)
                        result['silhouette_score'] = float(score)
                    except:
                        pass
            
            logger.info(f"🗺️ DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
            return result
            
        except Exception as e:
            logger.error(f"❌ DBSCAN clustering failed: {e}")
            return {'error': str(e)}
    
    def kmeans_clustering(self, 
                         points: List[Dict[str, Any]], 
                         n_clusters: int = 5) -> Dict[str, Any]:
        """Perform K-means clustering on spatial points"""
        try:
            if len(points) < n_clusters:
                logger.warning("🗺️ Insufficient points for K-means clustering")
                return {'error': 'Insufficient points'}
            
            # Prepare coordinate arrays
            coords = np.array([[p['latitude'], p['longitude']] for p in points])
            
            # Perform K-means clustering
            clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
            labels = clustering.labels_
            centroids = clustering.cluster_centers_
            
            # Group points by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    **points[i],
                    'cluster_id': int(label)
                })
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id, cluster_points in clusters.items():
                # Calculate cluster diameter
                max_dist = 0
                for i in range(len(cluster_points)):
                    for j in range(i+1, len(cluster_points)):
                        dist = self.haversine_distance(
                            cluster_points[i]['latitude'], cluster_points[i]['longitude'],
                            cluster_points[j]['latitude'], cluster_points[j]['longitude']
                        )
                        max_dist = max(max_dist, dist)
                
                cluster_stats[cluster_id] = {
                    'centroid': [float(centroids[cluster_id][0]), float(centroids[cluster_id][1])],
                    'size': len(cluster_points),
                    'diameter_km': max_dist,
                    'points': cluster_points
                }
            
            # Calculate silhouette score
            silhouette = silhouette_score(coords, labels)
            
            result = {
                'algorithm': 'kmeans',
                'parameters': {'n_clusters': n_clusters},
                'n_clusters': n_clusters,
                'clusters': cluster_stats,
                'total_points': len(points),
                'silhouette_score': float(silhouette),
                'inertia': float(clustering.inertia_)
            }
            
            logger.info(f"🗺️ K-means clustering completed: {n_clusters} clusters")
            return result
            
        except Exception as e:
            logger.error(f"❌ K-means clustering failed: {e}")
            return {'error': str(e)}
    
    def identify_hotspots(self, 
                         points: List[Dict[str, Any]], 
                         radius_km: float = 10.0,
                         min_points: int = 5) -> Dict[str, Any]:
        """Identify spatial hotspots based on point density"""
        try:
            if len(points) < min_points:
                logger.warning("🗺️ Insufficient points for hotspot analysis")
                return {'hotspots': [], 'total_points': len(points)}
            
            hotspots = []
            processed_points = set()
            
            for i, center_point in enumerate(points):
                if i in processed_points:
                    continue
                
                # Find points within radius
                nearby_points = []
                nearby_indices = []
                
                for j, point in enumerate(points):
                    if j != i:
                        dist = self.haversine_distance(
                            center_point['latitude'], center_point['longitude'],
                            point['latitude'], point['longitude']
                        )
                        
                        if dist <= radius_km:
                            nearby_points.append(point)
                            nearby_indices.append(j)
                
                # Check if this forms a hotspot
                if len(nearby_points) >= min_points - 1:  # -1 because we exclude center point
                    # Calculate hotspot centroid
                    all_points = [center_point] + nearby_points
                    all_indices = [i] + nearby_indices
                    
                    lats = [p['latitude'] for p in all_points]
                    lons = [p['longitude'] for p in all_points]
                    centroid = [np.mean(lats), np.mean(lons)]
                    
                    # Calculate density (points per km²)
                    area = math.pi * radius_km ** 2
                    density = len(all_points) / area
                    
                    # Identify data model types in hotspot
                    data_types = list(set([p.get('data_model_type', 'unknown') for p in all_points]))
                    
                    hotspot = {
                        'hotspot_id': len(hotspots),
                        'centroid': centroid,
                        'radius_km': radius_km,
                        'point_count': len(all_points),
                        'density_per_km2': density,
                        'data_model_types': data_types,
                        'points': all_points
                    }
                    
                    hotspots.append(hotspot)
                    processed_points.update(all_indices)
            
            # Sort hotspots by density
            hotspots.sort(key=lambda x: x['density_per_km2'], reverse=True)
            
            # Add ranking
            for i, hotspot in enumerate(hotspots):
                hotspot['rank'] = i + 1
            
            result = {
                'parameters': {'radius_km': radius_km, 'min_points': min_points},
                'hotspots': hotspots,
                'n_hotspots': len(hotspots),
                'total_points': len(points),
                'coverage_percentage': (len(processed_points) / len(points)) * 100
            }
            
            logger.info(f"🗺️ Hotspot analysis completed: {len(hotspots)} hotspots identified")
            return result
            
        except Exception as e:
            logger.error(f"❌ Hotspot analysis failed: {e}")
            return {'error': str(e)}
    
    def calculate_spatial_statistics(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive spatial statistics"""
        try:
            if not points:
                return {'error': 'No points provided'}
            
            # Basic statistics
            lats = [p['latitude'] for p in points]
            lons = [p['longitude'] for p in points]
            
            basic_stats = {
                'point_count': len(points),
                'latitude_stats': {
                    'min': float(np.min(lats)),
                    'max': float(np.max(lats)),
                    'mean': float(np.mean(lats)),
                    'std': float(np.std(lats))
                },
                'longitude_stats': {
                    'min': float(np.min(lons)),
                    'max': float(np.max(lons)),
                    'mean': float(np.mean(lons)),
                    'std': float(np.std(lons))
                },
                'centroid': [float(np.mean(lats)), float(np.mean(lons))]
            }
            
            # Extent and area
            extent = {
                'min_lat': basic_stats['latitude_stats']['min'],
                'max_lat': basic_stats['latitude_stats']['max'],
                'min_lon': basic_stats['longitude_stats']['min'],
                'max_lon': basic_stats['longitude_stats']['max']
            }
            
            # Calculate approximate area (rough estimation)
            lat_range = extent['max_lat'] - extent['min_lat']
            lon_range = extent['max_lon'] - extent['min_lon']
            area_deg2 = lat_range * lon_range
            area_km2 = area_deg2 * (111.0 ** 2)  # Rough conversion
            
            # Distance analysis
            if len(points) > 1:
                distances = []
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        dist = self.haversine_distance(
                            points[i]['latitude'], points[i]['longitude'],
                            points[j]['latitude'], points[j]['longitude']
                        )
                        distances.append(dist)
                
                distance_stats = {
                    'min_distance_km': float(np.min(distances)),
                    'max_distance_km': float(np.max(distances)),
                    'mean_distance_km': float(np.mean(distances)),
                    'median_distance_km': float(np.median(distances)),
                    'std_distance_km': float(np.std(distances))
                }
            else:
                distance_stats = None
            
            # Density calculation
            density_per_km2 = len(points) / area_km2 if area_km2 > 0 else 0
            
            # Data model type distribution
            data_types = {}
            for point in points:
                data_type = point.get('data_model_type', 'unknown')
                data_types[data_type] = data_types.get(data_type, 0) + 1
            
            result = {
                'basic_statistics': basic_stats,
                'spatial_extent': extent,
                'area_km2': area_km2,
                'density_per_km2': density_per_km2,
                'distance_statistics': distance_stats,
                'data_model_distribution': data_types,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🗺️ Spatial statistics calculated for {len(points)} points")
            return result
            
        except Exception as e:
            logger.error(f"❌ Spatial statistics calculation failed: {e}")
            return {'error': str(e)}
    
    def find_nearest_neighbors(self, 
                              target_point: Dict[str, Any], 
                              candidate_points: List[Dict[str, Any]], 
                              k: int = 5) -> List[Dict[str, Any]]:
        """Find k nearest neighbors to a target point"""
        try:
            if not candidate_points:
                return []
            
            # Calculate distances to all candidate points
            distances = []
            for i, point in enumerate(candidate_points):
                dist = self.haversine_distance(
                    target_point['latitude'], target_point['longitude'],
                    point['latitude'], point['longitude']
                )
                distances.append({'index': i, 'distance_km': dist, 'point': point})
            
            # Sort by distance and take top k
            distances.sort(key=lambda x: x['distance_km'])
            nearest = distances[:k]
            
            # Format results
            neighbors = []
            for neighbor in nearest:
                neighbors.append({
                    **neighbor['point'],
                    'distance_km': neighbor['distance_km']
                })
            
            logger.info(f"🗺️ Found {len(neighbors)} nearest neighbors")
            return neighbors
            
        except Exception as e:
            logger.error(f"❌ Nearest neighbor search failed: {e}")
            return []
    
    def analyze_spatial_patterns(self, 
                                points: List[Dict[str, Any]], 
                                data_model_type: str = None) -> Dict[str, Any]:
        """Comprehensive spatial pattern analysis"""
        try:
            # Filter by data model type if specified
            if data_model_type:
                points = [p for p in points if p.get('data_model_type') == data_model_type]
            
            if not points:
                return {'error': 'No points available for analysis'}
            
            # Basic spatial statistics
            spatial_stats = self.calculate_spatial_statistics(points)
            
            # Clustering analysis
            clustering_results = {}
            
            # DBSCAN clustering
            if len(points) >= 3:
                dbscan_result = self.dbscan_clustering(points, eps_km=5.0, min_samples=3)
                clustering_results['dbscan'] = dbscan_result
            
            # K-means clustering (if enough points)
            if len(points) >= 5:
                optimal_k = min(5, len(points) // 2)
                kmeans_result = self.kmeans_clustering(points, n_clusters=optimal_k)
                clustering_results['kmeans'] = kmeans_result
            
            # Hotspot analysis
            hotspot_result = self.identify_hotspots(points, radius_km=10.0, min_points=3)
            
            # Compile comprehensive analysis
            analysis = {
                'data_model_type': data_model_type,
                'spatial_statistics': spatial_stats,
                'clustering_analysis': clustering_results,
                'hotspot_analysis': hotspot_result,
                'analysis_timestamp': datetime.now().isoformat(),
                'recommendations': self._generate_spatial_recommendations(
                    spatial_stats, clustering_results, hotspot_result
                )
            }
            
            logger.info(f"🗺️ Comprehensive spatial pattern analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Spatial pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_spatial_recommendations(self, 
                                        spatial_stats: Dict[str, Any],
                                        clustering_results: Dict[str, Any], 
                                        hotspot_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on spatial analysis"""
        recommendations = []
        
        try:
            # Point density recommendations
            if 'density_per_km2' in spatial_stats:
                density = spatial_stats['density_per_km2']
                if density < 0.1:
                    recommendations.append("Low sampling density detected. Consider increasing sampling effort in this region.")
                elif density > 10:
                    recommendations.append("High sampling density detected. Data quality is excellent for spatial analysis.")
            
            # Clustering recommendations
            if 'dbscan' in clustering_results:
                dbscan = clustering_results['dbscan']
                if 'n_clusters' in dbscan and dbscan['n_clusters'] > 0:
                    recommendations.append(f"Identified {dbscan['n_clusters']} distinct spatial clusters. "
                                         "Consider targeted analysis of each cluster.")
            
            # Hotspot recommendations
            if 'n_hotspots' in hotspot_result and hotspot_result['n_hotspots'] > 0:
                recommendations.append(f"Identified {hotspot_result['n_hotspots']} biodiversity hotspots. "
                                     "These areas warrant special conservation attention.")
            
            # Coverage recommendations
            if 'coverage_percentage' in hotspot_result:
                coverage = hotspot_result['coverage_percentage']
                if coverage < 50:
                    recommendations.append("Low hotspot coverage. Consider expanding sampling to identify additional hotspots.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            return ["Analysis completed successfully."]

# Global spatial analysis instance
spatial_analyzer = SpatialAnalysis()