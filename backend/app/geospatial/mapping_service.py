"""
🗺️ Mapping Service - Interactive Geographic Visualization

Advanced mapping capabilities for marine data visualization.
Generates interactive maps with multiple data layers and real-time updates.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import folium
from folium import plugins
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from branca.colormap import LinearColormap, StepColormap
import numpy as np

logger = logging.getLogger(__name__)

class MappingService:
    """
    🗺️ Interactive Mapping Service for Marine Data Visualization
    
    Provides comprehensive mapping capabilities:
    - Multi-layer interactive maps
    - Real-time data visualization
    - Spatial clustering and heatmaps
    - Custom markers and popups
    - Export and sharing functionality
    """
    
    def __init__(self):
        """Initialize the mapping service"""
        self.default_center = [20.5937, 78.9629]  # India center
        self.default_zoom = 6
        self.color_schemes = {
            'edna': ['#2E8B57', '#32CD32'],      # Sea green palette
            'oceanographic': ['#1E90FF', '#0000CD'], # Blue palette
            'otolith': ['#FF6347', '#CD853F'],    # Orange/brown palette
            'taxonomy': ['#9370DB', '#8A2BE2'],   # Purple palette
            'fisheries': ['#FF4500', '#FF8C00']   # Red/orange palette
        }
        self.marker_icons = {
            'edna': 'leaf',
            'oceanographic': 'tint',
            'otolith': 'circle',
            'taxonomy': 'tag',
            'fisheries': 'ship'
        }
    
    def create_base_map(self, 
                       center: List[float] = None, 
                       zoom: int = None,
                       tiles: str = 'OpenStreetMap') -> folium.Map:
        """Create a base interactive map"""
        try:
            center = center or self.default_center
            zoom = zoom or self.default_zoom
            
            # Create base map
            m = folium.Map(
                location=center,
                zoom_start=zoom,
                tiles=tiles,
                prefer_canvas=True
            )
            
            # Add additional tile layers
            tile_layers = {
                'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                'Ocean': 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
                'Terrain': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}'
            }
            
            for name, url in tile_layers.items():
                folium.TileLayer(
                    tiles=url,
                    attr=f'{name} tiles',
                    name=name,
                    overlay=False,
                    control=True
                ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add measurement tool
            plugins.MeasureControl().add_to(m)
            
            # Add fullscreen button
            plugins.Fullscreen().add_to(m)
            
            logger.info(f"🗺️ Base map created with center {center}")
            return m
            
        except Exception as e:
            logger.error(f"❌ Base map creation failed: {e}")
            return None
    
    def add_point_layer(self, 
                       map_obj: folium.Map,
                       points_data: List[Dict[str, Any]],
                       layer_name: str,
                       data_model_type: str,
                       cluster: bool = True) -> folium.Map:
        """Add a point layer to the map"""
        try:
            if not points_data:
                logger.warning(f"🗺️ No data provided for {layer_name}")
                return map_obj
            
            # Get colors and icons for this data type
            colors = self.color_schemes.get(data_model_type, ['#666666', '#333333'])
            icon_name = self.marker_icons.get(data_model_type, 'circle')
            
            # Create feature group
            feature_group = folium.FeatureGroup(name=layer_name)
            
            if cluster:
                # Create marker cluster
                marker_cluster = plugins.MarkerCluster(name=f"{layer_name} Cluster")
                
                for point in points_data:
                    lat, lon = point['latitude'], point['longitude']
                    
                    # Create popup content
                    popup_html = self._create_point_popup(point, data_model_type)
                    
                    # Create marker
                    marker = folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(
                            color='blue' if data_model_type == 'oceanographic' else 'green',
                            icon=icon_name,
                            prefix='fa'
                        )
                    )
                    
                    marker.add_to(marker_cluster)
                
                marker_cluster.add_to(feature_group)
            
            else:
                # Add individual markers
                for point in points_data:
                    lat, lon = point['latitude'], point['longitude']
                    
                    popup_html = self._create_point_popup(point, data_model_type)
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        popup=folium.Popup(popup_html, max_width=300),
                        color=colors[1],
                        fillColor=colors[0],
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(feature_group)
            
            feature_group.add_to(map_obj)
            
            logger.info(f"🗺️ Added {len(points_data)} points to {layer_name}")
            return map_obj
            
        except Exception as e:
            logger.error(f"❌ Point layer creation failed: {e}")
            return map_obj
    
    def add_heatmap_layer(self,
                         map_obj: folium.Map,
                         points_data: List[Dict[str, Any]],
                         layer_name: str,
                         weight_field: str = None,
                         radius: int = 15) -> folium.Map:
        """Add a heatmap layer to the map"""
        try:
            if not points_data:
                logger.warning(f"🗺️ No data provided for heatmap {layer_name}")
                return map_obj
            
            # Prepare heatmap data
            heat_data = []
            for point in points_data:
                lat, lon = point['latitude'], point['longitude']
                
                if weight_field and weight_field in point and point[weight_field]:
                    weight = float(point[weight_field])
                    heat_data.append([lat, lon, weight])
                else:
                    heat_data.append([lat, lon])
            
            # Create heatmap
            heatmap = plugins.HeatMap(
                heat_data,
                name=layer_name,
                radius=radius,
                blur=10,
                max_zoom=1,
                gradient={
                    0.0: 'navy',
                    0.25: 'blue',
                    0.5: 'green',
                    0.75: 'yellow',
                    1.0: 'red'
                }
            )
            
            heatmap.add_to(map_obj)
            
            logger.info(f"🗺️ Added heatmap with {len(heat_data)} points")
            return map_obj
            
        except Exception as e:
            logger.error(f"❌ Heatmap creation failed: {e}")
            return map_obj
    
    def add_polygon_layer(self,
                         map_obj: folium.Map,
                         polygons_data: List[Dict[str, Any]],
                         layer_name: str,
                         color_field: str = None) -> folium.Map:
        """Add polygon layer (regions, zones, etc.) to the map"""
        try:
            if not polygons_data:
                logger.warning(f"🗺️ No polygon data provided for {layer_name}")
                return map_obj
            
            # Create feature group for polygons
            feature_group = folium.FeatureGroup(name=layer_name)
            
            for polygon_data in polygons_data:
                # Convert WKT or coordinates to polygon
                if 'geometry' in polygon_data:
                    # Assume it's WKT format
                    from shapely import wkt
                    geom = wkt.loads(polygon_data['geometry'])
                    
                    if geom.geom_type == 'Polygon':
                        coords = [[[lon, lat] for lon, lat in geom.exterior.coords]]
                    else:
                        continue
                
                elif 'coordinates' in polygon_data:
                    coords = polygon_data['coordinates']
                else:
                    continue
                
                # Determine color
                if color_field and color_field in polygon_data:
                    # Use color based on field value
                    color = self._get_color_for_value(polygon_data[color_field])
                else:
                    color = '#3388ff'
                
                # Create popup content
                popup_html = f"""
                <div style="font-family: Arial, sans-serif;">
                    <h4>{polygon_data.get('region_name', 'Region')}</h4>
                    <p><strong>Type:</strong> {polygon_data.get('region_type', 'N/A')}</p>
                    <p><strong>Area:</strong> {polygon_data.get('area_km2', 0):.2f} km²</p>
                </div>
                """
                
                # Add polygon to map
                folium.GeoJson(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coords
                        },
                        "properties": polygon_data
                    },
                    style_function=lambda feature, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 2,
                        'fillOpacity': 0.3
                    },
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(feature_group)
            
            feature_group.add_to(map_obj)
            
            logger.info(f"🗺️ Added {len(polygons_data)} polygons to {layer_name}")
            return map_obj
            
        except Exception as e:
            logger.error(f"❌ Polygon layer creation failed: {e}")
            return map_obj
    
    def add_density_grid(self,
                        map_obj: folium.Map,
                        density_data: List[Dict[str, Any]],
                        layer_name: str,
                        max_count: int = None) -> folium.Map:
        """Add density grid visualization to the map"""
        try:
            if not density_data:
                logger.warning(f"🗺️ No density data provided for {layer_name}")
                return map_obj
            
            # Calculate max count for color scaling
            if not max_count:
                max_count = max(cell['point_count'] for cell in density_data)
            
            # Create color map
            colormap = LinearColormap(
                colors=['yellow', 'orange', 'red'],
                vmin=0,
                vmax=max_count
            )
            
            # Create feature group
            feature_group = folium.FeatureGroup(name=layer_name)
            
            for cell in density_data:
                # Create rectangle for grid cell
                bounds = [
                    [cell['min_lat'], cell['min_lon']],
                    [cell['max_lat'], cell['max_lon']]
                ]
                
                color = colormap(cell['point_count'])
                
                folium.Rectangle(
                    bounds=bounds,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=1,
                    popup=f"Count: {cell['point_count']}"
                ).add_to(feature_group)
            
            feature_group.add_to(map_obj)
            
            # Add color legend
            colormap.caption = layer_name
            colormap.add_to(map_obj)
            
            logger.info(f"🗺️ Added density grid with {len(density_data)} cells")
            return map_obj
            
        except Exception as e:
            logger.error(f"❌ Density grid creation failed: {e}")
            return map_obj
    
    def create_multi_layer_map(self,
                             spatial_data: Dict[str, List[Dict[str, Any]]],
                             center: List[float] = None,
                             title: str = "Marine Data Visualization") -> folium.Map:
        """Create a comprehensive multi-layer map with all data types"""
        try:
            # Create base map
            map_obj = self.create_base_map(center=center)
            
            # Add title
            title_html = f'''
            <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
            '''
            map_obj.get_root().html.add_child(folium.Element(title_html))
            
            # Process each data type
            for data_type, data_list in spatial_data.items():
                if not data_list:
                    continue
                
                # Add point layer
                if 'points' in data_list:
                    self.add_point_layer(
                        map_obj, 
                        data_list['points'], 
                        f"{data_type.title()} Samples",
                        data_type,
                        cluster=True
                    )
                    
                    # Add heatmap layer
                    self.add_heatmap_layer(
                        map_obj,
                        data_list['points'],
                        f"{data_type.title()} Heatmap"
                    )
                
                # Add polygon layers if available
                if 'regions' in data_list:
                    self.add_polygon_layer(
                        map_obj,
                        data_list['regions'],
                        f"{data_type.title()} Regions"
                    )
            
            # Add search functionality
            plugins.Search(
                layer=None,
                search_label='point_name',
                search_zoom=15,
                geom_type='Point'
            ).add_to(map_obj)
            
            # Add draw tools
            draw = plugins.Draw(
                export=True,
                filename='marine_data_selection.geojson',
                position='topleft',
                draw_options={
                    'polyline': True,
                    'rectangle': True,
                    'polygon': True,
                    'circle': True,
                    'marker': True,
                    'circlemarker': False,
                }
            )
            draw.add_to(map_obj)
            
            logger.info(f"🗺️ Multi-layer map created with {len(spatial_data)} data types")
            return map_obj
            
        except Exception as e:
            logger.error(f"❌ Multi-layer map creation failed: {e}")
            return None
    
    def _create_point_popup(self, point: Dict[str, Any], data_model_type: str) -> str:
        """Create HTML popup content for a point"""
        try:
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 250px;">
                <h4 style="color: #2c3e50; margin-bottom: 10px;">
                    {point.get('point_name', 'Sample Point')}
                </h4>
                <div style="margin-bottom: 8px;">
                    <strong>Type:</strong> <span style="color: #3498db;">{data_model_type.title()}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Location:</strong> {point['latitude']:.4f}, {point['longitude']:.4f}
                </div>
            """
            
            # Add data type specific information
            if data_model_type == 'edna':
                popup_html += f"""
                <div style="margin-bottom: 8px;">
                    <strong>Sample ID:</strong> {point.get('record_id', 'N/A')}
                </div>
                """
                if point.get('depth_m'):
                    popup_html += f"<div><strong>Depth:</strong> {point['depth_m']} m</div>"
            
            elif data_model_type == 'oceanographic':
                popup_html += f"""
                <div style="margin-bottom: 8px;">
                    <strong>Station:</strong> {point.get('record_id', 'N/A')}
                </div>
                """
                if point.get('depth_m'):
                    popup_html += f"<div><strong>Depth:</strong> {point['depth_m']} m</div>"
            
            elif data_model_type == 'otolith':
                popup_html += f"""
                <div style="margin-bottom: 8px;">
                    <strong>Specimen:</strong> {point.get('record_id', 'N/A')}
                </div>
                """
            
            elif data_model_type == 'fisheries':
                popup_html += f"""
                <div style="margin-bottom: 8px;">
                    <strong>Vessel/Trip:</strong> {point.get('record_id', 'N/A')}
                </div>
                """
            
            # Add collection date if available
            if point.get('collection_date'):
                popup_html += f"""
                <div style="margin-bottom: 8px;">
                    <strong>Date:</strong> {point['collection_date'][:10]}
                </div>
                """
            
            # Add properties if available
            if point.get('properties'):
                properties = point['properties']
                if isinstance(properties, str):
                    import json
                    properties = json.loads(properties)
                
                if isinstance(properties, dict):
                    for key, value in properties.items():
                        if value is not None:
                            popup_html += f"""
                            <div style="margin-bottom: 4px;">
                                <strong>{key.replace('_', ' ').title()}:</strong> {value}
                            </div>
                            """
            
            popup_html += "</div>"
            
            return popup_html
            
        except Exception as e:
            logger.error(f"❌ Popup creation failed: {e}")
            return f"<div>{point.get('point_name', 'Sample Point')}</div>"
    
    def _get_color_for_value(self, value: Any) -> str:
        """Get color based on value (for choropleth maps)"""
        try:
            if isinstance(value, (int, float)):
                # Use value-based coloring
                if value < 10:
                    return '#ffffcc'
                elif value < 25:
                    return '#c2e699'
                elif value < 50:
                    return '#78c679'
                elif value < 100:
                    return '#31a354'
                else:
                    return '#006837'
            else:
                # Use categorical coloring
                colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
                return colors[hash(str(value)) % len(colors)]
        
        except Exception:
            return '#3388ff'  # Default blue
    
    def export_map(self, map_obj: folium.Map, filepath: str, format: str = 'html') -> bool:
        """Export map to file"""
        try:
            if format.lower() == 'html':
                map_obj.save(filepath)
                logger.info(f"🗺️ Map exported to {filepath}")
                return True
            else:
                logger.error(f"❌ Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Map export failed: {e}")
            return False
    
    def get_map_statistics(self, spatial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the spatial data being visualized"""
        try:
            stats = {
                'total_layers': len(spatial_data),
                'total_points': 0,
                'data_type_counts': {},
                'extent': {
                    'min_lat': float('inf'),
                    'max_lat': float('-inf'),
                    'min_lon': float('inf'),
                    'max_lon': float('-inf')
                }
            }
            
            for data_type, data_list in spatial_data.items():
                if isinstance(data_list, dict) and 'points' in data_list:
                    points = data_list['points']
                elif isinstance(data_list, list):
                    points = data_list
                else:
                    continue
                
                count = len(points)
                stats['total_points'] += count
                stats['data_type_counts'][data_type] = count
                
                # Update extent
                for point in points:
                    lat, lon = point['latitude'], point['longitude']
                    stats['extent']['min_lat'] = min(stats['extent']['min_lat'], lat)
                    stats['extent']['max_lat'] = max(stats['extent']['max_lat'], lat)
                    stats['extent']['min_lon'] = min(stats['extent']['min_lon'], lon)
                    stats['extent']['max_lon'] = max(stats['extent']['max_lon'], lon)
            
            # Calculate center
            if stats['total_points'] > 0:
                stats['center'] = [
                    (stats['extent']['min_lat'] + stats['extent']['max_lat']) / 2,
                    (stats['extent']['min_lon'] + stats['extent']['max_lon']) / 2
                ]
            else:
                stats['center'] = self.default_center
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Statistics calculation failed: {e}")
            return {'error': str(e)}

# Global mapping service instance
mapping_service = MappingService()