/**
 * 🗺️ Interactive Geospatial Map Component
 * 
 * Advanced interactive map for marine data visualization using React-Leaflet.
 * Features real-time data overlays, cluster analysis, and spatial query tools.
 * 
 * @author Ocean-Bio Development Team
 * @version 2.0.0
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, LayersControl, FeatureGroup, Circle, Rectangle } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import MarkerClusterGroup from 'react-leaflet-cluster';
import { Icon, LatLngBounds, LatLng } from 'leaflet';
import { Card, Select, Button, Switch, Slider, Space, Typography, Alert, Spin, Badge } from 'antd';
import { 
  FullscreenOutlined, 
  LayersOutlined, 
  SearchOutlined, 
  FilterOutlined,
  DownloadOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { useQuery, useMutation } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';

// API imports
import { geospatialAPI } from '../../services/api';

// Types
interface MapDataPoint {
  id: string;
  latitude: number;
  longitude: number;
  type: string;
  data: Record<string, any>;
  timestamp: string;
}

interface MapLayer {
  id: string;
  name: string;
  type: 'markers' | 'heatmap' | 'choropleth' | 'bathymetry';
  visible: boolean;
  data: MapDataPoint[];
  style?: Record<string, any>;
}

interface SpatialQuery {
  type: 'location' | 'bbox' | 'polygon';
  coordinates: number[];
  radius?: number;
  filters: {
    dataTypes: string[];
    dateRange: [string, string] | null;
  };
}

// Custom marker icons
const createMarkerIcon = (type: string, color: string = '#1890ff') => new Icon({
  iconUrl: `data:image/svg+xml,${encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}" width="24" height="24">
      <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
      <circle cx="12" cy="9" r="2.5" fill="white"/>
    </svg>
  `)}`,
  iconSize: [24, 24],
  iconAnchor: [12, 24],
  popupAnchor: [0, -24],
});

const { Option } = Select;
const { Title, Text } = Typography;

interface InteractiveMapProps {
  className?: string;
  height?: number | string;
  defaultCenter?: [number, number];
  defaultZoom?: number;
  enableClustering?: boolean;
  enableDrawing?: boolean;
  onSpatialQuery?: (query: SpatialQuery) => void;
  onDataPointClick?: (point: MapDataPoint) => void;
}

const InteractiveMap: React.FC<InteractiveMapProps> = ({
  className = '',
  height = 600,
  defaultCenter = [20.5937, 78.9629], // Center of India
  defaultZoom = 5,
  enableClustering = true,
  enableDrawing = false,
  onSpatialQuery,
  onDataPointClick
}) => {
  // State management
  const [mapLayers, setMapLayers] = useState<MapLayer[]>([]);
  const [selectedDataTypes, setSelectedDataTypes] = useState<string[]>(['all']);
  const [activeQuery, setActiveQuery] = useState<SpatialQuery | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [mapBounds, setMapBounds] = useState<LatLngBounds | null>(null);
  const [filterSettings, setFilterSettings] = useState({
    timeRange: null as [string, string] | null,
    depthRange: [0, 1000],
    showClusters: enableClustering
  });

  // Data fetching
  const {
    data: availableLayers,
    isLoading: layersLoading,
    error: layersError
  } = useQuery({
    queryKey: ['geospatial-layers'],
    queryFn: () => geospatialAPI.getAvailableLayers(),
    refetchInterval: 30000 // Refresh every 30 seconds
  });

  const {
    data: mapData,
    isLoading: dataLoading,
    refetch: refetchData
  } = useQuery({
    queryKey: ['map-data', selectedDataTypes, mapBounds],
    queryFn: () => geospatialAPI.queryByBoundingBox({
      min_lat: mapBounds?.getSouth() || -90,
      max_lat: mapBounds?.getNorth() || 90,
      min_lon: mapBounds?.getWest() || -180,
      max_lon: mapBounds?.getEast() || 180,
      data_types: selectedDataTypes.includes('all') ? undefined : selectedDataTypes,
      start_date: filterSettings.timeRange?.[0],
      end_date: filterSettings.timeRange?.[1]
    }),
    enabled: !!mapBounds,
    refetchOnWindowFocus: false
  });

  // Spatial query mutation
  const spatialQueryMutation = useMutation({
    mutationFn: (query: SpatialQuery) => {
      switch (query.type) {
        case 'location':
          return geospatialAPI.queryByLocation({
            latitude: query.coordinates[0],
            longitude: query.coordinates[1],
            radius_km: query.radius || 10,
            data_types: query.filters.dataTypes,
            start_date: query.filters.dateRange?.[0],
            end_date: query.filters.dateRange?.[1]
          });
        case 'bbox':
          return geospatialAPI.queryByBoundingBox({
            min_lat: Math.min(query.coordinates[0], query.coordinates[2]),
            max_lat: Math.max(query.coordinates[0], query.coordinates[2]),
            min_lon: Math.min(query.coordinates[1], query.coordinates[3]),
            max_lon: Math.max(query.coordinates[1], query.coordinates[3]),
            data_types: query.filters.dataTypes,
            start_date: query.filters.dateRange?.[0],
            end_date: query.filters.dateRange?.[1]
          });
        default:
          throw new Error('Unsupported query type');
      }
    },
    onSuccess: (data) => {
      toast.success(`Found ${data.total_records} records`);
      onSpatialQuery?.(activeQuery!);
    },
    onError: (error) => {
      console.error('Spatial query failed:', error);
      toast.error('Spatial query failed');
    }
  });

  // Process map data into layers
  const processedLayers = useMemo(() => {
    if (!mapData?.results?.features) return [];

    const layerGroups: Record<string, MapDataPoint[]> = {};

    mapData.results.features.forEach((feature: any) => {
      const type = feature.properties.type || 'unknown';
      if (!layerGroups[type]) {
        layerGroups[type] = [];
      }

      layerGroups[type].push({
        id: feature.properties.id || Math.random().toString(),
        latitude: feature.geometry.coordinates[1],
        longitude: feature.geometry.coordinates[0],
        type,
        data: feature.properties,
        timestamp: feature.properties.timestamp || new Date().toISOString()
      });
    });

    return Object.entries(layerGroups).map(([type, points]) => ({
      id: type,
      name: type.charAt(0).toUpperCase() + type.slice(1),
      type: 'markers' as const,
      visible: true,
      data: points
    }));
  }, [mapData]);

  // Handle map events
  const handleMapMove = useCallback((bounds: LatLngBounds) => {
    setMapBounds(bounds);
  }, []);

  const handleDrawCreated = useCallback((e: any) => {
    const { layerType, layer } = e;
    
    let query: SpatialQuery;
    
    if (layerType === 'circle') {
      const center = layer.getLatLng();
      const radius = layer.getRadius() / 1000; // Convert to km
      query = {
        type: 'location',
        coordinates: [center.lat, center.lng],
        radius,
        filters: {
          dataTypes: selectedDataTypes,
          dateRange: filterSettings.timeRange
        }
      };
    } else if (layerType === 'rectangle') {
      const bounds = layer.getBounds();
      query = {
        type: 'bbox',
        coordinates: [
          bounds.getSouth(),
          bounds.getWest(),
          bounds.getNorth(),
          bounds.getEast()
        ],
        filters: {
          dataTypes: selectedDataTypes,
          dateRange: filterSettings.timeRange
        }
      };
    } else {
      return; // Unsupported shape
    }

    setActiveQuery(query);
    spatialQueryMutation.mutate(query);
  }, [selectedDataTypes, filterSettings.timeRange, spatialQueryMutation]);

  const handleMarkerClick = useCallback((point: MapDataPoint) => {
    onDataPointClick?.(point);
  }, [onDataPointClick]);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(!isFullscreen);
  }, [isFullscreen]);

  // Render marker clusters or individual markers
  const renderMarkers = useCallback((layer: MapLayer) => {
    const markers = layer.data.map((point) => (
      <Marker
        key={point.id}
        position={[point.latitude, point.longitude]}
        icon={createMarkerIcon(point.type)}
        eventHandlers={{
          click: () => handleMarkerClick(point)
        }}
      >
        <Popup>
          <div className="p-3">
            <Title level={5}>{point.type}</Title>
            <Text strong>ID:</Text> <Text>{point.id}</Text><br/>
            <Text strong>Location:</Text> <Text>{point.latitude.toFixed(4)}, {point.longitude.toFixed(4)}</Text><br/>
            <Text strong>Time:</Text> <Text>{new Date(point.timestamp).toLocaleString()}</Text>
            {point.data.depth && (
              <>
                <br/><Text strong>Depth:</Text> <Text>{point.data.depth}m</Text>
              </>
            )}
            {point.data.temperature && (
              <>
                <br/><Text strong>Temperature:</Text> <Text>{point.data.temperature}°C</Text>
              </>
            )}
          </div>
        </Popup>
      </Marker>
    ));

    return filterSettings.showClusters ? (
      <MarkerClusterGroup chunkedLoading>
        {markers}
      </MarkerClusterGroup>
    ) : (
      <FeatureGroup>
        {markers}
      </FeatureGroup>
    );
  }, [handleMarkerClick, filterSettings.showClusters]);

  if (layersError) {
    return (
      <Alert
        message="Error loading map data"
        description="Failed to load geospatial layers. Please check your connection and try again."
        type="error"
        showIcon
        action={
          <Button size="small" onClick={() => window.location.reload()}>
            Retry
          </Button>
        }
      />
    );
  }

  return (
    <div className={`interactive-map-container ${className}`}>
      {/* Map Controls */}
      <Card 
        className="map-controls mb-4" 
        size="small"
        extra={
          <Space>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={() => refetchData()}
              loading={dataLoading}
            >
              Refresh
            </Button>
            <Button 
              icon={<FullscreenOutlined />} 
              onClick={toggleFullscreen}
            >
              {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            </Button>
          </Space>
        }
      >
        <Space wrap>
          <div>
            <Text strong>Data Types:</Text>
            <Select
              mode="multiple"
              style={{ width: 200, marginLeft: 8 }}
              placeholder="Select data types"
              value={selectedDataTypes}
              onChange={setSelectedDataTypes}
              loading={layersLoading}
            >
              <Option value="all">All Types</Option>
              {availableLayers?.available_layers.map((layer: any) => (
                <Option key={layer.id} value={layer.id}>
                  {layer.name}
                </Option>
              ))}
            </Select>
          </div>

          <div>
            <Text strong>Clustering:</Text>
            <Switch
              style={{ marginLeft: 8 }}
              checked={filterSettings.showClusters}
              onChange={(checked) => 
                setFilterSettings(prev => ({ ...prev, showClusters: checked }))
              }
            />
          </div>

          {dataLoading && (
            <div>
              <Spin size="small" />
              <Text style={{ marginLeft: 8 }}>Loading data...</Text>
            </div>
          )}

          {mapData && (
            <Badge 
              count={mapData.total_records} 
              style={{ backgroundColor: '#52c41a' }} 
              title={`${mapData.total_records} data points`}
            >
              <Text>Data Points</Text>
            </Badge>
          )}
        </Space>
      </Card>

      {/* Map Container */}
      <div 
        className={`map-wrapper ${isFullscreen ? 'fullscreen' : ''}`}
        style={{ height }}
      >
        <MapContainer
          center={defaultCenter}
          zoom={defaultZoom}
          style={{ height: '100%', width: '100%' }}
          whenReady={(map) => {
            const leafletMap = map.target;
            setMapBounds(leafletMap.getBounds());
            
            leafletMap.on('moveend', () => {
              handleMapMove(leafletMap.getBounds());
            });
          }}
        >
          {/* Base Tile Layers */}
          <LayersControl position="topright">
            <LayersControl.BaseLayer checked name="OpenStreetMap">
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
            </LayersControl.BaseLayer>
            
            <LayersControl.BaseLayer name="Satellite">
              <TileLayer
                attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              />
            </LayersControl.BaseLayer>

            <LayersControl.BaseLayer name="Bathymetry">
              <TileLayer
                attribution='&copy; <a href="https://www.gebco.net/">GEBCO</a>'
                url="https://tiles.arcgis.com/tiles/C8EMgrsFcRFL6LrL/arcgis/rest/services/GEBCO_basemap_NCEI/MapServer/tile/{z}/{y}/{x}"
              />
            </LayersControl.BaseLayer>

            {/* Data Layers */}
            {processedLayers.map((layer) => (
              <LayersControl.Overlay key={layer.id} checked name={layer.name}>
                <FeatureGroup>
                  {renderMarkers(layer)}
                </FeatureGroup>
              </LayersControl.Overlay>
            ))}
          </LayersControl>

          {/* Drawing Tools */}
          {enableDrawing && (
            <FeatureGroup>
              <EditControl
                position="topleft"
                onCreated={handleDrawCreated}
                draw={{
                  rectangle: true,
                  circle: true,
                  circlemarker: false,
                  marker: false,
                  polyline: false,
                  polygon: false
                }}
              />
            </FeatureGroup>
          )}
        </MapContainer>
      </div>

      {/* Active Query Display */}
      {activeQuery && spatialQueryMutation.isLoading && (
        <div className="query-overlay">
          <Spin size="large" />
          <Text>Executing spatial query...</Text>
        </div>
      )}

      <style jsx>{`
        .interactive-map-container {
          position: relative;
        }

        .map-wrapper {
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .map-wrapper.fullscreen {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 9999;
          height: 100vh !important;
        }

        .query-overlay {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
          z-index: 1000;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 10px;
        }

        :global(.leaflet-container) {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }

        :global(.leaflet-popup-content) {
          margin: 0;
        }
      `}</style>
    </div>
  );
};

export default InteractiveMap;