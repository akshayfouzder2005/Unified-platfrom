"""
WebSocket Manager for Real-time Communication
Handles WebSocket connections, broadcasting, and real-time data streaming
"""
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import uuid

logger = logging.getLogger(__name__)

class ConnectionInfo:
    """Information about a WebSocket connection"""
    def __init__(self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None):
        self.websocket = websocket
        self.client_id = client_id
        self.user_id = user_id
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.subscriptions: Set[str] = set()
        self.metadata: Dict[str, Any] = {}

class WebSocketManager:
    """
    Advanced WebSocket connection management
    Supports real-time analytics, notifications, and data streaming
    """
    
    def __init__(self):
        # Active connections
        self.connections: Dict[str, ConnectionInfo] = {}
        
        # Topic subscriptions (topic -> set of client_ids)
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # User connections (user_id -> set of client_ids)
        self.user_connections: Dict[str, Set[str]] = {}
        
        # Connection statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'disconnections': 0,
            'errors': 0
        }
        
        # Available topics for subscription
        self.available_topics = {
            'species_identification',      # AI species identification results
            'real_time_analytics',        # Live analytics data
            'fisheries_updates',         # Fisheries data updates
            'system_alerts',             # System notifications
            'user_notifications',        # User-specific notifications
            'data_ingestion_status',     # Data upload progress
            'model_performance',         # AI model performance metrics
            'platform_stats',           # Platform usage statistics
            'research_updates',          # Research project updates
            'environmental_data'         # Real-time environmental data
        }
        
        self.is_initialized = False
        self.redis_url = None
        # For compatibility with the init script
        self.connection_manager = self
    
    async def initialize(self, redis_url: Optional[str] = None):
        """Initialize the WebSocket manager"""
        try:
            logger.info("🔗 Initializing WebSocket manager...")
            
            self.redis_url = redis_url
            
            if redis_url:
                # Try to connect to Redis for pub/sub
                try:
                    # This would connect to Redis if available
                    logger.info("Redis URL provided for pub/sub capabilities")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}")
            else:
                logger.info("Running in standalone mode (no Redis)")
            
            self.is_initialized = True
            logger.info("✅ WebSocket manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            return False
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            if not client_id:
                client_id = str(uuid.uuid4())
            
            connection = ConnectionInfo(websocket, client_id, user_id)
            self.connections[client_id] = connection
            
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(client_id)
            
            self.stats['total_connections'] += 1
            self.stats['active_connections'] = len(self.connections)
            
            logger.info(f"WebSocket connected: {client_id} (user: {user_id})")
            
            await self.send_to_client(client_id, {
                'type': 'connection_established',
                'client_id': client_id,
                'server_time': datetime.now().isoformat(),
                'available_topics': list(self.available_topics),
                'message': 'Welcome to Ocean-Bio real-time analytics!'
            })
            
            return client_id
            
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {str(e)}")
            self.stats['errors'] += 1
            raise
    
    async def disconnect(self, client_id: str):
        """Disconnect and clean up a WebSocket connection"""
        try:
            if client_id in self.connections:
                connection = self.connections[client_id]
                
                if connection.user_id and connection.user_id in self.user_connections:
                    self.user_connections[connection.user_id].discard(client_id)
                    if not self.user_connections[connection.user_id]:
                        del self.user_connections[connection.user_id]
                
                for topic in connection.subscriptions:
                    if topic in self.subscriptions:
                        self.subscriptions[topic].discard(client_id)
                        if not self.subscriptions[topic]:
                            del self.subscriptions[topic]
                
                del self.connections[client_id]
                self.stats['active_connections'] = len(self.connections)
                self.stats['disconnections'] += 1
                
                logger.info(f"WebSocket disconnected: {client_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {client_id}: {str(e)}")
            self.stats['errors'] += 1
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]) -> bool:
        """Send data to a specific client"""
        try:
            if client_id in self.connections:
                connection = self.connections[client_id]
                
                message = {
                    'timestamp': datetime.now().isoformat(),
                    'client_id': client_id,
                    **data
                }
                
                await connection.websocket.send_text(json.dumps(message))
                self.stats['messages_sent'] += 1
                return True
            
            return False
            
        except WebSocketDisconnect:
            await self.disconnect(client_id)
            return False
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    async def broadcast_to_topic(self, topic: str, data: Dict[str, Any]) -> int:
        """Broadcast data to all clients subscribed to a topic"""
        try:
            if topic not in self.subscriptions:
                return 0
            
            client_ids = self.subscriptions[topic].copy()
            sent_count = 0
            
            message_data = {
                'topic': topic,
                'broadcast_id': str(uuid.uuid4()),
                **data
            }
            
            for client_id in client_ids:
                if await self.send_to_client(client_id, message_data):
                    sent_count += 1
            
            if sent_count > 0:
                logger.debug(f"Broadcast to topic '{topic}': {sent_count} clients")
            
            return sent_count
            
        except Exception as e:
            logger.error(f"Error broadcasting to topic {topic}: {str(e)}")
            self.stats['errors'] += 1
            return 0
    
    async def subscribe_to_topic(self, client_id: str, topic: str) -> bool:
        """Subscribe a client to a topic"""
        try:
            if client_id not in self.connections:
                return False
            
            if topic not in self.available_topics:
                await self.send_to_client(client_id, {
                    'type': 'subscription_error',
                    'error': f'Topic {topic} is not available',
                    'available_topics': list(self.available_topics)
                })
                return False
            
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            self.subscriptions[topic].add(client_id)
            
            self.connections[client_id].subscriptions.add(topic)
            
            await self.send_to_client(client_id, {
                'type': 'subscription_confirmed',
                'topic': topic,
                'message': f'Subscribed to {topic}'
            })
            
            logger.info(f"Client {client_id} subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing client {client_id} to topic {topic}: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    async def handle_client_message(self, client_id: str, message: str):
        """Handle incoming message from client"""
        try:
            self.stats['messages_received'] += 1
            
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await self.send_to_client(client_id, {
                    'type': 'error',
                    'error': 'Invalid JSON message'
                })
                return
            
            message_type = data.get('type', 'unknown')
            
            if message_type == 'subscribe':
                topic = data.get('topic')
                if topic:
                    await self.subscribe_to_topic(client_id, topic)
                    
            elif message_type == 'ping':
                if client_id in self.connections:
                    self.connections[client_id].last_ping = datetime.now()
                
                await self.send_to_client(client_id, {
                    'type': 'pong',
                    'server_time': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error handling client message from {client_id}: {str(e)}")
            self.stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            **self.stats,
            'topics_with_subscribers': len(self.subscriptions),
            'unique_users_connected': len(self.user_connections),
            'uptime': datetime.now().isoformat()
        }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'is_initialized': self.is_initialized,
            'active_connections': len(self.connections),
            'total_connections': self.stats['total_connections'],
            'messages_sent': self.stats['messages_sent'],
            'messages_received': self.stats['messages_received'],
            'available_topics': list(self.available_topics),
            'subscribed_topics': len(self.subscriptions),
            'redis_enabled': bool(self.redis_url)
        }
    
    def get_total_connections(self) -> int:
        """Get total number of connections"""
        return self.stats['total_connections']
    
    async def send_species_identification_update(self, identification_result: Dict[str, Any]):
        """Send species identification results to subscribers"""
        await self.broadcast_to_topic('species_identification', {
            'type': 'species_identification_result',
            'result': identification_result
        })
    
    async def send_analytics_update(self, analytics_data: Dict[str, Any]):
        """Send real-time analytics data to subscribers"""
        await self.broadcast_to_topic('real_time_analytics', {
            'type': 'analytics_update',
            'data': analytics_data
        })

# Global WebSocket manager instance
websocket_manager = WebSocketManager()