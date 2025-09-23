"""
Real-time Notification System for Ocean-Bio Platform
Handles alerts, notifications, and system-wide messaging
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json

logger = logging.getLogger(__name__)

class NotificationLevel(Enum):
    """Notification severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class NotificationType(Enum):
    """Types of notifications"""
    SYSTEM_ALERT = "system_alert"
    SPECIES_IDENTIFICATION = "species_identification"
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"

class Notification:
    """Individual notification object"""
    
    def __init__(
        self,
        notification_id: str,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        notification_type: NotificationType = NotificationType.SYSTEM_ALERT,
        data: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ):
        self.id = notification_id
        self.title = title
        self.message = message
        self.level = level
        self.type = notification_type
        self.data = data or {}
        self.created_at = datetime.now()
        self.expires_at = expires_at
        self.read = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'level': self.level.value,
            'type': self.type.value,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'read': self.read
        }

class NotificationSystem:
    """Real-time notification management system"""
    
    def __init__(self):
        self.notifications: Dict[str, Notification] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.thresholds: Dict[str, Any] = {}
        self.is_running = False
        self._notification_id_counter = 0
        
        # Default thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 0.05
        }
    
    async def initialize(self):
        """Initialize the notification system"""
        try:
            logger.info("🔔 Initializing notification system...")
            self.is_running = True
            
            # Start background task for notification cleanup
            asyncio.create_task(self._cleanup_expired_notifications())
            
            logger.info("✅ Notification system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize notification system: {str(e)}")
            return False
    
    async def shutdown(self):
        """Shutdown the notification system"""
        try:
            self.is_running = False
            self.notifications.clear()
            self.subscribers.clear()
            logger.info("🔔 Notification system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during notification system shutdown: {str(e)}")
    
    def create_notification(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        notification_type: NotificationType = NotificationType.SYSTEM_ALERT,
        data: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Create a new notification"""
        try:
            self._notification_id_counter += 1
            notification_id = f"notif_{self._notification_id_counter}_{int(datetime.now().timestamp())}"
            
            notification = Notification(
                notification_id=notification_id,
                title=title,
                message=message,
                level=level,
                notification_type=notification_type,
                data=data,
                expires_at=expires_at
            )
            
            self.notifications[notification_id] = notification
            
            # Notify subscribers
            asyncio.create_task(self._notify_subscribers(notification))
            
            logger.info(f"📢 Created notification: {title} ({level.value})")
            return notification_id
            
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")
            return ""
    
    async def _notify_subscribers(self, notification: Notification):
        """Notify all subscribers about a new notification"""
        try:
            notification_type = notification.type.value
            
            if notification_type in self.subscribers:
                for callback in self.subscribers[notification_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(notification)
                        else:
                            callback(notification)
                    except Exception as e:
                        logger.error(f"Error in notification subscriber: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error notifying subscribers: {str(e)}")
    
    def subscribe(self, notification_type: NotificationType, callback: Callable):
        """Subscribe to notifications of a specific type"""
        type_key = notification_type.value
        
        if type_key not in self.subscribers:
            self.subscribers[type_key] = []
        
        self.subscribers[type_key].append(callback)
        logger.info(f"📧 Added subscriber for {type_key} notifications")
    
    def unsubscribe(self, notification_type: NotificationType, callback: Callable):
        """Unsubscribe from notifications"""
        type_key = notification_type.value
        
        if type_key in self.subscribers and callback in self.subscribers[type_key]:
            self.subscribers[type_key].remove(callback)
            logger.info(f"📧 Removed subscriber for {type_key} notifications")
    
    def get_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        level: Optional[NotificationLevel] = None,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get notifications with optional filtering"""
        try:
            notifications = []
            
            for notification in self.notifications.values():
                # Apply filters
                if notification_type and notification.type != notification_type:
                    continue
                if level and notification.level != level:
                    continue
                if unread_only and notification.read:
                    continue
                
                notifications.append(notification.to_dict())
            
            # Sort by creation time (newest first)
            notifications.sort(key=lambda x: x['created_at'], reverse=True)
            return notifications
            
        except Exception as e:
            logger.error(f"Error getting notifications: {str(e)}")
            return []
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        try:
            if notification_id in self.notifications:
                self.notifications[notification_id].read = True
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            return False
    
    def delete_notification(self, notification_id: str) -> bool:
        """Delete a notification"""
        try:
            if notification_id in self.notifications:
                del self.notifications[notification_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting notification: {str(e)}")
            return False
    
    async def _cleanup_expired_notifications(self):
        """Background task to clean up expired notifications"""
        while self.is_running:
            try:
                current_time = datetime.now()
                expired_ids = []
                
                for notification_id, notification in self.notifications.items():
                    if notification.expires_at and current_time > notification.expires_at:
                        expired_ids.append(notification_id)
                
                for notification_id in expired_ids:
                    del self.notifications[notification_id]
                
                if expired_ids:
                    logger.info(f"🧹 Cleaned up {len(expired_ids)} expired notifications")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in notification cleanup: {str(e)}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    def check_threshold(self, metric_name: str, value: float) -> bool:
        """Check if a metric value exceeds its threshold"""
        try:
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                return value > threshold
            return False
            
        except Exception as e:
            logger.error(f"Error checking threshold for {metric_name}: {str(e)}")
            return False
    
    def set_threshold(self, metric_name: str, threshold_value: float):
        """Set a threshold for a metric"""
        self.thresholds[metric_name] = threshold_value
        logger.info(f"🎯 Set threshold for {metric_name}: {threshold_value}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get notification system statistics"""
        try:
            total_notifications = len(self.notifications)
            unread_count = sum(1 for n in self.notifications.values() if not n.read)
            
            level_counts = {}
            type_counts = {}
            
            for notification in self.notifications.values():
                level = notification.level.value
                ntype = notification.type.value
                
                level_counts[level] = level_counts.get(level, 0) + 1
                type_counts[ntype] = type_counts.get(ntype, 0) + 1
            
            return {
                'total_notifications': total_notifications,
                'unread_count': unread_count,
                'level_distribution': level_counts,
                'type_distribution': type_counts,
                'active_subscribers': sum(len(subs) for subs in self.subscribers.values()),
                'thresholds': self.thresholds,
                'is_running': self.is_running
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}

# Global notification system instance
notification_system = NotificationSystem()