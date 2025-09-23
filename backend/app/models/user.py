"""
User model for authentication and authorization.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from app.models.base import Base


class UserRole(PyEnum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    RESEARCHER = "researcher" 
    VIEWER = "viewer"


class UserStatus(PyEnum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(Base):
    """
    User model for authentication and authorization.
    
    Attributes:
        id: Primary key
        username: Unique username for login
        email: Unique email address
        full_name: User's full name
        hashed_password: Bcrypt hashed password
        role: User role (admin, researcher, viewer)
        is_active: Whether the user account is active
        is_verified: Whether the user email is verified
        created_at: Account creation timestamp
        last_login: Last login timestamp
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role.value}')>"

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN

    @property
    def is_researcher(self) -> bool:
        """Check if user has researcher role or higher."""
        return self.role in [UserRole.ADMIN, UserRole.RESEARCHER]

    @property
    def can_write(self) -> bool:
        """Check if user has write permissions."""
        return self.role in [UserRole.ADMIN, UserRole.RESEARCHER]

    @property
    def can_read(self) -> bool:
        """Check if user has read permissions."""
        return self.is_active and self.role in [UserRole.ADMIN, UserRole.RESEARCHER, UserRole.VIEWER]