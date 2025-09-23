"""
CRUD operations for user management.
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserUpdate
from app.core.auth import get_password_hash


class UserCRUD:
    """CRUD operations for User model."""

    def get_user(self, db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    def get_user_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()

    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()

    def get_users(self, db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Get list of users with pagination."""
        return db.query(User).offset(skip).limit(limit).all()

    def create_user(self, db: Session, user_create: UserCreate) -> User:
        """Create new user."""
        hashed_password = get_password_hash(user_create.password)
        
        db_user = User(
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            hashed_password=hashed_password,
            role=user_create.role,
            is_active=True,
            is_verified=False
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return db_user

    def update_user(self, db: Session, user_id: int, user_update: UserUpdate) -> Optional[User]:
        """Update existing user."""
        db_user = self.get_user(db, user_id)
        
        if not db_user:
            return None
        
        update_data = user_update.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(db_user, field, value)
        
        db.commit()
        db.refresh(db_user)
        
        return db_user

    def delete_user(self, db: Session, user_id: int) -> bool:
        """Delete user (soft delete by setting is_active to False)."""
        db_user = self.get_user(db, user_id)
        
        if not db_user:
            return False
        
        db_user.is_active = False
        db.commit()
        
        return True

    def update_last_login(self, db: Session, user_id: int) -> Optional[User]:
        """Update user's last login timestamp."""
        db_user = self.get_user(db, user_id)
        
        if not db_user:
            return None
        
        db_user.last_login = func.now()
        db.commit()
        db.refresh(db_user)
        
        return db_user

    def change_password(self, db: Session, user_id: int, new_password: str) -> bool:
        """Change user password."""
        db_user = self.get_user(db, user_id)
        
        if not db_user:
            return False
        
        db_user.hashed_password = get_password_hash(new_password)
        db.commit()
        
        return True

    def verify_user_email(self, db: Session, user_id: int) -> bool:
        """Mark user email as verified."""
        db_user = self.get_user(db, user_id)
        
        if not db_user:
            return False
        
        db_user.is_verified = True
        db.commit()
        
        return True

    def get_users_by_role(self, db: Session, role: UserRole) -> List[User]:
        """Get users by role."""
        return db.query(User).filter(User.role == role).all()

    def get_active_users(self, db: Session) -> List[User]:
        """Get all active users."""
        return db.query(User).filter(User.is_active == True).all()

    def user_exists(self, db: Session, username: str = None, email: str = None) -> bool:
        """Check if user exists by username or email."""
        query = db.query(User)
        
        if username:
            query = query.filter(User.username == username)
        
        if email:
            query = query.filter(User.email == email)
        
        return query.first() is not None

    def get_user_count(self, db: Session) -> int:
        """Get total number of users."""
        return db.query(User).count()

    def get_active_user_count(self, db: Session) -> int:
        """Get number of active users."""
        return db.query(User).filter(User.is_active == True).count()


# Global instance
user_crud = UserCRUD()