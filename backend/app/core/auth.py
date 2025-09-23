"""
Authentication utilities for JWT token handling and password security.
"""
from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.user import User
from app.schemas.user import TokenData


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthManager:
    """Authentication manager for handling JWT tokens and password security."""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt

    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            
            if username is None or user_id is None:
                raise credentials_exception
                
            token_data = TokenData(username=username, user_id=user_id)
            return token_data
        except JWTError:
            raise credentials_exception

    def authenticate_user(self, db: Session, username: str, password: str) -> Union[User, bool]:
        """Authenticate user with username and password."""
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return False
        if not user.is_active:
            return False
        if not self.verify_password(password, user.hashed_password):
            return False
        
        return user

    def create_token_for_user(self, user: User) -> dict:
        """Create token data for authenticated user."""
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,  # Convert to seconds
        }


# Global auth manager instance
auth_manager = AuthManager()


def get_password_hash(password: str) -> str:
    """Get password hash - convenience function."""
    return auth_manager.get_password_hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password - convenience function."""
    return auth_manager.verify_password(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token - convenience function."""
    return auth_manager.create_access_token(data, expires_delta)


def authenticate_user(db: Session, username: str, password: str) -> Union[User, bool]:
    """Authenticate user - convenience function."""
    return auth_manager.authenticate_user(db, username, password)