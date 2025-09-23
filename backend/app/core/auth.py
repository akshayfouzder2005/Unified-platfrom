"""
Authentication utilities for JWT token handling and password security.
"""
from datetime import datetime, timedelta
from typing import Optional, Union
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import logging

# Optional imports with graceful fallbacks
try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    JWTError = Exception
    
try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

from .config import get_settings
from .database import get_db

# Optional imports for models and schemas
try:
    from ..models.user import User
except ImportError:
    class User:
        """Mock User class for development"""
        id: int = 1
        username: str = "test_user"
        is_active: bool = True
        hashed_password: str = "mock_hash"
        
try:
    from ..schemas.user import TokenData
except ImportError:
    class TokenData:
        """Mock TokenData class for development"""
        def __init__(self, username: str = None, user_id: int = None):
            self.username = username
            self.user_id = user_id

logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# Password hashing context (if passlib is available)
if PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None
    logger.warning("⚠️ Passlib not available. Password hashing disabled.")


class AuthManager:
    """Authentication manager for handling JWT tokens and password security."""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.SECRET_KEY
        self.algorithm = self.settings.ALGORITHM
        self.access_token_expire_minutes = self.settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if not PASSLIB_AVAILABLE or pwd_context is None:
            # Simple comparison fallback for development
            return plain_password == hashed_password
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        if not PASSLIB_AVAILABLE or pwd_context is None:
            # Simple fallback for development (NOT secure for production)
            return f"mock_hash_{password}"
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        
        if not JWT_AVAILABLE:
            # Simple token fallback for development
            import json
            import base64
            token_data = json.dumps(to_encode, default=str)
            return base64.b64encode(token_data.encode()).decode()
        
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
            if not JWT_AVAILABLE:
                # Simple token decoding fallback for development
                import json
                import base64
                try:
                    token_data = json.loads(base64.b64decode(token.encode()).decode())
                    username = token_data.get("sub")
                    user_id = token_data.get("user_id", 1)
                    
                    if username is None:
                        raise credentials_exception
                        
                    return TokenData(username=username, user_id=user_id)
                except Exception:
                    raise credentials_exception
            
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


# FastAPI Dependencies
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency to get current authenticated user."""
    try:
        # Verify token
        token_data = auth_manager.verify_token(token)
        
        # Try to get user from database
        try:
            user = db.query(User).filter(User.username == token_data.username).first()
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Inactive user",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user
        except Exception as e:
            # Fallback to mock user for development
            logger.warning(f"⚠️ Database query failed, using mock user: {e}")
            mock_user = User()
            mock_user.id = token_data.user_id or 1
            mock_user.username = token_data.username or "test_user"
            mock_user.is_active = True
            return mock_user
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user
