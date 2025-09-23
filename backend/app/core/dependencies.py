"""
Authentication dependencies for protecting API endpoints.
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import auth_manager
from app.models.user import User, UserRole
from app.crud.user import user_crud


# Security scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token
        token_data = auth_manager.verify_token(credentials.credentials)
        
        # Get user from database
        user = user_crud.get_user_by_username(db, token_data.username)
        
        if user is None:
            raise credentials_exception
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user"
            )
        
        return user
        
    except Exception:
        raise credentials_exception


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    return current_user


def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Require admin role for access.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin role required."
        )
    return current_user


def require_researcher(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Require researcher role or higher for access.
    """
    if not current_user.is_researcher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Researcher role or higher required."
        )
    return current_user


def require_write_access(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Require write access permissions.
    """
    if not current_user.can_write:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Write access required."
        )
    return current_user


def require_read_access(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Require read access permissions.
    """
    if not current_user.can_read:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Read access required."
        )
    return current_user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if token is provided (optional authentication).
    """
    if not credentials:
        return None
    
    try:
        token_data = auth_manager.verify_token(credentials.credentials)
        user = user_crud.get_user_by_username(db, token_data.username)
        
        if user and user.is_active:
            return user
    except Exception:
        pass
    
    return None


def create_role_dependency(required_role: UserRole):
    """
    Create a dependency for specific role requirement.
    """
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role != required_role and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. {required_role.value.title()} role required."
            )
        return current_user
    
    return role_checker


# Convenience dependencies for specific roles
require_admin_role = create_role_dependency(UserRole.ADMIN)
require_researcher_role = create_role_dependency(UserRole.RESEARCHER)