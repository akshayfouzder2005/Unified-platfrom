"""
Authentication API endpoints for user management.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import auth_manager, authenticate_user
from app.core.dependencies import (
    get_current_active_user, 
    require_admin,
    get_optional_user
)
from app.crud.user import user_crud
from app.models.user import User, UserRole
from app.schemas.user import (
    User as UserSchema,
    UserCreate,
    UserUpdate,
    UserLogin,
    Token,
    PasswordChange
)
from app.core.exceptions import (
    ValidationError,
    DatabaseError,
    AuthenticationError
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_create: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    """
    try:
        # Check if username already exists
        if user_crud.user_exists(db, username=user_create.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        if user_crud.user_exists(db, email=user_create.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        db_user = user_crud.create_user(db, user_create)
        return db_user
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to create user: {str(e)}")


@router.post("/login", response_model=Token)
async def login_user(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access token.
    """
    try:
        # Authenticate user
        user = authenticate_user(db, user_credentials.username, user_credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        user_crud.update_last_login(db, user.id)
        
        # Create access token
        token_data = auth_manager.create_token_for_user(user)
        
        # Return token with user info
        return Token(
            **token_data,
            user=UserSchema.from_orm(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise AuthenticationError(f"Login failed: {str(e)}")


@router.post("/login/form", response_model=Token)
async def login_form(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Alternative login endpoint compatible with OAuth2PasswordRequestForm.
    """
    try:
        user = authenticate_user(db, form_data.username, form_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        user_crud.update_last_login(db, user.id)
        
        # Create access token
        token_data = auth_manager.create_token_for_user(user)
        
        return Token(
            **token_data,
            user=UserSchema.from_orm(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise AuthenticationError(f"Login failed: {str(e)}")


@router.get("/me", response_model=UserSchema)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user's profile information.
    """
    return current_user


@router.put("/me", response_model=UserSchema)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update current user's profile information.
    """
    try:
        # Users can only update their own basic info
        allowed_fields = {"full_name", "email"}
        update_data = {k: v for k, v in user_update.dict(exclude_unset=True).items() if k in allowed_fields}
        
        if "email" in update_data:
            # Check if new email is already taken
            if user_crud.user_exists(db, email=update_data["email"]):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        # Update user
        updated_user = user_crud.update_user(db, current_user.id, UserUpdate(**update_data))
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to update profile: {str(e)}")


@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change current user's password.
    """
    try:
        # Verify current password
        if not auth_manager.verify_password(password_change.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        success = user_crud.change_password(db, current_user.id, password_change.new_password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to change password: {str(e)}")


@router.get("/users", response_model=List[UserSchema])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Get list of all users (admin only).
    """
    try:
        users = user_crud.get_users(db, skip=skip, limit=limit)
        return users
        
    except Exception as e:
        raise DatabaseError(f"Failed to retrieve users: {str(e)}")


@router.get("/users/{user_id}", response_model=UserSchema)
async def get_user_by_id(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Get user by ID (admin only).
    """
    try:
        user = user_crud.get_user(db, user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to retrieve user: {str(e)}")


@router.put("/users/{user_id}", response_model=UserSchema)
async def update_user_by_id(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Update user by ID (admin only).
    """
    try:
        # Check if email is being changed and already exists
        update_data = user_update.dict(exclude_unset=True)
        if "email" in update_data:
            existing_user = user_crud.get_user_by_email(db, update_data["email"])
            if existing_user and existing_user.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        updated_user = user_crud.update_user(db, user_id, user_update)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to update user: {str(e)}")


@router.delete("/users/{user_id}")
async def delete_user_by_id(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Delete user by ID (admin only).
    """
    try:
        # Prevent admin from deleting themselves
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        success = user_crud.delete_user(db, user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to delete user: {str(e)}")


@router.get("/stats")
async def get_user_stats(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Get user statistics (admin only).
    """
    try:
        total_users = user_crud.get_user_count(db)
        active_users = user_crud.get_active_user_count(db)
        admin_users = len(user_crud.get_users_by_role(db, UserRole.ADMIN))
        researcher_users = len(user_crud.get_users_by_role(db, UserRole.RESEARCHER))
        viewer_users = len(user_crud.get_users_by_role(db, UserRole.VIEWER))
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "admin_users": admin_users,
            "researcher_users": researcher_users,
            "viewer_users": viewer_users
        }
        
    except Exception as e:
        raise DatabaseError(f"Failed to get user stats: {str(e)}")


@router.post("/verify-token", response_model=UserSchema)
async def verify_token(
    current_user: User = Depends(get_current_active_user)
):
    """
    Verify JWT token and return user info.
    """
    return current_user


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_active_user)
):
    """
    Logout user (client should discard the token).
    """
    return {"message": "Successfully logged out"}