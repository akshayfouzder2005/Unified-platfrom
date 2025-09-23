"""
Comprehensive authentication tests for Ocean-Bio platform.

Tests user authentication, authorization, password management, 
and security features.
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from app.core.security import verify_password, create_access_token
from app.models.user import User, UserRole, UserStatus


@pytest.mark.auth
class TestAuthentication:
    """Test user authentication functionality."""
    
    def test_user_registration(self, client: TestClient):
        """Test user registration endpoint."""
        user_data = {
            "username": "newuser",
            "email": "newuser@test.com",
            "password": "securepassword123",
            "full_name": "New User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert data["full_name"] == user_data["full_name"]
        assert data["role"] == "VIEWER"  # Default role
        assert data["status"] == "PENDING"  # Default status
        assert "id" in data
        assert "password" not in data
    
    def test_user_registration_duplicate_username(self, client: TestClient, admin_user: User):
        """Test registration with duplicate username."""
        user_data = {
            "username": admin_user.username,
            "email": "different@test.com",
            "password": "securepassword123",
            "full_name": "Different User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 400
        assert "Username already registered" in response.json()["detail"]
    
    def test_user_registration_duplicate_email(self, client: TestClient, admin_user: User):
        """Test registration with duplicate email."""
        user_data = {
            "username": "differentuser",
            "email": admin_user.email,
            "password": "securepassword123",
            "full_name": "Different User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]
    
    def test_user_login_success(self, client: TestClient, admin_user: User):
        """Test successful user login."""
        login_data = {
            "username": admin_user.username,
            "password": "testpassword123"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_user_login_invalid_credentials(self, client: TestClient, admin_user: User):
        """Test login with invalid credentials."""
        login_data = {
            "username": admin_user.username,
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_user_login_nonexistent_user(self, client: TestClient):
        """Test login with nonexistent user."""
        login_data = {
            "username": "nonexistent",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_user_login_inactive_user(self, client: TestClient, db_session, viewer_user: User):
        """Test login with inactive user."""
        # Deactivate user
        viewer_user.status = UserStatus.INACTIVE
        db_session.commit()
        
        login_data = {
            "username": viewer_user.username,
            "password": "testpassword123"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 400
        assert "Inactive user" in response.json()["detail"]


@pytest.mark.auth
class TestAuthorization:
    """Test user authorization and access control."""
    
    def test_get_current_user_with_valid_token(self, client: TestClient, admin_headers: dict):
        """Test getting current user with valid token."""
        response = client.get("/api/v1/auth/me", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin_test"
        assert data["role"] == "ADMIN"
        assert data["status"] == "ACTIVE"
    
    def test_get_current_user_without_token(self, client: TestClient):
        """Test getting current user without token."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
    
    def test_get_current_user_with_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]
    
    def test_admin_access_required(self, client: TestClient, researcher_headers: dict):
        """Test admin-only endpoint with researcher credentials."""
        # Try to access admin-only users list endpoint
        response = client.get("/api/v1/auth/users", headers=researcher_headers)
        
        assert response.status_code == 403
        assert "Not enough permissions" in response.json()["detail"]
    
    def test_admin_access_granted(self, client: TestClient, admin_headers: dict):
        """Test admin-only endpoint with admin credentials."""
        response = client.get("/api/v1/auth/users", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_researcher_access_to_data(self, client: TestClient, researcher_headers: dict):
        """Test researcher access to data endpoints."""
        response = client.get("/api/v1/fisheries/vessels", headers=researcher_headers)
        
        # Should have access (at least to read)
        assert response.status_code in [200, 404]  # 404 if no data exists
    
    def test_viewer_read_only_access(self, client: TestClient, viewer_headers: dict):
        """Test viewer read-only access."""
        # Should be able to read
        response = client.get("/api/v1/fisheries/vessels", headers=viewer_headers)
        assert response.status_code in [200, 404]
        
        # Should not be able to create
        vessel_data = {
            "vessel_name": "Test Vessel",
            "registration_number": "TEST-001",
            "vessel_type": "TRAWLER",
            "length_meters": 30.0,
            "owner_name": "Test Owner",
            "home_port": "Test Port"
        }
        
        response = client.post("/api/v1/fisheries/vessels", json=vessel_data, headers=viewer_headers)
        assert response.status_code == 403


@pytest.mark.auth
class TestPasswordManagement:
    """Test password-related functionality."""
    
    def test_password_change_success(self, client: TestClient, researcher_user: User, researcher_headers: dict):
        """Test successful password change."""
        password_data = {
            "current_password": "testpassword123",
            "new_password": "newsecurepassword123"
        }
        
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=researcher_headers)
        
        assert response.status_code == 200
        assert response.json()["message"] == "Password updated successfully"
    
    def test_password_change_wrong_current(self, client: TestClient, researcher_headers: dict):
        """Test password change with wrong current password."""
        password_data = {
            "current_password": "wrongpassword",
            "new_password": "newsecurepassword123"
        }
        
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=researcher_headers)
        
        assert response.status_code == 400
        assert "Incorrect current password" in response.json()["detail"]
    
    def test_password_reset_request(self, client: TestClient, researcher_user: User):
        """Test password reset request."""
        reset_data = {"email": researcher_user.email}
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        
        assert response.status_code == 200
        assert "Password reset instructions sent" in response.json()["message"]
    
    def test_password_reset_nonexistent_email(self, client: TestClient):
        """Test password reset with nonexistent email."""
        reset_data = {"email": "nonexistent@test.com"}
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        
        # Should still return success for security reasons
        assert response.status_code == 200


@pytest.mark.auth
class TestUserManagement:
    """Test user management functionality (admin features)."""
    
    def test_list_users_as_admin(self, client: TestClient, admin_headers: dict, admin_user: User, researcher_user: User):
        """Test listing users as admin."""
        response = client.get("/api/v1/auth/users", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2  # At least admin and researcher
        
        usernames = [user["username"] for user in data]
        assert admin_user.username in usernames
        assert researcher_user.username in usernames
    
    def test_get_user_by_id_as_admin(self, client: TestClient, admin_headers: dict, researcher_user: User):
        """Test getting specific user by ID as admin."""
        response = client.get(f"/api/v1/auth/users/{researcher_user.id}", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == researcher_user.id
        assert data["username"] == researcher_user.username
        assert data["email"] == researcher_user.email
    
    def test_update_user_role_as_admin(self, client: TestClient, admin_headers: dict, viewer_user: User):
        """Test updating user role as admin."""
        update_data = {"role": "RESEARCHER"}
        
        response = client.patch(f"/api/v1/auth/users/{viewer_user.id}", json=update_data, headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "RESEARCHER"
    
    def test_activate_user_as_admin(self, client: TestClient, admin_headers: dict, db_session, viewer_user: User):
        """Test activating user as admin."""
        # Set user as pending
        viewer_user.status = UserStatus.PENDING
        db_session.commit()
        
        response = client.post(f"/api/v1/auth/users/{viewer_user.id}/activate", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ACTIVE"
    
    def test_deactivate_user_as_admin(self, client: TestClient, admin_headers: dict, researcher_user: User):
        """Test deactivating user as admin."""
        response = client.post(f"/api/v1/auth/users/{researcher_user.id}/deactivate", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "INACTIVE"
    
    def test_delete_user_as_admin(self, client: TestClient, admin_headers: dict, viewer_user: User):
        """Test deleting user as admin."""
        response = client.delete(f"/api/v1/auth/users/{viewer_user.id}", headers=admin_headers)
        
        assert response.status_code == 200
        assert "User deleted successfully" in response.json()["message"]
        
        # Verify user is deleted
        response = client.get(f"/api/v1/auth/users/{viewer_user.id}", headers=admin_headers)
        assert response.status_code == 404


@pytest.mark.auth  
class TestTokenManagement:
    """Test JWT token functionality."""
    
    def test_token_expiration(self, client: TestClient, admin_user: User):
        """Test token expiration handling."""
        # Create expired token
        expired_token = create_access_token(
            data={"sub": admin_user.username},
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 401
        assert "Token expired" in response.json()["detail"]
    
    def test_token_refresh(self, client: TestClient, admin_user: User, admin_headers: dict):
        """Test token refresh functionality."""
        response = client.post("/api/v1/auth/refresh", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_logout(self, client: TestClient, admin_headers: dict):
        """Test user logout."""
        response = client.post("/api/v1/auth/logout", headers=admin_headers)
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]


@pytest.mark.unit
class TestSecurityUtilities:
    """Test security utility functions."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "testsecurepassword123"
        
        # Hash password
        from app.core.security import get_password_hash
        hashed = get_password_hash(password)
        
        # Verify password
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)
    
    def test_token_creation_and_verification(self, admin_user: User):
        """Test JWT token creation and verification."""
        from app.core.security import verify_token
        
        # Create token
        token = create_access_token(data={"sub": admin_user.username})
        
        # Verify token
        payload = verify_token(token)
        assert payload["sub"] == admin_user.username
    
    def test_invalid_token_verification(self):
        """Test verification of invalid tokens."""
        from app.core.security import verify_token
        
        # Test completely invalid token
        with pytest.raises(Exception):
            verify_token("invalid.token.string")
        
        # Test malformed token
        with pytest.raises(Exception):
            verify_token("not-a-token")


@pytest.mark.integration
class TestAuthenticationIntegration:
    """Integration tests for authentication flow."""
    
    def test_full_authentication_flow(self, client: TestClient):
        """Test complete authentication flow: register -> login -> access."""
        # 1. Register new user
        user_data = {
            "username": "integrationuser",
            "email": "integration@test.com",
            "password": "integrationpass123",
            "full_name": "Integration User"
        }
        
        register_response = client.post("/api/v1/auth/register", json=user_data)
        assert register_response.status_code == 201
        
        # 2. Login with new user
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        
        login_response = client.post("/api/v1/auth/login", data=login_data)
        assert login_response.status_code == 200
        
        token_data = login_response.json()
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        
        # 3. Access protected endpoint
        profile_response = client.get("/api/v1/auth/me", headers=headers)
        assert profile_response.status_code == 200
        
        profile_data = profile_response.json()
        assert profile_data["username"] == user_data["username"]
        assert profile_data["email"] == user_data["email"]
    
    def test_role_based_access_workflow(self, client: TestClient, admin_headers: dict):
        """Test role-based access control workflow."""
        # 1. Admin creates new user
        user_data = {
            "username": "roleuser",
            "email": "roleuser@test.com",
            "password": "rolepass123",
            "full_name": "Role Test User"
        }
        
        register_response = client.post("/api/v1/auth/register", json=user_data)
        new_user_id = register_response.json()["id"]
        
        # 2. Admin activates user
        activate_response = client.post(f"/api/v1/auth/users/{new_user_id}/activate", headers=admin_headers)
        assert activate_response.status_code == 200
        
        # 3. Admin promotes user to researcher
        update_response = client.patch(
            f"/api/v1/auth/users/{new_user_id}", 
            json={"role": "RESEARCHER"}, 
            headers=admin_headers
        )
        assert update_response.status_code == 200
        
        # 4. User logs in and accesses researcher features
        login_response = client.post("/api/v1/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        
        token_data = login_response.json()
        user_headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        
        # Should now have researcher access
        fisheries_response = client.get("/api/v1/fisheries/vessels", headers=user_headers)
        assert fisheries_response.status_code in [200, 404]  # 404 if no data