"""
Authentication API Routes for Fluxion Backend
Handles user authentication, registration, and session management.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr
from services.auth.enhanced_jwt_service import DeviceInfo, EnhancedJWTService
from services.user.user_service import UserService, UserType

router = APIRouter()
security = HTTPBearer()


# Request/Response models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    user_type: str = "individual"
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    country_of_residence: str = "US"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class MFAVerificationRequest(BaseModel):
    mfa_token: str
    mfa_code: str


@router.post("/register")
async def register_user(
    request: RegisterRequest,
    http_request: Request,
    user_service: UserService = Depends(),
):
    """Register a new user"""
    try:
        # Extract device info
        device_info = DeviceInfo(
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", ""),
            device_id=http_request.headers.get("x-device-id"),
        )

        # Prepare profile data
        profile_data = {
            "first_name": request.first_name,
            "last_name": request.last_name,
            "phone_number": request.phone_number,
            "country_of_residence": request.country_of_residence,
        }

        result = await user_service.register_user(
            email=request.email,
            password=request.password,
            user_type=UserType(request.user_type),
            profile_data=profile_data,
            device_info=device_info,
        )

        return {
            "success": True,
            "data": result,
            "message": "User registered successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login")
async def login_user(
    request: LoginRequest, http_request: Request, user_service: UserService = Depends()
):
    """Authenticate user and create session"""
    try:
        # Extract device info
        device_info = DeviceInfo(
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", ""),
            device_id=http_request.headers.get("x-device-id"),
        )

        result = await user_service.authenticate_user(
            email=request.email, password=request.password, device_info=device_info
        )

        return {"success": True, "data": result, "message": "Authentication successful"}

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.post("/verify-email")
async def verify_email(verification_token: str, user_service: UserService = Depends()):
    """Verify user email address"""
    try:
        result = await user_service.verify_email(verification_token)

        return {
            "success": True,
            "data": result,
            "message": "Email verified successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Email verification failed")


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(),  # Would implement JWT dependency
    user_service: UserService = Depends(),
):
    """Change user password"""
    try:
        user_id = current_user.get("user_id")

        result = await user_service.change_password(
            user_id=user_id,
            current_password=request.current_password,
            new_password=request.new_password,
        )

        return {
            "success": True,
            "data": result,
            "message": "Password changed successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Password change failed")


@router.post("/enable-mfa")
async def enable_mfa(
    current_user: Dict[str, Any] = Depends(),  # Would implement JWT dependency
    user_service: UserService = Depends(),
):
    """Enable multi-factor authentication"""
    try:
        user_id = current_user.get("user_id")

        result = await user_service.enable_mfa(user_id)

        return {"success": True, "data": result, "message": "MFA setup initiated"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="MFA setup failed")


@router.post("/verify-mfa-setup")
async def verify_mfa_setup(
    request: MFAVerificationRequest,
    current_user: Dict[str, Any] = Depends(),  # Would implement JWT dependency
    user_service: UserService = Depends(),
):
    """Verify MFA setup with authenticator code"""
    try:
        user_id = current_user.get("user_id")

        result = await user_service.verify_mfa_setup(
            user_id=user_id, mfa_code=request.mfa_code
        )

        return {"success": True, "data": result, "message": "MFA enabled successfully"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="MFA verification failed")


@router.post("/disable-mfa")
async def disable_mfa(
    password: str,
    current_user: Dict[str, Any] = Depends(),  # Would implement JWT dependency
    user_service: UserService = Depends(),
):
    """Disable multi-factor authentication"""
    try:
        user_id = current_user.get("user_id")

        result = await user_service.disable_mfa(user_id=user_id, password=password)

        return {"success": True, "data": result, "message": "MFA disabled successfully"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="MFA disable failed")


@router.post("/logout")
async def logout_user(
    current_user: Dict[str, Any] = Depends(),  # Would implement JWT dependency
    jwt_service: EnhancedJWTService = Depends(),
):
    """Logout user and revoke tokens"""
    try:
        user_id = current_user.get("user_id")

        # Revoke user sessions
        await jwt_service.revoke_user_sessions(user_id, "User logout")

        return {"success": True, "message": "Logged out successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me")
async def get_current_user(
    current_user: Dict[str, Any] = Depends(),  # Would implement JWT dependency
    user_service: UserService = Depends(),
):
    """Get current user information"""
    try:
        user_id = current_user.get("user_id")

        profile = await user_service.get_user_profile(user_id)

        return {
            "success": True,
            "data": profile,
            "message": "User profile retrieved successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve user profile")
