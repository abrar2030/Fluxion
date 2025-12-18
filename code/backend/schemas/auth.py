"""
Authentication schemas for Fluxion backend
"""

import re
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import EmailStr, Field, field_validator, model_validator
from schemas.base import BaseSchema


class UserRegister(BaseSchema):
    """User registration schema"""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ..., min_length=8, max_length=128, description="User password"
    )
    confirm_password: str = Field(..., description="Password confirmation")
    username: Optional[str] = Field(
        None, min_length=3, max_length=50, description="Username"
    )
    first_name: Optional[str] = Field(None, max_length=100, description="First name")
    last_name: Optional[str] = Field(None, max_length=100, description="Last name")
    phone_number: Optional[str] = Field(None, description="Phone number")
    country: Optional[str] = Field(None, max_length=100, description="Country")
    referral_code: Optional[str] = Field(
        None, max_length=20, description="Referral code"
    )
    terms_accepted: bool = Field(..., description="Terms and conditions accepted")
    privacy_accepted: bool = Field(..., description="Privacy policy accepted")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search("[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search("[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search("\\d", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search('[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError("Password must contain at least one special character")
        return v

    @model_validator(mode="after")
    def passwords_match(self):
        """Validate that password and confirm_password match"""
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserLogin(BaseSchema):
    """User login schema"""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    device_info: Optional[dict] = Field(None, description="Device information")
    remember_me: bool = Field(False, description="Remember user session")


class TokenResponse(BaseSchema):
    """Token response schema"""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_id: UUID = Field(..., description="User ID")
    session_id: UUID = Field(..., description="Session ID")


class RefreshTokenRequest(BaseSchema):
    """Refresh token request schema"""

    refresh_token: str = Field(..., description="JWT refresh token")


class PasswordChange(BaseSchema):
    """Password change schema"""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ..., min_length=8, max_length=128, description="New password"
    )
    confirm_new_password: str = Field(..., description="Confirm new password")

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search("[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search("[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search("\\d", v):
            raise ValueError("Password must contain at least one digit")
        return v

    @model_validator(mode="after")
    def passwords_match(self):
        """Validate that new_password and confirm_new_password match"""
        if self.new_password != self.confirm_new_password:
            raise ValueError("Passwords do not match")
        return self


class PasswordResetRequest(BaseSchema):
    """Password reset request schema"""

    email: EmailStr = Field(..., description="User email address")


class PasswordReset(BaseSchema):
    """Password reset schema"""

    token: str = Field(..., description="Password reset token")
    new_password: str = Field(
        ..., min_length=8, max_length=128, description="New password"
    )
    confirm_new_password: str = Field(..., description="Confirm new password")

    @model_validator(mode="after")
    def passwords_match(self):
        """Validate that new_password and confirm_new_password match"""
        if self.new_password != self.confirm_new_password:
            raise ValueError("Passwords do not match")
        return self


class EmailVerificationRequest(BaseSchema):
    """Email verification request schema"""

    token: str = Field(..., description="Email verification token")


class MFASetupResponse(BaseSchema):
    """MFA setup response schema"""

    secret: str = Field(..., description="MFA secret key")
    qr_code_url: str = Field(..., description="QR code URL")
    backup_codes: List[str] = Field(..., description="Backup codes")


class MFAVerifyRequest(BaseSchema):
    """MFA verification request schema"""

    token: str = Field(..., description="MFA token")
    code: str = Field(..., min_length=6, max_length=6, description="MFA code")


class SessionResponse(BaseSchema):
    """Session response schema"""

    session_id: UUID = Field(..., description="Session ID")
    user_id: UUID = Field(..., description="User ID")
    device_info: dict = Field(..., description="Device information")
    created_at: datetime = Field(..., description="Session creation time")
    expires_at: datetime = Field(..., description="Session expiration time")
    is_active: bool = Field(..., description="Session active status")


class SessionListResponse(BaseSchema):
    """Session list response schema"""

    sessions: List[SessionResponse] = Field(..., description="List of active sessions")
    current_session_id: UUID = Field(..., description="Current session ID")


class LogoutRequest(BaseSchema):
    """Logout request schema"""

    session_id: Optional[UUID] = Field(None, description="Specific session to logout")
    logout_all: bool = Field(False, description="Logout from all sessions")
