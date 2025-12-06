"""
Authentication schemas for Fluxion backend
"""

import re
from code.backend.schemas.base import BaseSchema, TimestampSchema
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import EmailStr, Field, validator


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

    @validator("password")
    def validate_password(cls: Any, v: Any) -> Any:
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

    @validator("confirm_password")
    def passwords_match(cls: Any, v: Any, values: Any) -> Any:
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator("username")
    def validate_username(cls: Any, v: Any) -> Any:
        if v and (not re.match("^[a-zA-Z0-9_]+$", v)):
            raise ValueError(
                "Username can only contain letters, numbers, and underscores"
            )
        return v

    @validator("terms_accepted", "privacy_accepted")
    def validate_acceptance(cls: Any, v: Any) -> Any:
        if not v:
            raise ValueError("Terms and privacy policy must be accepted")
        return v


class UserLogin(BaseSchema):
    """User login schema"""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Remember login session")
    mfa_code: Optional[str] = Field(
        None, min_length=6, max_length=6, description="MFA code"
    )


class TokenResponse(BaseSchema):
    """Token response schema"""

    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    scope: Optional[str] = Field(None, description="Token scope")


class RefreshTokenRequest(BaseSchema):
    """Refresh token request schema"""

    refresh_token: str = Field(..., description="Refresh token")


class PasswordChange(BaseSchema):
    """Password change schema"""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ..., min_length=8, max_length=128, description="New password"
    )
    confirm_password: str = Field(..., description="New password confirmation")

    @validator("new_password")
    def validate_new_password(cls: Any, v: Any) -> Any:
        """Validate new password strength"""
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

    @validator("confirm_password")
    def passwords_match(cls: Any, v: Any, values: Any) -> Any:
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class PasswordReset(BaseSchema):
    """Password reset schema"""

    email: EmailStr = Field(..., description="User email address")


class PasswordResetConfirm(BaseSchema):
    """Password reset confirmation schema"""

    token: str = Field(..., description="Reset token")
    new_password: str = Field(
        ..., min_length=8, max_length=128, description="New password"
    )
    confirm_password: str = Field(..., description="New password confirmation")

    @validator("new_password")
    def validate_new_password(cls: Any, v: Any) -> Any:
        """Validate new password strength"""
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

    @validator("confirm_password")
    def passwords_match(cls: Any, v: Any, values: Any) -> Any:
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class MFASetup(BaseSchema):
    """MFA setup schema"""

    method: str = Field(..., regex="^(totp|sms)$", description="MFA method")
    phone_number: Optional[str] = Field(None, description="Phone number for SMS MFA")

    @validator("phone_number")
    def validate_phone_for_sms(cls: Any, v: Any, values: Any) -> Any:
        if values.get("method") == "sms" and (not v):
            raise ValueError("Phone number is required for SMS MFA")
        return v


class MFASetupResponse(BaseSchema):
    """MFA setup response schema"""

    method: str = Field(..., description="MFA method")
    secret: Optional[str] = Field(None, description="TOTP secret")
    qr_code: Optional[str] = Field(None, description="QR code URL")
    backup_codes: List[str] = Field(..., description="Backup codes")


class MFAVerify(BaseSchema):
    """MFA verification schema"""

    code: str = Field(..., min_length=6, max_length=6, description="MFA code")
    method: str = Field(..., regex="^(totp|sms|backup)$", description="MFA method")


class MFADisable(BaseSchema):
    """MFA disable schema"""

    password: str = Field(..., description="Current password")
    code: str = Field(..., min_length=6, max_length=6, description="MFA code")


class EmailVerification(BaseSchema):
    """Email verification schema"""

    token: str = Field(..., description="Verification token")


class UserResponse(TimestampSchema):
    """User response schema"""

    id: UUID = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    username: Optional[str] = Field(None, description="Username")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    status: str = Field(..., description="User status")
    role: str = Field(..., description="User role")
    is_email_verified: bool = Field(..., description="Email verification status")
    is_phone_verified: bool = Field(..., description="Phone verification status")
    mfa_enabled: bool = Field(..., description="MFA enabled status")
    kyc_status: str = Field(..., description="KYC status")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")
    profile_completion_percentage: Optional[int] = Field(
        None, description="Profile completion percentage"
    )


class SessionResponse(TimestampSchema):
    """Session response schema"""

    id: UUID = Field(..., description="Session ID")
    user_id: UUID = Field(..., description="User ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    is_active: bool = Field(..., description="Session active status")
    expires_at: datetime = Field(..., description="Session expiry")
    last_activity_at: Optional[datetime] = Field(
        None, description="Last activity timestamp"
    )


class LoginAttempt(BaseSchema):
    """Login attempt schema"""

    email: str = Field(..., description="Email address")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    success: bool = Field(..., description="Login success status")
    failure_reason: Optional[str] = Field(None, description="Failure reason")
    timestamp: datetime = Field(..., description="Attempt timestamp")


class SecurityEvent(BaseSchema):
    """Security event schema"""

    event_type: str = Field(..., description="Event type")
    user_id: Optional[UUID] = Field(None, description="User ID")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    details: dict = Field(..., description="Event details")
    risk_score: Optional[float] = Field(None, description="Risk score")
    timestamp: datetime = Field(..., description="Event timestamp")


class PermissionResponse(BaseSchema):
    """Permission response schema"""

    id: UUID = Field(..., description="Permission ID")
    name: str = Field(..., description="Permission name")
    description: str = Field(..., description="Permission description")
    resource: str = Field(..., description="Resource")
    action: str = Field(..., description="Action")


class RoleResponse(BaseSchema):
    """Role response schema"""

    id: UUID = Field(..., description="Role ID")
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    permissions: List[PermissionResponse] = Field(..., description="Role permissions")


class APIKeyCreate(BaseSchema):
    """API key creation schema"""

    name: str = Field(..., max_length=100, description="API key name")
    description: Optional[str] = Field(
        None, max_length=500, description="API key description"
    )
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    permissions: List[str] = Field(..., description="API key permissions")


class APIKeyResponse(TimestampSchema):
    """API key response schema"""

    id: UUID = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    description: Optional[str] = Field(None, description="API key description")
    key_prefix: str = Field(..., description="API key prefix")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last used timestamp")
    is_active: bool = Field(..., description="API key active status")
    permissions: List[str] = Field(..., description="API key permissions")


class DeviceInfo(BaseSchema):
    """Device information schema"""

    device_id: Optional[str] = Field(None, description="Device ID")
    device_type: Optional[str] = Field(None, description="Device type")
    os: Optional[str] = Field(None, description="Operating system")
    browser: Optional[str] = Field(None, description="Browser")
    location: Optional[dict] = Field(None, description="Location information")


class TrustedDevice(TimestampSchema):
    """Trusted device schema"""

    id: UUID = Field(..., description="Device ID")
    user_id: UUID = Field(..., description="User ID")
    device_fingerprint: str = Field(..., description="Device fingerprint")
    device_name: str = Field(..., description="Device name")
    device_info: DeviceInfo = Field(..., description="Device information")
    is_trusted: bool = Field(..., description="Trusted status")
    last_seen_at: datetime = Field(..., description="Last seen timestamp")
    expires_at: Optional[datetime] = Field(None, description="Trust expiration")
