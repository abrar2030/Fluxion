"""
Authentication service for Fluxion backend
"""

import logging
import secrets
import string
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

import bcrypt
from config.settings import settings
from models.compliance import AuditLog, AuditLogLevel
from models.user import User, UserActivity, UserSession, UserStatus
from schemas.auth import PasswordChange, TokenResponse, UserLogin, UserRegister
from services.auth.jwt_service import JWTService
from services.auth.mfa_service import MFAService
from services.auth.security_service import SecurityService
from services.auth.session_service import SessionService
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication error exception"""

    pass


class AuthorizationError(Exception):
    """Authorization error exception"""

    pass


class AuthService:
    """Authentication service"""

    def __init__(
        self,
        jwt_service: JWTService,
        mfa_service: MFAService,
        session_service: SessionService,
        security_service: SecurityService,
    ):
        self.jwt_service = jwt_service
        self.mfa_service = mfa_service
        self.session_service = session_service
        self.security_service = security_service

    async def register_user(
        self,
        db: AsyncSession,
        user_data: UserRegister,
        ip_address: str,
        user_agent: str,
    ) -> Tuple[User, str]:
        """Register a new user"""
        try:
            # Check if user already exists
            existing_user = await self._get_user_by_email(db, user_data.email)
            if existing_user:
                raise AuthenticationError("User with this email already exists")

            # Check username uniqueness if provided
            if user_data.username:
                existing_username = await self._get_user_by_username(
                    db, user_data.username
                )
                if existing_username:
                    raise AuthenticationError("Username already taken")

            # Hash password
            hashed_password = self._hash_password(user_data.password)

            # Create user
            user = User(
                email=user_data.email,
                username=user_data.username,
                hashed_password=hashed_password,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                phone_number=user_data.phone_number,
                country=user_data.country,
                status=UserStatus.PENDING,
                terms_accepted_at=datetime.utcnow(),
                privacy_accepted_at=datetime.utcnow(),
            )

            db.add(user)
            await db.flush()

            # Generate email verification token
            verification_token = self._generate_verification_token()

            # Log registration
            await self._log_user_activity(
                db,
                user.id,
                None,
                "user_registration",
                f"User registered with email {user_data.email}",
                ip_address,
                user_agent,
            )

            await self._create_audit_log(
                db,
                user.id,
                "user_registration",
                "User Registration",
                f"New user registered: {user_data.email}",
                ip_address,
                user_agent,
            )

            await db.commit()

            logger.info(f"User registered successfully: {user.email}")
            return user, verification_token

        except Exception as e:
            await db.rollback()
            logger.error(f"User registration failed: {str(e)}")
            raise

    async def authenticate_user(
        self, db: AsyncSession, login_data: UserLogin, ip_address: str, user_agent: str
    ) -> Tuple[User, TokenResponse, Optional[UserSession]]:
        """Authenticate user and return tokens"""
        try:
            # Get user by email
            user = await self._get_user_by_email(db, login_data.email)
            if not user:
                await self._log_failed_login(
                    db, login_data.email, "user_not_found", ip_address, user_agent
                )
                raise AuthenticationError("Invalid credentials")

            # Check if user account is locked
            if user.is_locked():
                await self._log_failed_login(
                    db, login_data.email, "account_locked", ip_address, user_agent
                )
                raise AuthenticationError("Account is locked")

            # Verify password
            if not self._verify_password(login_data.password, user.hashed_password):
                await self._handle_failed_login(db, user, ip_address, user_agent)
                raise AuthenticationError("Invalid credentials")

            # Check if email is verified
            if not user.is_email_verified:
                raise AuthenticationError("Email not verified")

            # Check MFA if enabled
            if user.mfa_enabled:
                if not login_data.mfa_code:
                    raise AuthenticationError("MFA code required")

                if not await self.mfa_service.verify_code(user, login_data.mfa_code):
                    await self._handle_failed_login(db, user, ip_address, user_agent)
                    raise AuthenticationError("Invalid MFA code")

            # Reset failed login attempts
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login_at = datetime.utcnow()
            user.last_login_ip = ip_address

            # Create session
            session = await self.session_service.create_session(
                db, user.id, ip_address, user_agent, login_data.remember_me
            )

            # Generate tokens
            token_data = {
                "user_id": str(user.id),
                "session_id": str(session.id),
                "email": user.email,
                "role": user.role.value,
            }

            access_token = self.jwt_service.create_access_token(token_data)
            refresh_token = self.jwt_service.create_refresh_token(token_data)

            # Update session with tokens
            session.session_token = access_token
            session.refresh_token = refresh_token

            # Create token response
            token_response = TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=settings.auth.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )

            # Log successful login
            await self._log_user_activity(
                db,
                user.id,
                session.id,
                "user_login",
                "User logged in successfully",
                ip_address,
                user_agent,
            )

            await self._create_audit_log(
                db,
                user.id,
                "user_login",
                "User Login",
                f"User logged in: {user.email}",
                ip_address,
                user_agent,
            )

            await db.commit()

            logger.info(f"User authenticated successfully: {user.email}")
            return user, token_response, session

        except AuthenticationError:
            await db.rollback()
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"Authentication failed: {str(e)}")
            raise AuthenticationError("Authentication failed")

    async def refresh_token(
        self, db: AsyncSession, refresh_token: str, ip_address: str, user_agent: str
    ) -> TokenResponse:
        """Refresh access token"""
        try:
            # Verify refresh token
            payload = self.jwt_service.verify_refresh_token(refresh_token)
            user_id = UUID(payload["user_id"])
            session_id = UUID(payload["session_id"])

            # Get user and session
            user = await self._get_user_by_id(db, user_id)
            if not user or not user.can_login():
                raise AuthenticationError("Invalid user")

            session = await self.session_service.get_session(db, session_id)
            if not session or not session.is_valid():
                raise AuthenticationError("Invalid session")

            # Generate new tokens
            token_data = {
                "user_id": str(user.id),
                "session_id": str(session.id),
                "email": user.email,
                "role": user.role.value,
            }

            access_token = self.jwt_service.create_access_token(token_data)
            new_refresh_token = self.jwt_service.create_refresh_token(token_data)

            # Update session
            session.session_token = access_token
            session.refresh_token = new_refresh_token
            session.last_activity_at = datetime.utcnow()

            token_response = TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                token_type="bearer",
                expires_in=settings.auth.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )

            # Log token refresh
            await self._log_user_activity(
                db,
                user.id,
                session.id,
                "token_refresh",
                "Access token refreshed",
                ip_address,
                user_agent,
            )

            await db.commit()

            logger.info(f"Token refreshed for user: {user.email}")
            return token_response

        except Exception as e:
            await db.rollback()
            logger.error(f"Token refresh failed: {str(e)}")
            raise AuthenticationError("Token refresh failed")

    async def logout_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        session_id: UUID,
        ip_address: str,
        user_agent: str,
    ) -> None:
        """Logout user and invalidate session"""
        try:
            # Invalidate session
            await self.session_service.invalidate_session(db, session_id)

            # Log logout
            await self._log_user_activity(
                db,
                user_id,
                session_id,
                "user_logout",
                "User logged out",
                ip_address,
                user_agent,
            )

            await self._create_audit_log(
                db,
                user_id,
                "user_logout",
                "User Logout",
                "User logged out",
                ip_address,
                user_agent,
            )

            await db.commit()

            logger.info(f"User logged out: {user_id}")

        except Exception as e:
            await db.rollback()
            logger.error(f"Logout failed: {str(e)}")
            raise

    async def change_password(
        self,
        db: AsyncSession,
        user_id: UUID,
        password_data: PasswordChange,
        ip_address: str,
        user_agent: str,
    ) -> None:
        """Change user password"""
        try:
            user = await self._get_user_by_id(db, user_id)
            if not user:
                raise AuthenticationError("User not found")

            # Verify current password
            if not self._verify_password(
                password_data.current_password, user.hashed_password
            ):
                raise AuthenticationError("Invalid current password")

            # Hash new password
            new_hashed_password = self._hash_password(password_data.new_password)

            # Update password
            user.hashed_password = new_hashed_password
            user.password_changed_at = datetime.utcnow()

            # Invalidate all sessions except current one
            await self.session_service.invalidate_user_sessions(
                db, user_id, exclude_current=True
            )

            # Log password change
            await self._log_user_activity(
                db,
                user_id,
                None,
                "password_change",
                "Password changed successfully",
                ip_address,
                user_agent,
            )

            await self._create_audit_log(
                db,
                user_id,
                "password_change",
                "Password Change",
                "User changed password",
                ip_address,
                user_agent,
                level=AuditLogLevel.WARNING,
            )

            await db.commit()

            logger.info(f"Password changed for user: {user.email}")

        except Exception as e:
            await db.rollback()
            logger.error(f"Password change failed: {str(e)}")
            raise

    async def verify_email(
        self, db: AsyncSession, token: str, ip_address: str, user_agent: str
    ) -> User:
        """Verify user email"""
        try:
            # In a real implementation, you would verify the token
            # For now, we'll assume the token is valid and contains user_id

            # This is a simplified implementation
            # In production, you would store verification tokens in database
            # and verify them properly

            user_id = self._extract_user_id_from_token(token)
            user = await self._get_user_by_id(db, user_id)

            if not user:
                raise AuthenticationError("Invalid verification token")

            if user.is_email_verified:
                raise AuthenticationError("Email already verified")

            # Mark email as verified
            user.is_email_verified = True
            user.status = UserStatus.ACTIVE

            # Log email verification
            await self._log_user_activity(
                db,
                user.id,
                None,
                "email_verification",
                "Email verified successfully",
                ip_address,
                user_agent,
            )

            await self._create_audit_log(
                db,
                user.id,
                "email_verification",
                "Email Verification",
                f"Email verified: {user.email}",
                ip_address,
                user_agent,
            )

            await db.commit()

            logger.info(f"Email verified for user: {user.email}")
            return user

        except Exception as e:
            await db.rollback()
            logger.error(f"Email verification failed: {str(e)}")
            raise

    # Private helper methods

    async def _get_user_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email"""
        result = await db.execute(
            select(User).where(User.email == email, User.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def _get_user_by_username(
        self, db: AsyncSession, username: str
    ) -> Optional[User]:
        """Get user by username"""
        result = await db.execute(
            select(User).where(User.username == username, User.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def _get_user_by_id(self, db: AsyncSession, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(
            select(User).where(User.id == user_id, User.is_deleted == False)
        )
        return result.scalar_one_or_none()

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

    def _generate_verification_token(self) -> str:
        """Generate email verification token"""
        return "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(32)
        )

    def _extract_user_id_from_token(self, token: str) -> UUID:
        """Extract user ID from verification token"""
        # This is a simplified implementation
        # In production, you would properly decode and verify the token
        # For now, we'll return a dummy UUID
        return UUID("00000000-0000-0000-0000-000000000000")

    async def _handle_failed_login(
        self, db: AsyncSession, user: User, ip_address: str, user_agent: str
    ) -> None:
        """Handle failed login attempt"""
        user.failed_login_attempts += 1

        # Lock account if too many failed attempts
        if user.failed_login_attempts >= settings.auth.MAX_LOGIN_ATTEMPTS:
            user.status = UserStatus.LOCKED
            user.locked_until = datetime.utcnow() + timedelta(
                minutes=settings.auth.ACCOUNT_LOCKOUT_DURATION
            )

        await self._log_failed_login(
            db, user.email, "invalid_credentials", ip_address, user_agent
        )

    async def _log_failed_login(
        self,
        db: AsyncSession,
        email: str,
        reason: str,
        ip_address: str,
        user_agent: str,
    ) -> None:
        """Log failed login attempt"""
        await self._create_audit_log(
            db,
            None,
            "failed_login",
            "Failed Login Attempt",
            f"Failed login attempt for {email}: {reason}",
            ip_address,
            user_agent,
            level=AuditLogLevel.WARNING,
        )

    async def _log_user_activity(
        self,
        db: AsyncSession,
        user_id: UUID,
        session_id: Optional[UUID],
        activity_type: str,
        description: str,
        ip_address: str,
        user_agent: str,
    ) -> None:
        """Log user activity"""
        activity = UserActivity(
            user_id=user_id,
            session_id=session_id,
            activity_type=activity_type,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(activity)

    async def _create_audit_log(
        self,
        db: AsyncSession,
        user_id: Optional[UUID],
        event_type: str,
        title: str,
        description: str,
        ip_address: str,
        user_agent: str,
        level: AuditLogLevel = AuditLogLevel.INFO,
    ) -> None:
        """Create audit log entry"""
        audit_log = AuditLog(
            user_id=user_id,
            event_type=event_type,
            event_category="authentication",
            level=level,
            title=title,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(audit_log)
