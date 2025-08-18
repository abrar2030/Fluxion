"""
JWT service for Fluxion backend
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from config.settings import settings

logger = logging.getLogger(__name__)


class JWTError(Exception):
    """JWT error exception"""
    pass


class JWTService:
    """JWT token service"""
    
    def __init__(self):
        self.secret_key = settings.auth.JWT_SECRET_KEY
        self.algorithm = settings.auth.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.auth.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.auth.REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create access token"""
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "access"
            })
            
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            
            logger.debug(f"Access token created for user: {data.get('user_id')}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create access token: {str(e)}")
            raise JWTError("Failed to create access token")
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create refresh token"""
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "refresh"
            })
            
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            
            logger.debug(f"Refresh token created for user: {data.get('user_id')}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {str(e)}")
            raise JWTError("Failed to create refresh token")
    
    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode access token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access":
                raise JWTError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token expired")
            
            logger.debug(f"Access token verified for user: {payload.get('user_id')}")
            return payload
            
        except ExpiredSignatureError:
            logger.warning("Access token expired")
            raise JWTError("Token expired")
        except InvalidTokenError as e:
            logger.warning(f"Invalid access token: {str(e)}")
            raise JWTError("Invalid token")
        except Exception as e:
            logger.error(f"Failed to verify access token: {str(e)}")
            raise JWTError("Token verification failed")
    
    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode refresh token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "refresh":
                raise JWTError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token expired")
            
            logger.debug(f"Refresh token verified for user: {payload.get('user_id')}")
            return payload
            
        except ExpiredSignatureError:
            logger.warning("Refresh token expired")
            raise JWTError("Token expired")
        except InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {str(e)}")
            raise JWTError("Invalid token")
        except Exception as e:
            logger.error(f"Failed to verify refresh token: {str(e)}")
            raise JWTError("Token verification failed")
    
    def decode_token_without_verification(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode token without verification (for debugging)"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except Exception as e:
            logger.error(f"Failed to decode token: {str(e)}")
            return None
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """Get token expiry time"""
        try:
            payload = self.decode_token_without_verification(token)
            if payload and "exp" in payload:
                return datetime.fromtimestamp(payload["exp"])
            return None
        except Exception:
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired"""
        try:
            expiry = self.get_token_expiry(token)
            if expiry:
                return datetime.utcnow() > expiry
            return True
        except Exception:
            return True
    
    def get_token_remaining_time(self, token: str) -> Optional[timedelta]:
        """Get remaining time for token"""
        try:
            expiry = self.get_token_expiry(token)
            if expiry:
                remaining = expiry - datetime.utcnow()
                return remaining if remaining.total_seconds() > 0 else timedelta(0)
            return None
        except Exception:
            return None
    
    def create_password_reset_token(self, user_id: str, email: str) -> str:
        """Create password reset token"""
        try:
            data = {
                "user_id": user_id,
                "email": email,
                "type": "password_reset"
            }
            expire = datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
            
            data.update({
                "exp": expire,
                "iat": datetime.utcnow()
            })
            
            encoded_jwt = jwt.encode(data, self.secret_key, algorithm=self.algorithm)
            
            logger.debug(f"Password reset token created for user: {user_id}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create password reset token: {str(e)}")
            raise JWTError("Failed to create password reset token")
    
    def verify_password_reset_token(self, token: str) -> Dict[str, Any]:
        """Verify password reset token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "password_reset":
                raise JWTError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token expired")
            
            logger.debug(f"Password reset token verified for user: {payload.get('user_id')}")
            return payload
            
        except ExpiredSignatureError:
            logger.warning("Password reset token expired")
            raise JWTError("Token expired")
        except InvalidTokenError as e:
            logger.warning(f"Invalid password reset token: {str(e)}")
            raise JWTError("Invalid token")
        except Exception as e:
            logger.error(f"Failed to verify password reset token: {str(e)}")
            raise JWTError("Token verification failed")
    
    def create_email_verification_token(self, user_id: str, email: str) -> str:
        """Create email verification token"""
        try:
            data = {
                "user_id": user_id,
                "email": email,
                "type": "email_verification"
            }
            expire = datetime.utcnow() + timedelta(days=7)  # 7 days expiry
            
            data.update({
                "exp": expire,
                "iat": datetime.utcnow()
            })
            
            encoded_jwt = jwt.encode(data, self.secret_key, algorithm=self.algorithm)
            
            logger.debug(f"Email verification token created for user: {user_id}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create email verification token: {str(e)}")
            raise JWTError("Failed to create email verification token")
    
    def verify_email_verification_token(self, token: str) -> Dict[str, Any]:
        """Verify email verification token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "email_verification":
                raise JWTError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token expired")
            
            logger.debug(f"Email verification token verified for user: {payload.get('user_id')}")
            return payload
            
        except ExpiredSignatureError:
            logger.warning("Email verification token expired")
            raise JWTError("Token expired")
        except InvalidTokenError as e:
            logger.warning(f"Invalid email verification token: {str(e)}")
            raise JWTError("Invalid token")
        except Exception as e:
            logger.error(f"Failed to verify email verification token: {str(e)}")
            raise JWTError("Token verification failed")
    
    def create_api_key_token(self, user_id: str, api_key_id: str, permissions: list) -> str:
        """Create API key token"""
        try:
            data = {
                "user_id": user_id,
                "api_key_id": api_key_id,
                "permissions": permissions,
                "type": "api_key"
            }
            # API keys don't expire by default, but can have custom expiry
            
            data.update({
                "iat": datetime.utcnow()
            })
            
            encoded_jwt = jwt.encode(data, self.secret_key, algorithm=self.algorithm)
            
            logger.debug(f"API key token created for user: {user_id}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create API key token: {str(e)}")
            raise JWTError("Failed to create API key token")
    
    def verify_api_key_token(self, token: str) -> Dict[str, Any]:
        """Verify API key token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "api_key":
                raise JWTError("Invalid token type")
            
            logger.debug(f"API key token verified for user: {payload.get('user_id')}")
            return payload
            
        except InvalidTokenError as e:
            logger.warning(f"Invalid API key token: {str(e)}")
            raise JWTError("Invalid token")
        except Exception as e:
            logger.error(f"Failed to verify API key token: {str(e)}")
            raise JWTError("Token verification failed")

