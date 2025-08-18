"""
Comprehensive Security Middleware for Fluxion Backend
Implements enterprise-grade security controls including CSRF protection,
XSS prevention, security headers, and threat detection.
"""

import time
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from ipaddress import ip_address, ip_network

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config.settings import settings
from services.auth.jwt_service import JWTService
from services.security.threat_detection_service import ThreatDetectionService
from services.security.encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware implementing multiple security controls:
    - Security headers (HSTS, CSP, X-Frame-Options, etc.)
    - CSRF protection
    - XSS prevention
    - Rate limiting integration
    - IP whitelisting/blacklisting
    - Threat detection
    - Request/response sanitization
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.jwt_service = JWTService()
        self.threat_detection = ThreatDetectionService()
        self.encryption_service = EncryptionService()
        
        # Security configuration
        self.csrf_token_expiry = timedelta(hours=1)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.blocked_ips: Set[str] = set()
        self.allowed_ips: Set[str] = set()
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'eval\s*\(',
            r'document\.cookie',
            r'union\s+select',
            r'drop\s+table',
            r'insert\s+into',
            r'delete\s+from'
        ]
        
        # Initialize IP lists from settings
        self._load_ip_configurations()
        
        # Security headers configuration
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            'Content-Security-Policy': self._build_csp_header(),
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            'X-Permitted-Cross-Domain-Policies': 'none',
            'X-Download-Options': 'noopen'
        }
    
    def _load_ip_configurations(self):
        """Load IP whitelist and blacklist from configuration"""
        # In production, these would be loaded from database or configuration service
        pass
    
    def _build_csp_header(self) -> str:
        """Build Content Security Policy header"""
        if settings.app.ENVIRONMENT == 'development':
            return "default-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' data: https:; connect-src 'self' ws: wss:"
        else:
            return "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self'; font-src 'self'; object-src 'none'; media-src 'self'; frame-src 'none';"
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch method"""
        start_time = time.time()
        
        try:
            # Pre-request security checks
            await self._pre_request_security_checks(request)
            
            # Process request
            response = await call_next(request)
            
            # Post-request security enhancements
            await self._post_request_security_enhancements(request, response)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log security metrics
            self._log_security_metrics(request, response, time.time() - start_time)
            
            return response
            
        except HTTPException as e:
            logger.warning(f"Security middleware blocked request: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "error_code": "SECURITY_VIOLATION"}
            )
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal security error", "error_code": "SECURITY_ERROR"}
            )
    
    async def _pre_request_security_checks(self, request: Request):
        """Perform pre-request security validations"""
        # Check request size
        if hasattr(request, 'content_length') and request.content_length:
            if request.content_length > self.max_request_size:
                raise HTTPException(status_code=413, detail="Request too large")
        
        # IP-based access control
        client_ip = self._get_client_ip(request)
        if client_ip:
            await self._check_ip_access(client_ip)
        
        # Check for suspicious patterns in URL and headers
        await self._check_suspicious_patterns(request)
        
        # CSRF protection for state-changing operations
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            await self._validate_csrf_token(request)
        
        # Rate limiting check (delegated to rate limit middleware)
        # Threat detection
        await self._detect_threats(request)
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address from request"""
        # Check X-Forwarded-For header (from load balancer/proxy)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(',')[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()
        
        # Fall back to direct connection IP
        if request.client:
            return request.client.host
        
        return None
    
    async def _check_ip_access(self, client_ip: str):
        """Check IP-based access control"""
        try:
            ip_addr = ip_address(client_ip)
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                logger.warning(f"Blocked IP attempted access: {client_ip}")
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Check whitelist if configured
            if self.allowed_ips and client_ip not in self.allowed_ips:
                # Check if IP is in allowed networks
                allowed = False
                for allowed_ip in self.allowed_ips:
                    try:
                        if ip_addr in ip_network(allowed_ip, strict=False):
                            allowed = True
                            break
                    except ValueError:
                        # Not a network, check exact match
                        if client_ip == allowed_ip:
                            allowed = True
                            break
                
                if not allowed:
                    logger.warning(f"Non-whitelisted IP attempted access: {client_ip}")
                    raise HTTPException(status_code=403, detail="Access denied")
                    
        except ValueError as e:
            logger.error(f"Invalid IP address format: {client_ip}")
            raise HTTPException(status_code=400, detail="Invalid request")
    
    async def _check_suspicious_patterns(self, request: Request):
        """Check for suspicious patterns in request"""
        import re
        
        # Check URL path
        url_path = str(request.url.path)
        for pattern in self.suspicious_patterns:
            if re.search(pattern, url_path, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in URL: {pattern}")
                await self.threat_detection.log_threat(
                    request, "suspicious_pattern", f"Pattern: {pattern}"
                )
                raise HTTPException(status_code=400, detail="Invalid request")
        
        # Check headers
        for header_name, header_value in request.headers.items():
            for pattern in self.suspicious_patterns:
                if re.search(pattern, header_value, re.IGNORECASE):
                    logger.warning(f"Suspicious pattern detected in header {header_name}: {pattern}")
                    await self.threat_detection.log_threat(
                        request, "suspicious_header", f"Header: {header_name}, Pattern: {pattern}"
                    )
                    raise HTTPException(status_code=400, detail="Invalid request")
    
    async def _validate_csrf_token(self, request: Request):
        """Validate CSRF token for state-changing operations"""
        # Skip CSRF validation for API endpoints with proper authentication
        if request.url.path.startswith('/api/'):
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                # API requests with valid JWT tokens are exempt from CSRF
                try:
                    token = auth_header.split(' ')[1]
                    await self.jwt_service.verify_token(token)
                    return
                except Exception:
                    pass
        
        # Check CSRF token in header or form data
        csrf_token = request.headers.get('X-CSRF-Token')
        if not csrf_token:
            # Try to get from form data for non-JSON requests
            if request.headers.get('content-type', '').startswith('application/x-www-form-urlencoded'):
                form_data = await request.form()
                csrf_token = form_data.get('csrf_token')
        
        if not csrf_token:
            logger.warning("Missing CSRF token for state-changing operation")
            raise HTTPException(status_code=403, detail="CSRF token required")
        
        # Validate CSRF token
        if not await self._verify_csrf_token(csrf_token, request):
            logger.warning("Invalid CSRF token")
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
    
    async def _verify_csrf_token(self, token: str, request: Request) -> bool:
        """Verify CSRF token validity"""
        try:
            # In production, CSRF tokens would be stored in Redis or database
            # For now, we'll use a simple HMAC-based approach
            expected_token = self._generate_csrf_token(request)
            return secrets.compare_digest(token, expected_token)
        except Exception as e:
            logger.error(f"CSRF token verification error: {str(e)}")
            return False
    
    def _generate_csrf_token(self, request: Request) -> str:
        """Generate CSRF token for the session"""
        # In production, this would use session ID and timestamp
        session_id = request.headers.get('X-Session-ID', 'anonymous')
        timestamp = str(int(time.time() // 3600))  # Hour-based token
        
        token_data = f"{session_id}:{timestamp}:{settings.security.SECRET_KEY}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    async def _detect_threats(self, request: Request):
        """Detect potential security threats"""
        # Check for common attack patterns
        user_agent = request.headers.get('User-Agent', '')
        
        # Detect bot/scanner patterns
        suspicious_user_agents = [
            'sqlmap', 'nikto', 'nmap', 'masscan', 'zap', 'burp',
            'acunetix', 'nessus', 'openvas', 'w3af'
        ]
        
        for suspicious_ua in suspicious_user_agents:
            if suspicious_ua.lower() in user_agent.lower():
                logger.warning(f"Suspicious user agent detected: {user_agent}")
                await self.threat_detection.log_threat(
                    request, "suspicious_user_agent", f"User-Agent: {user_agent}"
                )
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Check request frequency (basic rate limiting)
        client_ip = self._get_client_ip(request)
        if client_ip:
            await self.threat_detection.check_request_frequency(client_ip, request)
    
    async def _post_request_security_enhancements(self, request: Request, response: Response):
        """Apply post-request security enhancements"""
        # Add request ID for tracking
        request_id = getattr(request.state, 'request_id', None)
        if not request_id:
            request_id = secrets.token_hex(16)
            request.state.request_id = request_id
        
        response.headers['X-Request-ID'] = request_id
        
        # Remove sensitive headers from response
        sensitive_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version']
        for header in sensitive_headers:
            if header in response.headers:
                del response.headers[header]
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value
        
        # Add cache control for sensitive endpoints
        if hasattr(response, 'url') and '/api/' in str(response.url):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
    
    def _log_security_metrics(self, request: Request, response: Response, duration: float):
        """Log security-related metrics"""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        security_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'client_ip': client_ip,
            'method': request.method,
            'path': str(request.url.path),
            'status_code': response.status_code,
            'user_agent': user_agent,
            'duration': duration,
            'request_id': getattr(request.state, 'request_id', None)
        }
        
        # Log to security monitoring system
        logger.info(f"Security metrics: {security_log}")
    
    async def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"IP {ip_address} blocked: {reason}")
        
        # In production, this would update the database/cache
        await self.threat_detection.log_ip_block(ip_address, reason)
    
    async def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip_address)
        logger.info(f"IP {ip_address} unblocked")
    
    def get_csrf_token(self, request: Request) -> str:
        """Generate CSRF token for client"""
        return self._generate_csrf_token(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Lightweight security headers middleware for cases where full SecurityMiddleware is not needed
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value
        
        return response

