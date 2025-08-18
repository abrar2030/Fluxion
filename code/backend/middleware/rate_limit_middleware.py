"""
Advanced Rate Limiting Middleware for Fluxion Backend
Implements sophisticated rate limiting with multiple algorithms,
user-based limits, and integration with threat detection.
"""

import time
import json
import asyncio
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    window_size: int = 60  # seconds
    block_duration: int = 300  # seconds (5 minutes)


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    requests_made: int
    requests_remaining: int
    reset_time: int
    blocked_until: Optional[int] = None
    violation_count: int = 0


class TokenBucket:
    """Token bucket algorithm implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket"""
        now = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_status(self) -> Dict:
        """Get current bucket status"""
        return {
            'tokens': self.tokens,
            'capacity': self.capacity,
            'refill_rate': self.refill_rate
        }


class SlidingWindowCounter:
    """Sliding window counter for precise rate limiting"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
    
    def add_request(self, timestamp: float) -> bool:
        """Add a request and check if limit is exceeded"""
        # Remove old requests outside the window
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        
        # Check if we can add this request
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(timestamp)
        return True
    
    def get_count(self, timestamp: float) -> int:
        """Get current request count in window"""
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        return len(self.requests)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with multiple algorithms and strategies:
    - Token bucket for burst handling
    - Sliding window for precise limits
    - User-based and IP-based limiting
    - Dynamic rate limit adjustment
    - Integration with Redis for distributed rate limiting
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Rate limiting configuration
        self.default_rules = {
            'anonymous': RateLimitRule(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=10
            ),
            'authenticated': RateLimitRule(
                requests_per_minute=120,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_limit=20
            ),
            'premium': RateLimitRule(
                requests_per_minute=300,
                requests_per_hour=15000,
                requests_per_day=150000,
                burst_limit=50
            ),
            'admin': RateLimitRule(
                requests_per_minute=1000,
                requests_per_hour=50000,
                requests_per_day=500000,
                burst_limit=100
            )
        }
        
        # Endpoint-specific rate limits
        self.endpoint_rules = {
            '/api/v1/auth/login': RateLimitRule(5, 20, 100, 2),
            '/api/v1/auth/register': RateLimitRule(3, 10, 50, 1),
            '/api/v1/auth/reset-password': RateLimitRule(2, 5, 20, 1),
            '/api/v1/transactions/create': RateLimitRule(30, 500, 2000, 5),
            '/api/v1/portfolio/update': RateLimitRule(60, 1000, 5000, 10),
        }
        
        # In-memory storage for development/testing
        self.ip_buckets: Dict[str, TokenBucket] = {}
        self.user_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindowCounter] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time
        self.violation_counts: Dict[str, int] = defaultdict(int)
        
        # Redis client for distributed rate limiting
        self.redis_client: Optional[redis.Redis] = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        try:
            if hasattr(settings, 'redis') and settings.redis.REDIS_URL:
                self.redis_client = redis.from_url(
                    str(settings.redis.REDIS_URL),
                    encoding='utf-8',
                    decode_responses=True,
                    max_connections=20
                )
                logger.info("Redis client initialized for distributed rate limiting")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis for rate limiting: {e}")
            self.redis_client = None
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch method"""
        try:
            # Check rate limits
            await self._check_rate_limits(request)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            await self._add_rate_limit_headers(request, response)
            
            return response
            
        except HTTPException as e:
            if e.status_code == 429:
                # Rate limit exceeded
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "retry_after": e.headers.get("Retry-After", "60")
                    },
                    headers=e.headers
                )
            raise
    
    async def _check_rate_limits(self, request: Request):
        """Check all applicable rate limits"""
        client_ip = self._get_client_ip(request)
        user_id = await self._get_user_id(request)
        endpoint = self._get_endpoint_key(request)
        
        current_time = time.time()
        
        # Check if IP is currently blocked
        if client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                retry_after = int(self.blocked_ips[client_ip] - current_time)
                raise HTTPException(
                    status_code=429,
                    detail="IP temporarily blocked due to rate limit violations",
                    headers={"Retry-After": str(retry_after)}
                )
            else:
                # Unblock IP
                del self.blocked_ips[client_ip]
                self.violation_counts[client_ip] = 0
        
        # Get applicable rate limit rules
        rules = await self._get_rate_limit_rules(request, user_id)
        
        # Check endpoint-specific limits first
        if endpoint in self.endpoint_rules:
            await self._check_endpoint_limit(client_ip, endpoint, current_time)
        
        # Check general rate limits
        await self._check_general_limits(client_ip, user_id, rules, current_time)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()
        
        # Fall back to direct connection IP
        if request.client:
            return request.client.host
        
        return 'unknown'
    
    async def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request (if authenticated)"""
        try:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                # In production, this would decode JWT token
                # For now, return a placeholder
                return 'authenticated_user'
        except Exception:
            pass
        return None
    
    def _get_endpoint_key(self, request: Request) -> str:
        """Get endpoint key for rate limiting"""
        return f"{request.method}:{request.url.path}"
    
    async def _get_rate_limit_rules(self, request: Request, user_id: Optional[str]) -> RateLimitRule:
        """Get applicable rate limit rules for the request"""
        # Determine user tier
        if user_id:
            # In production, this would check user's subscription tier
            user_tier = await self._get_user_tier(user_id)
            return self.default_rules.get(user_tier, self.default_rules['authenticated'])
        else:
            return self.default_rules['anonymous']
    
    async def _get_user_tier(self, user_id: str) -> str:
        """Get user's subscription tier"""
        # In production, this would query the database
        # For now, return default tier
        return 'authenticated'
    
    async def _check_endpoint_limit(self, client_ip: str, endpoint: str, current_time: float):
        """Check endpoint-specific rate limits"""
        if endpoint not in self.endpoint_rules:
            return
        
        rule = self.endpoint_rules[endpoint]
        key = f"{client_ip}:{endpoint}"
        
        if self.redis_client:
            await self._check_redis_limit(key, rule, current_time)
        else:
            await self._check_memory_limit(key, rule, current_time)
    
    async def _check_general_limits(self, client_ip: str, user_id: Optional[str], 
                                  rules: RateLimitRule, current_time: float):
        """Check general rate limits"""
        # IP-based limiting
        ip_key = f"ip:{client_ip}"
        await self._check_token_bucket_limit(ip_key, rules, current_time)
        
        # User-based limiting (if authenticated)
        if user_id:
            user_key = f"user:{user_id}"
            await self._check_token_bucket_limit(user_key, rules, current_time)
    
    async def _check_token_bucket_limit(self, key: str, rules: RateLimitRule, current_time: float):
        """Check token bucket rate limit"""
        if self.redis_client:
            await self._check_redis_token_bucket(key, rules, current_time)
        else:
            await self._check_memory_token_bucket(key, rules, current_time)
    
    async def _check_memory_token_bucket(self, key: str, rules: RateLimitRule, current_time: float):
        """Check token bucket limit using in-memory storage"""
        if key not in self.ip_buckets:
            # Create new token bucket
            capacity = rules.burst_limit
            refill_rate = rules.requests_per_minute / 60.0  # tokens per second
            self.ip_buckets[key] = TokenBucket(capacity, refill_rate)
        
        bucket = self.ip_buckets[key]
        if not bucket.consume(1):
            # Rate limit exceeded
            self.violation_counts[key] += 1
            
            # Block IP if too many violations
            if self.violation_counts[key] >= 5:
                block_duration = rules.block_duration * (2 ** (self.violation_counts[key] - 5))
                self.blocked_ips[key] = current_time + block_duration
                logger.warning(f"IP {key} blocked for {block_duration} seconds due to rate limit violations")
            
            retry_after = int(60 / bucket.refill_rate)  # Time to get one token
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)}
            )
    
    async def _check_redis_token_bucket(self, key: str, rules: RateLimitRule, current_time: float):
        """Check token bucket limit using Redis"""
        try:
            # Lua script for atomic token bucket operation
            lua_script = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local tokens_requested = tonumber(ARGV[4])
            
            local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket[1]) or capacity
            local last_refill = tonumber(bucket[2]) or current_time
            
            -- Refill tokens
            local elapsed = current_time - last_refill
            tokens = math.min(capacity, tokens + elapsed * refill_rate)
            
            if tokens >= tokens_requested then
                tokens = tokens - tokens_requested
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
                return {1, tokens}
            else
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, 3600)
                return {0, tokens}
            end
            """
            
            capacity = rules.burst_limit
            refill_rate = rules.requests_per_minute / 60.0
            
            result = await self.redis_client.eval(
                lua_script, 1, key, capacity, refill_rate, current_time, 1
            )
            
            if result[0] == 0:
                # Rate limit exceeded
                retry_after = int(60 / refill_rate)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry_after)}
                )
                
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fall back to memory-based limiting
            await self._check_memory_token_bucket(key, rules, current_time)
    
    async def _check_redis_limit(self, key: str, rule: RateLimitRule, current_time: float):
        """Check rate limit using Redis sliding window"""
        try:
            # Use Redis sorted sets for sliding window
            window_key = f"window:{key}"
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(
                window_key, 0, current_time - rule.window_size
            )
            
            # Count current requests
            count = await self.redis_client.zcard(window_key)
            
            if count >= rule.requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(rule.window_size)}
                )
            
            # Add current request
            await self.redis_client.zadd(window_key, {str(current_time): current_time})
            await self.redis_client.expire(window_key, rule.window_size)
            
        except Exception as e:
            logger.error(f"Redis sliding window error: {e}")
            # Fall back to memory-based limiting
            await self._check_memory_limit(key, rule, current_time)
    
    async def _check_memory_limit(self, key: str, rule: RateLimitRule, current_time: float):
        """Check rate limit using in-memory sliding window"""
        if key not in self.sliding_windows:
            self.sliding_windows[key] = SlidingWindowCounter(
                rule.window_size, rule.requests_per_minute
            )
        
        window = self.sliding_windows[key]
        if not window.add_request(current_time):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(rule.window_size)}
            )
    
    async def _add_rate_limit_headers(self, request: Request, response):
        """Add rate limit headers to response"""
        client_ip = self._get_client_ip(request)
        user_id = await self._get_user_id(request)
        
        # Get current rate limit status
        status = await self._get_rate_limit_status(client_ip, user_id)
        
        response.headers['X-RateLimit-Limit'] = str(status.requests_made + status.requests_remaining)
        response.headers['X-RateLimit-Remaining'] = str(status.requests_remaining)
        response.headers['X-RateLimit-Reset'] = str(status.reset_time)
        
        if status.blocked_until:
            response.headers['X-RateLimit-Blocked-Until'] = str(status.blocked_until)
    
    async def _get_rate_limit_status(self, client_ip: str, user_id: Optional[str]) -> RateLimitStatus:
        """Get current rate limit status"""
        # This is a simplified implementation
        # In production, this would aggregate status from all applicable limits
        
        key = f"ip:{client_ip}"
        if key in self.ip_buckets:
            bucket = self.ip_buckets[key]
            return RateLimitStatus(
                requests_made=int(bucket.capacity - bucket.tokens),
                requests_remaining=int(bucket.tokens),
                reset_time=int(time.time() + 60),
                blocked_until=self.blocked_ips.get(client_ip),
                violation_count=self.violation_counts.get(key, 0)
            )
        
        # Default status for new clients
        return RateLimitStatus(
            requests_made=0,
            requests_remaining=60,  # Default limit
            reset_time=int(time.time() + 60)
        )
    
    async def get_rate_limit_stats(self) -> Dict:
        """Get rate limiting statistics"""
        return {
            'active_buckets': len(self.ip_buckets),
            'blocked_ips': len(self.blocked_ips),
            'total_violations': sum(self.violation_counts.values()),
            'redis_connected': self.redis_client is not None
        }
    
    async def reset_rate_limit(self, identifier: str):
        """Reset rate limit for a specific identifier"""
        # Remove from all tracking structures
        keys_to_remove = [key for key in self.ip_buckets.keys() if identifier in key]
        for key in keys_to_remove:
            del self.ip_buckets[key]
            if key in self.violation_counts:
                del self.violation_counts[key]
        
        if identifier in self.blocked_ips:
            del self.blocked_ips[identifier]
        
        # Clear Redis data if available
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(f"*{identifier}*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Error clearing Redis rate limit data: {e}")
        
        logger.info(f"Rate limit reset for identifier: {identifier}")


class AdaptiveRateLimitMiddleware(RateLimitMiddleware):
    """
    Adaptive rate limiting that adjusts limits based on system load and user behavior
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.system_load_factor = 1.0
        self.user_behavior_scores: Dict[str, float] = defaultdict(lambda: 1.0)
    
    async def _get_rate_limit_rules(self, request: Request, user_id: Optional[str]) -> RateLimitRule:
        """Get adaptive rate limit rules"""
        base_rules = await super()._get_rate_limit_rules(request, user_id)
        
        # Adjust based on system load
        load_factor = await self._get_system_load_factor()
        
        # Adjust based on user behavior (if authenticated)
        behavior_factor = 1.0
        if user_id:
            behavior_factor = self.user_behavior_scores.get(user_id, 1.0)
        
        # Calculate adjusted limits
        adjustment_factor = load_factor * behavior_factor
        
        return RateLimitRule(
            requests_per_minute=int(base_rules.requests_per_minute * adjustment_factor),
            requests_per_hour=int(base_rules.requests_per_hour * adjustment_factor),
            requests_per_day=int(base_rules.requests_per_day * adjustment_factor),
            burst_limit=int(base_rules.burst_limit * adjustment_factor),
            window_size=base_rules.window_size,
            block_duration=base_rules.block_duration
        )
    
    async def _get_system_load_factor(self) -> float:
        """Calculate

