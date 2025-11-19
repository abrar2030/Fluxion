"""
Comprehensive Threat Detection Service for Fluxion Backend
Implements advanced threat detection, anomaly detection, and security monitoring
using machine learning and rule-based approaches.
"""

import ipaddress
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from statistics import mean
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import Request
from services.security.encryption_service import EncryptionService
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats"""

    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    CSRF_ATTACK = "csrf_attack"
    DDoS = "ddos"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    MALICIOUS_IP = "malicious_ip"
    BOT_ACTIVITY = "bot_activity"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ACCOUNT_TAKEOVER = "account_takeover"


@dataclass
class ThreatEvent:
    """Security threat event"""

    event_id: str
    timestamp: datetime
    threat_type: ThreatType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    request_path: str
    request_method: str
    user_agent: str
    risk_score: float
    confidence: float
    details: Dict[str, Any]
    indicators: List[str]
    recommended_actions: List[str]
    false_positive_probability: float


@dataclass
class BehaviorProfile:
    """User behavior profile for anomaly detection"""

    user_id: str
    typical_access_times: List[int]  # Hours of day
    typical_locations: Set[str]  # IP addresses or geographic locations
    typical_endpoints: Set[str]  # API endpoints accessed
    typical_request_frequency: float  # Requests per minute
    typical_session_duration: float  # Minutes
    risk_score_history: List[float]
    last_updated: datetime


class ThreatDetectionService:
    """
    Advanced threat detection service that uses multiple detection methods:
    - Rule-based detection for known attack patterns
    - Machine learning for anomaly detection
    - Behavioral analysis for user profiling
    - IP reputation and geolocation analysis
    - Statistical analysis for outlier detection
    """

    def __init__(self):
        self.encryption_service = EncryptionService()

        # Threat detection configuration
        self.max_failed_attempts = 5
        self.failed_attempt_window = timedelta(minutes=15)
        self.suspicious_request_threshold = 100  # requests per minute
        self.anomaly_threshold = 0.7

        # In-memory storage for real-time analysis
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.request_frequencies: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.user_profiles: Dict[str, BehaviorProfile] = {}
        self.ip_reputation: Dict[str, Dict[str, Any]] = {}
        self.threat_events: List[ThreatEvent] = []

        # Machine learning models
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self._initialize_ml_models()

        # Attack pattern signatures
        self.attack_patterns = {
            ThreatType.SQL_INJECTION: [
                r"(\bunion\b.*\bselect\b)",
                r"(\bselect\b.*\bfrom\b.*\bwhere\b)",
                r"(\bdrop\b.*\btable\b)",
                r"(\binsert\b.*\binto\b)",
                r"(\bdelete\b.*\bfrom\b)",
                r"(\bupdate\b.*\bset\b)",
                r"(\bor\b.*1\s*=\s*1)",
                r"(\band\b.*1\s*=\s*1)",
                r"(\bexec\b.*\bxp_)",
                r"(\bsp_executesql\b)",
            ],
            ThreatType.XSS_ATTACK: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"vbscript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"onclick\s*=",
                r"onmouseover\s*=",
                r"eval\s*\(",
                r"document\.cookie",
                r"document\.write",
            ],
            ThreatType.SUSPICIOUS_PATTERN: [
                r"\.\.\/",  # Directory traversal
                r"\/etc\/passwd",
                r"\/proc\/",
                r"cmd\.exe",
                r"powershell",
                r"base64",
                r"eval\(",
                r"exec\(",
                r"system\(",
                r"shell_exec\(",
            ],
        }

        # Known malicious user agents
        self.malicious_user_agents = {
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "zap",
            "burp",
            "acunetix",
            "nessus",
            "openvas",
            "w3af",
            "havij",
            "pangolin",
            "jsql",
            "bbqsql",
        }

        # Suspicious IP ranges (example - in production, use threat intelligence feeds)
        self.suspicious_ip_ranges = [
            ipaddress.ip_network("10.0.0.0/8"),  # Private networks (if external)
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
        ]

    def _initialize_ml_models(self):
        """Initialize machine learning models for anomaly detection"""
        try:
            # Initialize Isolation Forest for anomaly detection
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # Expected proportion of outliers
                random_state=42,
                n_estimators=100,
            )

            # In production, load pre-trained models
            logger.info("Machine learning models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            self.anomaly_detector = None

    async def analyze_request(self, request: Request) -> List[ThreatEvent]:
        """Analyze incoming request for security threats"""
        threats = []

        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        request_path = str(request.url.path)
        request_method = request.method
        user_id = await self._extract_user_id(request)

        # Rule-based threat detection
        rule_threats = await self._detect_rule_based_threats(
            request, client_ip, user_agent, request_path, request_method, user_id
        )
        threats.extend(rule_threats)

        # Behavioral anomaly detection
        if user_id:
            behavior_threats = await self._detect_behavioral_anomalies(
                request, user_id, client_ip
            )
            threats.extend(behavior_threats)

        # Statistical anomaly detection
        statistical_threats = await self._detect_statistical_anomalies(
            request, client_ip
        )
        threats.extend(statistical_threats)

        # IP reputation analysis
        ip_threats = await self._analyze_ip_reputation(client_ip, request)
        threats.extend(ip_threats)

        # Store threat events
        for threat in threats:
            self.threat_events.append(threat)
            await self._handle_threat_event(threat)

        return threats

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        if request.client:
            return request.client.host

        return "unknown"

    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        try:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                # In production, decode JWT token
                return "authenticated_user"  # Placeholder
        except Exception:
            pass
        return None

    async def _detect_rule_based_threats(
        self,
        request: Request,
        client_ip: str,
        user_agent: str,
        request_path: str,
        request_method: str,
        user_id: Optional[str],
    ) -> List[ThreatEvent]:
        """Detect threats using rule-based patterns"""
        threats = []

        # Check for malicious user agents
        if any(
            malicious_ua.lower() in user_agent.lower()
            for malicious_ua in self.malicious_user_agents
        ):
            threats.append(
                self._create_threat_event(
                    ThreatType.BOT_ACTIVITY,
                    ThreatLevel.HIGH,
                    client_ip,
                    user_id,
                    request_path,
                    request_method,
                    user_agent,
                    8.0,
                    0.9,
                    {"detected_tool": user_agent},
                    ["Malicious user agent detected"],
                    ["Block IP address", "Investigate source"],
                )
            )

        # Check for attack patterns in URL
        for threat_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_path, re.IGNORECASE):
                    threats.append(
                        self._create_threat_event(
                            threat_type,
                            ThreatLevel.HIGH,
                            client_ip,
                            user_id,
                            request_path,
                            request_method,
                            user_agent,
                            7.5,
                            0.8,
                            {"matched_pattern": pattern},
                            [f"Attack pattern detected: {pattern}"],
                            ["Block request", "Log for analysis"],
                        )
                    )

        # Check for brute force attacks
        if request_path.endswith("/login") and request_method == "POST":
            brute_force_threat = await self._check_brute_force(
                client_ip, user_id, request_path, request_method, user_agent
            )
            if brute_force_threat:
                threats.append(brute_force_threat)

        # Check for suspicious request frequency
        frequency_threat = await self._check_request_frequency(
            client_ip, request_path, request_method, user_agent, user_id
        )
        if frequency_threat:
            threats.append(frequency_threat)

        return threats

    async def _check_brute_force(
        self,
        client_ip: str,
        user_id: Optional[str],
        request_path: str,
        request_method: str,
        user_agent: str,
    ) -> Optional[ThreatEvent]:
        """Check for brute force attacks"""
        key = f"{client_ip}:{user_id or 'anonymous'}"
        current_time = datetime.now(timezone.utc)

        # Clean old attempts
        self.failed_attempts[key] = [
            attempt
            for attempt in self.failed_attempts[key]
            if current_time - attempt < self.failed_attempt_window
        ]

        # Add current attempt (assuming it might fail)
        self.failed_attempts[key].append(current_time)

        # Check if threshold exceeded
        if len(self.failed_attempts[key]) > self.max_failed_attempts:
            return self._create_threat_event(
                ThreatType.BRUTE_FORCE,
                ThreatLevel.HIGH,
                client_ip,
                user_id,
                request_path,
                request_method,
                user_agent,
                8.5,
                0.9,
                {
                    "failed_attempts": len(self.failed_attempts[key]),
                    "time_window_minutes": self.failed_attempt_window.total_seconds()
                    / 60,
                },
                ["Multiple failed login attempts detected"],
                ["Temporarily block IP", "Require CAPTCHA", "Alert security team"],
            )

        return None

    async def _check_request_frequency(
        self,
        client_ip: str,
        request_path: str,
        request_method: str,
        user_agent: str,
        user_id: Optional[str],
    ) -> Optional[ThreatEvent]:
        """Check for suspicious request frequency (potential DDoS)"""
        current_time = time.time()
        key = client_ip

        # Add current request timestamp
        self.request_frequencies[key].append(current_time)

        # Calculate requests per minute
        minute_ago = current_time - 60
        recent_requests = [
            ts for ts in self.request_frequencies[key] if ts > minute_ago
        ]
        requests_per_minute = len(recent_requests)

        if requests_per_minute > self.suspicious_request_threshold:
            return self._create_threat_event(
                ThreatType.DDoS,
                ThreatLevel.HIGH,
                client_ip,
                user_id,
                request_path,
                request_method,
                user_agent,
                7.0,
                0.8,
                {
                    "requests_per_minute": requests_per_minute,
                    "threshold": self.suspicious_request_threshold,
                },
                ["Abnormally high request frequency detected"],
                ["Rate limit IP", "Investigate traffic source"],
            )

        return None

    async def _detect_behavioral_anomalies(
        self, request: Request, user_id: str, client_ip: str
    ) -> List[ThreatEvent]:
        """Detect behavioral anomalies for authenticated users"""
        threats = []

        # Get or create user behavior profile
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = await self._create_user_profile(user_id)
            self.user_profiles[user_id] = profile

        current_time = datetime.now(timezone.utc)
        current_hour = current_time.hour

        # Check for unusual access time
        if (
            profile.typical_access_times
            and current_hour not in profile.typical_access_times
        ):
            # Calculate how unusual this time is
            time_distances = [
                min(abs(current_hour - t), 24 - abs(current_hour - t))
                for t in profile.typical_access_times
            ]
            min_distance = min(time_distances)

            if min_distance > 4:  # More than 4 hours from typical times
                threats.append(
                    self._create_threat_event(
                        ThreatType.ANOMALOUS_BEHAVIOR,
                        ThreatLevel.MEDIUM,
                        client_ip,
                        user_id,
                        str(request.url.path),
                        request.method,
                        request.headers.get("user-agent", ""),
                        5.0,
                        0.6,
                        {
                            "current_hour": current_hour,
                            "typical_hours": list(profile.typical_access_times),
                            "time_deviation": min_distance,
                        },
                        ["Unusual access time detected"],
                        ["Monitor user activity", "Require additional authentication"],
                    )
                )

        # Check for unusual location
        if profile.typical_locations and client_ip not in profile.typical_locations:
            # In production, use geolocation to compare geographic locations
            threats.append(
                self._create_threat_event(
                    ThreatType.ANOMALOUS_BEHAVIOR,
                    ThreatLevel.MEDIUM,
                    client_ip,
                    user_id,
                    str(request.url.path),
                    request.method,
                    request.headers.get("user-agent", ""),
                    6.0,
                    0.7,
                    {
                        "new_ip": client_ip,
                        "typical_ips": list(profile.typical_locations),
                    },
                    ["Access from unusual location detected"],
                    ["Verify user identity", "Monitor session"],
                )
            )

        # Update user profile
        await self._update_user_profile(profile, request, client_ip)

        return threats

    async def _detect_statistical_anomalies(
        self, request: Request, client_ip: str
    ) -> List[ThreatEvent]:
        """Detect anomalies using statistical analysis"""
        threats = []

        if not self.anomaly_detector:
            return threats

        try:
            # Extract features for anomaly detection
            features = await self._extract_request_features(request)

            # Reshape for sklearn
            features_array = np.array(features).reshape(1, -1)

            # Scale features
            features_scaled = self.scaler.fit_transform(features_array)

            # Predict anomaly
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1

            if is_anomaly and abs(anomaly_score) > self.anomaly_threshold:
                threats.append(
                    self._create_threat_event(
                        ThreatType.ANOMALOUS_BEHAVIOR,
                        ThreatLevel.MEDIUM,
                        client_ip,
                        await self._extract_user_id(request),
                        str(request.url.path),
                        request.method,
                        request.headers.get("user-agent", ""),
                        abs(anomaly_score) * 10,  # Scale to 0-10
                        0.7,
                        {"anomaly_score": float(anomaly_score), "features": features},
                        ["Statistical anomaly detected in request pattern"],
                        ["Monitor request", "Analyze pattern"],
                    )
                )

        except Exception as e:
            logger.warning(f"Statistical anomaly detection failed: {e}")

        return threats

    async def _extract_request_features(self, request: Request) -> List[float]:
        """Extract numerical features from request for ML analysis"""
        features = []

        # URL length
        features.append(len(str(request.url)))

        # Number of query parameters
        features.append(len(request.query_params))

        # Number of headers
        features.append(len(request.headers))

        # User agent length
        user_agent = request.headers.get("user-agent", "")
        features.append(len(user_agent))

        # Request method (encoded)
        method_encoding = {"GET": 1, "POST": 2, "PUT": 3, "DELETE": 4, "PATCH": 5}
        features.append(method_encoding.get(request.method, 0))

        # Time of day (hour)
        features.append(datetime.now().hour)

        # Day of week
        features.append(datetime.now().weekday())

        return features

    async def _analyze_ip_reputation(
        self, client_ip: str, request: Request
    ) -> List[ThreatEvent]:
        """Analyze IP reputation and geolocation"""
        threats = []

        try:
            ip_addr = ipaddress.ip_address(client_ip)

            # Check if IP is in suspicious ranges
            for suspicious_range in self.suspicious_ip_ranges:
                if ip_addr in suspicious_range:
                    threats.append(
                        self._create_threat_event(
                            ThreatType.MALICIOUS_IP,
                            ThreatLevel.MEDIUM,
                            client_ip,
                            await self._extract_user_id(request),
                            str(request.url.path),
                            request.method,
                            request.headers.get("user-agent", ""),
                            6.0,
                            0.8,
                            {
                                "ip_range": str(suspicious_range),
                                "reason": "IP in suspicious range",
                            },
                            ["IP address in suspicious range"],
                            ["Monitor traffic", "Consider blocking"],
                        )
                    )

            # Check IP reputation cache
            ip_reputation = self.ip_reputation.get(client_ip)
            if ip_reputation and ip_reputation.get("malicious", False):
                threats.append(
                    self._create_threat_event(
                        ThreatType.MALICIOUS_IP,
                        ThreatLevel.HIGH,
                        client_ip,
                        await self._extract_user_id(request),
                        str(request.url.path),
                        request.method,
                        request.headers.get("user-agent", ""),
                        8.0,
                        0.9,
                        ip_reputation,
                        ["Known malicious IP detected"],
                        ["Block IP immediately", "Alert security team"],
                    )
                )

        except ValueError:
            # Invalid IP address
            threats.append(
                self._create_threat_event(
                    ThreatType.SUSPICIOUS_PATTERN,
                    ThreatLevel.LOW,
                    client_ip,
                    await self._extract_user_id(request),
                    str(request.url.path),
                    request.method,
                    request.headers.get("user-agent", ""),
                    3.0,
                    0.5,
                    {"invalid_ip": client_ip},
                    ["Invalid IP address format"],
                    ["Log for analysis"],
                )
            )

        return threats

    def _create_threat_event(
        self,
        threat_type: ThreatType,
        threat_level: ThreatLevel,
        source_ip: str,
        user_id: Optional[str],
        request_path: str,
        request_method: str,
        user_agent: str,
        risk_score: float,
        confidence: float,
        details: Dict[str, Any],
        indicators: List[str],
        recommended_actions: List[str],
    ) -> ThreatEvent:
        """Create a threat event"""
        event_id = self.encryption_service.generate_secure_token(16)

        # Calculate false positive probability based on confidence and threat type
        false_positive_prob = self._calculate_false_positive_probability(
            threat_type, confidence, risk_score
        )

        return ThreatEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            threat_type=threat_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            request_path=request_path,
            request_method=request_method,
            user_agent=user_agent,
            risk_score=risk_score,
            confidence=confidence,
            details=details,
            indicators=indicators,
            recommended_actions=recommended_actions,
            false_positive_probability=false_positive_prob,
        )

    def _calculate_false_positive_probability(
        self, threat_type: ThreatType, confidence: float, risk_score: float
    ) -> float:
        """Calculate probability that this is a false positive"""
        # Base false positive rates by threat type
        base_rates = {
            ThreatType.BRUTE_FORCE: 0.05,
            ThreatType.SQL_INJECTION: 0.10,
            ThreatType.XSS_ATTACK: 0.15,
            ThreatType.DDoS: 0.20,
            ThreatType.MALICIOUS_IP: 0.10,
            ThreatType.BOT_ACTIVITY: 0.25,
            ThreatType.ANOMALOUS_BEHAVIOR: 0.40,
            ThreatType.SUSPICIOUS_PATTERN: 0.30,
        }

        base_rate = base_rates.get(threat_type, 0.25)

        # Adjust based on confidence and risk score
        adjusted_rate = base_rate * (1 - confidence) * (10 - risk_score) / 10

        return max(0.01, min(0.99, adjusted_rate))

    async def _create_user_profile(self, user_id: str) -> BehaviorProfile:
        """Create initial user behavior profile"""
        return BehaviorProfile(
            user_id=user_id,
            typical_access_times=[],
            typical_locations=set(),
            typical_endpoints=set(),
            typical_request_frequency=0.0,
            typical_session_duration=0.0,
            risk_score_history=[],
            last_updated=datetime.now(timezone.utc),
        )

    async def _update_user_profile(
        self, profile: BehaviorProfile, request: Request, client_ip: str
    ):
        """Update user behavior profile with new data"""
        current_time = datetime.now(timezone.utc)
        current_hour = current_time.hour

        # Update typical access times
        if current_hour not in profile.typical_access_times:
            profile.typical_access_times.append(current_hour)
            # Keep only recent patterns (last 30 days worth)
            if len(profile.typical_access_times) > 24:
                profile.typical_access_times = profile.typical_access_times[-24:]

        # Update typical locations
        profile.typical_locations.add(client_ip)
        # Keep only recent locations
        if len(profile.typical_locations) > 10:
            # In production, implement LRU cache
            pass

        # Update typical endpoints
        profile.typical_endpoints.add(str(request.url.path))
        if len(profile.typical_endpoints) > 50:
            # Keep most frequent endpoints
            pass

        profile.last_updated = current_time

    async def _handle_threat_event(self, threat: ThreatEvent):
        """Handle detected threat event"""
        # Log threat event
        logger.warning(
            f"Threat detected: {threat.threat_type.value} "
            f"(Level: {threat.threat_level.value}, Score: {threat.risk_score}) "
            f"from {threat.source_ip}"
        )

        # Take automated actions based on threat level
        if threat.threat_level == ThreatLevel.CRITICAL:
            await self._handle_critical_threat(threat)
        elif threat.threat_level == ThreatLevel.HIGH:
            await self._handle_high_threat(threat)
        elif threat.threat_level == ThreatLevel.MEDIUM:
            await self._handle_medium_threat(threat)

        # Store in persistent storage (in production)
        await self._store_threat_event(threat)

        # Send alerts if necessary
        await self._send_threat_alert(threat)

    async def _handle_critical_threat(self, threat: ThreatEvent):
        """Handle critical threat events"""
        # Immediate blocking
        await self._block_ip_address(threat.source_ip, "Critical threat detected")

        # Alert security team immediately
        await self._send_immediate_alert(threat)

        # Log to security incident system
        await self._create_security_incident(threat)

    async def _handle_high_threat(self, threat: ThreatEvent):
        """Handle high-level threat events"""
        # Temporary rate limiting
        await self._apply_rate_limiting(threat.source_ip)

        # Enhanced monitoring
        await self._enable_enhanced_monitoring(threat.source_ip)

        # Alert security team
        await self._send_threat_alert(threat)

    async def _handle_medium_threat(self, threat: ThreatEvent):
        """Handle medium-level threat events"""
        # Log for analysis
        await self._log_for_analysis(threat)

        # Increase monitoring
        await self._increase_monitoring(threat.source_ip)

    async def _block_ip_address(self, ip_address: str, reason: str):
        """Block IP address"""
        # In production, this would update firewall rules or WAF
        logger.info(f"Blocking IP {ip_address}: {reason}")

    async def _apply_rate_limiting(self, ip_address: str):
        """Apply rate limiting to IP address"""
        # In production, this would update rate limiting rules
        logger.info(f"Applying rate limiting to IP {ip_address}")

    async def _enable_enhanced_monitoring(self, ip_address: str):
        """Enable enhanced monitoring for IP address"""
        # In production, this would configure monitoring systems
        logger.info(f"Enabling enhanced monitoring for IP {ip_address}")

    async def _increase_monitoring(self, ip_address: str):
        """Increase monitoring level for IP address"""
        logger.info(f"Increasing monitoring for IP {ip_address}")

    async def _log_for_analysis(self, threat: ThreatEvent):
        """Log threat for further analysis"""
        logger.info(f"Logging threat {threat.event_id} for analysis")

    async def _store_threat_event(self, threat: ThreatEvent):
        """Store threat event in persistent storage"""
        # In production, store in database

    async def _send_threat_alert(self, threat: ThreatEvent):
        """Send threat alert to security team"""
        if threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # In production, send to SIEM, email, Slack, etc.
            logger.warning(f"SECURITY ALERT: {asdict(threat)}")

    async def _send_immediate_alert(self, threat: ThreatEvent):
        """Send immediate alert for critical threats"""
        # In production, send SMS, phone call, or other immediate notification
        logger.critical(f"IMMEDIATE SECURITY ALERT: {asdict(threat)}")

    async def _create_security_incident(self, threat: ThreatEvent):
        """Create security incident for critical threats"""
        # In production, create incident in incident management system
        logger.critical(f"Creating security incident for threat {threat.event_id}")

    # Public API methods

    async def log_threat(self, request: Request, threat_type: str, details: str):
        """Log a custom threat event"""
        client_ip = self._get_client_ip(request)
        user_id = await self._extract_user_id(request)

        threat = self._create_threat_event(
            ThreatType(threat_type),
            ThreatLevel.MEDIUM,
            client_ip,
            user_id,
            str(request.url.path),
            request.method,
            request.headers.get("user-agent", ""),
            5.0,
            0.7,
            {"custom_details": details},
            ["Custom threat logged"],
            ["Investigate"],
        )

        await self._handle_threat_event(threat)

    async def check_request_frequency(self, client_ip: str, request: Request):
        """Check request frequency for potential DDoS"""
        threat = await self._check_request_frequency(
            client_ip,
            str(request.url.path),
            request.method,
            request.headers.get("user-agent", ""),
            await self._extract_user_id(request),
        )

        if threat:
            await self._handle_threat_event(threat)

    async def log_ip_block(self, ip_address: str, reason: str):
        """Log IP block event"""
        logger.info(f"IP {ip_address} blocked: {reason}")
        # In production, store in database

    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        if not self.threat_events:
            return {
                "total_threats": 0,
                "threats_by_type": {},
                "threats_by_level": {},
                "top_source_ips": [],
            }

        threats_by_type = defaultdict(int)
        threats_by_level = defaultdict(int)
        source_ips = defaultdict(int)

        for threat in self.threat_events:
            threats_by_type[threat.threat_type.value] += 1
            threats_by_level[threat.threat_level.value] += 1
            source_ips[threat.source_ip] += 1

        return {
            "total_threats": len(self.threat_events),
            "threats_by_type": dict(threats_by_type),
            "threats_by_level": dict(threats_by_level),
            "top_source_ips": sorted(
                source_ips.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    async def get_user_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user risk profile"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {"user_id": user_id, "profile_exists": False}

        return {
            "user_id": user_id,
            "profile_exists": True,
            "typical_access_times": profile.typical_access_times,
            "typical_locations_count": len(profile.typical_locations),
            "typical_endpoints_count": len(profile.typical_endpoints),
            "average_risk_score": (
                mean(profile.risk_score_history) if profile.risk_score_history else 0
            ),
            "last_updated": profile.last_updated.isoformat(),
        }
