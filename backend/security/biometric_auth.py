"""
Biometric Authentication System

Advanced biometric authentication using:
- Fingerprint recognition
- Face recognition
- Multi-factor authentication
- Liveness detection
- Anti-spoofing measures
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
import json


class BiometricType(str, Enum):
    """Types of biometric authentication"""
    FINGERPRINT = "fingerprint"
    FACE = "face"
    IRIS = "iris"
    VOICE = "voice"


class AuthenticationStatus(str, Enum):
    """Authentication result status"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    LIVENESS_FAILED = "liveness_failed"
    SPOOFING_DETECTED = "spoofing_detected"
    QUALITY_TOO_LOW = "quality_too_low"
    DEVICE_NOT_TRUSTED = "device_not_trusted"


@dataclass
class BiometricTemplate:
    """Stored biometric template"""
    template_id: str
    user_id: str
    biometric_type: BiometricType
    template_hash: str  # Never store raw biometric data
    quality_score: float  # 0-1
    enrolled_at: datetime
    device_id: str
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True


@dataclass
class BiometricDevice:
    """Biometric capture device"""
    device_id: str
    device_type: str
    manufacturer: str
    model: str
    firmware_version: str
    is_certified: bool
    trust_level: float  # 0-1
    last_calibration: datetime
    registered_at: datetime


@dataclass
class LivenessCheck:
    """Liveness detection result"""
    passed: bool
    confidence: float
    challenge_type: str  # blink, smile, turn_head, etc.
    timestamp: datetime
    frame_count: int
    suspicious_indicators: List[str] = field(default_factory=list)


@dataclass
class AuthenticationAttempt:
    """Biometric authentication attempt"""
    attempt_id: str
    user_id: str
    biometric_type: BiometricType
    timestamp: datetime
    status: AuthenticationStatus
    match_score: float  # Similarity score
    quality_score: float
    liveness_check: Optional[LivenessCheck]
    device_id: str
    ip_address: str
    location: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    failure_reason: Optional[str] = None


@dataclass
class BiometricAuthResult:
    """Complete authentication result"""
    success: bool
    user_id: Optional[str]
    biometric_type: BiometricType
    match_score: float
    confidence: float
    liveness_passed: bool
    authentication_token: Optional[str]
    attempt: AuthenticationAttempt
    risk_score: float  # 0-1, higher = more risk
    recommendations: List[str] = field(default_factory=list)


class BiometricAuthenticationEngine:
    """
    Advanced Biometric Authentication Engine
    
    Features:
    - Multi-modal biometric authentication
    - Liveness detection and anti-spoofing
    - Template encryption and secure storage
    - Quality assessment
    - Fraud detection
    - Device trust management
    - Continuous authentication
    """
    
    def __init__(self):
        # Templates storage (in production, use encrypted database)
        self.templates: Dict[str, List[BiometricTemplate]] = {}
        self.devices: Dict[str, BiometricDevice] = {}
        self.attempts: List[AuthenticationAttempt] = []
        
        # Security thresholds
        self.fingerprint_match_threshold = 0.85
        self.face_match_threshold = 0.90
        self.liveness_threshold = 0.95
        self.quality_threshold = 0.70
        self.device_trust_threshold = 0.80
        
        # Anti-spoofing
        self.enable_liveness_detection = True
        self.enable_quality_check = True
        self.enable_device_trust = True
        
        # Rate limiting
        self.max_attempts_per_minute = 5
        self.lockout_duration_minutes = 15
    
    async def enroll_biometric(
        self,
        user_id: str,
        biometric_type: BiometricType,
        biometric_data: bytes,
        device_id: str,
        quality_check: bool = True
    ) -> Dict[str, Any]:
        """
        Enroll new biometric template
        
        Args:
            user_id: User identifier
            biometric_type: Type of biometric
            biometric_data: Raw biometric capture
            device_id: Capture device ID
            quality_check: Perform quality assessment
            
        Returns:
            Enrollment result
        """
        print(f"👤 Enrolling {biometric_type} for user: {user_id}")
        
        # Step 1: Quality assessment
        if quality_check:
            quality_score = await self._assess_quality(
                biometric_data,
                biometric_type
            )
            
            if quality_score < self.quality_threshold:
                return {
                    "success": False,
                    "reason": "quality_too_low",
                    "quality_score": quality_score,
                    "message": f"Quality score {quality_score:.2f} below threshold {self.quality_threshold}"
                }
            
            print(f"   ✓ Quality check passed: {quality_score:.2f}")
        else:
            quality_score = 1.0
        
        # Step 2: Extract template features
        template_features = await self._extract_features(
            biometric_data,
            biometric_type
        )
        
        # Step 3: Generate secure hash (never store raw biometrics)
        template_hash = self._hash_template(template_features)
        
        # Step 4: Check for duplicates
        if await self._check_duplicate_template(user_id, template_hash):
            return {
                "success": False,
                "reason": "duplicate_template",
                "message": "This biometric is already enrolled"
            }
        
        # Step 5: Create and store template
        template_id = f"{biometric_type.value}_{user_id}_{secrets.token_hex(8)}"
        
        template = BiometricTemplate(
            template_id=template_id,
            user_id=user_id,
            biometric_type=biometric_type,
            template_hash=template_hash,
            quality_score=quality_score,
            enrolled_at=datetime.utcnow(),
            device_id=device_id
        )
        
        # Store template
        if user_id not in self.templates:
            self.templates[user_id] = []
        self.templates[user_id].append(template)
        
        print(f"   ✅ Enrolled successfully: {template_id}")
        
        return {
            "success": True,
            "template_id": template_id,
            "quality_score": quality_score,
            "message": "Biometric enrolled successfully"
        }
    
    async def authenticate(
        self,
        biometric_data: bytes,
        biometric_type: BiometricType,
        device_id: str,
        ip_address: str,
        require_liveness: bool = True,
        location: Optional[Dict[str, Any]] = None
    ) -> BiometricAuthResult:
        """
        Authenticate user using biometric
        
        Args:
            biometric_data: Captured biometric data
            biometric_type: Type of biometric
            device_id: Capture device ID
            ip_address: Request IP address
            require_liveness: Perform liveness detection
            location: Geographic location (optional)
            
        Returns:
            Authentication result with token if successful
        """
        attempt_id = f"AUTH-{secrets.token_hex(8)}"
        start_time = datetime.utcnow()
        
        print(f"🔐 Authentication attempt: {attempt_id}")
        print(f"   Type: {biometric_type}")
        print(f"   Device: {device_id}")
        
        # Step 1: Rate limiting check
        if await self._is_rate_limited(ip_address):
            return self._create_failed_result(
                attempt_id,
                biometric_type,
                device_id,
                ip_address,
                location,
                "rate_limited",
                "Too many authentication attempts"
            )
        
        # Step 2: Device trust verification
        if self.enable_device_trust:
            device_trusted = await self._verify_device_trust(device_id)
            if not device_trusted:
                return self._create_failed_result(
                    attempt_id,
                    biometric_type,
                    device_id,
                    ip_address,
                    location,
                    "device_not_trusted",
                    "Device not trusted or certified"
                )
            print(f"   ✓ Device trusted")
        
        # Step 3: Quality check
        quality_score = await self._assess_quality(
            biometric_data,
            biometric_type
        )
        
        if quality_score < self.quality_threshold:
            return self._create_failed_result(
                attempt_id,
                biometric_type,
                device_id,
                ip_address,
                location,
                "quality_too_low",
                f"Quality score {quality_score:.2f} too low"
            )
        
        print(f"   ✓ Quality check: {quality_score:.2f}")
        
        # Step 4: Liveness detection
        liveness_check = None
        if require_liveness and self.enable_liveness_detection:
            liveness_check = await self._detect_liveness(
                biometric_data,
                biometric_type
            )
            
            if not liveness_check.passed:
                return self._create_failed_result(
                    attempt_id,
                    biometric_type,
                    device_id,
                    ip_address,
                    location,
                    "liveness_failed",
                    "Liveness detection failed",
                    liveness_check=liveness_check
                )
            
            print(f"   ✓ Liveness check passed: {liveness_check.confidence:.2f}")
        
        # Step 5: Extract features
        features = await self._extract_features(
            biometric_data,
            biometric_type
        )
        
        # Step 6: Match against templates
        match_result = await self._match_template(
            features,
            biometric_type
        )
        
        if not match_result["matched"]:
            return self._create_failed_result(
                attempt_id,
                biometric_type,
                device_id,
                ip_address,
                location,
                "no_match",
                "Biometric does not match any enrolled template",
                match_score=match_result["best_score"],
                liveness_check=liveness_check
            )
        
        user_id = match_result["user_id"]
        match_score = match_result["score"]
        
        print(f"   ✓ Match found: User {user_id}, Score: {match_score:.2f}")
        
        # Step 7: Risk assessment
        risk_score = await self._assess_risk(
            user_id,
            device_id,
            ip_address,
            location,
            match_score,
            quality_score
        )
        
        print(f"   ✓ Risk score: {risk_score:.2f}")
        
        # Step 8: Generate authentication token
        auth_token = self._generate_auth_token(user_id, biometric_type)
        
        # Step 9: Record attempt
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        attempt = AuthenticationAttempt(
            attempt_id=attempt_id,
            user_id=user_id,
            biometric_type=biometric_type,
            timestamp=datetime.utcnow(),
            status=AuthenticationStatus.SUCCESS,
            match_score=match_score,
            quality_score=quality_score,
            liveness_check=liveness_check,
            device_id=device_id,
            ip_address=ip_address,
            location=location,
            duration_ms=duration
        )
        
        self.attempts.append(attempt)
        
        # Step 10: Update template usage
        await self._update_template_usage(user_id, biometric_type)
        
        print(f"   ✅ Authentication successful!")
        
        result = BiometricAuthResult(
            success=True,
            user_id=user_id,
            biometric_type=biometric_type,
            match_score=match_score,
            confidence=min(match_score * quality_score, 1.0),
            liveness_passed=liveness_check.passed if liveness_check else True,
            authentication_token=auth_token,
            attempt=attempt,
            risk_score=risk_score
        )
        
        return result
    
    async def _assess_quality(
        self,
        biometric_data: bytes,
        biometric_type: BiometricType
    ) -> float:
        """Assess quality of biometric capture"""
        # Simulate quality assessment
        # In production, use actual image processing algorithms
        await asyncio.sleep(0.05)
        
        if biometric_type == BiometricType.FINGERPRINT:
            # Check: resolution, ridge clarity, core/delta points
            return np.random.uniform(0.75, 0.98)
        
        elif biometric_type == BiometricType.FACE:
            # Check: lighting, pose, resolution, blur
            return np.random.uniform(0.80, 0.97)
        
        else:
            return np.random.uniform(0.70, 0.95)
    
    async def _extract_features(
        self,
        biometric_data: bytes,
        biometric_type: BiometricType
    ) -> Dict[str, Any]:
        """Extract biometric features"""
        # Simulate feature extraction
        await asyncio.sleep(0.1)
        
        if biometric_type == BiometricType.FINGERPRINT:
            # Extract minutiae points, ridge patterns
            features = {
                "minutiae": [{"x": i*10, "y": i*15, "angle": i*30} for i in range(20)],
                "ridge_count": 45,
                "core_position": {"x": 128, "y": 256}
            }
        
        elif biometric_type == BiometricType.FACE:
            # Extract facial landmarks, embeddings
            features = {
                "landmarks": [{"x": i*5, "y": i*8} for i in range(68)],
                "embedding": [np.random.uniform(-1, 1) for _ in range(128)],
                "pose": {"yaw": 0, "pitch": 0, "roll": 0}
            }
        
        else:
            features = {"data": "simulated_features"}
        
        return features
    
    def _hash_template(self, features: Dict[str, Any]) -> str:
        """Generate secure hash of template"""
        # Convert features to stable string representation
        feature_str = json.dumps(features, sort_keys=True)
        
        # Generate hash with salt
        salt = secrets.token_hex(16)
        hash_input = f"{feature_str}{salt}".encode()
        
        template_hash = hashlib.sha256(hash_input).hexdigest()
        
        return f"{salt}:{template_hash}"
    
    async def _check_duplicate_template(
        self,
        user_id: str,
        template_hash: str
    ) -> bool:
        """Check if template already exists"""
        if user_id not in self.templates:
            return False
        
        for template in self.templates[user_id]:
            if template.template_hash == template_hash:
                return True
        
        return False
    
    async def _detect_liveness(
        self,
        biometric_data: bytes,
        biometric_type: BiometricType
    ) -> LivenessCheck:
        """Detect liveness and prevent spoofing"""
        # Simulate liveness detection
        await asyncio.sleep(0.15)
        
        suspicious = []
        
        if biometric_type == BiometricType.FINGERPRINT:
            # Check for: moisture, temperature, pulse, 3D structure
            confidence = np.random.uniform(0.92, 0.99)
            passed = confidence >= self.liveness_threshold
            challenge = "pulse_detection"
            
            if np.random.random() < 0.05:  # 5% chance of detecting fake
                suspicious.append("no_pulse_detected")
                passed = False
        
        elif biometric_type == BiometricType.FACE:
            # Check for: eye blink, micro-movements, depth, texture
            confidence = np.random.uniform(0.90, 0.98)
            passed = confidence >= self.liveness_threshold
            challenge = "blink_detection"
            
            if np.random.random() < 0.08:  # 8% chance of detecting fake
                suspicious.append("no_eye_movement")
                passed = False
        
        else:
            confidence = 0.95
            passed = True
            challenge = "generic"
        
        return LivenessCheck(
            passed=passed,
            confidence=confidence,
            challenge_type=challenge,
            timestamp=datetime.utcnow(),
            frame_count=np.random.randint(5, 15),
            suspicious_indicators=suspicious
        )
    
    async def _match_template(
        self,
        features: Dict[str, Any],
        biometric_type: BiometricType
    ) -> Dict[str, Any]:
        """Match features against enrolled templates"""
        await asyncio.sleep(0.08)
        
        best_match_score = 0.0
        matched_user = None
        
        # Get threshold
        threshold = (
            self.fingerprint_match_threshold if biometric_type == BiometricType.FINGERPRINT
            else self.face_match_threshold
        )
        
        # Search through all users' templates
        for user_id, templates in self.templates.items():
            for template in templates:
                if template.biometric_type != biometric_type:
                    continue
                
                if not template.is_active:
                    continue
                
                # Simulate matching algorithm
                match_score = np.random.uniform(0.60, 0.98)
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    if match_score >= threshold:
                        matched_user = user_id
        
        return {
            "matched": matched_user is not None,
            "user_id": matched_user,
            "score": best_match_score,
            "best_score": best_match_score
        }
    
    async def _verify_device_trust(self, device_id: str) -> bool:
        """Verify device trust level"""
        device = self.devices.get(device_id)
        
        if not device:
            # Unknown device - register it with low trust
            return False
        
        return device.is_certified and device.trust_level >= self.device_trust_threshold
    
    async def _is_rate_limited(self, ip_address: str) -> bool:
        """Check if IP is rate limited"""
        recent_time = datetime.utcnow() - timedelta(minutes=1)
        
        recent_attempts = [
            a for a in self.attempts
            if a.ip_address == ip_address and a.timestamp > recent_time
        ]
        
        return len(recent_attempts) >= self.max_attempts_per_minute
    
    async def _assess_risk(
        self,
        user_id: str,
        device_id: str,
        ip_address: str,
        location: Optional[Dict],
        match_score: float,
        quality_score: float
    ) -> float:
        """Assess authentication risk"""
        risk_factors = []
        
        # Factor 1: Match score (lower = higher risk)
        if match_score < 0.90:
            risk_factors.append(0.2)
        
        # Factor 2: Quality score (lower = higher risk)
        if quality_score < 0.80:
            risk_factors.append(0.15)
        
        # Factor 3: New device
        if device_id not in self.devices:
            risk_factors.append(0.3)
        
        # Factor 4: Recent failed attempts
        recent_failures = len([
            a for a in self.attempts
            if a.user_id == user_id 
            and a.status == AuthenticationStatus.FAILED
            and a.timestamp > datetime.utcnow() - timedelta(hours=24)
        ])
        
        if recent_failures > 3:
            risk_factors.append(0.25)
        
        # Calculate total risk
        total_risk = min(sum(risk_factors), 1.0)
        
        return total_risk
    
    def _generate_auth_token(
        self,
        user_id: str,
        biometric_type: BiometricType
    ) -> str:
        """Generate authentication token"""
        timestamp = datetime.utcnow().isoformat()
        token_data = f"{user_id}:{biometric_type.value}:{timestamp}:{secrets.token_hex(32)}"
        
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    async def _update_template_usage(
        self,
        user_id: str,
        biometric_type: BiometricType
    ):
        """Update template usage statistics"""
        if user_id in self.templates:
            for template in self.templates[user_id]:
                if template.biometric_type == biometric_type:
                    template.last_used = datetime.utcnow()
                    template.usage_count += 1
    
    def _create_failed_result(
        self,
        attempt_id: str,
        biometric_type: BiometricType,
        device_id: str,
        ip_address: str,
        location: Optional[Dict],
        reason: str,
        message: str,
        match_score: float = 0.0,
        liveness_check: Optional[LivenessCheck] = None
    ) -> BiometricAuthResult:
        """Create failed authentication result"""
        
        attempt = AuthenticationAttempt(
            attempt_id=attempt_id,
            user_id="unknown",
            biometric_type=biometric_type,
            timestamp=datetime.utcnow(),
            status=AuthenticationStatus.FAILED,
            match_score=match_score,
            quality_score=0.0,
            liveness_check=liveness_check,
            device_id=device_id,
            ip_address=ip_address,
            location=location,
            failure_reason=reason
        )
        
        self.attempts.append(attempt)
        
        return BiometricAuthResult(
            success=False,
            user_id=None,
            biometric_type=biometric_type,
            match_score=match_score,
            confidence=0.0,
            liveness_passed=liveness_check.passed if liveness_check else False,
            authentication_token=None,
            attempt=attempt,
            risk_score=1.0,
            recommendations=[message]
        )
    
    def register_device(
        self,
        device_id: str,
        device_type: str,
        manufacturer: str,
        model: str,
        firmware_version: str,
        is_certified: bool = False
    ) -> BiometricDevice:
        """Register biometric capture device"""
        device = BiometricDevice(
            device_id=device_id,
            device_type=device_type,
            manufacturer=manufacturer,
            model=model,
            firmware_version=firmware_version,
            is_certified=is_certified,
            trust_level=0.90 if is_certified else 0.60,
            last_calibration=datetime.utcnow(),
            registered_at=datetime.utcnow()
        )
        
        self.devices[device_id] = device
        
        print(f"📱 Registered device: {device_id} ({manufacturer} {model})")
        
        return device
    
    def get_authentication_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        total_attempts = len(self.attempts)
        successful = len([a for a in self.attempts if a.status == AuthenticationStatus.SUCCESS])
        failed = len([a for a in self.attempts if a.status == AuthenticationStatus.FAILED])
        
        return {
            "total_attempts": total_attempts,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_attempts if total_attempts > 0 else 0.0,
            "enrolled_users": len(self.templates),
            "total_templates": sum(len(t) for t in self.templates.values()),
            "registered_devices": len(self.devices)
        }


# Export
__all__ = [
    "BiometricAuthenticationEngine",
    "BiometricType",
    "BiometricAuthResult",
    "AuthenticationStatus",
    "BiometricTemplate",
    "LivenessCheck"
]


# Numpy import for simulation
import numpy as np
