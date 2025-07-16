"""
ML-Based Threat Detection System for Advanced Security.

This module implements machine learning-based threat detection including:
- Anomaly detection using isolation forests
- Behavioral analysis with sequence prediction
- Feature extraction for request patterns
- Real-time scoring with sub-50ms latency
- Integration with existing security monitoring
"""

import json
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from auth.security_logging import SecurityEventSeverity, SecurityEventType, security_auditor
from observability.security_monitoring import AttackType, SecurityMonitoringSystem, ThreatLevel

logger = logging.getLogger(__name__)


@dataclass
class ThreatFeatures:
    """Feature vector for threat detection."""

    # Request patterns
    request_frequency: float = 0.0
    request_size: float = 0.0
    endpoint_diversity: float = 0.0
    method_diversity: float = 0.0

    # Timing patterns
    time_since_last_request: float = 0.0
    requests_per_minute: float = 0.0
    requests_per_hour: float = 0.0

    # Geographic patterns
    country_risk_score: float = 0.0
    distance_from_usual_location: float = 0.0

    # User agent patterns
    user_agent_entropy: float = 0.0
    is_suspicious_user_agent: float = 0.0

    # Authentication patterns
    failed_login_rate: float = 0.0
    successful_login_rate: float = 0.0
    mfa_bypass_attempts: float = 0.0

    # API usage patterns
    error_rate: float = 0.0
    privileged_endpoint_access: float = 0.0
    data_access_volume: float = 0.0

    # Behavioral patterns
    deviation_from_baseline: float = 0.0
    session_duration: float = 0.0
    concurrent_sessions: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array."""
        return np.array(
            [
                self.request_frequency,
                self.request_size,
                self.endpoint_diversity,
                self.method_diversity,
                self.time_since_last_request,
                self.requests_per_minute,
                self.requests_per_hour,
                self.country_risk_score,
                self.distance_from_usual_location,
                self.user_agent_entropy,
                self.is_suspicious_user_agent,
                self.failed_login_rate,
                self.successful_login_rate,
                self.mfa_bypass_attempts,
                self.error_rate,
                self.privileged_endpoint_access,
                self.data_access_volume,
                self.deviation_from_baseline,
                self.session_duration,
                self.concurrent_sessions,
            ]
        )


@dataclass
class ThreatPrediction:
    """Threat prediction result."""

    risk_score: float
    threat_level: ThreatLevel
    confidence: float
    detected_attacks: List[AttackType]
    features_contribution: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserBehaviorBaseline:
    """User behavior baseline for anomaly detection."""

    user_id: str
    typical_request_patterns: Dict[str, float]
    typical_timing_patterns: Dict[str, float]
    typical_locations: List[Dict[str, Any]]
    typical_user_agents: List[str]
    last_updated: datetime = field(default_factory=datetime.utcnow)


class FeatureExtractor:
    """Extract features from request data for ML threat detection."""

    def __init__(self):
        self.user_request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ip_request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.user_baselines: Dict[str, UserBehaviorBaseline] = {}
        self.suspicious_user_agents = {
            "curl/",
            "wget/",
            "python-requests/",
            "sqlmap/",
            "nikto/",
            "nmap/",
            "burp/",
            "zap/",
            "metasploit/",
            "nessus/",
            "openvas/",
            "acunetix/",
        }
        self.country_risk_scores = {
            "US": 0.1,
            "CA": 0.1,
            "GB": 0.1,
            "DE": 0.1,
            "FR": 0.1,
            "AU": 0.1,
            "CN": 0.7,
            "RU": 0.8,
            "IR": 0.9,
            "KP": 0.9,
            "TOR": 1.0,
        }

    def extract_features(self, request_data: Dict[str, Any]) -> ThreatFeatures:
        """Extract threat detection features from request data."""
        try:
            user_id = request_data.get("user_id")
            ip_address = request_data.get("ip_address")
            timestamp = datetime.fromisoformat(
                request_data.get("timestamp", datetime.utcnow().isoformat())
            )

            # Store request in history
            if user_id:
                self.user_request_history[user_id].append(request_data)
            if ip_address:
                self.ip_request_history[ip_address].append(request_data)

            # Extract features
            features = ThreatFeatures()

            # Request pattern features
            features.request_frequency = self._calculate_request_frequency(user_id, ip_address)
            features.request_size = len(json.dumps(request_data))
            features.endpoint_diversity = self._calculate_endpoint_diversity(user_id, ip_address)
            features.method_diversity = self._calculate_method_diversity(user_id, ip_address)

            # Timing features
            features.time_since_last_request = self._calculate_time_since_last_request(
                user_id, ip_address
            )
            features.requests_per_minute = self._calculate_requests_per_minute(user_id, ip_address)
            features.requests_per_hour = self._calculate_requests_per_hour(user_id, ip_address)

            # Geographic features
            country = request_data.get("country", "US")
            features.country_risk_score = self.country_risk_scores.get(country, 0.5)
            features.distance_from_usual_location = self._calculate_location_distance(
                user_id, request_data
            )

            # User agent features
            user_agent = request_data.get("user_agent", "")
            features.user_agent_entropy = self._calculate_entropy(user_agent)
            features.is_suspicious_user_agent = self._is_suspicious_user_agent(user_agent)

            # Authentication features
            features.failed_login_rate = self._calculate_failed_login_rate(user_id, ip_address)
            features.successful_login_rate = self._calculate_successful_login_rate(
                user_id, ip_address
            )
            features.mfa_bypass_attempts = self._calculate_mfa_bypass_attempts(user_id, ip_address)

            # API usage features
            features.error_rate = self._calculate_error_rate(user_id, ip_address)
            features.privileged_endpoint_access = self._calculate_privileged_access(
                user_id, ip_address
            )
            features.data_access_volume = self._calculate_data_access_volume(user_id, ip_address)

            # Behavioral features
            features.deviation_from_baseline = self._calculate_baseline_deviation(
                user_id, request_data
            )
            features.session_duration = self._calculate_session_duration(user_id, request_data)
            features.concurrent_sessions = self._calculate_concurrent_sessions(user_id)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return ThreatFeatures()  # Return empty features on error

    def _calculate_request_frequency(self, user_id: str, ip_address: str) -> float:
        """Calculate request frequency for user/IP."""
        now = datetime.utcnow()
        recent_requests = 0

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(minutes=1):
                    recent_requests += 1

        return min(recent_requests / 60.0, 1.0)  # Normalize to 0-1

    def _calculate_endpoint_diversity(self, user_id: str, ip_address: str) -> float:
        """Calculate endpoint diversity (Shannon entropy)."""
        endpoints = []
        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            endpoints.extend([req.get("endpoint", "") for req in requests])

        if not endpoints:
            return 0.0

        return self._calculate_entropy("".join(endpoints))

    def _calculate_method_diversity(self, user_id: str, ip_address: str) -> float:
        """Calculate HTTP method diversity."""
        methods = []
        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            methods.extend([req.get("method", "GET") for req in requests])

        if not methods:
            return 0.0

        unique_methods = len(set(methods))
        return min(unique_methods / 10.0, 1.0)  # Normalize

    def _calculate_time_since_last_request(self, user_id: str, ip_address: str) -> float:
        """Calculate time since last request."""
        now = datetime.utcnow()
        last_request_time = None

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            if requests:
                req_time = datetime.fromisoformat(requests[-1].get("timestamp", now.isoformat()))
                if not last_request_time or req_time > last_request_time:
                    last_request_time = req_time

        if not last_request_time:
            return 1.0

        delta = (now - last_request_time).total_seconds()
        return min(delta / 3600.0, 1.0)  # Normalize to hours

    def _calculate_requests_per_minute(self, user_id: str, ip_address: str) -> float:
        """Calculate requests per minute."""
        now = datetime.utcnow()
        recent_requests = 0

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(minutes=1):
                    recent_requests += 1

        return min(recent_requests / 60.0, 1.0)

    def _calculate_requests_per_hour(self, user_id: str, ip_address: str) -> float:
        """Calculate requests per hour."""
        now = datetime.utcnow()
        recent_requests = 0

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(hours=1):
                    recent_requests += 1

        return min(recent_requests / 3600.0, 1.0)

    def _calculate_location_distance(self, user_id: str, request_data: Dict[str, Any]) -> float:
        """Calculate distance from user's usual location."""
        if not user_id or user_id not in self.user_baselines:
            return 0.0

        baseline = self.user_baselines[user_id]
        current_location = {
            "country": request_data.get("country", ""),
            "city": request_data.get("city", ""),
            "latitude": request_data.get("latitude", 0.0),
            "longitude": request_data.get("longitude", 0.0),
        }

        # Simple distance calculation (can be improved with geolocation libraries)
        typical_locations = baseline.typical_locations
        if not typical_locations:
            return 0.0

        min_distance = float("inf")
        for location in typical_locations:
            if location["country"] == current_location["country"]:
                distance = abs(location["latitude"] - current_location["latitude"]) + abs(
                    location["longitude"] - current_location["longitude"]
                )
                min_distance = min(min_distance, distance)

        return min(min_distance / 180.0, 1.0)  # Normalize to 0-1

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        from collections import Counter

        counts = Counter(text)
        total = len(text)
        entropy = 0.0

        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)

        return min(entropy / 8.0, 1.0)  # Normalize

    def _is_suspicious_user_agent(self, user_agent: str) -> float:
        """Check if user agent is suspicious."""
        if not user_agent:
            return 0.5

        user_agent_lower = user_agent.lower()
        for suspicious in self.suspicious_user_agents:
            if suspicious in user_agent_lower:
                return 1.0

        return 0.0

    def _calculate_failed_login_rate(self, user_id: str, ip_address: str) -> float:
        """Calculate failed login rate."""
        now = datetime.utcnow()
        failed_logins = 0
        total_logins = 0

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(hours=1) and req.get("endpoint", "").endswith(
                    "/login"
                ):
                    total_logins += 1
                    if req.get("status_code", 200) >= 400:
                        failed_logins += 1

        return failed_logins / max(total_logins, 1)

    def _calculate_successful_login_rate(self, user_id: str, ip_address: str) -> float:
        """Calculate successful login rate."""
        return 1.0 - self._calculate_failed_login_rate(user_id, ip_address)

    def _calculate_mfa_bypass_attempts(self, user_id: str, ip_address: str) -> float:
        """Calculate MFA bypass attempts."""
        now = datetime.utcnow()
        mfa_bypass_attempts = 0

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(hours=1):
                    endpoint = req.get("endpoint", "")
                    if "/mfa/" in endpoint and req.get("status_code", 200) >= 400:
                        mfa_bypass_attempts += 1

        return min(mfa_bypass_attempts / 10.0, 1.0)

    def _calculate_error_rate(self, user_id: str, ip_address: str) -> float:
        """Calculate error rate."""
        now = datetime.utcnow()
        errors = 0
        total_requests = 0

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(hours=1):
                    total_requests += 1
                    if req.get("status_code", 200) >= 400:
                        errors += 1

        return errors / max(total_requests, 1)

    def _calculate_privileged_access(self, user_id: str, ip_address: str) -> float:
        """Calculate privileged endpoint access."""
        now = datetime.utcnow()
        privileged_access = 0

        privileged_endpoints = ["/admin/", "/api/v1/system/", "/api/v1/security/"]

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(hours=1):
                    endpoint = req.get("endpoint", "")
                    if any(priv_endpoint in endpoint for priv_endpoint in privileged_endpoints):
                        privileged_access += 1

        return min(privileged_access / 10.0, 1.0)

    def _calculate_data_access_volume(self, user_id: str, ip_address: str) -> float:
        """Calculate data access volume."""
        now = datetime.utcnow()
        data_volume = 0

        for requests in [
            self.user_request_history.get(user_id, []),
            self.ip_request_history.get(ip_address, []),
        ]:
            for req in requests:
                req_time = datetime.fromisoformat(req.get("timestamp", now.isoformat()))
                if now - req_time < timedelta(hours=1):
                    data_volume += len(json.dumps(req))

        return min(data_volume / 1000000.0, 1.0)  # Normalize to MB

    def _calculate_baseline_deviation(self, user_id: str, request_data: Dict[str, Any]) -> float:
        """Calculate deviation from user's behavioral baseline."""
        if not user_id or user_id not in self.user_baselines:
            return 0.0

        baseline = self.user_baselines[user_id]
        current_patterns = self._extract_request_patterns(request_data)

        # Simple deviation calculation
        deviation = 0.0
        for pattern, value in current_patterns.items():
            baseline_value = baseline.typical_request_patterns.get(pattern, value)
            deviation += abs(value - baseline_value)

        return min(deviation / len(current_patterns), 1.0)

    def _extract_request_patterns(self, request_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract request patterns for baseline comparison."""
        return {
            "request_size": len(json.dumps(request_data)),
            "endpoint_length": len(request_data.get("endpoint", "")),
            "has_query_params": float("?" in request_data.get("endpoint", "")),
            "is_post_request": float(request_data.get("method", "GET") == "POST"),
        }

    def _calculate_session_duration(self, user_id: str, request_data: Dict[str, Any]) -> float:
        """Calculate session duration."""
        if not user_id:
            return 0.0

        requests = self.user_request_history.get(user_id, [])
        if len(requests) < 2:
            return 0.0

        first_request = datetime.fromisoformat(
            requests[0].get("timestamp", datetime.utcnow().isoformat())
        )
        last_request = datetime.fromisoformat(
            requests[-1].get("timestamp", datetime.utcnow().isoformat())
        )

        duration = (last_request - first_request).total_seconds()
        return min(duration / 3600.0, 1.0)  # Normalize to hours

    def _calculate_concurrent_sessions(self, user_id: str) -> float:
        """Calculate number of concurrent sessions."""
        # This would require session tracking - simplified implementation
        return 0.0


class MLThreatDetector:
    """Machine Learning-based threat detection system."""

    def __init__(self, security_monitor: SecurityMonitoringSystem):
        self.security_monitor = security_monitor
        self.feature_extractor = FeatureExtractor()
        self.model = IsolationForest(
            contamination=0.01,  # 1% of data expected to be anomalous
            random_state=42,
            n_estimators=100,
            max_samples=256,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "/tmp/threat_detection_model.pkl"
        self.scaler_path = "/tmp/threat_detection_scaler.pkl"

        # Performance tracking
        self.prediction_times = deque(maxlen=1000)
        self.recent_predictions = deque(maxlen=10000)

        # Load pre-trained model if available
        self._load_model()

    def train_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the threat detection model."""
        try:
            logger.info(f"Training threat detection model with {len(training_data)} samples")

            # Extract features from training data
            features = []
            for data in training_data:
                feature_vector = self.feature_extractor.extract_features(data)
                features.append(feature_vector.to_array())

            if not features:
                raise ValueError("No features extracted from training data")

            # Convert to numpy array
            X = np.array(features)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True

            # Save model
            self._save_model()

            # Calculate training metrics
            scores = self.model.decision_function(X_scaled)
            predictions = self.model.predict(X_scaled)

            anomaly_count = np.sum(predictions == -1)
            anomaly_percentage = (anomaly_count / len(predictions)) * 100

            logger.info(f"Model training completed. Anomaly rate: {anomaly_percentage:.2f}%")

            return {
                "trained": True,
                "samples_used": len(training_data),
                "anomaly_rate": anomaly_percentage,
                "feature_count": len(features[0]) if features else 0,
                "model_type": "IsolationForest",
            }

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {"trained": False, "error": str(e)}

    async def analyze_request(self, request_data: Dict[str, Any]) -> ThreatPrediction:
        """Analyze a request for threats using ML model."""
        start_time = time.time()

        try:
            # Extract features
            features = self.feature_extractor.extract_features(request_data)
            feature_array = features.to_array().reshape(1, -1)

            # Default prediction for untrained model
            if not self.is_trained:
                prediction = ThreatPrediction(
                    risk_score=0.1,
                    threat_level=ThreatLevel.LOW,
                    confidence=0.5,
                    detected_attacks=[],
                    features_contribution={},
                )
                return prediction

            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array)

            # Make prediction
            anomaly_score = self.model.decision_function(feature_array_scaled)[0]
            is_anomaly = self.model.predict(feature_array_scaled)[0] == -1

            # Convert anomaly score to risk score (0-1)
            risk_score = self._convert_anomaly_score_to_risk(anomaly_score)

            # Determine threat level
            threat_level = self._determine_threat_level(risk_score)

            # Calculate confidence
            confidence = min(abs(anomaly_score) / 2.0, 1.0)

            # Detect specific attack types
            detected_attacks = self._detect_attack_types(features, request_data)

            # Calculate feature contributions
            features_contribution = self._calculate_feature_contributions(features, anomaly_score)

            # Create prediction
            prediction = ThreatPrediction(
                risk_score=risk_score,
                threat_level=threat_level,
                confidence=confidence,
                detected_attacks=detected_attacks,
                features_contribution=features_contribution,
            )

            # Track prediction
            self.recent_predictions.append(prediction)

            # Log high-risk predictions
            if risk_score > 0.7:
                await self._log_high_risk_prediction(prediction, request_data)

            # Track performance
            end_time = time.time()
            prediction_time = (end_time - start_time) * 1000  # Convert to ms
            self.prediction_times.append(prediction_time)

            return prediction

        except Exception as e:
            logger.error(f"Threat analysis failed: {str(e)}")

            # Return safe default prediction
            return ThreatPrediction(
                risk_score=0.0,
                threat_level=ThreatLevel.LOW,
                confidence=0.0,
                detected_attacks=[],
                features_contribution={},
            )

    def _convert_anomaly_score_to_risk(self, anomaly_score: float) -> float:
        """Convert isolation forest anomaly score to risk score."""
        # Anomaly scores are typically between -1 and 1
        # More negative = more anomalous
        risk_score = max(0, (1 - anomaly_score) / 2)
        return min(risk_score, 1.0)

    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Determine threat level based on risk score."""
        if risk_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            return ThreatLevel.HIGH
        elif risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    def _detect_attack_types(
        self, features: ThreatFeatures, request_data: Dict[str, Any]
    ) -> List[AttackType]:
        """Detect specific attack types based on features."""
        detected_attacks = []

        # Brute force detection
        if features.failed_login_rate > 0.5 and features.request_frequency > 0.8:
            detected_attacks.append(AttackType.BRUTE_FORCE)

        # DDoS detection
        if features.request_frequency > 0.9 and features.requests_per_minute > 0.8:
            detected_attacks.append(AttackType.DDoS)

        # SQL injection detection
        endpoint = request_data.get("endpoint", "")
        if any(pattern in endpoint.lower() for pattern in ["select", "union", "drop", "insert"]):
            detected_attacks.append(AttackType.SQL_INJECTION)

        # Suspicious activity detection
        if features.is_suspicious_user_agent > 0.5 or features.country_risk_score > 0.7:
            detected_attacks.append(AttackType.SUSPICIOUS_ACTIVITY)

        # Privilege escalation detection
        if features.privileged_endpoint_access > 0.5 and features.mfa_bypass_attempts > 0.3:
            detected_attacks.append(AttackType.PRIVILEGE_ESCALATION)

        return detected_attacks

    def _calculate_feature_contributions(
        self, features: ThreatFeatures, anomaly_score: float
    ) -> Dict[str, float]:
        """Calculate feature contributions to the anomaly score."""
        feature_names = [
            "request_frequency",
            "request_size",
            "endpoint_diversity",
            "method_diversity",
            "time_since_last_request",
            "requests_per_minute",
            "requests_per_hour",
            "country_risk_score",
            "distance_from_usual_location",
            "user_agent_entropy",
            "is_suspicious_user_agent",
            "failed_login_rate",
            "successful_login_rate",
            "mfa_bypass_attempts",
            "error_rate",
            "privileged_endpoint_access",
            "data_access_volume",
            "deviation_from_baseline",
            "session_duration",
            "concurrent_sessions",
        ]

        feature_array = features.to_array()
        contributions = {}

        # Simple contribution calculation (can be improved with SHAP or similar)
        for i, name in enumerate(feature_names):
            contribution = abs(feature_array[i] * anomaly_score)
            contributions[name] = contribution

        return contributions

    async def _log_high_risk_prediction(
        self, prediction: ThreatPrediction, request_data: Dict[str, Any]
    ):
        """Log high-risk prediction as security event."""
        security_auditor.log_event(
            event_type=SecurityEventType.SUSPICIOUS_PATTERN,
            severity=SecurityEventSeverity.HIGH,
            message=f"High-risk threat detected: {prediction.threat_level}",
            user_id=request_data.get("user_id"),
            details={
                "risk_score": prediction.risk_score,
                "threat_level": prediction.threat_level,
                "confidence": prediction.confidence,
                "detected_attacks": [str(attack) for attack in prediction.detected_attacks],
                "ip_address": request_data.get("ip_address"),
                "endpoint": request_data.get("endpoint"),
                "user_agent": request_data.get("user_agent"),
            },
        )

    def _save_model(self):
        """Save trained model to disk."""
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

    def _load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.debug(f"No pre-trained model found: {str(e)}")
            self.is_trained = False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the threat detection system."""
        if not self.prediction_times:
            return {
                "average_prediction_time_ms": 0,
                "model_trained": self.is_trained,
                "recent_predictions": 0,
            }

        avg_time = np.mean(self.prediction_times)
        max_time = np.max(self.prediction_times)
        min_time = np.min(self.prediction_times)

        # Calculate threat level distribution
        threat_levels = [pred.threat_level for pred in self.recent_predictions]
        threat_level_counts = {}
        for level in ThreatLevel:
            threat_level_counts[level] = sum(1 for tl in threat_levels if tl == level)

        return {
            "average_prediction_time_ms": avg_time,
            "max_prediction_time_ms": max_time,
            "min_prediction_time_ms": min_time,
            "model_trained": self.is_trained,
            "recent_predictions": len(self.recent_predictions),
            "threat_level_distribution": threat_level_counts,
            "latency_target_met": avg_time < 50.0,  # Target: sub-50ms
        }

    def retrain_model(self, new_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrain the model with new data."""
        return self.train_model(new_data)

    def update_user_baseline(self, user_id: str, request_data: Dict[str, Any]):
        """Update user behavioral baseline."""
        if user_id not in self.feature_extractor.user_baselines:
            self.feature_extractor.user_baselines[user_id] = UserBehaviorBaseline(
                user_id=user_id,
                typical_request_patterns={},
                typical_timing_patterns={},
                typical_locations=[],
                typical_user_agents=[],
            )

        baseline = self.feature_extractor.user_baselines[user_id]

        # Update patterns
        patterns = self.feature_extractor._extract_request_patterns(request_data)
        for pattern, value in patterns.items():
            if pattern not in baseline.typical_request_patterns:
                baseline.typical_request_patterns[pattern] = value
            else:
                # Simple exponential smoothing
                baseline.typical_request_patterns[pattern] = (
                    0.9 * baseline.typical_request_patterns[pattern] + 0.1 * value
                )

        # Update location
        location = {
            "country": request_data.get("country", ""),
            "city": request_data.get("city", ""),
            "latitude": request_data.get("latitude", 0.0),
            "longitude": request_data.get("longitude", 0.0),
        }
        baseline.typical_locations.append(location)
        # Keep only recent locations
        baseline.typical_locations = baseline.typical_locations[-100:]

        # Update user agents
        user_agent = request_data.get("user_agent", "")
        if user_agent and user_agent not in baseline.typical_user_agents:
            baseline.typical_user_agents.append(user_agent)
            # Keep only recent user agents
            baseline.typical_user_agents = baseline.typical_user_agents[-10:]

        baseline.last_updated = datetime.utcnow()


# Global threat detector instance
ml_threat_detector = None


def get_ml_threat_detector() -> MLThreatDetector:
    """Get the global ML threat detector instance."""
    global ml_threat_detector
    if ml_threat_detector is None:
        security_monitor = SecurityMonitoringSystem()
        ml_threat_detector = MLThreatDetector(security_monitor)
    return ml_threat_detector
