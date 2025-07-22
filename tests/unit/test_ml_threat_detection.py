"""
Unit tests for ML Threat Detection System.

Tests cover:
- Feature extraction
- ML model training and prediction
- Threat level determination
- Attack type detection
- Performance metrics
- Behavioral baseline updates
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from auth.ml_threat_detection import (
    FeatureExtractor,
    MLThreatDetector,
    ThreatFeatures,
    ThreatPrediction,
    UserBehaviorBaseline,
    get_ml_threat_detector,
)
from observability.security_monitoring import (
    AttackType,
    SecurityMonitoringSystem,
    ThreatLevel,
)


class TestFeatureExtractor:
    """Test suite for FeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return FeatureExtractor()

    @pytest.fixture
    def sample_request(self):
        """Sample request data."""
        return {
            "user_id": "user123",
            "ip_address": "192.168.1.100",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/api/v1/users/profile",
            "method": "GET",
            "status_code": 200,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "country": "US",
            "city": "New York",
            "latitude": 40.7128,
            "longitude": -74.0060,
        }

    def test_extract_features_basic(self, extractor, sample_request):
        """Test basic feature extraction."""
        features = extractor.extract_features(sample_request)

        assert isinstance(features, ThreatFeatures)
        assert 0 <= features.request_frequency <= 1
        assert features.request_size > 0
        assert 0 <= features.endpoint_diversity <= 1
        assert 0 <= features.method_diversity <= 1
        assert 0 <= features.country_risk_score <= 1

    def test_extract_features_suspicious_user_agent(self, extractor, sample_request):
        """Test feature extraction with suspicious user agent."""
        sample_request["user_agent"] = "curl/7.68.0"
        features = extractor.extract_features(sample_request)

        assert features.is_suspicious_user_agent == 1.0

    def test_extract_features_high_risk_country(self, extractor, sample_request):
        """Test feature extraction with high-risk country."""
        sample_request["country"] = "TOR"  # Tor network
        features = extractor.extract_features(sample_request)

        assert features.country_risk_score == 1.0

    def test_request_frequency_calculation(self, extractor, sample_request):
        """Test request frequency calculation."""
        # First request
        features1 = extractor.extract_features(sample_request)
        assert features1.request_frequency >= 0

        # Second request quickly after
        features2 = extractor.extract_features(sample_request)
        assert features2.request_frequency > features1.request_frequency

    def test_entropy_calculation(self, extractor):
        """Test entropy calculation."""
        # Low entropy (repetitive)
        low_entropy = extractor._calculate_entropy("aaaaaa")
        assert low_entropy < 0.5

        # High entropy (random)
        high_entropy = extractor._calculate_entropy("abcdef123!@#")
        assert high_entropy > low_entropy

    def test_failed_login_rate_calculation(self, extractor):
        """Test failed login rate calculation."""
        user_id = "user123"
        ip_address = "192.168.1.100"

        # Add failed login attempts
        failed_request = {
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/api/v1/auth/login",
            "status_code": 401,
        }

        # Add successful login
        success_request = {
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/api/v1/auth/login",
            "status_code": 200,
        }

        extractor.extract_features(failed_request)
        extractor.extract_features(failed_request)
        extractor.extract_features(success_request)

        # Calculate rate
        failed_rate = extractor._calculate_failed_login_rate(user_id, ip_address)
        assert 0 <= failed_rate <= 1
        assert failed_rate > 0  # Should have some failed attempts

    def test_features_to_array(self):
        """Test converting features to numpy array."""
        features = ThreatFeatures(request_frequency=0.5, request_size=100, endpoint_diversity=0.3)

        array = features.to_array()
        assert isinstance(array, np.ndarray)
        assert len(array) == 20  # Should have 20 features
        assert array[0] == 0.5  # request_frequency
        assert array[1] == 100  # request_size
        assert array[2] == 0.3  # endpoint_diversity


class TestMLThreatDetector:
    """Test suite for MLThreatDetector."""

    @pytest.fixture
    def mock_security_monitor(self):
        """Mock security monitoring system."""
        return Mock(spec=SecurityMonitoringSystem)

    @pytest.fixture
    def detector(self, mock_security_monitor):
        """Create ML threat detector instance."""
        return MLThreatDetector(mock_security_monitor)

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data."""
        data = []
        for i in range(100):
            data.append(
                {
                    "user_id": f"user{i}",
                    "ip_address": f"192.168.1.{i}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoint": f"/api/v1/endpoint{i % 5}",
                    "method": "GET",
                    "status_code": 200,
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "country": "US",
                }
            )
        return data

    @pytest.fixture
    def sample_request(self):
        """Sample request for analysis."""
        return {
            "user_id": "user123",
            "ip_address": "192.168.1.100",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/api/v1/users/profile",
            "method": "GET",
            "status_code": 200,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "country": "US",
        }

    def test_train_model_success(self, detector, sample_training_data):
        """Test successful model training."""
        result = detector.train_model(sample_training_data)

        assert result["trained"] is True
        assert result["samples_used"] == len(sample_training_data)
        assert "anomaly_rate" in result
        assert "feature_count" in result
        assert detector.is_trained is True

    def test_train_model_empty_data(self, detector):
        """Test model training with empty data."""
        result = detector.train_model([])

        assert result["trained"] is False
        assert "error" in result
        assert detector.is_trained is False

    @pytest.mark.asyncio
    async def test_analyze_request_untrained_model(self, detector, sample_request):
        """Test request analysis with untrained model."""
        prediction = await detector.analyze_request(sample_request)

        assert isinstance(prediction, ThreatPrediction)
        assert prediction.risk_score == 0.1  # Default for untrained model
        assert prediction.threat_level == ThreatLevel.LOW
        assert prediction.confidence == 0.5
        assert prediction.detected_attacks == []

    @pytest.mark.asyncio
    async def test_analyze_request_trained_model(
        self, detector, sample_training_data, sample_request
    ):
        """Test request analysis with trained model."""
        # Train the model first
        detector.train_model(sample_training_data)

        # Analyze request
        prediction = await detector.analyze_request(sample_request)

        assert isinstance(prediction, ThreatPrediction)
        assert 0 <= prediction.risk_score <= 1
        assert prediction.threat_level in [
            ThreatLevel.LOW,
            ThreatLevel.MEDIUM,
            ThreatLevel.HIGH,
            ThreatLevel.CRITICAL,
        ]
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.detected_attacks, list)
        assert isinstance(prediction.features_contribution, dict)

    @pytest.mark.asyncio
    async def test_analyze_suspicious_request(self, detector, sample_training_data):
        """Test analysis of suspicious request."""
        # Train the model
        detector.train_model(sample_training_data)

        # Create suspicious request
        suspicious_request = {
            "user_id": "attacker",
            "ip_address": "10.0.0.1",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/api/v1/admin/users?id=1 UNION SELECT * FROM users",
            "method": "GET",
            "status_code": 200,
            "user_agent": "sqlmap/1.0",
            "country": "TOR",
        }

        prediction = await detector.analyze_request(suspicious_request)

        # Should detect higher risk
        assert prediction.risk_score > 0.5
        assert AttackType.SQL_INJECTION in prediction.detected_attacks
        assert AttackType.SUSPICIOUS_ACTIVITY in prediction.detected_attacks

    def test_convert_anomaly_score_to_risk(self, detector):
        """Test anomaly score to risk score conversion."""
        # Normal score (close to 0)
        normal_risk = detector._convert_anomaly_score_to_risk(0.0)
        assert 0.4 <= normal_risk <= 0.6

        # Anomalous score (negative)
        anomalous_risk = detector._convert_anomaly_score_to_risk(-0.5)
        assert anomalous_risk > normal_risk

        # Very anomalous score
        very_anomalous_risk = detector._convert_anomaly_score_to_risk(-1.0)
        assert very_anomalous_risk == 1.0

    def test_determine_threat_level(self, detector):
        """Test threat level determination."""
        assert detector._determine_threat_level(0.1) == ThreatLevel.LOW
        assert detector._determine_threat_level(0.5) == ThreatLevel.MEDIUM
        assert detector._determine_threat_level(0.7) == ThreatLevel.HIGH
        assert detector._determine_threat_level(0.9) == ThreatLevel.CRITICAL

    def test_detect_attack_types(self, detector):
        """Test attack type detection."""
        # Create features indicating brute force
        brute_force_features = ThreatFeatures(failed_login_rate=0.8, request_frequency=0.9)

        brute_force_request = {
            "endpoint": "/api/v1/auth/login",
            "user_agent": "Mozilla/5.0",
        }

        attacks = detector._detect_attack_types(brute_force_features, brute_force_request)
        assert AttackType.BRUTE_FORCE in attacks

        # Create features indicating SQL injection
        sql_injection_features = ThreatFeatures()
        sql_injection_request = {
            "endpoint": "/api/v1/users?id=1 UNION SELECT password FROM users",
            "user_agent": "Mozilla/5.0",
        }

        attacks = detector._detect_attack_types(sql_injection_features, sql_injection_request)
        assert AttackType.SQL_INJECTION in attacks

    def test_get_performance_metrics(self, detector):
        """Test performance metrics retrieval."""
        metrics = detector.get_performance_metrics()

        assert "average_prediction_time_ms" in metrics
        assert "model_trained" in metrics
        assert "recent_predictions" in metrics
        assert "latency_target_met" in metrics
        assert metrics["model_trained"] == detector.is_trained

    def test_update_user_baseline(self, detector):
        """Test user baseline updates."""
        user_id = "user123"
        request_data = {
            "user_id": user_id,
            "endpoint": "/api/v1/profile",
            "method": "GET",
            "user_agent": "Mozilla/5.0",
            "country": "US",
            "city": "New York",
            "latitude": 40.7128,
            "longitude": -74.0060,
        }

        # Update baseline
        detector.update_user_baseline(user_id, request_data)

        # Check that baseline was created
        assert user_id in detector.feature_extractor.user_baselines
        baseline = detector.feature_extractor.user_baselines[user_id]
        assert isinstance(baseline, UserBehaviorBaseline)
        assert baseline.user_id == user_id
        assert len(baseline.typical_locations) > 0
        assert len(baseline.typical_user_agents) > 0

    @pytest.mark.asyncio
    async def test_high_risk_logging(self, detector, sample_training_data):
        """Test that high-risk predictions are logged."""
        # Train model
        detector.train_model(sample_training_data)

        # Create high-risk request
        high_risk_request = {
            "user_id": "attacker",
            "ip_address": "10.0.0.1",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/api/v1/admin/delete_all_users",
            "method": "POST",
            "status_code": 200,
            "user_agent": "curl/7.68.0",
            "country": "TOR",
        }

        with patch("auth.ml_threat_detection.security_auditor") as mock_auditor:
            prediction = await detector.analyze_request(high_risk_request)

            # If prediction is high risk, should have logged
            if prediction.risk_score > 0.7:
                mock_auditor.log_event.assert_called()

    def test_model_persistence(self, detector, sample_training_data):
        """Test model saving and loading."""
        # Train model
        detector.train_model(sample_training_data)
        original_is_trained = detector.is_trained

        # Save model (happens automatically in train_model)
        assert detector.is_trained == original_is_trained

        # Create new detector and load model
        new_detector = MLThreatDetector(Mock())
        new_detector._load_model()

        # Should load successfully if model was saved
        if original_is_trained:
            assert new_detector.is_trained

    @pytest.mark.asyncio
    async def test_performance_tracking(self, detector, sample_training_data, sample_request):
        """Test performance tracking."""
        # Train model
        detector.train_model(sample_training_data)

        # Analyze request
        await detector.analyze_request(sample_request)

        # Check performance metrics
        metrics = detector.get_performance_metrics()
        assert metrics["recent_predictions"] > 0
        assert metrics["average_prediction_time_ms"] >= 0

        # Check latency target
        assert metrics["latency_target_met"] is True or metrics["latency_target_met"] is False

    def test_retrain_model(self, detector, sample_training_data):
        """Test model retraining."""
        # Initial training
        result1 = detector.train_model(sample_training_data)
        assert result1["trained"] is True

        # Retrain with new data
        new_data = sample_training_data + [
            {
                "user_id": "new_user",
                "ip_address": "192.168.2.1",
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": "/api/v1/new_endpoint",
                "method": "POST",
                "status_code": 200,
                "user_agent": "Mozilla/5.0",
                "country": "US",
            }
        ]

        result2 = detector.retrain_model(new_data)
        assert result2["trained"] is True
        assert result2["samples_used"] == len(new_data)

    def test_error_handling(self, detector):
        """Test error handling in various scenarios."""
        # Test with invalid request data
        invalid_request = {"invalid_field": "invalid_value"}

        # Should not crash
        features = detector.feature_extractor.extract_features(invalid_request)
        assert isinstance(features, ThreatFeatures)

        # Test training with invalid data
        result = detector.train_model([invalid_request])
        assert result["trained"] is False
        assert "error" in result

    def test_feature_contributions(self, detector):
        """Test feature contribution calculation."""
        features = ThreatFeatures(
            request_frequency=0.8,
            failed_login_rate=0.9,
            is_suspicious_user_agent=1.0,
        )

        contributions = detector._calculate_feature_contributions(features, -0.5)

        assert isinstance(contributions, dict)
        assert len(contributions) > 0
        assert "request_frequency" in contributions
        assert "failed_login_rate" in contributions
        assert "is_suspicious_user_agent" in contributions

    def test_get_global_detector(self):
        """Test global detector instance."""
        detector1 = get_ml_threat_detector()
        detector2 = get_ml_threat_detector()

        # Should return same instance
        assert detector1 is detector2
        assert isinstance(detector1, MLThreatDetector)


class TestThreatPrediction:
    """Test suite for ThreatPrediction."""

    def test_threat_prediction_creation(self):
        """Test threat prediction creation."""
        prediction = ThreatPrediction(
            risk_score=0.8,
            threat_level=ThreatLevel.HIGH,
            confidence=0.9,
            detected_attacks=[AttackType.BRUTE_FORCE],
            features_contribution={"request_frequency": 0.5},
        )

        assert prediction.risk_score == 0.8
        assert prediction.threat_level == ThreatLevel.HIGH
        assert prediction.confidence == 0.9
        assert AttackType.BRUTE_FORCE in prediction.detected_attacks
        assert "request_frequency" in prediction.features_contribution
        assert isinstance(prediction.timestamp, datetime)


class TestUserBehaviorBaseline:
    """Test suite for UserBehaviorBaseline."""

    def test_baseline_creation(self):
        """Test baseline creation."""
        baseline = UserBehaviorBaseline(
            user_id="user123",
            typical_request_patterns={"pattern1": 0.5},
            typical_timing_patterns={"timing1": 0.3},
            typical_locations=[{"country": "US"}],
            typical_user_agents=["Mozilla/5.0"],
        )

        assert baseline.user_id == "user123"
        assert baseline.typical_request_patterns["pattern1"] == 0.5
        assert baseline.typical_timing_patterns["timing1"] == 0.3
        assert len(baseline.typical_locations) == 1
        assert len(baseline.typical_user_agents) == 1
        assert isinstance(baseline.last_updated, datetime)
