"""
Unit tests for Zero-Trust Architecture.

Tests cover:
- Service registration and identity management
- Certificate generation and verification
- Policy evaluation and enforcement
- Continuous verification
- Identity-aware proxy validation
- Service mesh configuration
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request

from auth.zero_trust_architecture import (
    CertificateManager,
    ContinuousVerificationContext,
    IdentityAwareProxy,
    NetworkZone,
    ServiceIdentity,
    ServiceType,
    TrustLevel,
    ZeroTrustPolicy,
    ZeroTrustPolicyEngine,
    configure_default_zero_trust_policies,
    get_zero_trust_engine,
)


@pytest.mark.slow
class TestServiceIdentity:
    """Test suite for ServiceIdentity."""

    def test_service_identity_creation(self):
        """Test service identity creation."""
        identity = ServiceIdentity(
            service_name="test-service",
            service_type=ServiceType.API,
            network_zone=NetworkZone.DMZ,
            trust_level=TrustLevel.MEDIUM,
            certificate_fingerprint="abc123",
            allowed_operations=["read", "write"],
            rate_limits={"requests_per_second": 100},
        )

        assert identity.service_name == "test-service"
        assert identity.service_type == ServiceType.API
        assert identity.network_zone == NetworkZone.DMZ
        assert identity.trust_level == TrustLevel.MEDIUM
        assert identity.certificate_fingerprint == "abc123"
        assert "read" in identity.allowed_operations
        assert "write" in identity.allowed_operations

    def test_service_identity_validity(self):
        """Test service identity validity checking."""
        # Valid identity
        valid_identity = ServiceIdentity(
            service_name="test-service",
            service_type=ServiceType.API,
            network_zone=NetworkZone.DMZ,
            trust_level=TrustLevel.MEDIUM,
            certificate_fingerprint="abc123",
            valid_until=datetime.utcnow() + timedelta(hours=1),
        )
        assert valid_identity.is_valid()

        # Expired identity
        expired_identity = ServiceIdentity(
            service_name="test-service",
            service_type=ServiceType.API,
            network_zone=NetworkZone.DMZ,
            trust_level=TrustLevel.MEDIUM,
            certificate_fingerprint="abc123",
            valid_until=datetime.utcnow() - timedelta(hours=1),
        )
        assert not expired_identity.is_valid()

    def test_operation_authorization(self):
        """Test operation authorization checking."""
        identity = ServiceIdentity(
            service_name="test-service",
            service_type=ServiceType.API,
            network_zone=NetworkZone.DMZ,
            trust_level=TrustLevel.MEDIUM,
            certificate_fingerprint="abc123",
            allowed_operations=["read", "write"],
        )

        assert identity.can_perform_operation("read")
        assert identity.can_perform_operation("write")
        assert not identity.can_perform_operation("admin")


@pytest.mark.slow
class TestZeroTrustPolicy:
    """Test suite for ZeroTrustPolicy."""

    def test_policy_creation(self):
        """Test zero-trust policy creation."""
        policy = ZeroTrustPolicy(
            policy_id="test-policy",
            name="Test Policy",
            description="Test policy description",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ],
            minimum_trust_level=TrustLevel.MEDIUM,
        )

        assert policy.policy_id == "test-policy"
        assert policy.name == "Test Policy"
        assert "service-a" in policy.source_services
        assert "service-b" in policy.target_services
        assert "read" in policy.allowed_operations

    def test_policy_activity_check(self):
        """Test policy activity checking."""
        # Active policy
        active_policy = ZeroTrustPolicy(
            policy_id="active-policy",
            name="Active Policy",
            description="Active policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ],
            minimum_trust_level=TrustLevel.MEDIUM,
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        assert active_policy.is_active()

        # Expired policy
        expired_policy = ZeroTrustPolicy(
            policy_id="expired-policy",
            name="Expired Policy",
            description="Expired policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ],
            minimum_trust_level=TrustLevel.MEDIUM,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        assert not expired_policy.is_active()

    def test_policy_request_matching(self):
        """Test policy request matching."""
        policy = ZeroTrustPolicy(
            policy_id="test-policy",
            name="Test Policy",
            description="Test policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read", "write"],
            network_zones=[NetworkZone.DMZ],
            minimum_trust_level=TrustLevel.MEDIUM,
        )

        # Matching request
        assert policy.matches_request("service-a", "service-b", "read")
        assert policy.matches_request("service-a", "service-b", "write")

        # Non-matching requests
        assert not policy.matches_request("service-c", "service-b", "read")
        assert not policy.matches_request("service-a", "service-c", "read")
        assert not policy.matches_request("service-a", "service-b", "admin")


@pytest.mark.slow
class TestCertificateManager:
    """Test suite for CertificateManager."""

    def test_certificate_manager_initialization(self):
        """Test certificate manager initialization."""
        manager = CertificateManager()

        assert manager.ca_cert is not None
        assert manager.ca_key is not None
        assert hasattr(manager, "service_certificates")

    def test_service_certificate_issuance(self):
        """Test service certificate issuance."""
        manager = CertificateManager()

        cert_info = manager.issue_service_certificate("test-service", ServiceType.API)

        assert "certificate" in cert_info
        assert "private_key" in cert_info
        assert "ca_certificate" in cert_info
        assert "fingerprint" in cert_info
        assert cert_info["certificate"].startswith("-----BEGIN CERTIFICATE-----")
        assert cert_info["private_key"].startswith("-----BEGIN PRIVATE KEY-----")

    def test_certificate_verification(self):
        """Test certificate verification."""
        manager = CertificateManager()

        # Issue a certificate
        cert_info = manager.issue_service_certificate("test-service", ServiceType.API)

        # Verify the certificate
        assert manager.verify_certificate(cert_info["certificate"])

        # Test with invalid certificate
        invalid_cert = "-----BEGIN CERTIFICATE-----\nINVALID\n-----END CERTIFICATE-----"
        assert not manager.verify_certificate(invalid_cert)

    def test_certificate_info_extraction(self):
        """Test certificate information extraction."""
        manager = CertificateManager()

        # Issue a certificate
        cert_info = manager.issue_service_certificate("test-service", ServiceType.API)

        # Extract certificate info
        info = manager.get_certificate_info(cert_info["certificate"])

        assert "subject" in info
        assert "issuer" in info
        assert "fingerprint" in info
        assert "not_valid_before" in info
        assert "not_valid_after" in info
        assert info["is_valid"] is True
        assert info["subject"]["commonName"] == "test-service"


@pytest.mark.slow
class TestZeroTrustPolicyEngine:
    """Test suite for ZeroTrustPolicyEngine."""

    @pytest.fixture
    def policy_engine(self):
        """Create policy engine instance."""
        return ZeroTrustPolicyEngine()

    def test_service_registration(self, policy_engine):
        """Test service registration."""
        service_identity = policy_engine.register_service(
            "test-service", ServiceType.API, NetworkZone.DMZ, TrustLevel.MEDIUM
        )

        assert service_identity.service_name == "test-service"
        assert service_identity.service_type == ServiceType.API
        assert service_identity.network_zone == NetworkZone.DMZ
        assert service_identity.trust_level == TrustLevel.MEDIUM
        assert "test-service" in policy_engine.service_identities

    def test_policy_addition(self, policy_engine):
        """Test policy addition."""
        policy = ZeroTrustPolicy(
            policy_id="test-policy",
            name="Test Policy",
            description="Test policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ],
            minimum_trust_level=TrustLevel.MEDIUM,
        )

        policy_engine.add_policy(policy)

        assert "test-policy" in policy_engine.policies
        assert policy_engine.policies["test-policy"] == policy

    def test_request_evaluation_success(self, policy_engine):
        """Test successful request evaluation."""
        # Register services
        policy_engine.register_service(
            "service-a", ServiceType.API, NetworkZone.DMZ, TrustLevel.MEDIUM
        )
        policy_engine.register_service(
            "service-b",
            ServiceType.DATABASE,
            NetworkZone.INTERNAL,
            TrustLevel.HIGH,
        )

        # Add policy
        policy = ZeroTrustPolicy(
            policy_id="test-policy",
            name="Test Policy",
            description="Test policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ, NetworkZone.INTERNAL],
            minimum_trust_level=TrustLevel.MEDIUM,
        )
        policy_engine.add_policy(policy)

        # Evaluate request
        is_authorized, reason = policy_engine.evaluate_request(
            "service-a", "service-b", "read", {}
        )

        assert is_authorized
        assert reason == "Request authorized"

    def test_request_evaluation_failure(self, policy_engine):
        """Test failed request evaluation."""
        # Register services
        policy_engine.register_service(
            "service-a", ServiceType.API, NetworkZone.DMZ, TrustLevel.LOW
        )
        policy_engine.register_service(
            "service-b",
            ServiceType.DATABASE,
            NetworkZone.INTERNAL,
            TrustLevel.HIGH,
        )

        # Add policy with high trust requirement
        policy = ZeroTrustPolicy(
            policy_id="test-policy",
            name="Test Policy",
            description="Test policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ, NetworkZone.INTERNAL],
            minimum_trust_level=TrustLevel.HIGH,  # Higher than service-a's trust level
        )
        policy_engine.add_policy(policy)

        # Evaluate request
        is_authorized, reason = policy_engine.evaluate_request(
            "service-a", "service-b", "read", {}
        )

        assert not is_authorized
        assert "trust level" in reason.lower()

    def test_network_zone_access_control(self, policy_engine):
        """Test network zone access control."""
        # Test DMZ to Internal access (should be allowed)
        assert policy_engine._check_network_zone_access(
            NetworkZone.DMZ, NetworkZone.INTERNAL
        )

        # Test Internal to DMZ access (should be denied)
        assert not policy_engine._check_network_zone_access(
            NetworkZone.INTERNAL, NetworkZone.DMZ
        )

        # Test Isolated zone access (should only access itself)
        assert policy_engine._check_network_zone_access(
            NetworkZone.ISOLATED, NetworkZone.ISOLATED
        )
        assert not policy_engine._check_network_zone_access(
            NetworkZone.ISOLATED, NetworkZone.INTERNAL
        )

    def test_trust_level_hierarchy(self, policy_engine):
        """Test trust level hierarchy checking."""
        # Higher trust level should meet lower requirement
        assert policy_engine._check_trust_level(TrustLevel.HIGH, TrustLevel.MEDIUM)
        assert policy_engine._check_trust_level(TrustLevel.TRUSTED, TrustLevel.LOW)

        # Lower trust level should not meet higher requirement
        assert not policy_engine._check_trust_level(TrustLevel.LOW, TrustLevel.HIGH)
        assert not policy_engine._check_trust_level(
            TrustLevel.MEDIUM, TrustLevel.TRUSTED
        )

        # Same trust level should meet requirement
        assert policy_engine._check_trust_level(TrustLevel.MEDIUM, TrustLevel.MEDIUM)

    @pytest.mark.asyncio
    async def test_continuous_verification(self, policy_engine):
        """Test continuous verification."""
        # Register service
        policy_engine.register_service(
            "test-service", ServiceType.API, NetworkZone.DMZ, TrustLevel.MEDIUM
        )

        # Mock ML threat detector
        with patch(
            "auth.zero_trust_architecture.get_ml_threat_detector"
        ) as mock_detector:
            mock_prediction = Mock()
            mock_prediction.risk_score = 0.3
            mock_prediction.threat_level = Mock()
            mock_prediction.threat_level.value = "medium"
            mock_prediction.detected_attacks = []

            # Make the analyze_request method async
            async def mock_analyze_request(request_data):
                return mock_prediction

            mock_detector.return_value.analyze_request = mock_analyze_request

            # Test continuous verification
            request_data = {
                "service_name": "test-service",
                "user_id": "user123",
                "ip_address": "192.168.1.100",
            }

            trust_level = await policy_engine.continuous_verification(
                "session123", request_data
            )

            assert trust_level in [
                TrustLevel.UNTRUSTED,
                TrustLevel.LOW,
                TrustLevel.MEDIUM,
                TrustLevel.HIGH,
                TrustLevel.TRUSTED,
            ]
            assert "session123" in policy_engine.active_sessions

    def test_service_mesh_config_generation(self, policy_engine):
        """Test service mesh configuration generation."""
        # Register services
        policy_engine.register_service(
            "service-a", ServiceType.API, NetworkZone.DMZ, TrustLevel.MEDIUM
        )
        policy_engine.register_service(
            "service-b",
            ServiceType.DATABASE,
            NetworkZone.INTERNAL,
            TrustLevel.HIGH,
        )

        # Add policy
        policy = ZeroTrustPolicy(
            policy_id="test-policy",
            name="Test Policy",
            description="Test policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ, NetworkZone.INTERNAL],
            minimum_trust_level=TrustLevel.MEDIUM,
        )
        policy_engine.add_policy(policy)

        # Generate config
        config = policy_engine.get_service_mesh_config()

        assert "version" in config
        assert "services" in config
        assert "policies" in config
        assert "certificates" in config
        assert "service-a" in config["services"]
        assert "service-b" in config["services"]
        assert len(config["policies"]) == 1
        assert config["policies"][0]["id"] == "test-policy"

    def test_policy_conditions_checking(self, policy_engine):
        """Test policy conditions checking."""
        policy = ZeroTrustPolicy(
            policy_id="test-policy",
            name="Test Policy",
            description="Test policy",
            source_services=["service-a"],
            target_services=["service-b"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ],
            minimum_trust_level=TrustLevel.MEDIUM,
            conditions={
                "ip_whitelist": ["192.168.1.0/24"],
                "user_roles": ["admin", "user"],
            },
        )

        # Test matching conditions
        matching_context = {
            "client_ip": "192.168.1.100",
            "user_roles": ["admin"],
        }
        assert policy_engine._check_policy_conditions(policy, matching_context)

        # Test non-matching IP
        non_matching_context = {
            "client_ip": "10.0.0.1",
            "user_roles": ["admin"],
        }
        assert not policy_engine._check_policy_conditions(policy, non_matching_context)

        # Test non-matching roles
        non_matching_roles = {
            "client_ip": "192.168.1.100",
            "user_roles": ["guest"],
        }
        assert not policy_engine._check_policy_conditions(policy, non_matching_roles)


@pytest.mark.slow
class TestIdentityAwareProxy:
    """Test suite for IdentityAwareProxy."""

    @pytest.fixture
    def mock_policy_engine(self):
        """Mock policy engine."""
        engine = Mock()
        engine.certificate_manager = Mock()
        engine.evaluate_request = Mock(return_value=(True, "Request authorized"))
        engine.continuous_verification = Mock(return_value=TrustLevel.MEDIUM)
        return engine

    @pytest.fixture
    def proxy(self, mock_policy_engine):
        """Create identity-aware proxy."""
        return IdentityAwareProxy(mock_policy_engine)

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/v1/test"
        request.headers = {
            "X-Client-Certificate": "mock-certificate",
            "user-agent": "test-agent",
        }
        request.client.host = "192.168.1.100"
        return request

    @pytest.mark.asyncio
    async def test_validate_request_success(self, proxy, mock_request):
        """Test successful request validation."""
        # Configure mocks
        proxy.policy_engine.certificate_manager.verify_certificate.return_value = True
        proxy.policy_engine.certificate_manager.get_certificate_info.return_value = {
            "subject": {"commonName": "test-service"}
        }

        # Mock the async continuous_verification method
        async def mock_continuous_verification(session_id, request_context):
            return TrustLevel.MEDIUM

        proxy.policy_engine.continuous_verification = mock_continuous_verification

        # Validate request
        result = await proxy.validate_request(mock_request, "target-service", "read")

        assert result is True
        proxy.policy_engine.evaluate_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_request_no_certificate(self, proxy, mock_request):
        """Test request validation without client certificate."""
        # Remove certificate from headers
        mock_request.headers = {"user-agent": "test-agent"}

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.validate_request(mock_request, "target-service", "read")

        assert exc_info.value.status_code == 401
        assert "certificate required" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_validate_request_invalid_certificate(self, proxy, mock_request):
        """Test request validation with invalid certificate."""
        # Configure mock to reject certificate
        proxy.policy_engine.certificate_manager.verify_certificate.return_value = False

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.validate_request(mock_request, "target-service", "read")

        assert exc_info.value.status_code == 401
        assert "invalid" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_validate_request_policy_denial(self, proxy, mock_request):
        """Test request validation with policy denial."""
        # Configure mocks
        proxy.policy_engine.certificate_manager.verify_certificate.return_value = True
        proxy.policy_engine.certificate_manager.get_certificate_info.return_value = {
            "subject": {"commonName": "test-service"}
        }
        proxy.policy_engine.evaluate_request.return_value = (
            False,
            "Access denied",
        )

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.validate_request(mock_request, "target-service", "read")

        assert exc_info.value.status_code == 403
        assert "access denied" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_validate_request_low_trust(self, proxy, mock_request):
        """Test request validation with low trust level."""
        # Configure mocks
        proxy.policy_engine.certificate_manager.verify_certificate.return_value = True
        proxy.policy_engine.certificate_manager.get_certificate_info.return_value = {
            "subject": {"commonName": "test-service"}
        }

        # Mock the async continuous_verification method to return UNTRUSTED
        async def mock_continuous_verification(session_id, request_context):
            return TrustLevel.UNTRUSTED

        proxy.policy_engine.continuous_verification = mock_continuous_verification

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.validate_request(mock_request, "target-service", "read")

        assert exc_info.value.status_code == 403
        assert "trust level" in exc_info.value.detail.lower()

    def test_get_client_ip_forwarded(self, proxy, mock_request):
        """Test client IP extraction with forwarded headers."""
        mock_request.headers = {
            "X-Forwarded-For": "203.0.113.1, 198.51.100.1",
            "X-Real-IP": "203.0.113.2",
        }

        ip = proxy._get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_real_ip(self, proxy, mock_request):
        """Test client IP extraction with real IP header."""
        mock_request.headers = {"X-Real-IP": "203.0.113.2"}

        ip = proxy._get_client_ip(mock_request)
        assert ip == "203.0.113.2"

    def test_get_client_ip_direct(self, proxy, mock_request):
        """Test client IP extraction from direct connection."""
        mock_request.headers = {}

        ip = proxy._get_client_ip(mock_request)
        assert ip == "192.168.1.100"

    def test_generate_session_id(self, proxy, mock_request):
        """Test session ID generation."""
        session_id = proxy._generate_session_id(mock_request)

        assert isinstance(session_id, str)
        assert len(session_id) == 64  # SHA256 hex digest length

        # Same request should generate same session ID
        session_id2 = proxy._generate_session_id(mock_request)
        assert session_id == session_id2


@pytest.mark.slow
class TestContinuousVerificationContext:
    """Test suite for ContinuousVerificationContext."""

    def test_context_creation(self):
        """Test continuous verification context creation."""
        service_identity = ServiceIdentity(
            service_name="test-service",
            service_type=ServiceType.API,
            network_zone=NetworkZone.DMZ,
            trust_level=TrustLevel.MEDIUM,
            certificate_fingerprint="abc123",
        )

        context = ContinuousVerificationContext(
            session_id="session123",
            user_id="user456",
            service_identity=service_identity,
            initial_trust_level=TrustLevel.MEDIUM,
            current_trust_level=TrustLevel.MEDIUM,
            risk_score=0.3,
        )

        assert context.session_id == "session123"
        assert context.user_id == "user456"
        assert context.service_identity == service_identity
        assert context.initial_trust_level == TrustLevel.MEDIUM
        assert context.current_trust_level == TrustLevel.MEDIUM
        assert context.risk_score == 0.3

    def test_trust_level_update(self):
        """Test trust level update based on risk score."""
        service_identity = ServiceIdentity(
            service_name="test-service",
            service_type=ServiceType.API,
            network_zone=NetworkZone.DMZ,
            trust_level=TrustLevel.MEDIUM,
            certificate_fingerprint="abc123",
        )

        context = ContinuousVerificationContext(
            session_id="session123",
            user_id="user456",
            service_identity=service_identity,
            initial_trust_level=TrustLevel.MEDIUM,
            current_trust_level=TrustLevel.MEDIUM,
            risk_score=0.3,
        )

        # High risk score should lower trust level
        context.update_trust_level(0.9)
        assert context.current_trust_level == TrustLevel.UNTRUSTED
        assert context.risk_score == 0.9
        assert context.verification_count == 1

        # Low risk score should raise trust level
        context.update_trust_level(0.1)
        assert context.current_trust_level == TrustLevel.TRUSTED
        assert context.risk_score == 0.1
        assert context.verification_count == 2


@pytest.mark.slow
class TestGlobalFunctions:
    """Test suite for global functions."""

    def test_get_zero_trust_engine(self):
        """Test getting global zero-trust engine."""
        engine = get_zero_trust_engine()
        assert isinstance(engine, ZeroTrustPolicyEngine)

        # Should return same instance
        engine2 = get_zero_trust_engine()
        assert engine is engine2

    def test_configure_default_policies(self):
        """Test configuring default zero-trust policies."""
        # This function should run without errors
        configure_default_zero_trust_policies()

        # Check that some policies were created
        engine = get_zero_trust_engine()
        assert len(engine.policies) > 0
        assert len(engine.service_identities) > 0

    def test_default_operations_mapping(self):
        """Test default operations mapping for service types."""
        engine = ZeroTrustPolicyEngine()

        # Test API service operations
        api_ops = engine._get_default_operations(ServiceType.API)
        assert "read" in api_ops
        assert "write" in api_ops
        assert "execute" in api_ops

        # Test database service operations
        db_ops = engine._get_default_operations(ServiceType.DATABASE)
        assert "read" in db_ops
        assert "write" in db_ops
        assert "admin" in db_ops

        # Test admin service operations
        admin_ops = engine._get_default_operations(ServiceType.ADMIN)
        assert "read" in admin_ops
        assert "write" in admin_ops
        assert "execute" in admin_ops
        assert "admin" in admin_ops

    def test_default_rate_limits_mapping(self):
        """Test default rate limits mapping for service types."""
        engine = ZeroTrustPolicyEngine()

        # Test API service rate limits
        api_limits = engine._get_default_rate_limits(ServiceType.API)
        assert "requests_per_second" in api_limits
        assert "requests_per_minute" in api_limits
        assert api_limits["requests_per_second"] == 100

        # Test database service rate limits
        db_limits = engine._get_default_rate_limits(ServiceType.DATABASE)
        assert "requests_per_second" in db_limits
        assert "requests_per_minute" in db_limits
        assert db_limits["requests_per_second"] == 50

        # Test admin service rate limits (should be more restrictive)
        admin_limits = engine._get_default_rate_limits(ServiceType.ADMIN)
        assert "requests_per_second" in admin_limits
        assert "requests_per_minute" in admin_limits
        assert admin_limits["requests_per_second"] == 5  # More restrictive


@pytest.mark.slow
class TestErrorHandling:
    """Test suite for error handling in zero-trust architecture."""

    def test_certificate_manager_error_handling(self):
        """Test certificate manager error handling."""
        # Test with invalid paths
        manager = CertificateManager(
            ca_cert_path="/invalid/path", ca_key_path="/invalid/path"
        )

        # Should still work by creating new certificates
        assert manager.ca_cert is not None
        assert manager.ca_key is not None

    def test_invalid_certificate_handling(self):
        """Test handling of invalid certificates."""
        manager = CertificateManager()

        # Test with completely invalid certificate
        assert not manager.verify_certificate("invalid certificate")

        # Test with malformed PEM
        malformed_pem = (
            "-----BEGIN CERTIFICATE-----\nMALFORMED\n-----END CERTIFICATE-----"
        )
        assert not manager.verify_certificate(malformed_pem)

    @pytest.mark.asyncio
    async def test_proxy_error_handling(self):
        """Test identity-aware proxy error handling."""
        # Create proxy with mock engine that raises exception
        mock_engine = Mock()
        mock_engine.certificate_manager.verify_certificate.side_effect = Exception(
            "Test error"
        )

        proxy = IdentityAwareProxy(mock_engine)

        # Mock request
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/v1/test"
        request.headers = {"X-Client-Certificate": "mock-cert"}
        request.client.host = "192.168.1.100"

        # Should raise HTTPException for internal errors
        with pytest.raises(HTTPException) as exc_info:
            await proxy.validate_request(request, "target-service", "read")

        assert exc_info.value.status_code == 500
        assert "internal" in exc_info.value.detail.lower()

    def test_policy_evaluation_with_missing_services(self):
        """Test policy evaluation with missing services."""
        engine = ZeroTrustPolicyEngine()

        # Try to evaluate request with non-existent services
        is_authorized, reason = engine.evaluate_request(
            "non-existent-service", "also-non-existent", "read", {}
        )

        assert not is_authorized
        assert "not registered" in reason.lower()

    def test_policy_evaluation_with_expired_identity(self):
        """Test policy evaluation with expired service identity."""
        engine = ZeroTrustPolicyEngine()

        # Register service with expired identity
        service_identity = ServiceIdentity(
            service_name="expired-service",
            service_type=ServiceType.API,
            network_zone=NetworkZone.DMZ,
            trust_level=TrustLevel.MEDIUM,
            certificate_fingerprint="abc123",
            valid_until=datetime.utcnow() - timedelta(hours=1),  # Expired
        )

        engine.service_identities["expired-service"] = service_identity
        engine.register_service("target-service", ServiceType.API, NetworkZone.DMZ)

        # Evaluate request
        is_authorized, reason = engine.evaluate_request(
            "expired-service", "target-service", "read", {}
        )

        assert not is_authorized
        assert "expired" in reason.lower()
