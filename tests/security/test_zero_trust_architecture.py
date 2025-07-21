"""
Comprehensive tests for Zero-Trust Network Architecture.

Tests cover:
- Mutual TLS (mTLS) certificate management and rotation
- Service mesh configuration and deployment
- Identity-aware proxy with request validation
- Continuous verification system
- Performance requirements (<10ms latency overhead)
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from security.zero_trust.identity_proxy import (
    IdentityAwareProxy,
    ProxyConfig,
    RequestContext,
    SessionRiskScore,
)
from security.zero_trust.mtls_manager import (
    CertificateInfo,
    CertificateRotationPolicy,
    MTLSManager,
    RotationStrategy,
)
from security.zero_trust.service_mesh_config import (
    ServiceMeshConfig,
    ServiceMeshType,
    ServicePolicy,
    TrafficPolicy,
    generate_istio_config,
    generate_linkerd_config,
)


class TestMTLSManager:
    """Test mutual TLS certificate management."""

    @pytest.fixture
    def mtls_manager(self, tmp_path):
        """Create MTLSManager instance for testing."""
        return MTLSManager(
            ca_cert_path=str(tmp_path / "ca-cert.pem"),
            ca_key_path=str(tmp_path / "ca-key.pem"),
            cert_store_path=str(tmp_path / "certs"),
        )

    def test_ca_certificate_generation(self, mtls_manager):
        """Test CA certificate generation."""
        # CA should be automatically generated on initialization
        assert mtls_manager.ca_cert is not None
        assert mtls_manager.ca_key is not None

        # Verify CA certificate properties
        assert (
            mtls_manager.ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[
                0
            ].value
            == "Zero Trust CA"
        )
        assert mtls_manager.ca_cert.issuer == mtls_manager.ca_cert.subject

        # Check certificate is valid
        now = datetime.utcnow()
        assert mtls_manager.ca_cert.not_valid_before <= now
        assert mtls_manager.ca_cert.not_valid_after > now

    def test_service_certificate_generation(self, mtls_manager):
        """Test service certificate generation."""
        service_name = "test-service"
        cert_info = mtls_manager.generate_service_certificate(
            service_name=service_name,
            dns_names=["test-service.local", "test-service.cluster.local"],
            validity_days=30,
        )

        assert isinstance(cert_info, CertificateInfo)
        assert cert_info.service_name == service_name
        assert cert_info.fingerprint is not None
        assert len(cert_info.fingerprint) == 64  # SHA256 hex string

        # Verify certificate properties
        cert = x509.load_pem_x509_certificate(cert_info.certificate.encode())
        assert (
            cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            == service_name
        )
        assert cert.issuer == mtls_manager.ca_cert.subject

        # Check SAN extensions
        san_ext = cert.extensions.get_extension_for_oid(
            x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )
        dns_names = [name.value for name in san_ext.value]
        assert "test-service.local" in dns_names
        assert "test-service.cluster.local" in dns_names

    def test_certificate_validation(self, mtls_manager):
        """Test certificate validation."""
        # Generate a valid certificate
        cert_info = mtls_manager.generate_service_certificate("valid-service")

        # Test valid certificate
        is_valid, message = mtls_manager.validate_certificate(cert_info.certificate)
        assert is_valid is True
        assert message == "Certificate is valid"

        # Test with invalid certificate (self-signed)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        invalid_cert = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "invalid")])
            )
            .issuer_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "invalid")])
            )
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=1))
            .sign(private_key, hashes.SHA256())
        )
        invalid_pem = invalid_cert.public_bytes(serialization.Encoding.PEM).decode()

        is_valid, message = mtls_manager.validate_certificate(invalid_pem)
        assert is_valid is False
        assert "not issued by trusted CA" in message

    def test_certificate_rotation(self, mtls_manager):
        """Test certificate rotation mechanism."""
        service_name = "rotation-test"

        # Generate initial certificate
        initial_cert = mtls_manager.generate_service_certificate(service_name)
        initial_fingerprint = initial_cert.fingerprint

        # Set rotation policy
        policy = CertificateRotationPolicy(
            strategy=RotationStrategy.TIME_BASED,
            rotation_interval_days=30,
            overlap_period_days=7,
            auto_rotate=True,
        )
        mtls_manager.set_rotation_policy(service_name, policy)

        # Simulate certificate nearing expiration
        with patch.object(
            mtls_manager, "_should_rotate_certificate", return_value=True
        ):
            new_cert = mtls_manager.rotate_certificate(service_name)

        assert new_cert.fingerprint != initial_fingerprint
        assert new_cert.service_name == service_name

        # Verify both certificates are valid during overlap period
        is_valid, _ = mtls_manager.validate_certificate(initial_cert.certificate)
        assert is_valid is True
        is_valid, _ = mtls_manager.validate_certificate(new_cert.certificate)
        assert is_valid is True

    def test_certificate_revocation(self, mtls_manager):
        """Test certificate revocation."""
        cert_info = mtls_manager.generate_service_certificate("revoke-test")

        # Certificate should be valid initially
        is_valid, _ = mtls_manager.validate_certificate(cert_info.certificate)
        assert is_valid is True

        # Revoke certificate
        success = mtls_manager.revoke_certificate(
            cert_info.fingerprint, reason="Compromised"
        )
        assert success is True

        # Certificate should now be invalid
        is_valid, message = mtls_manager.validate_certificate(cert_info.certificate)
        assert is_valid is False
        assert "revoked" in message.lower()

    def test_certificate_persistence(self, mtls_manager, tmp_path):
        """Test certificate storage and retrieval."""
        service_name = "persist-test"
        cert_info = mtls_manager.generate_service_certificate(service_name)

        # Save certificate
        mtls_manager.save_certificate(cert_info)

        # Create new manager instance and load certificate
        new_manager = MTLSManager(
            ca_cert_path=str(tmp_path / "ca-cert.pem"),
            ca_key_path=str(tmp_path / "ca-key.pem"),
            cert_store_path=str(tmp_path / "certs"),
        )

        loaded_cert = new_manager.load_certificate(service_name)
        assert loaded_cert is not None
        assert loaded_cert.fingerprint == cert_info.fingerprint
        assert loaded_cert.certificate == cert_info.certificate

    def test_performance_certificate_generation(self, mtls_manager):
        """Test certificate generation performance (<10ms requirement)."""
        # Warm up
        mtls_manager.generate_service_certificate("warmup")

        # Measure generation time
        start_time = time.time()
        for i in range(10):
            mtls_manager.generate_service_certificate(f"perf-test-{i}")
        end_time = time.time()

        avg_time_ms = ((end_time - start_time) / 10) * 1000
        assert avg_time_ms < 10, (
            f"Certificate generation took {avg_time_ms:.2f}ms on average"
        )


class TestIdentityAwareProxy:
    """Test identity-aware proxy functionality."""

    @pytest.fixture
    def proxy(self):
        """Create IdentityAwareProxy instance for testing."""
        config = ProxyConfig(
            enable_mtls=True,
            enable_risk_scoring=True,
            max_risk_score=0.7,
            session_timeout_minutes=30,
        )
        return IdentityAwareProxy(config)

    @pytest.fixture
    def mock_request(self):
        """Create mock request for testing."""
        request = MagicMock()
        request.headers = {
            "X-Client-Certificate": "mock-cert-pem",
            "User-Agent": "test-client/1.0",
            "X-Forwarded-For": "192.168.1.100",
        }
        request.method = "POST"
        request.url.path = "/api/test"
        request.client.host = "192.168.1.100"
        return request

    @pytest.mark.asyncio
    async def test_request_validation(self, proxy, mock_request):
        """Test request validation at every hop."""
        with patch.object(
            proxy.mtls_manager,
            "validate_certificate",
            return_value=(True, "Valid"),
        ):
            context = await proxy.validate_request(
                request=mock_request,
                source_service="service-a",
                target_service="service-b",
                operation="read",
            )

        assert isinstance(context, RequestContext)
        assert context.is_valid is True
        assert context.source_service == "service-a"
        assert context.target_service == "service-b"
        assert context.client_ip == "192.168.1.100"

    @pytest.mark.asyncio
    async def test_dynamic_permission_evaluation(self, proxy):
        """Test dynamic permission evaluation."""
        # Setup test permissions
        proxy.add_service_policy(
            ServicePolicy(
                source_service="frontend",
                target_service="backend",
                allowed_operations=["read", "write"],
                conditions={
                    "time_window": {"start": "08:00", "end": "18:00"},
                    "max_requests_per_minute": 100,
                },
            )
        )

        # Test allowed operation
        is_allowed = await proxy.evaluate_permission(
            source="frontend",
            target="backend",
            operation="read",
            context={"time": "10:00", "request_count": 50},
        )
        assert is_allowed is True

        # Test denied operation
        is_allowed = await proxy.evaluate_permission(
            source="frontend",
            target="backend",
            operation="delete",
            context={"time": "10:00", "request_count": 50},
        )
        assert is_allowed is False

        # Test time window restriction
        is_allowed = await proxy.evaluate_permission(
            source="frontend",
            target="backend",
            operation="read",
            context={"time": "20:00", "request_count": 50},
        )
        assert is_allowed is False

    @pytest.mark.asyncio
    async def test_session_risk_scoring(self, proxy):
        """Test session risk scoring."""
        session_id = "test-session-123"

        # Initial risk score
        risk_score = await proxy.calculate_session_risk(
            session_id=session_id,
            factors={
                "location_change": False,
                "unusual_time": False,
                "failed_attempts": 0,
                "anomaly_score": 0.1,
            },
        )
        assert isinstance(risk_score, SessionRiskScore)
        assert risk_score.score < 0.3  # Low risk

        # Increase risk factors
        risk_score = await proxy.calculate_session_risk(
            session_id=session_id,
            factors={
                "location_change": True,
                "unusual_time": True,
                "failed_attempts": 3,
                "anomaly_score": 0.7,
            },
        )
        assert risk_score.score > 0.7  # High risk
        assert risk_score.requires_reauthentication is True

    @pytest.mark.asyncio
    async def test_continuous_verification(self, proxy, mock_request):
        """Test continuous verification system."""
        session_id = "continuous-test-123"

        # Start continuous verification
        verification_task = asyncio.create_task(
            proxy.start_continuous_verification(session_id, mock_request)
        )

        # Simulate multiple verification checks
        for i in range(5):
            await asyncio.sleep(0.1)
            status = proxy.get_verification_status(session_id)
            assert status["active"] is True
            assert status["verification_count"] > i

        # Stop verification
        proxy.stop_continuous_verification(session_id)
        await verification_task

        final_status = proxy.get_verification_status(session_id)
        assert final_status["active"] is False

    @pytest.mark.asyncio
    async def test_performance_request_validation(self, proxy, mock_request):
        """Test request validation performance (<10ms overhead)."""
        with patch.object(
            proxy.mtls_manager,
            "validate_certificate",
            return_value=(True, "Valid"),
        ):
            # Warm up
            await proxy.validate_request(mock_request, "src", "dst", "op")

            # Measure validation time
            start_time = time.time()
            for _ in range(100):
                await proxy.validate_request(mock_request, "src", "dst", "op")
            end_time = time.time()

            avg_time_ms = ((end_time - start_time) / 100) * 1000
            assert avg_time_ms < 10, (
                f"Request validation took {avg_time_ms:.2f}ms on average"
            )


class TestServiceMeshConfig:
    """Test service mesh configuration generation."""

    @pytest.fixture
    def mesh_config(self):
        """Create ServiceMeshConfig instance for testing."""
        return ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO)

    def test_istio_config_generation(self, mesh_config):
        """Test Istio configuration generation."""
        # Add services
        mesh_config.add_service("frontend", namespace="default", port=8080)
        mesh_config.add_service("backend", namespace="default", port=9090)

        # Add traffic policy
        mesh_config.add_traffic_policy(
            TrafficPolicy(
                name="frontend-to-backend",
                source_service="frontend",
                destination_service="backend",
                tls_mode="ISTIO_MUTUAL",
                retry_attempts=3,
                timeout_seconds=30,
            )
        )

        # Generate Istio config
        istio_config = generate_istio_config(mesh_config)

        assert "apiVersion" in istio_config
        assert istio_config["apiVersion"] == "networking.istio.io/v1beta1"
        assert "kind" in istio_config
        assert len(istio_config["items"]) > 0

        # Verify DestinationRule for mTLS
        destination_rules = [
            item for item in istio_config["items"] if item["kind"] == "DestinationRule"
        ]
        assert len(destination_rules) > 0
        assert (
            destination_rules[0]["spec"]["trafficPolicy"]["tls"]["mode"]
            == "ISTIO_MUTUAL"
        )

    def test_linkerd_config_generation(self):
        """Test Linkerd configuration generation."""
        mesh_config = ServiceMeshConfig(mesh_type=ServiceMeshType.LINKERD)

        # Add services
        mesh_config.add_service("api-gateway", namespace="prod", port=443)
        mesh_config.add_service("auth-service", namespace="prod", port=8443)

        # Add service policy
        mesh_config.add_service_policy(
            ServicePolicy(
                source_service="api-gateway",
                target_service="auth-service",
                allowed_operations=["authenticate", "validate"],
                mtls_required=True,
            )
        )

        # Generate Linkerd config
        linkerd_config = generate_linkerd_config(mesh_config)

        assert "apiVersion" in linkerd_config
        assert linkerd_config["apiVersion"] == "policy.linkerd.io/v1beta1"
        assert "kind" in linkerd_config

        # Verify ServerAuthorization
        server_auth = [
            item
            for item in linkerd_config["items"]
            if item["kind"] == "ServerAuthorization"
        ]
        assert len(server_auth) > 0
        assert server_auth[0]["spec"]["client"]["meshTLS"]["identities"] is not None

    def test_traffic_encryption_policies(self, mesh_config):
        """Test traffic encryption policy configuration."""
        # Add encryption policy
        mesh_config.set_encryption_policy(
            service="database",
            min_tls_version="1.3",
            cipher_suites=[
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
            ],
            client_cert_required=True,
        )

        config = mesh_config.get_config()
        encryption = config["encryption_policies"]["database"]

        assert encryption["min_tls_version"] == "1.3"
        assert len(encryption["cipher_suites"]) == 2
        assert encryption["client_cert_required"] is True

    def test_service_routing_rules(self, mesh_config):
        """Test service routing and load balancing rules."""
        # Add routing rule
        mesh_config.add_routing_rule(
            service="api",
            version_weights={"v1": 80, "v2": 20},  # Canary deployment
            sticky_sessions=True,
            load_balancer_type="ROUND_ROBIN",
        )

        config = mesh_config.get_config()
        routing = config["routing_rules"]["api"]

        assert routing["version_weights"]["v1"] == 80
        assert routing["version_weights"]["v2"] == 20
        assert routing["sticky_sessions"] is True
        assert routing["load_balancer_type"] == "ROUND_ROBIN"

    def test_yaml_export(self, mesh_config, tmp_path):
        """Test exporting configuration to YAML."""
        # Configure mesh
        mesh_config.add_service("web", namespace="frontend", port=80)
        mesh_config.add_traffic_policy(
            TrafficPolicy(
                name="web-policy",
                source_service="web",
                destination_service="api",
                tls_mode="SIMPLE",
            )
        )

        # Export to YAML
        yaml_path = tmp_path / "service-mesh-config.yaml"
        mesh_config.export_yaml(str(yaml_path))

        assert yaml_path.exists()

        # Verify YAML content
        import yaml

        with open(yaml_path) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config["mesh_type"] == "istio"
        assert "web" in loaded_config["services"]


class TestZeroTrustIntegration:
    """Test integration of all zero-trust components."""

    @pytest.fixture
    def zero_trust_system(self, tmp_path):
        """Create integrated zero-trust system for testing."""
        mtls_manager = MTLSManager(
            ca_cert_path=str(tmp_path / "ca-cert.pem"),
            ca_key_path=str(tmp_path / "ca-key.pem"),
        )

        proxy_config = ProxyConfig(
            enable_mtls=True,
            enable_risk_scoring=True,
            mtls_manager=mtls_manager,
        )
        proxy = IdentityAwareProxy(proxy_config)

        mesh_config = ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO)

        return {
            "mtls_manager": mtls_manager,
            "proxy": proxy,
            "mesh_config": mesh_config,
        }

    @pytest.mark.asyncio
    async def test_end_to_end_service_communication(self, zero_trust_system):
        """Test end-to-end service communication with zero-trust."""
        mtls = zero_trust_system["mtls_manager"]
        proxy = zero_trust_system["proxy"]
        mesh = zero_trust_system["mesh_config"]

        # Generate certificates for services
        frontend_cert = mtls.generate_service_certificate("frontend")
        mtls.generate_service_certificate("backend")

        # Configure service mesh
        mesh.add_service("frontend", namespace="default", port=8080)
        mesh.add_service("backend", namespace="default", port=9090)
        mesh.add_traffic_policy(
            TrafficPolicy(
                name="frontend-backend",
                source_service="frontend",
                destination_service="backend",
                tls_mode="ISTIO_MUTUAL",
            )
        )

        # Setup proxy policies
        proxy.add_service_policy(
            ServicePolicy(
                source_service="frontend",
                target_service="backend",
                allowed_operations=["read", "write"],
                mtls_required=True,
            )
        )

        # Simulate service request
        mock_request = MagicMock()
        mock_request.headers = {"X-Client-Certificate": frontend_cert.certificate}
        mock_request.method = "POST"
        mock_request.url.path = "/api/data"
        mock_request.client.host = "10.0.0.1"

        # Validate request through proxy
        context = await proxy.validate_request(
            request=mock_request,
            source_service="frontend",
            target_service="backend",
            operation="write",
        )

        assert context.is_valid is True
        assert context.mtls_verified is True

    @pytest.mark.asyncio
    async def test_certificate_rotation_with_zero_downtime(self, zero_trust_system):
        """Test certificate rotation without service interruption."""
        mtls = zero_trust_system["mtls_manager"]
        proxy = zero_trust_system["proxy"]

        service_name = "rotation-service"
        original_cert = mtls.generate_service_certificate(service_name)

        # Set aggressive rotation policy
        policy = CertificateRotationPolicy(
            strategy=RotationStrategy.TIME_BASED,
            rotation_interval_days=1,
            overlap_period_days=0.5,
            auto_rotate=True,
        )
        mtls.set_rotation_policy(service_name, policy)

        # Start continuous requests
        request_results = []

        async def make_requests():
            for i in range(20):
                mock_request = MagicMock()
                mock_request.headers = {
                    "X-Client-Certificate": original_cert.certificate
                }

                try:
                    context = await proxy.validate_request(
                        mock_request, service_name, "target", "read"
                    )
                    request_results.append((i, context.is_valid))
                except Exception:
                    request_results.append((i, False))

                await asyncio.sleep(0.1)

        # Start requests
        request_task = asyncio.create_task(make_requests())

        # Trigger rotation midway
        await asyncio.sleep(1)
        with patch.object(mtls, "_should_rotate_certificate", return_value=True):
            mtls.rotate_certificate(service_name)

        # Wait for requests to complete
        await request_task

        # Verify no failed requests during rotation
        failed_requests = [r for r in request_results if not r[1]]
        assert len(failed_requests) == 0, (
            f"Failed requests during rotation: {failed_requests}"
        )

    def test_performance_full_stack(self, zero_trust_system):
        """Test full zero-trust stack performance."""
        mtls = zero_trust_system["mtls_manager"]
        mesh = zero_trust_system["mesh_config"]

        # Measure full configuration generation
        start_time = time.time()

        # Generate certificates for 10 services
        for i in range(10):
            mtls.generate_service_certificate(f"service-{i}")

        # Configure service mesh for all services
        for i in range(10):
            mesh.add_service(f"service-{i}", namespace="default", port=8000 + i)

        # Add policies between services
        for i in range(9):
            mesh.add_traffic_policy(
                TrafficPolicy(
                    name=f"policy-{i}",
                    source_service=f"service-{i}",
                    destination_service=f"service-{i + 1}",
                    tls_mode="ISTIO_MUTUAL",
                )
            )

        # Generate final configuration
        istio_config = generate_istio_config(mesh)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        # Should complete within reasonable time for 10 services
        assert total_time_ms < 1000, (
            f"Full stack configuration took {total_time_ms:.2f}ms"
        )
        assert len(istio_config["items"]) >= 19  # At least 10 services + 9 policies
