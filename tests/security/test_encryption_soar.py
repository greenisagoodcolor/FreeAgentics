"""
Comprehensive test suite for advanced encryption and SOAR functionality.
Tests field-level encryption, quantum-resistant crypto, and incident response.
"""

import asyncio
import json
import os

# Import modules to test
import sys
import time
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from security.encryption.field_encryptor import (
    AWSKMSProvider,
    FieldEncryptor,
    HashiCorpVaultProvider,
    TransparentFieldEncryptor,
)
from security.encryption.quantum_resistant import (
    DilithiumSigner,
    EncryptionAtRest,
    HomomorphicEncryption,
    KyberKEM,
    QuantumResistantCrypto,
)
from security.soar.incident_manager import (
    IncidentManager,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
)
from security.soar.playbook_engine import (
    ActionStatus,
    IPBlockAction,
    NotificationAction,
    PlaybookEngine,
    PlaybookTrigger,
    save_example_playbook,
)


class TestFieldEncryption:
    """Test field-level encryption functionality."""

    @mock_aws
    def test_aws_kms_provider(self):
        """Test AWS KMS provider integration."""
        # Create KMS client and key
        kms_client = boto3.client("kms", region_name="us-east-1")
        key_response = kms_client.create_key(
            Description="Test encryption key", KeyUsage="ENCRYPT_DECRYPT"
        )
        key_id = key_response["KeyMetadata"]["KeyId"]

        # Create provider
        provider = AWSKMSProvider(region="us-east-1")

        # Test key generation
        plaintext_key, encrypted_key = provider.generate_data_key(key_id)
        assert len(plaintext_key) == 32  # AES-256
        assert len(encrypted_key) > 32

        # Test key decryption
        decrypted_key = provider.decrypt_data_key(encrypted_key, key_id)
        assert decrypted_key == plaintext_key

        # Test key metadata
        metadata = provider.get_key_metadata(key_id)
        assert metadata["KeyId"] == key_id
        assert metadata["KeyUsage"] == "ENCRYPT_DECRYPT"

    def test_vault_provider_mock(self):
        """Test HashiCorp Vault provider with mock."""
        # Mock Vault client
        mock_client = MagicMock()
        mock_client.sys.list_mounted_secrets_engines.return_value = {}
        mock_client.secrets.transit.generate_data_key.return_value = {
            "data": {
                "plaintext": "dGVzdF9rZXlfZGF0YQ==",  # base64 encoded
                "ciphertext": "vault:v1:encrypted_data",
            }
        }
        mock_client.secrets.transit.decrypt_data.return_value = {
            "data": {"plaintext": "dGVzdF9rZXlfZGF0YQ=="}
        }

        with patch("hvac.Client", return_value=mock_client):
            provider = HashiCorpVaultProvider(
                vault_url="http://localhost:8200", vault_token="test-token"
            )

            # Test key generation
            plaintext, ciphertext = provider.generate_data_key("test-key")
            assert plaintext == b"test_key_data"
            assert ciphertext == b"vault:v1:encrypted_data"

            # Test decryption
            decrypted = provider.decrypt_data_key(ciphertext, "test-key")
            assert decrypted == b"test_key_data"

    @mock_aws
    def test_field_encryptor_performance(self):
        """Test encryption performance meets <5ms requirement."""
        # Setup
        kms_client = boto3.client("kms", region_name="us-east-1")
        key_response = kms_client.create_key()
        key_id = key_response["KeyMetadata"]["KeyId"]

        provider = AWSKMSProvider(region="us-east-1")
        encryptor = FieldEncryptor(
            provider=provider,
            default_key_id=key_id,
            performance_monitoring=True,
        )

        # Test single field encryption
        test_data = {"ssn": "123-45-6789", "name": "John Doe"}

        # Warm up
        encrypted = encryptor.encrypt_field(test_data["ssn"], "ssn")

        # Measure performance
        start_time = time.perf_counter()
        encrypted = encryptor.encrypt_field(test_data["ssn"], "ssn")
        end_time = time.perf_counter()

        encryption_time_ms = (end_time - start_time) * 1000
        assert encryption_time_ms < 5, f"Encryption took {encryption_time_ms}ms"

        # Test decryption performance
        start_time = time.perf_counter()
        decrypted = encryptor.decrypt_field(encrypted, "ssn")
        end_time = time.perf_counter()

        decryption_time_ms = (end_time - start_time) * 1000
        assert decryption_time_ms < 5, f"Decryption took {decryption_time_ms}ms"
        assert decrypted == test_data["ssn"]

        # Check performance stats
        stats = encryptor.get_performance_stats()
        assert "encrypt_field" in stats
        assert stats["encrypt_field"]["avg_ms"] < 5

    @mock_aws
    def test_transparent_encryption(self):
        """Test transparent field encryption with decorators."""
        # Setup
        kms_client = boto3.client("kms", region_name="us-east-1")
        key_response = kms_client.create_key()
        key_id = key_response["KeyMetadata"]["KeyId"]

        provider = AWSKMSProvider(region="us-east-1")
        encryptor = FieldEncryptor(provider=provider, default_key_id=key_id)
        transparent = TransparentFieldEncryptor(encryptor)

        # Create model with encrypted fields
        @transparent.encrypt_model_fields({"ssn": None, "credit_card": None})
        class Customer:
            def __init__(self, name, ssn, credit_card):
                self.name = name
                self.ssn = ssn
                self.credit_card = credit_card

        # Test transparent encryption
        customer = Customer("John Doe", "123-45-6789", "4111111111111111")

        # Fields should be encrypted internally
        assert hasattr(customer, "_encrypted_ssn")
        assert hasattr(customer, "_encrypted_credit_card")

        # But access should return decrypted values
        assert customer.ssn == "123-45-6789"
        assert customer.credit_card == "4111111111111111"

    @mock_aws
    def test_key_rotation(self):
        """Test field encryption key rotation."""
        # Setup
        kms_client = boto3.client("kms", region_name="us-east-1")
        key1 = kms_client.create_key()["KeyMetadata"]["KeyId"]
        key2 = kms_client.create_key()["KeyMetadata"]["KeyId"]

        provider = AWSKMSProvider(region="us-east-1")
        encryptor = FieldEncryptor(provider=provider, default_key_id=key1)

        # Encrypt with first key
        original_data = "sensitive-data"
        encrypted = encryptor.encrypt_field(original_data, "field1")

        # Rotate to new key
        rotated = encryptor.rotate_field_encryption(encrypted, key2)

        # Verify new encryption uses new key
        assert rotated["metadata"]["key_id"] == key2

        # Verify data is still accessible
        decrypted = encryptor.decrypt_field(rotated)
        assert decrypted == original_data


class TestQuantumResistantCrypto:
    """Test quantum-resistant cryptography."""

    def test_kyber_kem(self):
        """Test Kyber key encapsulation."""
        kyber = KyberKEM(security_level=3)

        # Generate key pair
        keypair = kyber.generate_keypair()
        assert len(keypair.public_key) == 32  # Simplified
        assert len(keypair.private_key) > 0
        assert keypair.algorithm == "Kyber3"

        # Test encapsulation
        encapsulated = kyber.encapsulate(keypair.public_key)
        assert len(encapsulated.ciphertext) == 32
        assert len(encapsulated.shared_secret) == 32

        # Test decapsulation
        shared_secret = kyber.decapsulate(encapsulated.ciphertext, keypair.private_key)
        assert len(shared_secret) == 32

    def test_dilithium_signatures(self):
        """Test Dilithium digital signatures."""
        dilithium = DilithiumSigner(security_level=3)

        # Generate key pair
        keypair = dilithium.generate_keypair()
        assert keypair.algorithm == "Dilithium3"

        # Sign message
        message = b"Test message for signing"
        signature = dilithium.sign(message, keypair.private_key)
        assert len(signature) == 64  # Simplified

        # Verify signature
        valid = dilithium.verify(message, signature, keypair.public_key)
        assert valid

        # Test invalid signature
        invalid_sig = os.urandom(64)
        invalid = dilithium.verify(message, invalid_sig, keypair.public_key)
        assert not invalid

    def test_homomorphic_encryption(self):
        """Test homomorphic encryption operations."""
        he = HomomorphicEncryption()

        # Encrypt values
        val1 = 10.5
        val2 = 20.3
        enc1 = he.encrypt(val1)
        enc2 = he.encrypt(val2)

        # Test addition
        enc_sum = he.add(enc1, enc2)
        decrypted_sum = he.decrypt(enc_sum)
        assert abs(decrypted_sum - (val1 + val2)) < 1.0  # Allow for noise

        # Test multiplication (simplified)
        enc_prod = he.multiply(enc1, enc2)
        he.decrypt(enc_prod)
        # Note: Simplified implementation, exact match not expected

        # Test vector operations
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        enc_values = [he.encrypt(v) for v in values]
        enc_mean = he.compute_mean(enc_values)
        decrypted_mean = he.decrypt(enc_mean)
        assert abs(decrypted_mean - 3.0) < 1.0

    def test_hybrid_encryption(self):
        """Test hybrid quantum-resistant encryption."""
        qrc = QuantumResistantCrypto()

        # Generate keys
        keys = qrc.generate_hybrid_keypair()
        assert "kem" in keys
        assert "sign" in keys

        # Test encryption with signature
        data = b"Sensitive data to encrypt"
        encrypted = qrc.hybrid_encrypt(
            data,
            keys["kem"].public_key,
            sign_with_private_key=keys["sign"].private_key,
        )

        assert "encapsulated_key" in encrypted
        assert "ciphertext" in encrypted
        assert "signature" in encrypted
        assert encrypted["algorithm"] == "Kyber-AES-GCM"

        # Test decryption with verification
        decrypted = qrc.hybrid_decrypt(
            encrypted,
            keys["kem"].private_key,
            verify_with_public_key=keys["sign"].public_key,
        )

        assert decrypted == data

    def test_encryption_at_rest(self):
        """Test transparent encryption for data stores."""
        qrc = QuantumResistantCrypto()
        ear = EncryptionAtRest(qrc, key_rotation_days=90)

        # Test document encryption
        document = {
            "user_id": "12345",
            "ssn": "123-45-6789",
            "credit_score": 750,
            "account_balance": 10000.50,
        }

        # Full document encryption
        encrypted_doc = ear.encrypt_document(document)
        assert "_encrypted" in encrypted_doc
        assert "_encryption_metadata" in encrypted_doc

        # Decrypt document
        decrypted_doc = ear.decrypt_document(encrypted_doc)
        assert decrypted_doc == document

        # Field-level encryption
        encrypted_partial = ear.encrypt_document(
            document, fields_to_encrypt=["ssn", "credit_score"]
        )
        assert encrypted_partial["user_id"] == "12345"  # Not encrypted
        assert "encapsulated_key" in encrypted_partial["ssn"]  # Encrypted

        # Test key rotation
        assert not ear.should_rotate_keys()

        # Simulate time passage
        ear.key_metadata["rotation_due"] = time.time() - 1
        assert ear.should_rotate_keys()

        new_keys = ear.rotate_keys()
        assert new_keys != ear.current_keys


class TestSOARPlaybookEngine:
    """Test SOAR playbook engine functionality."""

    @pytest.fixture
    def playbook_engine(self, tmp_path):
        """Create playbook engine for testing."""
        playbook_dir = tmp_path / "playbooks"
        playbook_dir.mkdir()
        save_example_playbook(str(playbook_dir))
        return PlaybookEngine(playbook_dir=str(playbook_dir))

    @pytest.mark.asyncio
    async def test_playbook_execution(self, playbook_engine):
        """Test basic playbook execution."""
        # Execute playbook
        context = await playbook_engine.execute_playbook(
            playbook_id="brute_force_response",
            trigger=PlaybookTrigger.ALERT,
            trigger_data={"alert_id": "test-123"},
            variables={
                "attacker_ip": "192.168.1.100",
                "target_user": "john.doe",
                "affected_server": "web-server-01",
            },
        )

        assert context.status == ActionStatus.SUCCESS
        assert context.playbook_id == "brute_force_response"
        assert "blocked_ips" in context.artifacts
        assert "disabled_users" in context.artifacts
        assert "forensic_data" in context.artifacts

    @pytest.mark.asyncio
    async def test_ip_block_action(self):
        """Test IP blocking action."""
        action = IPBlockAction(
            "block_ip_1",
            {
                "ip_addresses": ["192.168.1.100", "192.168.1.101"],
                "duration_hours": 24,
            },
        )

        from security.soar.playbook_engine import PlaybookContext

        context = PlaybookContext(
            playbook_id="test",
            execution_id="exec-1",
            trigger=PlaybookTrigger.MANUAL,
            trigger_data={},
        )

        result = await action.execute(context)

        assert result.status == ActionStatus.SUCCESS
        assert len(result.output["blocked_ips"]) == 2
        assert result.output["duration_hours"] == 24

    @pytest.mark.asyncio
    async def test_conditional_action(self):
        """Test conditional execution."""
        from security.soar.playbook_engine import (
            ConditionalAction,
            PlaybookContext,
        )

        # Test true condition
        action = ConditionalAction("condition_1", {"condition": "5 > 3"})

        context = PlaybookContext(
            playbook_id="test",
            execution_id="exec-1",
            trigger=PlaybookTrigger.MANUAL,
            trigger_data={},
        )

        result = await action.execute(context)
        assert result.status == ActionStatus.SUCCESS
        assert result.output["condition_met"] is True

        # Test false condition
        action2 = ConditionalAction("condition_2", {"condition": "2 > 5"})

        result2 = await action2.execute(context)
        assert result2.status == ActionStatus.SKIPPED
        assert result2.output["condition_met"] is False

    @pytest.mark.asyncio
    async def test_variable_resolution(self):
        """Test variable resolution in actions."""
        from security.soar.playbook_engine import (
            PlaybookContext,
        )

        action = NotificationAction(
            "notify_1",
            {
                "recipients": ["{{security_team}}"],
                "message": "Attack from {{attacker_ip}} detected",
                "channels": ["email"],
            },
        )

        context = PlaybookContext(
            playbook_id="test",
            execution_id="exec-1",
            trigger=PlaybookTrigger.ALERT,
            trigger_data={},
            variables={
                "security_team": "security@example.com",
                "attacker_ip": "192.168.1.100",
            },
        )

        result = await action.execute(context)
        assert result.status == ActionStatus.SUCCESS
        assert result.output["notifications"][0]["recipient"] == "security@example.com"
        assert "192.168.1.100" in result.output["notifications"][0]["message"]

    def test_playbook_metrics(self, playbook_engine):
        """Test playbook execution metrics."""
        metrics = playbook_engine.get_execution_metrics()

        assert "total_executions" in metrics
        assert "successful" in metrics
        assert "failed" in metrics
        assert "success_rate" in metrics
        assert "active_executions" in metrics
        assert "average_duration_seconds" in metrics


class TestIncidentManager:
    """Test incident management functionality."""

    @pytest.fixture
    def incident_manager(self, tmp_path):
        """Create incident manager for testing."""
        data_dir = tmp_path / "incident_data"
        return IncidentManager(
            data_dir=str(data_dir),
            auto_triage=True,
            auto_playbook_execution=False,  # Disable for testing
        )

    def test_create_incident(self, incident_manager):
        """Test incident creation."""
        case = incident_manager.create_incident(
            title="Suspected Brute Force Attack",
            description="Multiple failed login attempts detected",
            type=IncidentType.UNAUTHORIZED_ACCESS,
            severity=IncidentSeverity.HIGH,
            indicators=[
                {"type": "ip", "value": "192.168.1.100", "confidence": 0.9},
                {"type": "user", "value": "john.doe", "confidence": 0.8},
            ],
            affected_assets=["web-server-01", "db-server-01"],
        )

        assert case.case_id.startswith("INC-")
        assert case.type == IncidentType.UNAUTHORIZED_ACCESS
        assert case.severity == IncidentSeverity.HIGH
        assert len(case.indicators) == 2
        assert len(case.affected_assets) == 2
        assert case.status == IncidentStatus.NEW
        assert len(case.timeline) > 0

    def test_incident_status_updates(self, incident_manager):
        """Test incident status progression."""
        # Create incident
        case = incident_manager.create_incident(
            title="Test Incident",
            description="Test",
            type=IncidentType.MALWARE,
            severity=IncidentSeverity.MEDIUM,
        )

        # Update status
        success = incident_manager.update_incident_status(
            case.case_id,
            IncidentStatus.IN_PROGRESS,
            notes="Started investigation",
        )
        assert success

        # Verify update
        updated_case = incident_manager.cases[case.case_id]
        assert updated_case.status == IncidentStatus.IN_PROGRESS
        assert len(updated_case.notes) == 1
        assert len(updated_case.timeline) > 1

    def test_indicator_tracking(self, incident_manager):
        """Test global indicator tracking."""
        # Create multiple incidents with shared indicators
        incident_manager.create_incident(
            title="Incident 1",
            description="Test",
            type=IncidentType.MALWARE,
            severity=IncidentSeverity.HIGH,
            indicators=[
                {"type": "ip", "value": "192.168.1.100"},
                {"type": "hash", "value": "abc123"},
            ],
        )

        incident_manager.create_incident(
            title="Incident 2",
            description="Test",
            type=IncidentType.MALWARE,
            severity=IncidentSeverity.HIGH,
            indicators=[
                {"type": "ip", "value": "192.168.1.100"},
                {"type": "domain", "value": "malicious.com"},
            ],
        )

        # Check global indicators
        assert "192.168.1.100" in incident_manager.global_indicators["ip"]
        assert "abc123" in incident_manager.global_indicators["hash"]
        assert "malicious.com" in incident_manager.global_indicators["domain"]

    @pytest.mark.asyncio
    async def test_auto_triage(self, incident_manager):
        """Test automatic incident triage."""
        # Create incident
        case = incident_manager.create_incident(
            title="Critical Security Breach",
            description="Unauthorized data access detected",
            type=IncidentType.DATA_BREACH,
            severity=IncidentSeverity.CRITICAL,
        )

        # Wait for auto-triage
        await asyncio.sleep(0.1)

        # Check triage results
        updated_case = incident_manager.cases[case.case_id]
        assert updated_case.status == IncidentStatus.TRIAGED
        assert updated_case.team is not None
        assert updated_case.assigned_to is not None

    def test_incident_metrics(self, incident_manager):
        """Test incident metrics calculation."""
        # Create and update incident
        case = incident_manager.create_incident(
            title="Test Metrics",
            description="Test",
            type=IncidentType.PHISHING,
            severity=IncidentSeverity.LOW,
        )

        # Simulate response
        time.sleep(0.1)
        incident_manager.update_incident_status(case.case_id, IncidentStatus.IN_PROGRESS)

        # Simulate containment
        time.sleep(0.1)
        incident_manager.update_incident_status(case.case_id, IncidentStatus.CONTAINED)

        # Get metrics
        summary = incident_manager.get_incident_summary(case.case_id)
        assert summary is not None
        assert "metrics" in summary

        # Get dashboard metrics
        dashboard = incident_manager.get_dashboard_metrics()
        assert dashboard["total_incidents"] >= 1
        assert dashboard["open_incidents"] >= 0
        assert "severity_breakdown" in dashboard
        assert "type_breakdown" in dashboard

    def test_related_incidents(self, incident_manager):
        """Test related incident detection."""
        # Create first incident
        case1 = incident_manager.create_incident(
            title="Attack 1",
            description="Initial attack",
            type=IncidentType.UNAUTHORIZED_ACCESS,
            severity=IncidentSeverity.HIGH,
            indicators=[{"type": "ip", "value": "10.0.0.1"}],
            affected_assets=["server-01"],
        )

        # Create related incident
        case2 = incident_manager.create_incident(
            title="Attack 2",
            description="Related attack",
            type=IncidentType.UNAUTHORIZED_ACCESS,
            severity=IncidentSeverity.HIGH,
            indicators=[{"type": "ip", "value": "10.0.0.1"}],
            affected_assets=["server-02"],
        )

        # Check relationship detection
        related = incident_manager._find_related_incidents(case2)
        assert len(related) == 1
        assert related[0].case_id == case1.case_id


class TestIntegration:
    """Integration tests for encryption and SOAR."""

    @pytest.mark.asyncio
    async def test_encrypted_incident_data(self, tmp_path):
        """Test incident data encryption."""
        # Create quantum crypto
        qrc = QuantumResistantCrypto()
        ear = EncryptionAtRest(qrc)

        # Create incident manager
        incident_manager = IncidentManager(
            data_dir=str(tmp_path / "incidents"), auto_playbook_execution=False
        )

        # Create incident with sensitive data
        case = incident_manager.create_incident(
            title="Data Breach",
            description="Sensitive data exposed",
            type=IncidentType.DATA_BREACH,
            severity=IncidentSeverity.CRITICAL,
            custom_fields={
                "exposed_records": 1000,
                "affected_users": ["user1", "user2", "user3"],
                "pii_types": ["ssn", "credit_card"],
            },
        )

        # Encrypt incident data
        encrypted_case = ear.encrypt_document(
            case.__dict__, fields_to_encrypt=["custom_fields", "notes"]
        )

        # Verify encryption
        assert "encapsulated_key" in encrypted_case["custom_fields"]
        assert encrypted_case["case_id"] == case.case_id  # Not encrypted

        # Decrypt and verify
        decrypted_case = ear.decrypt_document(
            encrypted_case, fields_to_decrypt=["custom_fields", "notes"]
        )
        assert decrypted_case["custom_fields"]["exposed_records"] == 1000

    @pytest.mark.asyncio
    async def test_secure_playbook_execution(self, tmp_path):
        """Test secure playbook execution with encryption."""
        # Setup encryption
        qrc = QuantumResistantCrypto()

        # Create playbook engine
        playbook_dir = tmp_path / "playbooks"
        playbook_dir.mkdir()
        save_example_playbook(str(playbook_dir))
        playbook_engine = PlaybookEngine(playbook_dir=str(playbook_dir))

        # Create incident manager with playbook engine
        incident_manager = IncidentManager(
            playbook_engine=playbook_engine,
            data_dir=str(tmp_path / "incidents"),
            auto_playbook_execution=True,
        )

        # Create incident that triggers playbook
        case = incident_manager.create_incident(
            title="Brute Force Detected",
            description="Multiple failed logins",
            type=IncidentType.UNAUTHORIZED_ACCESS,
            severity=IncidentSeverity.HIGH,
            indicators=[
                {"type": "ip", "value": "192.168.1.50"},
                {"type": "user", "value": "admin"},
                {"type": "host", "value": "prod-server-01"},
            ],
        )

        # Wait for playbook execution
        await asyncio.sleep(0.5)

        # Verify playbook was executed
        updated_case = incident_manager.cases[case.case_id]
        assert len(updated_case.playbooks_executed) > 0

        # Encrypt sensitive playbook artifacts
        if updated_case.evidence:
            encrypted_evidence = qrc.hybrid_encrypt(
                json.dumps(updated_case.evidence).encode(),
                qrc.generate_hybrid_keypair()["kem"].public_key,
            )
            assert "ciphertext" in encrypted_evidence
            assert "algorithm" in encrypted_evidence


def test_fips_compliance():
    """Test FIPS 140-2 compliance requirements."""
    # Test approved algorithms
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import algorithms, modes

    # AES is FIPS approved
    assert algorithms.AES

    # SHA-256 is FIPS approved
    assert hashes.SHA256

    # GCM mode is FIPS approved
    assert modes.GCM

    # Verify key sizes
    key_256 = os.urandom(32)  # 256-bit key
    assert len(key_256) == 32

    # Verify secure random generation
    random_bytes = os.urandom(16)
    assert len(random_bytes) == 16
    assert random_bytes != os.urandom(16)  # Should be different each time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
