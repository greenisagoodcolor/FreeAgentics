#!/usr/bin/env python3
"""
Basic test for encryption and SOAR functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_field_encryption():
    """Test basic field encryption functionality."""
    print("Testing field encryption...")

    from security.encryption.field_encryptor import FieldEncryptor

    # Mock provider for testing
    class MockProvider:
        def generate_data_key(self, key_id):
            return b"test_plaintext_key_32_bytes_long", b"encrypted_key_data"

        def decrypt_data_key(self, encrypted_key, key_id):
            return b"test_plaintext_key_32_bytes_long"

    # Create encryptor
    encryptor = FieldEncryptor(
        provider=MockProvider(),
        default_key_id="test-key",
        performance_monitoring=True,
    )

    # Test encryption/decryption
    test_data = "sensitive-information"
    encrypted = encryptor.encrypt_field(test_data, "test_field")
    decrypted = encryptor.decrypt_field(encrypted, "test_field")

    assert decrypted == test_data
    print("✓ Field encryption works correctly")

    # Test performance
    stats = encryptor.get_performance_stats()
    assert "encrypt_field" in stats
    print("✓ Performance monitoring works")


def test_quantum_resistant():
    """Test quantum-resistant cryptography."""
    print("\nTesting quantum-resistant crypto...")

    from security.encryption.quantum_resistant import (
        DilithiumSigner,
        KyberKEM,
        QuantumResistantCrypto,
    )

    # Test Kyber KEM
    kyber = KyberKEM()
    keypair = kyber.generate_keypair()
    assert keypair.algorithm == "Kyber3"

    encapsulated = kyber.encapsulate(keypair.public_key)
    shared_secret = kyber.decapsulate(encapsulated.ciphertext, keypair.private_key)
    assert len(shared_secret) == 32
    print("✓ Kyber KEM works")

    # Test Dilithium signatures
    dilithium = DilithiumSigner()
    sign_keypair = dilithium.generate_keypair()
    message = b"test message"
    signature = dilithium.sign(message, sign_keypair.private_key)
    valid = dilithium.verify(message, signature, sign_keypair.public_key)
    assert valid
    print("✓ Dilithium signatures work")

    # Test hybrid encryption
    qrc = QuantumResistantCrypto()
    keys = qrc.generate_hybrid_keypair()

    data = b"secret data"
    encrypted = qrc.hybrid_encrypt(data, keys["kem"].public_key)
    decrypted = qrc.hybrid_decrypt(encrypted, keys["kem"].private_key)
    assert decrypted == data
    print("✓ Hybrid encryption works")


async def test_soar_playbook():
    """Test SOAR playbook functionality."""
    print("\nTesting SOAR playbook engine...")

    from security.soar.playbook_engine import (
        IPBlockAction,
        PlaybookContext,
        PlaybookEngine,
        PlaybookTrigger,
    )

    # Test IP block action
    action = IPBlockAction(
        "test_block",
        {"ip_addresses": ["192.168.1.1", "192.168.1.2"], "duration_hours": 24},
    )

    context = PlaybookContext(
        playbook_id="test",
        execution_id="exec-1",
        trigger=PlaybookTrigger.MANUAL,
        trigger_data={},
    )

    result = await action.execute(context)
    assert result.output["blocked_ips"] == ["192.168.1.1", "192.168.1.2"]
    print("✓ IP block action works")

    # Test playbook engine
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PlaybookEngine(playbook_dir=tmpdir)
        metrics = engine.get_execution_metrics()
        assert "total_executions" in metrics
        print("✓ Playbook engine works")


def test_incident_manager():
    """Test incident management."""
    print("\nTesting incident manager...")

    import tempfile

    from security.soar.incident_manager import (
        IncidentManager,
        IncidentSeverity,
        IncidentType,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncidentManager(data_dir=tmpdir, auto_triage=False, auto_playbook_execution=False)

        # Create incident
        case = manager.create_incident(
            title="Test Incident",
            description="Test description",
            type=IncidentType.MALWARE,
            severity=IncidentSeverity.HIGH,
            indicators=[{"type": "ip", "value": "192.168.1.100", "confidence": 0.9}],
        )

        assert case.case_id.startswith("INC-")
        assert len(case.indicators) == 1
        print("✓ Incident creation works")

        # Test metrics
        metrics = manager.get_dashboard_metrics()
        assert metrics["total_incidents"] == 1
        print("✓ Incident metrics work")


def test_encryption_at_rest():
    """Test encryption at rest."""
    print("\nTesting encryption at rest...")

    from security.encryption.quantum_resistant import (
        EncryptionAtRest,
        QuantumResistantCrypto,
    )

    qrc = QuantumResistantCrypto()
    ear = EncryptionAtRest(qrc)

    # Test document encryption
    document = {
        "id": "doc-123",
        "content": "sensitive content",
        "metadata": {"author": "user"},
    }

    encrypted = ear.encrypt_document(document, ["content"])
    decrypted = ear.decrypt_document(encrypted, ["content"])

    assert decrypted["content"] == "sensitive content"
    assert decrypted["id"] == "doc-123"  # Not encrypted
    print("✓ Encryption at rest works")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Advanced Encryption and SOAR Features")
    print("=" * 50)

    try:
        test_field_encryption()
        test_quantum_resistant()
        await test_soar_playbook()
        test_incident_manager()
        test_encryption_at_rest()

        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
