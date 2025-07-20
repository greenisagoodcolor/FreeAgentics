"""Test coverage for security module __init__ files."""

import pytest


def test_security_init_import():
    """Test that security module can be imported."""
    import security

    # Module should have docstring
    assert security.__doc__ is not None
    assert "Security module" in security.__doc__


def test_security_encryption_init_import():
    """Test that security.encryption module imports correctly."""
    from security import encryption

    # Check that all expected exports are available
    expected_exports = [
        "FieldEncryptor",
        "AWSKMSProvider",
        "HashiCorpVaultProvider",
        "TransparentFieldEncryptor",
        "create_field_encryptor",
        "KyberKEM",
        "DilithiumSigner",
        "HomomorphicEncryption",
        "QuantumResistantCrypto",
        "EncryptionAtRest",
        "QuantumKeyPair",
        "EncryptedValue",
    ]

    for export in expected_exports:
        assert hasattr(encryption, export), f"Missing export: {export}"

    # Check __all__ is properly defined
    assert hasattr(encryption, "__all__")
    assert isinstance(encryption.__all__, list)
    assert len(encryption.__all__) == len(expected_exports)


def test_security_soar_init_import():
    """Test that security.soar module can be imported."""
    from security import soar

    # Module should have docstring
    assert soar.__doc__ is not None

    # Check expected exports
    expected_exports = [
        "IncidentManager",
        "PlaybookEngine",
        "IncidentSeverity",
        "IncidentStatus",
        "IncidentType",
    ]

    for export in expected_exports:
        assert hasattr(soar, export), f"Missing export: {export}"


def test_security_zero_trust_init_import():
    """Test that security.zero_trust module can be imported."""
    from security import zero_trust

    # Module should have docstring
    assert zero_trust.__doc__ is not None
    assert "Zero-Trust Network Architecture" in zero_trust.__doc__

    # Check expected exports
    expected_exports = [
        "IdentityAwareProxy",
        "MTLSManager",
        "ServiceMeshConfig",
        "ProxyConfig",
        "RequestContext",
        "ServicePolicy",
        "SessionRiskScore",
        "CertificateInfo",
        "CertificateRotationPolicy",
        "RotationStrategy",
        "ServiceMeshType",
        "TrafficPolicy",
        "generate_istio_config",
        "generate_linkerd_config",
    ]

    for export in expected_exports:
        assert hasattr(zero_trust, export), f"Missing export: {export}"

    # Check version info
    assert hasattr(zero_trust, "__version__")
    assert zero_trust.__version__ == "1.0.0"
    assert hasattr(zero_trust, "__author__")
    assert zero_trust.__author__ == "Zero Trust Security Team"
