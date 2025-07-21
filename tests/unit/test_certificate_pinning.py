"""
Unit tests for enhanced certificate pinning implementation.

Tests for Task #14.5 - Certificate Pinning for Mobile Applications
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, mock_open, patch

import pytest

from auth.certificate_pinning import (
    CertificateValidator,
    MobileCertificatePinner,
    PinConfiguration,
    mobile_cert_pinner,
    setup_production_pins,
)


class TestPinConfiguration:
    """Test PinConfiguration dataclass."""

    def test_default_configuration(self):
        """Test default pin configuration."""
        config = PinConfiguration()

        assert config.primary_pins == []
        assert config.backup_pins == []
        assert config.max_age == 5184000  # 60 days
        assert config.include_subdomains is True
        assert config.enforce_pinning is True
        assert config.allow_fallback is True
        assert config.mobile_specific is False

    def test_custom_configuration(self):
        """Test custom pin configuration."""
        config = PinConfiguration(
            primary_pins=["sha256-primary"],
            backup_pins=["sha256-backup"],
            max_age=2592000,  # 30 days
            mobile_specific=True,
            report_uri="/pin-violations",
        )

        assert config.primary_pins == ["sha256-primary"]
        assert config.backup_pins == ["sha256-backup"]
        assert config.max_age == 2592000
        assert config.mobile_specific is True
        assert config.report_uri == "/pin-violations"

    def test_mobile_user_agents(self):
        """Test mobile user agent configuration."""
        config = PinConfiguration()

        expected_agents = [
            "FreeAgentics-iOS",
            "FreeAgentics-Android",
            "FreeAgentics-Mobile",
        ]
        assert config.mobile_user_agents == expected_agents

    def test_emergency_bypass(self):
        """Test emergency bypass configuration."""
        config = PinConfiguration(
            emergency_bypass=True,
            emergency_bypass_until=datetime.now() + timedelta(hours=24),
        )

        assert config.emergency_bypass is True
        assert config.emergency_bypass_until is not None


class TestCertificateValidator:
    """Test certificate validation utilities."""

    def test_validate_pin_format_valid(self):
        """Test validation of correct pin format."""
        valid_pin = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        assert CertificateValidator.validate_pin_format(valid_pin) is True

    def test_validate_pin_format_invalid_prefix(self):
        """Test validation of incorrect pin prefix."""
        invalid_pin = "sha1-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        assert CertificateValidator.validate_pin_format(invalid_pin) is False

    def test_validate_pin_format_invalid_encoding(self):
        """Test validation of incorrect pin encoding."""
        invalid_pin = "sha256-INVALID_BASE64!"
        assert CertificateValidator.validate_pin_format(invalid_pin) is False

    def test_validate_pin_format_invalid_length(self):
        """Test validation of incorrect pin length."""
        invalid_pin = "sha256-QUE="  # Too short
        assert CertificateValidator.validate_pin_format(invalid_pin) is False

    def test_generate_backup_pin(self):
        """Test backup pin generation."""
        backup_pin = CertificateValidator.generate_backup_pin()

        assert backup_pin.startswith("sha256-")
        assert CertificateValidator.validate_pin_format(backup_pin) is True

    @patch("auth.certificate_pinning.socket.create_connection")
    @patch("auth.certificate_pinning.ssl.create_default_context")
    @patch("auth.certificate_pinning.x509.load_der_x509_certificate")
    def test_get_certificate_from_server(
        self, mock_load_cert, mock_context, mock_socket
    ):
        """Test certificate retrieval from server."""
        # Mock certificate
        mock_cert = Mock()
        mock_cert.public_bytes.return_value = (
            b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"
        )
        mock_load_cert.return_value = mock_cert

        # Mock SSL socket and certificate retrieval
        mock_cert_der = b"mock_certificate_der_data"
        mock_ssl_socket = Mock()
        mock_ssl_socket.getpeercert.return_value = mock_cert_der

        # Mock context manager for SSL socket
        mock_wrapped_socket = Mock()
        mock_wrapped_socket.__enter__ = Mock(return_value=mock_ssl_socket)
        mock_wrapped_socket.__exit__ = Mock(return_value=None)

        # Mock context manager for regular socket
        mock_socket_instance = Mock()
        mock_socket_instance.__enter__ = Mock(return_value=mock_socket_instance)
        mock_socket_instance.__exit__ = Mock(return_value=None)

        mock_context_instance = Mock()
        mock_context_instance.wrap_socket.return_value = mock_wrapped_socket
        mock_context.return_value = mock_context_instance

        mock_socket.return_value = mock_socket_instance

        result = CertificateValidator.get_certificate_from_server("example.com")

        assert "BEGIN CERTIFICATE" in result
        mock_socket.assert_called_once_with(("example.com", 443), timeout=10)


class TestMobileCertificatePinner:
    """Test enhanced mobile certificate pinner."""

    def setup_method(self):
        """Set up test environment."""
        self.pinner = MobileCertificatePinner()

    def test_initialization(self):
        """Test pinner initialization."""
        assert hasattr(self.pinner, "domain_configs")
        assert hasattr(self.pinner, "pin_cache")
        assert hasattr(self.pinner, "failure_count")

    def test_add_domain_pins(self):
        """Test adding domain pins."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            backup_pins=["sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="],
        )

        self.pinner.add_domain_pins("example.com", config)

        assert "example.com" in self.pinner.domain_configs
        assert (
            self.pinner.domain_configs["example.com"].primary_pins
            == config.primary_pins
        )

    def test_add_domain_pins_invalid_format(self):
        """Test adding domain pins with invalid format."""
        config = PinConfiguration(primary_pins=["invalid-pin-format"])

        with pytest.raises(ValueError, match="Invalid pin format"):
            self.pinner.add_domain_pins("example.com", config)

    def test_get_pinning_header_basic(self):
        """Test basic pinning header generation."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            backup_pins=["sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="],
        )
        self.pinner.add_domain_pins("example.com", config)

        header = self.pinner.get_pinning_header("example.com")

        assert header is not None
        assert "pin-sha256=" in header
        assert "max-age=5184000" in header
        assert "includeSubDomains" in header

    def test_get_pinning_header_mobile_specific(self):
        """Test mobile-specific pinning header generation."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            mobile_specific=True,
        )
        self.pinner.add_domain_pins("example.com", config)

        # Should return None for non-mobile user agent
        header = self.pinner.get_pinning_header("example.com", "Mozilla/5.0 Desktop")
        assert header is None

        # Should return header for mobile user agent
        header = self.pinner.get_pinning_header("example.com", "FreeAgentics-iOS/1.0")
        assert header is not None

    def test_get_pinning_header_emergency_bypass(self):
        """Test emergency bypass functionality."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            emergency_bypass=True,
            emergency_bypass_until=datetime.now() + timedelta(hours=1),
        )
        self.pinner.add_domain_pins("example.com", config)

        header = self.pinner.get_pinning_header("example.com")
        assert header is None  # Should be bypassed

    def test_get_pinning_header_no_pins(self):
        """Test header generation when no pins are configured."""
        header = self.pinner.get_pinning_header("nonexistent.com")
        assert header is None

    def test_validate_certificate_chain_valid(self):
        """Test valid certificate chain validation."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            enforce_pinning=True,
        )
        self.pinner.add_domain_pins("example.com", config)

        # Mock certificate chain that matches the pin
        cert_chain = ["-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"]

        with patch.object(CertificateValidator, "extract_spki_pin") as mock_extract:
            mock_extract.return_value = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

            result = self.pinner.validate_certificate_chain("example.com", cert_chain)
            assert result is True

    def test_validate_certificate_chain_invalid(self):
        """Test invalid certificate chain validation."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            enforce_pinning=True,
        )
        self.pinner.add_domain_pins("example.com", config)

        # Mock certificate chain that doesn't match the pin
        cert_chain = ["-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"]

        with patch.object(CertificateValidator, "extract_spki_pin") as mock_extract:
            mock_extract.return_value = "DIFFERENT_PIN_HASH"

            result = self.pinner.validate_certificate_chain("example.com", cert_chain)
            assert result is False
            assert self.pinner.failure_count.get("example.com", 0) > 0

    def test_validate_certificate_chain_no_enforcement(self):
        """Test certificate chain validation when enforcement is disabled."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            enforce_pinning=False,
        )
        self.pinner.add_domain_pins("example.com", config)

        cert_chain = ["-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"]

        # Should always return True when enforcement is disabled
        result = self.pinner.validate_certificate_chain("example.com", cert_chain)
        assert result is True

    def test_emergency_bypass_domain(self):
        """Test emergency domain bypass."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="]
        )
        self.pinner.add_domain_pins("example.com", config)

        self.pinner.emergency_bypass_domain("example.com", duration_hours=12)

        domain_config = self.pinner.domain_configs["example.com"]
        assert domain_config.emergency_bypass is True
        assert domain_config.emergency_bypass_until is not None

    def test_get_mobile_pinning_config(self):
        """Test mobile app pinning configuration."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            backup_pins=["sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="],
            include_subdomains=True,
            max_age=2592000,
        )
        self.pinner.add_domain_pins("example.com", config)

        mobile_config = self.pinner.get_mobile_pinning_config("example.com")

        assert mobile_config is not None
        assert mobile_config["domain"] == "example.com"
        assert len(mobile_config["pins"]) == 2
        assert mobile_config["includeSubdomains"] is True
        assert mobile_config["maxAge"] == 2592000

    def test_get_mobile_pinning_config_not_found(self):
        """Test mobile app config for non-existent domain."""
        mobile_config = self.pinner.get_mobile_pinning_config("nonexistent.com")
        assert mobile_config is None

    @patch.dict(
        os.environ,
        {
            "CERT_PIN_FREEAGENTICS_COM": "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "CERT_BACKUP_PIN_FREEAGENTICS_COM": "sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=",
        },
    )
    def test_load_env_configuration(self):
        """Test loading configuration from environment variables."""
        pinner = MobileCertificatePinner()

        # Should have loaded pins from environment for production domains
        # The _load_env_configuration method checks for freeagentics.com domains
        assert "freeagentics.com" in pinner.domain_configs
        config = pinner.domain_configs["freeagentics.com"]
        assert (
            "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=" in config.primary_pins
        )
        assert (
            "sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=" in config.backup_pins
        )

    def test_load_file_configuration(self):
        """Test loading configuration from file."""
        config_data = {
            "example.com": {
                "primary_pins": ["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
                "backup_pins": ["sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="],
                "max_age": 2592000,
                "include_subdomains": True,
                "mobile_specific": True,
                "report_uri": "/pin-violations",
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("os.path.exists", return_value=True):
                pinner = MobileCertificatePinner()

                assert "example.com" in pinner.domain_configs
                config = pinner.domain_configs["example.com"]
                assert config.max_age == 2592000
                assert config.mobile_specific is True
                assert config.report_uri == "/pin-violations"

    @patch("requests.post")
    def test_report_pin_failure(self, mock_post):
        """Test pin failure reporting."""
        config = PinConfiguration(
            primary_pins=["sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="],
            report_uri="/pin-violations",
            enforce_pinning=True,
        )
        self.pinner.add_domain_pins("example.com", config)

        # Mock certificate chain that doesn't match
        cert_chain = ["-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"]

        with patch.object(CertificateValidator, "extract_spki_pin") as mock_extract:
            mock_extract.return_value = "DIFFERENT_PIN_HASH"

            self.pinner.validate_certificate_chain("example.com", cert_chain)

            # Should have made a POST request to report the failure
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["timeout"] == 5
            assert "json" in call_args[1]


class TestGlobalMobilePinner:
    """Test global mobile certificate pinner instance."""

    def test_global_instance_exists(self):
        """Test that global mobile pinner instance exists."""
        assert mobile_cert_pinner is not None
        assert isinstance(mobile_cert_pinner, MobileCertificatePinner)

    @patch.dict(os.environ, {"PRODUCTION": "true"})
    def test_setup_production_pins(self):
        """Test production pins setup."""
        with patch.object(mobile_cert_pinner, "add_domain_pins") as mock_add:
            setup_production_pins()

            # Should have called add_domain_pins for production domains
            assert mock_add.call_count > 0
