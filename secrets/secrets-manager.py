#!/usr/bin/env python3
"""
Production Secrets Manager for FreeAgentics
Handles secure generation, storage, and retrieval of secrets
"""

import argparse
import base64
import logging
import os
import secrets
import string
import sys
from pathlib import Path
from typing import Any, Dict

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SecretsManager:
    """Manages application secrets securely"""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.secrets_dir = Path(__file__).parent
        self.config_file = self.secrets_dir / f"secrets_config_{environment}.json"

    def generate_secret(self, length: int = 32, include_symbols: bool = True) -> str:
        """Generate a cryptographically secure random secret"""
        alphabet = string.ascii_letters + string.digits
        if include_symbols:
            alphabet += "!@#$%^&*"

        return "".join(secrets.choice(alphabet) for _ in range(length))

    def generate_jwt_keypair(self) -> tuple[str, str]:
        """Generate RSA keypair for JWT signing"""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        # Serialize public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        return private_pem, public_pem

    def generate_encryption_key(self) -> str:
        """Generate Fernet encryption key"""
        return Fernet.generate_key().decode("utf-8")

    def create_default_secrets(self) -> Dict[str, Any]:
        """Create default set of secrets for the application"""
        logger.info(f"Generating secrets for {self.environment} environment")

        # Generate JWT keypair
        jwt_private, jwt_public = self.generate_jwt_keypair()

        secrets = {
            # Application secrets
            "secret_key": self.generate_secret(64),
            "jwt_secret": self.generate_secret(64),
            "jwt_private_key": jwt_private,
            "jwt_public_key": jwt_public,
            "encryption_key": self.generate_encryption_key(),
            # Database secrets
            "postgres_password": self.generate_secret(32),
            "postgres_user": "freeagentics",
            "postgres_db": "freeagentics",
            # Redis secrets
            "redis_password": self.generate_secret(32),
            # Monitoring secrets
            "grafana_admin_password": self.generate_secret(16),
            # API keys (placeholders - replace with actual keys)
            "openai_api_key": "sk-placeholder-replace-with-actual-key",
            "anthropic_api_key": "placeholder-replace-with-actual-key",
            # Webhook URLs (placeholders)
            "slack_webhook": "https://hooks.slack.com/placeholder",
            # SSL/TLS settings
            "ssl_cert_path": "/etc/ssl/certs/freeagentics.crt",
            "ssl_key_path": "/etc/ssl/private/freeagentics.key",
            "ssl_ca_bundle_path": "/etc/ssl/certs/ca-bundle.crt",
            # Security settings
            "https_only": True,
            "secure_cookies": True,
            "access_token_expire_minutes": 30,
            "refresh_token_expire_days": 7,
            # Environment metadata
            "environment": self.environment,
            "generated_at": "2025-07-14T00:00:00Z",
        }

        return secrets

    def save_secrets_to_files(self, secrets: Dict[str, Any]) -> None:
        """Save secrets to individual files"""
        logger.info("Saving secrets to files")

        # Create secrets directory if it doesn't exist
        secrets_output_dir = self.secrets_dir / f"{self.environment}_secrets"
        secrets_output_dir.mkdir(exist_ok=True)

        # Save each secret to a separate file
        for key, value in secrets.items():
            if isinstance(value, str) and not key.endswith("_path"):
                file_path = secrets_output_dir / f"{key}.txt"
                with open(file_path, "w") as f:
                    f.write(value)
                os.chmod(file_path, 0o600)  # Read only for owner
                logger.info(f"Saved {key} to {file_path}")

    def save_docker_env_file(self, secrets: Dict[str, Any]) -> None:
        """Save secrets as Docker environment file"""
        env_file = self.secrets_dir / f".env.{self.environment}"

        logger.info(f"Saving Docker environment file to {env_file}")

        with open(env_file, "w") as f:
            f.write(f"# FreeAgentics {self.environment.title()} Environment Variables\n")
            f.write(f"# Generated on {secrets.get('generated_at', 'unknown')}\n")
            f.write("# DO NOT COMMIT THIS FILE TO VERSION CONTROL\n\n")

            for key, value in secrets.items():
                if isinstance(value, str):
                    f.write(f"{key.upper()}={value}\n")
                elif isinstance(value, bool):
                    f.write(f"{key.upper()}={'true' if value else 'false'}\n")
                else:
                    f.write(f"{key.upper()}={value}\n")

        os.chmod(env_file, 0o600)

    def save_kubernetes_secrets(self, secrets: Dict[str, Any]) -> None:
        """Generate Kubernetes secrets manifest"""
        k8s_file = self.secrets_dir / f"k8s-secrets-{self.environment}.yaml"

        logger.info(f"Generating Kubernetes secrets manifest: {k8s_file}")

        # Encode secrets in base64
        encoded_secrets = {}
        for key, value in secrets.items():
            if isinstance(value, str) and not key.endswith("_path"):
                encoded_secrets[key] = base64.b64encode(value.encode()).decode()

        manifest = f"""apiVersion: v1
kind: Secret
metadata:
  name: freeagentics-secrets
  namespace: freeagentics-{self.environment}
type: Opaque
data:
"""

        for key, encoded_value in encoded_secrets.items():
            manifest += f"  {key}: {encoded_value}\n"

        with open(k8s_file, "w") as f:
            f.write(manifest)

        os.chmod(k8s_file, 0o600)

    def generate_ssl_certificate(self, domain: str = "localhost") -> None:
        """Generate self-signed SSL certificate for development"""
        if self.environment == "production":
            logger.warning("Use proper SSL certificates in production!")
            return

        logger.info(f"Generating self-signed SSL certificate for {domain}")

        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Create certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FreeAgentics"),
                x509.NameAttribute(NameOID.COMMON_NAME, domain),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(domain),
                        x509.DNSName(f"*.{domain}"),
                        x509.DNSName("localhost"),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Save certificate and private key
        ssl_dir = self.secrets_dir / "ssl"
        ssl_dir.mkdir(exist_ok=True)

        cert_path = ssl_dir / f"{domain}.crt"
        key_path = ssl_dir / f"{domain}.key"

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        os.chmod(cert_path, 0o644)
        os.chmod(key_path, 0o600)

        logger.info(f"SSL certificate saved to {cert_path}")
        logger.info(f"SSL private key saved to {key_path}")


def main():
    parser = argparse.ArgumentParser(description="FreeAgentics Secrets Manager")
    parser.add_argument(
        "--environment",
        "-e",
        default="production",
        choices=["development", "staging", "production"],
        help="Target environment",
    )
    parser.add_argument(
        "--output",
        "-o",
        choices=["files", "docker", "k8s", "all"],
        default="all",
        help="Output format",
    )
    parser.add_argument("--domain", "-d", default="localhost", help="Domain for SSL certificate")
    parser.add_argument("--generate-ssl", action="store_true", help="Generate SSL certificate")

    args = parser.parse_args()

    manager = SecretsManager(args.environment)

    try:
        # Generate secrets
        secrets = manager.create_default_secrets()

        # Save in requested formats
        if args.output in ["files", "all"]:
            manager.save_secrets_to_files(secrets)

        if args.output in ["docker", "all"]:
            manager.save_docker_env_file(secrets)

        if args.output in ["k8s", "all"]:
            manager.save_kubernetes_secrets(secrets)

        # Generate SSL certificate if requested
        if args.generate_ssl:
            manager.generate_ssl_certificate(args.domain)

        logger.info("Secrets generation completed successfully!")
        logger.warning("Remember to:")
        logger.warning("1. Review and update placeholder values")
        logger.warning("2. Store secrets securely")
        logger.warning("3. Never commit actual secrets to version control")

    except Exception as e:
        logger.error(f"Error generating secrets: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
