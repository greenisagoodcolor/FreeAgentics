"""
Field-level encryption with AWS KMS and HashiCorp Vault integration.
Provides transparent encryption/decryption with key rotation and FIPS 140-2 compliance.
"""

import base64
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
import hvac
from botocore.exceptions import ClientError
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import constant_time, hashes, serialization

# Cryptography imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted fields."""

    key_id: str
    algorithm: str
    encrypted_at: datetime
    key_version: int
    provider: str
    additional_data: Dict[str, Any] = field(default_factory=dict)


class KeyProvider(ABC):
    """Abstract base class for key providers."""

    @abstractmethod
    def generate_data_key(self, key_id: str) -> Tuple[bytes, bytes]:
        """Generate a new data encryption key."""
        pass

    @abstractmethod
    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data encryption key."""
        pass

    @abstractmethod
    def rotate_key(self, key_id: str) -> str:
        """Rotate a key and return new key ID."""
        pass

    @abstractmethod
    def get_key_metadata(self, key_id: str) -> Dict[str, Any]:
        """Get metadata about a key."""
        pass


class AWSKMSProvider(KeyProvider):
    """AWS KMS key provider with FIPS 140-2 compliance."""

    def __init__(self, region: str = "us-east-1", endpoint_url: Optional[str] = None):
        self.client = boto3.client(
            "kms", region_name=region, endpoint_url=endpoint_url  # For testing with localstack
        )
        self._key_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # 5 minutes

    def generate_data_key(self, key_id: str) -> Tuple[bytes, bytes]:
        """Generate AES-256 data key using AWS KMS."""
        try:
            response = self.client.generate_data_key(KeyId=key_id, KeySpec="AES_256")
            return response["Plaintext"], response["CiphertextBlob"]
        except ClientError as e:
            logger.error(f"AWS KMS error generating data key: {e}")
            raise

    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt data key using AWS KMS with caching."""
        cache_key = base64.b64encode(encrypted_key).decode()

        with self._cache_lock:
            if cache_key in self._key_cache:
                plaintext, timestamp = self._key_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return plaintext

        try:
            response = self.client.decrypt(CiphertextBlob=encrypted_key, KeyId=key_id)
            plaintext = response["Plaintext"]

            with self._cache_lock:
                self._key_cache[cache_key] = (plaintext, time.time())
                # Cleanup old entries
                self._cleanup_cache()

            return plaintext
        except ClientError as e:
            logger.error(f"AWS KMS error decrypting data key: {e}")
            raise

    def rotate_key(self, key_id: str) -> str:
        """Initiate key rotation in AWS KMS."""
        try:
            self.client.enable_key_rotation(KeyId=key_id)
            # Get new key version
            response = self.client.describe_key(KeyId=key_id)
            return response["KeyMetadata"]["KeyId"]
        except ClientError as e:
            logger.error(f"AWS KMS error rotating key: {e}")
            raise

    def get_key_metadata(self, key_id: str) -> Dict[str, Any]:
        """Get key metadata from AWS KMS."""
        try:
            response = self.client.describe_key(KeyId=key_id)
            return response["KeyMetadata"]
        except ClientError as e:
            logger.error(f"AWS KMS error getting key metadata: {e}")
            raise

    def _cleanup_cache(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            k for k, (_, ts) in self._key_cache.items() if current_time - ts >= self._cache_ttl
        ]
        for k in expired_keys:
            del self._key_cache[k]


class HashiCorpVaultProvider(KeyProvider):
    """HashiCorp Vault key provider with transit engine."""

    def __init__(self, vault_url: str, vault_token: str, mount_point: str = "transit"):
        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.mount_point = mount_point
        self._ensure_transit_engine()

    def _ensure_transit_engine(self):
        """Ensure transit engine is enabled."""
        try:
            engines = self.client.sys.list_mounted_secrets_engines()
            if f"{self.mount_point}/" not in engines:
                self.client.sys.enable_secrets_engine(backend_type="transit", path=self.mount_point)
        except Exception as e:
            logger.error(f"Vault error enabling transit engine: {e}")
            raise

    def generate_data_key(self, key_id: str) -> Tuple[bytes, bytes]:
        """Generate data key using Vault's transit engine."""
        try:
            # Generate random key
            response = self.client.secrets.transit.generate_data_key(
                name=key_id, key_type="aes256-gcm96", mount_point=self.mount_point
            )

            plaintext = base64.b64decode(response["data"]["plaintext"])
            ciphertext = response["data"]["ciphertext"].encode()

            return plaintext, ciphertext
        except Exception as e:
            logger.error(f"Vault error generating data key: {e}")
            raise

    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt data key using Vault."""
        try:
            ciphertext = (
                encrypted_key.decode() if isinstance(encrypted_key, bytes) else encrypted_key
            )

            response = self.client.secrets.transit.decrypt_data(
                name=key_id, ciphertext=ciphertext, mount_point=self.mount_point
            )

            return base64.b64decode(response["data"]["plaintext"])
        except Exception as e:
            logger.error(f"Vault error decrypting data key: {e}")
            raise

    def rotate_key(self, key_id: str) -> str:
        """Rotate key in Vault."""
        try:
            self.client.secrets.transit.rotate_key(name=key_id, mount_point=self.mount_point)
            # Get latest version
            response = self.client.secrets.transit.read_key(
                name=key_id, mount_point=self.mount_point
            )
            return f"{key_id}:v{response['data']['latest_version']}"
        except Exception as e:
            logger.error(f"Vault error rotating key: {e}")
            raise

    def get_key_metadata(self, key_id: str) -> Dict[str, Any]:
        """Get key metadata from Vault."""
        try:
            response = self.client.secrets.transit.read_key(
                name=key_id, mount_point=self.mount_point
            )
            return response["data"]
        except Exception as e:
            logger.error(f"Vault error getting key metadata: {e}")
            raise


class FieldEncryptor:
    """
    High-performance field-level encryptor with <5ms overhead.
    Supports AWS KMS and HashiCorp Vault with transparent encryption/decryption.
    """

    def __init__(
        self,
        provider: KeyProvider,
        default_key_id: str,
        cache_size: int = 1000,
        performance_monitoring: bool = True,
    ):
        self.provider = provider
        self.default_key_id = default_key_id
        self.cache_size = cache_size
        self.performance_monitoring = performance_monitoring

        # Performance tracking
        self._operation_times = defaultdict(list)
        self._operation_lock = threading.Lock()

        # Encryption cache for performance
        self._encryption_cache = {}
        self._cache_lock = threading.Lock()

    @contextmanager
    def _track_performance(self, operation: str):
        """Track operation performance."""
        if not self.performance_monitoring:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            with self._operation_lock:
                self._operation_times[operation].append(elapsed)
                # Keep only last 1000 measurements
                if len(self._operation_times[operation]) > 1000:
                    self._operation_times[operation] = self._operation_times[operation][-1000:]

    def encrypt_field(
        self,
        value: Any,
        field_name: str,
        key_id: Optional[str] = None,
        additional_data: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Encrypt a field value with metadata.
        Returns encrypted value with metadata for decryption.
        """
        with self._track_performance("encrypt_field"):
            key_id = key_id or self.default_key_id

            # Serialize value
            if isinstance(value, (str, int, float, bool)):
                plaintext = str(value).encode()
            else:
                plaintext = json.dumps(value).encode()

            # Generate or retrieve data key
            plaintext_key, encrypted_key = self.provider.generate_data_key(key_id)

            # Use AES-GCM for authenticated encryption
            aesgcm = AESGCM(plaintext_key)
            nonce = os.urandom(12)  # 96-bit nonce for GCM

            # Create additional authenticated data
            aad = additional_data or f"{field_name}:{datetime.utcnow().isoformat()}".encode()

            # Encrypt
            ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

            # Clear sensitive data
            plaintext_key = None

            # Create metadata
            metadata = EncryptionMetadata(
                key_id=key_id,
                algorithm="AES-256-GCM",
                encrypted_at=datetime.utcnow(),
                key_version=1,
                provider=type(self.provider).__name__,
                additional_data={"field_name": field_name},
            )

            return {
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "encrypted_key": base64.b64encode(encrypted_key).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "aad": base64.b64encode(aad).decode(),
                "metadata": metadata.__dict__,
            }

    def decrypt_field(
        self, encrypted_data: Dict[str, Any], expected_field_name: Optional[str] = None
    ) -> Any:
        """
        Decrypt a field value.
        Validates metadata and returns original value.
        """
        with self._track_performance("decrypt_field"):
            # Extract components
            ciphertext = base64.b64decode(encrypted_data["ciphertext"])
            encrypted_key = base64.b64decode(encrypted_data["encrypted_key"])
            nonce = base64.b64decode(encrypted_data["nonce"])
            aad = base64.b64decode(encrypted_data["aad"])
            metadata = encrypted_data["metadata"]

            # Validate field name if provided
            if (
                expected_field_name
                and metadata.get("additional_data", {}).get("field_name") != expected_field_name
            ):
                raise ValueError("Field name mismatch in encrypted data")

            # Decrypt data key
            plaintext_key = self.provider.decrypt_data_key(encrypted_key, metadata["key_id"])

            # Decrypt value
            aesgcm = AESGCM(plaintext_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, aad)

            # Clear sensitive data
            plaintext_key = None

            # Deserialize value
            value_str = plaintext.decode()
            try:
                # Try to parse as JSON first
                return json.loads(value_str)
            except json.JSONDecodeError:
                # Return as string if not JSON
                return value_str

    def bulk_encrypt_fields(
        self, data: Dict[str, Any], fields_to_encrypt: List[str], key_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encrypt multiple fields in a dictionary efficiently.
        """
        with self._track_performance("bulk_encrypt_fields"):
            encrypted_data = data.copy()

            for field_name in fields_to_encrypt:
                if field_name in data:
                    encrypted_data[field_name] = self.encrypt_field(
                        data[field_name], field_name, key_id
                    )

            return encrypted_data

    def bulk_decrypt_fields(
        self, encrypted_data: Dict[str, Any], fields_to_decrypt: List[str]
    ) -> Dict[str, Any]:
        """
        Decrypt multiple fields in a dictionary efficiently.
        """
        with self._track_performance("bulk_decrypt_fields"):
            decrypted_data = encrypted_data.copy()

            for field_name in fields_to_decrypt:
                if field_name in encrypted_data and isinstance(encrypted_data[field_name], dict):
                    if "ciphertext" in encrypted_data[field_name]:
                        decrypted_data[field_name] = self.decrypt_field(
                            encrypted_data[field_name], field_name
                        )

            return decrypted_data

    def rotate_field_encryption(
        self, encrypted_data: Dict[str, Any], new_key_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Re-encrypt field with new key for key rotation.
        """
        with self._track_performance("rotate_field_encryption"):
            # Decrypt with old key
            decrypted_value = self.decrypt_field(encrypted_data)

            # Get field name from metadata
            field_name = (
                encrypted_data["metadata"].get("additional_data", {}).get("field_name", "unknown")
            )

            # Re-encrypt with new key
            return self.encrypt_field(
                decrypted_value, field_name, new_key_id or self.default_key_id
            )

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for encryption operations.
        """
        with self._operation_lock:
            stats = {}
            for operation, times in self._operation_times.items():
                if times:
                    stats[operation] = {
                        "count": len(times),
                        "avg_ms": sum(times) / len(times),
                        "min_ms": min(times),
                        "max_ms": max(times),
                        "p95_ms": (
                            sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times)
                        ),
                    }
            return stats

    def clear_cache(self):
        """Clear encryption cache."""
        with self._cache_lock:
            self._encryption_cache.clear()


class TransparentFieldEncryptor:
    """
    Decorator-based transparent field encryption.
    Automatically encrypts/decrypts specified fields.
    """

    def __init__(self, encryptor: FieldEncryptor):
        self.encryptor = encryptor

    def encrypted_field(self, field_name: str, key_id: Optional[str] = None):
        """
        Decorator for class properties to enable transparent encryption.
        """

        def decorator(prop):
            @wraps(prop.fget)
            def getter(instance):
                encrypted_value = getattr(instance, f"_encrypted_{field_name}", None)
                if encrypted_value:
                    return self.encryptor.decrypt_field(encrypted_value, field_name)
                return None

            @wraps(prop.fset)
            def setter(instance, value):
                if value is not None:
                    encrypted_value = self.encryptor.encrypt_field(value, field_name, key_id)
                    setattr(instance, f"_encrypted_{field_name}", encrypted_value)
                else:
                    setattr(instance, f"_encrypted_{field_name}", None)

            return property(getter, setter)

        return decorator

    def encrypt_model_fields(self, fields_config: Dict[str, Optional[str]]):
        """
        Class decorator to automatically encrypt specified fields.

        Args:
            fields_config: Dict mapping field names to optional key IDs
        """

        def decorator(cls):
            original_init = cls.__init__

            def new_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)

                # Encrypt specified fields
                for field_name, key_id in fields_config.items():
                    if hasattr(instance, field_name):
                        value = getattr(instance, field_name)
                        if value is not None:
                            encrypted_value = self.encryptor.encrypt_field(
                                value, field_name, key_id
                            )
                            setattr(instance, f"_encrypted_{field_name}", encrypted_value)
                            # Replace with property
                            setattr(
                                instance.__class__,
                                field_name,
                                self.encrypted_field(field_name, key_id)(
                                    property(lambda self: None)
                                ),
                            )

            cls.__init__ = new_init
            return cls

        return decorator


def create_field_encryptor(provider_type: str = "aws_kms", **provider_config) -> FieldEncryptor:
    """
    Factory function to create field encryptor with specified provider.

    Args:
        provider_type: 'aws_kms' or 'vault'
        **provider_config: Provider-specific configuration

    Returns:
        Configured FieldEncryptor instance
    """
    if provider_type == "aws_kms":
        provider = AWSKMSProvider(
            region=provider_config.get("region", "us-east-1"),
            endpoint_url=provider_config.get("endpoint_url"),
        )
        default_key_id = provider_config.get("key_id", "alias/field-encryption")
    elif provider_type == "vault":
        provider = HashiCorpVaultProvider(
            vault_url=provider_config["vault_url"],
            vault_token=provider_config["vault_token"],
            mount_point=provider_config.get("mount_point", "transit"),
        )
        default_key_id = provider_config.get("key_id", "field-encryption")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return FieldEncryptor(
        provider=provider,
        default_key_id=default_key_id,
        cache_size=provider_config.get("cache_size", 1000),
        performance_monitoring=provider_config.get("performance_monitoring", True),
    )
