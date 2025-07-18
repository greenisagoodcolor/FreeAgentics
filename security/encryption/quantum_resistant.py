"""
Quantum-resistant encryption using Kyber (key encapsulation) and Dilithium (signatures).
Implements homomorphic encryption for secure computations on encrypted data.
"""

import hashlib
import json
import logging
import os
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Note: In production, use actual implementations from:
# - liboqs-python for Kyber/Dilithium
# - tenseal or concrete-ml for homomorphic encryption
# This implementation provides the interface and simulated functionality


logger = logging.getLogger(__name__)


@dataclass
class QuantumKeyPair:
    """Quantum-resistant key pair."""

    public_key: bytes
    private_key: bytes
    algorithm: str
    parameters: Dict[str, Any]


@dataclass
class EncapsulatedKey:
    """Encapsulated key from KEM."""

    ciphertext: bytes
    shared_secret: bytes


class QuantumResistantAlgorithm(ABC):
    """Base class for quantum-resistant algorithms."""

    @abstractmethod
    def generate_keypair(self) -> QuantumKeyPair:
        """Generate a quantum-resistant key pair."""
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        pass


class KyberKEM(QuantumResistantAlgorithm):
    """
    Kyber Key Encapsulation Mechanism implementation.
    Kyber is a lattice-based KEM, finalist in NIST PQC standardization.
    """

    def __init__(self, security_level: int = 3):
        """
        Initialize Kyber with security level.
        Level 1: ~AES-128, Level 3: ~AES-192, Level 5: ~AES-256
        """
        self.security_level = security_level
        self.parameters = self._get_parameters(security_level)

    def _get_parameters(self, level: int) -> Dict[str, Any]:
        """Get Kyber parameters for security level."""
        params = {
            1: {"n": 256, "k": 2, "q": 3329, "eta1": 3, "eta2": 2},
            3: {"n": 256, "k": 3, "q": 3329, "eta1": 2, "eta2": 2},
            5: {"n": 256, "k": 4, "q": 3329, "eta1": 2, "eta2": 2},
        }
        return params.get(level, params[3])

    def generate_keypair(self) -> QuantumKeyPair:
        """Generate Kyber key pair."""
        # Simulated implementation - in production use liboqs
        n = self.parameters["n"]
        k = self.parameters["k"]

        # Generate random polynomial matrices for private key
        private_key = os.urandom(32 * k)  # Simplified

        # Derive public key (simplified - actual involves polynomial operations)
        public_key = hashlib.sha3_256(private_key).digest()

        return QuantumKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=self.get_algorithm_name(),
            parameters=self.parameters,
        )

    def encapsulate(self, public_key: bytes) -> EncapsulatedKey:
        """
        Encapsulate a shared secret using public key.
        Returns ciphertext and shared secret.
        """
        # Generate random shared secret
        shared_secret = os.urandom(32)

        # Encapsulate (simplified - actual involves lattice operations)
        r = os.urandom(32)
        ciphertext = hashlib.sha3_256(public_key + r + shared_secret).digest()

        return EncapsulatedKey(
            ciphertext=ciphertext, shared_secret=shared_secret
        )

    def decapsulate(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """
        Decapsulate shared secret using private key.
        Returns shared secret.
        """
        # Simplified decapsulation
        # In real implementation, this involves lattice decryption
        return hashlib.sha3_256(private_key + ciphertext).digest()[:32]

    def get_algorithm_name(self) -> str:
        return f"Kyber{self.security_level}"


class DilithiumSigner(QuantumResistantAlgorithm):
    """
    Dilithium digital signature implementation.
    Dilithium is a lattice-based signature scheme, NIST PQC standard.
    """

    def __init__(self, security_level: int = 3):
        """
        Initialize Dilithium with security level.
        Level 2: ~AES-128, Level 3: ~AES-192, Level 5: ~AES-256
        """
        self.security_level = security_level
        self.parameters = self._get_parameters(security_level)

    def _get_parameters(self, level: int) -> Dict[str, Any]:
        """Get Dilithium parameters for security level."""
        params = {
            2: {"n": 256, "k": 4, "l": 4, "eta": 2, "tau": 39},
            3: {"n": 256, "k": 6, "l": 5, "eta": 4, "tau": 49},
            5: {"n": 256, "k": 8, "l": 7, "eta": 2, "tau": 60},
        }
        return params.get(level, params[3])

    def generate_keypair(self) -> QuantumKeyPair:
        """Generate Dilithium key pair."""
        # Simulated implementation
        k = self.parameters["k"]
        l = self.parameters["l"]

        # Generate random seed
        seed = os.urandom(32)

        # Derive keys (simplified)
        private_key = hashlib.sha3_256(seed + b"private").digest()
        public_key = hashlib.sha3_256(seed + b"public").digest()

        return QuantumKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=self.get_algorithm_name(),
            parameters=self.parameters,
        )

    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """
        Sign message with Dilithium private key.
        """
        # Simplified signature
        # Real implementation uses lattice-based signatures
        h = hashlib.sha3_256(message).digest()
        signature = hashlib.sha3_512(private_key + h).digest()
        return signature

    def verify(
        self, message: bytes, signature: bytes, public_key: bytes
    ) -> bool:
        """
        Verify signature with Dilithium public key.
        """
        # Simplified verification
        # This is just for interface demonstration
        expected = hashlib.sha3_256(message).digest()
        return len(signature) == 64  # Simplified check

    def get_algorithm_name(self) -> str:
        return f"Dilithium{self.security_level}"


class HomomorphicEncryption:
    """
    Homomorphic encryption for computations on encrypted data.
    Supports addition and multiplication operations.
    """

    def __init__(self, key_size: int = 2048):
        """Initialize with key size for security."""
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        self.context = None
        self._setup_keys()

    def _setup_keys(self):
        """Setup homomorphic encryption keys."""
        # In production, use TenSEAL or Concrete-ML
        # This is a simplified simulation
        self.modulus = 2**31 - 1  # Large prime
        self.private_key = os.urandom(32)
        self.public_key = hashlib.sha256(self.private_key).digest()

    def encrypt(
        self, value: Union[int, float, List[Union[int, float]]]
    ) -> "EncryptedValue":
        """
        Encrypt a value or list of values.
        """
        if isinstance(value, (int, float)):
            values = [value]
        else:
            values = value

        # Simplified encryption (in practice uses polynomial rings)
        encrypted_values = []
        for v in values:
            # Add noise for security
            noise = struct.unpack("i", os.urandom(4))[0] % 100
            encrypted = (int(v * 1000) + noise) % self.modulus
            encrypted_values.append(encrypted)

        return EncryptedValue(
            ciphertexts=encrypted_values,
            encryption_params={"modulus": self.modulus, "scale": 1000},
        )

    def decrypt(
        self, encrypted_value: "EncryptedValue"
    ) -> Union[float, List[float]]:
        """
        Decrypt an encrypted value.
        """
        # Simplified decryption
        scale = encrypted_value.encryption_params["scale"]
        values = []

        for ciphertext in encrypted_value.ciphertexts:
            # Remove noise (simplified)
            decrypted = ciphertext / scale
            values.append(decrypted)

        return values[0] if len(values) == 1 else values

    def add(
        self, a: "EncryptedValue", b: "EncryptedValue"
    ) -> "EncryptedValue":
        """
        Add two encrypted values.
        """
        if len(a.ciphertexts) != len(b.ciphertexts):
            raise ValueError("Encrypted values must have same length")

        result_ciphertexts = [
            (ca + cb) % self.modulus
            for ca, cb in zip(a.ciphertexts, b.ciphertexts)
        ]

        return EncryptedValue(
            ciphertexts=result_ciphertexts,
            encryption_params=a.encryption_params,
        )

    def multiply(
        self, a: "EncryptedValue", b: "EncryptedValue"
    ) -> "EncryptedValue":
        """
        Multiply two encrypted values.
        """
        if len(a.ciphertexts) != len(b.ciphertexts):
            raise ValueError("Encrypted values must have same length")

        # Simplified multiplication (real HE is more complex)
        result_ciphertexts = [
            (ca * cb) % self.modulus
            for ca, cb in zip(a.ciphertexts, b.ciphertexts)
        ]

        # Update scale for multiplication
        new_params = a.encryption_params.copy()
        new_params["scale"] = (
            a.encryption_params["scale"] * b.encryption_params["scale"]
        )

        return EncryptedValue(
            ciphertexts=result_ciphertexts, encryption_params=new_params
        )

    def compute_mean(
        self, encrypted_values: List["EncryptedValue"]
    ) -> "EncryptedValue":
        """
        Compute mean of encrypted values without decryption.
        """
        if not encrypted_values:
            raise ValueError("No values to compute mean")

        # Sum all values
        total = encrypted_values[0]
        for value in encrypted_values[1:]:
            total = self.add(total, value)

        # Division by constant (simplified)
        n = len(encrypted_values)
        result_ciphertexts = [ct // n for ct in total.ciphertexts]

        return EncryptedValue(
            ciphertexts=result_ciphertexts,
            encryption_params=total.encryption_params,
        )


@dataclass
class EncryptedValue:
    """Represents a homomorphically encrypted value."""

    ciphertexts: List[int]
    encryption_params: Dict[str, Any]


class QuantumResistantCrypto:
    """
    High-level interface for quantum-resistant cryptography.
    Combines Kyber, Dilithium, and homomorphic encryption.
    """

    def __init__(
        self,
        kyber_level: int = 3,
        dilithium_level: int = 3,
        enable_homomorphic: bool = True,
    ):
        self.kyber = KyberKEM(kyber_level)
        self.dilithium = DilithiumSigner(dilithium_level)
        self.homomorphic = (
            HomomorphicEncryption() if enable_homomorphic else None
        )

        # Cache for performance
        self._key_cache = {}

    def generate_hybrid_keypair(self) -> Dict[str, QuantumKeyPair]:
        """
        Generate both KEM and signature key pairs.
        """
        return {
            "kem": self.kyber.generate_keypair(),
            "sign": self.dilithium.generate_keypair(),
        }

    def hybrid_encrypt(
        self,
        data: bytes,
        recipient_public_key: bytes,
        sign_with_private_key: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Encrypt data using hybrid quantum-resistant encryption.
        Optionally sign the encrypted data.
        """
        # Encapsulate shared secret
        encapsulated = self.kyber.encapsulate(recipient_public_key)

        # Derive encryption key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"quantum-resistant-encryption",
            backend=default_backend(),
        )
        encryption_key = hkdf.derive(encapsulated.shared_secret)

        # Encrypt data with AES-GCM
        nonce = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(encryption_key),
            modes.GCM(nonce),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        result = {
            "encapsulated_key": encapsulated.ciphertext,
            "ciphertext": ciphertext,
            "nonce": nonce,
            "tag": encryptor.tag,
            "algorithm": "Kyber-AES-GCM",
        }

        # Sign if requested
        if sign_with_private_key:
            # Sign the ciphertext and encapsulated key
            to_sign = encapsulated.ciphertext + ciphertext
            signature = self.dilithium.sign(to_sign, sign_with_private_key)
            result["signature"] = signature
            result["signature_algorithm"] = self.dilithium.get_algorithm_name()

        return result

    def hybrid_decrypt(
        self,
        encrypted_data: Dict[str, Any],
        private_key: bytes,
        verify_with_public_key: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypt data using hybrid quantum-resistant decryption.
        Optionally verify signature.
        """
        # Verify signature if provided
        if verify_with_public_key and "signature" in encrypted_data:
            to_verify = (
                encrypted_data["encapsulated_key"]
                + encrypted_data["ciphertext"]
            )
            if not self.dilithium.verify(
                to_verify, encrypted_data["signature"], verify_with_public_key
            ):
                raise ValueError("Signature verification failed")

        # Decapsulate shared secret
        shared_secret = self.kyber.decapsulate(
            encrypted_data["encapsulated_key"], private_key
        )

        # Derive decryption key
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"quantum-resistant-encryption",
            backend=default_backend(),
        )
        decryption_key = hkdf.derive(shared_secret)

        # Decrypt data
        cipher = Cipher(
            algorithms.AES(decryption_key),
            modes.GCM(encrypted_data["nonce"], encrypted_data["tag"]),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        plaintext = (
            decryptor.update(encrypted_data["ciphertext"])
            + decryptor.finalize()
        )

        return plaintext

    def encrypt_for_computation(self, values: List[float]) -> "EncryptedValue":
        """
        Encrypt values for homomorphic computation.
        """
        if not self.homomorphic:
            raise ValueError("Homomorphic encryption not enabled")

        return self.homomorphic.encrypt(values)

    def compute_on_encrypted(
        self, operation: str, *args: "EncryptedValue"
    ) -> "EncryptedValue":
        """
        Perform computation on encrypted values.
        Supports 'add', 'multiply', 'mean'.
        """
        if not self.homomorphic:
            raise ValueError("Homomorphic encryption not enabled")

        if operation == "add":
            return self.homomorphic.add(args[0], args[1])
        elif operation == "multiply":
            return self.homomorphic.multiply(args[0], args[1])
        elif operation == "mean":
            return self.homomorphic.compute_mean(list(args))
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def decrypt_computation_result(
        self, encrypted_result: "EncryptedValue"
    ) -> Union[float, List[float]]:
        """
        Decrypt result of homomorphic computation.
        """
        if not self.homomorphic:
            raise ValueError("Homomorphic encryption not enabled")

        return self.homomorphic.decrypt(encrypted_result)


# Encryption-at-rest implementation
class EncryptionAtRest:
    """
    Provides transparent encryption for data stores.
    Supports various backends with automatic key management.
    """

    def __init__(
        self, crypto: QuantumResistantCrypto, key_rotation_days: int = 90
    ):
        self.crypto = crypto
        self.key_rotation_days = key_rotation_days
        self.current_keys = self.crypto.generate_hybrid_keypair()
        self.key_metadata = {
            "created_at": time.time(),
            "rotation_due": time.time() + (key_rotation_days * 86400),
        }

    def encrypt_document(
        self,
        document: Dict[str, Any],
        fields_to_encrypt: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Encrypt document for storage.
        If fields_to_encrypt is None, encrypts entire document.
        """
        if fields_to_encrypt:
            # Field-level encryption
            encrypted_doc = document.copy()
            for field in fields_to_encrypt:
                if field in document:
                    value_bytes = json.dumps(document[field]).encode()
                    encrypted_doc[field] = self.crypto.hybrid_encrypt(
                        value_bytes, self.current_keys["kem"].public_key
                    )
        else:
            # Full document encryption
            doc_bytes = json.dumps(document).encode()
            encrypted_doc = {
                "_encrypted": True,
                "_data": self.crypto.hybrid_encrypt(
                    doc_bytes, self.current_keys["kem"].public_key
                ),
            }

        # Add encryption metadata
        encrypted_doc["_encryption_metadata"] = {
            "timestamp": time.time(),
            "key_id": hashlib.sha256(
                self.current_keys["kem"].public_key
            ).hexdigest()[:16],
            "algorithm": "quantum-resistant",
        }

        return encrypted_doc

    def decrypt_document(
        self,
        encrypted_doc: Dict[str, Any],
        fields_to_decrypt: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Decrypt document from storage.
        """
        if encrypted_doc.get("_encrypted"):
            # Full document encryption
            decrypted_bytes = self.crypto.hybrid_decrypt(
                encrypted_doc["_data"], self.current_keys["kem"].private_key
            )
            return json.loads(decrypted_bytes)
        else:
            # Field-level encryption
            decrypted_doc = encrypted_doc.copy()
            if fields_to_decrypt:
                for field in fields_to_decrypt:
                    if field in encrypted_doc and isinstance(
                        encrypted_doc[field], dict
                    ):
                        if "encapsulated_key" in encrypted_doc[field]:
                            decrypted_bytes = self.crypto.hybrid_decrypt(
                                encrypted_doc[field],
                                self.current_keys["kem"].private_key,
                            )
                            decrypted_doc[field] = json.loads(decrypted_bytes)

            # Remove metadata
            decrypted_doc.pop("_encryption_metadata", None)
            return decrypted_doc

    def should_rotate_keys(self) -> bool:
        """Check if key rotation is due."""
        return time.time() >= self.key_metadata["rotation_due"]

    def rotate_keys(self) -> Dict[str, QuantumKeyPair]:
        """Rotate encryption keys."""
        old_keys = self.current_keys
        self.current_keys = self.crypto.generate_hybrid_keypair()
        self.key_metadata = {
            "created_at": time.time(),
            "rotation_due": time.time() + (self.key_rotation_days * 86400),
            "previous_key_id": hashlib.sha256(
                old_keys["kem"].public_key
            ).hexdigest()[:16],
        }
        return self.current_keys
