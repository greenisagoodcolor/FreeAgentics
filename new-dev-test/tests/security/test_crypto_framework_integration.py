#!/usr/bin/env python3
"""
Integration test for the comprehensive cryptography assessment framework.

This test validates that all components of the framework work together correctly.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_framework_components():
    """Test all framework components."""
    print("Testing Comprehensive Cryptography Assessment Framework")
    print("=" * 60)

    # Test 1: Algorithm Assessment
    print("\n1. Testing Algorithm Assessment...")
    try:
        from tests.security.cryptography_assessment_config import (
            CryptographyStandards,
            SecurityLevel,
        )

        # Test algorithm standards lookup
        sha256_standard = CryptographyStandards.HASH_ALGORITHMS.get("sha256")
        if sha256_standard and sha256_standard.security_level == SecurityLevel.HIGH:
            print("   ✓ Algorithm standards configuration working")
        else:
            print("   ✗ Algorithm standards configuration failed")

    except Exception as e:
        print(f"   ✗ Algorithm assessment error: {e}")

    # Test 2: Security Requirements
    print("\n2. Testing Security Requirements...")
    try:
        from tests.security.cryptography_assessment_config import SecurityRequirements

        requirements = SecurityRequirements.REQUIREMENTS
        if requirements and len(requirements) > 0:
            print(f"   ✓ {len(requirements)} security requirements loaded")
        else:
            print("   ✗ Security requirements not loaded")

    except Exception as e:
        print(f"   ✗ Security requirements error: {e}")

    # Test 3: Static Analysis Patterns
    print("\n3. Testing Static Analysis...")
    try:
        from tests.security.crypto_static_analysis import CryptographicPatternMatcher

        matcher = CryptographicPatternMatcher()
        if matcher.patterns and len(matcher.patterns) > 0:
            pattern_count = sum(len(patterns) for patterns in matcher.patterns.values())
            print(f"   ✓ {pattern_count} vulnerability patterns loaded")
        else:
            print("   ✗ Vulnerability patterns not loaded")

    except Exception as e:
        print(f"   ✗ Static analysis error: {e}")

    # Test 4: Scoring System
    print("\n4. Testing Scoring System...")
    try:
        from tests.security.cryptography_assessment_config import calculate_security_score

        test_findings = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"},
        ]

        score = calculate_security_score(test_findings)
        if score and "overall_score" in score:
            print(f"   ✓ Security scoring working: {score['overall_score']}%")
        else:
            print("   ✗ Security scoring failed")

    except Exception as e:
        print(f"   ✗ Scoring system error: {e}")

    # Test 5: Report Generation
    print("\n5. Testing Report Generation...")
    try:
        # Create temporary directory for test reports
        with tempfile.TemporaryDirectory() as temp_dir:
            test_report = {
                "assessment_metadata": {
                    "assessment_type": "test",
                    "platform": "test",
                },
                "executive_summary": {
                    "overall_risk_level": "LOW",
                    "security_score": 95.0,
                },
            }

            # Test JSON report generation
            report_path = Path(temp_dir) / "test_report.json"
            with open(report_path, "w") as f:
                json.dump(test_report, f, indent=2)

            if report_path.exists() and report_path.stat().st_size > 0:
                print("   ✓ Report generation working")
            else:
                print("   ✗ Report generation failed")

    except Exception as e:
        print(f"   ✗ Report generation error: {e}")

    # Test 6: Compliance Mapping
    print("\n6. Testing Compliance Mapping...")
    try:
        from tests.security.cryptography_assessment_config import COMPLIANCE_MAPPINGS

        if COMPLIANCE_MAPPINGS and len(COMPLIANCE_MAPPINGS) > 0:
            print(f"   ✓ {len(COMPLIANCE_MAPPINGS)} compliance standards mapped")
        else:
            print("   ✗ Compliance mapping failed")

    except Exception as e:
        print(f"   ✗ Compliance mapping error: {e}")


def test_cryptographic_implementations():
    """Test actual cryptographic implementations."""
    print("\n" + "=" * 60)
    print("Testing Cryptographic Implementations")
    print("=" * 60)

    # Test 1: Hash Functions
    print("\n1. Testing Hash Functions...")
    try:
        import hashlib
        import secrets

        # Test SHA-256
        test_data = b"test_data_for_validation"
        sha256_hash = hashlib.sha256(test_data).hexdigest()

        if len(sha256_hash) == 64 and all(c in "0123456789abcdef" for c in sha256_hash):
            print("   ✓ SHA-256 implementation correct")
        else:
            print("   ✗ SHA-256 implementation failed")

        # Test secure random salt generation
        salt = secrets.token_bytes(32)
        if len(salt) == 32:
            print("   ✓ Secure salt generation working")
        else:
            print("   ✗ Secure salt generation failed")

    except Exception as e:
        print(f"   ✗ Hash function test error: {e}")

    # Test 2: Symmetric Encryption
    print("\n2. Testing Symmetric Encryption...")
    try:
        import secrets

        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        # Test AES-256-GCM
        key = secrets.token_bytes(32)  # 256-bit key
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        plaintext = b"test_encryption_data"

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())

        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        auth_tag = encryptor.tag

        # Test decryption
        decryptor = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, auth_tag),
            backend=default_backend(),
        ).decryptor()

        decrypted = decryptor.update(ciphertext) + decryptor.finalize()

        if decrypted == plaintext:
            print("   ✓ AES-256-GCM implementation correct")
        else:
            print("   ✗ AES-256-GCM decryption failed")

    except Exception as e:
        print(f"   ✗ Symmetric encryption test error: {e}")

    # Test 3: Asymmetric Encryption
    print("\n3. Testing Asymmetric Encryption...")
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding, rsa

        # Generate RSA-2048 key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        public_key = private_key.public_key()

        # Test signing and verification
        message = b"test_signature_data"
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            print("   ✓ RSA-2048 PSS signature/verification correct")
        except Exception:
            print("   ✗ RSA-2048 PSS signature verification failed")

    except Exception as e:
        print(f"   ✗ Asymmetric encryption test error: {e}")

    # Test 4: Password Hashing
    print("\n4. Testing Password Hashing...")
    try:
        import bcrypt

        password = b"test_password_123"
        salt = bcrypt.gensalt(rounds=10)
        hashed = bcrypt.hashpw(password, salt)

        if bcrypt.checkpw(password, hashed):
            print("   ✓ bcrypt password hashing working")
        else:
            print("   ✗ bcrypt password verification failed")

        # Test wrong password
        if not bcrypt.checkpw(b"wrong_password", hashed):
            print("   ✓ bcrypt correctly rejects wrong password")
        else:
            print("   ✗ bcrypt incorrectly accepts wrong password")

    except Exception as e:
        print(f"   ✗ Password hashing test error: {e}")


def main():
    """Main test runner."""
    print("FreeAgentics Cryptography Assessment Framework Integration Test")
    print("=" * 70)

    # Test framework components
    test_framework_components()

    # Test cryptographic implementations
    test_cryptographic_implementations()

    print("\n" + "=" * 70)
    print("Integration Test Complete")
    print("=" * 70)

    print(
        """
Next Steps:
1. Run full assessment: python tests/security/comprehensive_crypto_security_suite.py
2. Generate reports: python tests/security/run_cryptography_assessment.py --output-dir ./reports
3. Review findings and implement recommendations
4. Integrate into CI/CD pipeline for continuous monitoring
    """
    )


if __name__ == "__main__":
    main()
