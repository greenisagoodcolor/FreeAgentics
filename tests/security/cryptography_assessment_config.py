"""
Cryptography Assessment Configuration

This module defines configuration settings, standards, and criteria
for the comprehensive cryptography assessment framework.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class SecurityLevel(Enum):
    """Security levels for cryptographic implementations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComplianceStandard(Enum):
    """Compliance standards for cryptographic assessments."""

    NIST_SP_800_57 = "nist_sp_800_57"
    FIPS_140_2 = "fips_140_2"
    OWASP_CRYPTOGRAPHIC_STORAGE = "owasp_cryptographic_storage"
    RFC_8446_TLS_1_3 = "rfc_8446_tls_1_3"
    ENISA_CRYPTOGRAPHIC_GUIDELINES = "enisa_cryptographic_guidelines"


@dataclass
class AlgorithmStandard:
    """Standard definition for cryptographic algorithms."""

    name: str
    minimum_key_size: int
    recommended_key_size: int
    security_level: SecurityLevel
    compliance_standards: List[ComplianceStandard]
    deprecation_year: Optional[int] = None
    replacement_algorithms: List[str] = None


@dataclass
class SecurityRequirement:
    """Security requirement definition."""

    requirement_id: str
    description: str
    security_level: SecurityLevel
    compliance_standards: List[ComplianceStandard]
    test_criteria: List[str]
    remediation_guidance: str


class CryptographyStandards:
    """Cryptography standards and best practices."""

    # Hash Algorithm Standards
    HASH_ALGORITHMS = {
        "sha256": AlgorithmStandard(
            name="SHA-256",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
                ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE,
            ],
        ),
        "sha384": AlgorithmStandard(
            name="SHA-384",
            minimum_key_size=384,
            recommended_key_size=384,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
        ),
        "sha512": AlgorithmStandard(
            name="SHA-512",
            minimum_key_size=512,
            recommended_key_size=512,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
        ),
        "sha1": AlgorithmStandard(
            name="SHA-1",
            minimum_key_size=160,
            recommended_key_size=160,
            security_level=SecurityLevel.CRITICAL,
            compliance_standards=[],
            deprecation_year=2017,
            replacement_algorithms=["SHA-256", "SHA-384", "SHA-512"],
        ),
        "md5": AlgorithmStandard(
            name="MD5",
            minimum_key_size=128,
            recommended_key_size=128,
            security_level=SecurityLevel.CRITICAL,
            compliance_standards=[],
            deprecation_year=2012,
            replacement_algorithms=["SHA-256", "SHA-384", "SHA-512"],
        ),
    }

    # Symmetric Encryption Standards
    SYMMETRIC_ALGORITHMS = {
        "aes-256-gcm": AlgorithmStandard(
            name="AES-256-GCM",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
                ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE,
            ],
        ),
        "aes-128-gcm": AlgorithmStandard(
            name="AES-128-GCM",
            minimum_key_size=128,
            recommended_key_size=128,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
        ),
        "chacha20-poly1305": AlgorithmStandard(
            name="ChaCha20-Poly1305",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[ComplianceStandard.RFC_8446_TLS_1_3],
        ),
        "des": AlgorithmStandard(
            name="DES",
            minimum_key_size=56,
            recommended_key_size=56,
            security_level=SecurityLevel.CRITICAL,
            compliance_standards=[],
            deprecation_year=1998,
            replacement_algorithms=["AES-128", "AES-256"],
        ),
        "3des": AlgorithmStandard(
            name="3DES",
            minimum_key_size=112,
            recommended_key_size=168,
            security_level=SecurityLevel.CRITICAL,
            compliance_standards=[],
            deprecation_year=2023,
            replacement_algorithms=["AES-128", "AES-256"],
        ),
    }

    # Asymmetric Encryption Standards
    ASYMMETRIC_ALGORITHMS = {
        "rsa-2048": AlgorithmStandard(
            name="RSA-2048",
            minimum_key_size=2048,
            recommended_key_size=2048,
            security_level=SecurityLevel.MEDIUM,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
        ),
        "rsa-3072": AlgorithmStandard(
            name="RSA-3072",
            minimum_key_size=3072,
            recommended_key_size=3072,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
        ),
        "rsa-4096": AlgorithmStandard(
            name="RSA-4096",
            minimum_key_size=4096,
            recommended_key_size=4096,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
        ),
        "ecdsa-p256": AlgorithmStandard(
            name="ECDSA P-256",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
                ComplianceStandard.RFC_8446_TLS_1_3,
            ],
        ),
        "ecdsa-p384": AlgorithmStandard(
            name="ECDSA P-384",
            minimum_key_size=384,
            recommended_key_size=384,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
        ),
        "ed25519": AlgorithmStandard(
            name="Ed25519",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[ComplianceStandard.RFC_8446_TLS_1_3],
        ),
        "rsa-1024": AlgorithmStandard(
            name="RSA-1024",
            minimum_key_size=1024,
            recommended_key_size=1024,
            security_level=SecurityLevel.CRITICAL,
            compliance_standards=[],
            deprecation_year=2010,
            replacement_algorithms=["RSA-2048", "RSA-3072", "ECDSA P-256"],
        ),
    }

    # Key Derivation Function Standards
    KDF_ALGORITHMS = {
        "pbkdf2-sha256": AlgorithmStandard(
            name="PBKDF2-SHA256",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE,
            ],
        ),
        "scrypt": AlgorithmStandard(
            name="Scrypt",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE],
        ),
        "argon2": AlgorithmStandard(
            name="Argon2",
            minimum_key_size=256,
            recommended_key_size=256,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE],
        ),
        "bcrypt": AlgorithmStandard(
            name="bcrypt",
            minimum_key_size=184,  # bcrypt output size
            recommended_key_size=184,
            security_level=SecurityLevel.HIGH,
            compliance_standards=[ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE],
        ),
    }


class SecurityRequirements:
    """Security requirements for cryptographic implementations."""

    REQUIREMENTS = [
        SecurityRequirement(
            requirement_id="CRYPTO-001",
            description="Use only approved cryptographic algorithms",
            security_level=SecurityLevel.CRITICAL,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
            test_criteria=[
                "No deprecated hash algorithms (MD5, SHA-1)",
                "No weak symmetric ciphers (DES, 3DES, RC4)",
                "No weak asymmetric keys (RSA < 2048)",
                "All algorithms meet minimum security standards",
            ],
            remediation_guidance="Replace deprecated algorithms with approved alternatives. Update configuration to disable weak ciphers.",
        ),
        SecurityRequirement(
            requirement_id="CRYPTO-002",
            description="Implement proper key management",
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE,
            ],
            test_criteria=[
                "Keys generated using cryptographically secure random number generator",
                "Key storage protected with appropriate permissions",
                "Key rotation implemented for long-term keys",
                "Key lifecycle management in place",
            ],
            remediation_guidance="Implement secure key generation, storage, and rotation procedures. Use hardware security modules where appropriate.",
        ),
        SecurityRequirement(
            requirement_id="CRYPTO-003",
            description="Use authenticated encryption for data protection",
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE,
                ComplianceStandard.NIST_SP_800_57,
            ],
            test_criteria=[
                "Authenticated encryption modes used (GCM, CCM, Poly1305)",
                "Integrity verification implemented",
                "No unauthenticated encryption modes",
                "Proper IV/nonce generation",
            ],
            remediation_guidance="Replace unauthenticated encryption with authenticated encryption modes. Implement proper IV/nonce generation.",
        ),
        SecurityRequirement(
            requirement_id="CRYPTO-004",
            description="Implement secure SSL/TLS configuration",
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.RFC_8446_TLS_1_3,
                ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE,
            ],
            test_criteria=[
                "TLS 1.2 minimum version",
                "Strong cipher suites only",
                "Perfect forward secrecy enabled",
                "Certificate validation implemented",
            ],
            remediation_guidance="Update TLS configuration to use strong protocols and cipher suites. Implement certificate pinning where appropriate.",
        ),
        SecurityRequirement(
            requirement_id="CRYPTO-005",
            description="Protect against timing attacks",
            security_level=SecurityLevel.MEDIUM,
            compliance_standards=[ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE],
            test_criteria=[
                "Constant-time operations for sensitive comparisons",
                "No timing-based information leakage",
                "Proper error handling without timing differences",
                "Use of timing-safe comparison functions",
            ],
            remediation_guidance="Implement constant-time comparison functions. Review error handling for timing leaks.",
        ),
        SecurityRequirement(
            requirement_id="CRYPTO-006",
            description="Use secure random number generation",
            security_level=SecurityLevel.HIGH,
            compliance_standards=[
                ComplianceStandard.NIST_SP_800_57,
                ComplianceStandard.FIPS_140_2,
            ],
            test_criteria=[
                "Cryptographically secure pseudo-random number generator (CSPRNG)",
                "Proper entropy source",
                "No predictable patterns in generated values",
                "Adequate seed material",
            ],
            remediation_guidance="Use approved CSPRNG implementations. Ensure proper entropy collection and seeding.",
        ),
        SecurityRequirement(
            requirement_id="CRYPTO-007",
            description="Implement proper password storage",
            security_level=SecurityLevel.CRITICAL,
            compliance_standards=[ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE],
            test_criteria=[
                "Strong password hashing function (bcrypt, scrypt, Argon2)",
                "Unique salt per password",
                "Appropriate work factor/iterations",
                "No plaintext password storage",
            ],
            remediation_guidance="Implement approved password hashing with unique salts and appropriate work factors.",
        ),
        SecurityRequirement(
            requirement_id="CRYPTO-008",
            description="Implement certificate pinning for critical connections",
            security_level=SecurityLevel.MEDIUM,
            compliance_standards=[ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE],
            test_criteria=[
                "Certificate pinning implemented for API endpoints",
                "Pin validation working correctly",
                "Backup pins configured",
                "Pin failure reporting implemented",
            ],
            remediation_guidance="Implement certificate pinning for critical connections. Configure backup pins and monitoring.",
        ),
    ]


class AssessmentConfiguration:
    """Configuration for cryptography assessment."""

    # Test configuration
    TEST_CONFIG = {
        "timing_attack_threshold": 1.2,  # 20% timing variance threshold
        "randomness_sample_size": 1000,
        "randomness_entropy_threshold": 200,  # Minimum unique bytes in random data
        "key_generation_iterations": 100,
        "padding_oracle_test_iterations": 50,
        "minimum_rsa_key_size": 2048,
        "minimum_pbkdf2_iterations": 10000,
        "minimum_bcrypt_rounds": 10,
        "tls_minimum_version": "TLSv1.2",
        "jwt_maximum_lifetime_minutes": 60,
        "refresh_token_maximum_lifetime_days": 30,
    }

    # Vulnerability patterns to detect
    VULNERABILITY_PATTERNS = {
        "weak_hash_patterns": [
            r"hashlib\.md5\(",
            r"hashlib\.sha1\(",
            r"\.md5\(",
            r"\.sha1\(",
            r"MD5\(",
            r"SHA1\(",
        ],
        "weak_cipher_patterns": [
            r"DES\(",
            r"3DES\(",
            r"RC4\(",
            r"modes\.ECB\(",
            r"Cipher.*ECB",
        ],
        "weak_random_patterns": [
            r"random\.random\(",
            r"random\.randint\(",
            r"random\.choice\(",
            r"Math\.random\(",
            r"rand\(\)",
        ],
        "hardcoded_key_patterns": [
            r'key\s*=\s*["\'][^"\']{10,}["\']',
            r'secret\s*=\s*["\'][^"\']{10,}["\']',
            r'password\s*=\s*["\'][^"\']{5,}["\']',
        ],
    }

    # Compliance mappings
    COMPLIANCE_MAPPINGS = {
        ComplianceStandard.NIST_SP_800_57: {
            "document": "NIST Special Publication 800-57 Part 1",
            "url": "https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final",
            "key_requirements": [
                "Use of approved cryptographic algorithms",
                "Minimum key lengths for security levels",
                "Key management best practices",
            ],
        },
        ComplianceStandard.FIPS_140_2: {
            "document": "FIPS 140-2 Security Requirements for Cryptographic Modules",
            "url": "https://csrc.nist.gov/publications/detail/fips/140/2/final",
            "key_requirements": [
                "FIPS-approved cryptographic algorithms",
                "Key management requirements",
                "Physical security requirements",
            ],
        },
        ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE: {
            "document": "OWASP Cryptographic Storage Cheat Sheet",
            "url": "https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html",
            "key_requirements": [
                "Use strong encryption algorithms",
                "Proper key management",
                "Secure password storage",
            ],
        },
        ComplianceStandard.RFC_8446_TLS_1_3: {
            "document": "RFC 8446 - The Transport Layer Security (TLS) Protocol Version 1.3",
            "url": "https://tools.ietf.org/rfc/rfc8446.txt",
            "key_requirements": [
                "TLS 1.3 security requirements",
                "Cipher suite recommendations",
                "Perfect forward secrecy",
            ],
        },
    }

    # Reporting configuration
    REPORT_CONFIG = {
        "output_format": "json",  # json, html, pdf
        "include_remediation": True,
        "include_compliance_mapping": True,
        "severity_levels": ["critical", "high", "medium", "low", "info"],
        "generate_executive_summary": True,
        "include_technical_details": True,
    }


# Assessment scoring weights
SCORING_WEIGHTS = {
    SecurityLevel.CRITICAL: 100,
    SecurityLevel.HIGH: 50,
    SecurityLevel.MEDIUM: 25,
    SecurityLevel.LOW: 10,
    SecurityLevel.INFO: 1,
}


def get_algorithm_standard(
    algorithm_type: str, algorithm_name: str
) -> Optional[AlgorithmStandard]:
    """Get algorithm standard by type and name."""
    standards_map = {
        "hash": CryptographyStandards.HASH_ALGORITHMS,
        "symmetric": CryptographyStandards.SYMMETRIC_ALGORITHMS,
        "asymmetric": CryptographyStandards.ASYMMETRIC_ALGORITHMS,
        "kdf": CryptographyStandards.KDF_ALGORITHMS,
    }

    if algorithm_type in standards_map:
        return standards_map[algorithm_type].get(algorithm_name.lower())
    return None


def calculate_security_score(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall security score based on findings."""
    total_score = 0
    max_score = 0

    severity_counts = {level.value: 0 for level in SecurityLevel}

    for finding in findings:
        severity = finding.get("severity", SecurityLevel.INFO.value)
        severity_counts[severity] += 1

        # Deduct points for vulnerabilities
        if severity in SCORING_WEIGHTS:
            total_score -= SCORING_WEIGHTS[SecurityLevel(severity)]

        # Add max possible points
        max_score += SCORING_WEIGHTS[SecurityLevel.HIGH]

    # Calculate percentage (0-100)
    if max_score > 0:
        score_percentage = max(0, (total_score + max_score) / max_score * 100)
    else:
        score_percentage = 100

    return {
        "overall_score": round(score_percentage, 2),
        "severity_counts": severity_counts,
        "total_findings": len(findings),
        "score_breakdown": {
            "total_deducted": abs(min(0, total_score)),
            "max_possible": max_score,
            "percentage": round(score_percentage, 2),
        },
    }


def get_compliance_status(
    findings: List[Dict[str, Any]],
) -> Dict[ComplianceStandard, str]:
    """Determine compliance status for each standard."""
    compliance_status = {}

    for standard in ComplianceStandard:
        critical_findings = [
            f
            for f in findings
            if f.get("severity") == SecurityLevel.CRITICAL.value
            and standard in f.get("compliance_standards", [])
        ]

        high_findings = [
            f
            for f in findings
            if f.get("severity") == SecurityLevel.HIGH.value
            and standard in f.get("compliance_standards", [])
        ]

        if critical_findings:
            compliance_status[standard] = "NON_COMPLIANT"
        elif high_findings:
            compliance_status[standard] = "PARTIAL_COMPLIANCE"
        else:
            compliance_status[standard] = "COMPLIANT"

    return compliance_status
