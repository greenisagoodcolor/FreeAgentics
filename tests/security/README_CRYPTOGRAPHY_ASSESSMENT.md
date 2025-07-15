# Comprehensive Cryptography Assessment Framework

## Overview

This comprehensive cryptography assessment framework provides production-ready security testing for the FreeAgentics platform. It validates all cryptographic implementations against industry standards and security best practices through multiple assessment approaches.

## Framework Components

### 1. Dynamic Cryptographic Testing (`test_cryptography_assessment.py`)
- **Algorithm Assessment**: Tests for weak algorithms (MD5, SHA-1, DES), validates strong algorithms (AES-256, SHA-256, RSA-2048+)
- **Key Management**: Validates key generation strength, storage security, rotation procedures
- **Encryption Implementation**: Tests symmetric/asymmetric encryption, digital signatures, authenticated encryption
- **SSL/TLS Security**: Protocol validation, cipher suite assessment, certificate validation
- **Vulnerability Testing**: Timing attacks, side-channel resistance, randomness quality, oracle attacks

### 2. Static Code Analysis (`crypto_static_analysis.py`)
- **Pattern Matching**: Detects weak algorithms, hardcoded secrets, insecure random usage
- **AST Analysis**: Python-specific cryptographic vulnerability detection
- **CWE Mapping**: Maps vulnerabilities to Common Weakness Enumeration
- **OWASP Classification**: Categorizes findings by OWASP Top 10

### 3. Configuration Security Review
- **Environment Variables**: Validates production secrets, configuration security
- **File Permissions**: Checks cryptographic asset permissions
- **TLS Configuration**: Reviews SSL/TLS deployment settings
- **Key Management**: Assesses key storage and lifecycle policies

### 4. Compliance Validation
- **NIST SP 800-57**: Key management and algorithm compliance
- **FIPS 140-2**: Cryptographic module validation
- **OWASP Guidelines**: Cryptographic storage best practices
- **RFC 8446**: TLS 1.3 security requirements

### 5. Executive Reporting
- **Risk Assessment**: Overall security posture evaluation
- **Business Impact**: Risk-to-business impact analysis
- **Remediation Roadmap**: Prioritized action plan with timelines
- **Compliance Status**: Standards compliance mapping

## Quick Start

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements-production.txt

# Additional security testing dependencies
pip install pytest cryptography bcrypt passlib pyjwt
```

### Running Individual Components

#### 1. Dynamic Cryptographic Testing
```bash
# Run comprehensive dynamic tests
python tests/security/run_cryptography_assessment.py --output-dir ./reports --verbose

# Run with pytest
pytest tests/security/test_cryptography_assessment.py -v
```

#### 2. Static Code Analysis
```bash
# Analyze entire project
python tests/security/crypto_static_analysis.py /home/green/FreeAgentics --output ./static_analysis.json --verbose

# Analyze specific directory
python tests/security/crypto_static_analysis.py ./auth --format text
```

#### 3. Comprehensive Assessment Suite
```bash
# Run complete assessment
python tests/security/comprehensive_crypto_security_suite.py --project-root . --output-dir ./crypto_reports --verbose

# Quick assessment (faster, fewer tests)
python tests/security/comprehensive_crypto_security_suite.py --quick
```

## Detailed Usage

### Configuration

The assessment framework uses `cryptography_assessment_config.py` for standards and criteria:

```python
from tests.security.cryptography_assessment_config import (
    SecurityLevel,
    ComplianceStandard,
    CryptographyStandards
)

# Check if algorithm meets standards
standard = CryptographyStandards.HASH_ALGORITHMS.get('sha256')
print(f"SHA-256 Security Level: {standard.security_level}")
```

### Custom Assessment

```python
from tests.security.test_cryptography_assessment import CryptographicAlgorithmAssessment

# Run specific assessment
assessor = CryptographicAlgorithmAssessment()
results = assessor.assess_hash_algorithms()

for passed in results['passed']:
    print(f"✓ {passed}")

for failed in results['failed']:
    print(f"✗ {failed}")
```

### Integration with CI/CD

```yaml
# .github/workflows/crypto-security.yml
name: Cryptography Security Assessment

on: [push, pull_request]

jobs:
  crypto-assessment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-production.txt
          pip install pytest cryptography
      
      - name: Run Cryptographic Assessment
        run: |
          python tests/security/run_cryptography_assessment.py --output-dir ./crypto-reports
      
      - name: Upload Assessment Reports
        uses: actions/upload-artifact@v3
        with:
          name: crypto-assessment-reports
          path: ./crypto-reports/
```

## Assessment Categories

### 1. Algorithm Security
- **Weak Algorithms**: MD5, SHA-1, DES, 3DES, RC4
- **Strong Algorithms**: SHA-256/384/512, AES-256-GCM, ChaCha20-Poly1305, RSA-2048+, ECDSA P-256+
- **Key Sizes**: Validates minimum key lengths per NIST guidelines
- **Algorithm Configuration**: Proper modes, padding, parameters

### 2. Key Management
- **Generation**: Cryptographically secure random number generation
- **Storage**: File permissions, encryption at rest, HSM integration
- **Lifecycle**: Rotation policies, expiration, revocation
- **Distribution**: Secure key exchange, certificate management

### 3. Implementation Security
- **Authenticated Encryption**: GCM, CCM, Poly1305 modes
- **Timing Resistance**: Constant-time operations
- **Side-Channel Protection**: Power analysis, cache timing
- **Error Handling**: Information leakage prevention

### 4. Protocol Security
- **TLS Configuration**: Version requirements, cipher suites
- **Certificate Validation**: Chain verification, hostname checking
- **Perfect Forward Secrecy**: Ephemeral key exchange
- **Certificate Pinning**: Mobile app security

## Vulnerability Categories

### Critical Severity
- Use of broken cryptographic algorithms (MD5, SHA-1, DES)
- Hardcoded cryptographic secrets
- Disabled certificate verification
- Use of unauthenticated encryption

### High Severity
- Weak key sizes (RSA-1024, weak ECDSA curves)
- Use of weak random number generators
- Missing authentication in encryption
- Timing attack vulnerabilities

### Medium Severity
- Suboptimal algorithm configurations
- Missing perfect forward secrecy
- Weak key derivation parameters
- Information disclosure in error messages

### Low Severity
- Use of deprecated but not broken algorithms
- Missing security headers
- Suboptimal cipher suite ordering
- Documentation and policy issues

## Compliance Standards

### NIST SP 800-57
- **Key Management**: Lifecycle, storage, distribution
- **Algorithm Selection**: Approved algorithms and key sizes
- **Security Levels**: 80, 112, 128, 192, 256-bit equivalent security

### FIPS 140-2
- **Cryptographic Modules**: Validated implementations
- **Key Management**: Security requirements
- **Physical Security**: Tamper evidence and response

### OWASP Cryptographic Storage
- **Data Protection**: Encryption requirements
- **Key Management**: Storage and handling
- **Implementation**: Secure coding practices

### RFC 8446 (TLS 1.3)
- **Protocol Security**: Version requirements
- **Cipher Suites**: Approved algorithms
- **Perfect Forward Secrecy**: Mandatory requirement

## Report Structure

### Executive Summary
```json
{
  "overall_risk_level": "MEDIUM",
  "security_score": 78.5,
  "total_findings": 23,
  "critical_findings": 0,
  "high_severity_findings": 3,
  "key_recommendations": [
    "Upgrade RSA key size to 2048 bits minimum",
    "Replace SHA-1 usage with SHA-256",
    "Implement certificate pinning"
  ]
}
```

### Detailed Findings
```json
{
  "finding_id": "CRYPTO-001",
  "severity": "high",
  "category": "weak_algorithm",
  "description": "Use of SHA-1 hash algorithm detected",
  "file_path": "auth/legacy_auth.py",
  "line_number": 45,
  "recommendation": "Replace SHA-1 with SHA-256 or stronger",
  "cwe_id": "CWE-327",
  "owasp_category": "A2:2021 – Cryptographic Failures"
}
```

### Remediation Roadmap
```json
{
  "immediate_actions": {
    "timeline": "24-48 hours",
    "priority": "CRITICAL",
    "actions": [
      "Replace hardcoded JWT secret with environment variable",
      "Disable SSL 3.0 and TLS 1.0 protocols"
    ]
  },
  "short_term_actions": {
    "timeline": "1-2 weeks",
    "priority": "HIGH",
    "actions": [
      "Upgrade RSA keys to 2048 bits minimum",
      "Implement authenticated encryption"
    ]
  }
}
```

## Customization

### Adding Custom Algorithms
```python
# In cryptography_assessment_config.py
CUSTOM_ALGORITHMS = {
    'my_custom_hash': AlgorithmStandard(
        name='MyCustomHash',
        minimum_key_size=256,
        recommended_key_size=256,
        security_level=SecurityLevel.HIGH,
        compliance_standards=[ComplianceStandard.NIST_SP_800_57]
    )
}
```

### Custom Vulnerability Patterns
```python
# In crypto_static_analysis.py
CUSTOM_PATTERNS = {
    VulnerabilityType.CUSTOM_ISSUE: [
        {
            'pattern': r'my_insecure_function\(',
            'severity': SecurityLevel.HIGH,
            'description': 'Use of custom insecure function',
            'recommendation': 'Replace with secure alternative'
        }
    ]
}
```

### Custom Compliance Standards
```python
# Add custom standard
class CustomStandard(Enum):
    COMPANY_CRYPTO_POLICY = "company_crypto_policy"

# Map requirements
COMPANY_REQUIREMENTS = [
    SecurityRequirement(
        requirement_id="COMPANY-001",
        description="All encryption must use AES-256",
        security_level=SecurityLevel.HIGH,
        compliance_standards=[CustomStandard.COMPANY_CRYPTO_POLICY]
    )
]
```

## Best Practices

### For Developers
1. **Run assessments regularly** during development
2. **Address critical findings immediately**
3. **Use approved algorithms** from the standards list
4. **Implement proper key management** from the start
5. **Follow secure coding practices** for cryptographic operations

### For Security Teams
1. **Integrate into CI/CD** pipelines
2. **Review assessment reports** regularly
3. **Maintain compliance mappings** up to date
4. **Track remediation progress** over time
5. **Customize standards** for organizational needs

### For Operations Teams
1. **Monitor certificate expiration** dates
2. **Implement key rotation** procedures
3. **Maintain secure configurations** in production
4. **Respond to security findings** promptly
5. **Document incident response** procedures

## Troubleshooting

### Common Issues

#### ImportError: Module not found
```bash
# Ensure all dependencies are installed
pip install -r requirements-production.txt
pip install pytest cryptography bcrypt passlib pyjwt

# Add project root to Python path
export PYTHONPATH=/home/green/FreeAgentics:$PYTHONPATH
```

#### Permission Denied on Key Files
```bash
# Set proper permissions
chmod 600 auth/keys/jwt_private.pem
chmod 644 auth/keys/jwt_public.pem
```

#### Assessment Takes Too Long
```bash
# Use quick mode
python comprehensive_crypto_security_suite.py --quick

# Run specific components only
python run_cryptography_assessment.py
```

### Getting Help

For issues or questions:
1. Check the assessment logs for detailed error messages
2. Review the configuration files for proper settings
3. Ensure all dependencies are correctly installed
4. Verify file permissions for cryptographic assets

## Security Considerations

- **Assessment Data**: Reports may contain sensitive information
- **Test Environment**: Run in isolated environments when possible
- **Production Impact**: Some tests may affect performance
- **False Positives**: Review findings for context appropriateness
- **Compliance**: Verify standards alignment with organizational requirements

## License and Disclaimer

This assessment framework is provided for security testing purposes. Users are responsible for:
- Validating findings in their specific context
- Implementing appropriate remediation measures
- Maintaining compliance with applicable regulations
- Regular updates and maintenance of the framework

The framework provides guidance based on industry standards but does not guarantee complete security or compliance.