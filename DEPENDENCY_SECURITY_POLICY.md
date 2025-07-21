# FreeAgentics Dependency Security Policy
# Generated on: 2025-07-21T12:05:32.095178

## Security Requirements

### Version Pinning
- ALL production dependencies MUST use exact version pinning (==)
- Development dependencies MAY use compatible version ranges (~=)
- NO floating dependencies (>=, >) allowed in production

### Vulnerability Monitoring
- Dependencies MUST be scanned with pip-audit before each release
- Safety scans MUST be performed on all requirements files
- Critical vulnerabilities MUST be patched within 24 hours
- High vulnerabilities MUST be patched within 7 days

### Approved Security Tools
- pip-audit (for CVE scanning)
- safety (for vulnerability database)
- bandit (for Python security linting)
- semgrep (for additional security patterns)

### Dependency Update Process
1. Run security scans on current dependencies
2. Update vulnerable packages to secure versions
3. Test compatibility with updated packages
4. Update ALL requirements files consistently
5. Regenerate pip freeze for reproducible builds
6. Validate deployment with updated dependencies

### Restricted Packages
The following packages are BANNED due to security issues:
- py==1.11.0 (CVE-2022-42969)

### Required Security Headers
All HTTP responses MUST include:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000; includeSubDomains

### Container Security
- Use minimal base images (Alpine Linux preferred)
- Run containers as non-root user
- Scan container images for vulnerabilities
- Keep base images updated

## Compliance Verification

Run these commands to verify compliance:

```bash
# CVE scanning
pip-audit -f json -o pip_audit_report.json

# Vulnerability scanning  
safety scan --json --output safety_report.json

# Version consistency check
python scripts/validate_dependencies.py

# Container security scan
docker scan freeagentics:latest
```

## Emergency Response

For critical security vulnerabilities:
1. Immediately update to secure version
2. Deploy emergency patch
3. Notify security team
4. Document incident
5. Review security processes
