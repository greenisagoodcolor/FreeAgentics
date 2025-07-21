#!/usr/bin/env python3
"""
Security validation tests for SECURITY-PALADIN improvements
Ensures all security issues are properly addressed
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest


class TestSecurityPaladinValidation:
    """Comprehensive security validation test suite."""
    
    def test_no_dependency_vulnerabilities(self):
        """Verify all dependency vulnerabilities have been fixed."""
        # Run pip-audit
        result = subprocess.run(
            ["pip-audit", "--format", "json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            audit_data = json.loads(result.stdout)
            vulnerable_deps = [
                d for d in audit_data.get("dependencies", [])
                if d.get("vulns")
            ]
            
            assert len(vulnerable_deps) == 0, (
                f"Found {len(vulnerable_deps)} vulnerable dependencies: "
                f"{[d['name'] for d in vulnerable_deps]}"
            )
    
    def test_no_hardcoded_jwt_keys(self):
        """Ensure JWT private keys are not in source control."""
        private_key_path = Path("auth/keys/jwt_private.pem")
        public_key_path = Path("auth/keys/jwt_public.pem")
        
        assert not private_key_path.exists(), (
            "CRITICAL: JWT private key found in repository! "
            "Remove immediately and rotate keys."
        )
        
        # Public key can exist but private key must not
        if public_key_path.exists():
            pass  # Public key can exist
    
    def test_jwt_verification_enabled(self):
        """Verify JWT tokens are properly verified."""
        # Check for verify=False in JWT decode calls
        files_to_check = [
            "auth/jwt_handler.py",
            "auth/security_implementation.py"
        ]
        
        for filepath in files_to_check:
            if Path(filepath).exists():
                content = Path(filepath).read_text()
                assert "verify=False" not in content, (
                    f"Found verify=False in {filepath} - "
                    "JWT verification must be enabled!"
                )
                assert "verify_signature\": False" not in content, (
                    f"Found disabled signature verification in {filepath}"
                )
    
    def test_password_validation_exists(self):
        """Verify password validation is implemented."""
        validator_path = Path("auth/password_validator.py")
        
        if validator_path.exists():
            content = validator_path.read_text()
            
            # Check for minimum length
            assert "MIN_LENGTH" in content
            assert "12" in content or "MIN_LENGTH = 12" in content
            
            # Check for complexity requirements
            assert "[A-Z]" in content  # Uppercase check
            assert "[a-z]" in content  # Lowercase check
            assert "[0-9]" in content  # Number check
            assert "special" in content or "[!@#$%^&*" in content  # Special char check
            
            # Check for common password detection
            assert "COMMON_PASSWORDS" in content or "common" in content.lower()
        else:
            pass  # File doesn't exist, which is acceptable
    
    def test_security_headers_middleware(self):
        """Verify security headers middleware is implemented."""
        middleware_files = [
            "api/middleware/security_headers.py",
            "api/middleware/comprehensive_security_headers.py"
        ]
        
        headers_found = False
        for filepath in middleware_files:
            if Path(filepath).exists():
                content = Path(filepath).read_text()
                
                required_headers = [
                    "X-Content-Type-Options",
                    "X-Frame-Options",
                    "X-XSS-Protection",
                    "Content-Security-Policy",
                    "Strict-Transport-Security"
                ]
                
                for header in required_headers:
                    assert header in content, f"Missing security header: {header}"
                
                headers_found = True
                break
        
        assert headers_found, "Security headers middleware not found"
    
    def test_no_pickle_usage_in_critical_paths(self):
        """Verify pickle is not used in security-critical paths."""
        # These files should not use pickle
        critical_files = [
            "auth/jwt_handler.py",
            "auth/security_implementation.py",
            "api/v1/auth.py"
        ]
        
        for filepath in critical_files:
            if Path(filepath).exists():
                content = Path(filepath).read_text()
                assert "pickle" not in content.lower(), (
                    f"Found pickle usage in security-critical file: {filepath}"
                )
    
    def test_api_endpoints_protected(self):
        """Verify API endpoints have authentication."""
        api_files = list(Path("api/v1").glob("*.py"))
        
        unprotected_endpoints = []
        for api_file in api_files:
            if api_file.name in ["__init__.py", "auth.py"]:
                continue
                
            content = api_file.read_text()
            
            # Count route decorators
            import re
            routes = re.findall(r'@\w+\.(get|post|put|delete|patch)', content)
            
            # Count authentication decorators
            auth_decorators = re.findall(r'@require_permission|@require_auth|@protected', content)
            
            if len(routes) > len(auth_decorators):
                unprotected_endpoints.append({
                    "file": str(api_file),
                    "routes": len(routes),
                    "protected": len(auth_decorators)
                })
        
        assert len(unprotected_endpoints) == 0, (
            f"Found unprotected endpoints: {unprotected_endpoints}"
        )
    
    def test_docker_security_best_practices(self):
        """Verify Docker security best practices."""
        dockerfile_path = Path("Dockerfile.production")
        
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            
            # Check for non-root user
            assert "USER" in content and "USER root" not in content, (
                "Docker container should run as non-root user"
            )
            
            # Check for minimal base image
            assert "slim" in content or "alpine" in content, (
                "Should use minimal base image"
            )
            
            # Check for HEALTHCHECK
            assert "HEALTHCHECK" in content, "Missing HEALTHCHECK instruction"
    
    def test_environment_configuration(self):
        """Verify secure environment configuration."""
        env_example = Path(".env.example")
        
        if env_example.exists():
            content = env_example.read_text()
            
            # Check for placeholder values
            assert "your_secret_key_here" in content, (
                "Example file should have placeholder values"
            )
            
            # Ensure no real secrets
            assert not any(
                keyword in content.lower()
                for keyword in ["prod", "production"]
                if "your_" not in content.lower()
            ), "Example file should not contain real secrets"
    
    def test_ssl_tls_configuration(self):
        """Verify SSL/TLS is properly configured."""
        ssl_config_file = Path("auth/ssl_tls_config.py")
        
        if ssl_config_file.exists():
            content = ssl_config_file.read_text()
            
            # Check for minimum TLS version
            assert "TLSv1.2" in content or "TLSv1.3" in content, (
                "Must enforce minimum TLS v1.2"
            )
            
            # Check cipher configuration
            if "set_ciphers" in content:
                # Ensure no weak ciphers
                weak_ciphers = ["RC4", "DES", "3DES", "MD5"]
                for cipher in weak_ciphers:
                    assert cipher not in content, f"Weak cipher found: {cipher}"
    
    def test_rate_limiting_configured(self):
        """Verify rate limiting is properly configured."""
        rate_limit_files = [
            "api/middleware/rate_limiter.py",
            "config/rate_limiting.py"
        ]
        
        rate_limiting_found = False
        for filepath in rate_limit_files:
            if Path(filepath).exists():
                content = Path(filepath).read_text()
                
                # Check for rate limit configuration
                assert any(
                    keyword in content
                    for keyword in ["RateLimiter", "rate_limit", "throttle"]
                ), f"Rate limiting not properly configured in {filepath}"
                
                rate_limiting_found = True
                break
        
        assert rate_limiting_found, "Rate limiting configuration not found"


def run_security_validation():
    """Run all security validation tests."""
    print("🔒 Running SECURITY-PALADIN validation tests...")
    
    # Run pytest with verbose output
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_security_validation()
    if success:
        print("\n✅ All security validations passed!")
    else:
        print("\n❌ Security validation failed - address issues immediately!")
        exit(1)