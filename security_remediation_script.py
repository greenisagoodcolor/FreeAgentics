#!/usr/bin/env python3
"""
Security Remediation Script for FreeAgentics
Addresses critical security issues identified by SECURITY-PALADIN
"""

import re
from pathlib import Path

# Color codes for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(message: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{message}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}✅ {message}{RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{YELLOW}⚠️  {message}{RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{RED}❌ {message}{RESET}")


def fix_jwt_verification():
    """Fix JWT verification issues by removing verify=False."""
    print_header("Fixing JWT Verification Issues")

    files_to_fix = [
        ("auth/jwt_handler.py", 468),
        ("auth/security_implementation.py", 490),
        ("auth/security_implementation.py", 810),
    ]

    for filepath, line_num in files_to_fix:
        try:
            path = Path(filepath)
            if not path.exists():
                print_warning(f"File not found: {filepath}")
                continue

            content = path.read_text()

            # Replace jwt.decode with verify=False
            pattern = r"jwt\.decode\([^)]+verify\s*=\s*False[^)]*\)"
            matches = list(re.finditer(pattern, content))

            if matches:
                # Replace verify=False with proper verification
                new_content = re.sub(
                    r"(jwt\.decode\([^,]+,[^,]+)(,\s*verify\s*=\s*False)([^)]*\))",
                    r"\1\3",
                    content,
                )

                # Also ensure options are properly set
                new_content = re.sub(
                    r"(jwt\.decode\([^)]+)\)",
                    r'\1, options={"verify_signature": True})',
                    new_content,
                )

                path.write_text(new_content)
                print_success(f"Fixed JWT verification in {filepath}")
            else:
                print_warning(f"No verify=False found in {filepath}")

        except Exception as e:
            print_error(f"Error fixing {filepath}: {e}")


def add_password_validation():
    """Add comprehensive password validation."""
    print_header("Adding Password Validation")

    password_validator_code = '''
import re
from typing import List, Tuple


class PasswordValidator:
    """Comprehensive password validation following security best practices."""

    MIN_LENGTH = 12
    COMMON_PASSWORDS = {
        "password", "123456", "password123", "admin", "letmein",
        "qwerty", "abc123", "111111", "123123", "password1"
    }

    @classmethod
    def validate(cls, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password strength.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Length check
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters long")

        # Complexity checks
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if not re.search(r'[0-9]', password):
            errors.append("Password must contain at least one number")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        # Common password check
        if password.lower() in cls.COMMON_PASSWORDS:
            errors.append("Password is too common and easily guessable")

        # Sequential character check
        if cls._has_sequential_chars(password):
            errors.append("Password contains sequential characters")

        return len(errors) == 0, errors

    @staticmethod
    def _has_sequential_chars(password: str, threshold: int = 3) -> bool:
        """Check for sequential characters like 'abc' or '123'."""
        for i in range(len(password) - threshold + 1):
            substr = password[i:i + threshold]

            # Check ascending
            if all(ord(substr[j]) == ord(substr[j-1]) + 1 for j in range(1, len(substr))):
                return True

            # Check descending
            if all(ord(substr[j]) == ord(substr[j-1]) - 1 for j in range(1, len(substr))):
                return True

        return False
'''

    try:
        # Create password validator file
        validator_path = Path("auth/password_validator.py")
        validator_path.write_text(password_validator_code)
        print_success("Created password_validator.py")

        # Update auth module to use password validator
        auth_init_path = Path("auth/__init__.py")
        if auth_init_path.exists():
            content = auth_init_path.read_text()
            if "PasswordValidator" not in content:
                content += "\nfrom .password_validator import PasswordValidator\n"
                auth_init_path.write_text(content)
                print_success("Updated auth/__init__.py to include PasswordValidator")

    except Exception as e:
        print_error(f"Error creating password validator: {e}")


def secure_jwt_keys():
    """Move JWT keys to environment variables."""
    print_header("Securing JWT Keys")

    # Check if private key exists
    private_key_path = Path("auth/keys/jwt_private.pem")
    Path("auth/keys/jwt_public.pem")

    if private_key_path.exists():
        print_warning("CRITICAL: Private key found in repository!")
        print_warning("Action required: Remove this file and rotate keys immediately")

        # Create .gitignore entry
        gitignore_path = Path(".gitignore")
        try:
            if gitignore_path.exists():
                content = gitignore_path.read_text()
                if "auth/keys/" not in content:
                    content += (
                        "\n# JWT Keys - NEVER commit these!\nauth/keys/*.pem\nauth/keys/*.key\n"
                    )
                    gitignore_path.write_text(content)
                    print_success("Added JWT keys to .gitignore")
        except Exception as e:
            print_error(f"Error updating .gitignore: {e}")

        # Create key management documentation
        key_mgmt_doc = """# JWT Key Management Guide

## CRITICAL SECURITY NOTICE

JWT private keys have been detected in the repository. This is a critical security breach.

### Immediate Actions Required:

1. **Remove keys from repository:**
   ```bash
   git rm -r auth/keys/
   git commit -m "Remove exposed JWT keys"
   git push
   ```

2. **Rotate all JWT keys immediately**

3. **Use environment variables:**
   ```python
   # In your JWT handler:
   import os
   from cryptography.hazmat.primitives import serialization

   # Load from environment
   private_key = os.environ['JWT_PRIVATE_KEY']
   public_key = os.environ['JWT_PUBLIC_KEY']
   ```

4. **For production:**
   - Use a proper secret management service (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Never store keys in code or configuration files
   - Rotate keys regularly

5. **Generate new keys:**
   ```bash
   # Generate new RSA key pair
   openssl genrsa -out jwt_private.pem 4096
   openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem

   # Store in environment variables
   export JWT_PRIVATE_KEY="$(cat jwt_private.pem)"
   export JWT_PUBLIC_KEY="$(cat jwt_public.pem)"
   ```
"""

        try:
            key_mgmt_path = Path("auth/JWT_KEY_MANAGEMENT.md")
            key_mgmt_path.write_text(key_mgmt_doc)
            print_success("Created JWT key management guide")
        except Exception as e:
            print_error(f"Error creating key management guide: {e}")


def add_security_headers_middleware():
    """Add comprehensive security headers middleware."""
    print_header("Adding Security Headers Middleware")

    security_headers_code = '''
from fastapi import Request
from fastapi.responses import Response
from typing import Callable


class SecurityHeadersMiddleware:
    """Middleware to add comprehensive security headers to all responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp

        # Strict Transport Security (if HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return response
'''

    try:
        # Create security headers middleware
        headers_path = Path("api/middleware/comprehensive_security_headers.py")
        headers_path.parent.mkdir(exist_ok=True)
        headers_path.write_text(security_headers_code)
        print_success("Created comprehensive security headers middleware")

        # Add integration note
        print_warning("Remember to add this middleware to your FastAPI app:")
        print("    app.add_middleware(SecurityHeadersMiddleware)")

    except Exception as e:
        print_error(f"Error creating security headers middleware: {e}")


def create_security_checklist():
    """Create a security checklist for ongoing maintenance."""
    print_header("Creating Security Checklist")

    checklist = """# Security Checklist for FreeAgentics

## Daily Checks
- [ ] Review security monitoring alerts
- [ ] Check for new dependency vulnerabilities: `pip-audit`
- [ ] Monitor failed authentication attempts

## Weekly Checks
- [ ] Run security scans: `bandit -r . && semgrep --config=auto`
- [ ] Review and update dependencies
- [ ] Check for unusual API access patterns
- [ ] Verify all endpoints have proper authentication

## Monthly Checks
- [ ] Full security audit
- [ ] Review and rotate API keys
- [ ] Update security documentation
- [ ] Security training for team members

## Before Each Release
- [ ] Run full security test suite
- [ ] Verify no secrets in code: `git secrets --scan`
- [ ] Check OWASP Top 10 compliance
- [ ] Perform penetration testing
- [ ] Review Docker image security

## Incident Response
1. Isolate affected systems
2. Assess impact and scope
3. Contain the incident
4. Eradicate the threat
5. Recover systems
6. Document lessons learned

## Security Contacts
- Security Lead: [Add contact]
- Incident Response: [Add contact]
- External Security Team: [Add contact]
"""

    try:
        checklist_path = Path("SECURITY_CHECKLIST.md")
        checklist_path.write_text(checklist)
        print_success("Created security checklist")
    except Exception as e:
        print_error(f"Error creating security checklist: {e}")


def main():
    """Run all security remediation tasks."""
    print_header("FreeAgentics Security Remediation Script")
    print("This script addresses critical security issues identified by SECURITY-PALADIN\n")

    # Run remediation tasks
    fix_jwt_verification()
    add_password_validation()
    secure_jwt_keys()
    add_security_headers_middleware()
    create_security_checklist()

    print_header("Remediation Summary")
    print_success("Security remediation tasks completed!")
    print_warning("CRITICAL: Remove JWT private keys from repository immediately!")
    print_warning("Remember to test all changes before deploying to production")
    print("\nNext steps:")
    print("1. Remove auth/keys/*.pem files from repository")
    print("2. Rotate all JWT keys")
    print("3. Update authentication to use PasswordValidator")
    print("4. Add SecurityHeadersMiddleware to FastAPI app")
    print("5. Run security tests to verify fixes")


if __name__ == "__main__":
    main()
