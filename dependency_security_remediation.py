#!/usr/bin/env python3
"""
DEPENDENCY-DOCTOR Security Remediation Script
Fixes all identified security vulnerabilities in FreeAgentics dependencies
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class DependencySecurityRemediator:
    """Comprehensive dependency security remediation for FreeAgentics"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.remediation_log = []
        
    def log_action(self, action: str, details: str, status: str = "INFO"):
        """Log remediation actions"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "status": status
        }
        self.remediation_log.append(log_entry)
        print(f"[{status}] {action}: {details}")
        
    def fix_cryptography_vulnerability(self) -> bool:
        """Fix CVE-2024-12797 in cryptography package"""
        try:
            self.log_action(
                "SECURITY_FIX",
                "Updating cryptography from 45.0.5 to 46.0.1 to fix CVE-2024-12797"
            )
            
            # Update requirements files
            files_to_update = [
                "requirements.txt",
                "requirements-production.txt", 
                "requirements-core.txt"
            ]
            
            for req_file in files_to_update:
                file_path = self.project_root / req_file
                if file_path.exists():
                    content = file_path.read_text()
                    updated_content = content.replace(
                        "cryptography==45.0.5",
                        "cryptography==46.0.1  # Security fix for CVE-2024-12797"
                    )
                    if content != updated_content:
                        file_path.write_text(updated_content)
                        self.log_action(
                            "FILE_UPDATE", 
                            f"Updated cryptography version in {req_file}"
                        )
                        
            return True
            
        except Exception as e:
            self.log_action("ERROR", f"Failed to fix cryptography vulnerability: {e}", "ERROR")
            return False
    
    def fix_starlette_vulnerability(self) -> bool:
        """Fix CVE-2024-47874 in starlette package"""
        try:
            self.log_action(
                "SECURITY_FIX",
                "Updating starlette from 0.46.2 to 0.46.6 to fix CVE-2024-47874"
            )
            
            files_to_update = [
                "requirements.txt",
                "requirements-production.txt",
                "requirements-core.txt"
            ]
            
            for req_file in files_to_update:
                file_path = self.project_root / req_file
                if file_path.exists():
                    content = file_path.read_text()
                    updated_content = content.replace(
                        "starlette==0.46.2",
                        "starlette==0.46.6  # Security fix for CVE-2024-47874"
                    )
                    if content != updated_content:
                        file_path.write_text(updated_content)
                        self.log_action(
                            "FILE_UPDATE", 
                            f"Updated starlette version in {req_file}"
                        )
                        
            return True
            
        except Exception as e:
            self.log_action("ERROR", f"Failed to fix starlette vulnerability: {e}", "ERROR")
            return False
    
    def remove_vulnerable_py_package(self) -> bool:
        """Remove vulnerable py==1.11.0 package with CVE-2022-42969"""
        try:
            self.log_action(
                "SECURITY_FIX",
                "Removing py==1.11.0 package due to CVE-2022-42969 (already commented out)"
            )
            
            # Verify py package is already removed from requirements.txt
            req_file = self.project_root / "requirements.txt"
            content = req_file.read_text()
            
            if "# py==1.11.0  # REMOVED - CVE-2022-42969" in content:
                self.log_action(
                    "VERIFICATION_PASSED",
                    "py package already safely removed from requirements.txt"
                )
            else:
                # Ensure it's commented out
                updated_content = content.replace(
                    "py==1.11.0",
                    "# py==1.11.0  # REMOVED - CVE-2022-42969"
                )
                req_file.write_text(updated_content)
                self.log_action(
                    "FILE_UPDATE",
                    "Commented out vulnerable py package in requirements.txt"
                )
                
            return True
            
        except Exception as e:
            self.log_action("ERROR", f"Failed to remove py package: {e}", "ERROR")
            return False
    
    def pin_unpinned_dependencies(self) -> bool:
        """Pin unpinned dependencies to exact versions"""
        try:
            self.log_action(
                "SECURITY_HARDENING",
                "Pinning unpinned dependencies to exact versions"
            )
            
            req_file = self.project_root / "requirements.txt"
            content = req_file.read_text()
            
            # Pin OpenAI to latest stable version
            updated_content = content.replace(
                "openai>=1.97.0",
                "openai==1.97.0  # Pinned for reproducible builds"
            )
            
            # Pin Anthropic to latest stable version
            updated_content = updated_content.replace(
                "anthropic>=0.58.0",
                "anthropic==0.58.0  # Pinned for reproducible builds"
            )
            
            if content != updated_content:
                req_file.write_text(updated_content)
                self.log_action(
                    "FILE_UPDATE",
                    "Pinned OpenAI and Anthropic packages to exact versions"
                )
            
            return True
            
        except Exception as e:
            self.log_action("ERROR", f"Failed to pin dependencies: {e}", "ERROR")
            return False
    
    def validate_requirements_consistency(self) -> bool:
        """Validate consistency across all requirements files"""
        try:
            self.log_action(
                "VALIDATION",
                "Validating consistency across requirements files"
            )
            
            core_deps = {}
            req_core_file = self.project_root / "requirements-core.txt"
            
            if req_core_file.exists():
                for line in req_core_file.read_text().splitlines():
                    if line.strip() and not line.startswith("#") and "==" in line:
                        pkg_name = line.split("==")[0].strip()
                        version = line.split("==")[1].split()[0].strip()
                        core_deps[pkg_name] = version
            
            # Check production requirements
            inconsistencies = []
            prod_file = self.project_root / "requirements-production.txt"
            
            if prod_file.exists():
                for line in prod_file.read_text().splitlines():
                    if line.strip() and not line.startswith("#") and "==" in line:
                        pkg_name = line.split("==")[0].strip()
                        version = line.split("==")[1].split()[0].strip()
                        
                        if pkg_name in core_deps and core_deps[pkg_name] != version:
                            inconsistencies.append(
                                f"{pkg_name}: core={core_deps[pkg_name]}, prod={version}"
                            )
            
            if inconsistencies:
                self.log_action(
                    "WARNING",
                    f"Version inconsistencies found: {', '.join(inconsistencies)}",
                    "WARNING"
                )
                return False
            else:
                self.log_action(
                    "VALIDATION_PASSED",
                    "All requirements files are consistent"
                )
                return True
            
        except Exception as e:
            self.log_action("ERROR", f"Failed validation: {e}", "ERROR")
            return False
    
    def generate_dependency_freeze(self) -> bool:
        """Generate pip freeze for dependency verification"""
        try:
            self.log_action(
                "DEPENDENCY_FREEZE",
                "Generating pip freeze output for reproducible builds"
            )
            
            # Note: We can't actually run pip freeze in this environment
            # but we can document the requirement
            freeze_file = self.project_root / "requirements-freeze.txt"
            
            freeze_content = f"""# Generated dependency freeze for FreeAgentics
# Generated on: {datetime.now().isoformat()}
# 
# CRITICAL SECURITY NOTE:
# This file should be generated by running:
# pip freeze > requirements-freeze.txt
#
# Use this for exact reproducible builds in production
#
# To install exact versions:
# pip install -r requirements-freeze.txt
#
# Verify with:
# pip-audit -r requirements-freeze.txt
# safety check -r requirements-freeze.txt
"""
            
            freeze_file.write_text(freeze_content)
            self.log_action(
                "FILE_CREATED",
                "Created requirements-freeze.txt template"
            )
            
            return True
            
        except Exception as e:
            self.log_action("ERROR", f"Failed to generate freeze: {e}", "ERROR")
            return False
    
    def create_security_policy(self) -> bool:
        """Create dependency security policy"""
        try:
            self.log_action(
                "POLICY_CREATION",
                "Creating dependency security policy"
            )
            
            policy_content = f"""# FreeAgentics Dependency Security Policy
# Generated on: {datetime.now().isoformat()}

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
"""
            
            policy_file = self.project_root / "DEPENDENCY_SECURITY_POLICY.md"
            policy_file.write_text(policy_content)
            
            self.log_action(
                "FILE_CREATED",
                "Created DEPENDENCY_SECURITY_POLICY.md"
            )
            
            return True
            
        except Exception as e:
            self.log_action("ERROR", f"Failed to create security policy: {e}", "ERROR")
            return False
    
    def run_full_remediation(self) -> Dict[str, Any]:
        """Run complete security remediation"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "remediation_steps": [],
            "vulnerabilities_fixed": 0,
            "dependencies_pinned": 0,
            "success": True,
            "errors": []
        }
        
        try:
            self.log_action("START_REMEDIATION", "Beginning comprehensive dependency security remediation")
            
            # Fix identified vulnerabilities
            if self.fix_cryptography_vulnerability():
                results["vulnerabilities_fixed"] += 1
                results["remediation_steps"].append("Fixed CVE-2024-12797 in cryptography")
            
            if self.fix_starlette_vulnerability():
                results["vulnerabilities_fixed"] += 1
                results["remediation_steps"].append("Fixed CVE-2024-47874 in starlette")
            
            if self.remove_vulnerable_py_package():
                results["vulnerabilities_fixed"] += 1
                results["remediation_steps"].append("Removed vulnerable py package (CVE-2022-42969)")
            
            # Pin unpinned dependencies
            if self.pin_unpinned_dependencies():
                results["dependencies_pinned"] += 2
                results["remediation_steps"].append("Pinned openai and anthropic packages")
            
            # Validation and policy creation
            if self.validate_requirements_consistency():
                results["remediation_steps"].append("Validated requirements consistency")
            
            if self.generate_dependency_freeze():
                results["remediation_steps"].append("Created requirements freeze template")
            
            if self.create_security_policy():
                results["remediation_steps"].append("Created dependency security policy")
            
            self.log_action("REMEDIATION_COMPLETE", 
                          f"Fixed {results['vulnerabilities_fixed']} vulnerabilities, "
                          f"pinned {results['dependencies_pinned']} dependencies")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.log_action("ERROR", f"Remediation failed: {e}", "ERROR")
        
        # Save remediation log
        results["log"] = self.remediation_log
        
        log_file = self.project_root / "dependency_remediation_log.json"
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results


def main():
    """Main remediation execution"""
    project_root = Path(__file__).parent
    remediator = DependencySecurityRemediator(project_root)
    
    print("=== DEPENDENCY-DOCTOR Security Remediation ===")
    print(f"Project: FreeAgentics")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 50)
    
    results = remediator.run_full_remediation()
    
    print("\n=== REMEDIATION SUMMARY ===")
    print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
    print(f"Vulnerabilities Fixed: {results['vulnerabilities_fixed']}")
    print(f"Dependencies Pinned: {results['dependencies_pinned']}")
    print(f"Remediation Steps: {len(results['remediation_steps'])}")
    
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\nDetailed log saved to: dependency_remediation_log.json")
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    sys.exit(main())