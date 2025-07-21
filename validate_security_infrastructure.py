#!/usr/bin/env python3
"""
Validate Security Testing Infrastructure

Simple validation script to check if all security modules are properly structured.
"""

import sys
from pathlib import Path


def validate_file_structure():
    """Validate that all security files exist"""
    required_files = [
        "security/testing/sast_scanner.py",
        "security/testing/dast_integration.py",
        "security/testing/dependency_monitor.py",
        "security/testing/threat_intelligence.py",
        "tests/security/test_security_scanners.py",
        ".github/workflows/security-scan.yml",
        "scripts/security/generate_security_report.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required security files exist")
    return True


def validate_python_syntax():
    """Validate Python syntax of security modules"""
    security_files = [
        "security/testing/sast_scanner.py",
        "security/testing/dast_integration.py",
        "security/testing/dependency_monitor.py",
        "security/testing/threat_intelligence.py",
        "tests/security/test_security_scanners.py",
        "scripts/security/generate_security_report.py",
    ]

    for file_path in security_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Basic syntax check
            compile(content, file_path, "exec")
            print(f"✅ {file_path} syntax is valid")

        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            print(f"⚠️  Warning in {file_path}: {e}")

    return True


def validate_security_features():
    """Validate key security features are implemented"""
    features = {
        "SAST Scanner": {
            "file": "security/testing/sast_scanner.py",
            "checks": [
                "class BanditScanner",
                "class SemgrepScanner",
                "class SafetyScanner",
                "class SASTScanner",
                "OWASP",
            ],
        },
        "DAST Integration": {
            "file": "security/testing/dast_integration.py",
            "checks": [
                "class AuthenticationTester",
                "class APIScanner",
                "class ZAPScanner",
                "OWASP ZAP",
            ],
        },
        "Dependency Monitor": {
            "file": "security/testing/dependency_monitor.py",
            "checks": [
                "class DependencyScanner",
                "class VulnerabilityDatabase",
                "class UpdateManager",
                "CVE",
            ],
        },
        "Threat Intelligence": {
            "file": "security/testing/threat_intelligence.py",
            "checks": [
                "class ThreatIntelligenceEngine",
                "class OTXFeed",
                "class AbuseIPDBFeed",
                "AlienVault",
            ],
        },
    }

    for feature_name, feature_info in features.items():
        file_path = feature_info["file"]
        if not Path(file_path).exists():
            print(f"❌ {feature_name}: File {file_path} not found")
            continue

        try:
            with open(file_path, "r") as f:
                content = f.read()

            missing_checks = []
            for check in feature_info["checks"]:
                if check not in content:
                    missing_checks.append(check)

            if missing_checks:
                print(f"⚠️  {feature_name}: Missing features: {missing_checks}")
            else:
                print(f"✅ {feature_name}: All key features present")

        except Exception as e:
            print(f"❌ {feature_name}: Error reading file: {e}")


def validate_github_workflow():
    """Validate GitHub Actions workflow"""
    workflow_path = ".github/workflows/security-scan.yml"

    if not Path(workflow_path).exists():
        print(f"❌ GitHub workflow not found: {workflow_path}")
        return False

    try:
        with open(workflow_path, "r") as f:
            content = f.read()

        required_jobs = [
            "sast-scan",
            "dependency-scan",
            "container-scan",
            "secrets-scan",
            "security-report",
        ]

        missing_jobs = []
        for job in required_jobs:
            if job not in content:
                missing_jobs.append(job)

        if missing_jobs:
            print(f"⚠️  GitHub workflow missing jobs: {missing_jobs}")
        else:
            print("✅ GitHub workflow has all required security jobs")

        # Check for security tools
        tools = ["bandit", "semgrep", "safety", "trivy", "trufflehog"]
        present_tools = [tool for tool in tools if tool in content]
        print(f"✅ Security tools configured: {present_tools}")

        return len(missing_jobs) == 0

    except Exception as e:
        print(f"❌ Error validating GitHub workflow: {e}")
        return False


def validate_test_coverage():
    """Validate test coverage for security modules"""
    test_file = "tests/security/test_security_scanners.py"

    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    try:
        with open(test_file, "r") as f:
            content = f.read()

        test_classes = [
            "TestSASTScanner",
            "TestDependencyMonitor",
            "TestThreatIntelligence",
            "TestDASTIntegration",
            "TestOWASPCompliance",
        ]

        present_tests = [test for test in test_classes if test in content]
        missing_tests = [test for test in test_classes if test not in content]

        print(f"✅ Present test classes: {present_tests}")
        if missing_tests:
            print(f"⚠️  Missing test classes: {missing_tests}")

        return len(missing_tests) == 0

    except Exception as e:
        print(f"❌ Error validating tests: {e}")
        return False


def main():
    """Main validation function"""
    print("🔒 Validating FreeAgentics Security Testing Infrastructure")
    print("=" * 60)

    results = []

    # Run all validations
    results.append(("File Structure", validate_file_structure()))
    results.append(("Python Syntax", validate_python_syntax()))
    results.append(("Security Features", validate_security_features()))
    results.append(("GitHub Workflow", validate_github_workflow()))
    results.append(("Test Coverage", validate_test_coverage()))

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} validations passed")

    if passed == total:
        print("\n🎉 Security testing infrastructure is properly configured!")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install bandit semgrep safety")
        print("2. Configure API keys for threat intelligence feeds")
        print("3. Run security scans: python security/testing/sast_scanner.py")
        print("4. Set up GitHub Actions secrets for automation")
        return True
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
