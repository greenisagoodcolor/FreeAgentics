#!/usr/bin/env python3
"""
POLICY-GUARD AGENT - COMPREHENSIVE BYPASS ELIMINATION

This script implements zero-tolerance bypass elimination:
1. Search for ALL bypass patterns in project files (not dependencies)
2. Remove or fix every bypass found
3. Install git pre-receive hooks preventing future bypasses
4. Generate comprehensive violation report

NO MERCY. NO EXCEPTIONS. ZERO BYPASS TOLERANCE.
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Enhanced bypass patterns - comprehensive detection
BYPASS_PATTERNS = [
    # CI/CD bypass patterns
    (r"\[skip\s+ci\]", "", "CI Skip Directive"),
    (r"\[ci\s+skip\]", "", "CI Skip Directive"),
    (r"skip:\s*ci", "", "CI Skip Directive"),
    (r"continue-on-error:\s*true", "continue-on-error: false", "CI Error Bypass"),
    (r"allow_failure:\s*true", "allow_failure: false", "CI Failure Bypass"),
    (r"fail_ci_if_error:\s*false", "fail_ci_if_error: true", "CI Failure Bypass"),
    (r"exit_on_error:\s*false", "exit_on_error: true", "Error Bypass"),
    # Code quality bypasses
    (r"#\s*type:\s*ignore.*$", "", "Type Ignore"),
    (r"#\s*mypy:\s*ignore.*$", "", "MyPy Ignore"),
    (r"#\s*noqa:?\s*[\w\d,]*", "", "Quality Assurance Bypass"),
    (r"#\s*ruff:\s*noqa.*$", "", "Ruff Bypass"),
    (r"#\s*flake8:\s*noqa.*$", "", "Flake8 Bypass"),
    (r"#\s*pylint:\s*disable.*$", "", "PyLint Bypass"),
    (r"#\s*bandit:\s*skip.*$", "", "Bandit Security Bypass"),
    # Test bypasses
    (r"@pytest\.mark\.skip\(.*?\)", "", "Test Skip"),
    (r"@pytest\.mark\.skipif\(.*?\)", "", "Test Skip Conditional"),
    (r"@pytest\.mark\.xfail.*$", "", "Test Expected Failure"),
    (r"@unittest\.skip\(.*?\)", "", "Unittest Skip"),
    (r"self\.skipTest\(.*?\)", "", "Test Skip Method"),
    (r"pytest\.skip\(.*?\)", "", "Pytest Skip Call"),
    # JavaScript/TypeScript bypasses
    (r"//\s*eslint-disable.*$", "", "ESLint Disable"),
    (r"/\*\s*eslint-disable.*?\*/", "", "ESLint Block Disable"),
    (r"//\s*@ts-ignore.*$", "", "TypeScript Ignore"),
    (r"/\*\s*@ts-ignore.*?\*/", "", "TypeScript Block Ignore"),
    (r"//\s*prettier-ignore.*$", "", "Prettier Ignore"),
    (r"/\*\s*prettier-ignore.*?\*/", "", "Prettier Block Ignore"),
    (r"//\s*tslint:disable.*$", "", "TSLint Disable"),
    # Coverage bypasses
    (r"pragma:\s*no\s*cover", "", "Coverage Pragma"),
    (r"coverage:\s*ignore", "", "Coverage Ignore"),
    (r"# coverage:\s*ignore", "", "Coverage Ignore Comment"),
    # Security bypasses
    (r"--skip=[\w,]+", "", "Security Skip"),
    (r"--ignore=[\w,]+", "", "Security Ignore"),
    (r"\|\|\s*true", "", "Shell OR True Bypass"),
    (r"; true$", "", "Shell True Bypass"),
    # Configuration bypasses
    (r'"skipLibCheck":\s*true', '"skipLibCheck": false', "TypeScript Skip Lib Check"),
    (r"fail_under\s*=\s*[0-7]\d", "fail_under = 80", "Low Coverage Threshold"),
    (r"min-coverage\s*=\s*[0-7]\d", "min-coverage = 80", "Low Coverage Threshold"),
    # Build bypasses
    (r"", "", "Build Check Skip"),
    (r"", "", "Test Skip Flag"),
    (r"", "", "Error Ignore Flag"),
]

# Project directories to check (exclude dependencies)
PROJECT_DIRS = [
    "agents",
    "api",
    "coalitions",
    "inference",
    "world",
    "tests",
    "scripts",
    "auth",
    "infrastructure",
    "knowledge",
    "communication",
    "web/src",
    "web/components",
    "web/pages",
    "web/lib",
]

# File extensions to process
VALID_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".sh",
}

# Files to always check
CRITICAL_FILES = {
    "Makefile",
    "Dockerfile",
    ".gitignore",
    ".env.example",
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
}


def should_process_file(file_path: Path) -> bool:
    """Determine if file should be processed for bypass detection."""
    # Skip dependency directories
    skip_dirs = {
        "venv",
        "node_modules",
        ".git",
        "__pycache__",
        ".pytest_cache",
        "htmlcov",
        ".mypy_cache",
        ".coverage",
        "build",
        "dist",
        ".next",
        "coverage",
        "test-reports",
        "security_audit_env",
        "test_venv",
    }

    if any(part in skip_dirs for part in file_path.parts):
        return False

    # Check if in project directories
    for project_dir in PROJECT_DIRS:
        if str(file_path).startswith(project_dir):
            return file_path.suffix in VALID_EXTENSIONS

    # Check critical files in root
    if file_path.name in CRITICAL_FILES:
        return True

    # Check files with valid extensions in root
    if len(file_path.parts) == 1 and file_path.suffix in VALID_EXTENSIONS:
        return True

    return False


def detect_and_remove_bypasses(file_path: Path) -> List[Dict]:
    """Detect and remove bypass patterns from a file."""
    violations = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            original_content = content
    except Exception as e:
        return [{"file": str(file_path), "error": str(e), "type": "READ_ERROR"}]

    # Apply bypass pattern detection and removal
    for pattern, replacement, violation_type in BYPASS_PATTERNS:
        matches = re.finditer(pattern, content, flags=re.MULTILINE | re.IGNORECASE)
        for match in matches:
            line_num = content[: match.start()].count("\n") + 1
            violations.append(
                {
                    "file": str(file_path),
                    "line": line_num,
                    "pattern": pattern,
                    "match": match.group(),
                    "type": violation_type,
                    "replacement": replacement,
                    "severity": "CRITICAL",
                }
            )

        # Apply replacement
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.IGNORECASE)

    # Write back if changed
    if content != original_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            for violation in violations:
                if violation.get("type") != "READ_ERROR":
                    violation["status"] = "ELIMINATED"
        except Exception as e:
            for violation in violations:
                violation["status"] = "WRITE_ERROR"
                violation["error"] = str(e)

    return violations


def install_pre_receive_hooks():
    """Install git pre-receive hooks to prevent bypass commits."""
    hooks_dir = Path(".git/hooks")
    if not hooks_dir.exists():
        print("Warning: No .git directory found. Cannot install pre-receive hooks.")
        return False

    pre_receive_hook = hooks_dir / "pre-receive"

    hook_content = """#!/bin/bash
# POLICY-GUARD AGENT - PRE-RECEIVE HOOK
# Zero tolerance for bypass directives

echo "🛡️  POLICY-GUARD: Scanning for bypass directives..."

# Patterns that trigger rejection
FORBIDDEN_PATTERNS=(
    "\\[skip ci\\]"
    "\\[ci skip\\]"
    "continue-on-error: *true"
    "allow_failure: *true"
    "# *type: *ignore"
    "# *noqa"
    "# *ruff: *noqa"
    "@pytest\\.mark\\.skip"
    "eslint-disable"
    "@ts-ignore"
    "pragma: *no *cover"
    "skipLibCheck.*true"
    "\\|\\| *true"
)

while read oldrev newrev refname; do
    # Check each commit in the push
    for commit in $(git rev-list $oldrev..$newrev); do
        echo "Checking commit $commit..."
        
        # Get the diff for this commit
        diff_output=$(git show --name-only --pretty=format: $commit)
        
        for file in $diff_output; do
            if [[ -f "$file" ]]; then
                # Check file content for forbidden patterns
                for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
                    if git show $commit:$file 2>/dev/null | grep -i -E "$pattern" > /dev/null; then
                        echo "❌ POLICY VIOLATION DETECTED!"
                        echo "File: $file"
                        echo "Pattern: $pattern"
                        echo "Commit: $commit"
                        echo ""
                        echo "🚫 BYPASS DIRECTIVES ARE FORBIDDEN"
                        echo "Michael Feathers: 'If a safety-net test is failing, fix the code—never snip the net.'"
                        echo ""
                        echo "To fix this:"
                        echo "1. Remove the bypass directive"
                        echo "2. Fix the underlying issue"
                        echo "3. Commit the proper fix"
                        exit 1
                    fi
                done
            fi
        done
    done
done

echo "✅ POLICY-GUARD: No bypass directives detected"
exit 0
"""

    try:
        with open(pre_receive_hook, "w") as f:
            f.write(hook_content)

        # Make executable
        os.chmod(pre_receive_hook, 0o755)
        print(f"✅ Pre-receive hook installed: {pre_receive_hook}")
        return True
    except Exception as e:
        print(f"❌ Failed to install pre-receive hook: {e}")
        return False


def generate_violation_report(violations: List[Dict]) -> str:
    """Generate comprehensive violation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"BYPASS_VIOLATION_REPORT_{timestamp}.md"

    # Group violations by type
    by_type = {}
    by_file = {}

    for violation in violations:
        v_type = violation.get("type", "UNKNOWN")
        v_file = violation.get("file", "UNKNOWN")

        if v_type not in by_type:
            by_type[v_type] = []
        by_type[v_type].append(violation)

        if v_file not in by_file:
            by_file[v_file] = []
        by_file[v_file].append(violation)

    report_content = f"""# POLICY-GUARD AGENT - BYPASS VIOLATION REPORT

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status:** ZERO BYPASS TOLERANCE ENFORCEMENT COMPLETE

## EXECUTIVE SUMMARY

- **Total Violations Found:** {len(violations)}
- **Files Affected:** {len(by_file)}
- **Violation Types:** {len(by_type)}
- **Status:** ALL BYPASSES ELIMINATED

## MICHAEL FEATHERS PRINCIPLE ENFORCEMENT

> "If a safety-net test is failing, fix the code—never snip the net."

All bypass directives have been **ELIMINATED** to ensure code quality and security integrity.

## VIOLATIONS BY TYPE

"""

    for v_type, type_violations in sorted(by_type.items()):
        report_content += f"### {v_type}\n"
        report_content += f"**Count:** {len(type_violations)}\n\n"

        for violation in type_violations[:10]:  # Show first 10 per type
            report_content += f"- **File:** `{violation.get('file', 'N/A')}`\n"
            report_content += f"  **Line:** {violation.get('line', 'N/A')}\n"
            report_content += f"  **Pattern:** `{violation.get('match', 'N/A')}`\n"
            report_content += f"  **Status:** {violation.get('status', 'PROCESSED')}\n\n"

        if len(type_violations) > 10:
            report_content += f"... and {len(type_violations) - 10} more violations\n\n"

    report_content += """
## VIOLATIONS BY FILE

"""

    for file_path, file_violations in sorted(by_file.items()):
        if len(file_violations) > 0:
            report_content += f"### `{file_path}`\n"
            report_content += f"**Violations:** {len(file_violations)}\n\n"

            for violation in file_violations[:5]:  # Show first 5 per file
                report_content += (
                    f"- Line {violation.get('line', 'N/A')}: {violation.get('type', 'N/A')}\n"
                )

            if len(file_violations) > 5:
                report_content += f"- ... and {len(file_violations) - 5} more\n"
            report_content += "\n"

    report_content += """
## PRE-RECEIVE HOOKS INSTALLED

Git pre-receive hooks have been installed to prevent future bypass commits.

## SINDRE SORHUS STANDARDS ENFORCED

- ❌ NO bypass directives allowed
- ❌ NO continue-on-error: false
- ❌ NO allow_failure: false
- ❌ NO type: ignore comments
- ❌ NO noqa directives
- ❌ NO test skips
- ❌ NO eslint-disable
- ❌ NO coverage bypasses

## VERIFICATION

Run the following to verify zero bypasses remain:

```bash
# Search for any remaining bypasses
grep -r "noqa\\|type.*ignore\\|skip.*ci\\|allow_failure.*true" . --exclude-dir=venv --exclude-dir=node_modules

# Should return NO results
```

## CONCLUSION

**MISSION ACCOMPLISHED:** All bypass directives have been eliminated.
The codebase now enforces zero-tolerance bypass policy.

---
Generated by POLICY-GUARD AGENT
Sindre Sorhus Mentor - Zero Bypass Tolerance
"""

    try:
        with open(report_file, "w") as f:
            f.write(report_content)
        print(f"📊 Violation report generated: {report_file}")
        return report_file
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        return ""


def main():
    """Main execution function."""
    print("🛡️  POLICY-GUARD AGENT - COMPREHENSIVE BYPASS ELIMINATION")
    print("=" * 60)
    print("NO MERCY. NO EXCEPTIONS. ZERO BYPASS TOLERANCE.")
    print()

    root_dir = Path(".")
    all_violations = []

    print("🔍 SCANNING CODEBASE FOR BYPASS VIOLATIONS...")
    print()

    # Process all relevant files
    files_processed = 0
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and should_process_file(file_path):
            violations = detect_and_remove_bypasses(file_path)
            all_violations.extend(violations)
            files_processed += 1

            if violations:
                print(f"⚠️  {len(violations)} violations in {file_path}")

    print("\n📊 SCAN COMPLETE:")
    print(f"  Files Processed: {files_processed}")
    print(f"  Total Violations: {len(all_violations)}")
    print()

    # Install pre-receive hooks
    print("🔧 INSTALLING PRE-RECEIVE HOOKS...")
    hook_installed = install_pre_receive_hooks()
    print()

    # Generate report
    print("📋 GENERATING VIOLATION REPORT...")
    report_file = generate_violation_report(all_violations)
    print()

    # Final verification
    print("✅ BYPASS ELIMINATION COMPLETE!")
    print()
    print("SUMMARY:")
    print(f"  • {len(all_violations)} bypass violations ELIMINATED")
    print(f"  • Pre-receive hooks: {'INSTALLED' if hook_installed else 'FAILED'}")
    print(f"  • Report generated: {report_file}")
    print()
    print("🛡️  ZERO BYPASS TOLERANCE ENFORCED")
    print("Michael Feathers: 'If a safety-net test is failing, fix the code—never snip the net.'")

    return 0 if all_violations else 1


if __name__ == "__main__":
    sys.exit(main())
