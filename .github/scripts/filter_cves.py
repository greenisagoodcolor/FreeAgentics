#!/usr/bin/env python3
"""
CVE Filtering Script for CI/CD Pipeline
======================================

Committee-approved allowlist for disputed/false positive CVEs.
Maintains zero-tolerance for legitimate vulnerabilities while handling edge cases.

Usage: python filter_cves.py safety_report.json
"""

import json
import sys
from typing import Dict, List, Any

# Committee Decision: Disputed CVEs with technical justification
DISPUTED_CVE_ALLOWLIST = {
    "51457": {
        "cve": "CVE-2022-42969",
        "package": "py",
        "reason": "ReDoS via SVN - DISPUTED by maintainers, not applicable to production use",
        "committee_approval": "2025-07-22",
        "justification": "Package py is deprecated test infrastructure, vulnerability requires SVN usage"
    }
}

def load_safety_report(filename: str) -> Dict[str, Any]:
    """Load and parse Safety JSON report."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Safety report file not found: {filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in safety report: {e}")
        sys.exit(1)

def filter_vulnerabilities(vulnerabilities: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter vulnerabilities based on allowlist.
    
    Returns:
        tuple: (filtered_vulnerabilities, allowlisted_vulnerabilities)
    """
    filtered = []
    allowlisted = []
    
    for vuln in vulnerabilities:
        vuln_id = vuln.get("vulnerability_id", "")
        
        if vuln_id in DISPUTED_CVE_ALLOWLIST:
            allowlist_entry = DISPUTED_CVE_ALLOWLIST[vuln_id]
            print(f"⚠️  Allowing disputed CVE: {vuln_id} ({allowlist_entry['cve']})")
            print(f"    Package: {allowlist_entry['package']}")
            print(f"    Reason: {allowlist_entry['reason']}")
            allowlisted.append(vuln)
        else:
            filtered.append(vuln)
    
    return filtered, allowlisted

def count_by_severity(vulnerabilities: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count vulnerabilities by severity level."""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
    
    for vuln in vulnerabilities:
        severity = (vuln.get("severity", "") or "").lower()
        if severity in counts:
            counts[severity] += 1
        else:
            counts["unknown"] += 1
    
    return counts

def main():
    """Main CVE filtering logic."""
    if len(sys.argv) != 2:
        print("Usage: python filter_cves.py safety_report.json")
        sys.exit(1)
    
    report_file = sys.argv[1]
    
    # Load safety report
    safety_report = load_safety_report(report_file)
    
    # Extract vulnerabilities
    all_vulnerabilities = safety_report.get("vulnerabilities", [])
    total_vulns = len(all_vulnerabilities)
    
    print(f"🔍 Processing {total_vulns} vulnerabilities from safety scan...")
    
    # Filter vulnerabilities
    filtered_vulns, allowlisted_vulns = filter_vulnerabilities(all_vulnerabilities)
    
    # Count by severity (after filtering)
    severity_counts = count_by_severity(filtered_vulns)
    
    # Report results
    print(f"\n📊 Vulnerability Summary (after allowlist filtering):")
    print(f"   Total found: {total_vulns}")
    print(f"   Allowlisted: {len(allowlisted_vulns)}")
    print(f"   Remaining: {len(filtered_vulns)}")
    print(f"   Critical: {severity_counts['critical']}")
    print(f"   High: {severity_counts['high']}")
    print(f"   Medium: {severity_counts['medium']}")
    print(f"   Low: {severity_counts['low']}")
    
    # Check zero-tolerance thresholds
    critical_count = severity_counts['critical']
    high_count = severity_counts['high']
    
    # Committee policy: MAX_CRITICAL_VULNS=0, MAX_HIGH_VULNS=0
    MAX_CRITICAL_VULNS = 0
    MAX_HIGH_VULNS = 0
    
    if critical_count > MAX_CRITICAL_VULNS or high_count > MAX_HIGH_VULNS:
        print(f"\n❌ SECURITY POLICY VIOLATION:")
        print(f"   Critical vulnerabilities: {critical_count} (max allowed: {MAX_CRITICAL_VULNS})")
        print(f"   High vulnerabilities: {high_count} (max allowed: {MAX_HIGH_VULNS})")
        
        # Show the violating vulnerabilities
        if filtered_vulns:
            print(f"\n🔴 Blocking vulnerabilities:")
            for vuln in filtered_vulns:
                severity = vuln.get("severity", "unknown")
                if severity in ["critical", "high"]:
                    cve = vuln.get("CVE", "No CVE")
                    package = vuln.get("package_name", "unknown")
                    advisory = vuln.get("advisory", "")[:100]
                    print(f"   - {cve} in {package} [{severity.upper()}]")
                    if advisory:
                        print(f"     {advisory}...")
        
        sys.exit(1)
    
    print(f"\n✅ Security scan passed - no critical/high vulnerabilities found")
    if allowlisted_vulns:
        print(f"   (Disputed CVEs properly allowlisted)")

if __name__ == "__main__":
    main()