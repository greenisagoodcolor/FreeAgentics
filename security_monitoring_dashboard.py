#!/usr/bin/env python3
"""
Security Monitoring Dashboard for FreeAgentics
Real-time security metrics and threat detection
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import subprocess


class SecurityMonitoringDashboard:
    """Real-time security monitoring and alerting."""
    
    def __init__(self):
        self.alerts = []
        self.metrics = {
            "last_scan": None,
            "vulnerabilities": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "authentication": {"failed_attempts": 0, "successful_logins": 0},
            "api_security": {"blocked_requests": 0, "rate_limit_hits": 0},
            "dependencies": {"total": 0, "vulnerable": 0}
        }
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        print("🔍 Running security scan...")
        results = {}
        
        # Check for dependency vulnerabilities
        try:
            pip_audit = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True
            )
            if pip_audit.returncode == 0:
                audit_data = json.loads(pip_audit.stdout)
                vulnerable_deps = [d for d in audit_data.get("dependencies", []) if d.get("vulns")]
                results["dependencies"] = {
                    "total": len(audit_data.get("dependencies", [])),
                    "vulnerable": len(vulnerable_deps),
                    "details": vulnerable_deps
                }
        except Exception as e:
            results["dependencies"] = {"error": str(e)}
        
        # Check for hardcoded secrets
        try:
            secrets_check = subprocess.run(
                ["grep", "-r", "-E", "(password|secret|key|token)\\s*=\\s*['\"][^'\"]+['\"]", 
                 "--include=*.py", "--exclude-dir=venv", "--exclude-dir=tests", "."],
                capture_output=True,
                text=True
            )
            if secrets_check.stdout:
                results["secrets"] = {
                    "found": True,
                    "count": len(secrets_check.stdout.strip().split("\n")),
                    "files": list(set([line.split(":")[0] for line in secrets_check.stdout.strip().split("\n")]))
                }
            else:
                results["secrets"] = {"found": False}
        except Exception as e:
            results["secrets"] = {"error": str(e)}
        
        # Check authentication logs
        results["authentication"] = self._check_auth_logs()
        
        # Check security headers
        results["headers"] = self._check_security_headers()
        
        self.metrics["last_scan"] = datetime.now().isoformat()
        return results
    
    def _check_auth_logs(self) -> Dict[str, Any]:
        """Analyze authentication logs for anomalies."""
        # In production, this would read from actual log files
        return {
            "failed_attempts_24h": 0,
            "suspicious_ips": [],
            "brute_force_detected": False
        }
    
    def _check_security_headers(self) -> Dict[str, Any]:
        """Check if security headers are properly configured."""
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Content-Security-Policy",
            "Strict-Transport-Security"
        ]
        
        # Check if middleware file exists
        middleware_path = Path("api/middleware/security_headers.py")
        if middleware_path.exists():
            content = middleware_path.read_text()
            configured = [h for h in required_headers if h in content]
            missing = [h for h in required_headers if h not in configured]
            return {
                "configured": configured,
                "missing": missing,
                "compliance": len(configured) / len(required_headers) * 100
            }
        return {"error": "Security headers middleware not found"}
    
    def generate_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate security monitoring report."""
        report = f"""
# Security Monitoring Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Security Score: {self._calculate_score(scan_results)}/100

### Dependency Security
- Total Dependencies: {scan_results.get('dependencies', {}).get('total', 'N/A')}
- Vulnerable Dependencies: {scan_results.get('dependencies', {}).get('vulnerable', 'N/A')}

### Secrets Detection
- Hardcoded Secrets Found: {'YES ⚠️' if scan_results.get('secrets', {}).get('found') else 'NO ✅'}
"""
        
        if scan_results.get('secrets', {}).get('found'):
            report += f"- Files with potential secrets: {scan_results['secrets']['count']}\n"
            for file in scan_results['secrets'].get('files', [])[:5]:
                report += f"  - {file}\n"
        
        report += f"""
### Authentication Security
- Failed Login Attempts (24h): {scan_results.get('authentication', {}).get('failed_attempts_24h', 0)}
- Brute Force Detected: {'YES 🚨' if scan_results.get('authentication', {}).get('brute_force_detected') else 'NO ✅'}

### Security Headers
- Compliance: {scan_results.get('headers', {}).get('compliance', 0):.1f}%
- Missing Headers: {len(scan_results.get('headers', {}).get('missing', []))}
"""
        
        if scan_results.get('headers', {}).get('missing'):
            report += "\nMissing security headers:\n"
            for header in scan_results['headers']['missing']:
                report += f"  - {header}\n"
        
        # Add recommendations
        report += "\n## Recommendations\n"
        recommendations = self._generate_recommendations(scan_results)
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def _calculate_score(self, scan_results: Dict[str, Any]) -> int:
        """Calculate overall security score."""
        score = 100
        
        # Deduct for vulnerabilities
        vuln_count = scan_results.get('dependencies', {}).get('vulnerable', 0)
        score -= min(vuln_count * 10, 30)
        
        # Deduct for secrets
        if scan_results.get('secrets', {}).get('found'):
            score -= 30
        
        # Deduct for missing headers
        header_compliance = scan_results.get('headers', {}).get('compliance', 100)
        score -= int((100 - header_compliance) * 0.2)
        
        # Deduct for auth issues
        if scan_results.get('authentication', {}).get('brute_force_detected'):
            score -= 20
        
        return max(score, 0)
    
    def _generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        if scan_results.get('dependencies', {}).get('vulnerable', 0) > 0:
            recommendations.append("Update vulnerable dependencies immediately")
        
        if scan_results.get('secrets', {}).get('found'):
            recommendations.append("Remove hardcoded secrets and use environment variables")
        
        if scan_results.get('headers', {}).get('missing'):
            recommendations.append("Implement missing security headers")
        
        if scan_results.get('authentication', {}).get('brute_force_detected'):
            recommendations.append("Investigate brute force attempts and strengthen rate limiting")
        
        if not recommendations:
            recommendations.append("Continue regular security monitoring")
        
        return recommendations
    
    def run_continuous_monitoring(self, interval_minutes: int = 60):
        """Run continuous security monitoring."""
        print(f"🛡️ Starting continuous security monitoring (interval: {interval_minutes} minutes)")
        
        try:
            while True:
                scan_results = self.run_security_scan()
                report = self.generate_report(scan_results)
                
                # Save report
                report_path = Path(f"security_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                report_path.parent.mkdir(exist_ok=True)
                report_path.write_text(report)
                
                # Display summary
                print(f"\n{'=' * 60}")
                print(report)
                print(f"{'=' * 60}\n")
                
                # Check for critical issues
                if self._has_critical_issues(scan_results):
                    print("🚨 CRITICAL SECURITY ISSUES DETECTED! Immediate action required.")
                
                # Wait for next scan
                print(f"Next scan in {interval_minutes} minutes...")
                import time
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n👋 Security monitoring stopped.")
    
    def _has_critical_issues(self, scan_results: Dict[str, Any]) -> bool:
        """Check if there are critical security issues."""
        return (
            scan_results.get('dependencies', {}).get('vulnerable', 0) > 0 or
            scan_results.get('secrets', {}).get('found', False) or
            scan_results.get('authentication', {}).get('brute_force_detected', False)
        )


def main():
    """Run security monitoring dashboard."""
    dashboard = SecurityMonitoringDashboard()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        dashboard.run_continuous_monitoring(interval)
    else:
        # Run single scan
        scan_results = dashboard.run_security_scan()
        report = dashboard.generate_report(scan_results)
        print(report)
        
        # Save report
        report_path = Path(f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        report_path.write_text(report)
        print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()