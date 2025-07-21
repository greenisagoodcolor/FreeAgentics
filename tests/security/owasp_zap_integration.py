"""
OWASP ZAP Integration for Automated Security Testing

This module integrates with OWASP ZAP (Zed Attack Proxy) to provide
automated security scanning capabilities including:
- Active scanning
- Passive scanning
- Spider/crawler
- API scanning
- Authenticated scanning
- Custom scan policies
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from zapv2 import ZAPv2


class OWASPZAPIntegration:
    """OWASP ZAP integration for automated security testing"""

    def __init__(
        self,
        zap_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        target_url: str = "http://localhost:8000",
    ):
        """
        Initialize ZAP integration

        Args:
            zap_url: ZAP proxy URL
            api_key: ZAP API key (required for API operations)
            target_url: Target application URL
        """
        self.zap_url = zap_url
        self.api_key = api_key or os.getenv("ZAP_API_KEY", "changeme")
        self.target_url = target_url

        # Initialize ZAP client
        self.zap = ZAPv2(
            apikey=self.api_key, proxies={"http": zap_url, "https": zap_url}
        )

        self.scan_results = {
            "passive_scan": [],
            "active_scan": [],
            "spider_results": [],
            "ajax_spider_results": [],
            "api_scan": [],
            "authenticated_scan": [],
        }

    async def run_full_scan(self, include_active: bool = True) -> Dict:
        """
        Run comprehensive security scan

        Args:
            include_active: Whether to include active scanning (can be destructive)

        Returns:
            Comprehensive scan results
        """
        print(f"Starting OWASP ZAP scan on {self.target_url}")

        # Start new session
        self._start_new_session()

        # Configure scan policies
        self._configure_scan_policies()

        # Run spider
        await self._run_spider()

        # Run AJAX spider for modern apps
        await self._run_ajax_spider()

        # Passive scan (happens automatically)
        await self._wait_for_passive_scan()

        # Active scan (if enabled)
        if include_active:
            await self._run_active_scan()

        # API-specific scanning
        await self._run_api_scan()

        # Authenticated scanning
        await self._run_authenticated_scan()

        # Generate report
        report = self._generate_comprehensive_report()

        return report

    def _start_new_session(self):
        """Start a new ZAP session"""
        try:
            # Create new session
            self.zap.core.new_session()

            # Set target
            self.zap.core.access_url(self.target_url)

            # Configure context
            context_name = "FreeAgentics"
            context_id = self.zap.context.new_context(context_name)

            # Include target in context
            self.zap.context.include_in_context(context_name, f"{self.target_url}.*")

            self.context_id = context_id
            self.context_name = context_name

        except Exception as e:
            print(f"Error starting ZAP session: {e}")

    def _configure_scan_policies(self):
        """Configure ZAP scan policies"""
        try:
            # Create custom scan policy
            policy_name = "FreeAgentics Security Policy"

            # Enable all scan rules
            self.zap.ascan.enable_all_scanners(scanpolicyname=policy_name)

            # Configure specific rules with higher strength
            critical_rules = [
                "40018",  # SQL Injection
                "40012",  # Cross Site Scripting (Reflected)
                "40014",  # Cross Site Scripting (Persistent)
                "40016",  # Web Browser XSS Protection Not Enabled
                "40017",  # Cross-Domain JavaScript Source File Inclusion
                "90019",  # Server Side Code Injection
                "90020",  # Remote OS Command Injection
                "90021",  # Path Traversal
                "90022",  # Application Error Disclosure
                "90023",  # XML External Entity Attack
                "90024",  # Generic Padding Oracle
                "90025",  # Expression Language Injection
                "90026",  # SOAP Action Spoofing
                "90027",  # Cookie Slack Detector
                "90028",  # Insecure HTTP Method
                "90029",  # SOAP XML Injection
                "90030",  # WSDL File Detection
                "90033",  # Loosely Scoped Cookie
                "90034",  # Cloud Metadata Potentially Exposed
                "40003",  # CRLF Injection
                "40008",  # Parameter Tampering
                "40009",  # Server Side Include
                "40013",  # Session Fixation
                "40019",  # External Redirect
                "40020",  # Anti-CSRF Tokens Check
                "40021",  # Persistent XSS - Prime
                "40022",  # SQL Injection - Oracle
                "40023",  # SQL Injection - PostgreSQL
                "40024",  # SQL Injection - SQLite
                "40025",  # Proxy Disclosure
                "40026",  # Cross Domain Misconfiguration
                "40027",  # SQL Injection - MySQL
                "40028",  # ELMAH Information Leak
                "40029",  # Trace.axd Information Leak
                "40031",  # Out of Band XSS
                "40032",  # .htaccess Information Leak
                "40034",  # .env Information Leak
            ]

            for rule_id in critical_rules:
                try:
                    self.zap.ascan.set_scanner_attack_strength(
                        rule_id, "HIGH", scanpolicyname=policy_name
                    )
                except Exception:
                    pass

            self.scan_policy = policy_name

        except Exception as e:
            print(f"Error configuring scan policies: {e}")

    async def _run_spider(self):
        """Run traditional spider/crawler"""
        print("Running spider...")

        try:
            # Start spider
            scan_id = self.zap.spider.scan(self.target_url)

            # Wait for spider to complete
            while int(self.zap.spider.status(scan_id)) < 100:
                await asyncio.sleep(2)
                progress = self.zap.spider.status(scan_id)
                print(f"Spider progress: {progress}%")

            # Get results
            results = self.zap.spider.results(scan_id)
            self.scan_results["spider_results"] = results

            print(f"Spider found {len(results)} URLs")

        except Exception as e:
            print(f"Error during spider scan: {e}")

    async def _run_ajax_spider(self):
        """Run AJAX spider for modern JavaScript applications"""
        print("Running AJAX spider...")

        try:
            # Start AJAX spider
            self.zap.ajaxSpider.scan(self.target_url)

            # Wait for AJAX spider
            while self.zap.ajaxSpider.status != "stopped":
                await asyncio.sleep(2)
                status = self.zap.ajaxSpider.status
                results = self.zap.ajaxSpider.number_of_results
                print(f"AJAX Spider status: {status}, Results: {results}")

            # Get results
            results = self.zap.ajaxSpider.results()
            self.scan_results["ajax_spider_results"] = results

            print(f"AJAX spider found {len(results)} URLs")

        except Exception as e:
            print(f"Error during AJAX spider scan: {e}")

    async def _wait_for_passive_scan(self):
        """Wait for passive scanning to complete"""
        print("Waiting for passive scan...")

        try:
            while int(self.zap.pscan.records_to_scan) > 0:
                await asyncio.sleep(2)
                records = self.zap.pscan.records_to_scan
                print(f"Passive scan records remaining: {records}")

            # Get passive scan results
            alerts = self.zap.core.alerts(baseurl=self.target_url)
            self.scan_results["passive_scan"] = alerts

            print(f"Passive scan found {len(alerts)} alerts")

        except Exception as e:
            print(f"Error during passive scan: {e}")

    async def _run_active_scan(self):
        """Run active security scan"""
        print("Running active scan...")

        try:
            # Start active scan
            scan_id = self.zap.ascan.scan(
                self.target_url, scanpolicyname=self.scan_policy
            )

            # Wait for active scan
            while int(self.zap.ascan.status(scan_id)) < 100:
                await asyncio.sleep(5)
                progress = self.zap.ascan.status(scan_id)
                print(f"Active scan progress: {progress}%")

            # Get results
            alerts = self.zap.core.alerts(baseurl=self.target_url)
            self.scan_results["active_scan"] = alerts

            print(f"Active scan found {len(alerts)} alerts")

        except Exception as e:
            print(f"Error during active scan: {e}")

    async def _run_api_scan(self):
        """Run API-specific security scan"""
        print("Running API scan...")

        try:
            # Import OpenAPI spec if available
            openapi_url = f"{self.target_url}/openapi.json"

            try:
                self.zap.openapi.import_url(openapi_url, self.target_url)
                print("Imported OpenAPI specification")
            except Exception:
                print("No OpenAPI spec found, using standard API scanning")

            # Configure API scan
            api_endpoints = [
                "/api/v1/auth/login",
                "/api/v1/auth/register",
                "/api/v1/users",
                "/api/v1/resources",
                "/api/v1/agents",
                "/api/v1/admin",
            ]

            # Scan each endpoint
            for endpoint in api_endpoints:
                full_url = f"{self.target_url}{endpoint}"

                # Access URL to add to sites
                self.zap.core.access_url(full_url)

                # Run targeted scan
                scan_id = self.zap.ascan.scan(full_url)

                # Wait for completion
                while int(self.zap.ascan.status(scan_id)) < 100:
                    await asyncio.sleep(2)

            # Get API-specific alerts
            api_alerts = []
            for endpoint in api_endpoints:
                alerts = self.zap.core.alerts(baseurl=f"{self.target_url}{endpoint}")
                api_alerts.extend(alerts)

            self.scan_results["api_scan"] = api_alerts

            print(f"API scan found {len(api_alerts)} alerts")

        except Exception as e:
            print(f"Error during API scan: {e}")

    async def _run_authenticated_scan(self):
        """Run authenticated security scan"""
        print("Running authenticated scan...")

        try:
            # Configure authentication
            # This would typically use real credentials
            auth_params = {
                "loginUrl": f"{self.target_url}/api/v1/auth/login",
                "username": "test_user",
                "password": "test_password",
            }

            # Set up form-based authentication
            self.zap.authentication.set_authentication_method(
                self.context_id,
                "formBasedAuthentication",
                authMethodConfigParams=json.dumps(auth_params),
            )

            # Create user
            user_id = self.zap.users.new_user(self.context_id, "test_user")

            # Set credentials
            self.zap.users.set_authentication_credentials(
                self.context_id,
                user_id,
                f"username={auth_params['username']}&password={auth_params['password']}",
            )

            # Enable user
            self.zap.users.set_user_enabled(self.context_id, user_id, True)

            # Force user mode
            self.zap.forcedUser.set_forced_user(self.context_id, user_id)
            self.zap.forcedUser.set_forced_user_mode_enabled(True)

            # Run authenticated scan
            scan_id = self.zap.ascan.scan_as_user(
                self.target_url,
                self.context_id,
                user_id,
                scanpolicyname=self.scan_policy,
            )

            # Wait for completion
            while int(self.zap.ascan.status(scan_id)) < 100:
                await asyncio.sleep(5)
                progress = self.zap.ascan.status(scan_id)
                print(f"Authenticated scan progress: {progress}%")

            # Get results
            auth_alerts = self.zap.core.alerts(baseurl=self.target_url)
            self.scan_results["authenticated_scan"] = auth_alerts

            print(f"Authenticated scan found {len(auth_alerts)} alerts")

        except Exception as e:
            print(f"Error during authenticated scan: {e}")

    def _generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive security report"""

        # Aggregate all alerts
        all_alerts = []
        for scan_type, alerts in self.scan_results.items():
            if scan_type.endswith("_results"):
                continue  # Skip spider results
            all_alerts.extend(alerts)

        # Deduplicate alerts
        unique_alerts = {}
        for alert in all_alerts:
            key = f"{alert.get('alert')}_{alert.get('url')}"
            if key not in unique_alerts or alert.get("risk") > unique_alerts[key].get(
                "risk"
            ):
                unique_alerts[key] = alert

        # Categorize by risk
        risk_summary = {
            "High": [],
            "Medium": [],
            "Low": [],
            "Informational": [],
        }

        for alert in unique_alerts.values():
            risk = alert.get("risk", "Informational")
            risk_summary[risk].append(alert)

        # Generate OWASP mapping
        owasp_mapping = self._map_to_owasp(unique_alerts.values())

        # Create report
        report = {
            "scan_info": {
                "target_url": self.target_url,
                "scan_date": datetime.now().isoformat(),
                "zap_version": self.zap.core.version,
                "total_alerts": len(unique_alerts),
                "urls_discovered": len(self.scan_results.get("spider_results", []))
                + len(self.scan_results.get("ajax_spider_results", [])),
            },
            "risk_summary": {
                "high": len(risk_summary["High"]),
                "medium": len(risk_summary["Medium"]),
                "low": len(risk_summary["Low"]),
                "informational": len(risk_summary["Informational"]),
            },
            "detailed_findings": risk_summary,
            "owasp_mapping": owasp_mapping,
            "scan_coverage": {
                "spider_urls": len(self.scan_results.get("spider_results", [])),
                "ajax_urls": len(self.scan_results.get("ajax_spider_results", [])),
                "passive_alerts": len(self.scan_results.get("passive_scan", [])),
                "active_alerts": len(self.scan_results.get("active_scan", [])),
                "api_alerts": len(self.scan_results.get("api_scan", [])),
                "auth_alerts": len(self.scan_results.get("authenticated_scan", [])),
            },
            "recommendations": self._generate_recommendations(risk_summary),
        }

        # Export reports in multiple formats
        self._export_reports(report)

        return report

    def _map_to_owasp(self, alerts: List[Dict]) -> Dict:
        """Map alerts to OWASP Top 10"""

        owasp_mapping = {
            "A01:2021 – Broken Access Control": [],
            "A02:2021 – Cryptographic Failures": [],
            "A03:2021 – Injection": [],
            "A04:2021 – Insecure Design": [],
            "A05:2021 – Security Misconfiguration": [],
            "A06:2021 – Vulnerable and Outdated Components": [],
            "A07:2021 – Identification and Authentication Failures": [],
            "A08:2021 – Software and Data Integrity Failures": [],
            "A09:2021 – Security Logging and Monitoring Failures": [],
            "A10:2021 – Server-Side Request Forgery": [],
        }

        # Map ZAP alerts to OWASP categories
        alert_to_owasp = {
            "SQL Injection": "A03:2021 – Injection",
            "Cross Site Scripting": "A03:2021 – Injection",
            "Path Traversal": "A01:2021 – Broken Access Control",
            "Remote File Include": "A03:2021 – Injection",
            "Server Side Include": "A03:2021 – Injection",
            "Authentication": "A07:2021 – Identification and Authentication Failures",
            "Session": "A07:2021 – Identification and Authentication Failures",
            "CSRF": "A01:2021 – Broken Access Control",
            "Cryptographic": "A02:2021 – Cryptographic Failures",
            "Configuration": "A05:2021 – Security Misconfiguration",
            "Component": "A06:2021 – Vulnerable and Outdated Components",
            "SSRF": "A10:2021 – Server-Side Request Forgery",
        }

        for alert in alerts:
            alert_name = alert.get("alert", "")
            mapped = False

            for keyword, owasp_cat in alert_to_owasp.items():
                if keyword.lower() in alert_name.lower():
                    owasp_mapping[owasp_cat].append(alert)
                    mapped = True
                    break

            if not mapped:
                # Default mapping
                owasp_mapping["A05:2021 – Security Misconfiguration"].append(alert)

        return owasp_mapping

    def _generate_recommendations(self, risk_summary: Dict) -> List[Dict]:
        """Generate security recommendations"""

        recommendations = []

        if risk_summary["High"]:
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "title": "Address High Risk Vulnerabilities",
                    "description": f"Found {len(risk_summary['High'])} high-risk vulnerabilities that require immediate attention",
                    "actions": [
                        "Review and fix all SQL injection vulnerabilities",
                        "Implement proper input validation and sanitization",
                        "Update security headers and access controls",
                    ],
                }
            )

        if risk_summary["Medium"]:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "title": "Fix Medium Risk Issues",
                    "description": f"Found {len(risk_summary['Medium'])} medium-risk issues",
                    "actions": [
                        "Implement CSRF protection on all state-changing operations",
                        "Review and update session management",
                        "Enhance error handling to prevent information disclosure",
                    ],
                }
            )

        recommendations.extend(
            [
                {
                    "priority": "MEDIUM",
                    "title": "Implement Security Headers",
                    "description": "Add comprehensive security headers",
                    "actions": [
                        "Implement Content-Security-Policy",
                        "Add X-Frame-Options",
                        "Enable HSTS",
                        "Set X-Content-Type-Options",
                    ],
                },
                {
                    "priority": "MEDIUM",
                    "title": "Regular Security Testing",
                    "description": "Establish regular security testing practices",
                    "actions": [
                        "Integrate ZAP into CI/CD pipeline",
                        "Run weekly automated scans",
                        "Perform quarterly manual penetration testing",
                    ],
                },
            ]
        )

        return recommendations

    def _export_reports(self, report: Dict):
        """Export reports in multiple formats"""

        # JSON report
        with open("zap_security_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # HTML report
        try:
            html_report = self.zap.core.htmlreport()
            with open("zap_security_report.html", "w") as f:
                f.write(html_report)
        except Exception:
            pass

        # XML report
        try:
            xml_report = self.zap.core.xmlreport()
            with open("zap_security_report.xml", "w") as f:
                f.write(xml_report)
        except Exception:
            pass

        print("Reports exported:")
        print("- zap_security_report.json")
        print("- zap_security_report.html")
        print("- zap_security_report.xml")

    def configure_ci_cd_integration(self) -> Dict:
        """Configure ZAP for CI/CD integration"""

        ci_config = {
            "docker_command": "docker run -t owasp/zap2docker-stable zap-baseline.py -t "
            + self.target_url,
            "github_action": {
                "name": "OWASP ZAP Scan",
                "uses": "zaproxy/action-full-scan@v0.4.0",
                "with": {
                    "target": self.target_url,
                    "rules_file_name": ".zap/rules.tsv",
                    "cmd_options": "-a",
                },
            },
            "jenkins_pipeline": """
                stage('Security Scan') {
                    steps {
                        script {
                            def zapHome = tool 'ZAP'
                            sh "${zapHome}/zap.sh -cmd -quickurl ${TARGET_URL} -quickprogress -quickout zap_report.html"
                        }
                    }
                }
            """,
            "fail_thresholds": {"high": 0, "medium": 5, "low": 10},
        }

        return ci_config


# Example usage and testing
async def test_zap_integration():
    """Test ZAP integration"""

    # Initialize ZAP integration
    zap_scanner = OWASPZAPIntegration(target_url="http://localhost:8000")

    # Run full scan
    report = await zap_scanner.run_full_scan(include_active=True)

    # Check results
    print("\n" + "=" * 80)
    print("ZAP SECURITY SCAN RESULTS")
    print("=" * 80)
    print(f"Total Alerts: {report['scan_info']['total_alerts']}")
    print(f"High Risk: {report['risk_summary']['high']}")
    print(f"Medium Risk: {report['risk_summary']['medium']}")
    print(f"Low Risk: {report['risk_summary']['low']}")
    print(f"URLs Discovered: {report['scan_info']['urls_discovered']}")

    # Display high-risk findings
    if report["detailed_findings"]["High"]:
        print("\nHIGH RISK FINDINGS:")
        for alert in report["detailed_findings"]["High"][:5]:  # Show first 5
            print(f"- {alert.get('alert')}")
            print(f"  URL: {alert.get('url')}")
            print(f"  Description: {alert.get('description', '')[:100]}...")

    return report


if __name__ == "__main__":
    asyncio.run(test_zap_integration())
