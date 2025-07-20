"""
Dynamic Application Security Testing (DAST) Integration

Integrates OWASP ZAP for dynamic security testing of running applications.
Performs API endpoint scanning, authentication bypass testing, and more.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from zapv2 import ZAPv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """ZAP risk levels"""

    INFORMATIONAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @property
    def name(self) -> str:
        names = {
            self.INFORMATIONAL: "Informational",
            self.LOW: "Low",
            self.MEDIUM: "Medium",
            self.HIGH: "High",
        }
        return names[self]


@dataclass
class Alert:
    """Security alert from DAST scan"""

    name: str
    risk: RiskLevel
    confidence: str
    description: str
    solution: str
    reference: str
    cwe_id: Optional[int] = None
    wasc_id: Optional[int] = None
    instances: List[Dict[str, Any]] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class DASTConfig:
    """Configuration for DAST scanning"""

    target_url: str
    zap_proxy: str = "http://127.0.0.1:8080"
    api_key: Optional[str] = None
    ajax_spider: bool = True
    active_scan: bool = True
    passive_scan_wait: int = 30
    scan_policy: Optional[str] = None
    context_name: str = "FreeAgentics"
    auth_config: Optional[Dict[str, Any]] = None
    max_scan_duration: int = 3600  # 1 hour
    risk_threshold: RiskLevel = RiskLevel.MEDIUM
    exclude_urls: List[str] = field(default_factory=list)
    include_urls: List[str] = field(default_factory=list)
    custom_headers: Dict[str, str] = field(default_factory=dict)


class AuthenticationTester:
    """Test authentication bypass vulnerabilities"""

    def __init__(self, zap: ZAPv2, config: DASTConfig):
        self.zap = zap
        self.config = config

    async def test_authentication_bypass(self) -> List[Dict[str, Any]]:
        """Test for authentication bypass vulnerabilities"""
        logger.info("Testing for authentication bypass vulnerabilities...")
        vulnerabilities = []

        # Test endpoints that should require authentication
        protected_endpoints = [
            "/api/v1/admin",
            "/api/v1/agents/create",
            "/api/v1/agents/delete",
            "/api/v1/knowledge/update",
            "/api/v1/monitoring/logs",
            "/api/v1/security/audit",
        ]

        # Common authentication bypass techniques
        bypass_techniques = [
            # No authentication
            {"headers": {}},
            # Invalid tokens
            {"headers": {"Authorization": "Bearer invalid_token"}},
            {
                "headers": {
                    "Authorization": "Bearer eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiJ9."
                }
            },
            # SQL injection in auth headers
            {"headers": {"Authorization": "Bearer ' OR '1'='1"}},
            # Directory traversal in auth
            {"headers": {"Authorization": "Bearer ../../admin"}},
            # Null byte injection
            {"headers": {"Authorization": "Bearer admin%00"}},
            # Case manipulation
            {"headers": {"authorization": "bearer valid_token"}},
            # Additional headers that might bypass
            {"headers": {"X-Forwarded-For": "127.0.0.1"}},
            {"headers": {"X-Real-IP": "127.0.0.1"}},
            {"headers": {"X-Original-URL": "/admin"}},
            {"headers": {"X-Rewrite-URL": "/admin"}},
        ]

        for endpoint in protected_endpoints:
            url = urljoin(self.config.target_url, endpoint)

            for technique in bypass_techniques:
                try:
                    # Test with bypass technique
                    response = await self._test_endpoint(
                        url, technique["headers"]
                    )

                    # Check if bypass was successful
                    if response["status"] in [
                        200,
                        201,
                    ] and not self._is_auth_error(response["body"]):
                        vulnerabilities.append(
                            {
                                "type": "authentication_bypass",
                                "endpoint": endpoint,
                                "technique": technique,
                                "response_status": response["status"],
                                "evidence": response["body"][
                                    :500
                                ],  # First 500 chars
                            }
                        )
                        logger.warning(
                            f"Authentication bypass found at {endpoint} using {technique}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error testing {endpoint} with {technique}: {e}"
                    )

        return vulnerabilities

    async def test_jwt_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Test for JWT-specific vulnerabilities"""
        logger.info("Testing for JWT vulnerabilities...")
        vulnerabilities = []

        # Test login endpoint to get valid JWT
        login_url = urljoin(self.config.target_url, "/api/v1/auth/login")
        valid_token = await self._get_valid_token(login_url)

        if not valid_token:
            logger.warning("Could not obtain valid JWT token for testing")
            return vulnerabilities

        # JWT vulnerability tests
        jwt_tests = [
            {
                "name": "None algorithm",
                "token": self._create_none_algorithm_token(valid_token),
                "description": "JWT with 'none' algorithm should be rejected",
            },
            {
                "name": "Weak secret",
                "token": self._create_weak_secret_token(),
                "description": "JWT signed with common weak secret",
            },
            {
                "name": "Expired token",
                "token": self._create_expired_token(valid_token),
                "description": "Expired JWT should be rejected",
            },
            {
                "name": "Algorithm confusion",
                "token": self._create_algorithm_confusion_token(valid_token),
                "description": "JWT with algorithm confusion attack",
            },
        ]

        # Test protected endpoint with vulnerable JWTs
        test_endpoint = urljoin(self.config.target_url, "/api/v1/agents")

        for test in jwt_tests:
            try:
                response = await self._test_endpoint(
                    test_endpoint, {"Authorization": f"Bearer {test['token']}"}
                )

                if response["status"] in [200, 201]:
                    vulnerabilities.append(
                        {
                            "type": "jwt_vulnerability",
                            "name": test["name"],
                            "description": test["description"],
                            "endpoint": test_endpoint,
                            "response_status": response["status"],
                        }
                    )
                    logger.warning(f"JWT vulnerability found: {test['name']}")

            except Exception as e:
                logger.error(
                    f"Error testing JWT vulnerability {test['name']}: {e}"
                )

        return vulnerabilities

    async def _test_endpoint(
        self, url: str, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Test an endpoint with given headers"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, ssl=False
            ) as response:
                body = await response.text()
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": body,
                }

    async def _get_valid_token(self, login_url: str) -> Optional[str]:
        """Get a valid JWT token by logging in"""
        # This would need actual credentials or test credentials
        # For now, return None
        return None

    def _is_auth_error(self, response_body: str) -> bool:
        """Check if response indicates authentication error"""
        auth_error_indicators = [
            "unauthorized",
            "forbidden",
            "authentication required",
            "401",
            "403",
            "access denied",
        ]
        return any(
            indicator in response_body.lower()
            for indicator in auth_error_indicators
        )

    def _create_none_algorithm_token(self, valid_token: str) -> str:
        """Create JWT with 'none' algorithm"""
        # In real implementation, decode the token and re-encode with none algorithm
        # For security reasons, not implementing actual token manipulation
        return "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiJ9."

    def _create_weak_secret_token(self) -> str:
        """Create JWT with weak secret"""
        # Would create token signed with common weak secrets like 'secret', '123456', etc.
        return "weak_secret_token_placeholder"

    def _create_expired_token(self, valid_token: str) -> str:
        """Create expired JWT"""
        # Would modify exp claim to be in the past
        return "expired_token_placeholder"

    def _create_algorithm_confusion_token(self, valid_token: str) -> str:
        """Create token for algorithm confusion attack"""
        # Would change RS256 to HS256 and sign with public key
        return "algorithm_confusion_placeholder"


class APIScanner:
    """Scan API endpoints for vulnerabilities"""

    def __init__(self, zap: ZAPv2, config: DASTConfig):
        self.zap = zap
        self.config = config

    def import_openapi_spec(self, spec_url: str) -> bool:
        """Import OpenAPI specification for comprehensive API scanning"""
        try:
            logger.info(f"Importing OpenAPI spec from {spec_url}")
            self.zap.openapi.import_url(spec_url, self.config.target_url)
            return True
        except Exception as e:
            logger.error(f"Failed to import OpenAPI spec: {e}")
            return False

    def scan_api_endpoints(self) -> List[str]:
        """Scan all discovered API endpoints"""
        logger.info("Scanning API endpoints...")
        endpoints = []

        # Get all URLs in ZAP's sites tree
        sites = self.zap.core.sites
        for site in sites:
            urls = self.zap.core.urls(site)
            endpoints.extend(urls)

        # Filter for API endpoints
        api_endpoints = [url for url in endpoints if "/api/" in url]

        logger.info(f"Found {len(api_endpoints)} API endpoints")
        return api_endpoints

    def test_api_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Test for common API vulnerabilities"""
        vulnerabilities = []

        # Test for missing rate limiting
        rate_limit_vulns = self._test_rate_limiting()
        vulnerabilities.extend(rate_limit_vulns)

        # Test for CORS misconfigurations
        cors_vulns = self._test_cors()
        vulnerabilities.extend(cors_vulns)

        # Test for API versioning issues
        version_vulns = self._test_api_versioning()
        vulnerabilities.extend(version_vulns)

        # Test for excessive data exposure
        data_exposure_vulns = self._test_excessive_data_exposure()
        vulnerabilities.extend(data_exposure_vulns)

        return vulnerabilities

    def _test_rate_limiting(self) -> List[Dict[str, Any]]:
        """Test for missing rate limiting"""
        vulnerabilities = []
        test_endpoints = [
            "/api/v1/auth/login",
            "/api/v1/agents",
            "/api/v1/knowledge/query",
        ]

        for endpoint in test_endpoints:
            url = urljoin(self.config.target_url, endpoint)

            # Send rapid requests
            start_time = time.time()
            successful_requests = 0

            for i in range(100):  # Send 100 rapid requests
                try:
                    response = requests.get(
                        url, headers=self.config.custom_headers, timeout=1
                    )
                    if response.status_code != 429:  # Not rate limited
                        successful_requests += 1
                except Exception as e:
                    # Log but continue - some request failures expected during rate limit test
                    logger.debug(f"Request failed during rate limit test: {e}")
                    continue

            elapsed_time = time.time() - start_time

            # If most requests succeeded quickly, likely no rate limiting
            if successful_requests > 90 and elapsed_time < 10:
                vulnerabilities.append(
                    {
                        "type": "missing_rate_limiting",
                        "endpoint": endpoint,
                        "requests_sent": 100,
                        "successful_requests": successful_requests,
                        "time_elapsed": elapsed_time,
                    }
                )

        return vulnerabilities

    def _test_cors(self) -> List[Dict[str, Any]]:
        """Test for CORS misconfigurations"""
        vulnerabilities = []
        malicious_origins = [
            "http://evil.com",
            "null",
            "file://",
            "http://localhost.evil.com",
        ]

        for origin in malicious_origins:
            headers = {"Origin": origin}
            url = urljoin(self.config.target_url, "/api/v1/health")

            try:
                response = requests.options(url, headers=headers, timeout=30)
                acao = response.headers.get("Access-Control-Allow-Origin", "")
                acac = response.headers.get(
                    "Access-Control-Allow-Credentials", ""
                )

                # Check for overly permissive CORS
                if acao == "*" or acao == origin:
                    if acac == "true":
                        vulnerabilities.append(
                            {
                                "type": "cors_misconfiguration",
                                "severity": "high",
                                "origin_tested": origin,
                                "acao_header": acao,
                                "credentials_allowed": True,
                            }
                        )
            except Exception as e:
                # Log CORS test failures but continue testing other endpoints
                logger.debug(f"CORS test failed for {url}: {e}")
                continue

        return vulnerabilities

    def _test_api_versioning(self) -> List[Dict[str, Any]]:
        """Test for API versioning vulnerabilities"""
        vulnerabilities = []
        version_patterns = [
            "/api/v0/",
            "/api/v1/",
            "/api/v2/",
            "/api/beta/",
            "/api/internal/",
        ]

        for pattern in version_patterns:
            # Test if old/internal API versions are accessible
            test_url = self.config.target_url.replace("/api/v1/", pattern)
            try:
                response = requests.get(
                    urljoin(test_url, "health"), timeout=30
                )
                if response.status_code == 200:
                    vulnerabilities.append(
                        {
                            "type": "exposed_api_version",
                            "version": pattern,
                            "url": test_url,
                            "status_code": response.status_code,
                        }
                    )
            except Exception as e:
                # Log API versioning test failures but continue testing other endpoints
                logger.debug(f"API versioning test failed for {test_url}: {e}")
                continue

        return vulnerabilities

    def _test_excessive_data_exposure(self) -> List[Dict[str, Any]]:
        """Test for excessive data exposure in API responses"""
        vulnerabilities = []
        sensitive_fields = [
            "password",
            "secret",
            "token",
            "ssn",
            "credit_card",
            "api_key",
            "private_key",
        ]

        # Get sample API responses from ZAP history
        messages = self.zap.core.messages()
        for msg in messages[-100:]:  # Check last 100 messages
            if "api" in msg.get("requestHeader", ""):
                response_body = msg.get("responseBody", "")
                try:
                    # Try to parse as JSON
                    data = json.loads(response_body)
                    exposed_fields = self._find_sensitive_fields(
                        data, sensitive_fields
                    )
                    if exposed_fields:
                        vulnerabilities.append(
                            {
                                "type": "excessive_data_exposure",
                                "url": msg.get("requestHeader", "").split(" ")[
                                    1
                                ],
                                "exposed_fields": exposed_fields,
                                "sample_data": str(data)[:200],
                            }
                        )
                except Exception as e:
                    # Log but continue - some request failures expected during rate limit test
                    logger.debug(f"Request failed during rate limit test: {e}")
                    continue

        return vulnerabilities

    def _find_sensitive_fields(
        self, data: Any, sensitive_fields: List[str]
    ) -> List[str]:
        """Recursively find sensitive field names in data"""
        found = []

        if isinstance(data, dict):
            for key, value in data.items():
                if any(
                    sensitive in key.lower() for sensitive in sensitive_fields
                ):
                    found.append(key)
                found.extend(
                    self._find_sensitive_fields(value, sensitive_fields)
                )
        elif isinstance(data, list):
            for item in data:
                found.extend(
                    self._find_sensitive_fields(item, sensitive_fields)
                )

        return found


class ZAPScanner:
    """OWASP ZAP scanner integration"""

    def __init__(self, config: DASTConfig):
        self.config = config
        self.zap = ZAPv2(
            apikey=config.api_key,
            proxies={"http": config.zap_proxy, "https": config.zap_proxy},
        )
        self.session_name = f"dast_scan_{int(time.time())}"

    def start_session(self) -> None:
        """Start a new ZAP session"""
        logger.info(f"Starting new ZAP session: {self.session_name}")
        self.zap.core.new_session(name=self.session_name, overwrite=True)

        # Set up context
        self.zap.context.new_context(self.config.context_name)

        # Include/exclude URLs
        for url in self.config.include_urls:
            self.zap.context.include_in_context(self.config.context_name, url)

        for url in self.config.exclude_urls:
            self.zap.context.exclude_from_context(
                self.config.context_name, url
            )

    def spider_target(self) -> None:
        """Spider the target application"""
        logger.info(f"Starting spider scan of {self.config.target_url}")

        # Traditional spider
        scan_id = self.zap.spider.scan(self.config.target_url)
        self._wait_for_spider(scan_id)

        # AJAX spider if enabled
        if self.config.ajax_spider:
            logger.info("Starting AJAX spider scan")
            self.zap.ajaxSpider.scan(self.config.target_url)
            self._wait_for_ajax_spider()

    def passive_scan(self) -> None:
        """Wait for passive scanning to complete"""
        logger.info("Waiting for passive scanning to complete")
        time.sleep(self.config.passive_scan_wait)

        while int(self.zap.pscan.records_to_scan) > 0:
            logger.info(
                f"Passive scan records remaining: {self.zap.pscan.records_to_scan}"
            )
            time.sleep(5)

    def active_scan(self) -> Optional[str]:
        """Run active scanning"""
        if not self.config.active_scan:
            return None

        logger.info(f"Starting active scan of {self.config.target_url}")

        # Configure scan policy if provided
        if self.config.scan_policy:
            scan_policy_name = self.config.scan_policy
        else:
            scan_policy_name = "Default Policy"

        # Start active scan
        scan_id = self.zap.ascan.scan(
            self.config.target_url, scanpolicyname=scan_policy_name
        )

        # Wait for active scan
        start_time = time.time()
        while int(self.zap.ascan.status(scan_id)) < 100:
            if time.time() - start_time > self.config.max_scan_duration:
                logger.warning("Active scan timeout reached, stopping scan")
                self.zap.ascan.stop(scan_id)
                break

            progress = int(self.zap.ascan.status(scan_id))
            logger.info(f"Active scan progress: {progress}%")
            time.sleep(10)

        return scan_id

    def get_alerts(self) -> List[Alert]:
        """Get all alerts from the scan"""
        alerts = []
        raw_alerts = self.zap.core.alerts(baseurl=self.config.target_url)

        for raw_alert in raw_alerts:
            alert = Alert(
                name=raw_alert.get("name", ""),
                risk=RiskLevel(int(raw_alert.get("risk", 0))),
                confidence=raw_alert.get("confidence", ""),
                description=raw_alert.get("description", ""),
                solution=raw_alert.get("solution", ""),
                reference=raw_alert.get("reference", ""),
                cwe_id=int(raw_alert.get("cweid"))
                if raw_alert.get("cweid")
                else None,
                wasc_id=int(raw_alert.get("wascid"))
                if raw_alert.get("wascid")
                else None,
                instances=[
                    {
                        "uri": raw_alert.get("url", ""),
                        "method": raw_alert.get("method", ""),
                        "param": raw_alert.get("param", ""),
                    }
                ],
                tags=raw_alert.get("tags", {}),
            )
            alerts.append(alert)

        return alerts

    def _wait_for_spider(self, scan_id: str) -> None:
        """Wait for spider scan to complete"""
        while int(self.zap.spider.status(scan_id)) < 100:
            progress = int(self.zap.spider.status(scan_id))
            logger.info(f"Spider progress: {progress}%")
            time.sleep(5)

    def _wait_for_ajax_spider(self) -> None:
        """Wait for AJAX spider to complete"""
        while self.zap.ajaxSpider.status == "running":
            logger.info("AJAX Spider still running...")
            time.sleep(5)


class DASTOrchestrator:
    """Orchestrate complete DAST scan"""

    def __init__(self, config: DASTConfig):
        self.config = config
        self.zap_scanner = ZAPScanner(config)
        self.api_scanner = APIScanner(self.zap_scanner.zap, config)
        self.auth_tester = AuthenticationTester(self.zap_scanner.zap, config)

    async def run_scan(self) -> Dict[str, Any]:
        """Run complete DAST scan"""
        logger.info("Starting DAST scan")
        scan_results = {
            "start_time": time.time(),
            "target": self.config.target_url,
            "vulnerabilities": [],
            "alerts": [],
            "stats": {},
        }

        try:
            # Start ZAP session
            self.zap_scanner.start_session()

            # Import OpenAPI spec if available
            openapi_url = urljoin(
                self.config.target_url, "/api/v1/openapi.json"
            )
            self.api_scanner.import_openapi_spec(openapi_url)

            # Spider the target
            self.zap_scanner.spider_target()

            # Run authentication tests
            auth_vulns = await self.auth_tester.test_authentication_bypass()
            scan_results["vulnerabilities"].extend(auth_vulns)

            jwt_vulns = await self.auth_tester.test_jwt_vulnerabilities()
            scan_results["vulnerabilities"].extend(jwt_vulns)

            # Run API-specific tests
            api_vulns = self.api_scanner.test_api_vulnerabilities()
            scan_results["vulnerabilities"].extend(api_vulns)

            # Passive scan
            self.zap_scanner.passive_scan()

            # Active scan
            self.zap_scanner.active_scan()

            # Get alerts
            alerts = self.zap_scanner.get_alerts()
            scan_results["alerts"] = [
                self._alert_to_dict(alert) for alert in alerts
            ]

            # Calculate statistics
            scan_results["stats"] = self._calculate_stats(
                alerts, scan_results["vulnerabilities"]
            )

            # Check if scan passed based on risk threshold
            scan_results["passed"] = self._check_threshold(alerts)

        except Exception as e:
            logger.error(f"Error during DAST scan: {e}")
            scan_results["error"] = str(e)

        scan_results["end_time"] = time.time()
        scan_results["duration"] = (
            scan_results["end_time"] - scan_results["start_time"]
        )

        return scan_results

    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert Alert to dictionary"""
        return {
            "name": alert.name,
            "risk": alert.risk.name,
            "confidence": alert.confidence,
            "description": alert.description,
            "solution": alert.solution,
            "reference": alert.reference,
            "cwe_id": alert.cwe_id,
            "wasc_id": alert.wasc_id,
            "instances": alert.instances,
            "tags": alert.tags,
        }

    def _calculate_stats(
        self, alerts: List[Alert], custom_vulns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate scan statistics"""
        risk_counts = {level.name: 0 for level in RiskLevel}
        for alert in alerts:
            risk_counts[alert.risk.name] += 1

        return {
            "total_alerts": len(alerts),
            "custom_vulnerabilities": len(custom_vulns),
            "risk_distribution": risk_counts,
            "unique_vulnerabilities": len(set(alert.name for alert in alerts)),
        }

    def _check_threshold(self, alerts: List[Alert]) -> bool:
        """Check if scan meets risk threshold"""
        for alert in alerts:
            if alert.risk.value >= self.config.risk_threshold.value:
                return False
        return True

    def export_report(
        self, scan_results: Dict[str, Any], output_path: Path
    ) -> None:
        """Export scan results to file"""
        with open(output_path, "w") as f:
            json.dump(scan_results, f, indent=2)

        logger.info(f"DAST scan results exported to {output_path}")


def main():
    """Main entry point for DAST scanner"""
    import argparse

    parser = argparse.ArgumentParser(description="DAST Security Scanner")
    parser.add_argument("--target", required=True, help="Target URL to scan")
    parser.add_argument(
        "--zap-proxy", default="http://127.0.0.1:8080", help="ZAP proxy URL"
    )
    parser.add_argument("--api-key", help="ZAP API key")
    parser.add_argument(
        "--no-active-scan", action="store_true", help="Skip active scanning"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dast-results.json"),
        help="Output file for results",
    )

    args = parser.parse_args()

    # Create configuration
    config = DASTConfig(
        target_url=args.target,
        zap_proxy=args.zap_proxy,
        api_key=args.api_key,
        active_scan=not args.no_active_scan,
    )

    # Run scan
    orchestrator = DASTOrchestrator(config)
    scan_results = asyncio.run(orchestrator.run_scan())

    # Export results
    orchestrator.export_report(scan_results, args.output)

    # Exit with appropriate code
    import sys

    sys.exit(0 if scan_results.get("passed", False) else 1)


if __name__ == "__main__":
    main()
