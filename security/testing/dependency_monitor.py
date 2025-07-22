"""
Dependency Security Monitor

Monitors dependencies for security vulnerabilities using multiple sources.
Integrates with Snyk, GitHub Security Advisories, and manages automated updates.
"""

import asyncio
import json
import logging
import re
import subprocess  # nosec B404 # Required for dependency monitoring tools (pip, git, gh)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import toml
from packaging import version
from packaging.requirements import Requirement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def score(self) -> int:
        """Get numerical score for comparison"""
        scores = {
            VulnerabilitySeverity.CRITICAL: 4,
            VulnerabilitySeverity.HIGH: 3,
            VulnerabilitySeverity.MEDIUM: 2,
            VulnerabilitySeverity.LOW: 1,
        }
        return scores[self]


@dataclass
class Vulnerability:
    """Dependency vulnerability information"""

    id: str
    package: str
    installed_version: str
    affected_versions: str
    fixed_versions: List[str]
    severity: VulnerabilitySeverity
    title: str
    description: str
    published_date: datetime
    cve_ids: List[str] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)
    exploit_available: bool = False
    references: List[str] = field(default_factory=list)
    patched: bool = False


@dataclass
class Dependency:
    """Dependency information"""

    name: str
    version: str
    source: str  # pip, npm, etc.
    direct: bool = True
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    latest_version: Optional[str] = None
    update_available: bool = False
    license: Optional[str] = None


@dataclass
class MonitorConfig:
    """Configuration for dependency monitoring"""

    project_root: Path
    check_interval: int = 3600  # Check every hour
    severity_threshold: VulnerabilitySeverity = VulnerabilitySeverity.HIGH
    auto_update: bool = False
    auto_update_severity: VulnerabilitySeverity = VulnerabilitySeverity.CRITICAL
    create_prs: bool = True
    snyk_token: Optional[str] = None
    github_token: Optional[str] = None
    pypi_mirror: str = "https://pypi.org/pypi"
    npm_registry: str = "https://registry.npmjs.org"
    ignored_vulnerabilities: Set[str] = field(default_factory=set)
    ignored_packages: Set[str] = field(default_factory=set)
    max_update_frequency: timedelta = timedelta(days=1)


class DependencyScanner:
    """Scan project for dependencies"""

    def __init__(self, config: MonitorConfig):
        self.config = config

    def scan_dependencies(self) -> List[Dependency]:
        """Scan all project dependencies"""
        dependencies = []

        # Scan Python dependencies
        python_deps = self._scan_python_dependencies()
        dependencies.extend(python_deps)

        # Scan Node.js dependencies
        node_deps = self._scan_node_dependencies()
        dependencies.extend(node_deps)

        # Scan Go dependencies
        go_deps = self._scan_go_dependencies()
        dependencies.extend(go_deps)

        return dependencies

    def _scan_python_dependencies(self) -> List[Dependency]:
        """Scan Python dependencies from requirements files and pip"""
        dependencies = []
        seen = set()

        # Scan requirements files
        req_files = list(self.config.project_root.glob("*requirements*.txt"))
        for req_file in req_files:
            with open(req_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            req = Requirement(line)
                            if req.name not in seen:
                                seen.add(req.name)
                                # Get actual installed version
                                installed_version = self._get_pip_version(req.name)
                                if installed_version:
                                    dependencies.append(
                                        Dependency(
                                            name=req.name,
                                            version=installed_version,
                                            source="pip",
                                            direct=True,
                                        )
                                    )
                        except Exception as e:
                            logger.warning(f"Failed to parse requirement: {line} - {e}")

        # Also scan pyproject.toml if it exists
        pyproject_path = self.config.project_root / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path) as f:
                data = toml.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                for dep in deps:
                    try:
                        req = Requirement(dep)
                        if req.name not in seen:
                            seen.add(req.name)
                            installed_version = self._get_pip_version(req.name)
                            if installed_version:
                                dependencies.append(
                                    Dependency(
                                        name=req.name,
                                        version=installed_version,
                                        source="pip",
                                        direct=True,
                                    )
                                )
                    except Exception:  # nosec B110 # Safe fallback for malformed requirements files
                        pass

        # Get all installed packages (including transitive)
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                all_packages = json.loads(result.stdout)
                for pkg in all_packages:
                    if pkg["name"] not in seen:
                        dependencies.append(
                            Dependency(
                                name=pkg["name"],
                                version=pkg["version"],
                                source="pip",
                                direct=False,
                            )
                        )
        except Exception:  # nosec B110 # Safe fallback if pip list fails
            pass

        return dependencies

    def _scan_node_dependencies(self) -> List[Dependency]:
        """Scan Node.js dependencies from package.json"""
        dependencies = []
        package_json_files = list(self.config.project_root.glob("**/package.json"))

        for package_json in package_json_files:
            # Skip node_modules
            if "node_modules" in str(package_json):
                continue

            try:
                with open(package_json) as f:
                    data = json.load(f)

                # Direct dependencies
                for name, version_spec in data.get("dependencies", {}).items():
                    # Get actual installed version
                    installed_version = self._get_npm_version(name, package_json.parent)
                    if installed_version:
                        dependencies.append(
                            Dependency(
                                name=name,
                                version=installed_version,
                                source="npm",
                                direct=True,
                            )
                        )

                # Dev dependencies
                for name, version_spec in data.get("devDependencies", {}).items():
                    installed_version = self._get_npm_version(name, package_json.parent)
                    if installed_version:
                        dependencies.append(
                            Dependency(
                                name=name,
                                version=installed_version,
                                source="npm",
                                direct=True,
                            )
                        )

            except Exception as e:
                logger.warning(f"Failed to parse {package_json}: {e}")

        return dependencies

    def _scan_go_dependencies(self) -> List[Dependency]:
        """Scan Go dependencies from go.mod"""
        dependencies = []
        go_mod_path = self.config.project_root / "go.mod"

        if go_mod_path.exists():
            try:
                result = subprocess.run(
                    ["go", "list", "-m", "-json", "all"],
                    cwd=self.config.project_root,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    # Parse JSON lines
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            dep_info = json.loads(line)
                            if dep_info.get("Path") and dep_info.get("Version"):
                                dependencies.append(
                                    Dependency(
                                        name=dep_info["Path"],
                                        version=dep_info["Version"],
                                        source="go",
                                        direct=dep_info.get("Main", False),
                                    )
                                )
            except Exception as e:
                logger.warning(f"Failed to scan Go dependencies: {e}")

        return dependencies

    def _get_pip_version(self, package_name: str) -> Optional[str]:
        """Get installed pip package version"""
        try:
            result = subprocess.run(["pip", "show", package_name], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        return line.split(":")[1].strip()
        except Exception:
            pass
        return None

    def _get_npm_version(self, package_name: str, working_dir: Path) -> Optional[str]:
        """Get installed npm package version"""
        try:
            result = subprocess.run(
                ["npm", "list", package_name, "--json"],
                cwd=working_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Navigate through the structure to find the version
                deps = data.get("dependencies", {})
                if package_name in deps:
                    return deps[package_name].get("version")
        except Exception:
            pass
        return None


class VulnerabilityDatabase:
    """Interface to vulnerability databases"""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def check_vulnerabilities(self, dependencies: List[Dependency]) -> List[Dependency]:
        """Check dependencies for known vulnerabilities"""
        # Check from multiple sources
        tasks = []

        # Python vulnerabilities from PyUp Safety DB
        python_deps = [d for d in dependencies if d.source == "pip"]
        if python_deps:
            tasks.append(self._check_python_vulnerabilities(python_deps))

        # NPM vulnerabilities
        npm_deps = [d for d in dependencies if d.source == "npm"]
        if npm_deps:
            tasks.append(self._check_npm_vulnerabilities(npm_deps))

        # Snyk vulnerabilities (if token available)
        if self.config.snyk_token:
            tasks.append(self._check_snyk_vulnerabilities(dependencies))

        # GitHub Security Advisories
        if self.config.github_token:
            tasks.append(self._check_github_advisories(dependencies))

        # Run all checks concurrently
        if tasks:
            await asyncio.gather(*tasks)

        return dependencies

    async def _check_python_vulnerabilities(self, dependencies: List[Dependency]) -> None:
        """Check Python packages against PyUp Safety database"""
        try:
            # Download the latest safety database
            async with self.session.get(
                "https://raw.githubusercontent.com/pyupio/safety-db/master/data/insecure_full.json"
            ) as response:
                if response.status == 200:
                    safety_db = await response.json()

                    for dep in dependencies:
                        if dep.name in safety_db:
                            for vuln_data in safety_db[dep.name]:
                                # Check if version is affected
                                if self._is_version_affected(
                                    dep.version, vuln_data.get("specs", [])
                                ):
                                    vuln = Vulnerability(
                                        id=vuln_data.get(
                                            "id",
                                            f"PYUP-{dep.name}-{len(dep.vulnerabilities)}",
                                        ),
                                        package=dep.name,
                                        installed_version=dep.version,
                                        affected_versions=", ".join(vuln_data.get("specs", [])),
                                        fixed_versions=self._extract_fixed_versions(
                                            vuln_data.get("specs", [])
                                        ),
                                        severity=self._estimate_severity(vuln_data),
                                        title=vuln_data.get(
                                            "advisory",
                                            "Vulnerability in " + dep.name,
                                        ),
                                        description=vuln_data.get("description", ""),
                                        published_date=datetime.now(),  # Safety DB doesn't provide dates
                                        cve_ids=(
                                            [vuln_data.get("cve")] if vuln_data.get("cve") else []
                                        ),
                                    )
                                    dep.vulnerabilities.append(vuln)

        except Exception as e:
            logger.error(f"Error checking Python vulnerabilities: {e}")

    async def _check_npm_vulnerabilities(self, dependencies: List[Dependency]) -> None:
        """Check NPM packages for vulnerabilities"""
        for dep in dependencies:
            try:
                async with self.session.get(f"{self.config.npm_registry}/{dep.name}") as response:
                    if response.status == 200:
                        data = await response.json()

                        # Check if there's a security holding
                        if "security" in data:
                            security_info = data["security"]
                            vuln = Vulnerability(
                                id=f"NPM-{dep.name}-SECURITY",
                                package=dep.name,
                                installed_version=dep.version,
                                affected_versions="<"
                                + security_info.get("patched_version", "unknown"),
                                fixed_versions=[security_info.get("patched_version", "unknown")],
                                severity=VulnerabilitySeverity.HIGH,
                                title="Security holding on package",
                                description=security_info.get(
                                    "message",
                                    "Package has been marked with security issues",
                                ),
                                published_date=datetime.now(),
                            )
                            dep.vulnerabilities.append(vuln)

            except Exception as e:
                logger.warning(f"Error checking NPM vulnerabilities for {dep.name}: {e}")

    async def _check_snyk_vulnerabilities(self, dependencies: List[Dependency]) -> None:
        """Check vulnerabilities using Snyk API"""
        # Group by source for batch checking
        by_source: Dict[str, List[Dependency]] = {}
        for dep in dependencies:
            if dep.source not in by_source:
                by_source[dep.source] = []
            by_source[dep.source].append(dep)

        headers = {"Authorization": f"token {self.config.snyk_token}"}

        for source, deps in by_source.items():
            # Prepare package list for Snyk
            packages = [{"name": d.name, "version": d.version} for d in deps]

            try:
                async with self.session.post(
                    f"https://snyk.io/api/v1/test/{source}",
                    json={"packages": packages},
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._process_snyk_results(data, deps)
            except Exception as e:
                logger.error(f"Error checking Snyk vulnerabilities: {e}")

    async def _check_github_advisories(self, dependencies: List[Dependency]) -> None:
        """Check GitHub Security Advisories"""
        headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        for dep in dependencies:
            ecosystem = self._map_to_github_ecosystem(dep.source)
            if not ecosystem:
                continue

            query = """
            query($ecosystem: String!, $package: String!) {
                securityVulnerabilities(ecosystem: $ecosystem, package: $package, first: 100) {
                    nodes {
                        advisory {
                            ghsaId
                            summary
                            description
                            severity
                            publishedAt
                            identifiers {
                                type
                                value
                            }
                            references {
                                url
                            }
                        }
                        vulnerableVersionRange
                        firstPatchedVersion {
                            identifier
                        }
                    }
                }
            }
            """

            try:
                async with self.session.post(
                    "https://api.github.com/graphql",
                    json={
                        "query": query,
                        "variables": {
                            "ecosystem": ecosystem,
                            "package": dep.name,
                        },
                    },
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._process_github_advisories(data, dep)
            except Exception as e:
                logger.warning(f"Error checking GitHub advisories for {dep.name}: {e}")

    def _is_version_affected(self, version_str: str, specs: List[str]) -> bool:
        """Check if version matches vulnerability specs"""
        try:
            current = version.parse(version_str)
            for spec in specs:
                # Parse version spec (e.g., "<2.0.0", ">=1.0.0,<1.5.0")
                if self._matches_spec(current, spec):
                    return True
        except Exception:
            pass
        return False

    def _matches_spec(self, current_version, spec: str) -> bool:
        """Check if version matches a specification"""
        # Simple implementation - in production, use packaging library
        if spec.startswith("<"):
            target = version.parse(spec[1:])
            return current_version < target
        elif spec.startswith("<="):
            target = version.parse(spec[2:])
            return current_version <= target
        elif spec.startswith(">"):
            target = version.parse(spec[1:])
            return current_version > target
        elif spec.startswith(">="):
            target = version.parse(spec[2:])
            return current_version >= target
        elif spec.startswith("=="):
            target = version.parse(spec[2:])
            return current_version == target
        return False

    def _extract_fixed_versions(self, specs: List[str]) -> List[str]:
        """Extract fixed versions from vulnerability specs"""
        fixed = []
        for spec in specs:
            # Look for >= patterns which indicate fixed versions
            if spec.startswith(">="):
                fixed.append(spec[2:])
        return fixed

    def _estimate_severity(self, vuln_data: Dict[str, Any]) -> VulnerabilitySeverity:
        """Estimate severity from vulnerability data"""
        # Look for severity indicators in the text
        text = (vuln_data.get("advisory", "") + vuln_data.get("description", "")).lower()

        if any(word in text for word in ["critical", "severe", "remote code execution", "rce"]):
            return VulnerabilitySeverity.CRITICAL
        elif any(word in text for word in ["high", "sql injection", "xss", "authentication"]):
            return VulnerabilitySeverity.HIGH
        elif any(word in text for word in ["medium", "moderate", "dos"]):
            return VulnerabilitySeverity.MEDIUM
        else:
            return VulnerabilitySeverity.LOW

    def _map_to_github_ecosystem(self, source: str) -> Optional[str]:
        """Map package source to GitHub ecosystem name"""
        mapping = {
            "pip": "PIP",
            "npm": "NPM",
            "go": "GO",
            "maven": "MAVEN",
            "nuget": "NUGET",
        }
        return mapping.get(source)

    def _process_snyk_results(self, data: Dict[str, Any], dependencies: List[Dependency]) -> None:
        """Process Snyk vulnerability results"""
        # Implementation depends on Snyk API response format
        pass

    def _process_github_advisories(self, data: Dict[str, Any], dependency: Dependency) -> None:
        """Process GitHub advisory results"""
        vulnerabilities = data.get("data", {}).get("securityVulnerabilities", {}).get("nodes", [])

        for vuln_node in vulnerabilities:
            advisory = vuln_node.get("advisory", {})

            # Check if version is in vulnerable range
            vuln_range = vuln_node.get("vulnerableVersionRange", "")
            if self._is_version_affected(dependency.version, [vuln_range]):
                cve_ids = [
                    ident["value"]
                    for ident in advisory.get("identifiers", [])
                    if ident["type"] == "CVE"
                ]

                vuln = Vulnerability(
                    id=advisory.get("ghsaId", ""),
                    package=dependency.name,
                    installed_version=dependency.version,
                    affected_versions=vuln_range,
                    fixed_versions=[vuln_node.get("firstPatchedVersion", {}).get("identifier", "")],
                    severity=VulnerabilitySeverity(advisory.get("severity", "LOW").lower()),
                    title=advisory.get("summary", ""),
                    description=advisory.get("description", ""),
                    published_date=datetime.fromisoformat(
                        advisory.get("publishedAt", "").replace("Z", "+00:00")
                    ),
                    cve_ids=cve_ids,
                    references=[ref["url"] for ref in advisory.get("references", [])],
                )
                dependency.vulnerabilities.append(vuln)


class UpdateManager:
    """Manage dependency updates"""

    def __init__(self, config: MonitorConfig):
        self.config = config

    async def check_updates(self, dependencies: List[Dependency]) -> None:
        """Check for available updates"""
        for dep in dependencies:
            if dep.source == "pip":
                latest = await self._get_latest_pip_version(dep.name)
            elif dep.source == "npm":
                latest = await self._get_latest_npm_version(dep.name)
            else:
                latest = None

            if latest and latest != dep.version:
                dep.latest_version = latest
                dep.update_available = True

    async def _get_latest_pip_version(self, package_name: str) -> Optional[str]:
        """Get latest version from PyPI"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.pypi_mirror}/{package_name}/json"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("info", {}).get("version")
        except Exception:
            pass
        return None

    async def _get_latest_npm_version(self, package_name: str) -> Optional[str]:
        """Get latest version from NPM"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.npm_registry}/{package_name}/latest"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("version")
        except Exception:
            pass
        return None

    def create_update_pr(self, dependencies: List[Dependency]) -> Optional[str]:
        """Create PR for dependency updates"""
        if not self.config.create_prs or not self.config.github_token:
            return None

        updates_needed = [d for d in dependencies if d.update_available and self._should_update(d)]

        if not updates_needed:
            return None

        # Create branch
        branch_name = f"dependency-updates-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(["git", "checkout", "-b", branch_name])

        # Update dependencies
        for dep in updates_needed:
            self._update_dependency(dep)

        # Commit changes
        subprocess.run(["git", "add", "-A"])
        commit_message = self._generate_commit_message(updates_needed)
        subprocess.run(["git", "commit", "-m", commit_message])

        # Push branch
        subprocess.run(["git", "push", "origin", branch_name])

        # Create PR using GitHub CLI
        pr_body = self._generate_pr_body(updates_needed)
        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--title",
                "Security: Dependency Updates",
                "--body",
                pr_body,
                "--label",
                "security,dependencies",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logger.error(f"Failed to create PR: {result.stderr}")
            return None

    def _should_update(self, dependency: Dependency) -> bool:
        """Check if dependency should be updated"""
        if dependency.name in self.config.ignored_packages:
            return False

        # Check if any vulnerability requires update
        for vuln in dependency.vulnerabilities:
            if vuln.id in self.config.ignored_vulnerabilities:
                continue

            if vuln.severity.score >= self.config.auto_update_severity.score:
                return True

        return False

    def _update_dependency(self, dependency: Dependency) -> None:
        """Update a single dependency"""
        if dependency.source == "pip":
            self._update_pip_dependency(dependency)
        elif dependency.source == "npm":
            self._update_npm_dependency(dependency)

    def _update_pip_dependency(self, dependency: Dependency) -> None:
        """Update Python dependency in requirements files"""
        req_files = list(self.config.project_root.glob("*requirements*.txt"))

        for req_file in req_files:
            content = req_file.read_text()
            # Update version in requirements
            pattern = rf"^{re.escape(dependency.name)}[=<>~!].*$"
            replacement = f"{dependency.name}=={dependency.latest_version}"
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

            if new_content != content:
                req_file.write_text(new_content)

    def _update_npm_dependency(self, dependency: Dependency) -> None:
        """Update Node.js dependency"""
        package_json_files = list(self.config.project_root.glob("**/package.json"))

        for package_json in package_json_files:
            if "node_modules" in str(package_json):
                continue

            with open(package_json) as f:
                data = json.load(f)

            updated = False
            for dep_type in ["dependencies", "devDependencies"]:
                if dep_type in data and dependency.name in data[dep_type]:
                    data[dep_type][dependency.name] = f"^{dependency.latest_version}"
                    updated = True

            if updated:
                with open(package_json, "w") as f:
                    json.dump(data, f, indent=2)

    def _generate_commit_message(self, updates: List[Dependency]) -> str:
        """Generate commit message for updates"""
        if len(updates) == 1:
            dep = updates[0]
            return f"security: update {dep.name} from {dep.version} to {dep.latest_version}"
        else:
            return f"security: update {len(updates)} dependencies"

    def _generate_pr_body(self, updates: List[Dependency]) -> str:
        """Generate PR body with update details"""
        body = "## Dependency Security Updates\n\n"
        body += (
            "This PR updates the following dependencies to address security vulnerabilities:\n\n"
        )

        for dep in updates:
            body += f"### {dep.name}\n"
            body += f"- **Current version:** {dep.version}\n"
            body += f"- **Updated to:** {dep.latest_version}\n"

            if dep.vulnerabilities:
                body += "- **Vulnerabilities fixed:**\n"
                for vuln in dep.vulnerabilities:
                    body += f"  - {vuln.severity.value.upper()}: {vuln.title}\n"
                    if vuln.cve_ids:
                        body += f"    - CVE: {', '.join(vuln.cve_ids)}\n"

            body += "\n"

        body += "## Testing\n\n"
        body += "- [ ] All tests pass\n"
        body += "- [ ] No breaking changes identified\n"
        body += "- [ ] Application starts successfully\n"

        return body


class DependencyMonitor:
    """Main dependency monitoring orchestrator"""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.scanner = DependencyScanner(config)
        self.update_manager = UpdateManager(config)
        self.last_update_times: Dict[str, datetime] = {}

    async def run_continuous_monitoring(self) -> None:
        """Run continuous dependency monitoring"""
        logger.info("Starting continuous dependency monitoring")

        while True:
            try:
                await self.check_dependencies()
                await asyncio.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def check_dependencies(self) -> Dict[str, Any]:
        """Run single dependency check"""
        logger.info("Scanning dependencies...")
        start_time = datetime.now()

        # Scan dependencies
        dependencies = self.scanner.scan_dependencies()
        logger.info(f"Found {len(dependencies)} dependencies")

        # Check for vulnerabilities
        async with VulnerabilityDatabase(self.config) as vuln_db:
            dependencies = await vuln_db.check_vulnerabilities(dependencies)

        # Check for updates
        await self.update_manager.check_updates(dependencies)

        # Filter vulnerabilities
        vulnerable_deps = [d for d in dependencies if d.vulnerabilities]
        critical_vulns = []
        high_vulns = []

        for dep in vulnerable_deps:
            for vuln in dep.vulnerabilities:
                if vuln.severity == VulnerabilitySeverity.CRITICAL:
                    critical_vulns.append((dep, vuln))
                elif vuln.severity == VulnerabilitySeverity.HIGH:
                    high_vulns.append((dep, vuln))

        # Generate report
        report = {
            "scan_time": start_time.isoformat(),
            "total_dependencies": len(dependencies),
            "vulnerable_dependencies": len(vulnerable_deps),
            "critical_vulnerabilities": len(critical_vulns),
            "high_vulnerabilities": len(high_vulns),
            "dependencies_with_updates": len([d for d in dependencies if d.update_available]),
        }

        # Auto-update if configured
        if self.config.auto_update and (
            critical_vulns
            or (self.config.auto_update_severity == VulnerabilitySeverity.HIGH and high_vulns)
        ):
            pr_url = await self._handle_auto_update(dependencies)
            if pr_url:
                report["auto_update_pr"] = pr_url

        # Alert on critical vulnerabilities
        if critical_vulns:
            await self._send_critical_alert(critical_vulns)

        # Save detailed report
        self._save_report(dependencies, report)

        return report

    async def _handle_auto_update(self, dependencies: List[Dependency]) -> Optional[str]:
        """Handle automatic updates"""
        # Check update frequency
        now = datetime.now()
        last_update = self.last_update_times.get("auto_update", datetime.min)

        if now - last_update < self.config.max_update_frequency:
            logger.info("Skipping auto-update due to frequency limit")
            return None

        pr_url = self.update_manager.create_update_pr(dependencies)
        if pr_url:
            self.last_update_times["auto_update"] = now

        return pr_url

    async def _send_critical_alert(
        self, critical_vulns: List[Tuple[Dependency, Vulnerability]]
    ) -> None:
        """Send alert for critical vulnerabilities"""
        logger.critical(f"CRITICAL VULNERABILITIES DETECTED: {len(critical_vulns)} issues found")

        for dep, vuln in critical_vulns:
            logger.critical(
                f"  - {dep.name} {dep.version}: {vuln.title} (CVE: {', '.join(vuln.cve_ids)})"
            )

        # In production, this would send to alerting system

    def _save_report(self, dependencies: List[Dependency], summary: Dict[str, Any]) -> None:
        """Save detailed vulnerability report"""
        report_path = (
            self.config.project_root
            / f"dependency-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )

        full_report = {
            "summary": summary,
            "dependencies": [
                {
                    "name": d.name,
                    "version": d.version,
                    "latest_version": d.latest_version,
                    "source": d.source,
                    "direct": d.direct,
                    "update_available": d.update_available,
                    "vulnerabilities": [
                        {
                            "id": v.id,
                            "severity": v.severity.value,
                            "title": v.title,
                            "cve_ids": v.cve_ids,
                            "fixed_versions": v.fixed_versions,
                        }
                        for v in d.vulnerabilities
                    ],
                }
                for d in dependencies
                if d.vulnerabilities or d.update_available
            ],
        }

        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)

        logger.info(f"Dependency report saved to {report_path}")


def main():
    """Main entry point"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Dependency Security Monitor")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory",
    )
    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--auto-update", action="store_true", help="Enable automatic updates")
    parser.add_argument("--snyk-token", help="Snyk API token")
    parser.add_argument("--github-token", help="GitHub API token")

    args = parser.parse_args()

    # Create configuration
    config = MonitorConfig(
        project_root=args.project_root,
        auto_update=args.auto_update,
        snyk_token=args.snyk_token or os.environ.get("SNYK_TOKEN"),
        github_token=args.github_token or os.environ.get("GITHUB_TOKEN"),
    )

    # Create monitor
    monitor = DependencyMonitor(config)

    if args.continuous:
        # Run continuous monitoring
        asyncio.run(monitor.run_continuous_monitoring())
    else:
        # Run single check
        report = asyncio.run(monitor.check_dependencies())
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
