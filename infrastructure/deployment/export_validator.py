"""
Export Validator for Deployment Packages

Validates exported agent packages and ensures they are ready for deployment
on target hardware platforms.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status levels."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class HardwarePlatform(Enum):
    """Supported hardware platforms."""

    RASPBERRY_PI = "raspberry_pi"
    MAC_MINI = "mac_mini"
    JETSON_NANO = "jetson_nano"
    X86_LINUX = "x86_linux"
    GENERIC_ARM = "generic_arm"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical


@dataclass
class PackageManifest:
    """Package manifest information."""

    package_name: str
    version: str
    agent_class: str
    created_at: str
    platform: str
    dependencies: List[str]
    files: Dict[str, str]  # filename -> hash
    metadata: Dict[str, Any]


@dataclass
class HardwareRequirements:
    """Hardware requirements for deployment."""

    min_ram_mb: int
    min_storage_mb: int
    min_cpu_cores: int
    gpu_required: bool
    architecture: str  # arm64, x86_64, etc.
    os_family: str  # linux, darwin, etc.


class ExportValidator:
    """
    Validates exported agent packages for deployment.

    Performs comprehensive checks including:
    - Package structure and completeness
    - File integrity
    - Configuration validation
    - Dependency verification
    - Hardware compatibility
    """

    def __init__(self) -> None:
        """Initialize export validator."""
        self.required_files = {
            "manifest.json",
            "agent_config.json",
            "gnn_model.json",
            "requirements.txt",
            "run.sh",
            "README.md",
        }

        self.optional_files = {
            "docker-compose.yml",
            "Dockerfile",
            "health_check.py",
            "install.sh",
            "uninstall.sh",
        }

        self.hardware_profiles = {
            HardwarePlatform.RASPBERRY_PI: HardwareRequirements(
                min_ram_mb=1024,
                min_storage_mb=2048,
                min_cpu_cores=4,
                gpu_required=False,
                architecture="arm64",
                os_family="linux",
            ),
            HardwarePlatform.MAC_MINI: HardwareRequirements(
                min_ram_mb=8192,
                min_storage_mb=10240,
                min_cpu_cores=8,
                gpu_required=False,
                architecture="arm64",
                os_family="darwin",
            ),
            HardwarePlatform.JETSON_NANO: HardwareRequirements(
                min_ram_mb=4096,
                min_storage_mb=8192,
                min_cpu_cores=4,
                gpu_required=True,
                architecture="arm64",
                os_family="linux",
            ),
        }

    def validate_package(
        self, package_path: Path, target_platform: Optional[HardwarePlatform] = None
    ) -> List[ValidationResult]:
        """
        Validate an exported package.

        Args:
            package_path: Path to package file or directory
            target_platform: Target hardware platform

        Returns:
            List of validation results
        """
        results = []

        # Extract package if compressed
        if package_path.is_file():
            extract_dir = self._extract_package(package_path)
            if not extract_dir:
                results.append(
                    ValidationResult(
                        check_name="package_extraction",
                        status=ValidationStatus.FAILED,
                        message="Failed to extract package",
                        severity="critical",
                    )
                )
                return results
        else:
            extract_dir = package_path

        try:
            # Run validation checks
            results.extend(self._check_structure(extract_dir))
            results.extend(self._check_manifest(extract_dir))
            results.extend(self._check_files(extract_dir))
            results.extend(self._check_configuration(extract_dir))
            results.extend(self._check_dependencies(extract_dir))
            results.extend(self._check_scripts(extract_dir))

            if target_platform:
                results.extend(self._check_hardware_compatibility(extract_dir, target_platform))

            # Generate summary
            results.append(self._generate_summary(results))

        finally:
            # Cleanup if we extracted
            if package_path.is_file() and extract_dir.exists():
                shutil.rmtree(extract_dir)

        return results

    def _extract_package(self, package_path: Path) -> Optional[Path]:
        """Extract compressed package."""
        extract_dir = Path(tempfile.mkdtemp(prefix="freeagentics_validate_"))

        try:
            if package_path.suffix == ".zip":
                with zipfile.ZipFile(package_path, "r") as zf:
                    zf.extractall(extract_dir)
            elif package_path.suffix in [".tar", ".gz", ".tgz"]:
                with tarfile.open(package_path, "r:*") as tf:
                    tf.extractall(extract_dir)
            else:
                logger.error(f"Unsupported package format: {package_path.suffix}")
                shutil.rmtree(extract_dir)
                return None

            # Find actual package root (might be nested)
            manifest_paths = list(extract_dir.rglob("manifest.json"))
            if manifest_paths:
                return manifest_paths[0].parent

            return extract_dir

        except Exception as e:
            logger.error(f"Failed to extract package: {e}")
            shutil.rmtree(extract_dir)
            return None

    def _check_structure(self, package_dir: Path) -> List[ValidationResult]:
        """Check package structure."""
        results = []

        # Check for required files
        missing_files = []
        for required_file in self.required_files:
            if not (package_dir / required_file).exists():
                missing_files.append(required_file)

        if missing_files:
            results.append(
                ValidationResult(
                    check_name="package_structure",
                    status=ValidationStatus.FAILED,
                    message=f"Missing required files: {', '.join(missing_files)}",
                    details={"missing_files": missing_files},
                    severity="error",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="package_structure",
                    status=ValidationStatus.PASSED,
                    message="All required files present",
                )
            )

        # Check for recommended files
        missing_optional = []
        for optional_file in self.optional_files:
            if not (package_dir / optional_file).exists():
                missing_optional.append(optional_file)

        if missing_optional:
            results.append(
                ValidationResult(
                    check_name="optional_files",
                    status=ValidationStatus.WARNING,
                    message=f"Missing optional files: {', '.join(missing_optional)}",
                    details={"missing_files": missing_optional},
                    severity="warning",
                )
            )

        return results

    def _check_manifest(self, package_dir: Path) -> List[ValidationResult]:
        """Check package manifest."""
        results = []
        manifest_path = package_dir / "manifest.json"

        if not manifest_path.exists():
            return [
                ValidationResult(
                    check_name="manifest",
                    status=ValidationStatus.FAILED,
                    message="Manifest file not found",
                    severity="critical",
                )
            ]

        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)

            # Validate manifest structure
            required_fields = [
                "package_name",
                "version",
                "agent_class",
                "created_at",
                "platform",
                "files",
            ]

            missing_fields = [f for f in required_fields if f not in manifest_data]

            if missing_fields:
                results.append(
                    ValidationResult(
                        check_name="manifest_structure",
                        status=ValidationStatus.FAILED,
                        message=f"Missing manifest fields: {', '.join(missing_fields)}",
                        severity="error",
                    )
                )
            else:
                # Create manifest object
                manifest = PackageManifest(
                    package_name=manifest_data["package_name"],
                    version=manifest_data["version"],
                    agent_class=manifest_data["agent_class"],
                    created_at=manifest_data["created_at"],
                    platform=manifest_data["platform"],
                    dependencies=manifest_data.get("dependencies", []),
                    files=manifest_data["files"],
                    metadata=manifest_data.get("metadata", {}),
                )

                results.append(
                    ValidationResult(
                        check_name="manifest_structure",
                        status=ValidationStatus.PASSED,
                        message="Manifest structure valid",
                        details={"manifest": manifest.__dict__},
                    )
                )

                # Verify file hashes
                hash_results = self._verify_file_hashes(package_dir, manifest.files)
                results.extend(hash_results)

        except json.JSONDecodeError as e:
            results.append(
                ValidationResult(
                    check_name="manifest_parse",
                    status=ValidationStatus.FAILED,
                    message=f"Invalid manifest JSON: {e}",
                    severity="critical",
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    check_name="manifest_validation",
                    status=ValidationStatus.FAILED,
                    message=f"Manifest validation error: {e}",
                    severity="error",
                )
            )

        return results

    def _verify_file_hashes(
        self, package_dir: Path, file_hashes: Dict[str, str]
    ) -> List[ValidationResult]:
        """Verify file integrity using hashes."""
        results = []
        corrupted_files = []

        for filename, expected_hash in file_hashes.items():
            file_path = package_dir / filename

            if not file_path.exists():
                corrupted_files.append(f"{filename} (missing)")
                continue

            # Calculate actual hash
            actual_hash = self._calculate_file_hash(file_path)

            if actual_hash != expected_hash:
                corrupted_files.append(f"{filename} (hash mismatch)")

        if corrupted_files:
            results.append(
                ValidationResult(
                    check_name="file_integrity",
                    status=ValidationStatus.FAILED,
                    message=f"File integrity check failed: {', '.join(corrupted_files)}",
                    details={"corrupted_files": corrupted_files},
                    severity="critical",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="file_integrity",
                    status=ValidationStatus.PASSED,
                    message="All file hashes verified",
                )
            )

        return results

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _check_files(self, package_dir: Path) -> List[ValidationResult]:
        """Check individual files."""
        results = []

        # Check agent config
        config_path = package_dir / "agent_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                # Validate config structure
                if "agent_class" in config and "personality" in config:
                    results.append(
                        ValidationResult(
                            check_name="agent_config",
                            status=ValidationStatus.PASSED,
                            message="Agent configuration valid",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check_name="agent_config",
                            status=ValidationStatus.FAILED,
                            message="Invalid agent configuration structure",
                            severity="error",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="agent_config",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to parse agent config: {e}",
                        severity="error",
                    )
                )

        # Check GNN model
        model_path = package_dir / "gnn_model.json"
        if model_path.exists():
            try:
                with open(model_path) as f:
                    model = json.load(f)

                # Basic model validation
                if "metadata" in model and "layers" in model:
                    results.append(
                        ValidationResult(
                            check_name="gnn_model",
                            status=ValidationStatus.PASSED,
                            message="GNN model structure valid",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check_name="gnn_model",
                            status=ValidationStatus.WARNING,
                            message="GNN model may be incomplete",
                            severity="warning",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="gnn_model",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to parse GNN model: {e}",
                        severity="error",
                    )
                )

        return results

    def _check_configuration(self, package_dir: Path) -> List[ValidationResult]:
        """Check configuration files."""
        results = []

        # Check for environment configuration
        env_files = [".env", ".env.example", "config.yaml", "config.json"]
        env_found = False

        for env_file in env_files:
            if (package_dir / env_file).exists():
                env_found = True
                break

        if not env_found:
            results.append(
                ValidationResult(
                    check_name="environment_config",
                    status=ValidationStatus.WARNING,
                    message="No environment configuration found",
                    severity="warning",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="environment_config",
                    status=ValidationStatus.PASSED,
                    message="Environment configuration present",
                )
            )

        return results

    def _check_dependencies(self, package_dir: Path) -> List[ValidationResult]:
        """Check package dependencies."""
        results = []

        # Check Python dependencies
        req_path = package_dir / "requirements.txt"
        if req_path.exists():
            try:
                with open(req_path) as f:
                    requirements = f.read().strip().split("\n")

                # Check for problematic dependencies
                problematic = []
                for req in requirements:
                    if req and not req.startswith("#"):
                        # Check for version pinning
                        if "==" not in req and ">=" not in req:
                            problematic.append(f"{req} (no version)")

                if problematic:
                    results.append(
                        ValidationResult(
                            check_name="python_dependencies",
                            status=ValidationStatus.WARNING,
                            message=f"Unpinned dependencies: {', '.join(problematic[:3])}",
                            details={"unpinned": problematic},
                            severity="warning",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check_name="python_dependencies",
                            status=ValidationStatus.PASSED,
                            message="All Python dependencies properly pinned",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="python_dependencies",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to parse requirements.txt: {e}",
                        severity="error",
                    )
                )

        # Check for package.json (Node.js dependencies)
        package_json_path = package_dir / "package.json"
        if package_json_path.exists():
            try:
                with open(package_json_path) as f:
                    package_data = json.load(f)

                if "dependencies" in package_data:
                    results.append(
                        ValidationResult(
                            check_name="node_dependencies",
                            status=ValidationStatus.PASSED,
                            message="Node.js dependencies found",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="node_dependencies",
                        status=ValidationStatus.WARNING,
                        message=f"Failed to parse package.json: {e}",
                        severity="warning",
                    )
                )

        return results

    def _check_scripts(self, package_dir: Path) -> List[ValidationResult]:
        """Check executable scripts."""
        results = []

        # Check run script
        run_script = package_dir / "run.sh"
        if run_script.exists():
            # Check if executable
            if os.access(run_script, os.X_OK):
                results.append(
                    ValidationResult(
                        check_name="run_script",
                        status=ValidationStatus.PASSED,
                        message="Run script is executable",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check_name="run_script",
                        status=ValidationStatus.FAILED,
                        message="Run script is not executable",
                        severity="error",
                    )
                )

            # Check script content
            try:
                with open(run_script) as f:
                    content = f.read()

                # Basic checks
                if "#!/bin/" not in content:
                    results.append(
                        ValidationResult(
                            check_name="run_script_shebang",
                            status=ValidationStatus.WARNING,
                            message="Run script missing shebang",
                            severity="warning",
                        )
                    )

                if "python" not in content and "node" not in content:
                    results.append(
                        ValidationResult(
                            check_name="run_script_content",
                            status=ValidationStatus.WARNING,
                            message="Run script doesn't appear to launch application",
                            severity="warning",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="run_script_read",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to read run script: {e}",
                        severity="error",
                    )
                )

        # Check health check script
        health_check = package_dir / "health_check.py"
        if health_check.exists():
            results.append(
                ValidationResult(
                    check_name="health_check",
                    status=ValidationStatus.PASSED,
                    message="Health check script present",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="health_check",
                    status=ValidationStatus.WARNING,
                    message="No health check script found",
                    severity="warning",
                )
            )

        return results

    def _check_hardware_compatibility(
        self, package_dir: Path, target_platform: HardwarePlatform
    ) -> List[ValidationResult]:
        """Check hardware compatibility."""
        results = []

        if target_platform not in self.hardware_profiles:
            results.append(
                ValidationResult(
                    check_name="hardware_compatibility",
                    status=ValidationStatus.SKIPPED,
                    message=f"No profile for platform: {target_platform.value}",
                )
            )
            return results

        requirements = self.hardware_profiles[target_platform]

        # Check manifest for platform compatibility
        manifest_path = package_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                # Check platform
                if manifest.get("platform") != target_platform.value:
                    results.append(
                        ValidationResult(
                            check_name="platform_match",
                            status=ValidationStatus.WARNING,
                            message=f"Package built for {manifest.get('platform')}, "
                            f"target is {target_platform.value}",
                            severity="warning",
                        )
                    )

                # Check architecture
                if "architecture" in manifest:
                    if manifest["architecture"] != requirements.architecture:
                        results.append(
                            ValidationResult(
                                check_name="architecture_compatibility",
                                status=ValidationStatus.FAILED,
                                message=f"Architecture mismatch: package is "
                                f"{manifest['architecture']}, target needs "
                                f"{requirements.architecture}",
                                severity="critical",
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                check_name="architecture_compatibility",
                                status=ValidationStatus.PASSED,
                                message="Architecture compatible",
                            )
                        )

                # Check resource requirements
                if "requirements" in manifest:
                    pkg_reqs = manifest["requirements"]

                    if pkg_reqs.get("min_ram_mb", 0) > requirements.min_ram_mb:
                        results.append(
                            ValidationResult(
                                check_name="memory_requirements",
                                status=ValidationStatus.WARNING,
                                message=f"Package requires {pkg_reqs['min_ram_mb']}MB RAM, "
                                f"platform has {requirements.min_ram_mb}MB",
                                severity="warning",
                            )
                        )

                    if pkg_reqs.get("gpu_required", False) and not requirements.gpu_required:
                        results.append(
                            ValidationResult(
                                check_name="gpu_requirements",
                                status=ValidationStatus.FAILED,
                                message="Package requires GPU but platform doesn't have one",
                                severity="critical",
                            )
                        )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="hardware_compatibility",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to check hardware compatibility: {e}",
                        severity="error",
                    )
                )

        return results

    def _generate_summary(self, results: List[ValidationResult]) -> ValidationResult:
        """Generate validation summary."""
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)

        if failed > 0:
            status = ValidationStatus.FAILED
            message = f"Validation failed: {failed} errors, {warnings} warnings"
            severity = "critical"
        elif warnings > 0:
            status = ValidationStatus.WARNING
            message = f"Validation passed with {warnings} warnings"
            severity = "warning"
        else:
            status = ValidationStatus.PASSED
            message = f"Validation passed: {passed} checks"
            severity = "info"

        return ValidationResult(
            check_name="validation_summary",
            status=status,
            message=message,
            details={
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "total": len(results),
            },
            severity=severity,
        )


class DeploymentVerifier:
    """
    Verifies deployment on target hardware.
    """

    def __init__(self) -> None:
        """Initialize deployment verifier."""
        self.test_timeout = 300  # 5 minutes

    def verify_deployment(
        self, package_dir: Path, platform: HardwarePlatform
    ) -> List[ValidationResult]:
        """
        Verify deployment on target platform.

        Args:
            package_dir: Deployed package directory
            platform: Target platform

        Returns:
            Verification results
        """
        results = []

        # Check if package is running
        results.extend(self._check_process_running(package_dir))

        # Check health endpoint
        results.extend(self._check_health_endpoint(package_dir))

        # Check resource usage
        results.extend(self._check_resource_usage(platform))

        # Check logs for errors
        results.extend(self._check_logs(package_dir))

        # Run functional tests
        results.extend(self._run_functional_tests(package_dir))

        return results

    def _check_process_running(self, package_dir: Path) -> List[ValidationResult]:
        """Check if agent process is running."""
        results = []

        # Look for PID file
        pid_file = package_dir / "agent.pid"
        if pid_file.exists():
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())

                # Check if process exists
                try:
                    os.kill(pid, 0)
                    results.append(
                        ValidationResult(
                            check_name="process_running",
                            status=ValidationStatus.PASSED,
                            message=f"Agent process running (PID: {pid})",
                        )
                    )
                except OSError:
                    results.append(
                        ValidationResult(
                            check_name="process_running",
                            status=ValidationStatus.FAILED,
                            message=f"Agent process not found (PID: {pid})",
                            severity="critical",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="process_check",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to check process: {e}",
                        severity="error",
                    )
                )
        else:
            results.append(
                ValidationResult(
                    check_name="pid_file",
                    status=ValidationStatus.WARNING,
                    message="No PID file found",
                    severity="warning",
                )
            )

        return results

    def _check_health_endpoint(self, package_dir: Path) -> List[ValidationResult]:
        """Check health endpoint if available."""
        results = []

        # Try to find health check configuration
        config_files = ["config.json", "agent_config.json", ".env"]
        health_port = None

        for config_file in config_files:
            config_path = package_dir / config_file
            if config_path.exists():
                try:
                    if config_file.endswith(".json"):
                        with open(config_path) as f:
                            config = json.load(f)
                        health_port = config.get("health_port", config.get("port"))
                    elif config_file == ".env":
                        with open(config_path) as f:
                            for line in f:
                                if "HEALTH_PORT" in line or "PORT" in line:
                                    health_port = int(line.split("=")[1].strip())

                    if health_port:
                        break

                except Exception:
                    continue

        if health_port:
            try:
                import requests  # type: ignore[import-untyped]

                response = requests.get(f"http://localhost:{health_port}/health", timeout=5)

                if response.status_code == 200:
                    results.append(
                        ValidationResult(
                            check_name="health_endpoint",
                            status=ValidationStatus.PASSED,
                            message="Health endpoint responding",
                            details={"response": response.json()},
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check_name="health_endpoint",
                            status=ValidationStatus.WARNING,
                            message=f"Health endpoint returned {response.status_code}",
                            severity="warning",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="health_endpoint",
                        status=ValidationStatus.FAILED,
                        message=f"Health endpoint not accessible: {e}",
                        severity="error",
                    )
                )
        else:
            results.append(
                ValidationResult(
                    check_name="health_endpoint",
                    status=ValidationStatus.SKIPPED,
                    message="No health endpoint configured",
                )
            )

        return results

    def _check_resource_usage(self, platform: HardwarePlatform) -> List[ValidationResult]:
        """Check resource usage on platform."""
        results = []

        try:
            import psutil

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                results.append(
                    ValidationResult(
                        check_name="cpu_usage",
                        status=ValidationStatus.WARNING,
                        message=f"High CPU usage: {cpu_percent}%",
                        severity="warning",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check_name="cpu_usage",
                        status=ValidationStatus.PASSED,
                        message=f"CPU usage normal: {cpu_percent}%",
                    )
                )

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                results.append(
                    ValidationResult(
                        check_name="memory_usage",
                        status=ValidationStatus.WARNING,
                        message=f"High memory usage: {memory.percent}%",
                        severity="warning",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check_name="memory_usage",
                        status=ValidationStatus.PASSED,
                        message=f"Memory usage normal: {memory.percent}%",
                    )
                )

            # Check disk usage
            disk = psutil.disk_usage("/")
            if disk.percent > 90:
                results.append(
                    ValidationResult(
                        check_name="disk_usage",
                        status=ValidationStatus.WARNING,
                        message=f"Low disk space: {disk.percent}% used",
                        severity="warning",
                    )
                )

        except ImportError:
            results.append(
                ValidationResult(
                    check_name="resource_monitoring",
                    status=ValidationStatus.SKIPPED,
                    message="psutil not available for resource monitoring",
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    check_name="resource_check",
                    status=ValidationStatus.FAILED,
                    message=f"Failed to check resources: {e}",
                    severity="error",
                )
            )

        return results

    def _check_logs(self, package_dir: Path) -> List[ValidationResult]:
        """Check logs for errors."""
        results = []

        log_dir = package_dir / "logs"
        if not log_dir.exists():
            log_dir = package_dir

        # Find log files
        log_files = list(log_dir.glob("*.log"))

        if not log_files:
            results.append(
                ValidationResult(
                    check_name="log_files",
                    status=ValidationStatus.WARNING,
                    message="No log files found",
                    severity="warning",
                )
            )
            return results

        # Check recent logs for errors
        error_count = 0
        warning_count = 0

        for log_file in log_files:
            try:
                with open(log_file) as f:
                    # Read last 1000 lines
                    lines = f.readlines()[-1000:]

                    for line in lines:
                        line_lower = line.lower()
                        if "error" in line_lower or "exception" in line_lower:
                            error_count += 1
                        elif "warning" in line_lower or "warn" in line_lower:
                            warning_count += 1

            except Exception:
                continue

        if error_count > 0:
            results.append(
                ValidationResult(
                    check_name="log_errors",
                    status=ValidationStatus.WARNING,
                    message=f"Found {error_count} errors in logs",
                    details={
                        "error_count": error_count,
                        "warning_count": warning_count,
                    },
                    severity="warning",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="log_errors",
                    status=ValidationStatus.PASSED,
                    message="No errors found in logs",
                )
            )

        return results

    def _run_functional_tests(self, package_dir: Path) -> List[ValidationResult]:
        """Run functional tests if available."""
        results = []

        # Look for test script
        test_scripts = ["test.sh", "run_tests.sh", "functional_tests.py"]
        test_script = None

        for script in test_scripts:
            script_path = package_dir / script
            if script_path.exists():
                test_script = script_path
                break

        if test_script:
            try:
                # Run test script
                result = subprocess.run(
                    [str(test_script)],
                    cwd=package_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout,
                )

                if result.returncode == 0:
                    results.append(
                        ValidationResult(
                            check_name="functional_tests",
                            status=ValidationStatus.PASSED,
                            message="Functional tests passed",
                            details={"stdout": result.stdout[-500:]},  # Last 500 chars
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check_name="functional_tests",
                            status=ValidationStatus.FAILED,
                            message="Functional tests failed",
                            details={
                                "returncode": result.returncode,
                                "stderr": result.stderr[-500:],
                            },
                            severity="error",
                        )
                    )

            except subprocess.TimeoutExpired:
                results.append(
                    ValidationResult(
                        check_name="functional_tests",
                        status=ValidationStatus.FAILED,
                        message="Functional tests timed out",
                        severity="error",
                    )
                )
            except Exception as e:
                results.append(
                    ValidationResult(
                        check_name="functional_tests",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to run tests: {e}",
                        severity="error",
                    )
                )
        else:
            results.append(
                ValidationResult(
                    check_name="functional_tests",
                    status=ValidationStatus.SKIPPED,
                    message="No functional tests found",
                )
            )

        return results


def validate_export(package_path: str, target_platform: Optional[str] = None) -> bool:
    """
    Validate an exported package.

    Args:
        package_path: Path to package
        target_platform: Target hardware platform

    Returns:
        True if validation passed
    """
    validator = ExportValidator()

    platform = None
    if target_platform:
        try:
            platform = HardwarePlatform(target_platform)
        except ValueError:
            logger.warning(f"Unknown platform: {target_platform}")

    results = validator.validate_package(Path(package_path), platform)

    # Print results
    print("\n=== Export Validation Results ===")

    failed = False
    for result in results:
        icon = {
            ValidationStatus.PASSED: "✓",
            ValidationStatus.WARNING: "⚠",
            ValidationStatus.FAILED: "✗",
            ValidationStatus.SKIPPED: "○",
        }.get(result.status, "?")

        print(f"{icon} {result.check_name}: {result.message}")

        if result.status == ValidationStatus.FAILED:
            failed = True

    print("\n" + "=" * 40)

    return not failed
