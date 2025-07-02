#!/usr/bin/env python3
"""
Test Runner Setup and Configuration Module for FreeAgentics Repository

This module implements comprehensive test runner configuration following
expert committee guidance from Kent Beck (TDD) and Gary Bernhardt (Testing Strategies).

Expert Committee Guidance:
- Kent Beck: "Tests should run fast and provide immediate feedback"
- Gary Bernhardt: "Test execution strategy should match test boundaries"
- Michael Feathers: "Test runners must be reliable for safe refactoring"
- Rich Harris: "Developer tools should be fast and intuitive"

Following Clean Code and SOLID principles with optimized execution strategies.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from test_categorization import TestCategorizationResult
from test_discovery import TestDiscoveryResult, TestFramework


class RunnerType(Enum):
    """Supported test runner types"""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    CYPRESS = "cypress"
    PLAYWRIGHT = "playwright"
    VITEST = "vitest"
    CUSTOM = "custom"


class ExecutionMode(Enum):
    """Test execution modes for different scenarios"""

    FAST = "fast"  # Quick feedback loop
    COMPLETE = "complete"  # Full test suite
    CRITICAL = "critical"  # Critical path only
    PARALLEL = "parallel"  # Parallel execution
    SEQUENTIAL = "sequential"  # Sequential execution
    SMOKE = "smoke"  # Smoke tests only


@dataclass
class RunnerConfig:
    """
    Configuration for a specific test runner.

    Following expert committee guidance for optimal test execution
    with framework-specific optimizations.
    """

    runner_type: RunnerType
    command: str
    config_file: Optional[str] = None

    # Execution settings
    parallel: bool = True
    max_workers: int = 4
    timeout: int = 300  # seconds

    # Coverage settings
    coverage_enabled: bool = True
    coverage_threshold: float = 80.0
    coverage_report_format: str = "xml"

    # Output settings
    output_format: str = "junit"
    output_file: Optional[str] = None
    verbose: bool = False

    # Environment settings
    environment_vars: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None

    # Framework-specific settings
    extra_args: List[str] = field(default_factory=list)
    plugins: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "runner_type": self.runner_type.value,
            "command": self.command,
            "config_file": self.config_file,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "timeout": self.timeout,
            "coverage_enabled": self.coverage_enabled,
            "coverage_threshold": self.coverage_threshold,
            "coverage_report_format": self.coverage_report_format,
            "output_format": self.output_format,
            "output_file": self.output_file,
            "verbose": self.verbose,
            "environment_vars": self.environment_vars,
            "working_directory": self.working_directory,
            "extra_args": self.extra_args,
            "plugins": self.plugins,
        }


@dataclass
class ExecutionPlan:
    """
    Test execution plan with optimized scheduling.

    Implements expert committee guidance for efficient test execution
    with proper parallelization and dependency management.
    """

    name: str
    mode: ExecutionMode

    # Test selection
    test_patterns: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    excluded_patterns: List[str] = field(default_factory=list)

    # Execution configuration
    runner_configs: List[RunnerConfig] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)

    # Parallel execution
    parallel_groups: Dict[str, List[str]] = field(default_factory=dict)
    max_parallel_runners: int = 4

    # Quality gates
    fail_fast: bool = True
    required_coverage: float = 80.0
    max_failures: int = 0

    # Reporting
    generate_reports: bool = True
    report_formats: List[str] = field(default_factory=lambda: ["junit", "html"])

    # Estimated metrics
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0  # CI/CD cost estimate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "mode": self.mode.value,
            "test_patterns": self.test_patterns,
            "test_files": self.test_files,
            "excluded_patterns": self.excluded_patterns,
            "runner_configs": [rc.to_dict() for rc in self.runner_configs],
            "execution_order": self.execution_order,
            "parallel_groups": self.parallel_groups,
            "max_parallel_runners": self.max_parallel_runners,
            "fail_fast": self.fail_fast,
            "required_coverage": self.required_coverage,
            "max_failures": self.max_failures,
            "generate_reports": self.generate_reports,
            "report_formats": self.report_formats,
            "estimated_duration": self.estimated_duration,
            "estimated_cost": self.estimated_cost,
        }


@dataclass
class TestRunnerSetupResult:
    """
    Complete test runner setup results.

    Provides comprehensive configuration for CI/CD integration
    following expert committee guidance.
    """

    runner_configs: List[RunnerConfig] = field(default_factory=list)
    execution_plans: List[ExecutionPlan] = field(default_factory=list)

    # Configuration files generated
    config_files_created: List[str] = field(default_factory=list)

    # Validation results
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

    # Performance metrics
    estimated_total_duration: float = 0.0
    parallel_efficiency: float = 0.0

    # Setup metadata
    setup_duration: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "runner_configs": [rc.to_dict() for rc in self.runner_configs],
            "execution_plans": [ep.to_dict() for ep in self.execution_plans],
            "config_files_created": self.config_files_created,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "estimated_total_duration": self.estimated_total_duration,
            "parallel_efficiency": self.parallel_efficiency,
            "setup_duration": self.setup_duration,
            "recommendations": self.recommendations,
        }


class TestRunnerSetup:
    """
    Comprehensive test runner setup engine following expert committee guidance.

    Implements intelligent runner configuration using:
    1. Framework detection and optimization
    2. Parallel execution planning
    3. Coverage configuration
    4. CI/CD integration setup
    5. Performance optimization

    Following SOLID principles:
    - Single Responsibility: Only sets up test runners
    - Open/Closed: Extensible for new runners
    - Liskov Substitution: Runner configurations are interchangeable
    - Interface Segregation: Clear runner interfaces
    - Dependency Inversion: Depends on abstractions
    """

    # Default runner configurations
    DEFAULT_CONFIGS = {
        RunnerType.PYTEST: {
            "command": "python -m pytest",
            "config_file": "pytest.ini",
            "plugins": ["pytest-cov", "pytest-xdist", "pytest-html"],
            "extra_args": ["--tb=short", "--strict-markers"],
        },
        RunnerType.JEST: {
            "command": "npx jest",
            "config_file": "jest.config.js",
            "plugins": [],
            "extra_args": ["--passWithNoTests"],
        },
        RunnerType.UNITTEST: {
            "command": "python -m unittest",
            "config_file": None,
            "plugins": [],
            "extra_args": ["discover", "-v"],
        },
    }

    # Framework-specific optimizations
    FRAMEWORK_OPTIMIZATIONS = {
        TestFramework.PYTEST: {
            "parallel_args": ["-n", "auto"],
            "coverage_args": ["--cov=.", "--cov-report=xml"],
            "fast_args": ["--ff", "-x"],
            "output_args": ["--junit-xml=test-results.xml"],
        },
        TestFramework.JEST: {
            "parallel_args": ["--maxWorkers=50%"],
            "coverage_args": ["--coverage", "--coverageReporters=lcov"],
            "fast_args": ["--bail", "--onlyChanged"],
            "output_args": ["--outputFile=test-results.json"],
        },
    }

    def __init__(
        self,
        project_root: Union[str, Path],
        output_dir: Optional[str] = None,
        ci_mode: bool = False,
    ) -> None:
        """
        Initialize test runner setup engine.

        Args:
            project_root: Root directory of the project
            output_dir: Directory for generated configuration files
            ci_mode: Whether to optimize for CI/CD environment
        """
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(
            output_dir) if output_dir else self.project_root / ".test_configs"
        self.ci_mode = ci_mode
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # Setup statistics
        self._setup_stats = {"configs_created": 0, "plans_generated": 0, "errors": []}

    def setup_test_runners(
        self,
        discovery_result: TestDiscoveryResult,
        categorization_result: TestCategorizationResult,
    ) -> TestRunnerSetupResult:
        """
        Setup test runners based on discovery and categorization results.

        Args:
            discovery_result: Results from test discovery
            categorization_result: Results from test categorization

        Returns:
            TestRunnerSetupResult: Comprehensive setup results
        """
        import time

        start_time = time.time()

        self.logger.info("Starting test runner setup")

        result = TestRunnerSetupResult()

        try:
            # Step 1: Analyze frameworks and create runner configurations
            result.runner_configs = self._create_runner_configs(discovery_result)

            # Step 2: Generate execution plans
            result.execution_plans = self._generate_execution_plans(
                discovery_result, categorization_result
            )

            # Step 3: Create configuration files
            result.config_files_created = self._create_config_files(
                result.runner_configs)

            # Step 4: Validate configurations
            result.validation_passed, result.validation_errors = self._validate_setup(
                result)

            # Step 5: Calculate performance metrics
            result.estimated_total_duration = self._calculate_total_duration(result)
            result.parallel_efficiency = self._calculate_parallel_efficiency(result)

            # Step 6: Generate recommendations
            result.recommendations = self._generate_recommendations(result)

            # Step 7: Final metrics
            result.setup_duration = time.time() - start_time

            self.logger.info(
                f"Test runner setup complete: {len(result.runner_configs)} runners configured"
            )

        except Exception as e:
            self.logger.error(f"Test runner setup failed: {e}")
            result.validation_passed = False
            result.validation_errors.append(str(e))

        return result

    def _create_runner_configs(
            self,
            discovery_result: TestDiscoveryResult) -> List[RunnerConfig]:
        """Create runner configurations based on discovered frameworks"""
        configs = []

        # Analyze framework usage
        framework_usage = discovery_result.by_framework

        for framework, count in framework_usage.items():
            if count == 0:
                continue

            runner_type = self._map_framework_to_runner(framework)
            if runner_type == RunnerType.CUSTOM:
                continue

            config = self._create_framework_config(runner_type, count)
            configs.append(config)
            self._setup_stats["configs_created"] += 1

        return configs

    def _map_framework_to_runner(self, framework: str) -> RunnerType:
        """Map framework name to runner type"""
        mapping = {
            "pytest": RunnerType.PYTEST,
            "unittest": RunnerType.UNITTEST,
            "jest": RunnerType.JEST,
            "mocha": RunnerType.MOCHA,
            "cypress": RunnerType.CYPRESS,
            "playwright": RunnerType.PLAYWRIGHT,
            "vitest": RunnerType.VITEST,
        }

        return mapping.get(framework.lower(), RunnerType.CUSTOM)

    def _create_framework_config(
            self,
            runner_type: RunnerType,
            test_count: int) -> RunnerConfig:
        """Create optimized configuration for a specific framework"""
        defaults = self.DEFAULT_CONFIGS.get(runner_type, {})

        config = RunnerConfig(
            runner_type=runner_type,
            command=defaults.get("command", f"{runner_type.value}"),
            config_file=defaults.get("config_file"),
            plugins=defaults.get("plugins", []),
            extra_args=defaults.get("extra_args", []),
        )

        # Optimize based on test count and CI mode
        if test_count > 10:
            config.parallel = True
            config.max_workers = min(8, max(2, test_count // 10))

        if self.ci_mode:
            config.timeout = 600  # 10 minutes in CI
            config.verbose = True
            config.coverage_enabled = True
            config.output_format = "junit"
            config.output_file = f"test-results-{runner_type.value}.xml"

        # Framework-specific optimizations
        if runner_type == RunnerType.PYTEST:
            if config.parallel:
                config.extra_args.extend(["-n", str(config.max_workers)])
            if config.coverage_enabled:
                config.extra_args.extend(["--cov=.", "--cov-report=xml"])

        elif runner_type == RunnerType.JEST:
            if config.parallel:
                config.extra_args.append(f"--maxWorkers={config.max_workers}")
            if config.coverage_enabled:
                config.extra_args.extend(["--coverage", "--coverageReporters=lcov"])

        return config

    def _generate_execution_plans(
        self,
        discovery_result: TestDiscoveryResult,
        categorization_result: TestCategorizationResult,
    ) -> List[ExecutionPlan]:
        """Generate optimized execution plans for different scenarios"""
        plans = []

        # Plan 1: Fast feedback (smoke + unit tests)
        fast_plan = ExecutionPlan(
            name="fast",
            mode=ExecutionMode.FAST,
            fail_fast=True,
            max_failures=3,
            required_coverage=60.0,
        )

        # Add smoke and fast tests
        for categorized in categorization_result.categorized_tests:
            if (
                categorized.test_file.test_type.value == "smoke"
                or categorized.execution_tier.value == "fast"
            ):
                fast_plan.test_files.append(str(categorized.test_file.path))

        fast_plan.estimated_duration = sum(
            cat.estimated_duration
            for cat in categorization_result.categorized_tests
            if str(cat.test_file.path) in fast_plan.test_files
        )

        plans.append(fast_plan)

        # Plan 2: Complete test suite
        complete_plan = ExecutionPlan(
            name="complete",
            mode=ExecutionMode.COMPLETE,
            fail_fast=False,
            required_coverage=80.0,
            parallel_groups=categorization_result.parallel_execution_groups,
        )

        complete_plan.test_files = [
            str(cat.test_file.path) for cat in categorization_result.categorized_tests
        ]
        complete_plan.estimated_duration = categorization_result.estimated_total_duration

        plans.append(complete_plan)

        # Plan 3: Critical path only
        critical_plan = ExecutionPlan(
            name="critical",
            mode=ExecutionMode.CRITICAL,
            fail_fast=True,
            max_failures=0,
            required_coverage=90.0,
        )

        critical_plan.test_files = categorization_result.critical_path_tests
        critical_plan.estimated_duration = sum(
            cat.estimated_duration
            for cat in categorization_result.categorized_tests
            if str(cat.test_file.path) in critical_plan.test_files
        )

        plans.append(critical_plan)

        # Plan 4: Parallel execution (for CI/CD)
        if categorization_result.parallel_execution_groups:
            parallel_plan = ExecutionPlan(
                name="parallel",
                mode=ExecutionMode.PARALLEL,
                parallel_groups=categorization_result.parallel_execution_groups,
                max_parallel_runners=len(
                    categorization_result.parallel_execution_groups),
                fail_fast=False,
                required_coverage=80.0,
            )

            parallel_plan.test_files = complete_plan.test_files
            # Estimate parallel duration (assume 70% efficiency)
            parallel_plan.estimated_duration = complete_plan.estimated_duration * 0.3

            plans.append(parallel_plan)

        self._setup_stats["plans_generated"] = len(plans)
        return plans

    def _create_config_files(self, runner_configs: List[RunnerConfig]) -> List[str]:
        """Create configuration files for test runners"""
        created_files = []

        for config in runner_configs:
            if config.config_file:
                config_path = self._create_runner_config_file(config)
                if config_path:
                    created_files.append(config_path)

        # Create master test configuration
        master_config_path = self._create_master_config(runner_configs)
        if master_config_path:
            created_files.append(master_config_path)

        return created_files

    def _create_runner_config_file(self, config: RunnerConfig) -> Optional[str]:
        """Create configuration file for a specific runner"""
        try:
            if config.runner_type == RunnerType.PYTEST:
                return self._create_pytest_config(config)
            elif config.runner_type == RunnerType.JEST:
                return self._create_jest_config(config)
            elif config.runner_type == RunnerType.UNITTEST:
                # unittest doesn't need a config file
                return None

        except Exception as e:
            error_msg = f"Error creating config for {config.runner_type.value}: {e}"
            self.logger.warning(error_msg)
            self._setup_stats["errors"].append(error_msg)

        return None

    def _create_pytest_config(self, config: RunnerConfig) -> str:
        """Create pytest.ini configuration file"""
        pytest_config = f"""[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = {' '.join(config.extra_args)}
markers =
    smoke: Quick sanity check tests
    slow: Tests that take more than 10 seconds
    integration: Integration tests
    unit: Unit tests
    critical: Critical path tests
    flaky: Tests that occasionally fail
timeout = {config.timeout}
"""

        if config.parallel:
            pytest_config += f"addopts = -n {config.max_workers}\n"

        config_path = self.output_dir / "pytest.ini"
        config_path.write_text(pytest_config)

        return str(config_path)

    def _create_jest_config(self, config: RunnerConfig) -> str:
        """Create jest.config.js configuration file"""
        jest_config = f"""module.exports = {{
  testEnvironment: 'node',
  testMatch: [
    '**/__tests__/**/*.(js|jsx|ts|tsx)',
    '**/*.(test|spec).(js|jsx|ts|tsx)'
  ],
  collectCoverageFrom: [
    '**/*.(js|jsx|ts|tsx)',
    '!**/*.d.ts',
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  testTimeout: {config.timeout * 1000},
  maxWorkers: {config.max_workers if config.parallel else 1},
  verbose: {str(config.verbose).lower()},
  reporters: [
    'default',
    ['jest-junit', {{
      outputDirectory: 'test-results',
      outputName: 'jest-results.xml',
    }}]
  ]
}};
"""

        config_path = self.output_dir / "jest.config.js"
        config_path.write_text(jest_config)

        return str(config_path)

    def _create_master_config(self, runner_configs: List[RunnerConfig]) -> str:
        """Create master test configuration file"""
        master_config = {
            "test_runners": [config.to_dict() for config in runner_configs],
            "project_root": str(self.project_root),
            "output_directory": str(self.output_dir),
            "ci_mode": self.ci_mode,
            "created_at": "2025-06-20T08:30:00Z",
        }

        config_path = self.output_dir / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(master_config, f, indent=2)

        return str(config_path)

    def _validate_setup(self, result: TestRunnerSetupResult) -> tuple[bool, list[str]]:
        """Validate the test runner setup"""
        errors = []

        # Check if we have at least one runner
        if not result.runner_configs:
            errors.append("No test runners configured")

        # Check if we have execution plans
        if not result.execution_plans:
            errors.append("No execution plans generated")

        # Validate each runner configuration
        for config in result.runner_configs:
            if not shutil.which(config.command.split()[0]):
                errors.append(f"Command not found: {config.command}")

        # Check for required files
        for plan in result.execution_plans:
            for test_file in plan.test_files:
                if not Path(test_file).exists():
                    errors.append(f"Test file not found: {test_file}")

        return len(errors) == 0, errors

    def _calculate_total_duration(self, result: TestRunnerSetupResult) -> float:
        """Calculate estimated total execution duration"""
        if not result.execution_plans:
            return 0.0

        # Use the complete plan duration
        complete_plan = next(
            (plan for plan in result.execution_plans if plan.mode == ExecutionMode.COMPLETE),
            result.execution_plans[0],
        )

        return complete_plan.estimated_duration

    def _calculate_parallel_efficiency(self, result: TestRunnerSetupResult) -> float:
        """Calculate parallel execution efficiency"""
        complete_plan = next(
            (plan for plan in result.execution_plans if plan.mode == ExecutionMode.COMPLETE),
            None,
        )
        parallel_plan = next(
            (plan for plan in result.execution_plans if plan.mode == ExecutionMode.PARALLEL),
            None,
        )

        if not complete_plan or not parallel_plan:
            return 0.0

        if complete_plan.estimated_duration == 0:
            return 0.0

        # Efficiency = (sequential_time - parallel_time) / sequential_time
        efficiency = (
            complete_plan.estimated_duration - parallel_plan.estimated_duration
        ) / complete_plan.estimated_duration
        return max(0.0, min(1.0, efficiency))

    def _generate_recommendations(self, result: TestRunnerSetupResult) -> List[str]:
        """Generate actionable recommendations for test runner optimization"""
        recommendations = []

        # Runner configuration recommendations
        if len(result.runner_configs) > 3:
            recommendations.append(
                "🔧 Multiple test runners detected - consider standardizing on fewer frameworks"
            )

        # Performance recommendations
        if result.estimated_total_duration > 600:  # 10 minutes
            recommendations.append(
                "⏱️ Long execution time - implement parallel execution strategies"
            )

        if result.parallel_efficiency < 0.5:
            recommendations.append(
                "⚡ Low parallel efficiency - optimize test parallelization")

        # Validation recommendations
        if not result.validation_passed:
            recommendations.append(
                "❌ Configuration validation failed - fix setup errors before proceeding"
            )

        # CI/CD recommendations
        if self.ci_mode and result.estimated_total_duration > 300:  # 5 minutes
            recommendations.append(
                "🚀 CI/CD optimization needed - consider test splitting and caching"
            )

        # Coverage recommendations
        for plan in result.execution_plans:
            if plan.required_coverage > 90:
                recommendations.append(
                    "📊 High coverage requirements may slow development - balance quality vs speed"
                )

        return recommendations

    def generate_ci_config(self, result: TestRunnerSetupResult,
                           ci_platform: str = "github") -> str:
        """Generate CI/CD configuration file"""
        if ci_platform.lower() == "github":
            return self._generate_github_actions_config(result)
        elif ci_platform.lower() == "gitlab":
            return self._generate_gitlab_ci_config(result)
        else:
            raise ValueError(f"Unsupported CI platform: {ci_platform}")

    def _generate_github_actions_config(self, result: TestRunnerSetupResult) -> str:
        """Generate GitHub Actions workflow configuration"""
        config = """name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        node-version: [18, 20]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Set up Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}

    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist

    - name: Install Node.js dependencies
      run: |
        npm ci

    - name: Run Python tests
      run: |
        pytest --cov=. --cov-report=xml --junit-xml=test-results-python.xml

    - name: Run JavaScript tests
      run: |
        npm test -- --coverage --watchAll=false

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: test-results-*.xml
"""

        config_path = self.output_dir / "github-actions.yml"
        config_path.write_text(config)

        return str(config_path)


def create_test_runner_setup(project_root: str, **kwargs: Any) -> TestRunnerSetup:
    """
    Factory function to create a TestRunnerSetup instance.

    Args:
        project_root: Root directory of the project
        **kwargs: Additional configuration options

    Returns:
        TestRunnerSetup: Configured test runner setup instance
    """
    return TestRunnerSetup(project_root, **kwargs)


def main():
    """Main function for command-line usage"""
    import argparse

    from test_categorization import create_test_categorizer
    from test_discovery import create_test_discovery

    parser = argparse.ArgumentParser(description="Setup and configure test runners")
    parser.add_argument("project_root", help="Project root directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for configurations")
    parser.add_argument("--ci-mode", action="store_true", help="Optimize for CI/CD")
    parser.add_argument(
        "--ci-platform",
        default="github",
        help="CI platform (github, gitlab)")
    parser.add_argument("--validate", action="store_true", help="Validate setup only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Step 1: Discover tests
    print("Step 1: Discovering tests...")
    discovery = create_test_discovery(args.project_root)
    discovery_result = discovery.discover_tests()
    print(f"  Found {len(discovery_result.test_files)} test files")

    # Step 2: Categorize tests
    print("Step 2: Categorizing tests...")
    categorizer = create_test_categorizer(args.project_root)
    categorization_result = categorizer.categorize_tests(discovery_result)
    print(f"  Categorized {categorization_result.total_tests} tests")

    # Step 3: Setup test runners
    print("Step 3: Setting up test runners...")
    setup = create_test_runner_setup(
        args.project_root, output_dir=args.output_dir, ci_mode=args.ci_mode
    )
    result = setup.setup_test_runners(discovery_result, categorization_result)

    # Step 4: Generate CI configuration
    if not args.validate:
        print("Step 4: Generating CI configuration...")
        ci_config_path = setup.generate_ci_config(result, args.ci_platform)
        print(f"  CI configuration saved to {ci_config_path}")

    # Summary
    print("\n📊 Test Runner Setup Summary:")
    print(f"  🔧 Runners configured: {len(result.runner_configs)}")
    print(f"  📋 Execution plans: {len(result.execution_plans)}")
    print(f"  📁 Config files created: {len(result.config_files_created)}")
    print(f"  ✅ Validation passed: {result.validation_passed}")
    print(f"  ⏱️ Estimated duration: {result.estimated_total_duration:.1f}s")
    print(f"  ⚡ Parallel efficiency: {result.parallel_efficiency:.1%}")

    if result.validation_errors:
        print("\n❌ Validation Errors:")
        for error in result.validation_errors:
            print(f"  {error}")

    if result.recommendations:
        print("\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")

    print(f"\n✅ Test runner setup completed in {result.setup_duration:.2f}s")


if __name__ == "__main__":
    main()
