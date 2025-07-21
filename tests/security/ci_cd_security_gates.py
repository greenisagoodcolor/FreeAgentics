"""
CI/CD Security Gates Configuration

This module provides security gate configurations for various CI/CD platforms
to ensure security tests pass before deployment.
"""

from pathlib import Path
from typing import Dict

import yaml


class SecurityGatesConfig:
    """Security gates configuration for CI/CD pipelines"""

    @staticmethod
    def github_actions_config() -> Dict:
        """Generate GitHub Actions security workflow"""

        workflow = {
            "name": "Security Testing",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]},
            },
            "jobs": {
                "security-tests": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3",
                        },
                        {
                            "name": "Setup Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.10"},
                        },
                        {
                            "name": "Install dependencies",
                            "run": """
                                pip install --upgrade pip
                                pip install -r requirements.txt
                                pip install -r requirements-dev.txt
                            """,
                        },
                        {
                            "name": "Run security test suite",
                            "run": """
                                python -m pytest tests/security/comprehensive_security_test_suite.py -v
                            """,
                        },
                        {
                            "name": "Run penetration tests",
                            "run": """
                                python tests/security/run_comprehensive_penetration_tests.py
                            """,
                        },
                        {
                            "name": "Check security headers",
                            "run": """
                                python -m pytest tests/unit/test_security_headers_validation.py -v
                            """,
                        },
                        {
                            "name": "OWASP Dependency Check",
                            "uses": "dependency-check/Dependency-Check_Action@main",
                            "with": {
                                "project": "FreeAgentics",
                                "path": ".",
                                "format": "ALL",
                                "args": "--enableRetired --enableExperimental",
                            },
                        },
                        {
                            "name": "Run Bandit Security Scan",
                            "run": """
                                pip install bandit
                                bandit -r . -f json -o bandit-report.json
                            """,
                        },
                        {
                            "name": "Run Safety Check",
                            "run": """
                                pip install safety
                                safety check --json > safety-report.json
                            """,
                        },
                        {
                            "name": "OWASP ZAP Baseline Scan",
                            "uses": "zaproxy/action-baseline@v0.7.0",
                            "with": {
                                "target": "http://localhost:8000",
                                "allow_issue_writing": False,
                                "fail_action": True,
                            },
                        },
                        {
                            "name": "Upload security reports",
                            "uses": "actions/upload-artifact@v3",
                            "if": "always()",
                            "with": {
                                "name": "security-reports",
                                "path": """
                                    security_test_report.json
                                    bandit-report.json
                                    safety-report.json
                                    zap_report.html
                                """,
                            },
                        },
                        {
                            "name": "Security gate check",
                            "run": """
                                python tests/security/check_security_gates.py
                            """,
                        },
                    ],
                    "services": {
                        "postgres": {
                            "image": "postgres:14",
                            "env": {
                                "POSTGRES_PASSWORD": "postgres",
                                "POSTGRES_DB": "freeagentics_test",
                            },
                            "options": "--health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5",
                        },
                        "redis": {
                            "image": "redis:7",
                            "options": '--health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5',
                        },
                    },
                }
            },
        }

        return workflow

    @staticmethod
    def gitlab_ci_config() -> Dict:
        """Generate GitLab CI security configuration"""

        config = {
            "stages": ["test", "security", "deploy"],
            "variables": {"PIP_CACHE_DIR": "$CI_PROJECT_DIR/.cache/pip"},
            "cache": {"paths": [".cache/pip"]},
            "security-tests": {
                "stage": "security",
                "image": "python:3.10",
                "services": ["postgres:14", "redis:latest"],
                "variables": {
                    "POSTGRES_DB": "freeagentics_test",
                    "POSTGRES_USER": "postgres",
                    "POSTGRES_PASSWORD": "postgres",
                    "DATABASE_URL": "postgresql://postgres:postgres@postgres/freeagentics_test",
                },
                "before_script": [
                    "pip install --upgrade pip",
                    "pip install -r requirements.txt",
                    "pip install -r requirements-dev.txt",
                ],
                "script": [
                    "python -m pytest tests/security/ -v --junitxml=security-report.xml",
                    "python tests/security/run_comprehensive_penetration_tests.py",
                    "bandit -r . -f json -o bandit-report.json",
                    "safety check --json > safety-report.json",
                    "python tests/security/check_security_gates.py",
                ],
                "artifacts": {
                    "reports": {"junit": "security-report.xml"},
                    "paths": [
                        "security_test_report.json",
                        "bandit-report.json",
                        "safety-report.json",
                    ],
                    "when": "always",
                },
                "allow_failure": False,
            },
            "dependency-scanning": {
                "stage": "security",
                "image": {
                    "name": "owasp/dependency-check:latest",
                    "entrypoint": [""],
                },
                "script": [
                    "/usr/share/dependency-check/bin/dependency-check.sh --scan . --format ALL --project FreeAgentics"
                ],
                "artifacts": {"paths": ["dependency-check-report.*"]},
            },
            "container-scanning": {
                "stage": "security",
                "image": "registry.gitlab.com/gitlab-org/security-products/analyzers/container-scanning:latest",
                "variables": {
                    "CI_APPLICATION_REPOSITORY": "$CI_REGISTRY_IMAGE",
                    "CI_APPLICATION_TAG": "$CI_COMMIT_SHA",
                },
                "script": ["/analyzer run"],
                "artifacts": {
                    "reports": {
                        "container_scanning": "gl-container-scanning-report.json"
                    }
                },
            },
            "sast": {
                "stage": "security",
                "image": "registry.gitlab.com/gitlab-org/security-products/analyzers/semgrep:latest",
                "script": ["/analyzer run"],
                "artifacts": {"reports": {"sast": "gl-sast-report.json"}},
            },
        }

        return config

    @staticmethod
    def jenkins_pipeline() -> str:
        """Generate Jenkins security pipeline"""

        pipeline = """
pipeline {
    agent any

    environment {
        PYTHON_VERSION = '3.10'
        DATABASE_URL = 'postgresql://postgres:postgres@localhost/freeagentics_test'
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    python${PYTHON_VERSION} -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install -r requirements-dev.txt
                '''
            }
        }

        stage('Security Tests') {
            parallel {
                stage('Unit Security Tests') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            python -m pytest tests/security/ -v --junitxml=results/security-tests.xml
                        '''
                    }
                }

                stage('Penetration Tests') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            python tests/security/run_comprehensive_penetration_tests.py
                        '''
                    }
                }

                stage('SAST Scanning') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            bandit -r . -f json -o results/bandit-report.json
                            pylint **/*.py --output-format=json > results/pylint-report.json
                        '''
                    }
                }

                stage('Dependency Scanning') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            safety check --json > results/safety-report.json
                            pip-audit --format json > results/pip-audit-report.json
                        '''
                    }
                }
            }
        }

        stage('OWASP ZAP Scan') {
            steps {
                script {
                    docker.image('owasp/zap2docker-stable').inside {
                        sh '''
                            zap-baseline.py -t http://localhost:8000 -r zap-report.html
                        '''
                    }
                }
            }
        }

        stage('Security Gate') {
            steps {
                sh '''
                    . venv/bin/activate
                    python tests/security/check_security_gates.py
                '''
            }
        }
    }

    post {
        always {
            junit 'results/*-tests.xml'
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'results',
                reportFiles: '*.html',
                reportName: 'Security Reports'
            ])
            archiveArtifacts artifacts: 'results/*.json', fingerprint: true
        }

        failure {
            emailext (
                subject: "Security Gate Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Security vulnerabilities detected. Check the reports for details.",
                to: "${env.SECURITY_TEAM_EMAIL}"
            )
        }
    }
}
"""
        return pipeline

    @staticmethod
    def circleci_config() -> Dict:
        """Generate CircleCI security configuration"""

        config = {
            "version": 2.1,
            "orbs": {
                "python": "circleci/python@2.1.1",
                "security": "salto/security@0.2.0",
            },
            "jobs": {
                "security-tests": {
                    "docker": [
                        {"image": "cimg/python:3.10"},
                        {
                            "image": "cimg/postgres:14.0",
                            "environment": {
                                "POSTGRES_USER": "postgres",
                                "POSTGRES_PASSWORD": "postgres",
                                "POSTGRES_DB": "freeagentics_test",
                            },
                        },
                        {"image": "cimg/redis:7.0"},
                    ],
                    "steps": [
                        "checkout",
                        {
                            "python/install-packages": {
                                "pkg-manager": "pip",
                                "pip-dependency-file": "requirements.txt",
                            }
                        },
                        {
                            "run": {
                                "name": "Install dev dependencies",
                                "command": "pip install -r requirements-dev.txt",
                            }
                        },
                        {
                            "run": {
                                "name": "Run security test suite",
                                "command": "python -m pytest tests/security/ -v",
                            }
                        },
                        {
                            "run": {
                                "name": "Run penetration tests",
                                "command": "python tests/security/run_comprehensive_penetration_tests.py",
                            }
                        },
                        {
                            "security/scan": {
                                "scan-type": "sast",
                                "fail-on-issues": True,
                            }
                        },
                        {
                            "run": {
                                "name": "Dependency check",
                                "command": """
                                    safety check --json > safety-report.json
                                    pip-audit --format json > pip-audit-report.json
                                """,
                            }
                        },
                        {
                            "run": {
                                "name": "Security gate check",
                                "command": "python tests/security/check_security_gates.py",
                            }
                        },
                        {"store_artifacts": {"path": "security_test_report.json"}},
                        {"store_test_results": {"path": "test-results"}},
                    ],
                }
            },
            "workflows": {"security": {"jobs": ["security-tests"]}},
        }

        return config

    @staticmethod
    def azure_devops_pipeline() -> Dict:
        """Generate Azure DevOps security pipeline"""

        pipeline = {
            "trigger": ["main", "develop"],
            "pool": {"vmImage": "ubuntu-latest"},
            "variables": {"pythonVersion": "3.10"},
            "stages": [
                {
                    "stage": "SecurityTests",
                    "displayName": "Security Testing",
                    "jobs": [
                        {
                            "job": "SecurityScan",
                            "displayName": "Run Security Scans",
                            "steps": [
                                {
                                    "task": "UsePythonVersion@0",
                                    "inputs": {
                                        "versionSpec": "$(pythonVersion)",
                                        "addToPath": True,
                                    },
                                },
                                {
                                    "script": """
                                        pip install --upgrade pip
                                        pip install -r requirements.txt
                                        pip install -r requirements-dev.txt
                                    """,
                                    "displayName": "Install dependencies",
                                },
                                {
                                    "script": "python -m pytest tests/security/ -v --junitxml=junit/security-tests.xml",
                                    "displayName": "Run security tests",
                                },
                                {
                                    "script": "python tests/security/run_comprehensive_penetration_tests.py",
                                    "displayName": "Run penetration tests",
                                },
                                {
                                    "task": "WhiteSource@21",
                                    "displayName": "WhiteSource security scan",
                                },
                                {
                                    "script": """
                                        bandit -r . -f json -o $(Build.ArtifactStagingDirectory)/bandit-report.json
                                        safety check --json > $(Build.ArtifactStagingDirectory)/safety-report.json
                                    """,
                                    "displayName": "SAST and dependency scanning",
                                },
                                {
                                    "script": "python tests/security/check_security_gates.py",
                                    "displayName": "Security gate validation",
                                },
                                {
                                    "task": "PublishTestResults@2",
                                    "inputs": {
                                        "testResultsFormat": "JUnit",
                                        "testResultsFiles": "**/security-tests.xml",
                                    },
                                },
                                {
                                    "task": "PublishBuildArtifacts@1",
                                    "inputs": {
                                        "pathToPublish": "$(Build.ArtifactStagingDirectory)",
                                        "artifactName": "SecurityReports",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }

        return pipeline

    @staticmethod
    def generate_all_configs(output_dir: str = "."):
        """Generate all CI/CD security configurations"""

        output_path = Path(output_dir)

        # GitHub Actions
        github_dir = output_path / ".github" / "workflows"
        github_dir.mkdir(parents=True, exist_ok=True)

        with open(github_dir / "security-tests.yml", "w") as f:
            yaml.dump(
                SecurityGatesConfig.github_actions_config(),
                f,
                default_flow_style=False,
            )

        # GitLab CI
        with open(output_path / ".gitlab-ci-security.yml", "w") as f:
            yaml.dump(
                SecurityGatesConfig.gitlab_ci_config(),
                f,
                default_flow_style=False,
            )

        # Jenkins
        with open(output_path / "Jenkinsfile.security", "w") as f:
            f.write(SecurityGatesConfig.jenkins_pipeline())

        # CircleCI
        circleci_dir = output_path / ".circleci"
        circleci_dir.mkdir(exist_ok=True)

        with open(circleci_dir / "config-security.yml", "w") as f:
            yaml.dump(
                SecurityGatesConfig.circleci_config(),
                f,
                default_flow_style=False,
            )

        # Azure DevOps
        with open(output_path / "azure-pipelines-security.yml", "w") as f:
            yaml.dump(
                SecurityGatesConfig.azure_devops_pipeline(),
                f,
                default_flow_style=False,
            )

        print("Generated CI/CD security configurations:")
        print("- .github/workflows/security-tests.yml")
        print("- .gitlab-ci-security.yml")
        print("- Jenkinsfile.security")
        print("- .circleci/config-security.yml")
        print("- azure-pipelines-security.yml")


if __name__ == "__main__":
    # Generate all configurations
    SecurityGatesConfig.generate_all_configs()
