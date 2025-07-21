"""
RBAC Test Configuration.

This module provides configuration settings for RBAC authorization matrix tests.
It defines test scenarios, user roles, permissions, and security test parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from auth.security_implementation import Permission, UserRole


@dataclass
class TestUser:
    """Test user configuration."""

    username: str
    email: str
    password: str
    role: UserRole
    expected_permissions: Set[Permission] = field(default_factory=set)
    should_have_access: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TestEndpoint:
    """Test endpoint configuration."""

    path: str
    method: str
    required_permission: Permission
    description: str
    test_data: Optional[Dict] = None


@dataclass
class SecurityTestScenario:
    """Security test scenario configuration."""

    name: str
    description: str
    test_type: str
    severity: str
    expected_outcome: str
    test_data: Dict = field(default_factory=dict)


class RBACTestConfig:
    """Configuration for RBAC authorization matrix tests."""

    # Test user configurations
    TEST_USERS = {
        "admin": TestUser(
            username="admin_test_user",
            email="admin@rbac.test",
            password="AdminPass123!",
            role=UserRole.ADMIN,
            expected_permissions={
                Permission.CREATE_AGENT,
                Permission.DELETE_AGENT,
                Permission.VIEW_AGENTS,
                Permission.MODIFY_AGENT,
                Permission.CREATE_COALITION,
                Permission.VIEW_METRICS,
                Permission.ADMIN_SYSTEM,
            },
            should_have_access={
                "create_agent": True,
                "delete_agent": True,
                "view_agents": True,
                "modify_agent": True,
                "view_metrics": True,
                "admin_system": True,
            },
        ),
        "researcher": TestUser(
            username="researcher_test_user",
            email="researcher@rbac.test",
            password="ResearchPass123!",
            role=UserRole.RESEARCHER,
            expected_permissions={
                Permission.CREATE_AGENT,
                Permission.VIEW_AGENTS,
                Permission.MODIFY_AGENT,
                Permission.CREATE_COALITION,
                Permission.VIEW_METRICS,
            },
            should_have_access={
                "create_agent": True,
                "delete_agent": False,
                "view_agents": True,
                "modify_agent": True,
                "view_metrics": True,
                "admin_system": False,
            },
        ),
        "agent_manager": TestUser(
            username="manager_test_user",
            email="manager@rbac.test",
            password="ManagerPass123!",
            role=UserRole.AGENT_MANAGER,
            expected_permissions={
                Permission.CREATE_AGENT,
                Permission.VIEW_AGENTS,
                Permission.MODIFY_AGENT,
                Permission.VIEW_METRICS,
            },
            should_have_access={
                "create_agent": True,
                "delete_agent": False,
                "view_agents": True,
                "modify_agent": True,
                "view_metrics": True,
                "admin_system": False,
            },
        ),
        "observer": TestUser(
            username="observer_test_user",
            email="observer@rbac.test",
            password="ObserverPass123!",
            role=UserRole.OBSERVER,
            expected_permissions={
                Permission.VIEW_AGENTS,
                Permission.VIEW_METRICS,
            },
            should_have_access={
                "create_agent": False,
                "delete_agent": False,
                "view_agents": True,
                "modify_agent": False,
                "view_metrics": True,
                "admin_system": False,
            },
        ),
    }

    # Test endpoint configurations
    TEST_ENDPOINTS = {
        "agents": {
            "create": TestEndpoint(
                path="/api/v1/agents",
                method="POST",
                required_permission=Permission.CREATE_AGENT,
                description="Create new agent",
                test_data={"name": "Test Agent", "template": "basic-explorer"},
            ),
            "list": TestEndpoint(
                path="/api/v1/agents",
                method="GET",
                required_permission=Permission.VIEW_AGENTS,
                description="List all agents",
            ),
            "get": TestEndpoint(
                path="/api/v1/agents/{agent_id}",
                method="GET",
                required_permission=Permission.VIEW_AGENTS,
                description="Get specific agent",
            ),
            "update": TestEndpoint(
                path="/api/v1/agents/{agent_id}/status",
                method="PATCH",
                required_permission=Permission.MODIFY_AGENT,
                description="Update agent status",
                test_data={"status": "active"},
            ),
            "delete": TestEndpoint(
                path="/api/v1/agents/{agent_id}",
                method="DELETE",
                required_permission=Permission.DELETE_AGENT,
                description="Delete agent",
            ),
            "metrics": TestEndpoint(
                path="/api/v1/agents/{agent_id}/metrics",
                method="GET",
                required_permission=Permission.VIEW_METRICS,
                description="View agent metrics",
            ),
        },
        "gmn": {
            "create_from_gmn": TestEndpoint(
                path="/api/v1/agents/from-gmn",
                method="POST",
                required_permission=Permission.CREATE_AGENT,
                description="Create agent from GMN specification",
                test_data={
                    "name": "GMN Test Agent",
                    "gmn_spec": "[nodes]\nposition: state {num_states: 25}\n[edges]\n",
                    "planning_horizon": 3,
                },
            ),
            "get_gmn": TestEndpoint(
                path="/api/v1/agents/{agent_id}/gmn",
                method="GET",
                required_permission=Permission.VIEW_AGENTS,
                description="Get agent GMN specification",
            ),
            "update_gmn": TestEndpoint(
                path="/api/v1/agents/{agent_id}/gmn",
                method="PUT",
                required_permission=Permission.MODIFY_AGENT,
                description="Update agent GMN specification",
                test_data="[nodes]\nposition: state {num_states: 25}\n[edges]\n",
            ),
            "examples": TestEndpoint(
                path="/api/v1/gmn/examples",
                method="GET",
                required_permission=Permission.VIEW_AGENTS,
                description="Get GMN examples",
            ),
        },
        "templates": {
            "list": TestEndpoint(
                path="/api/v1/templates",
                method="GET",
                required_permission=Permission.VIEW_AGENTS,
                description="List agent templates",
            ),
        },
        "system": {
            "health": TestEndpoint(
                path="/api/v1/system/health",
                method="GET",
                required_permission=Permission.ADMIN_SYSTEM,
                description="System health check",
            ),
            "metrics": TestEndpoint(
                path="/api/v1/system/metrics",
                method="GET",
                required_permission=Permission.ADMIN_SYSTEM,
                description="System metrics",
            ),
            "config": TestEndpoint(
                path="/api/v1/system/config",
                method="GET",
                required_permission=Permission.ADMIN_SYSTEM,
                description="System configuration",
            ),
        },
    }

    # Security test scenarios
    SECURITY_SCENARIOS = {
        "sql_injection": SecurityTestScenario(
            name="SQL Injection Attack",
            description="Test protection against SQL injection in authentication",
            test_type="injection",
            severity="high",
            expected_outcome="blocked",
            test_data={
                "payloads": [
                    "admin'; DROP TABLE users; --",
                    "' OR '1'='1",
                    "admin' --",
                    "admin' /*",
                    "admin' OR 1=1 --",
                ]
            },
        ),
        "xss_attack": SecurityTestScenario(
            name="XSS Attack",
            description="Test protection against XSS in user input",
            test_type="injection",
            severity="medium",
            expected_outcome="sanitized",
            test_data={
                "payloads": [
                    "<script>alert('XSS')</script>",
                    "javascript:alert('XSS')",
                    "<img src=x onerror=alert('XSS')>",
                    "<svg onload=alert('XSS')>",
                ]
            },
        ),
        "privilege_escalation": SecurityTestScenario(
            name="Vertical Privilege Escalation",
            description="Test protection against privilege escalation",
            test_type="authorization",
            severity="critical",
            expected_outcome="blocked",
            test_data={
                "test_user_role": UserRole.OBSERVER,
                "attempted_actions": [
                    "create_agent",
                    "delete_agent",
                    "admin_system",
                ],
            },
        ),
        "horizontal_escalation": SecurityTestScenario(
            name="Horizontal Privilege Escalation",
            description="Test protection against cross-user resource access",
            test_type="authorization",
            severity="high",
            expected_outcome="blocked",
            test_data={"test_scenario": "cross_user_access"},
        ),
        "token_manipulation": SecurityTestScenario(
            name="Token Manipulation",
            description="Test protection against token manipulation attacks",
            test_type="authentication",
            severity="high",
            expected_outcome="blocked",
            test_data={
                "attack_types": [
                    "token_modification",
                    "token_reuse",
                    "token_forgery",
                ]
            },
        ),
        "session_hijacking": SecurityTestScenario(
            name="Session Hijacking",
            description="Test protection against session hijacking",
            test_type="authentication",
            severity="high",
            expected_outcome="blocked",
            test_data={
                "attack_methods": [
                    "session_fixation",
                    "session_replay",
                ]
            },
        ),
        "brute_force": SecurityTestScenario(
            name="Brute Force Attack",
            description="Test protection against brute force attacks",
            test_type="authentication",
            severity="medium",
            expected_outcome="rate_limited",
            test_data={
                "attack_rate": 100,  # requests per minute
                "attack_duration": 60,  # seconds
            },
        ),
        "concurrent_access": SecurityTestScenario(
            name="Concurrent Access Attack",
            description="Test system stability under concurrent access",
            test_type="performance",
            severity="medium",
            expected_outcome="stable",
            test_data={
                "concurrent_users": 50,
                "requests_per_user": 10,
            },
        ),
    }

    # Test data generators
    AGENT_TEST_DATA = {
        "basic_agent": {
            "name": "Basic Test Agent",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        },
        "advanced_agent": {
            "name": "Advanced Test Agent",
            "template": "goal-optimizer",
            "parameters": {
                "optimization_target": "efficiency",
                "learning_rate": 0.01,
            },
        },
        "gmn_agent": {
            "name": "GMN Test Agent",
            "gmn_spec": """
[nodes]
position: state {num_states: 25}
obs_position: observation {num_observations: 6}
move: action {num_actions: 6}
position_belief: belief
exploration_pref: preference {preferred_observation: 1}
position_likelihood: likelihood
position_transition: transition

[edges]
position -> position_likelihood: depends_on
position_likelihood -> obs_position: generates
position -> position_transition: depends_on
move -> position_transition: depends_on
exploration_pref -> obs_position: depends_on
position_belief -> position: depends_on
            """,
            "planning_horizon": 3,
        },
    }

    # Test configuration parameters
    TEST_CONFIG = {
        "timeout": 30,  # seconds
        "max_retries": 3,
        "rate_limit_window": 60,  # seconds
        "rate_limit_max_requests": 100,
        "concurrent_test_users": 10,
        "performance_threshold": 1.0,  # seconds
        "memory_threshold": 100,  # MB
        "database_cleanup": True,
        "auth_cleanup": True,
        "temp_file_cleanup": True,
        "verbose_logging": False,
        "generate_reports": True,
        "report_format": "json",
        "security_scan_enabled": True,
        "performance_monitoring": True,
    }

    # Expected HTTP status codes for different scenarios
    EXPECTED_STATUS_CODES = {
        "success": 200,
        "created": 201,
        "unauthorized": 401,
        "forbidden": 403,
        "not_found": 404,
        "rate_limited": 429,
        "server_error": 500,
    }

    # Role hierarchy (higher roles inherit lower role permissions)
    ROLE_HIERARCHY = {
        UserRole.ADMIN: 4,
        UserRole.RESEARCHER: 3,
        UserRole.AGENT_MANAGER: 2,
        UserRole.OBSERVER: 1,
    }

    @classmethod
    def get_user_config(cls, role: str) -> TestUser:
        """Get test user configuration for a specific role."""
        return cls.TEST_USERS.get(role)

    @classmethod
    def get_endpoint_config(cls, category: str, operation: str) -> TestEndpoint:
        """Get test endpoint configuration."""
        return cls.TEST_ENDPOINTS.get(category, {}).get(operation)

    @classmethod
    def get_security_scenario(cls, scenario_name: str) -> SecurityTestScenario:
        """Get security test scenario configuration."""
        return cls.SECURITY_SCENARIOS.get(scenario_name)

    @classmethod
    def get_all_roles(cls) -> List[UserRole]:
        """Get all user roles."""
        return list(cls.TEST_USERS.keys())

    @classmethod
    def get_all_endpoints(cls) -> List[Tuple[str, str, TestEndpoint]]:
        """Get all test endpoints."""
        endpoints = []
        for category, operations in cls.TEST_ENDPOINTS.items():
            for operation, endpoint in operations.items():
                endpoints.append((category, operation, endpoint))
        return endpoints

    @classmethod
    def get_role_permissions(cls, role: UserRole) -> Set[Permission]:
        """Get expected permissions for a role."""
        user_config = cls.TEST_USERS.get(role.value)
        return user_config.expected_permissions if user_config else set()

    @classmethod
    def should_have_access(cls, role: UserRole, operation: str) -> bool:
        """Check if a role should have access to an operation."""
        user_config = cls.TEST_USERS.get(role.value)
        return (
            user_config.should_have_access.get(operation, False)
            if user_config
            else False
        )

    @classmethod
    def get_agent_test_data(cls, agent_type: str) -> Dict:
        """Get agent test data for a specific type."""
        return cls.AGENT_TEST_DATA.get(agent_type, cls.AGENT_TEST_DATA["basic_agent"])

    @classmethod
    def get_test_config(cls, key: str):
        """Get test configuration parameter."""
        return cls.TEST_CONFIG.get(key)

    @classmethod
    def get_expected_status_code(cls, scenario: str) -> int:
        """Get expected HTTP status code for a scenario."""
        return cls.EXPECTED_STATUS_CODES.get(scenario, 200)

    @classmethod
    def is_higher_role(cls, role1: UserRole, role2: UserRole) -> bool:
        """Check if role1 is higher than role2 in the hierarchy."""
        return cls.ROLE_HIERARCHY.get(role1, 0) > cls.ROLE_HIERARCHY.get(role2, 0)


# Test data validation
def validate_test_config():
    """Validate test configuration for consistency."""
    errors = []

    # Validate that all users have valid roles
    for user_name, user_config in RBACTestConfig.TEST_USERS.items():
        if not isinstance(user_config.role, UserRole):
            errors.append(f"User {user_name} has invalid role: {user_config.role}")

    # Validate that all endpoints have valid permissions
    for category, operations in RBACTestConfig.TEST_ENDPOINTS.items():
        for operation, endpoint in operations.items():
            if not isinstance(endpoint.required_permission, Permission):
                errors.append(
                    f"Endpoint {category}.{operation} has invalid permission: {endpoint.required_permission}"
                )

    # Validate security scenarios
    for scenario_name, scenario in RBACTestConfig.SECURITY_SCENARIOS.items():
        if scenario.severity not in ["low", "medium", "high", "critical"]:
            errors.append(
                f"Security scenario {scenario_name} has invalid severity: {scenario.severity}"
            )

    if errors:
        raise ValueError(f"Test configuration validation failed: {errors}")

    return True


# Initialize configuration validation
if __name__ == "__main__":
    try:
        validate_test_config()
        print("✅ RBAC test configuration validation passed")
    except ValueError as e:
        print(f"❌ RBAC test configuration validation failed: {e}")
        exit(1)
