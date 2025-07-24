"""
Business Logic Testing Module

This module implements comprehensive business logic testing including:
- Workflow bypass attempts
- State manipulation
- Race condition exploitation
- Payment logic bypass (if applicable)
- Multi-step process attacks
- Resource allocation bypasses
- Agent creation/deletion logic testing
- Coalition formation vulnerabilities
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from auth.security_implementation import UserRole

from .penetration_testing_framework import (
    BasePenetrationTest,
    SeverityLevel,
    TestResult,
    VulnerabilityFinding,
    VulnerabilityType,
)

logger = logging.getLogger(__name__)


class BusinessLogicTests(BasePenetrationTest):
    """Comprehensive business logic vulnerability testing."""

    async def execute(self) -> TestResult:
        """Execute all business logic tests."""
        start_time = time.time()

        try:
            # Test workflow bypass attempts
            await self._test_workflow_bypass()

            # Test state manipulation
            await self._test_state_manipulation()

            # Test race condition exploitation
            await self._test_race_conditions()

            # Test multi-step process attacks
            await self._test_multi_step_processes()

            # Test resource allocation bypasses
            await self._test_resource_allocation()

            # Test agent lifecycle vulnerabilities
            await self._test_agent_lifecycle()

            # Test coalition formation logic
            await self._test_coalition_logic()

            # Test permission escalation through business logic
            await self._test_business_logic_escalation()

            # Test data validation bypasses
            await self._test_data_validation_bypass()

            # Test transaction logic vulnerabilities
            await self._test_transaction_logic()

            execution_time = time.time() - start_time

            return TestResult(
                test_name="BusinessLogicTests",
                success=True,
                vulnerabilities=self.vulnerabilities,
                execution_time=execution_time,
                metadata={
                    "business_flows_tested": 10,
                    "race_condition_tests": 50,
                    "concurrent_operations": 20,
                },
            )

        except Exception as e:
            logger.error(f"Business logic test failed: {e}")
            return TestResult(
                test_name="BusinessLogicTests",
                success=False,
                vulnerabilities=self.vulnerabilities,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def _test_workflow_bypass(self):
        """Test workflow bypass vulnerabilities."""
        logger.info("Testing workflow bypass vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.RESEARCHER)
        token = self.get_auth_token(username, password)

        # Test agent creation workflow bypass
        await self._test_agent_creation_bypass(token)

        # Test authentication workflow bypass
        await self._test_auth_workflow_bypass()

        # Test resource access workflow bypass
        await self._test_resource_workflow_bypass(token)

    async def _test_agent_creation_bypass(self, token: str):
        """Test agent creation workflow bypass."""
        # Normal workflow: Create -> Configure -> Activate
        # Try to bypass steps or change order

        # Test 1: Try to activate agent without configuration
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "bypass_test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Try to activate immediately without configuration
                    activate_response = self.client.post(
                        f"/api/v1/agents/{agent_id}/activate",
                        headers=self.get_auth_headers(token),
                    )

                    if activate_response.status_code in [200, 202]:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                severity=SeverityLevel.MEDIUM,
                                title="Agent Activation Workflow Bypass",
                                description="Agent can be activated without proper configuration steps",
                                affected_endpoint=f"/api/v1/agents/{agent_id}/activate",
                                proof_of_concept="Created agent and immediately activated without configuration",
                                exploitation_steps=[
                                    "1. Create new agent",
                                    "2. Skip configuration step",
                                    "3. Directly activate agent",
                                    "4. Use improperly configured agent",
                                ],
                                remediation_steps=[
                                    "Implement proper workflow state validation",
                                    "Require configuration before activation",
                                    "Add workflow state checks",
                                    "Implement business rule validation",
                                ],
                                cwe_id="CWE-840",
                                cvss_score=5.3,
                                test_method="agent_creation_bypass",
                            )
                        )

            except Exception as e:
                logger.debug(f"Agent creation bypass test error: {e}")

    async def _test_auth_workflow_bypass(self):
        """Test authentication workflow bypass."""
        # Test login workflow bypass attempts

        # Test 1: Try to access protected resources during password reset
        reset_response = self.client.post(
            "/api/v1/auth/reset-password", json={"email": "test@example.com"}
        )

        if reset_response.status_code in [200, 202]:
            # Try to access protected resources immediately
            access_response = self.client.get("/api/v1/auth/me")

            if access_response.status_code == 200:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                        severity=SeverityLevel.HIGH,
                        title="Authentication Workflow Bypass",
                        description="Can access protected resources during password reset process",
                        affected_endpoint="/api/v1/auth/me",
                        proof_of_concept="Accessed protected endpoint during password reset workflow",
                        exploitation_steps=[
                            "1. Initiate password reset process",
                            "2. Access protected resources during reset",
                            "3. Bypass authentication requirements",
                        ],
                        remediation_steps=[
                            "Implement proper session state management",
                            "Invalidate sessions during password reset",
                            "Add workflow state validation",
                        ],
                        cwe_id="CWE-287",
                        cvss_score=7.5,
                        test_method="auth_workflow_bypass",
                    )
                )

    async def _test_resource_workflow_bypass(self, token: str):
        """Test resource access workflow bypass."""
        # Test if resources can be accessed in improper states

        # Create a coalition
        coalition_response = self.client.post(
            "/api/v1/coalitions",
            json={"name": "test_coalition", "type": "research"},
            headers=self.get_auth_headers(token),
        )

        if coalition_response.status_code in [200, 201]:
            try:
                coalition_data = coalition_response.json()
                coalition_id = coalition_data.get("id")

                if coalition_id:
                    # Try to add agents to coalition before it's properly configured
                    add_agent_response = self.client.post(
                        f"/api/v1/coalitions/{coalition_id}/agents",
                        json={"agent_id": "test_agent_id"},
                        headers=self.get_auth_headers(token),
                    )

                    if add_agent_response.status_code in [200, 201]:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                severity=SeverityLevel.MEDIUM,
                                title="Coalition Workflow Bypass",
                                description="Can add agents to coalition before proper configuration",
                                affected_endpoint=f"/api/v1/coalitions/{coalition_id}/agents",
                                proof_of_concept="Added agent to unconfigured coalition",
                                exploitation_steps=[
                                    "1. Create coalition",
                                    "2. Skip configuration steps",
                                    "3. Add agents to unconfigured coalition",
                                    "4. Use improperly configured coalition",
                                ],
                                remediation_steps=[
                                    "Implement coalition state validation",
                                    "Require proper configuration before agent addition",
                                    "Add workflow state checks",
                                ],
                                cwe_id="CWE-840",
                                cvss_score=4.3,
                                test_method="resource_workflow_bypass",
                            )
                        )

            except Exception as e:
                logger.debug(f"Resource workflow bypass test error: {e}")

    async def _test_state_manipulation(self):
        """Test state manipulation vulnerabilities."""
        logger.info("Testing state manipulation vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.RESEARCHER)
        token = self.get_auth_token(username, password)

        # Test agent state manipulation
        await self._test_agent_state_manipulation(token)

        # Test user state manipulation
        await self._test_user_state_manipulation(token)

        # Test session state manipulation
        await self._test_session_state_manipulation(token)

    async def _test_agent_state_manipulation(self, token: str):
        """Test agent state manipulation."""
        # Create an agent
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "state_test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Test invalid state transitions
                    invalid_transitions = [
                        {
                            "state": "deleted",
                            "description": "Force delete state",
                        },
                        {
                            "state": "system",
                            "description": "Force system state",
                        },
                        {
                            "state": "admin_controlled",
                            "description": "Force admin control",
                        },
                        {
                            "status": "active",
                            "owner": "admin",
                            "description": "Change ownership",
                        },
                    ]

                    for transition in invalid_transitions:
                        update_response = self.client.patch(
                            f"/api/v1/agents/{agent_id}",
                            json=transition,
                            headers=self.get_auth_headers(token),
                        )

                        if update_response.status_code in [200, 202]:
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                    severity=SeverityLevel.MEDIUM,
                                    title=f"Agent State Manipulation - {transition['description']}",
                                    description=f"Can manipulate agent state: {transition}",
                                    affected_endpoint=f"/api/v1/agents/{agent_id}",
                                    proof_of_concept=f"State change accepted: {json.dumps(transition)}",
                                    exploitation_steps=[
                                        "1. Create or access agent",
                                        "2. Send state manipulation request",
                                        "3. Force invalid state transition",
                                        "4. Exploit inconsistent state",
                                    ],
                                    remediation_steps=[
                                        "Implement state transition validation",
                                        "Use state machine pattern for agent lifecycle",
                                        "Validate state changes server-side",
                                        "Implement proper authorization for state changes",
                                    ],
                                    cwe_id="CWE-362",
                                    cvss_score=5.4,
                                    test_method="agent_state_manipulation",
                                )
                            )

            except Exception as e:
                logger.debug(f"Agent state manipulation test error: {e}")

    async def _test_user_state_manipulation(self, token: str):
        """Test user state manipulation."""
        # Try to manipulate user state through API calls
        state_manipulations = [
            {"is_active": False, "description": "Disable own account"},
            {"role": "admin", "description": "Escalate role"},
            {
                "permissions": ["admin_system"],
                "description": "Add admin permissions",
            },
            {
                "last_login": "2030-01-01T00:00:00Z",
                "description": "Manipulate login timestamp",
            },
        ]

        for manipulation in state_manipulations:
            update_response = self.client.patch(
                "/api/v1/auth/me",
                json=manipulation,
                headers=self.get_auth_headers(token),
            )

            if update_response.status_code in [200, 202]:
                try:
                    data = update_response.json()
                    if any(key in data for key in manipulation.keys()):
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                severity=SeverityLevel.HIGH,
                                title=f"User State Manipulation - {manipulation['description']}",
                                description=f"Can manipulate user state: {manipulation}",
                                affected_endpoint="/api/v1/auth/me",
                                proof_of_concept=f"State change accepted: {json.dumps(manipulation)}",
                                exploitation_steps=[
                                    "1. Access user profile endpoint",
                                    "2. Send state manipulation request",
                                    "3. Change protected user attributes",
                                    "4. Gain unauthorized privileges",
                                ],
                                remediation_steps=[
                                    "Implement field-level access controls",
                                    "Validate user state changes",
                                    "Use separate endpoints for different operations",
                                    "Implement proper authorization for user modifications",
                                ],
                                cwe_id="CWE-639",
                                cvss_score=8.1,
                                test_method="user_state_manipulation",
                            )
                        )
                except Exception as e:
                    logger.debug(f"User state manipulation response parsing error: {e}")

    async def _test_session_state_manipulation(self, token: str):
        """Test session state manipulation."""
        # Try to manipulate session state
        session_manipulations = [
            {
                "session_id": "admin_session",
                "description": "Change session ID",
            },
            {"user_id": "admin", "description": "Change user ID in session"},
            {"role": "admin", "description": "Change role in session"},
            {
                "expires": "2030-01-01T00:00:00Z",
                "description": "Extend session",
            },
        ]

        for manipulation in session_manipulations:
            # Try as query parameters
            response = self.client.get(
                "/api/v1/auth/me",
                params=manipulation,
                headers=self.get_auth_headers(token),
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check if session manipulation was successful
                    for key, value in manipulation.items():
                        if key in data and data[key] == value:
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                    severity=SeverityLevel.HIGH,
                                    title=f"Session State Manipulation - {manipulation['description']}",
                                    description=f"Session state manipulated via parameters: {manipulation}",
                                    affected_endpoint="/api/v1/auth/me",
                                    proof_of_concept=f"Parameter manipulation: {json.dumps(manipulation)}",
                                    exploitation_steps=[
                                        "1. Access authenticated endpoint",
                                        "2. Add state manipulation parameters",
                                        "3. Change session state",
                                        "4. Gain unauthorized access",
                                    ],
                                    remediation_steps=[
                                        "Use server-side session storage only",
                                        "Ignore client-provided session data",
                                        "Implement proper session validation",
                                        "Use secure session management",
                                    ],
                                    cwe_id="CWE-384",
                                    cvss_score=7.5,
                                    test_method="session_state_manipulation",
                                )
                            )
                except Exception as e:
                    logger.debug(f"Session state manipulation response parsing error: {e}")

    async def _test_race_conditions(self):
        """Test race condition vulnerabilities."""
        logger.info("Testing race condition vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.RESEARCHER)
        token = self.get_auth_token(username, password)

        # Test concurrent agent creation
        await self._test_concurrent_agent_creation(token)

        # Test concurrent resource access
        await self._test_concurrent_resource_access(token)

        # Test concurrent state changes
        await self._test_concurrent_state_changes(token)

    async def _test_concurrent_agent_creation(self, token: str):
        """Test race conditions in agent creation."""

        def create_agent(thread_id):
            """Create agent in thread."""
            try:
                response = self.client.post(
                    "/api/v1/agents",
                    json={"name": f"race_agent_{thread_id}", "type": "basic"},
                    headers=self.get_auth_headers(token),
                )
                return (
                    response.status_code,
                    response.json() if response.text else {},
                )
            except Exception as e:
                return 500, {"error": str(e)}

        # Create multiple agents concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_agent, i) for i in range(20)]
            results = [future.result() for future in as_completed(futures)]

        successful_creations = sum(1 for status, _ in results if status in [200, 201])

        # Check for race condition indicators
        agent_ids = []
        for status, data in results:
            if status in [200, 201] and data:
                agent_id = data.get("id") or data.get("agent_id")
                if agent_id:
                    agent_ids.append(agent_id)

        # Check for duplicate IDs (race condition in ID generation)
        unique_ids = set(agent_ids)
        if len(unique_ids) < len(agent_ids):
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.RACE_CONDITION,
                    severity=SeverityLevel.MEDIUM,
                    title="Race Condition in Agent Creation - Duplicate IDs",
                    description=f"Concurrent agent creation resulted in {len(agent_ids) - len(unique_ids)} duplicate IDs",
                    affected_endpoint="/api/v1/agents",
                    proof_of_concept=f"Created {len(agent_ids)} agents, {len(unique_ids)} unique IDs",
                    exploitation_steps=[
                        "1. Send multiple concurrent agent creation requests",
                        "2. Exploit race condition in ID generation",
                        "3. Create agents with duplicate IDs",
                        "4. Cause data inconsistency",
                    ],
                    remediation_steps=[
                        "Use atomic operations for ID generation",
                        "Implement proper locking mechanisms",
                        "Use database constraints for uniqueness",
                        "Implement transaction isolation",
                    ],
                    cwe_id="CWE-362",
                    cvss_score=5.3,
                    test_method="concurrent_agent_creation",
                )
            )

        # Check if too many agents were created (resource exhaustion)
        if successful_creations > 15:  # Expected some to fail due to rate limiting
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.RACE_CONDITION,
                    severity=SeverityLevel.LOW,
                    title="Race Condition - Resource Exhaustion",
                    description=f"No rate limiting on concurrent operations: {successful_creations} agents created",
                    affected_endpoint="/api/v1/agents",
                    proof_of_concept=f"Successfully created {successful_creations}/20 agents concurrently",
                    exploitation_steps=[
                        "1. Send many concurrent requests",
                        "2. Exhaust system resources",
                        "3. Cause denial of service",
                    ],
                    remediation_steps=[
                        "Implement rate limiting per user",
                        "Add resource creation limits",
                        "Implement proper concurrency controls",
                    ],
                    cwe_id="CWE-400",
                    cvss_score=3.7,
                    test_method="concurrent_agent_creation",
                )
            )

    async def _test_concurrent_resource_access(self, token: str):
        """Test race conditions in resource access."""
        # Create an agent first
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "race_test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:

                    def access_agent(operation):
                        """Perform operation on agent."""
                        try:
                            if operation == "read":
                                response = self.client.get(
                                    f"/api/v1/agents/{agent_id}",
                                    headers=self.get_auth_headers(token),
                                )
                            elif operation == "update":
                                response = self.client.patch(
                                    f"/api/v1/agents/{agent_id}",
                                    json={"name": f"updated_{time.time()}"},
                                    headers=self.get_auth_headers(token),
                                )
                            elif operation == "delete":
                                response = self.client.delete(
                                    f"/api/v1/agents/{agent_id}",
                                    headers=self.get_auth_headers(token),
                                )
                            else:
                                return 400, {}

                            return (
                                response.status_code,
                                response.json() if response.text else {},
                            )
                        except Exception as e:
                            return 500, {"error": str(e)}

                    # Perform concurrent operations
                    operations = [
                        "read",
                        "update",
                        "read",
                        "update",
                        "delete",
                        "read",
                    ]
                    with ThreadPoolExecutor(max_workers=6) as executor:
                        futures = [executor.submit(access_agent, op) for op in operations]
                        results = [future.result() for future in as_completed(futures)]

                    # Analyze results for race condition indicators
                    read_after_delete = False
                    for i, (status, data) in enumerate(results):
                        if operations[i] == "read" and status == 200:
                            # Check if read succeeded after delete
                            delete_results = [
                                r for j, r in enumerate(results) if operations[j] == "delete"
                            ]
                            if any(s in [200, 204] for s, _ in delete_results):
                                read_after_delete = True

                    if read_after_delete:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.RACE_CONDITION,
                                severity=SeverityLevel.MEDIUM,
                                title="Race Condition in Resource Access",
                                description="Can read resource after deletion due to race condition",
                                affected_endpoint=f"/api/v1/agents/{agent_id}",
                                proof_of_concept="Concurrent read/delete operations show inconsistent state",
                                exploitation_steps=[
                                    "1. Perform concurrent operations on same resource",
                                    "2. Exploit race condition in state management",
                                    "3. Access deleted or inconsistent resources",
                                ],
                                remediation_steps=[
                                    "Implement proper resource locking",
                                    "Use database transactions",
                                    "Implement optimistic or pessimistic locking",
                                    "Add proper synchronization",
                                ],
                                cwe_id="CWE-362",
                                cvss_score=5.4,
                                test_method="concurrent_resource_access",
                            )
                        )

            except Exception as e:
                logger.debug(f"Concurrent resource access test error: {e}")

    async def _test_concurrent_state_changes(self, token: str):
        """Test race conditions in state changes."""

        # Create user accounts concurrently and try to manipulate states
        def create_and_modify_user(thread_id):
            """Create user and modify state."""
            try:
                # Create user
                username = f"race_user_{thread_id}_{int(time.time() * 1000)}"
                create_response = self.client.post(
                    "/api/v1/auth/register",
                    json={
                        "username": username,
                        "email": f"{username}@test.com",
                        "password": "password123",
                        "role": "observer",
                    },
                )

                if create_response.status_code in [200, 201]:
                    # Try to immediately escalate privileges
                    user_data = create_response.json()
                    if "access_token" in user_data:
                        escalate_response = self.client.patch(
                            "/api/v1/auth/me",
                            json={"role": "admin"},
                            headers={"Authorization": f"Bearer {user_data['access_token']}"},
                        )
                        return escalate_response.status_code, (
                            escalate_response.json() if escalate_response.text else {}
                        )

                return create_response.status_code, {}
            except Exception as e:
                return 500, {"error": str(e)}

        # Test concurrent user creation and privilege escalation
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_and_modify_user, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]

        # Check if any privilege escalation succeeded
        successful_escalations = sum(
            1 for status, data in results if status == 200 and data.get("role") == "admin"
        )

        if successful_escalations > 0:
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.RACE_CONDITION,
                    severity=SeverityLevel.CRITICAL,
                    title="Race Condition in Privilege Escalation",
                    description=f"{successful_escalations} users gained admin privileges through race condition",
                    affected_endpoint="/api/v1/auth/me",
                    proof_of_concept=f"Concurrent operations resulted in {successful_escalations} privilege escalations",
                    exploitation_steps=[
                        "1. Create user account",
                        "2. Immediately attempt privilege escalation",
                        "3. Exploit race condition in role assignment",
                        "4. Gain administrative access",
                    ],
                    remediation_steps=[
                        "Implement atomic user creation and role assignment",
                        "Use database transactions for user operations",
                        "Add proper validation for role changes",
                        "Implement rate limiting for sensitive operations",
                    ],
                    cwe_id="CWE-362",
                    cvss_score=9.1,
                    test_method="concurrent_state_changes",
                )
            )

    async def _test_multi_step_processes(self):
        """Test multi-step process vulnerabilities."""
        logger.info("Testing multi-step process vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.RESEARCHER)
        token = self.get_auth_token(username, password)

        # Test agent deployment process
        await self._test_agent_deployment_process(token)

        # Test coalition formation process
        await self._test_coalition_formation_process(token)

        # Test user registration process
        await self._test_registration_process()

    async def _test_agent_deployment_process(self, token: str):
        """Test agent deployment multi-step process."""
        # Normal process: Create -> Configure -> Deploy -> Activate

        # Step 1: Create agent
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "deploy_test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Try to skip steps in the deployment process

                    # Test: Skip configuration and go directly to deployment
                    deploy_response = self.client.post(
                        f"/api/v1/agents/{agent_id}/deploy",
                        headers=self.get_auth_headers(token),
                    )

                    if deploy_response.status_code in [200, 202]:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                severity=SeverityLevel.MEDIUM,
                                title="Agent Deployment Process Bypass",
                                description="Can deploy agent without configuration step",
                                affected_endpoint=f"/api/v1/agents/{agent_id}/deploy",
                                proof_of_concept="Deployed agent without required configuration",
                                exploitation_steps=[
                                    "1. Create agent",
                                    "2. Skip configuration step",
                                    "3. Deploy unconfigured agent",
                                    "4. Use improperly deployed agent",
                                ],
                                remediation_steps=[
                                    "Implement process step validation",
                                    "Require configuration before deployment",
                                    "Add deployment readiness checks",
                                    "Implement proper state machine",
                                ],
                                cwe_id="CWE-840",
                                cvss_score=5.3,
                                test_method="agent_deployment_process",
                            )
                        )

                    # Test: Try to perform operations in wrong order
                    activate_response = self.client.post(
                        f"/api/v1/agents/{agent_id}/activate",
                        headers=self.get_auth_headers(token),
                    )

                    if activate_response.status_code in [200, 202]:
                        # Check if agent is actually activated without proper deployment
                        status_response = self.client.get(
                            f"/api/v1/agents/{agent_id}",
                            headers=self.get_auth_headers(token),
                        )

                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data.get("status") == "active":
                                self.add_vulnerability(
                                    VulnerabilityFinding(
                                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                        severity=SeverityLevel.MEDIUM,
                                        title="Agent Activation Process Bypass",
                                        description="Can activate agent without deployment",
                                        affected_endpoint=f"/api/v1/agents/{agent_id}/activate",
                                        proof_of_concept="Activated agent without deployment step",
                                        exploitation_steps=[
                                            "1. Create agent",
                                            "2. Skip deployment step",
                                            "3. Activate undeployed agent",
                                            "4. Use agent in invalid state",
                                        ],
                                        remediation_steps=[
                                            "Implement strict process flow validation",
                                            "Check prerequisites before each step",
                                            "Use workflow engine for complex processes",
                                        ],
                                        cwe_id="CWE-840",
                                        cvss_score=5.3,
                                        test_method="agent_deployment_process",
                                    )
                                )

            except Exception as e:
                logger.debug(f"Agent deployment process test error: {e}")

    async def _test_coalition_formation_process(self, token: str):
        """Test coalition formation multi-step process."""
        # Normal process: Create -> Configure -> Add Members -> Activate

        # Step 1: Create coalition
        create_response = self.client.post(
            "/api/v1/coalitions",
            json={"name": "test_coalition", "type": "research"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                coalition_data = create_response.json()
                coalition_id = coalition_data.get("id")

                if coalition_id:
                    # Try to activate coalition without members
                    activate_response = self.client.post(
                        f"/api/v1/coalitions/{coalition_id}/activate",
                        headers=self.get_auth_headers(token),
                    )

                    if activate_response.status_code in [200, 202]:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                severity=SeverityLevel.LOW,
                                title="Coalition Activation Without Members",
                                description="Can activate coalition without adding members",
                                affected_endpoint=f"/api/v1/coalitions/{coalition_id}/activate",
                                proof_of_concept="Activated empty coalition",
                                exploitation_steps=[
                                    "1. Create coalition",
                                    "2. Skip member addition",
                                    "3. Activate empty coalition",
                                    "4. Use coalition without proper setup",
                                ],
                                remediation_steps=[
                                    "Validate coalition has members before activation",
                                    "Implement minimum member requirements",
                                    "Add business rule validation",
                                ],
                                cwe_id="CWE-840",
                                cvss_score=3.1,
                                test_method="coalition_formation_process",
                            )
                        )

            except Exception as e:
                logger.debug(f"Coalition formation process test error: {e}")

    async def _test_registration_process(self):
        """Test user registration multi-step process."""
        # Some systems have multi-step registration (register -> verify email -> activate)

        # Test: Try to use account immediately after registration
        register_response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": f"process_test_{int(time.time())}",
                "email": "process@test.com",
                "password": "password123",
                "role": "observer",
            },
        )

        if register_response.status_code in [200, 201]:
            try:
                data = register_response.json()
                if "access_token" in data:
                    # User can immediately use account - check if this bypasses verification
                    token = data["access_token"]

                    # Try to access protected resources immediately
                    protected_response = self.client.get(
                        "/api/v1/auth/me",
                        headers={"Authorization": f"Bearer {token}"},
                    )

                    if protected_response.status_code == 200:
                        user_data = protected_response.json()

                        # Check if account should require verification
                        if not user_data.get(
                            "email_verified", True
                        ):  # Assuming false means needs verification
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                    severity=SeverityLevel.LOW,
                                    title="Registration Process Bypass",
                                    description="Can use account before email verification",
                                    affected_endpoint="/api/v1/auth/register",
                                    proof_of_concept="Immediate access to protected resources after registration",
                                    exploitation_steps=[
                                        "1. Register account",
                                        "2. Skip email verification",
                                        "3. Access protected resources",
                                        "4. Use unverified account",
                                    ],
                                    remediation_steps=[
                                        "Require email verification before account activation",
                                        "Limit unverified account capabilities",
                                        "Implement proper verification workflow",
                                    ],
                                    cwe_id="CWE-287",
                                    cvss_score=3.7,
                                    test_method="registration_process",
                                )
                            )

            except Exception as e:
                logger.debug(f"Registration process test error: {e}")

    async def _test_resource_allocation(self):
        """Test resource allocation bypasses."""
        logger.info("Testing resource allocation bypasses")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.OBSERVER)
        token = self.get_auth_token(username, password)

        # Test agent creation limits
        await self._test_agent_creation_limits(token)

        # Test resource consumption limits
        await self._test_resource_consumption_limits(token)

    async def _test_agent_creation_limits(self, token: str):
        """Test agent creation limit bypass."""
        # Try to create many agents to test limits
        created_agents = 0
        max_attempts = 20

        for i in range(max_attempts):
            response = self.client.post(
                "/api/v1/agents",
                json={"name": f"limit_test_agent_{i}", "type": "basic"},
                headers=self.get_auth_headers(token),
            )

            if response.status_code in [200, 201]:
                created_agents += 1
            elif response.status_code == 429:  # Rate limited
                break
            elif response.status_code == 403:  # Forbidden due to limits
                break

        # If observer role can create many agents, this might be a vulnerability
        if created_agents > 10:  # Arbitrary threshold for observer role
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                    severity=SeverityLevel.LOW,
                    title="Excessive Resource Allocation for Observer Role",
                    description=f"Observer role can create {created_agents} agents",
                    affected_endpoint="/api/v1/agents",
                    proof_of_concept=f"Created {created_agents} agents with observer role",
                    exploitation_steps=[
                        "1. Use low-privilege account",
                        "2. Create excessive resources",
                        "3. Consume system resources",
                        "4. Potentially cause DoS",
                    ],
                    remediation_steps=[
                        "Implement role-based resource limits",
                        "Add quota system for resource creation",
                        "Monitor resource usage per user",
                        "Implement proper rate limiting",
                    ],
                    cwe_id="CWE-770",
                    cvss_score=3.7,
                    test_method="agent_creation_limits",
                )
            )

    async def _test_resource_consumption_limits(self, token: str):
        """Test resource consumption limit bypass."""
        # Test large data uploads/requests
        large_data_tests = [
            # Large agent name
            {"name": "A" * 10000, "type": "basic"},
            # Large configuration
            {
                "name": "test",
                "type": "basic",
                "config": {"data": "X" * 100000},
            },
        ]

        for test_data in large_data_tests:
            response = self.client.post(
                "/api/v1/agents",
                json=test_data,
                headers=self.get_auth_headers(token),
            )

            if response.status_code in [200, 201]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                        severity=SeverityLevel.LOW,
                        title="No Resource Consumption Limits",
                        description="Can send large data payloads without limits",
                        affected_endpoint="/api/v1/agents",
                        proof_of_concept=f"Large payload accepted: {len(str(test_data))} bytes",
                        exploitation_steps=[
                            "1. Send large data payloads",
                            "2. Consume server resources",
                            "3. Potentially cause DoS",
                        ],
                        remediation_steps=[
                            "Implement payload size limits",
                            "Add request size validation",
                            "Implement resource usage monitoring",
                        ],
                        cwe_id="CWE-770",
                        cvss_score=3.1,
                        test_method="resource_consumption_limits",
                    )
                )

    async def _test_agent_lifecycle(self):
        """Test agent lifecycle vulnerabilities."""
        logger.info("Testing agent lifecycle vulnerabilities")

        # Create test user with agent management permissions
        username, password, user_id = self.create_test_user(UserRole.AGENT_MANAGER)
        token = self.get_auth_token(username, password)

        # Test agent lifecycle state transitions
        await self._test_agent_lifecycle_transitions(token)

        # Test agent ownership transfer
        await self._test_agent_ownership_transfer(token)

    async def _test_agent_lifecycle_transitions(self, token: str):
        """Test agent lifecycle state transitions."""
        # Create an agent
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "lifecycle_test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Test invalid lifecycle transitions
                    invalid_transitions = [
                        # Try to delete active agent
                        ("delete", f"/api/v1/agents/{agent_id}"),
                        # Try to reset agent to initial state
                        (
                            "patch",
                            f"/api/v1/agents/{agent_id}",
                            {"state": "created"},
                        ),
                    ]

                    for method, endpoint, data in [
                        (t[0], t[1], t[2] if len(t) > 2 else None) for t in invalid_transitions
                    ]:
                        if method == "delete":
                            response = self.client.delete(
                                endpoint, headers=self.get_auth_headers(token)
                            )
                        elif method == "patch" and data:
                            response = self.client.patch(
                                endpoint,
                                json=data,
                                headers=self.get_auth_headers(token),
                            )

                        if hasattr(response, "status_code") and response.status_code in [
                            200,
                            202,
                            204,
                        ]:
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                    severity=SeverityLevel.MEDIUM,
                                    title="Agent Lifecycle Transition Bypass",
                                    description=f"Invalid lifecycle transition allowed: {method} {endpoint}",
                                    affected_endpoint=endpoint,
                                    proof_of_concept=f"Performed {method} operation in invalid state",
                                    exploitation_steps=[
                                        "1. Create agent",
                                        "2. Perform invalid state transition",
                                        "3. Agent in inconsistent state",
                                    ],
                                    remediation_steps=[
                                        "Implement proper state machine validation",
                                        "Check current state before transitions",
                                        "Add lifecycle validation rules",
                                    ],
                                    cwe_id="CWE-362",
                                    cvss_score=5.4,
                                    test_method="agent_lifecycle_transitions",
                                )
                            )

            except Exception as e:
                logger.debug(f"Agent lifecycle test error: {e}")

    async def _test_agent_ownership_transfer(self, token: str):
        """Test agent ownership transfer vulnerabilities."""
        # Create another user
        user2_name, user2_pass, user2_id = self.create_test_user(UserRole.AGENT_MANAGER)

        # Create an agent with first user
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "ownership_test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Try to transfer ownership
                    transfer_response = self.client.patch(
                        f"/api/v1/agents/{agent_id}",
                        json={"owner": user2_id},
                        headers=self.get_auth_headers(token),
                    )

                    if transfer_response.status_code in [200, 202]:
                        # Check if ownership was actually transferred
                        check_response = self.client.get(
                            f"/api/v1/agents/{agent_id}",
                            headers=self.get_auth_headers(token),
                        )

                        if check_response.status_code == 200:
                            check_data = check_response.json()
                            if check_data.get("owner") == user2_id:
                                self.add_vulnerability(
                                    VulnerabilityFinding(
                                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                        severity=SeverityLevel.MEDIUM,
                                        title="Unauthorized Agent Ownership Transfer",
                                        description="Can transfer agent ownership without proper authorization",
                                        affected_endpoint=f"/api/v1/agents/{agent_id}",
                                        proof_of_concept=f"Transferred ownership from original_user to {user2_id}",
                                        exploitation_steps=[
                                            "1. Create agent",
                                            "2. Transfer ownership to another user",
                                            "3. Lose control of agent",
                                            "4. Potential privilege escalation",
                                        ],
                                        remediation_steps=[
                                            "Implement proper ownership transfer validation",
                                            "Require confirmation from new owner",
                                            "Add audit logging for ownership changes",
                                            "Implement proper authorization checks",
                                        ],
                                        cwe_id="CWE-639",
                                        cvss_score=6.1,
                                        test_method="agent_ownership_transfer",
                                    )
                                )

            except Exception as e:
                logger.debug(f"Agent ownership transfer test error: {e}")

    async def _test_coalition_logic(self):
        """Test coalition formation logic vulnerabilities."""
        logger.info("Testing coalition logic vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.RESEARCHER)
        token = self.get_auth_token(username, password)

        # Test coalition member limits
        await self._test_coalition_member_limits(token)

        # Test coalition permission escalation
        await self._test_coalition_permission_escalation(token)

    async def _test_coalition_member_limits(self, token: str):
        """Test coalition member limit bypass."""
        # Create a coalition
        create_response = self.client.post(
            "/api/v1/coalitions",
            json={"name": "limit_test_coalition", "type": "research"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                coalition_data = create_response.json()
                coalition_id = coalition_data.get("id")

                if coalition_id:
                    # Try to add many members
                    members_added = 0
                    for i in range(100):  # Try to add 100 members
                        add_response = self.client.post(
                            f"/api/v1/coalitions/{coalition_id}/members",
                            json={"user_id": f"fake_user_{i}"},
                            headers=self.get_auth_headers(token),
                        )

                        if add_response.status_code in [200, 201]:
                            members_added += 1
                        elif add_response.status_code in [400, 403, 429]:
                            break

                    if members_added > 50:  # Arbitrary threshold
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                severity=SeverityLevel.LOW,
                                title="No Coalition Member Limits",
                                description=f"Can add {members_added} members to coalition",
                                affected_endpoint=f"/api/v1/coalitions/{coalition_id}/members",
                                proof_of_concept=f"Added {members_added} members without limit",
                                exploitation_steps=[
                                    "1. Create coalition",
                                    "2. Add excessive members",
                                    "3. Consume system resources",
                                ],
                                remediation_steps=[
                                    "Implement coalition member limits",
                                    "Add validation for member additions",
                                    "Implement resource usage monitoring",
                                ],
                                cwe_id="CWE-770",
                                cvss_score=3.1,
                                test_method="coalition_member_limits",
                            )
                        )

            except Exception as e:
                logger.debug(f"Coalition member limits test error: {e}")

    async def _test_coalition_permission_escalation(self, token: str):
        """Test coalition permission escalation."""
        # Test if coalition membership can escalate permissions

        # Create a coalition
        create_response = self.client.post(
            "/api/v1/coalitions",
            json={"name": "permission_test_coalition", "type": "admin"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                coalition_data = create_response.json()
                coalition_id = coalition_data.get("id")

                if coalition_id:
                    # Check if creating admin coalition escalates permissions
                    check_response = self.client.get(
                        "/api/v1/auth/permissions",
                        headers=self.get_auth_headers(token),
                    )

                    if check_response.status_code == 200:
                        perms_data = check_response.json()
                        if perms_data.get("can_admin_system", False):
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                                    severity=SeverityLevel.HIGH,
                                    title="Coalition-based Permission Escalation",
                                    description="Creating admin coalition grants admin permissions",
                                    affected_endpoint="/api/v1/coalitions",
                                    proof_of_concept="Admin permissions gained through coalition creation",
                                    exploitation_steps=[
                                        "1. Create coalition with admin type",
                                        "2. Gain admin permissions",
                                        "3. Access administrative functions",
                                    ],
                                    remediation_steps=[
                                        "Separate coalition types from user permissions",
                                        "Implement proper permission validation",
                                        "Don't grant permissions based on resource creation",
                                    ],
                                    cwe_id="CWE-269",
                                    cvss_score=8.1,
                                    test_method="coalition_permission_escalation",
                                )
                            )

            except Exception as e:
                logger.debug(f"Coalition permission escalation test error: {e}")

    async def _test_business_logic_escalation(self):
        """Test privilege escalation through business logic."""
        logger.info("Testing business logic privilege escalation")

        # Create limited user
        username, password, user_id = self.create_test_user(UserRole.OBSERVER)
        token = self.get_auth_token(username, password)

        # Test resource creation privilege escalation
        await self._test_resource_creation_escalation(token)

        # Test collaboration privilege escalation
        await self._test_collaboration_escalation(token)

    async def _test_resource_creation_escalation(self, token: str):
        """Test privilege escalation through resource creation."""
        # Try to create resources that might grant additional permissions
        escalation_attempts = [
            {"name": "admin_agent", "type": "admin", "permissions": ["admin"]},
            {"name": "system_agent", "type": "system", "owner": "system"},
            {
                "name": "test_agent",
                "type": "privileged",
                "access_level": "admin",
            },
        ]

        for attempt in escalation_attempts:
            response = self.client.post(
                "/api/v1/agents",
                json=attempt,
                headers=self.get_auth_headers(token),
            )

            if response.status_code in [200, 201]:
                # Check if this granted additional permissions
                perms_response = self.client.get(
                    "/api/v1/auth/permissions",
                    headers=self.get_auth_headers(token),
                )

                if perms_response.status_code == 200:
                    perms_data = perms_response.json()
                    if perms_data.get("can_admin_system", False) or perms_data.get(
                        "can_delete_agents", False
                    ):
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                                severity=SeverityLevel.HIGH,
                                title="Resource Creation Privilege Escalation",
                                description=f"Creating agent with {attempt} grants elevated permissions",
                                affected_endpoint="/api/v1/agents",
                                proof_of_concept=f"Agent creation payload: {json.dumps(attempt)}",
                                exploitation_steps=[
                                    "1. Create agent with special attributes",
                                    "2. Gain elevated permissions",
                                    "3. Access restricted functionality",
                                ],
                                remediation_steps=[
                                    "Validate agent creation parameters",
                                    "Don't grant permissions based on resource attributes",
                                    "Implement proper access control separation",
                                ],
                                cwe_id="CWE-269",
                                cvss_score=8.1,
                                test_method="resource_creation_escalation",
                            )
                        )

    async def _test_collaboration_escalation(self, token: str):
        """Test privilege escalation through collaboration features."""
        # Test if joining or creating collaborations can escalate privileges

        collaboration_tests = [
            {
                "endpoint": "/api/v1/coalitions",
                "data": {"name": "admin_coalition", "type": "admin"},
            },
            {
                "endpoint": "/api/v1/projects",
                "data": {"name": "admin_project", "level": "admin"},
            },
        ]

        for test in collaboration_tests:
            if self.client.__dict__.get(test["endpoint"]):  # Check if endpoint exists
                response = self.client.post(
                    test["endpoint"],
                    json=test["data"],
                    headers=self.get_auth_headers(token),
                )

                if response.status_code in [200, 201]:
                    # Check for permission escalation
                    perms_response = self.client.get(
                        "/api/v1/auth/permissions",
                        headers=self.get_auth_headers(token),
                    )

                    if perms_response.status_code == 200:
                        perms_data = perms_response.json()
                        if perms_data.get("can_admin_system", False):
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                                    severity=SeverityLevel.HIGH,
                                    title="Collaboration Privilege Escalation",
                                    description=f"Creating {test['endpoint']} resource grants admin permissions",
                                    affected_endpoint=test["endpoint"],
                                    proof_of_concept=f"Resource creation: {json.dumps(test['data'])}",
                                    exploitation_steps=[
                                        "1. Create collaboration resource",
                                        "2. Gain admin permissions",
                                        "3. Access administrative functions",
                                    ],
                                    remediation_steps=[
                                        "Separate resource creation from permission grants",
                                        "Implement proper permission validation",
                                        "Use role-based access control",
                                    ],
                                    cwe_id="CWE-269",
                                    cvss_score=8.1,
                                    test_method="collaboration_escalation",
                                )
                            )

    async def _test_data_validation_bypass(self):
        """Test data validation bypass vulnerabilities."""
        logger.info("Testing data validation bypass")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.RESEARCHER)
        token = self.get_auth_token(username, password)

        # Test business rule validation bypass
        await self._test_business_rule_bypass(token)

        # Test constraint bypass
        await self._test_constraint_bypass(token)

    async def _test_business_rule_bypass(self, token: str):
        """Test business rule validation bypass."""
        # Test invalid business data
        invalid_data_tests = [
            # Negative values where they shouldn't be allowed
            {"name": "test_agent", "priority": -1, "cost": -100},
            # Future dates in past contexts
            {"name": "test_agent", "created_at": "2030-01-01T00:00:00Z"},
            # Invalid combinations
            {"name": "test_agent", "type": "nonexistent", "status": "invalid"},
        ]

        for test_data in invalid_data_tests:
            response = self.client.post(
                "/api/v1/agents",
                json=test_data,
                headers=self.get_auth_headers(token),
            )

            if response.status_code in [200, 201]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                        severity=SeverityLevel.LOW,
                        title="Business Rule Validation Bypass",
                        description=f"Invalid business data accepted: {test_data}",
                        affected_endpoint="/api/v1/agents",
                        proof_of_concept=f"Invalid data: {json.dumps(test_data)}",
                        exploitation_steps=[
                            "1. Send invalid business data",
                            "2. Bypass validation rules",
                            "3. Create resources with invalid state",
                        ],
                        remediation_steps=[
                            "Implement comprehensive business rule validation",
                            "Add data constraint checks",
                            "Validate data relationships and logic",
                        ],
                        cwe_id="CWE-20",
                        cvss_score=3.7,
                        test_method="business_rule_bypass",
                    )
                )

    async def _test_constraint_bypass(self, token: str):
        """Test constraint bypass vulnerabilities."""
        # Test constraint violations
        constraint_tests = [
            # Duplicate names (if uniqueness required)
            {"name": "duplicate_agent", "type": "basic"},
            # Empty required fields
            {"name": "", "type": "basic"},
            # Oversized fields
            {"name": "A" * 1000, "type": "basic"},
        ]

        for test_data in constraint_tests:
            response = self.client.post(
                "/api/v1/agents",
                json=test_data,
                headers=self.get_auth_headers(token),
            )

            if response.status_code in [200, 201]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                        severity=SeverityLevel.LOW,
                        title="Data Constraint Bypass",
                        description=f"Data constraints not enforced: {test_data}",
                        affected_endpoint="/api/v1/agents",
                        proof_of_concept=f"Constraint violation: {json.dumps(test_data)}",
                        exploitation_steps=[
                            "1. Send data violating constraints",
                            "2. Bypass validation",
                            "3. Create invalid resources",
                        ],
                        remediation_steps=[
                            "Implement proper data validation",
                            "Add constraint checks",
                            "Use database constraints",
                        ],
                        cwe_id="CWE-20",
                        cvss_score=3.1,
                        test_method="constraint_bypass",
                    )
                )

    async def _test_transaction_logic(self):
        """Test transaction logic vulnerabilities."""
        logger.info("Testing transaction logic vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user(UserRole.RESEARCHER)
        token = self.get_auth_token(username, password)

        # Test atomicity violations
        await self._test_atomicity_violations(token)

        # Test consistency violations
        await self._test_consistency_violations(token)

    async def _test_atomicity_violations(self, token: str):
        """Test transaction atomicity violations."""
        # Test operations that should be atomic but might not be

        # Test: Create agent with dependencies
        create_data = {
            "name": "atomic_test_agent",
            "type": "basic",
            "dependencies": ["nonexistent_dependency"],
            "configuration": {"invalid": "config"},
        }

        response = self.client.post(
            "/api/v1/agents",
            json=create_data,
            headers=self.get_auth_headers(token),
        )

        if response.status_code in [200, 201]:
            # Check if partial creation occurred
            try:
                agent_data = response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Check agent status
                    check_response = self.client.get(
                        f"/api/v1/agents/{agent_id}",
                        headers=self.get_auth_headers(token),
                    )

                    if check_response.status_code == 200:
                        agent_status = check_response.json()

                        # If agent exists but dependencies failed, atomicity violated
                        if agent_status.get(
                            "status"
                        ) != "error" and "nonexistent_dependency" in str(create_data):
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                    severity=SeverityLevel.LOW,
                                    title="Transaction Atomicity Violation",
                                    description="Agent created despite dependency failures",
                                    affected_endpoint="/api/v1/agents",
                                    proof_of_concept="Agent created with invalid dependencies",
                                    exploitation_steps=[
                                        "1. Create resource with invalid dependencies",
                                        "2. Partial creation occurs",
                                        "3. System in inconsistent state",
                                    ],
                                    remediation_steps=[
                                        "Implement proper transaction management",
                                        "Validate all dependencies before creation",
                                        "Use database transactions for atomic operations",
                                    ],
                                    cwe_id="CWE-362",
                                    cvss_score=3.1,
                                    test_method="atomicity_violations",
                                )
                            )

            except Exception as e:
                logger.debug(f"Atomicity test error: {e}")

    async def _test_consistency_violations(self, token: str):
        """Test data consistency violations."""
        # Test operations that might leave data in inconsistent state

        # Create an agent
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "consistency_test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Try to create conflicting state
                    self.client.patch(
                        f"/api/v1/agents/{agent_id}",
                        json={"status": "active"},
                        headers=self.get_auth_headers(token),
                    )

                    update2 = self.client.patch(
                        f"/api/v1/agents/{agent_id}",
                        json={"status": "deleted", "active": True},
                        headers=self.get_auth_headers(token),
                    )

                    if update2.status_code in [200, 202]:
                        # Check final state
                        final_response = self.client.get(
                            f"/api/v1/agents/{agent_id}",
                            headers=self.get_auth_headers(token),
                        )

                        if final_response.status_code == 200:
                            final_data = final_response.json()

                            # Check for inconsistent state (deleted but active)
                            if (
                                final_data.get("status") == "deleted"
                                and final_data.get("active") is True
                            ):
                                self.add_vulnerability(
                                    VulnerabilityFinding(
                                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_BYPASS,
                                        severity=SeverityLevel.LOW,
                                        title="Data Consistency Violation",
                                        description="Agent in inconsistent state (deleted but active)",
                                        affected_endpoint=f"/api/v1/agents/{agent_id}",
                                        proof_of_concept="Agent marked as deleted but still active",
                                        exploitation_steps=[
                                            "1. Update resource with conflicting states",
                                            "2. Create data inconsistency",
                                            "3. System in invalid state",
                                        ],
                                        remediation_steps=[
                                            "Implement data consistency checks",
                                            "Add state validation rules",
                                            "Use database constraints",
                                            "Implement proper state machine",
                                        ],
                                        cwe_id="CWE-362",
                                        cvss_score=3.1,
                                        test_method="consistency_violations",
                                    )
                                )

            except Exception as e:
                logger.debug(f"Consistency test error: {e}")
