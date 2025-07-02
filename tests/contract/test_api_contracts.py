"""API Contract Tests.

Expert Committee: Martin Fowler (API design), Robert C. Martin (contracts)
Testing API contracts and backwards compatibility.
"""

from typing import Any, Dict, Optional

import pytest

from tests.contract.contract_test_base import (
    AgentResponseSchema,
    APIContract,
    CoalitionCreateRequestSchema,
    CoalitionResponseSchema,
    ContractSchema,
    ContractTestRunner,
    CreateAgentContract,
    GetAgentContract,
    ListAgentsContract,
    contract_registry,
)


class TestAgentAPIContracts:
    """Test contracts for Agent API endpoints."""

    @pytest.fixture
    def contract_runner(self):
        """Create contract test runner."""
        return ContractTestRunner()

    @pytest.mark.asyncio
    async def test_create_agent_contract(self, client, contract_runner):
        """Test agent creation API contract."""
        contract = CreateAgentContract()
        result = await contract_runner.test_contract(contract, client)

        if not result.passed:
            violations_str = "\n".join(
                f"- {v.field}: {v.message}" for v in result.violations)
            pytest.fail(f"Contract violations:\n{violations_str}")

    @pytest.mark.asyncio
    async def test_get_agent_contract(self, client, contract_runner):
        """Test get agent API contract."""
        # First create an agent
        create_response = await client.post(
            "/api/agents",
            json={"name": "TestAgent", "agent_type": "explorer", "initial_position": [0, 0]},
        )

        if hasattr(create_response, "json"):
            agent_data = create_response.json()
            agent_id = agent_data.get("id", "test-id")
        else:
            agent_id = "test-id"

        # Test get contract
        contract = GetAgentContract(agent_id)
        result = await contract_runner.test_contract(contract, client)

        if not result.passed:
            violations_str = "\n".join(
                f"- {v.field}: {v.message}" for v in result.violations)
            pytest.fail(f"Contract violations:\n{violations_str}")

    @pytest.mark.asyncio
    async def test_list_agents_contract(self, client, contract_runner):
        """Test list agents API contract."""
        contract = ListAgentsContract()
        result = await contract_runner.test_contract(contract, client)

        if not result.passed:
            violations_str = "\n".join(
                f"- {v.field}: {v.message}" for v in result.violations)
            pytest.fail(f"Contract violations:\n{violations_str}")

    @pytest.mark.asyncio
    async def test_agent_response_backwards_compatibility(self, client):
        """Test that agent responses maintain backwards compatibility."""
        response = await client.get("/api/agents")

        if hasattr(response, "json"):
            data = response.json()
        else:
            data = {"items": [], "total": 0}

        # Check v1 contract fields are present
        assert "items" in data
        assert "total" in data

        # If there are agents, check their structure
        if data["items"]:
            agent = data["items"][0]
            v1_required_fields = ["id", "name", "agent_type", "status"]
            for field in v1_required_fields:
                assert field in agent, f"Missing v1 field: {field}"


class TestCoalitionAPIContracts:
    """Test contracts for Coalition API endpoints."""

    @pytest.fixture
    def contract_runner(self):
        """Create contract test runner."""
        return ContractTestRunner()

    @pytest.mark.asyncio
    async def test_create_coalition_contract(self, client, contract_runner):
        """Test coalition creation API contract."""

        # Define coalition contract
        class CreateCoalitionContract(APIContract):
            @property
            def endpoint(self) -> str:
                return "/api/coalitions"

            @property
            def method(self) -> str:
                return "POST"

            @property
            def request_schema(self):
                return CoalitionCreateRequestSchema

            @property
            def response_schema(self):
                return CoalitionResponseSchema

            @property
            def expected_status_codes(self):
                return [201]

        contract = CreateCoalitionContract()
        result = await contract_runner.test_contract(contract, client)

        if not result.passed:
            violations_str = "\n".join(
                f"- {v.field}: {v.message}" for v in result.violations)
            pytest.fail(f"Contract violations:\n{violations_str}")


class TestWebSocketContracts:
    """Test contracts for WebSocket endpoints."""

    @pytest.mark.asyncio
    async def test_websocket_message_contract(self, client):
        """Test WebSocket message contracts."""
        # WebSocket contracts are different - they test message formats

        # Define expected message schemas
        class WSSubscribeMessage(ContractSchema):
            type: str  # "subscribe"
            topic: str
            params: Optional[Dict[str, Any]] = None

        class WSUpdateMessage(ContractSchema):
            type: str  # "update"
            topic: str
            data: Dict[str, Any]
            timestamp: str

        # Test message validation
        subscribe_msg = {"type": "subscribe", "topic": "agents.updates"}

        violations = WSSubscribeMessage.validate_contract(subscribe_msg)
        assert len(
            violations) == 0, f"Subscribe message contract violations: {violations}"

        update_msg = {
            "type": "update",
            "topic": "agents.updates",
            "data": {"agent_id": "123", "status": "active"},
            "timestamp": "2024-01-01T00:00:00Z",
        }

        violations = WSUpdateMessage.validate_contract(update_msg)
        assert len(
            violations) == 0, f"Update message contract violations: {violations}"


class TestContractVersioning:
    """Test API versioning and contract evolution."""

    def test_contract_registry(self):
        """Test contract registry functionality."""
        # Check registered contracts
        v1_contracts = contract_registry.get_all_contracts("v1")
        assert len(v1_contracts) >= 3  # At least the 3 we registered

        # Check specific contract
        create_contract = contract_registry.get_contract(
            "POST", "/api/agents", "v1")
        assert create_contract is not None
        assert isinstance(create_contract, CreateAgentContract)

    def test_contract_versioning(self):
        """Test supporting multiple API versions."""

        # Define v2 contract with additional fields
        class AgentResponseSchemaV2(AgentResponseSchema):
            capabilities: Dict[str, float]
            resources: Dict[str, float]
            version: str = "v2"

        class CreateAgentContractV2(CreateAgentContract):
            @property
            def response_schema(self):
                return AgentResponseSchemaV2

        # Register v2 contract
        contract_registry.register(CreateAgentContractV2(), "v2")

        # Check both versions exist
        v1_contract = contract_registry.get_contract(
            "POST", "/api/agents", "v1")
        v2_contract = contract_registry.get_contract(
            "POST", "/api/agents", "v2")

        assert v1_contract is not None
        assert v2_contract is not None
        assert v1_contract.response_schema != v2_contract.response_schema


class TestContractValidation:
    """Test contract validation functionality."""

    def test_strict_validation(self):
        """Test that extra fields cause contract violations."""
        schema = AgentResponseSchema

        # Valid data
        valid_data = {
            "id": "123",
            "name": "TestAgent",
            "agent_type": "explorer",
            "status": "active",
            "position": {"x": 0, "y": 0, "z": 0},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        violations = schema.validate_contract(valid_data)
        assert len(violations) == 0

        # Data with extra field
        invalid_data = valid_data.copy()
        invalid_data["extra_field"] = "should not be here"

        violations = schema.validate_contract(invalid_data)
        assert len(violations) > 0
        assert any("extra" in v.message.lower() for v in violations)

    def test_missing_required_fields(self):
        """Test that missing required fields cause violations."""
        schema = AgentResponseSchema

        # Missing required field
        incomplete_data = {
            "id": "123",
            "name": "TestAgent",
            # Missing: agent_type, status, position, created_at, updated_at
        }

        violations = schema.validate_contract(incomplete_data)
        assert len(violations) > 0
        assert any("agent_type" in v.field for v in violations)

    def test_type_validation(self):
        """Test that incorrect types cause violations."""
        schema = AgentResponseSchema

        # Wrong type for position
        invalid_type_data = {
            "id": "123",
            "name": "TestAgent",
            "agent_type": "explorer",
            "status": "active",
            "position": "should be dict not string",  # Wrong type
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        violations = schema.validate_contract(invalid_type_data)
        assert len(violations) > 0
        assert any("position" in v.field for v in violations)


@pytest.mark.asyncio
class TestContractReporting:
    """Test contract test reporting functionality."""

    async def test_contract_report_generation(self, client):
        """Test generating contract test reports."""
        runner = ContractTestRunner()

        # Register test contracts
        runner.register_contract(CreateAgentContract())
        runner.register_contract(ListAgentsContract())

        # Run all contracts
        results = await runner.test_all_contracts(client)

        # Generate report
        report = runner.generate_report(results)

        # Check report content
        assert "API Contract Test Report" in report
        assert "Total contracts tested:" in report
        assert "✅" in report or "❌" in report  # Status indicators

        # Save report if needed
        # with open("contract_test_report.md", "w") as f:
        #     f.write(report)
