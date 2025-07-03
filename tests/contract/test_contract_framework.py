"""Test the contract framework itself."""

import pytest

from tests.contract.contract_test_base import (
    ContractTestRunner,
    CreateAgentContract,
    ListAgentsContract,
)


class TestContractFramework:
    """Test the contract framework functionality."""

    @pytest.mark.asyncio
    async def test_contract_with_mock_client(self):
        """Test contract framework with a simple mock client."""

        # Create a simple mock client
        class SimpleMockClient:
            async def get(self, url, **kwargs):
                if "/api/agents" in url:
                    return type(
                        "Response",
                        (),
                        {
                            "json": lambda self: {
                                "items": [],
                                "total": 0,
                                "page": 1,
                                "per_page": 20,
                            },
                            "status_code": 200,
                        },
                    )()
                return type(
                    "Response",
                    (),
                    {"json": lambda self: {"error": "Not found"}, "status_code": 404},
                )()

            async def post(self, url, **kwargs):
                if "/api/agents" in url:
                    return type(
                        "Response",
                        (),
                        {
                            "json": lambda self: {
                                "id": "test-123",
                                "name": "Test Agent",
                                "agent_type": "explorer",
                                "status": "active",
                                "position": {"x": 0, "y": 0, "z": 0},
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": "2024-01-01T00:00:00Z",
                            },
                            "status_code": 201,
                        },
                    )()
                return type(
                    "Response",
                    (),
                    {"json": lambda self: {"error": "Bad request"}, "status_code": 400},
                )()

        # Test with our framework
        runner = ContractTestRunner()
        client = SimpleMockClient()

        # Test list agents contract
        list_contract = ListAgentsContract()
        result = await runner.test_contract(list_contract, client)

        assert (
            result.passed
        ), f"Contract failed with violations: {
            result.violations}"
        assert result.endpoint == "/api/agents"
        assert result.method == "GET"

        # Test create agent contract
        create_contract = CreateAgentContract()
        result = await runner.test_contract(create_contract, client)

        assert (
            result.passed
        ), f"Contract failed with violations: {
            result.violations}"
        assert result.endpoint == "/api/agents"
        assert result.method == "POST"

    @pytest.mark.asyncio
    async def test_contract_violations(self):
        """Test that contract violations are properly detected."""

        # Create a mock client that returns invalid data
        class BadMockClient:
            async def get(self, url, **kwargs):
                # Return data missing required fields - items and total are
                # required
                return type(
                    "Response",
                    (),
                    {
                        # Missing total (required field)
                        "json": lambda self: {"items": []},
                        "status_code": 200,
                    },
                )()

        runner = ContractTestRunner()
        client = BadMockClient()

        contract = ListAgentsContract()
        result = await runner.test_contract(contract, client)

        assert not result.passed
        assert len(result.violations) > 0
        # Should have violations for missing required fields
        violation_fields = [v.field for v in result.violations]
        assert any("total" in field for field in violation_fields)

    @pytest.mark.asyncio
    async def test_contract_with_conftest_client(self, client):
        """Test using the client fixture from conftest."""
        runner = ContractTestRunner()

        # The client fixture should work with our contracts
        contract = ListAgentsContract()
        result = await runner.test_contract(contract, client)

        # Even if it fails, we should get a proper result object
        assert isinstance(result.passed, bool)
        assert result.endpoint == "/api/agents"
        assert result.method == "GET"

        # If it failed, check why
        if not result.passed:
            print(f"Violations: {[(v.field, v.message) for v in result.violations]}")
            print(f"Warnings: {result.warnings}")
