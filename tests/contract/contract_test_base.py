"""
Contract Testing Framework for FreeAgentics.

This module provides a base framework for API contract testing,
ensuring backward compatibility and API stability.

Following contract testing best practices from Martin Fowler and consumer-driven contracts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

import pytest
from pydantic import BaseModel, ValidationError


@dataclass
class ContractViolation:
    """Represents a contract violation."""

    field: str
    expected: Any
    actual: Any
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ContractTestResult:
    """Result of a contract test."""

    passed: bool
    endpoint: str
    method: str
    violations: List[ContractViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ContractSchema(BaseModel):
    """Base class for contract schemas using Pydantic."""

    class Config:
        extra = "forbid"  # Strict mode - no extra fields allowed
        validate_assignment = True

    @classmethod
    def validate_contract(
            cls, data: Dict[str, Any]) -> List[ContractViolation]:
        """Validate data against contract schema."""
        violations = []
        try:
            cls(**data)
        except ValidationError as e:
            for error in e.errors():
                violation = ContractViolation(
                    field=".".join(str(loc) for loc in error["loc"]),
                    expected=error.get("type"),
                    actual=error.get("input"),
                    message=error["msg"],
                )
                violations.append(violation)
        return violations


class APIContract(ABC):
    """Base class for API contracts."""

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """API endpoint path."""

    @property
    @abstractmethod
    def method(self) -> str:
        """HTTP method (GET, POST, etc.)."""

    @property
    @abstractmethod
    def request_schema(self) -> Optional[Type[ContractSchema]]:
        """Request body schema."""

    @property
    @abstractmethod
    def response_schema(self) -> Type[ContractSchema]:
        """Response body schema."""

    @property
    def expected_status_codes(self) -> List[int]:
        """Expected status codes for success."""
        return [200, 201]

    def validate_request(self,
                         request_data: Dict[str,
                                            Any]) -> List[ContractViolation]:
        """Validate request data against contract."""
        if self.request_schema is None:
            return []
        return self.request_schema.validate_contract(request_data)

    def validate_response(
        self, response_data: Dict[str, Any], status_code: int
    ) -> ContractTestResult:
        """Validate response data against contract."""
        violations = []
        warnings = []

        # Check status code
        if status_code not in self.expected_status_codes:
            warnings.append(
                f"Unexpected status code: {status_code}. Expected: {
                    self.expected_status_codes}")

        # Validate response schema
        violations.extend(
            self.response_schema.validate_contract(response_data))

        return ContractTestResult(
            passed=len(violations) == 0,
            endpoint=self.endpoint,
            method=self.method,
            violations=violations,
            warnings=warnings,
        )


class ContractTestBase:
    """Base class for contract tests."""

    def __init__(self):
        self.contracts: List[APIContract] = []
        self.results: List[ContractTestResult] = []

    def register_contract(self, contract: APIContract):
        """Register a contract for testing."""
        self.contracts.append(contract)

    async def test_contract(
            self,
            contract: APIContract,
            client: Any) -> ContractTestResult:
        """Test a single contract."""
        # This is a base implementation - override in subclasses
        raise NotImplementedError("Subclasses must implement test_contract")

    async def test_all_contracts(
            self, client: Any) -> List[ContractTestResult]:
        """Test all registered contracts."""
        results = []
        for contract in self.contracts:
            result = await self.test_contract(contract, client)
            results.append(result)
        return results

    def generate_report(self, results: List[ContractTestResult]) -> str:
        """Generate a contract test report."""
        report = ["# API Contract Test Report", ""]
        report.append(f"Generated at: {datetime.now().isoformat()}")
        report.append(f"Total contracts tested: {len(results)}")
        report.append("")

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        report.append(f"✅ Passed: {passed}")
        report.append(f"❌ Failed: {failed}")
        report.append("")

        # Detailed results
        for result in results:
            status = "✅" if result.passed else "❌"
            report.append(f"## {status} {result.method} {result.endpoint}")

            if result.violations:
                report.append("### Violations:")
                for violation in result.violations:
                    report.append(
                        f"- **{violation.field}**: {violation.message}")
                    report.append(f"  - Expected: {violation.expected}")
                    report.append(f"  - Actual: {violation.actual}")

            if result.warnings:
                report.append("### Warnings:")
                for warning in result.warnings:
                    report.append(f"- {warning}")

            report.append("")

        return "\n".join(report)


class ContractRegistry:
    """Registry for API contracts with versioning support."""

    def __init__(self):
        self._contracts: Dict[str, Dict[str, APIContract]] = {}
        self._versions: Dict[str, List[str]] = {}

    def register(self, contract: APIContract, version: str = "v1"):
        """Register a contract with version."""
        key = f"{contract.method}:{contract.endpoint}"

        if key not in self._contracts:
            self._contracts[key] = {}
            self._versions[key] = []

        self._contracts[key][version] = contract
        if version not in self._versions[key]:
            self._versions[key].append(version)

    def get_contract(
        self, method: str, endpoint: str, version: str = "v1"
    ) -> Optional[APIContract]:
        """Get a specific contract."""
        key = f"{method}:{endpoint}"
        return self._contracts.get(key, {}).get(version)

    def get_all_versions(self, method: str, endpoint: str) -> List[str]:
        """Get all versions of a contract."""
        key = f"{method}:{endpoint}"
        return self._versions.get(key, [])

    def get_all_contracts(
            self,
            version: Optional[str] = None) -> List[APIContract]:
        """Get all contracts, optionally filtered by version."""
        contracts = []
        for endpoint_contracts in self._contracts.values():
            if version:
                if version in endpoint_contracts:
                    contracts.append(endpoint_contracts[version])
            else:
                contracts.extend(endpoint_contracts.values())
        return contracts


# Contract validation decorators
def contract_test(contract_class: Type[APIContract]):
    """Decorator to mark a test method as a contract test."""

    def decorator(test_func):
        test_func._contract = contract_class
        return test_func

    return decorator


def validate_response_contract(schema: Type[ContractSchema]):
    """Decorator to validate response against contract schema."""

    def decorator(test_func):
        async def wrapper(*args, **kwargs):
            response = await test_func(*args, **kwargs)

            # Extract response data
            if hasattr(response, "json"):
                data = response.json()
            elif isinstance(response, dict):
                data = response
            else:
                raise ValueError(
                    f"Cannot extract data from response: {
                        type(response)}")

            # Validate
            violations = schema.validate_contract(data)
            if violations:
                pytest.fail(f"Contract violations: {violations}")

            return response

        return wrapper

    return decorator


# Example contract schemas
class AgentCreateRequestSchema(ContractSchema):
    """Schema for agent creation request."""

    name: str
    agent_type: str
    initial_position: Optional[List[float]] = None
    personality: Optional[Dict[str, float]] = None
    constraints: Optional[Dict[str, Any]] = None


class AgentResponseSchema(ContractSchema):
    """Schema for agent response."""

    id: str
    name: str
    agent_type: str
    status: str
    position: Dict[str, float]
    created_at: str
    updated_at: str


class CoalitionCreateRequestSchema(ContractSchema):
    """Schema for coalition creation request."""

    name: str
    member_ids: List[str]
    business_type: str
    rules: Optional[Dict[str, Any]] = None


class CoalitionResponseSchema(ContractSchema):
    """Schema for coalition response."""

    id: str
    name: str
    members: List[str]
    business_type: str
    status: str
    formation_time: str
    synergy_score: float


class ErrorResponseSchema(ContractSchema):
    """Schema for error responses."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str


# Example contract implementations
class CreateAgentContract(APIContract):
    """Contract for agent creation endpoint."""

    @property
    def endpoint(self) -> str:
        return "/api/agents"

    @property
    def method(self) -> str:
        return "POST"

    @property
    def request_schema(self) -> Type[ContractSchema]:
        return AgentCreateRequestSchema

    @property
    def response_schema(self) -> Type[ContractSchema]:
        return AgentResponseSchema

    @property
    def expected_status_codes(self) -> List[int]:
        return [201]


class GetAgentContract(APIContract):
    """Contract for getting agent by ID."""

    def __init__(self, agent_id: str = "{id}"):
        self.agent_id = agent_id

    @property
    def endpoint(self) -> str:
        return f"/api/agents/{self.agent_id}"

    @property
    def method(self) -> str:
        return "GET"

    @property
    def request_schema(self) -> None:
        return None

    @property
    def response_schema(self) -> Type[ContractSchema]:
        return AgentResponseSchema


class ListAgentsContract(APIContract):
    """Contract for listing agents."""

    @property
    def endpoint(self) -> str:
        return "/api/agents"

    @property
    def method(self) -> str:
        return "GET"

    @property
    def request_schema(self) -> None:
        return None

    @property
    def response_schema(self) -> Type[ContractSchema]:
        # For list endpoints, we need a wrapper schema
        class AgentListResponseSchema(ContractSchema):
            items: List[AgentResponseSchema]
            total: int
            page: int = 1
            per_page: int = 20

        return AgentListResponseSchema


# Contract test runner
class ContractTestRunner(ContractTestBase):
    """Runner for contract tests with mock client support."""

    async def test_contract(
            self,
            contract: APIContract,
            client: Any) -> ContractTestResult:
        """Test a contract against a client."""
        # Prepare request
        url = contract.endpoint
        method = contract.method.lower()

        # Get client method - handle both sync and async methods
        client_method = getattr(client, method, None)

        # Check if it's callable (could be a regular method or async method)
        if client_method is None:
            return ContractTestResult(
                passed=False,
                endpoint=contract.endpoint,
                method=contract.method,
                violations=[
                    ContractViolation(
                        field="method",
                        expected=contract.method,
                        actual="not found",
                        message=f"Client does not have method '{method}'",
                    )
                ],
            )

        # For bound methods, check if they're callable
        if not (callable(client_method) or hasattr(client_method, "__call__")):
            return ContractTestResult(
                passed=False,
                endpoint=contract.endpoint,
                method=contract.method,
                violations=[
                    ContractViolation(
                        field="method",
                        expected=contract.method,
                        actual="not callable",
                        message=f"Client method '{method}' is not callable",
                    )
                ],
            )

        # Make request
        try:
            if contract.request_schema and method in ["post", "put", "patch"]:
                # Generate sample request data
                sample_data = self._generate_sample_data(
                    contract.request_schema)
                response = await client_method(url, json=sample_data)
            else:
                response = await client_method(url)

            # Extract response data - handle different response types
            if hasattr(response, "json"):
                # Real HTTP response object
                if callable(response.json):
                    response_data = response.json()
                else:
                    response_data = response.json
                status_code = getattr(response, "status_code", 200)
            elif isinstance(response, dict):
                # Direct dict response from mock
                response_data = response
                status_code = 200
            else:
                # Try to extract json method from mock response
                if hasattr(response, "json") and callable(response.json):
                    response_data = response.json()
                    status_code = getattr(response, "status_code", 200)
                else:
                    response_data = {"error": "Invalid response format"}
                    status_code = 500

            # Validate response
            return contract.validate_response(response_data, status_code)

        except Exception as e:
            return ContractTestResult(
                passed=False,
                endpoint=contract.endpoint,
                method=contract.method,
                violations=[
                    ContractViolation(
                        field="request",
                        expected="successful request",
                        actual=str(e),
                        message=f"Request failed: {str(e)}",
                    )
                ],
            )

    def _generate_sample_data(
            self, schema: Type[ContractSchema]) -> Dict[str, Any]:
        """Generate sample data for a schema."""
        # This is a simple implementation - could be enhanced
        sample_data = {}

        # Pydantic v2 uses model_fields
        if hasattr(schema, "model_fields"):
            for field_name, field_info in schema.model_fields.items():
                # In Pydantic v2, required is determined by whether field has
                # default
                is_required = field_info.is_required()

                if is_required:
                    # Generate based on type annotation
                    field_type = field_info.annotation

                    # Handle Optional types
                    if hasattr(field_type, "__origin__"):
                        if field_type.__origin__ is Union:
                            # Get the non-None type from Optional[T]
                            args = field_type.__args__
                            field_type = next(
                                (arg for arg in args if arg is not type(None)), str)

                    if field_type == str:
                        sample_data[field_name] = f"test_{field_name}"
                    elif field_type == int:
                        sample_data[field_name] = 1
                    elif field_type == float:
                        sample_data[field_name] = 1.0
                    elif field_type == bool:
                        sample_data[field_name] = True
                    elif field_type == list or (
                        hasattr(field_type, "__origin__") and field_type.__origin__ is list
                    ):
                        sample_data[field_name] = []
                    elif field_type == dict or (
                        hasattr(field_type, "__origin__") and field_type.__origin__ is dict
                    ):
                        sample_data[field_name] = {}
                    else:
                        # Default to string for unknown types
                        sample_data[field_name] = f"test_{field_name}"

        return sample_data


# Global contract registry
contract_registry = ContractRegistry()

# Register default contracts
contract_registry.register(CreateAgentContract(), "v1")
contract_registry.register(GetAgentContract(), "v1")
contract_registry.register(ListAgentsContract(), "v1")
