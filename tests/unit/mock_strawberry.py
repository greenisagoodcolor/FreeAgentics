"""
Mock strawberry module for testing GraphQL schema without external dependencies
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union


# Mock ID type
class ID(str):
    """Mock Strawberry ID type"""

    pass


# Mock decorators
def type(cls):
    """Mock @strawberry.type decorator"""
    return dataclass(cls)


def input(cls):
    """Mock @strawberry.input decorator"""
    return dataclass(cls)


def enum(cls):
    """Mock @strawberry.enum decorator"""
    return cls


def field(resolver: Optional[Callable] = None, **kwargs):
    """Mock @strawberry.field decorator"""
    if resolver:
        return resolver

    def decorator(func):
        return func

    return decorator


def mutation(func):
    """Mock @strawberry.mutation decorator"""
    return func


def subscription(func):
    """Mock @strawberry.subscription decorator"""
    return func


# Mock types
class Info:
    """Mock GraphQL Info context"""

    def __init__(self):
        self.context = {}
        self.field_name = "test_field"
        self.parent_type = None


# Mock JSON scalar
JSON = Dict[str, Any]


# Mock test client
class GraphQLTestClient:
    """Mock GraphQL test client"""

    def __init__(self, schema):
        self.schema = schema


# Mock execution result
@dataclass
class ExecutionResult:
    """Mock GraphQL execution result"""

    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[Any]] = None


# Mock schema
class Schema:
    """Mock GraphQL schema"""

    def __init__(self, query=None, mutation=None, subscription=None):
        self.query_type = query
        self.mutation_type = mutation
        self.subscription_type = subscription

    async def execute(self, query: str, variable_values: Optional[Dict] = None):
        """Mock execute method"""
        # Simple mock implementation
        if "invalidField" in query:
            return ExecutionResult(
                data=None, errors=[{"message": "Field 'invalidField' doesn't exist"}]
            )

        # Mock introspection
        if "__schema" in query:
            return ExecutionResult(
                data={
                    "__schema": {
                        "queryType": {
                            "name": "Query",
                            "fields": [
                                {"name": "agent", "type": {"name": "Agent"}},
                                {"name": "agents", "type": {"name": "AgentList"}},
                                {"name": "coalition", "type": {"name": "Coalition"}},
                                {"name": "coalitions", "type": {"name": "CoalitionList"}},
                                {"name": "worldState", "type": {"name": "WorldState"}},
                                {
                                    "name": "simulationMetrics",
                                    "type": {"name": "SimulationMetrics"},
                                },
                            ],
                        }
                    }
                }
            )

        # Mock worldState query
        if "worldState" in query:
            return ExecutionResult(
                data={"worldState": {"totalAgents": 0, "totalCoalitions": 0, "activeAgents": 0}}
            )

        # Mock simulationMetrics query
        if "simulationMetrics" in query:
            return ExecutionResult(
                data={
                    "simulationMetrics": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "fps": 60.0,
                        "agentCount": 0,
                        "coalitionCount": 0,
                        "totalInteractions": 0,
                    }
                }
            )

        # Mock createAgent mutation
        if "createAgent" in query:
            return ExecutionResult(
                data={
                    "createAgent": {
                        "id": "1",
                        "name": "Test Agent",
                        "agentClass": "EXPLORER",
                        "energy": 100.0,
                    }
                }
            )

        # Mock agent query
        if "agent(" in query:
            return ExecutionResult(data={"agent": None})

        # Default response
        return ExecutionResult(data={})


# Mock scalars module
class scalars:
    JSON = JSON


# Mock types module
class types:
    Info = Info
    ExecutionResult = ExecutionResult


# Mock test module
class test:
    GraphQLTestClient = GraphQLTestClient
