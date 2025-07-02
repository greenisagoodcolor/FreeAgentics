"""
Coalition Packaging System

Handles serialization, compression, and packaging of coalition states for deployment.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StateComponent(Enum):
    """Types of state components that need serialization"""

    AGENT_STATE = "agent_state"
    SHARED_KNOWLEDGE = "shared_knowledge"
    COMMUNICATION_CHANNELS = "communication_channels"
    ENVIRONMENTAL_CONTEXT = "environmental_context"
    COALITION_METADATA = "coalition_metadata"
    RESOURCE_ALLOCATIONS = "resource_allocations"
    ACTIVE_GOALS = "active_goals"
    PERFORMANCE_METRICS = "performance_metrics"
    SECURITY_CREDENTIALS = "security_credentials"
    CONFIGURATION = "configuration"


@dataclass
class SerializationRequirement:
    """Defines requirements for serializing a state component"""

    component_type: StateComponent
    data_format: str  # json, binary, mixed
    compression_enabled: bool = True
    encryption_required: bool = False
    versioning_enabled: bool = True
    validation_rules: List[str] = field(default_factory=list)
    dependencies: List[StateComponent] = field(default_factory=list)

    def validate_data(self, data: Any) -> bool:
        """Validate data against requirement rules"""
        # Basic validation - can be extended
        if not data:
            return False

        # Apply validation rules
        for rule in self.validation_rules:
            if rule == "non_empty" and not data:
                return False
            elif rule == "dict_type" and not isinstance(data, dict):
                return False
            elif rule == "list_type" and not isinstance(data, list):
                return False

        return True


@dataclass
class StateSnapshot:
    """Represents a complete state snapshot of a coalition"""

    coalition_id: str
    timestamp: datetime
    version: str
    components: Dict[StateComponent, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksums: Dict[str, str] = field(default_factory=dict)

    def add_component(self, component_type: StateComponent, data: Any) -> None:
        """Add a state component with automatic checksum"""
        self.components[component_type] = data
        # Calculate checksum
        data_str = json.dumps(
            data, sort_keys=True) if not isinstance(
            data, bytes) else data
        checksum = hashlib.sha256(
            data_str.encode() if isinstance(data_str, str) else data_str
        ).hexdigest()
        self.checksums[component_type.value] = checksum

    def verify_integrity(self) -> bool:
        """Verify integrity of all components using checksums"""
        for component_type, data in self.components.items():
            data_str = json.dumps(
                data, sort_keys=True) if not isinstance(
                data, bytes) else data
            checksum = hashlib.sha256(
                data_str.encode() if isinstance(data_str, str) else data_str
            ).hexdigest()

            if self.checksums.get(component_type.value) != checksum:
                logger.error(f"Checksum mismatch for {component_type.value}")
                return False

        return True


class SerializationRequirementsManager:
    """Manages serialization requirements for coalition packaging"""

    def __init__(self) -> None:
        """Initialize with default requirements"""
        self.requirements = self._define_default_requirements()

    def _define_default_requirements(
        self,
    ) -> Dict[StateComponent, SerializationRequirement]:
        """Define default serialization requirements for each component"""
        return {
            StateComponent.AGENT_STATE: SerializationRequirement(
                component_type=StateComponent.AGENT_STATE,
                data_format="json",
                compression_enabled=True,
                encryption_required=False,
                versioning_enabled=True,
                validation_rules=["non_empty", "dict_type"],
                dependencies=[],
            ),
            StateComponent.SHARED_KNOWLEDGE: SerializationRequirement(
                component_type=StateComponent.SHARED_KNOWLEDGE,
                data_format="mixed",  # Can be JSON or binary (e.g., embeddings)
                compression_enabled=True,
                encryption_required=False,
                versioning_enabled=True,
                validation_rules=["non_empty"],
                dependencies=[StateComponent.AGENT_STATE],
            ),
            StateComponent.COMMUNICATION_CHANNELS: SerializationRequirement(
                component_type=StateComponent.COMMUNICATION_CHANNELS,
                data_format="json",
                compression_enabled=True,
                encryption_required=True,  # Contains sensitive comm data
                versioning_enabled=True,
                validation_rules=["non_empty", "dict_type"],
                dependencies=[],
            ),
            StateComponent.ENVIRONMENTAL_CONTEXT: SerializationRequirement(
                component_type=StateComponent.ENVIRONMENTAL_CONTEXT,
                data_format="json",
                compression_enabled=True,
                encryption_required=False,
                versioning_enabled=True,
                validation_rules=["dict_type"],
                dependencies=[],
            ),
            StateComponent.COALITION_METADATA: SerializationRequirement(
                component_type=StateComponent.COALITION_METADATA,
                data_format="json",
                compression_enabled=False,  # Usually small
                encryption_required=False,
                versioning_enabled=True,
                validation_rules=["non_empty", "dict_type"],
                dependencies=[],
            ),
            StateComponent.RESOURCE_ALLOCATIONS: SerializationRequirement(
                component_type=StateComponent.RESOURCE_ALLOCATIONS,
                data_format="json",
                compression_enabled=True,
                encryption_required=False,
                versioning_enabled=True,
                validation_rules=["dict_type"],
                dependencies=[StateComponent.AGENT_STATE],
            ),
            StateComponent.ACTIVE_GOALS: SerializationRequirement(
                component_type=StateComponent.ACTIVE_GOALS,
                data_format="json",
                compression_enabled=True,
                encryption_required=False,
                versioning_enabled=True,
                validation_rules=["list_type"],
                dependencies=[StateComponent.COALITION_METADATA],
            ),
            StateComponent.PERFORMANCE_METRICS: SerializationRequirement(
                component_type=StateComponent.PERFORMANCE_METRICS,
                data_format="json",
                compression_enabled=True,
                encryption_required=False,
                versioning_enabled=False,  # Point-in-time data
                validation_rules=["dict_type"],
                dependencies=[],
            ),
            StateComponent.SECURITY_CREDENTIALS: SerializationRequirement(
                component_type=StateComponent.SECURITY_CREDENTIALS,
                data_format="binary",
                compression_enabled=False,
                encryption_required=True,  # Always encrypt credentials
                versioning_enabled=True,
                validation_rules=["non_empty"],
                dependencies=[],
            ),
            StateComponent.CONFIGURATION: SerializationRequirement(
                component_type=StateComponent.CONFIGURATION,
                data_format="json",
                compression_enabled=False,
                encryption_required=False,
                versioning_enabled=True,
                validation_rules=["non_empty", "dict_type"],
                dependencies=[],
            ),
        }

    def get_requirement(
            self,
            component_type: StateComponent) -> SerializationRequirement:
        """Get serialization requirement for a component type"""
        return self.requirements.get(component_type)

    def get_serialization_order(self) -> List[StateComponent]:
        """Get the order in which components should be serialized based on
        dependencies"""
        # Topological sort based on dependencies
        visited = set()
        order = []

        def visit(component: StateComponent):
            if component in visited:
                return

            visited.add(component)
            requirement = self.requirements.get(component)

            if requirement:
                for dep in requirement.dependencies:
                    visit(dep)

            order.append(component)

        for component in StateComponent:
            visit(component)

        return order

    def validate_component_data(
            self,
            component_type: StateComponent,
            data: Any) -> bool:
        """Validate component data against its requirements"""
        requirement = self.get_requirement(component_type)
        if not requirement:
            logger.warning(f"No requirement found for {component_type}")
            return True

        return requirement.validate_data(data)

    def generate_state_schema(self) -> Dict[str, Any]:
        """Generate a schema document for state serialization"""
        schema = {"version": "1.0", "components": {}}

        for component_type, requirement in self.requirements.items():
            schema["components"][component_type.value] = {
                "data_format": requirement.data_format,
                "compression": requirement.compression_enabled,
                "encryption": requirement.encryption_required,
                "versioning": requirement.versioning_enabled,
                "validation_rules": requirement.validation_rules,
                "dependencies": [dep.value for dep in requirement.dependencies],
            }

        return schema


@dataclass
class DataStructureInfo:
    """Information about data structures that need preservation"""

    name: str
    type_info: str
    relationships: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    serialization_notes: str = ""


class StateDataCatalog:
    """Catalogs all data structures and relationships for serialization"""

    def __init__(self) -> None:
        """Initialize the data catalog"""
        self.structures = self._define_data_structures()

    def _define_data_structures(self) -> Dict[str, DataStructureInfo]:
        """Define all data structures that need serialization"""
        return {
            # Agent-related structures
            "agent_profile": DataStructureInfo(
                name="agent_profile",
                type_info="Dict[str, Any]",
                relationships=["agent_state", "capabilities", "personality"],
                constraints=["agent_id must be unique", "readiness_score between 0-1"],
                serialization_notes="Include full agent configuration and learned parameters",
            ),
            "agent_memory": DataStructureInfo(
                name="agent_memory",
                type_info="Dict[str, List[Memory]]",
                relationships=["experiences", "beliefs", "patterns"],
                constraints=["timestamp ordering", "memory size limits"],
                serialization_notes="Can be compressed using semantic deduplication",
            ),
            # Knowledge structures
            "knowledge_graph": DataStructureInfo(
                name="knowledge_graph",
                type_info="NetworkX.DiGraph",
                relationships=["nodes", "edges", "metadata"],
                constraints=["DAG structure for certain subgraphs"],
                serialization_notes="Use GraphML or custom format for full preservation",
            ),
            "belief_state": DataStructureInfo(
                name="belief_state",
                type_info="Dict[str, BeliefDistribution]",
                relationships=["agent_id", "belief_values", "confidence"],
                constraints=["probabilities sum to 1", "confidence in [0,1]"],
                serialization_notes="Preserve full probability distributions",
            ),
            # Communication structures
            "message_history": DataStructureInfo(
                name="message_history",
                type_info="List[ConversationMessage]",
                relationships=["sender", "recipient", "thread_id"],
                constraints=["chronological ordering", "referential integrity"],
                serialization_notes="Can be truncated based on recency",
            ),
            "active_conversations": DataStructureInfo(
                name="active_conversations",
                type_info="Dict[str, ConversationState]",
                relationships=["participants", "context", "goals"],
                constraints=["participant availability", "context consistency"],
                serialization_notes="Include full conversation context",
            ),
            # Coalition structures
            "coalition_state": DataStructureInfo(
                name="coalition_state",
                type_info="Coalition",
                relationships=["members", "goals", "resources", "contracts"],
                constraints=["member consensus", "resource availability"],
                serialization_notes="Include all binding agreements and commitments",
            ),
            "shared_goals": DataStructureInfo(
                name="shared_goals",
                type_info="List[CoalitionGoal]",
                relationships=["coalition_id", "subtasks", "dependencies"],
                constraints=["dependency ordering", "resource requirements"],
                serialization_notes="Preserve goal hierarchy and progress",
            ),
            # Resource structures
            "resource_pool": DataStructureInfo(
                name="resource_pool",
                type_info="Dict[str, ResourceAllocation]",
                relationships=["owner", "allocated_to", "constraints"],
                constraints=["non-negative quantities", "allocation limits"],
                serialization_notes="Track both committed and available resources",
            ),
            # Environmental structures
            "spatial_state": DataStructureInfo(
                name="spatial_state",
                type_info="Dict[str, HexCell]",
                relationships=["coordinates", "properties", "occupants"],
                constraints=["valid H3 indices", "occupancy rules"],
                serialization_notes="Can use H3 compression for sparse grids",
            ),
            "world_state": DataStructureInfo(
                name="world_state",
                type_info="WorldState",
                relationships=["spatial_state", "resources", "entities"],
                constraints=["consistency rules", "physics constraints"],
                serialization_notes="Large structure - consider differential updates",
            ),
        }

    def get_structure_info(self, name: str) -> Optional[DataStructureInfo]:
        """Get information about a specific data structure"""
        return self.structures.get(name)

    def get_related_structures(self, name: str) -> List[str]:
        """Get all structures related to a given structure"""
        info = self.get_structure_info(name)
        if not info:
            return []

        related = set(info.relationships)

        # Recursively find all related structures
        to_check = list(related)
        while to_check:
            current = to_check.pop()
            current_info = self.get_structure_info(current)
            if current_info:
                for rel in current_info.relationships:
                    if rel not in related:
                        related.add(rel)
                        to_check.append(rel)

        return list(related)

    def generate_serialization_plan(
            self, components: List[StateComponent]) -> Dict[str, Any]:
        """Generate a detailed serialization plan for given components"""
        plan = {
            "components": {},
            "data_structures": {},
            "relationships": {},
            "estimated_size": {},
        }

        # Map components to data structures
        component_structures = {
            StateComponent.AGENT_STATE: ["agent_profile", "agent_memory"],
            StateComponent.SHARED_KNOWLEDGE: ["knowledge_graph", "belief_state"],
            StateComponent.COMMUNICATION_CHANNELS: [
                "message_history",
                "active_conversations",
            ],
            StateComponent.COALITION_METADATA: ["coalition_state", "shared_goals"],
            StateComponent.RESOURCE_ALLOCATIONS: ["resource_pool"],
            StateComponent.ENVIRONMENTAL_CONTEXT: ["spatial_state", "world_state"],
        }

        for component in components:
            structures = component_structures.get(component, [])
            plan["components"][component.value] = structures

            for struct_name in structures:
                struct_info = self.get_structure_info(struct_name)
                if struct_info:
                    plan["data_structures"][struct_name] = asdict(struct_info)

                    # Add relationships
                    related = self.get_related_structures(struct_name)
                    plan["relationships"][struct_name] = related

                    # Estimate size (placeholder - would need actual implementation)
                    plan["estimated_size"][struct_name] = self._estimate_structure_size(
                        struct_name)

        return plan

    def _estimate_structure_size(self, structure_name: str) -> Dict[str, int]:
        """Estimate the serialized size of a data structure"""
        # Placeholder estimates in bytes
        size_estimates = {
            "agent_profile": {"min": 1024, "max": 10240, "typical": 5120},
            "agent_memory": {"min": 10240, "max": 1048576, "typical": 102400},
            "knowledge_graph": {"min": 5120, "max": 10485760, "typical": 1048576},
            "belief_state": {"min": 2048, "max": 20480, "typical": 10240},
            "message_history": {"min": 1024, "max": 1048576, "typical": 51200},
            "active_conversations": {"min": 512, "max": 10240, "typical": 2048},
            "coalition_state": {"min": 2048, "max": 20480, "typical": 5120},
            "shared_goals": {"min": 1024, "max": 10240, "typical": 5120},
            "resource_pool": {"min": 512, "max": 5120, "typical": 2048},
            "spatial_state": {"min": 10240, "max": 104857600, "typical": 1048576},
            "world_state": {"min": 102400, "max": 1073741824, "typical": 10485760},
        }

        return size_estimates.get(
            structure_name, {
                "min": 1024, "max": 10240, "typical": 5120})
