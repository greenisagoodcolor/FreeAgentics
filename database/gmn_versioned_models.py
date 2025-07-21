"""Enhanced GMN storage models with version tracking and efficient querying.

Following TDD principles and YAGNI, this module extends the existing GMN storage
with minimal essential features for version tracking, rollback, and performance.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base
from database.types import GUID

if TYPE_CHECKING:
    from database.models import Agent


class GMNVersionStatus(PyEnum):
    """Version status enumeration for GMN specifications."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class GMNVersionedSpecification(Base):
    """Enhanced GMN specification model with version tracking.

    This model extends the existing GMN storage with version tracking,
    parent-child relationships, and performance optimizations.

    Key design decisions (following YAGNI):
    - Version lineage through parent_version_id (minimal tree structure)
    - JSON storage for parsed data (efficient for complex structures)
    - Essential indexes for performance (agent_id, version_number, status)
    - Checksum for integrity verification
    - Simple rollback support through version chain
    """

    __tablename__ = "gmn_versioned_specifications"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    agent_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("agents.id"), nullable=False
    )

    # Version tracking (core enhancement)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parent_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("gmn_versioned_specifications.id"),
        nullable=True,
    )

    # Specification content
    specification_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_specification: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Metadata
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version_tag: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # Human-readable version like "v1.2.3"

    # Version metadata
    version_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )  # Change summary, author, etc.

    # Status and validation
    status: Mapped[GMNVersionStatus] = mapped_column(
        Enum(GMNVersionStatus), default=GMNVersionStatus.DRAFT
    )

    # Data integrity
    specification_checksum: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # SHA-256 of specification_text

    # Performance metrics (minimal YAGNI approach)
    node_count: Mapped[int] = mapped_column(Integer, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, default=0)
    complexity_score: Mapped[Optional[float]] = mapped_column(
        Integer, nullable=True
    )  # Simple metric for querying by complexity

    # Rollback support
    rollback_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )  # Stores rollback information if this version was created via rollback

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deprecated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", foreign_keys=[agent_id])

    # Parent-child version relationships
    parent_version: Mapped[Optional["GMNVersionedSpecification"]] = relationship(
        "GMNVersionedSpecification",
        remote_side=[id],
        back_populates="child_versions",
    )
    child_versions: Mapped[List["GMNVersionedSpecification"]] = relationship(
        "GMNVersionedSpecification", back_populates="parent_version"
    )

    # Constraints and indexes for performance
    __table_args__ = (
        # Unique constraint: one active version per agent
        Index(
            "idx_gmn_versioned_agent_active_unique",
            "agent_id",
            unique=True,
            postgresql_where="status = 'active'",
        ),
        # Performance indexes
        Index("idx_gmn_versioned_agent_version", "agent_id", "version_number"),
        Index("idx_gmn_versioned_status", "status"),
        Index("idx_gmn_versioned_created_at", "created_at"),
        Index("idx_gmn_versioned_name", "name"),
        Index("idx_gmn_versioned_complexity", "complexity_score"),
        Index("idx_gmn_versioned_parent", "parent_version_id"),
        # Checksum index for integrity queries
        Index("idx_gmn_versioned_checksum", "specification_checksum"),
        # Composite indexes for common queries
        Index(
            "idx_gmn_versioned_agent_status_version",
            "agent_id",
            "status",
            "version_number",
        ),
        Index("idx_gmn_versioned_agent_created", "agent_id", "created_at"),
        # Data integrity constraints
        CheckConstraint("version_number > 0", name="ck_gmn_version_positive"),
        CheckConstraint("node_count >= 0", name="ck_gmn_node_count_non_negative"),
        CheckConstraint("edge_count >= 0", name="ck_gmn_edge_count_non_negative"),
        CheckConstraint(
            "complexity_score IS NULL OR (complexity_score >= 0.0 AND complexity_score <= 1.0)",
            name="ck_gmn_complexity_range",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "version_number": self.version_number,
            "parent_version_id": (
                str(self.parent_version_id) if self.parent_version_id else None
            ),
            "name": self.name,
            "description": self.description,
            "version_tag": self.version_tag,
            "status": self.status.value,
            "specification_text": self.specification_text,
            "parsed_specification": self.parsed_specification,
            "version_metadata": self.version_metadata,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "complexity_score": self.complexity_score,
            "specification_checksum": self.specification_checksum,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "activated_at": (
                self.activated_at.isoformat() if self.activated_at else None
            ),
            "deprecated_at": (
                self.deprecated_at.isoformat() if self.deprecated_at else None
            ),
        }

    def is_compatible_with(self, other: "GMNVersionedSpecification") -> bool:
        """Check compatibility with another version (basic structure check)."""
        if not other.parsed_specification or not self.parsed_specification:
            return False

        # Basic compatibility: same number of nodes and similar structure
        self_nodes = self.parsed_specification.get("nodes", [])
        other_nodes = other.parsed_specification.get("nodes", [])

        if len(self_nodes) != len(other_nodes):
            return False

        # Check node names and types match
        self_node_sigs = {
            (node.get("name"), node.get("type"))
            for node in self_nodes
            if "name" in node and "type" in node
        }
        other_node_sigs = {
            (node.get("name"), node.get("type"))
            for node in other_nodes
            if "name" in node and "type" in node
        }

        return self_node_sigs == other_node_sigs


class GMNVersionTransition(Base):
    """Model for tracking version transitions and changes.

    This lightweight model tracks transitions between versions
    for audit trail and rollback support.
    """

    __tablename__ = "gmn_version_transitions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    agent_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("agents.id"), nullable=False
    )
    from_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("gmn_versioned_specifications.id"),
        nullable=True,
    )
    to_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("gmn_versioned_specifications.id"),
        nullable=False,
    )

    # Transition metadata
    transition_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "create", "update", "rollback", "activate", "deprecate"
    transition_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Change summary (minimal diff information)
    changes_summary: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", foreign_keys=[agent_id])
    from_version: Mapped[Optional["GMNVersionedSpecification"]] = relationship(
        "GMNVersionedSpecification", foreign_keys=[from_version_id]
    )
    to_version: Mapped["GMNVersionedSpecification"] = relationship(
        "GMNVersionedSpecification", foreign_keys=[to_version_id]
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_gmn_transition_agent", "agent_id"),
        Index("idx_gmn_transition_from_version", "from_version_id"),
        Index("idx_gmn_transition_to_version", "to_version_id"),
        Index("idx_gmn_transition_type", "transition_type"),
        Index("idx_gmn_transition_created", "created_at"),
        Index("idx_gmn_transition_agent_created", "agent_id", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "from_version_id": (
                str(self.from_version_id) if self.from_version_id else None
            ),
            "to_version_id": str(self.to_version_id),
            "transition_type": self.transition_type,
            "transition_reason": self.transition_reason,
            "changes_summary": self.changes_summary,
            "created_at": self.created_at.isoformat(),
        }


# Migration/evolution note:
# This schema is designed to work alongside the existing GMNSpecification model
# during a transition period. The enhanced repository can use both models
# with a gradual migration strategy.
