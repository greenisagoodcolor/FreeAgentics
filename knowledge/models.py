"""Knowledge Graph models with pgvector and H3 support."""

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, String, Text, Index, JSON, Integer, Float
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.base import Base

# Try to import pgvector types
try:
    from pgvector.sqlalchemy import Vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    # Fallback for development without pgvector
    from sqlalchemy import JSON as Vector

    PGVECTOR_AVAILABLE = False


class KnowledgeNode(Base):
    """Represents a node in the knowledge graph."""

    __tablename__ = "kg_nodes"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    graph_id = Column(PGUUID(as_uuid=True), nullable=False)  # For collective KG

    # Node properties
    label = Column(Text, nullable=False)
    node_type = Column(String(50), nullable=False)  # concept, observation, belief, etc.
    properties = Column(JSON, default={})

    # Vector embedding for semantic search
    embedding = Column(Vector(768), nullable=True)  # 768-dim for compatibility

    # H3 geospatial index for location-based queries
    h3_index = Column(BigInteger, nullable=True, index=True)
    h3_resolution = Column(Integer, default=9)  # H3 resolution level

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    agent = relationship("Agent", back_populates="knowledge_nodes")
    edges_from = relationship(
        "KnowledgeEdge", foreign_keys="KnowledgeEdge.source_id", back_populates="source"
    )
    edges_to = relationship(
        "KnowledgeEdge", foreign_keys="KnowledgeEdge.target_id", back_populates="target"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_kg_nodes_agent_graph", "agent_id", "graph_id"),
        Index("idx_kg_nodes_h3", "h3_index"),
        Index("idx_kg_nodes_type", "node_type"),
    )

    if PGVECTOR_AVAILABLE:
        # Vector similarity index (IVFFlat for pgvector)
        __table_args__ += (
            Index(
                "idx_kg_nodes_embedding",
                "embedding",
                postgresql_using="ivfflat",
                postgresql_with={"lists": 100},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )


class KnowledgeEdge(Base):
    """Represents an edge (relationship) between knowledge nodes."""

    __tablename__ = "kg_edges"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(BigInteger, ForeignKey("kg_nodes.id"), nullable=False)
    target_id = Column(BigInteger, ForeignKey("kg_nodes.id"), nullable=False)

    # Edge properties
    relationship_type = Column(
        String(50), nullable=False
    )  # causes, implies, contradicts, etc.
    weight = Column(Float, default=1.0)
    properties = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    source = relationship(
        "KnowledgeNode", foreign_keys=[source_id], back_populates="edges_from"
    )
    target = relationship(
        "KnowledgeNode", foreign_keys=[target_id], back_populates="edges_to"
    )

    # Indexes
    __table_args__ = (
        Index("idx_kg_edges_source_target", "source_id", "target_id"),
        Index("idx_kg_edges_type", "relationship_type"),
    )


class AgentBeliefSnapshot(Base):
    """Stores historical snapshots of agent belief states as vectors."""

    __tablename__ = "agent_belief_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)

    # Belief state as vector
    belief_vector = Column(Vector(768), nullable=False)

    # Metadata
    inference_step = Column(Integer, nullable=False)
    free_energy = Column(Float, nullable=True)
    action_taken = Column(String(50), nullable=True)

    # Location at time of snapshot
    h3_index = Column(BigInteger, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    agent = relationship("Agent", back_populates="belief_snapshots")

    # Indexes
    __table_args__ = (
        Index("idx_belief_snapshots_agent_step", "agent_id", "inference_step"),
        Index("idx_belief_snapshots_h3", "h3_index"),
    )

    if PGVECTOR_AVAILABLE:
        __table_args__ += (
            Index(
                "idx_belief_vector",
                "belief_vector",
                postgresql_using="ivfflat",
                postgresql_with={"lists": 100},
                postgresql_ops={"belief_vector": "vector_cosine_ops"},
            ),
        )


# Update the Agent model to include knowledge graph relationships
def update_agent_model():
    """Add knowledge graph relationships to Agent model."""
    from database.models import Agent

    if not hasattr(Agent, "knowledge_nodes"):
        Agent.knowledge_nodes = relationship(
            "KnowledgeNode", back_populates="agent", cascade="all, delete-orphan"
        )

    if not hasattr(Agent, "belief_snapshots"):
        Agent.belief_snapshots = relationship(
            "AgentBeliefSnapshot", back_populates="agent", cascade="all, delete-orphan"
        )
