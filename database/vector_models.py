"""
Database models for vector storage and similarity search.

This module provides SQLAlchemy models for storing and searching vector embeddings
using pgvector extension, optimized for Active Inference agent memories and knowledge.
"""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class VectorEmbedding(Base):
    """Base model for storing vector embeddings with metadata."""

    __tablename__ = "vector_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Vector data (512-dimensional for typical embeddings)
    embedding = Column(Vector(512), nullable=False)

    # Metadata
    content = Column(Text, nullable=False)  # Original content
    content_type = Column(String(50), nullable=False)  # text, image, audio, etc.
    source_type = Column(String(50), nullable=False)  # agent_memory, knowledge, observation
    source_id = Column(String(100), nullable=True)  # Reference to source entity

    # Temporal information
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Search optimization
    similarity_threshold = Column(Float, default=0.8)  # Minimum similarity for matches
    is_active = Column(Boolean, default=True)

    # Additional metadata as JSON
    metadata = Column(JSONB, nullable=True)

    def __repr__(self):
        return f"<VectorEmbedding(id={self.id}, type={self.content_type})>"


class AgentMemory(Base):
    """Vector-enhanced agent memory storage."""

    __tablename__ = "agent_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(String(100), nullable=False, index=True)

    # Memory content
    memory_type = Column(String(50), nullable=False)  # belief, observation, action, goal
    content = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)

    # Vector embedding for similarity search
    embedding_id = Column(UUID(as_uuid=True), ForeignKey("vector_embeddings.id"), nullable=True)
    embedding = relationship("VectorEmbedding", backref="agent_memories")

    # Temporal context
    timestamp = Column(DateTime, default=datetime.utcnow)
    decay_factor = Column(Float, default=1.0)  # For memory decay over time

    # Hierarchical location using h3 (if available)
    h3_location = Column(String(20), nullable=True, index=True)

    # Active Inference specific
    precision = Column(Float, default=1.0)  # Belief precision
    prediction_error = Column(Float, nullable=True)  # Prediction error

    # Metadata
    tags = Column(JSONB, nullable=True)  # Flexible tagging
    is_consolidated = Column(Boolean, default=False)  # For memory consolidation

    def __repr__(self):
        return f"<AgentMemory(id={self.id}, agent={self.agent_id}, type={self.memory_type})>"


class KnowledgeVector(Base):
    """Vector storage for knowledge graph entities."""

    __tablename__ = "knowledge_vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Knowledge graph reference
    node_id = Column(String(100), nullable=False, index=True)
    node_type = Column(String(50), nullable=False)

    # Vector representation
    embedding_id = Column(UUID(as_uuid=True), ForeignKey("vector_embeddings.id"), nullable=False)
    embedding = relationship("VectorEmbedding", backref="knowledge_vectors")

    # Knowledge metadata
    importance_score = Column(Float, default=1.0)
    access_frequency = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)

    # Hierarchical indexing
    h3_location = Column(String(20), nullable=True, index=True)
    hierarchy_level = Column(Integer, default=0)

    def __repr__(self):
        return f"<KnowledgeVector(id={self.id}, node={self.node_id})>"


class SemanticCluster(Base):
    """Clusters of semantically similar vectors."""

    __tablename__ = "semantic_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Cluster metadata
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    cluster_type = Column(String(50), nullable=False)  # memory, knowledge, mixed

    # Cluster centroid
    centroid = Column(Vector(512), nullable=False)
    radius = Column(Float, nullable=False)  # Cluster radius for membership

    # Statistics
    member_count = Column(Integer, default=0)
    coherence_score = Column(Float, default=1.0)  # Internal cluster coherence

    # Temporal information
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Hierarchical location
    h3_location = Column(String(20), nullable=True, index=True)

    def __repr__(self):
        return f"<SemanticCluster(id={self.id}, name={self.name}, members={self.member_count})>"


class VectorSearchIndex(Base):
    """Optimized indexes for vector similarity search."""

    __tablename__ = "vector_search_indexes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Index metadata
    index_name = Column(String(100), nullable=False, unique=True)
    index_type = Column(String(50), nullable=False)  # ivfflat, hnsw
    dimension = Column(Integer, nullable=False)

    # Search parameters
    similarity_metric = Column(String(20), default="cosine")  # cosine, euclidean, inner_product
    search_k = Column(Integer, default=10)  # Number of results to return

    # Performance statistics
    total_vectors = Column(Integer, default=0)
    avg_search_time_ms = Column(Float, nullable=True)
    last_rebuild = Column(DateTime, nullable=True)

    # Configuration
    build_parameters = Column(JSONB, nullable=True)
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<VectorSearchIndex(name={self.index_name}, type={self.index_type})>"


# Create indexes for optimal performance
def create_vector_indexes():
    """Create database indexes for vector operations."""

    indexes = [
        # Embedding similarity search indexes
        "CREATE INDEX IF NOT EXISTS idx_embeddings_cosine ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_euclidean ON vector_embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);",
        # Agent memory indexes
        "CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_id ON agent_memories (agent_id);",
        "CREATE INDEX IF NOT EXISTS idx_agent_memories_timestamp ON agent_memories (timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_agent_memories_h3 ON agent_memories (h3_location) WHERE h3_location IS NOT NULL;",
        # Knowledge vector indexes
        "CREATE INDEX IF NOT EXISTS idx_knowledge_vectors_node ON knowledge_vectors (node_id);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_vectors_importance ON knowledge_vectors (importance_score DESC);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_vectors_h3 ON knowledge_vectors (h3_location) WHERE h3_location IS NOT NULL;",
        # Composite indexes for common queries
        "CREATE INDEX IF NOT EXISTS idx_embeddings_type_active ON vector_embeddings (content_type, is_active) WHERE is_active = true;",
        "CREATE INDEX IF NOT EXISTS idx_memories_agent_type ON agent_memories (agent_id, memory_type);",
    ]

    return indexes


def create_vector_functions():
    """Create database functions for vector operations."""

    functions = [
        """
        CREATE OR REPLACE FUNCTION find_similar_embeddings(
            query_embedding vector(512),
            similarity_threshold float DEFAULT 0.7,
            max_results integer DEFAULT 10
        )
        RETURNS TABLE(
            id uuid,
            content text,
            similarity float,
            content_type varchar(50)
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT
                e.id,
                e.content,
                1 - (e.embedding <=> query_embedding) as similarity,
                e.content_type
            FROM vector_embeddings e
            WHERE e.is_active = true
                AND 1 - (e.embedding <=> query_embedding) >= similarity_threshold
            ORDER BY e.embedding <=> query_embedding
            LIMIT max_results;
        END;
        $$ LANGUAGE plpgsql;
        """,
        """
        CREATE OR REPLACE FUNCTION find_agent_similar_memories(
            target_agent_id varchar(100),
            query_embedding vector(512),
            similarity_threshold float DEFAULT 0.7,
            max_results integer DEFAULT 10
        )
        RETURNS TABLE(
            memory_id uuid,
            content text,
            memory_type varchar(50),
            similarity float,
            confidence float,
            timestamp timestamp
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT
                am.id as memory_id,
                am.content,
                am.memory_type,
                1 - (ve.embedding <=> query_embedding) as similarity,
                am.confidence,
                am.timestamp
            FROM agent_memories am
            JOIN vector_embeddings ve ON am.embedding_id = ve.id
            WHERE am.agent_id = target_agent_id
                AND ve.is_active = true
                AND 1 - (ve.embedding <=> query_embedding) >= similarity_threshold
            ORDER BY ve.embedding <=> query_embedding
            LIMIT max_results;
        END;
        $$ LANGUAGE plpgsql;
        """,
        """
        CREATE OR REPLACE FUNCTION update_cluster_centroid(
            cluster_id uuid
        )
        RETURNS void AS $$
        DECLARE
            new_centroid vector(512);
            member_count integer;
        BEGIN
            -- Calculate new centroid from cluster members
            SELECT
                AVG(ve.embedding)::vector(512),
                COUNT(*)
            INTO new_centroid, member_count
            FROM semantic_clusters sc
            JOIN vector_embeddings ve ON (ve.metadata->>'cluster_id')::uuid = sc.id
            WHERE sc.id = cluster_id AND ve.is_active = true;

            -- Update cluster with new centroid
            UPDATE semantic_clusters
            SET centroid = new_centroid,
                member_count = member_count,
                updated_at = NOW()
            WHERE id = cluster_id;
        END;
        $$ LANGUAGE plpgsql;
        """,
    ]

    return functions
