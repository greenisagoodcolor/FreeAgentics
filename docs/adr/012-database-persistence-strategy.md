# ADR-012: Database and Persistence Strategy

## Status
Accepted

## Context
FreeAgentics requires a robust persistence strategy to handle agent state, coalition data, world simulations, and business intelligence. The system must support high-frequency updates, complex queries, real-time analytics, and horizontal scaling while maintaining data consistency and performance.

## Decision
We will implement a polyglot persistence architecture using PostgreSQL as the primary operational database, Redis for caching and real-time data, and InfluxDB for time-series metrics, with clear data modeling patterns for each storage type.

## Persistence Architecture

### 1. Primary Database: PostgreSQL

#### Core Data Models
```sql
-- agents/models.sql
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Agent state (encrypted JSON)
    belief_state BYTEA,
    personality_traits JSONB,
    energy_level INTEGER DEFAULT 100,
    location_h3 VARCHAR(15),

    -- Status tracking
    status VARCHAR(20) DEFAULT 'active',
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_energy CHECK (energy_level >= 0 AND energy_level <= 100),
    CONSTRAINT valid_status CHECK (status IN ('active', 'inactive', 'suspended'))
);

CREATE INDEX idx_agents_tenant_type ON agents(tenant_id, agent_type);
CREATE INDEX idx_agents_location ON agents(location_h3);
CREATE INDEX idx_agents_last_active ON agents(last_active_at);
```

```sql
-- coalitions/models.sql
CREATE TABLE coalitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    business_model VARCHAR(100) NOT NULL,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Coalition configuration
    formation_criteria JSONB,
    business_config JSONB,

    -- Financial tracking
    initial_capital DECIMAL(15,2) DEFAULT 0,
    current_capital DECIMAL(15,2) DEFAULT 0,
    total_revenue DECIMAL(15,2) DEFAULT 0,

    -- Status
    status VARCHAR(20) DEFAULT 'forming',
    dissolved_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_status CHECK (
        status IN ('forming', 'active', 'profitable', 'dissolved')
    )
);

CREATE TABLE coalition_members (
    coalition_id UUID REFERENCES coalitions(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    role VARCHAR(50) DEFAULT 'member',
    contribution_score DECIMAL(5,2) DEFAULT 0,

    PRIMARY KEY (coalition_id, agent_id)
);

CREATE INDEX idx_coalition_members_agent ON coalition_members(agent_id);
```

#### Data Access Layer
```python
# infrastructure/database/repositories/agent_repository.py
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from cryptography.fernet import Fernet

class AgentRepository:
    """Repository pattern for agent data access."""

    def __init__(self, session: Session, encryption_key: bytes):
        self.session = session
        self.cipher = Fernet(encryption_key)

    async def save_agent(self, agent: Agent) -> Agent:
        """Save agent with encrypted belief state."""

        # Encrypt sensitive data
        encrypted_beliefs = self.cipher.encrypt(
            json.dumps(agent.beliefs.tolist()).encode()
        )

        db_agent = AgentModel(
            id=agent.id,
            name=agent.name,
            agent_type=agent.agent_type,
            tenant_id=agent.tenant_id,
            belief_state=encrypted_beliefs,
            personality_traits=agent.personality.to_dict(),
            energy_level=agent.energy,
            location_h3=agent.location,
            status=agent.status
        )

        self.session.add(db_agent)
        await self.session.commit()
        return agent

    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Retrieve agent with decrypted belief state."""

        db_agent = await self.session.get(AgentModel, agent_id)
        if not db_agent:
            return None

        # Decrypt belief state
        decrypted_beliefs = json.loads(
            self.cipher.decrypt(db_agent.belief_state).decode()
        )

        return Agent(
            id=db_agent.id,
            name=db_agent.name,
            agent_type=db_agent.agent_type,
            beliefs=np.array(decrypted_beliefs),
            personality=Personality.from_dict(db_agent.personality_traits),
            energy=db_agent.energy_level,
            location=db_agent.location_h3
        )

    async def get_agents_by_location(
        self,
        location: str,
        radius: int = 3
    ) -> List[Agent]:
        """Get agents within H3 cell radius."""

        # Use H3 spatial query
        nearby_cells = h3.k_ring(location, radius)

        query = self.session.query(AgentModel).filter(
            AgentModel.location_h3.in_(nearby_cells),
            AgentModel.status == 'active'
        )

        db_agents = await query.all()
        return [self._db_to_domain(agent) for agent in db_agents]
```

### 2. Caching Layer: Redis

#### Cache Strategy
```python
# infrastructure/cache/redis_cache.py
import redis.asyncio as redis
import json
from typing import Optional, List

class AgentCache:
    """Redis-based caching for agent data."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes

    async def cache_agent_beliefs(
        self,
        agent_id: str,
        beliefs: np.ndarray,
        ttl: int = None
    ):
        """Cache agent belief state for fast access."""
        key = f"agent:beliefs:{agent_id}"
        value = json.dumps(beliefs.tolist())

        await self.redis.setex(
            key,
            ttl or self.default_ttl,
            value
        )

    async def get_cached_beliefs(
        self,
        agent_id: str
    ) -> Optional[np.ndarray]:
        """Retrieve cached belief state."""
        key = f"agent:beliefs:{agent_id}"
        cached = await self.redis.get(key)

        if cached:
            beliefs_list = json.loads(cached)
            return np.array(beliefs_list)

        return None

    async def cache_coalition_members(
        self,
        coalition_id: str,
        member_ids: List[str]
    ):
        """Cache coalition membership for fast lookups."""
        key = f"coalition:members:{coalition_id}"

        # Use Redis set for O(1) membership checks
        await self.redis.delete(key)
        if member_ids:
            await self.redis.sadd(key, *member_ids)
            await self.redis.expire(key, 600)  # 10 minutes

    async def is_coalition_member(
        self,
        coalition_id: str,
        agent_id: str
    ) -> bool:
        """Check if agent is coalition member (cached)."""
        key = f"coalition:members:{coalition_id}"
        return await self.redis.sismember(key, agent_id)
```

#### Real-time Updates
```python
# infrastructure/cache/real_time_updates.py
class RealTimeUpdates:
    """Redis pub/sub for real-time agent updates."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def publish_agent_update(
        self,
        agent_id: str,
        update_type: str,
        data: dict
    ):
        """Publish agent state change."""
        message = {
            'agent_id': agent_id,
            'type': update_type,
            'data': data,
            'timestamp': time.time()
        }

        channel = f"agent:updates:{agent_id}"
        await self.redis.publish(channel, json.dumps(message))

        # Also publish to general updates channel
        await self.redis.publish(
            "agent:updates:all",
            json.dumps(message)
        )

    async def subscribe_to_agent_updates(
        self,
        agent_id: str
    ) -> redis.client.PubSub:
        """Subscribe to specific agent updates."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"agent:updates:{agent_id}")
        return pubsub
```

### 3. Time-Series Database: InfluxDB

#### Metrics Storage
```python
# infrastructure/metrics/influx_metrics.py
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class MetricsCollector:
    """InfluxDB-based metrics collection for agents and coalitions."""

    def __init__(self, client: InfluxDBClient):
        self.client = client
        self.write_api = client.write_api(write_options=SYNCHRONOUS)
        self.bucket = "freeagentics_metrics"
        self.org = "freeagentics"

    def record_agent_belief_update(
        self,
        agent_id: str,
        beliefs: np.ndarray,
        free_energy: float
    ):
        """Record agent belief state change."""
        point = (
            Point("agent_beliefs")
            .tag("agent_id", agent_id)
            .field("free_energy", free_energy)
            .field("belief_entropy", -np.sum(beliefs * np.log(beliefs + 1e-10)))
            .field("max_belief", float(np.max(beliefs)))
            .field("belief_variance", float(np.var(beliefs)))
        )

        # Add individual belief dimensions
        for i, belief in enumerate(beliefs):
            point = point.field(f"belief_{i}", float(belief))

        self.write_api.write(
            bucket=self.bucket,
            org=self.org,
            record=point
        )

    def record_coalition_performance(
        self,
        coalition_id: str,
        member_count: int,
        revenue: float,
        efficiency: float
    ):
        """Record coalition business metrics."""
        point = (
            Point("coalition_performance")
            .tag("coalition_id", coalition_id)
            .field("member_count", member_count)
            .field("revenue", revenue)
            .field("efficiency", efficiency)
            .field("revenue_per_member", revenue / member_count if member_count > 0 else 0)
        )

        self.write_api.write(
            bucket=self.bucket,
            org=self.org,
            record=point
        )

    def query_agent_performance_history(
        self,
        agent_id: str,
        hours: int = 24
    ) -> List[dict]:
        """Query agent performance over time."""
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -{hours}h)
        |> filter(fn: (r) => r._measurement == "agent_beliefs")
        |> filter(fn: (r) => r.agent_id == "{agent_id}")
        |> aggregateWindow(every: 1h, fn: mean)
        '''

        result = self.client.query_api().query(org=self.org, query=query)

        metrics = []
        for table in result:
            for record in table.records:
                metrics.append({
                    'time': record.get_time(),
                    'field': record.get_field(),
                    'value': record.get_value()
                })

        return metrics
```

### 4. Data Migration and Versioning

#### Schema Management
```python
# infrastructure/database/migrations/migration_manager.py
from alembic import command
from alembic.config import Config
import importlib

class MigrationManager:
    """Manages database schema migrations and data versioning."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.alembic_cfg = Config("alembic.ini")
        self.alembic_cfg.set_main_option("sqlalchemy.url", database_url)

    def run_migrations(self):
        """Run pending database migrations."""
        command.upgrade(self.alembic_cfg, "head")

    def create_migration(self, message: str):
        """Create new migration from model changes."""
        command.revision(
            self.alembic_cfg,
            message=message,
            autogenerate=True
        )

    def rollback_migration(self, revision: str = "-1"):
        """Rollback to previous migration."""
        command.downgrade(self.alembic_cfg, revision)
```

#### Data Backup Strategy
```python
# infrastructure/backup/backup_manager.py
import asyncio
import subprocess
from datetime import datetime

class BackupManager:
    """Manages automated database backups and recovery."""

    def __init__(self, config: dict):
        self.postgres_config = config['postgres']
        self.redis_config = config['redis']
        self.backup_storage = config['backup_storage']

    async def create_postgres_backup(self) -> str:
        """Create PostgreSQL backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"freeagentics_backup_{timestamp}.sql"

        command = [
            "pg_dump",
            "-h", self.postgres_config['host'],
            "-p", str(self.postgres_config['port']),
            "-U", self.postgres_config['username'],
            "-d", self.postgres_config['database'],
            "-f", backup_file,
            "--verbose"
        ]

        process = await asyncio.create_subprocess_exec(
            *command,
            env={"PGPASSWORD": self.postgres_config['password']},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            # Upload to backup storage (S3, etc.)
            await self._upload_backup(backup_file)
            return backup_file
        else:
            raise Exception(f"Backup failed: {stderr.decode()}")

    async def create_redis_backup(self) -> str:
        """Create Redis backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"redis_backup_{timestamp}.rdb"

        # Trigger Redis BGSAVE
        redis_client = redis.Redis(**self.redis_config)
        await redis_client.bgsave()

        # Wait for backup completion and copy RDB file
        # Implementation depends on Redis configuration

        return backup_file
```

### 5. Performance Optimization

#### Database Optimization
```sql
-- Performance indexes and optimizations
CREATE INDEX CONCURRENTLY idx_agents_composite
ON agents(tenant_id, status, last_active_at)
WHERE status = 'active';

CREATE INDEX CONCURRENTLY idx_coalition_members_performance
ON coalition_members(coalition_id, contribution_score DESC);

-- Partitioning for large tables
CREATE TABLE agent_history (
    id UUID,
    agent_id UUID,
    recorded_at TIMESTAMP WITH TIME ZONE,
    belief_state BYTEA,
    energy_level INTEGER
) PARTITION BY RANGE (recorded_at);

-- Create monthly partitions
CREATE TABLE agent_history_2024_01 PARTITION OF agent_history
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

#### Query Optimization
```python
# infrastructure/database/query_optimizer.py
class QueryOptimizer:
    """Database query optimization utilities."""

    @staticmethod
    def bulk_load_agents(session: Session, agents: List[Agent]):
        """Optimized bulk loading of agents."""
        # Use SQLAlchemy's bulk operations for better performance
        agent_data = [
            {
                'id': agent.id,
                'name': agent.name,
                'agent_type': agent.agent_type,
                'tenant_id': agent.tenant_id,
                'personality_traits': agent.personality.to_dict(),
                'energy_level': agent.energy,
                'location_h3': agent.location
            }
            for agent in agents
        ]

        session.bulk_insert_mappings(AgentModel, agent_data)
        session.commit()

    @staticmethod
    async def efficient_coalition_query(
        session: Session,
        location: str,
        business_model: str
    ) -> List[Coalition]:
        """Optimized query for coalitions by location and business model."""

        # Use spatial query with proper indexes
        query = """
        SELECT c.* FROM coalitions c
        JOIN coalition_members cm ON c.id = cm.coalition_id
        JOIN agents a ON cm.agent_id = a.id
        WHERE c.business_model = :business_model
        AND a.location_h3 = ANY(SELECT unnest(:nearby_locations))
        AND c.status = 'active'
        GROUP BY c.id
        HAVING COUNT(cm.agent_id) >= 2
        """

        nearby_locations = h3.k_ring(location, 2)

        result = await session.execute(
            text(query),
            {
                'business_model': business_model,
                'nearby_locations': list(nearby_locations)
            }
        )

        return [Coalition.from_db_row(row) for row in result]
```

## Architectural Compliance

### Directory Structure (ADR-002)
- Database models in `infrastructure/database/models/`
- Repositories in `infrastructure/database/repositories/`
- Migrations in `infrastructure/database/migrations/`
- Cache layer in `infrastructure/cache/`

### Dependency Rules (ADR-003)
- Core domain entities are persistence-agnostic
- Infrastructure layer implements persistence interfaces
- No database dependencies in domain layer

### Naming Conventions (ADR-004)
- Database tables use snake_case: `coalition_members`
- Repository classes use PascalCase: `AgentRepository`
- Cache keys use colon notation: `agent:beliefs:id`

## Performance Targets

### Database Performance
- **Query Response Time**: <100ms for 95% of queries
- **Write Throughput**: 1,000+ agent updates per second
- **Concurrent Connections**: Support 500+ simultaneous connections
- **Data Retention**: 1 year of historical data with efficient queries

### Cache Performance
- **Cache Hit Rate**: >90% for frequently accessed data
- **Cache Response Time**: <10ms for Redis operations
- **Real-time Updates**: <100ms latency for pub/sub messages

## Testing Strategy

### Database Testing
- **Unit Tests**: Repository and model testing
- **Integration Tests**: Full database flow testing
- **Performance Tests**: Load testing with realistic data volumes
- **Migration Tests**: Schema migration validation

### Data Integrity Testing
- **Consistency Tests**: Multi-table transaction validation
- **Backup/Recovery Tests**: Disaster recovery validation
- **Encryption Tests**: Data security validation

## Consequences

### Positive
- Scalable persistence supporting growth
- Optimized for agent and coalition use cases
- Real-time capabilities with caching
- Comprehensive backup and recovery

### Negative
- Increased infrastructure complexity
- Multiple database technologies to maintain
- Performance tuning requirements
- Backup storage costs

### Risks and Mitigations
- **Risk**: Database performance degradation under load
  - **Mitigation**: Performance monitoring and query optimization
- **Risk**: Data loss during failures
  - **Mitigation**: Automated backups and replication
- **Risk**: Cache inconsistency with database
  - **Mitigation**: Cache invalidation patterns and TTL management

## Related Decisions
- ADR-002: Canonical Directory Structure
- ADR-003: Dependency Rules
- ADR-009: Performance and Optimization Strategy
- ADR-011: Security and Authentication Architecture

This ADR ensures FreeAgentics has a robust, scalable persistence layer that supports the unique requirements of multi-agent systems while maintaining performance and data integrity.
