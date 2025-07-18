"""
Integration tests for Coalition functionality using real database.

This test demonstrates how to use the real PostgreSQL database infrastructure
instead of mocks for testing coalition formation and management.
"""

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from database.models import (
    Agent,
    AgentRole,
    AgentStatus,
    Coalition,
    CoalitionStatus,
)
from tests.db_infrastructure.factories import AgentFactory, CoalitionFactory
from tests.db_infrastructure.fixtures import db_session
from tests.db_infrastructure.test_config import DatabaseTestCase


class TestCoalitionDatabase(DatabaseTestCase):
    """Test coalition functionality with real database."""

    def test_create_coalition_with_agents(self, db_session: Session):
        """Test creating a coalition and adding agents."""
        # Create agents using factory
        agents = [AgentFactory(name=f"Agent {i}") for i in range(3)]
        db_session.add_all(agents)
        db_session.commit()

        # Create coalition
        coalition = CoalitionFactory(
            name="Test Coalition",
            description="Testing coalition formation",
            status=CoalitionStatus.FORMING,
        )
        db_session.add(coalition)
        db_session.commit()

        # Add agents to coalition through the association table
        from database.models import agent_coalition_association

        # Add leader
        db_session.execute(
            agent_coalition_association.insert().values(
                agent_id=agents[0].id,
                coalition_id=coalition.id,
                role=AgentRole.LEADER,
                contribution_score=1.0,
                trust_score=1.0,
            )
        )

        # Add members
        for agent in agents[1:]:
            db_session.execute(
                agent_coalition_association.insert().values(
                    agent_id=agent.id,
                    coalition_id=coalition.id,
                    role=AgentRole.MEMBER,
                    contribution_score=0.5,
                    trust_score=0.8,
                )
            )

        db_session.commit()

        # Verify coalition has agents
        db_session.refresh(coalition)
        assert len(coalition.agents) == 3

        # Verify agents are in coalition
        for agent in agents:
            db_session.refresh(agent)
            assert coalition in agent.coalitions

    def test_coalition_lifecycle(self, db_session: Session):
        """Test coalition lifecycle from formation to dissolution."""
        # Create coalition in forming state
        coalition = CoalitionFactory(status=CoalitionStatus.FORMING)
        db_session.add(coalition)
        db_session.commit()

        # Add agents
        agents = [AgentFactory() for _ in range(5)]
        db_session.add_all(agents)
        db_session.commit()

        # Add agents to coalition
        coalition.agents.extend(agents)
        db_session.commit()

        # Update coalition status to active
        coalition.status = CoalitionStatus.ACTIVE
        coalition.objectives["primary"] = "Resource collection"
        coalition.shared_state["resources_collected"] = 0
        db_session.commit()

        # Simulate coalition activity
        coalition.shared_state["resources_collected"] = 150
        coalition.performance_metrics = {
            "efficiency": 0.85,
            "coordination_score": 0.92,
            "goal_completion": 0.75,
        }
        db_session.commit()

        # Start disbanding
        coalition.status = CoalitionStatus.DISBANDING
        db_session.commit()

        # Final dissolution
        coalition.status = CoalitionStatus.DISSOLVED
        coalition.dissolved_at = datetime.utcnow()
        db_session.commit()

        # Verify final state
        db_session.refresh(coalition)
        assert coalition.status == CoalitionStatus.DISSOLVED
        assert coalition.dissolved_at is not None
        assert coalition.performance_metrics["efficiency"] == 0.85

    def test_agent_role_management(self, db_session: Session):
        """Test managing agent roles within coalitions."""
        # Create coalition and agents
        coalition = CoalitionFactory()
        agents = [AgentFactory() for _ in range(4)]
        db_session.add(coalition)
        db_session.add_all(agents)
        db_session.commit()

        # Assign different roles
        from database.models import agent_coalition_association

        roles = [
            AgentRole.LEADER,
            AgentRole.COORDINATOR,
            AgentRole.MEMBER,
            AgentRole.OBSERVER,
        ]

        for agent, role in zip(agents, roles):
            db_session.execute(
                agent_coalition_association.insert().values(
                    agent_id=agent.id, coalition_id=coalition.id, role=role
                )
            )

        db_session.commit()

        # Query agents by role
        result = db_session.execute(
            agent_coalition_association.select().where(
                agent_coalition_association.c.coalition_id == coalition.id,
                agent_coalition_association.c.role == AgentRole.LEADER,
            )
        ).first()

        assert result is not None
        assert result.agent_id == agents[0].id

    def test_coalition_performance_tracking(self, db_session: Session):
        """Test tracking coalition performance over time."""
        # Create active coalition
        coalition = CoalitionFactory(
            status=CoalitionStatus.ACTIVE,
            objectives={"goal": "maximize_efficiency"},
        )
        agents = [AgentFactory(status=AgentStatus.ACTIVE) for _ in range(3)]

        db_session.add(coalition)
        db_session.add_all(agents)
        db_session.commit()

        coalition.agents.extend(agents)
        db_session.commit()

        # Simulate performance updates over time
        performance_history = []

        for hour in range(5):
            # Update shared state
            coalition.shared_state["tasks_completed"] = hour * 10
            coalition.shared_state["resources_used"] = hour * 5

            # Calculate and update metrics
            efficiency = (hour * 10) / max((hour * 5), 1)
            coalition.performance_metrics = {
                "efficiency": efficiency,
                "uptime_hours": hour,
                "task_completion_rate": 0.8 + (hour * 0.02),
                "timestamp": (
                    datetime.utcnow() + timedelta(hours=hour)
                ).isoformat(),
            }

            performance_history.append(
                {
                    "hour": hour,
                    "efficiency": efficiency,
                    "tasks": coalition.shared_state["tasks_completed"],
                }
            )

            db_session.commit()

        # Verify final metrics
        db_session.refresh(coalition)
        assert coalition.performance_metrics["uptime_hours"] == 4
        assert coalition.shared_state["tasks_completed"] == 40

        # Verify we can query historical data
        assert len(performance_history) == 5
        assert (
            performance_history[-1]["efficiency"] == 2.0
        )  # 40 tasks / 20 resources

    def test_multi_coalition_agent_membership(self, db_session: Session):
        """Test agents being members of multiple coalitions."""
        # Create agents
        versatile_agent = AgentFactory(name="Versatile Agent")
        other_agents = [AgentFactory() for _ in range(4)]

        db_session.add(versatile_agent)
        db_session.add_all(other_agents)
        db_session.commit()

        # Create multiple coalitions
        coalitions = [
            CoalitionFactory(name="Research Coalition"),
            CoalitionFactory(name="Exploration Coalition"),
            CoalitionFactory(name="Defense Coalition"),
        ]
        db_session.add_all(coalitions)
        db_session.commit()

        # Add versatile agent to all coalitions
        from database.models import agent_coalition_association

        for i, coalition in enumerate(coalitions):
            # Add versatile agent with different roles
            role = [
                AgentRole.COORDINATOR,
                AgentRole.MEMBER,
                AgentRole.OBSERVER,
            ][i]
            db_session.execute(
                agent_coalition_association.insert().values(
                    agent_id=versatile_agent.id,
                    coalition_id=coalition.id,
                    role=role,
                )
            )

            # Add some other agents too
            db_session.execute(
                agent_coalition_association.insert().values(
                    agent_id=other_agents[i].id,
                    coalition_id=coalition.id,
                    role=AgentRole.MEMBER,
                )
            )

        db_session.commit()

        # Verify versatile agent is in all coalitions
        db_session.refresh(versatile_agent)
        assert len(versatile_agent.coalitions) == 3
        coalition_names = {c.name for c in versatile_agent.coalitions}
        assert coalition_names == {
            "Research Coalition",
            "Exploration Coalition",
            "Defense Coalition",
        }

        # Verify other agents are in only one coalition each
        for i, agent in enumerate(other_agents[:3]):
            db_session.refresh(agent)
            assert len(agent.coalitions) == 1
            assert agent.coalitions[0] == coalitions[i]

    def test_coalition_trust_and_contribution_tracking(
        self, db_session: Session
    ):
        """Test tracking trust scores and contributions within coalitions."""
        # Create coalition and agents
        coalition = CoalitionFactory(name="Trust Test Coalition")
        agents = [AgentFactory() for _ in range(3)]

        db_session.add(coalition)
        db_session.add_all(agents)
        db_session.commit()

        # Add agents with initial trust scores
        from database.models import agent_coalition_association

        for agent in agents:
            db_session.execute(
                agent_coalition_association.insert().values(
                    agent_id=agent.id,
                    coalition_id=coalition.id,
                    role=AgentRole.MEMBER,
                    trust_score=1.0,
                    contribution_score=0.0,
                )
            )

        db_session.commit()

        # Simulate contributions and trust updates
        # Agent 0: High contributor, maintains trust
        # Agent 1: Medium contributor, trust decreases slightly
        # Agent 2: Low contributor, trust decreases significantly

        contribution_updates = [
            (agents[0].id, 0.9, 0.95),  # (agent_id, contribution, trust)
            (agents[1].id, 0.5, 0.85),
            (agents[2].id, 0.1, 0.6),
        ]

        for agent_id, contribution, trust in contribution_updates:
            db_session.execute(
                agent_coalition_association.update()
                .where(
                    agent_coalition_association.c.agent_id == agent_id,
                    agent_coalition_association.c.coalition_id == coalition.id,
                )
                .values(contribution_score=contribution, trust_score=trust)
            )

        db_session.commit()

        # Query and verify scores
        results = db_session.execute(
            agent_coalition_association.select()
            .where(agent_coalition_association.c.coalition_id == coalition.id)
            .order_by(agent_coalition_association.c.contribution_score.desc())
        ).all()

        assert len(results) == 3
        assert results[0].contribution_score == 0.9  # Highest contributor
        assert results[0].trust_score == 0.95
        assert results[2].contribution_score == 0.1  # Lowest contributor
        assert results[2].trust_score == 0.6

    def test_coalition_query_performance(self, db_session: Session):
        """Test efficient querying of coalition data."""
        # Create many coalitions and agents for performance testing
        coalitions = [CoalitionFactory() for _ in range(10)]
        agents = [AgentFactory() for _ in range(50)]

        db_session.add_all(coalitions)
        db_session.add_all(agents)
        db_session.commit()

        # Distribute agents across coalitions
        import random

        from database.models import agent_coalition_association

        for agent in agents:
            # Each agent joins 1-3 coalitions
            num_coalitions = random.randint(1, 3)
            selected_coalitions = random.sample(coalitions, num_coalitions)

            for coalition in selected_coalitions:
                db_session.execute(
                    agent_coalition_association.insert().values(
                        agent_id=agent.id,
                        coalition_id=coalition.id,
                        role=random.choice(list(AgentRole)),
                    )
                )

        db_session.commit()

        # Test efficient queries
        # 1. Find all active coalitions with more than 5 members
        from sqlalchemy import func

        active_large_coalitions = (
            db_session.query(Coalition)
            .join(Coalition.agents)
            .filter(Coalition.status == CoalitionStatus.ACTIVE)
            .group_by(Coalition.id)
            .having(func.count(Agent.id) > 5)
            .all()
        )

        # 2. Find agents in multiple coalitions
        from sqlalchemy import select

        multi_coalition_agents = (
            db_session.execute(
                select(Agent)
                .join(Agent.coalitions)
                .group_by(Agent.id)
                .having(func.count(Coalition.id) > 1)
            )
            .scalars()
            .all()
        )

        assert len(multi_coalition_agents) > 0

        # 3. Get coalition statistics
        coalition_stats = db_session.execute(
            select(
                Coalition.id,
                Coalition.name,
                func.count(Agent.id).label("member_count"),
                func.avg(agent_coalition_association.c.trust_score).label(
                    "avg_trust"
                ),
            )
            .select_from(Coalition)
            .join(agent_coalition_association)
            .join(Agent)
            .group_by(Coalition.id, Coalition.name)
        ).all()

        assert len(coalition_stats) > 0
        for stat in coalition_stats:
            assert stat.member_count > 0
            assert 0 <= stat.avg_trust <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
