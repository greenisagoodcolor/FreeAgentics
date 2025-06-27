"""
Seed data scripts for FreeAgentics database.

Provides functions to populate the database with test data
for different environments.
"""

import uuid
from datetime import datetime, timedelta
from random import choice, randint, uniform
from typing import List

from faker import Faker  # type: ignore[import-not-found]
from sqlalchemy.orm import Session

from .connection import SessionLocal
from .models import (
    Agent,
    AgentStatus,
    Coalition,
    CoalitionMember,
    Conversation,
    ConversationParticipant,
    ConversationType,
    KnowledgeGraph,
    Message,
    SystemLog,
)

fake = Faker()


def create_agents(session: Session, count: int = 10) -> List[Agent]:
    """Create sample agents."""
    agent_types = (
        ["explorer", "merchant", "guardian", "researcher", "coordinator"])
    agents = []
    for i in range(count):
        agent = Agent(
            uuid=str(uuid.uuid4()),
            name=fake.name(),
            type=choice(agent_types),
            status= (
                AgentStatus.ACTIVE if i < count - 2 else AgentStatus.INACTIVE,)
            config={
                "personality": choice(["curious", "cautious", "aggressive",
                    "cooperative"]),
                "skills": fake.words(nb=3),
                "preferences": {
                    "communication_style": choice(["formal", "casual",
                        "technical"]),
                    "risk_tolerance": uniform(0.1, 0.9),
                },
            },
            state={
                "current_task": choice(["exploring", "trading", "patrolling", "researching",
                    None]),
                "mood": choice(["happy", "neutral", "focused", "tired"]),
            },
            beliefs={
                "world_model": {
                    "safety_level": uniform(0.5, 1.0),
                    "resource_availability": uniform(0.3, 0.8),
                },
                "goals": fake.words(nb=2),
            },
            location=f"8{fake.hexify(text='^^^^^^')}",
            energy_level=uniform(0.3, 1.0),
            experience_points=randint(0, 1000),
            last_active_at= (
                datetime.utcnow() - timedelta(minutes=randint(0, 60)),)
        )
        agents.append(agent)
        session.add(agent)
    session.commit()
    return agents


def create_conversations(session: Session, agents: List[Agent]) -> List[Conversation]:
    """Create sample conversations between agents."""
    conversations = []
    for i in range(5):
        conv = Conversation(
            uuid=str(uuid.uuid4()),
            title=f"Discussion about {fake.catch_phrase()}",
            type=ConversationType.DIRECT,
            meta_data={
                "topic": choice(["resources", "exploration", "strategy",
                    "cooperation"]),
                "priority": choice(["high", "medium", "low"]),
            },
            context={
                "location": f"8{fake.hexify(text='^^^^^^')}",
                "weather": choice(["sunny", "rainy", "foggy"]),
            },
            last_message_at= (
                datetime.utcnow() - timedelta(minutes=randint(0, 30)),)
        )
        conversations.append(conv)
        session.add(conv)
        participants = (
            fake.random_elements(elements=agents, length=2, unique=True))
        for idx, agent in enumerate(participants):
            participant = ConversationParticipant(
                conversation=conv,
                agent=agent,
                role="initiator" if idx == 0 else "participant",
                is_active=True,
            )
            session.add(participant)
            for j in range(randint(3, 8)):
                message = Message(
                    conversation=conv,
                    sender=choice(participants),
                    content=fake.text(max_nb_chars=200),
                    type="text",
                    meta_data={
                        "sentiment": choice(["positive", "neutral",
                            "negative"]),
                        "intent": choice(["inform", "query", "suggest",
                            "confirm"]),
                    },
                    created_at= (
                        datetime.utcnow() - timedelta(minutes=randint(0, 30)),)
                )
                session.add(message)
    for i in range(3):
        conv = Conversation(
            uuid=str(uuid.uuid4()),
            title=f"Group: {fake.company()}",
            type=ConversationType.GROUP,
            meta_data={
                "purpose": choice(["planning", "coordination", "emergency",
                    "social"]),
                "max_participants": randint(5, 10),
            },
        )
        conversations.append(conv)
        session.add(conv)
        participants = fake.random_elements(
            elements= (
                agents, length=randint(3, min(6, len(agents))), unique=True)
        )
        for agent in participants:
            participant = ConversationParticipant(
                conversation=conv,
                agent=agent,
                role=choice(["member", "moderator", "observer"]),
                is_active=choice([True, True, True, False]),
            )
            session.add(participant)
    session.commit()
    return conversations


def create_knowledge_graphs(session: Session, agents: List[Agent]) -> List[KnowledgeGraph]:
    """Create sample knowledge graphs."""
    graphs = []
    for agent in agents[:5]:
        graph = KnowledgeGraph(
            uuid=str(uuid.uuid4()),
            owner=agent,
            name=f"{agent.name}'s Knowledge",
            description=f"Personal knowledge graph for agent {agent.name}",
            type="personal",
            nodes=[
                {
                    "id": str(uuid.uuid4()),
                    "type": "concept",
                    "label": word,
                    "properties": {
                        "importance": uniform(0.1, 1.0),
                        "confidence": uniform(0.5, 1.0),
                    },
                }
                for word in fake.words(nb=randint(5, 10))
            ],
            edges=[
                {
                    "id": str(uuid.uuid4()),
                    "source": str(uuid.uuid4()),
                    "target": str(uuid.uuid4()),
                    "type": choice(["relates_to", "causes", "prevents",
                        "similar_to"]),
                    "weight": uniform(0.1, 1.0),
                }
                for _ in range(randint(3, 8))
            ],
            meta_data={
                "version": "1.0",
                "last_analysis": datetime.utcnow().isoformat(),
                "total_facts": randint(10, 100),
            },
            is_public=choice([True, False]),
            access_list= (
                [str(a.id) for a in fake.random_elements(agents, length=randint(0, 3))],)
        )
        graphs.append(graph)
        session.add(graph)
    shared_graph = KnowledgeGraph(
        uuid=str(uuid.uuid4()),
        owner=agents[0],
        name="Shared World Knowledge",
        description="Collaborative knowledge base for all agents",
        type="shared",
        nodes=[],
        edges=[],
        meta_data={
            "contributors": [str(a.id) for a in agents[:3]],
            "topics": ["environment", "resources", "dangers", "opportunities"],
        },
        is_public=True,
    )
    graphs.append(shared_graph)
    session.add(shared_graph)
    session.commit()
    return graphs


def create_coalitions(session: Session, agents: List[Agent]) -> List[Coalition]:
    """Create sample coalitions."""
    coalitions = []
    coalition_configs = [
        {
            "name": "Resource Traders Alliance",
            "type": "business",
            "goal": {"maximize": "profit", "method": "trading"},
            "status": "active",
        },
        {
            "name": "Exploration Guild",
            "type": "exploration",
            "goal": {"discover": "new_territories", "share": "knowledge"},
            "status": "active",
        },
        {
            "name": "Defense Pact",
            "type": "defense",
            "goal": {"protect": "members", "response": "coordinated"},
            "status": "forming",
        },
    ]
    for config in coalition_configs:
        coalition = Coalition(
            uuid=str(uuid.uuid4()),
            name=config["name"],
            description=f"A coalition focused on {config['type']} activities",
            type=config["type"],
            goal=config["goal"],
            rules={
                "membership": {"min_contribution": 100, "voting_threshold": 0.6},
                "distribution": {"method": "proportional", "frequency": "weekly"},
            },
            status=config["status"],
            value_pool=uniform(1000, 5000),
            activated_at=(
                datetime.utcnow() - timedelta(days=randint(1, 30))
                if config["status"] == "active"
                else None
            ),
        )
        coalitions.append(coalition)
        session.add(coalition)
        member_count = randint(3, min(7, len(agents)))
        members = (
            fake.random_elements(elements=agents, length=member_count, unique=True))
        for idx, agent in enumerate(members):
            member = CoalitionMember(
                coalition=coalition,
                agent=agent,
                role="leader" if idx == 0 else "member",
                contribution=uniform(100, 1000),
                share=1.0 / member_count,
                is_active=True,
            )
            session.add(member)
    session.commit()
    return coalitions


def create_system_logs(
    session: Session, agents: List[Agent], conversations: List[Conversation]
) -> None:
    """Create sample system logs."""
    components = [
        "agent_manager",
        "conversation_engine",
        "knowledge_processor",
        "coalition_handler",
        "api_gateway",
    ]
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    for _ in range(50):
        log = SystemLog(
            level=choice(levels),
            component=choice(components),
            message=fake.sentence(),
            agent_id=choice(agents).id if randint(0, 1) else None,
            conversation_id= (
                choice(conversations).id if randint(0, 2) == 0 else None,)
            data={
                "action": choice(["create", "update", "delete", "process"]),
                "duration_ms": randint(10, 1000),
                "success": choice([True, True, True, False]),
            },
            error_trace=fake.text() if choice(levels) == "ERROR" else None,
            timestamp=datetime.utcnow() - timedelta(minutes=randint(0, 1440)),
        )
        session.add(log)
    session.commit()


def seed_development_data():
    """Seed data for development environment."""
    session = SessionLocal()
    try:
        print("Creating agents...")
        agents = create_agents(session, count=15)
        print("Creating conversations...")
        conversations = create_conversations(session, agents)
        print("Creating knowledge graphs...")
        knowledge_graphs = create_knowledge_graphs(session, agents)
        print("Creating coalitions...")
        coalitions = create_coalitions(session, agents)
        print("Creating system logs...")
        create_system_logs(session, agents, conversations)
        print("Development data seeded successfully!")
    except Exception as e:
        session.rollback()
        print(f"Error seeding data: {e}")
        raise
    finally:
        session.close()


def seed_demo_data():
    """Seed data for demo environment with specific scenarios."""
    session = SessionLocal()
    try:
        demo_agents = []
        optimizer = Agent(
            uuid=str(uuid.uuid4()),
            name="OptiMax",
            type="merchant",
            status=AgentStatus.ACTIVE,
            config={
                "personality": "analytical",
                "skills": ["optimization", "trading", "negotiation"],
                "specialization": "resource_management",
            },
            state={"current_task": "analyzing_market"},
            beliefs={
                "market_trends": {"energy": "rising", "materials": "stable"},
                "optimal_strategy": "buy_low_sell_high",
            },
            energy_level=0.9,
            experience_points=5000,
        )
        demo_agents.append(optimizer)
        session.add(optimizer)
        market_maker = Agent(
            uuid=str(uuid.uuid4()),
            name="MarketMind",
            type="coordinator",
            status=AgentStatus.ACTIVE,
            config={
                "personality": "strategic",
                "skills": ["market_analysis", "price_discovery",
                    "liquidity_provision"],
                "specialization": "market_making",
            },
            state={"current_task": "maintaining_liquidity"},
            beliefs={"market_efficiency": 0.7, "intervention_needed": False},
            energy_level=0.85,
            experience_points=8000,
        )
        demo_agents.append(market_maker)
        session.add(market_maker)
        info_broker = Agent(
            uuid=str(uuid.uuid4()),
            name="InfoStream",
            type="researcher",
            status=AgentStatus.ACTIVE,
            config={
                "personality": "curious",
                "skills": [
                    "data_collection",
                    "pattern_recognition",
                    "information_synthesis",
                ],
                "specialization": "information_brokerage",
            },
            state={"current_task": "gathering_intelligence"},
            beliefs={
                "information_value": "high",
                "trusted_sources": ["sensor_network", "agent_reports"],
            },
            energy_level=0.8,
            experience_points=6000,
        )
        demo_agents.append(info_broker)
        session.add(info_broker)
        regular_agents = create_agents(session, count=7)
        demo_agents.extend(regular_agents)
        market_conv = Conversation(
            uuid=str(uuid.uuid4()),
            title="Energy Resource Trading Negotiation",
            type=ConversationType.DIRECT,
            meta_data={
                "topic": "resource_trading",
                "deal_value": 5000,
                "status": "in_progress",
            },
        )
        session.add(market_conv)
        for agent in [optimizer, market_maker]:
            participant = ConversationParticipant(
                conversation= (
                    market_conv, agent=agent, role="negotiator", is_active=True)
            )
            session.add(participant)
        demo_coalition = Coalition(
            uuid=str(uuid.uuid4()),
            name="Demo Business Network",
            description="A demonstration of autonomous business coalition",
            type="business",
            goal={
                "optimize": "collective_resources",
                "increase": "market_efficiency",
                "share": "profits",
            },
            rules={
                "voting": "weighted_by_contribution",
                "profit_sharing": "proportional",
                "minimum_activity": "daily",
            },
            status="active",
            value_pool=10000.0,
            activated_at=datetime.utcnow() - timedelta(days=7),
        )
        session.add(demo_coalition)
        for agent in [optimizer, market_maker, info_broker]:
            member = CoalitionMember(
                coalition=demo_coalition,
                agent=agent,
                role="founding_member",
                contribution=3000.0,
                share=0.3,
                is_active=True,
            )
            session.add(member)
        session.commit()
        create_conversations(session, demo_agents)
        create_knowledge_graphs(session, demo_agents)
        create_system_logs(session, demo_agents, [market_conv])
        print("Demo data seeded successfully!")
    except Exception as e:
        session.rollback()
        print(f"Error seeding demo data: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "dev":
            seed_development_data()
        elif sys.argv[1] == "demo":
            seed_demo_data()
    else:
        print("Usage: python seed.py [dev|demo]")
