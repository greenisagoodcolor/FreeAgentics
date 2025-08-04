#!/usr/bin/env python3
"""
Three-Agent Conversation Demo - FreeAgentics
============================================

A minimal demonstration of multi-agent conversation using three distinct agents:
- Alex (Analyst): Focuses on data and facts
- Blake (Creative): Generates ideas and possibilities  
- Casey (Synthesizer): Combines perspectives and makes decisions

This demo shows the core concepts of multi-agent systems through a simple,
educational conversation without external dependencies.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime

# Set up clean logging for the demo
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent personality types"""
    ANALYST = "analyst"
    CREATIVE = "creative" 
    SYNTHESIZER = "synthesizer"


@dataclass
class Message:
    """Simple message structure for agent communication"""
    sender: str
    recipient: Optional[str]  # None means broadcast to all
    content: str
    timestamp: datetime
    turn: int
    
    def __str__(self) -> str:
        recipient_str = f" -> {self.recipient}" if self.recipient else " -> ALL"
        return f"[Turn {self.turn}] {self.sender}{recipient_str}: {self.content}"


class ConversationAgent:
    """A simple agent that participates in conversations"""
    
    def __init__(self, name: str, role: AgentRole, personality: str):
        self.name = name
        self.role = role
        self.personality = personality
        self.conversation_history: List[Message] = []
        self.internal_state = {"focus": None, "mood": "neutral"}
        
    def receive_message(self, message: Message) -> None:
        """Process an incoming message"""
        if message.recipient and message.recipient != self.name:
            return  # Message not for us
        
        self.conversation_history.append(message)
        self._update_internal_state(message)
        
    def _update_internal_state(self, message: Message) -> None:
        """Update agent's internal state based on message"""
        # Simple state updates based on message content
        if "problem" in message.content.lower():
            self.internal_state["focus"] = "problem-solving"
        elif "idea" in message.content.lower():
            self.internal_state["focus"] = "brainstorming"
        elif "decision" in message.content.lower():
            self.internal_state["focus"] = "concluding"
            
    def generate_response(self, topic: str, turn: int) -> Optional[Message]:
        """Generate a response based on conversation context and role"""
        
        # Get recent context
        recent_messages = self.conversation_history[-3:] if self.conversation_history else []
        
        # Role-based response generation
        if self.role == AgentRole.ANALYST:
            content = self._generate_analytical_response(topic, recent_messages)
        elif self.role == AgentRole.CREATIVE:
            content = self._generate_creative_response(topic, recent_messages)
        else:  # SYNTHESIZER
            content = self._generate_synthesis_response(topic, recent_messages)
            
        if content is None:
            return None
            
        return Message(
            sender=self.name,
            recipient=None,  # Broadcast to all
            content=content,
            timestamp=datetime.now(),
            turn=turn
        )
    
    def _generate_analytical_response(self, topic: str, recent_messages: List[Message]) -> Optional[str]:
        """Generate analytical, fact-focused responses"""
        turn_count = len(self.conversation_history) + 1
        
        analytical_responses = [
            f"Let me analyze the '{topic}' problem systematically. What are the key constraints and requirements?",
            "Based on what's been shared, I think we need to consider budget, dietary restrictions, and team size. How many people are we planning for?",
            "Good points! I suggest we prioritize: 1) Survey team preferences, 2) Calculate cost per person, 3) Choose optimal timing. Should I draft a quick survey?",
            "From the data perspective, lunch-time pizza parties have 85% attendance vs 60% for after-work events. What's our primary goal here?",
            "Looking at our discussion, I can outline a practical action plan with clear metrics for success. Ready to finalize the details?"
        ]
        
        return analytical_responses[min(turn_count - 1, len(analytical_responses) - 1)]
    
    def _generate_creative_response(self, topic: str, recent_messages: List[Message]) -> Optional[str]:
        """Generate creative, idea-focused responses"""
        turn_count = len(self.conversation_history) + 1
        
        creative_responses = [
            f"Oh, I love working on '{topic}'! This opens up so many exciting possibilities. What if we thought completely outside the box?",
            "What if we made it a themed pizza party? Like 'Pizza Around the World' with different international toppings? Or a build-your-own pizza bar?",
            "Ooh, and we could gamify it! Pizza trivia, toppings guessing games, maybe even a pizza-making contest between teams?",
            "Here's a wild idea: what if we livestream the party planning process and let remote team members vote on toppings in real-time?",
            "I'm seeing this whole experience becoming legendary! Let's make it the pizza party people will still talk about years from now!"
        ]
        
        return creative_responses[min(turn_count - 1, len(creative_responses) - 1)]
    
    def _generate_synthesis_response(self, topic: str, recent_messages: List[Message]) -> Optional[str]:
        """Generate synthesizing, decision-focused responses"""
        turn_count = len(self.conversation_history) + 1
        
        synthesis_responses = [
            f"Great topic: '{topic}'. I'm here to help us find the best path forward by combining all our perspectives.",
            "I love how Alex is thinking about the logistics while Blake is envisioning the experience. What if we do a structured creative approach?",
            "Perfect! So we're looking at: practical planning (surveys, budget) + creative elements (themes, games). Let's balance both!",
            "This is coming together beautifully! We have data-driven planning with creative engagement. How about we prototype this approach?",
            "Excellent! We've got our framework: systematic planning meets creative execution. I think we're ready to make this happen!"
        ]
        
        return synthesis_responses[min(turn_count - 1, len(synthesis_responses) - 1)]
    
    def get_internal_state_summary(self) -> str:
        """Get a summary of the agent's current internal state"""
        return f"[{self.name} thinking: focus={self.internal_state['focus']}, mood={self.internal_state['mood']}]"


class ConversationCoordinator:
    """Manages turn-taking and message routing between agents"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.agents: Dict[str, ConversationAgent] = {}
        self.message_history: List[Message] = []
        self.current_turn = 0
        self.max_turns = 8
        
    def add_agent(self, agent: ConversationAgent) -> None:
        """Add an agent to the conversation"""
        self.agents[agent.name] = agent
        logger.info(f"✅ {agent.name} joined the conversation ({agent.role.value})")
        
    def broadcast_message(self, message: Message) -> None:
        """Send a message to all relevant agents"""
        self.message_history.append(message)
        
        for agent in self.agents.values():
            if agent.name != message.sender:  # Don't send message back to sender
                agent.receive_message(message)
    
    def run_conversation(self) -> None:
        """Execute the full conversation"""
        logger.info("\n" + "=" * 60)
        logger.info(f"🎭 Three-Agent Conversation: {self.topic}")
        logger.info("=" * 60)
        
        # Show agent introductions
        logger.info("\n👥 Meet the Agents:")
        for agent in self.agents.values():
            logger.info(f"  • {agent.name} ({agent.role.value}): {agent.personality}")
        
        logger.info(f"\n💬 Starting conversation about: {self.topic}")
        logger.info("-" * 60)
        
        # Get agent names in order for turn-taking
        agent_names = list(self.agents.keys())
        
        # Run conversation turns
        for turn in range(1, self.max_turns + 1):
            self.current_turn = turn
            
            # Determine which agent speaks (round-robin)
            speaker_index = (turn - 1) % len(agent_names)
            speaker_name = agent_names[speaker_index]
            speaker = self.agents[speaker_name]
            
            # Show agent's internal state
            logger.info(f"\n{speaker.get_internal_state_summary()}")
            
            # Generate and broadcast response
            message = speaker.generate_response(self.topic, turn)
            if message:
                self.broadcast_message(message)
                logger.info(f"💭 {message}")
                
                # Small pause for readability
                time.sleep(0.5)
            
            # Check if conversation should end
            if self._should_end_conversation():
                break
                
        logger.info("\n" + "-" * 60)
        logger.info("🎯 Conversation Summary:")
        self._show_conversation_summary()
        logger.info("=" * 60)
        
    def _should_end_conversation(self) -> bool:
        """Determine if conversation should end based on content"""
        if self.current_turn >= self.max_turns:
            return True
            
        # Check if agents are converging on a solution
        recent_messages = self.message_history[-3:] if len(self.message_history) >= 3 else []
        decision_keywords = ["finalize", "conclude", "decided", "agreed", "solution"]
        
        for message in recent_messages:
            if any(keyword in message.content.lower() for keyword in decision_keywords):
                logger.info("\n🎯 Agents are converging on a solution!")
                return True
                
        return False
    
    def _show_conversation_summary(self) -> None:
        """Display a summary of the conversation"""
        logger.info(f"  • Total turns: {len(self.message_history)}")
        logger.info(f"  • Topic: {self.topic}")
        
        # Show each agent's contribution
        for agent_name, agent in self.agents.items():
            agent_messages = [m for m in self.message_history if m.sender == agent_name]
            logger.info(f"  • {agent_name} contributed {len(agent_messages)} messages ({agent.role.value})")


def create_demo_agents() -> List[ConversationAgent]:
    """Create the three demo agents with distinct personalities"""
    
    agents = [
        ConversationAgent(
            name="Alex",
            role=AgentRole.ANALYST,
            personality="Systematic, data-driven, asks probing questions"
        ),
        ConversationAgent(
            name="Blake", 
            role=AgentRole.CREATIVE,
            personality="Imaginative, enthusiastic, thinks outside the box"
        ),
        ConversationAgent(
            name="Casey",
            role=AgentRole.SYNTHESIZER,
            personality="Diplomatic, integrative, builds consensus"
        )
    ]
    
    return agents


def run_demo() -> None:
    """Run the complete three-agent conversation demo"""
    
    # Demo topic - something relatable and engaging
    topic = "Planning the Perfect Team Pizza Party"
    
    # Create coordinator
    coordinator = ConversationCoordinator(topic)
    
    # Add agents
    agents = create_demo_agents()
    for agent in agents:
        coordinator.add_agent(agent)
    
    # Run the conversation
    coordinator.run_conversation()
    
    # Educational outro
    logger.info("\n🎓 What You Just Saw:")
    logger.info("  • Three agents with different roles and personalities")
    logger.info("  • Turn-based conversation with message passing")
    logger.info("  • Internal state updates based on conversation context")
    logger.info("  • Emergent consensus through multi-agent interaction")
    logger.info("\n💡 This demonstrates core multi-agent system concepts:")
    logger.info("  • Agent autonomy (each has distinct behavior)")
    logger.info("  • Communication protocols (structured message passing)")
    logger.info("  • Coordination mechanisms (turn-taking, consensus)")
    logger.info("  • Emergent intelligence (collective problem-solving)")
    logger.info("\n🔧 Try modifying the code to:")
    logger.info("  • Add more agents with different roles")
    logger.info("  • Change agent personalities and response patterns")
    logger.info("  • Experiment with different conversation topics")
    logger.info("  • Add more sophisticated internal state management")


if __name__ == "__main__":
    run_demo()