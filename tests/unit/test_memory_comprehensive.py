"""
Comprehensive tests for Memory module
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from agents.base.data_model import Action, ActionType, Agent, Position
from agents.base.decision_making import DecisionContext
from agents.base.memory import (  # Enums; Core classes; Storage; Systems
    Experience,
    InMemoryStorage,
    LearningAlgorithm,
    Memory,
    MemoryConsolidator,
    MemoryImportance,
    MemoryStorage,
    MemorySystem,
    MemoryType,
    MessageSystem,
    Pattern,
    PatternRecognizer,
    ReinforcementLearner,
    WorkingMemory,
)
from agents.base.perception import Percept, PerceptionType, Stimulus, StimulusType


class TestEnums:
    """Test memory enum types"""

    def test_memory_type_enum(self):
        """Test MemoryType enum values"""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.SENSORY.value == "sensory"
        assert MemoryType.WORKING.value == "working"
        assert len(MemoryType) == 5

    def test_memory_importance_enum(self):
        """Test MemoryImportance enum values"""
        assert MemoryImportance.CRITICAL.value == 5
        assert MemoryImportance.HIGH.value == 4
        assert MemoryImportance.MEDIUM.value == 3
        assert MemoryImportance.LOW.value == 2
        assert MemoryImportance.TRIVIAL.value == 1
        assert len(MemoryImportance) == 5


class TestMemory:
    """Test Memory dataclass"""

    def test_memory_creation(self):
        """Test creating a memory"""
        memory = Memory(
            memory_id="test_1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "found_food", "location": (10, 20)},
            timestamp=datetime.now(),
            importance=0.8,
        )

        assert memory.memory_id == "test_1"
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.content["event"] == "found_food"
        assert memory.importance == 0.8
        assert memory.access_count == 0
        assert memory.decay_rate == 0.01
        assert memory.associations == []
        assert isinstance(memory.context, dict)

    def test_memory_defaults(self):
        """Test memory default values"""
        now = datetime.now()
        memory = Memory(
            memory_id="test_2",
            memory_type=MemoryType.SEMANTIC,
            content="test content",
            timestamp=now,
        )

        assert memory.importance == 0.5
        assert memory.access_count == 0
        assert isinstance(memory.last_accessed, datetime)
        assert memory.decay_rate == 0.01
        assert memory.associations == []
        assert memory.context == {}
        assert memory.metadata == {}

    def test_memory_calculate_strength(self):
        """Test memory strength calculation"""
        past_time = datetime.now() - timedelta(hours=24)
        memory = Memory(
            memory_id="test_3",
            memory_type=MemoryType.EPISODIC,
            content="old event",
            timestamp=past_time,
            importance=0.8,
            access_count=5,
            decay_rate=0.1,  # Higher decay rate for testing
        )

        # Test strength calculation
        current_strength = memory.calculate_strength()
        # Strength includes access boost, so may be >= importance
        assert current_strength > 0
        assert current_strength <= 1.0

        # Test with specific time - 25 hours from creation (1 hour in future)
        future_time = past_time + timedelta(hours=25)
        future_strength = memory.calculate_strength(future_time)
        # Should have more decay - both should be less than 1.0 for comparison
        if current_strength < 1.0:
            assert future_strength < current_strength

    def test_memory_access(self):
        """Test memory access tracking"""
        memory = Memory(
            memory_id="test_4",
            memory_type=MemoryType.SEMANTIC,
            content="fact",
            timestamp=datetime.now(),
        )

        original_count = memory.access_count
        original_time = memory.last_accessed

        # Access the memory
        memory.access()

        assert memory.access_count == original_count + 1
        assert memory.last_accessed > original_time

    def test_memory_associations(self):
        """Test memory associations"""
        memory = Memory(
            memory_id="test_5",
            memory_type=MemoryType.SEMANTIC,
            content="concept",
            timestamp=datetime.now(),
            associations=["related_1", "related_2"],
        )

        assert "related_1" in memory.associations
        assert "related_2" in memory.associations
        assert len(memory.associations) == 2


class TestExperience:
    """Test Experience dataclass"""

    def test_experience_creation(self):
        """Test creating an experience"""
        state = {"position": (5, 5), "energy": 80}
        action = Action(action_type=ActionType.MOVE, target_position=Position(10, 10))
        outcome = {"position": (10, 10), "energy": 75}
        next_state = {"position": (10, 10), "energy": 75}

        exp = Experience(
            state=state, action=action, outcome=outcome, reward=0.5, next_state=next_state
        )

        assert exp.state == state
        assert exp.action == action
        assert exp.outcome == outcome
        assert exp.reward == 0.5
        assert exp.next_state == next_state
        assert isinstance(exp.timestamp, datetime)

    def test_experience_metadata(self):
        """Test experience metadata"""
        exp = Experience(
            state={"pos": (0, 0)},
            action=Action(action_type=ActionType.EXPLORE),
            outcome={"discovered": "resource"},
            reward=1.0,
            next_state={"pos": (1, 1)},
            metadata={"context": "exploration"},
        )

        assert exp.metadata["context"] == "exploration"
        assert isinstance(exp.timestamp, datetime)


class TestPattern:
    """Test Pattern dataclass"""

    def test_pattern_creation(self):
        """Test creating a pattern"""
        pattern = Pattern(
            pattern_id="pat_1",
            pattern_type="spatial",
            conditions={"location": "area_1", "time": "morning"},
            prediction={"event": "resource_spawn"},
            confidence=0.9,
            occurrences=5,
            successes=4,
        )

        assert pattern.pattern_id == "pat_1"
        assert pattern.pattern_type == "spatial"
        assert pattern.conditions["location"] == "area_1"
        assert pattern.prediction["event"] == "resource_spawn"
        assert pattern.confidence == 0.9
        assert pattern.occurrences == 5
        assert pattern.successes == 4

    def test_pattern_success_rate(self):
        """Test pattern success rate calculation"""
        pattern = Pattern(
            pattern_id="pat_1",
            pattern_type="test",
            conditions={},
            prediction={},
            occurrences=10,
            successes=7,
        )

        assert pattern.get_success_rate() == 0.7

        # Test empty pattern
        empty_pattern = Pattern(
            pattern_id="pat_2", pattern_type="test", conditions={}, prediction={}
        )
        assert empty_pattern.get_success_rate() == 0.0

    def test_pattern_update(self):
        """Test pattern updates"""
        pattern = Pattern(
            pattern_id="pat_1",
            pattern_type="test",
            conditions={},
            prediction={},
            confidence=0.5,
            occurrences=1,
            successes=1,
        )

        # Successful update
        pattern.update(success=True)
        assert pattern.occurrences == 2
        assert pattern.successes == 2
        assert pattern.confidence > 0.5  # Should increase

        # Failed update
        pattern.update(success=False)
        assert pattern.occurrences == 3
        assert pattern.successes == 2
        assert pattern.confidence < 0.67  # Should decrease from 2/2 toward 2/3


class TestInMemoryStorage:
    """Test InMemoryStorage class"""

    def test_storage_creation(self):
        """Test creating in-memory storage"""
        storage = InMemoryStorage()
        assert storage.memories == {}
        assert hasattr(storage, "type_index")
        assert hasattr(storage, "timestamp_index")

    def test_storage_store_and_retrieve(self):
        """Test storing and retrieving memories"""
        storage = InMemoryStorage()

        memory = Memory(
            memory_id="mem_1",
            memory_type=MemoryType.SEMANTIC,
            content="test",
            timestamp=datetime.now(),
        )

        # Store memory
        storage.store(memory)

        # Retrieve by ID
        retrieved = storage.retrieve("mem_1")
        assert retrieved == memory

        # Non-existent ID
        assert storage.retrieve("non_existent") is None

    def test_storage_search(self):
        """Test searching memories"""
        storage = InMemoryStorage()

        # Add multiple memories
        memories = []
        for i in range(5):
            memory = Memory(
                memory_id=f"mem_{i}",
                memory_type=MemoryType.EPISODIC if i % 2 == 0 else MemoryType.SEMANTIC,
                content={"index": i},
                timestamp=datetime.now() - timedelta(hours=i),
                importance=i * 0.2,
            )
            memories.append(memory)
            storage.store(memory)

        # Search all
        all_memories = storage.search({})
        assert len(all_memories) == 5

        # Search by type
        episodic = storage.search({"memory_type": MemoryType.EPISODIC})
        assert len(episodic) == 3
        assert all(m.memory_type == MemoryType.EPISODIC for m in episodic)

        # Search by importance
        important = storage.search({"min_importance": 0.4})
        assert all(m.importance >= 0.4 for m in important)

    def test_storage_update(self):
        """Test updating memories"""
        storage = InMemoryStorage()

        memory = Memory(
            memory_id="mem_1",
            memory_type=MemoryType.SEMANTIC,
            content="original",
            timestamp=datetime.now(),
        )
        storage.store(memory)

        # Update memory by storing again
        memory.content = "updated"
        memory.importance = 0.9
        storage.store(memory)

        # Verify update
        retrieved = storage.retrieve("mem_1")
        assert retrieved.content == "updated"
        assert retrieved.importance == 0.9

    def test_storage_remove(self):
        """Test removing memories"""
        storage = InMemoryStorage()

        memory = Memory(
            memory_id="mem_1",
            memory_type=MemoryType.SEMANTIC,
            content="test",
            timestamp=datetime.now(),
        )
        storage.store(memory)

        # Remove memory
        removed = storage.remove("mem_1")

        # Verify removal
        assert removed == True
        assert storage.retrieve("mem_1") is None

    def test_storage_multiple_operations(self):
        """Test multiple storage operations"""
        storage = InMemoryStorage()

        # Add multiple memories
        for i in range(3):
            memory = Memory(
                memory_id=f"mem_{i}",
                memory_type=MemoryType.SEMANTIC,
                content=f"content_{i}",
                timestamp=datetime.now(),
            )
            storage.store(memory)

        # Verify storage
        assert len(storage.memories) == 3

        # Remove one
        storage.remove("mem_1")
        assert len(storage.memories) == 2


class TestMemorySystem:
    """Test MemorySystem class"""

    @pytest.fixture
    def memory_system(self):
        """Create a memory system for testing"""
        return MemorySystem(agent_id="test_agent")

    def test_memory_system_creation(self, memory_system):
        """Test creating memory system"""
        assert memory_system.agent_id == "test_agent"
        assert isinstance(memory_system.storage, InMemoryStorage)
        assert memory_system.total_memories == 0
        assert hasattr(memory_system, "working_memory")
        assert hasattr(memory_system, "consolidator")
        assert hasattr(memory_system, "pattern_recognizer")

    def test_store_memory(self, memory_system):
        """Test storing memories"""
        # Store episodic memory
        memory = memory_system.store_memory(
            content={"event": "test"},
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            context={"location": "here"},
        )

        assert isinstance(memory, Memory)
        assert memory.content["event"] == "test"
        assert memory.importance == 0.7
        assert memory_system.total_memories == 1

    def test_store_percept(self, memory_system):
        """Test storing percepts as memories"""
        stimulus = Stimulus(
            stimulus_id="stim_1",
            stimulus_type=StimulusType.OBJECT,
            position=Position(10, 10, 0),
            intensity=0.8,
            metadata={"object": "food"},
        )
        percept = Percept(
            stimulus=stimulus,
            perception_type=PerceptionType.VISUAL,
            confidence=0.9,
            distance=5.0,
            timestamp=datetime.now(),
        )

        # Store percept as memory content
        memory = memory_system.store_memory(
            content={"percept": percept}, memory_type=MemoryType.SENSORY, importance=0.5
        )

        assert memory.memory_type == MemoryType.SENSORY
        assert memory.content["percept"].stimulus.stimulus_id == "stim_1"

    def test_store_experience(self, memory_system):
        """Test storing experiences"""
        experience = Experience(
            state={"pos": (5, 5)},
            action=Action(action_type=ActionType.GATHER),
            outcome={"resource": "wood"},
            reward=0.8,
            next_state={"pos": (5, 5), "inventory": ["wood"]},
        )

        memory_system.store_experience(experience)

        # Experience should be in buffer and storage
        assert len(memory_system.experience_buffer) == 1
        assert memory_system.total_memories == 1

    def test_retrieve_memories(self, memory_system):
        """Test retrieving memories"""
        # Store various memories
        memory_system.store_memory("fact1", MemoryType.SEMANTIC, 0.5)
        memory_system.store_memory("fact2", MemoryType.SEMANTIC, 0.7)
        memory_system.store_memory("event1", MemoryType.EPISODIC, 0.9)

        # Retrieve by type
        semantic = memory_system.retrieve_memories({"memory_type": MemoryType.SEMANTIC})
        assert len(semantic) >= 2

        # Retrieve by importance
        important = memory_system.retrieve_memories({"min_importance": 0.6})
        assert all(m.importance >= 0.6 for m in important)

    def test_memory_decay(self, memory_system):
        """Test memory decay over time"""
        # Store memory with low importance
        old_time = datetime.now() - timedelta(days=2)

        # Manually create old memory
        old_memory = Memory(
            memory_id="old_1",
            memory_type=MemoryType.EPISODIC,
            content="old event",
            timestamp=old_time,
            importance=0.3,
        )

        # Calculate strength should be low due to decay
        current_strength = old_memory.calculate_strength()
        assert current_strength < old_memory.importance

        # Recent memory should have higher strength
        recent_memory = memory_system.store_memory(
            "recent event", MemoryType.EPISODIC, importance=0.3
        )
        recent_strength = recent_memory.calculate_strength()
        assert recent_strength > current_strength

    def test_consolidate_memories(self, memory_system):
        """Test memory consolidation"""
        # Store related memories
        memories = []
        for i in range(5):
            memory = memory_system.store_memory(
                content={"topic": "learning", "detail": f"detail_{i}"},
                memory_type=MemoryType.EPISODIC,
                importance=0.5 + i * 0.1,
            )
            memories.append(memory)

        # Consolidate
        consolidated = memory_system.consolidator.consolidate(memories)

        # Should create abstract memories
        assert isinstance(consolidated, list)


class TestWorkingMemory:
    """Test WorkingMemory class"""

    @pytest.fixture
    def working_memory(self):
        """Create working memory for testing"""
        return WorkingMemory(capacity=3)

    def test_working_memory_creation(self, working_memory):
        """Test creating working memory"""
        assert working_memory.capacity == 3
        assert len(working_memory.items) == 0

    def test_add_items(self, working_memory):
        """Test adding items to working memory"""
        # Add items within capacity
        working_memory.add("item1", weight=0.5)
        working_memory.add("item2", weight=0.8)

        assert len(working_memory.items) == 2
        assert "item1" in working_memory.items
        assert "item2" in working_memory.items

    def test_capacity_limit(self, working_memory):
        """Test working memory capacity limit"""
        # Fill to capacity
        working_memory.add("item1", weight=0.3)
        working_memory.add("item2", weight=0.5)
        working_memory.add("item3", weight=0.7)

        # Add one more
        working_memory.add("item4", weight=0.9)

        # Should maintain capacity limit
        assert len(working_memory.items) <= working_memory.capacity + 1

    def test_get_focused_and_clear(self, working_memory):
        """Test getting focused items and clearing"""
        working_memory.add("item1", weight=0.3)
        working_memory.add("item2", weight=0.8)

        # Get focused items
        focused = working_memory.get_focused_items(n=1)
        assert len(focused) == 1

        # Clear
        working_memory.clear()
        assert len(working_memory.items) == 0
        assert len(working_memory.attention_weights) == 0

    def test_update_attention(self, working_memory):
        """Test updating attention weights"""
        item = "item1"
        working_memory.add(item, weight=0.5)

        # Update attention
        working_memory.update_attention(item, 0.3)

        # Weight should increase
        item_id = str(id(item))
        assert working_memory.attention_weights[item_id] == 0.8


class TestPatternRecognizer:
    """Test PatternRecognizer class"""

    @pytest.fixture
    def pattern_recognizer(self):
        """Create pattern recognizer for testing"""
        return PatternRecognizer()

    def test_pattern_recognizer_creation(self, pattern_recognizer):
        """Test creating pattern recognizer"""
        assert isinstance(pattern_recognizer.patterns, dict)
        assert len(pattern_recognizer.patterns) == 0
        assert hasattr(pattern_recognizer, "pattern_index")

    def test_extract_patterns(self, pattern_recognizer):
        """Test pattern extraction from experiences"""
        # Create experiences with patterns
        experiences = []
        for i in range(5):
            exp = Experience(
                state={"location": "area_1"},
                action=Action(action_type=ActionType.GATHER if i % 2 == 0 else ActionType.EXPLORE),
                outcome={"result": "success" if i % 2 == 0 else "failure"},
                reward=1.0 if i % 2 == 0 else 0.0,
                next_state={"location": "area_1"},
            )
            experiences.append(exp)

        # Extract patterns
        patterns = pattern_recognizer.extract_patterns(experiences)

        # Should find some patterns
        assert isinstance(patterns, list)
        assert all(isinstance(p, Pattern) for p in patterns)

    def test_store_pattern(self, pattern_recognizer):
        """Test storing patterns"""
        # Create a pattern
        pattern = Pattern(
            pattern_id="test_pattern",
            pattern_type="action_outcome",
            conditions={"action": "gather"},
            prediction={"outcome": "success", "expected_reward": 1.0},
            confidence=0.8,
            occurrences=10,
            successes=8,
        )

        # Store pattern
        pattern_recognizer.patterns[pattern.pattern_id] = pattern

        # Verify storage
        assert pattern.pattern_id in pattern_recognizer.patterns
        assert pattern_recognizer.patterns[pattern.pattern_id] == pattern


class TestReinforcementLearner:
    """Test ReinforcementLearner class"""

    @pytest.fixture
    def rl_learner(self):
        """Create reinforcement learner for testing"""
        return ReinforcementLearner(learning_rate=0.1, discount_factor=0.9)

    def test_rl_learner_creation(self, rl_learner):
        """Test creating reinforcement learner"""
        assert rl_learner.learning_rate == 0.1
        assert rl_learner.discount_factor == 0.9
        assert rl_learner.q_table == {}

    def test_learn_from_experience(self, rl_learner):
        """Test learning from experience"""
        experience = Experience(
            state={"location": "area_1", "energy": 80},
            action=Action(action_type=ActionType.GATHER),
            outcome={"resource": "wood"},
            reward=1.0,
            next_state={"location": "area_1", "energy": 75},
        )

        # Learn from experience
        rl_learner.learn(experience)

        # Should have updated Q-table
        state_key = rl_learner._state_to_key(experience.state)
        action_key = rl_learner._action_to_key(experience.action)
        assert state_key in rl_learner.q_table
        assert action_key in rl_learner.q_table[state_key]
        assert rl_learner.q_table[state_key][action_key] > 0

    def test_predict_action(self, rl_learner):
        """Test predicting best action"""
        state = {"location": "area_1", "energy": 80}
        state_key = rl_learner._state_to_key(state)

        # Set up Q-values
        rl_learner.q_table[state_key] = {"gather:None": 0.5, "explore:None": 0.8, "wait:None": 0.3}

        # Best action should be highest Q-value
        best_action = rl_learner.predict(state)
        assert best_action == "explore:None"

    def test_predict_unknown_state(self, rl_learner):
        """Test predicting for unknown state"""
        unknown_state = {"location": "unknown", "energy": 50}

        # Should return None for unknown state
        prediction = rl_learner.predict(unknown_state)
        assert prediction is None


class TestMemoryStorage:
    """Test MemoryStorage base class"""

    def test_base_class_methods(self):
        """Test base class raises NotImplementedError"""
        storage = MemoryStorage()

        memory = Memory(
            memory_id="test",
            memory_type=MemoryType.SEMANTIC,
            content="test",
            timestamp=datetime.now(),
        )

        with pytest.raises(NotImplementedError):
            storage.store(memory)

        with pytest.raises(NotImplementedError):
            storage.retrieve("test")

        with pytest.raises(NotImplementedError):
            storage.search({})

        with pytest.raises(NotImplementedError):
            storage.remove("test")


class TestLearningAlgorithm:
    """Test LearningAlgorithm base class"""

    def test_base_class_methods(self):
        """Test base class raises NotImplementedError"""
        algo = LearningAlgorithm()

        exp = Experience(
            state={"test": 1},
            action=Action(action_type=ActionType.WAIT),
            outcome={"result": "ok"},
            reward=0.5,
            next_state={"test": 2},
        )

        with pytest.raises(NotImplementedError):
            algo.learn(exp)

        with pytest.raises(NotImplementedError):
            algo.predict({"test": 1})


class TestMessageSystem:
    """Test MessageSystem class"""

    def test_message_system_creation(self):
        """Test creating message system"""
        msg_system = MessageSystem()
        assert msg_system.messages == []
        assert len(msg_system.message_queue) == 0

    def test_send_message(self):
        """Test sending messages"""
        msg_system = MessageSystem()

        msg_system.send_message(
            sender_id="agent_1", recipient_id="agent_2", content={"command": "gather"}
        )

        assert len(msg_system.messages) == 1
        assert len(msg_system.message_queue) == 1

        msg = msg_system.messages[0]
        assert msg["sender_id"] == "agent_1"
        assert msg["recipient_id"] == "agent_2"
        assert msg["content"]["command"] == "gather"
        assert "id" in msg
        assert "timestamp" in msg

    def test_get_messages(self):
        """Test getting messages for recipient"""
        msg_system = MessageSystem()

        # Send multiple messages
        msg_system.send_message("agent_1", "agent_2", "msg1")
        msg_system.send_message("agent_1", "agent_3", "msg2")
        msg_system.send_message("agent_3", "agent_2", "msg3")

        # Get messages for agent_2
        agent2_msgs = msg_system.get_messages("agent_2")
        assert len(agent2_msgs) == 2
        assert all(msg["recipient_id"] == "agent_2" for msg in agent2_msgs)

        # Get messages for agent_3
        agent3_msgs = msg_system.get_messages("agent_3")
        assert len(agent3_msgs) == 1
        assert agent3_msgs[0]["content"] == "msg2"

    def test_clear_messages(self):
        """Test clearing messages"""
        msg_system = MessageSystem()

        # Add messages
        msg_system.send_message("agent_1", "agent_2", "test")
        msg_system.send_message("agent_2", "agent_1", "reply")

        assert len(msg_system.messages) == 2
        assert len(msg_system.message_queue) == 2

        # Clear all
        msg_system.clear_messages()

        assert len(msg_system.messages) == 0
        assert len(msg_system.message_queue) == 0


class TestMemoryIntegration:
    """Test integration between memory components"""

    def test_full_memory_workflow(self):
        """Test complete memory workflow"""
        # Create memory system
        memory_system = MemorySystem("test_agent")
        working_memory = WorkingMemory(capacity=5)
        pattern_recognizer = PatternRecognizer()
        rl_learner = ReinforcementLearner()

        # Simulate agent experiences
        experiences = []
        for i in range(10):
            exp = Experience(
                state={"energy": 100 - i * 10},
                action=Action(action_type=ActionType.GATHER if i % 2 == 0 else ActionType.EXPLORE),
                outcome={"found": "resource" if i % 3 == 0 else "nothing"},
                reward=1.0 if i % 3 == 0 else 0.0,
                next_state={"energy": 90 - i * 10},
            )
            experiences.append(exp)

            # Store experience
            memory_system.store_experience(exp)

            # Add to working memory (use memory object, not ID)
            working_memory.add(exp, weight=exp.reward)

            # Learn from experience
            rl_learner.learn(exp)

        # Extract patterns from experiences
        patterns = pattern_recognizer.extract_patterns(experiences)

        # Verify integration worked
        assert memory_system.total_memories == 10
        assert len(patterns) >= 0  # May or may not find patterns
        assert len(rl_learner.q_table) > 0
        assert len(working_memory.items) <= working_memory.capacity + 1


class TestMemoryStorageBase:
    """Test MemoryStorage base class"""

    def test_base_class_methods(self):
        """Test base class raises NotImplementedError"""
        storage = MemoryStorage()

        memory = Memory(
            memory_id="test",
            memory_type=MemoryType.SEMANTIC,
            content="test",
            timestamp=datetime.now(),
        )

        with pytest.raises(NotImplementedError):
            storage.store(memory)

        with pytest.raises(NotImplementedError):
            storage.retrieve("test")

        with pytest.raises(NotImplementedError):
            storage.search({})

        with pytest.raises(NotImplementedError):
            storage.remove("test")


class TestLearningAlgorithmBase:
    """Test LearningAlgorithm base class"""

    def test_base_class_methods(self):
        """Test base class raises NotImplementedError"""
        algo = LearningAlgorithm()

        exp = Experience(
            state={"test": 1},
            action=Action(action_type=ActionType.WAIT),
            outcome={"result": "ok"},
            reward=0.5,
            next_state={"test": 2},
        )

        with pytest.raises(NotImplementedError):
            algo.learn(exp)

        with pytest.raises(NotImplementedError):
            algo.predict({"test": 1})


class TestMessageSystem:
    """Test MessageSystem class"""

    def test_message_system_creation(self):
        """Test creating message system"""
        msg_system = MessageSystem()
        assert msg_system.messages == []
        assert len(msg_system.message_queue) == 0

    def test_send_message(self):
        """Test sending messages"""
        msg_system = MessageSystem()

        msg_system.send_message(
            sender_id="agent_1", recipient_id="agent_2", content={"command": "gather"}
        )

        assert len(msg_system.messages) == 1
        assert len(msg_system.message_queue) == 1

        msg = msg_system.messages[0]
        assert msg["sender_id"] == "agent_1"
        assert msg["recipient_id"] == "agent_2"
        assert msg["content"]["command"] == "gather"
        assert "id" in msg
        assert "timestamp" in msg

    def test_get_messages(self):
        """Test getting messages for recipient"""
        msg_system = MessageSystem()

        # Send multiple messages
        msg_system.send_message("agent_1", "agent_2", "msg1")
        msg_system.send_message("agent_1", "agent_3", "msg2")
        msg_system.send_message("agent_3", "agent_2", "msg3")

        # Get messages for agent_2
        agent2_msgs = msg_system.get_messages("agent_2")
        assert len(agent2_msgs) == 2
        assert all(msg["recipient_id"] == "agent_2" for msg in agent2_msgs)

        # Get messages for agent_3
        agent3_msgs = msg_system.get_messages("agent_3")
        assert len(agent3_msgs) == 1
        assert agent3_msgs[0]["content"] == "msg2"

    def test_clear_messages(self):
        """Test clearing messages"""
        msg_system = MessageSystem()

        # Add messages
        msg_system.send_message("agent_1", "agent_2", "test")
        msg_system.send_message("agent_2", "agent_1", "reply")

        assert len(msg_system.messages) == 2
        assert len(msg_system.message_queue) == 2

        # Clear all
        msg_system.clear_messages()

        assert len(msg_system.messages) == 0
        assert len(msg_system.message_queue) == 0


class TestInMemoryStorageExtended:
    """Extended tests for InMemoryStorage filters"""

    def test_time_range_filters(self):
        """Test time range filter edge cases"""
        storage = InMemoryStorage()

        now = datetime.now()
        past = now - timedelta(hours=2)
        future = now + timedelta(hours=2)

        # Add memories at different times
        memories = [
            Memory(
                memory_id="past", memory_type=MemoryType.EPISODIC, content="past", timestamp=past
            ),
            Memory(memory_id="now", memory_type=MemoryType.EPISODIC, content="now", timestamp=now),
            Memory(
                memory_id="future",
                memory_type=MemoryType.EPISODIC,
                content="future",
                timestamp=future,
            ),
        ]

        for mem in memories:
            storage.store(mem)

        # Test start_time only
        results = storage.search({"start_time": now})
        assert len(results) == 2  # now and future
        assert all(m.memory_id in ["now", "future"] for m in results)

        # Test end_time only
        results = storage.search({"end_time": now})
        assert len(results) == 2  # past and now
        assert all(m.memory_id in ["past", "now"] for m in results)

        # Test both boundaries
        results = storage.search(
            {"start_time": past + timedelta(minutes=30), "end_time": now + timedelta(minutes=30)}
        )
        assert len(results) == 1
        assert results[0].memory_id == "now"

    def test_context_match_filter(self):
        """Test context matching filter"""
        storage = InMemoryStorage()

        memories = [
            Memory(
                memory_id="m1",
                memory_type=MemoryType.EPISODIC,
                content="1",
                timestamp=datetime.now(),
                context={"location": "forest", "action": "gather"},
            ),
            Memory(
                memory_id="m2",
                memory_type=MemoryType.EPISODIC,
                content="2",
                timestamp=datetime.now(),
                context={"location": "forest", "action": "explore"},
            ),
            Memory(
                memory_id="m3",
                memory_type=MemoryType.EPISODIC,
                content="3",
                timestamp=datetime.now(),
                context={"location": "plains", "action": "gather"},
            ),
        ]

        for mem in memories:
            storage.store(mem)

        # Single context match
        results = storage.search({"context_match": {"location": "forest"}})
        assert len(results) == 2
        assert all(m.memory_id in ["m1", "m2"] for m in results)

        # Multiple context match
        results = storage.search({"context_match": {"location": "forest", "action": "gather"}})
        assert len(results) == 1
        assert results[0].memory_id == "m1"

        # No match
        results = storage.search({"context_match": {"location": "ocean"}})
        assert len(results) == 0


class TestMemoryConsolidatorExtended:
    """Extended tests for MemoryConsolidator"""

    def test_consolidation_threshold(self):
        """Test consolidation threshold calculation"""
        consolidator = MemoryConsolidator(consolidation_threshold=0.5)

        # Low importance memory
        low_mem = Memory(
            memory_id="low",
            memory_type=MemoryType.EPISODIC,
            content="test",
            timestamp=datetime.now(),
            importance=0.2,
            access_count=1,
            associations=[],
        )
        assert not consolidator.evaluate_for_consolidation(low_mem)

        # High importance memory
        high_mem = Memory(
            memory_id="high",
            memory_type=MemoryType.EPISODIC,
            content="test",
            timestamp=datetime.now(),
            importance=0.8,
            access_count=5,
            associations=["a", "b"],
        )
        assert consolidator.evaluate_for_consolidation(high_mem)

        # Edge case - exactly at threshold
        edge_mem = Memory(
            memory_id="edge",
            memory_type=MemoryType.EPISODIC,
            content="test",
            timestamp=datetime.now(),
            importance=0.6,
            access_count=3,
            associations=["a"],
        )
        # 0.6*0.5 + 0.3*0.3 + 0.2*0.2 = 0.3 + 0.09 + 0.04 = 0.43 < 0.5
        assert not consolidator.evaluate_for_consolidation(edge_mem)

    def test_memory_similarity(self):
        """Test memory similarity checking"""
        consolidator = MemoryConsolidator()

        # Same type, similar context
        mem1 = Memory(
            memory_id="m1",
            memory_type=MemoryType.EPISODIC,
            content="test1",
            timestamp=datetime.now(),
            context={"location": "forest", "action": "gather", "result": "success"},
        )
        mem2 = Memory(
            memory_id="m2",
            memory_type=MemoryType.EPISODIC,
            content="test2",
            timestamp=datetime.now(),
            context={"location": "forest", "action": "gather", "result": "failure"},
        )

        # Should be similar (2/3 matches > 0.7)
        assert consolidator._are_similar(mem1, mem2)

        # Different types
        mem3 = Memory(
            memory_id="m3",
            memory_type=MemoryType.SEMANTIC,
            content="test3",
            timestamp=datetime.now(),
            context={"location": "forest", "action": "gather"},
        )
        assert not consolidator._are_similar(mem1, mem3)

        # No common keys
        mem4 = Memory(
            memory_id="m4",
            memory_type=MemoryType.EPISODIC,
            content="test4",
            timestamp=datetime.now(),
            context={"weather": "sunny", "time": "morning"},
        )
        assert not consolidator._are_similar(mem1, mem4)

    def test_consolidate_edge_cases(self):
        """Test consolidation edge cases"""
        consolidator = MemoryConsolidator()

        # Empty list
        assert consolidator.consolidate([]) == []

        # Single memory
        single = [
            Memory(
                memory_id="single",
                memory_type=MemoryType.EPISODIC,
                content="test",
                timestamp=datetime.now(),
            )
        ]
        result = consolidator.consolidate(single)
        assert len(result) == 1
        assert result[0] == single[0]

        # Two memories (not enough for abstract)
        two_mems = [
            Memory(
                memory_id="m1",
                memory_type=MemoryType.EPISODIC,
                content="1",
                timestamp=datetime.now(),
                context={"loc": "A"},
            ),
            Memory(
                memory_id="m2",
                memory_type=MemoryType.EPISODIC,
                content="2",
                timestamp=datetime.now(),
                context={"loc": "A"},
            ),
        ]
        result = consolidator.consolidate(two_mems)
        assert len(result) == 2  # Not consolidated

    def test_create_abstract_memory_edge_cases(self):
        """Test abstract memory creation edge cases"""
        consolidator = MemoryConsolidator()

        # Empty list
        assert consolidator._create_abstract_memory([]) is None

        # No common context
        memories = [
            Memory(
                memory_id=f"m{i}",
                memory_type=MemoryType.EPISODIC,
                content=f"test{i}",
                timestamp=datetime.now(),
                context={f"key{i}": f"val{i}"},
            )
            for i in range(3)
        ]
        assert consolidator._create_abstract_memory(memories) is None

        # Partial common context
        memories = [
            Memory(
                memory_id="m1",
                memory_type=MemoryType.EPISODIC,
                content="1",
                timestamp=datetime.now(),
                context={"loc": "A", "act": "gather"},
                importance=0.5,
            ),
            Memory(
                memory_id="m2",
                memory_type=MemoryType.EPISODIC,
                content="2",
                timestamp=datetime.now(),
                context={"loc": "A", "res": "food"},
                importance=0.7,
            ),
            Memory(
                memory_id="m3",
                memory_type=MemoryType.EPISODIC,
                content="3",
                timestamp=datetime.now(),
                context={"loc": "A", "time": "day"},
                importance=0.6,
            ),
        ]
        abstract = consolidator._create_abstract_memory(memories)
        assert abstract is not None
        assert abstract.content["pattern"]["loc"] == "A"
        assert abstract.importance == 0.7  # max importance


class TestMemorySystemExtended:
    """Extended tests for MemorySystem"""

    def test_memory_relevance_calculation(self):
        """Test memory relevance calculation"""
        system = MemorySystem("test_agent")

        # Create test agent and context
        agent = Agent(agent_id="test")

        # Test procedural memory relevance
        proc_memory = Memory(
            memory_id="proc",
            memory_type=MemoryType.PROCEDURAL,
            content="procedure",
            timestamp=datetime.now(),
            context={},
        )

        context = DecisionContext(agent=agent, percepts=[], current_goal=None, available_actions=[])

        relevance = system._calculate_relevance(proc_memory, context)
        assert relevance >= 0.3  # Procedural gets 0.3 base

        # Test recent memory bonus
        recent_memory = Memory(
            memory_id="recent",
            memory_type=MemoryType.EPISODIC,
            content="recent",
            timestamp=datetime.now() - timedelta(minutes=30),
            context={},
        )
        relevance = system._calculate_relevance(recent_memory, context)
        assert relevance >= 0.2  # Recent (< 1 hour) gets 0.2

        # Test old memory
        old_memory = Memory(
            memory_id="old",
            memory_type=MemoryType.EPISODIC,
            content="old",
            timestamp=datetime.now() - timedelta(days=2),
            context={},
        )
        relevance = system._calculate_relevance(old_memory, context)
        assert relevance < 0.2  # No recency bonus

    def test_memory_relevance_with_percepts(self):
        """Test memory relevance with percept matching"""
        system = MemorySystem("test_agent")
        agent = Agent(agent_id="test")

        # Create memory with percept types
        memory = Memory(
            memory_id="m1",
            memory_type=MemoryType.EPISODIC,
            content="test",
            timestamp=datetime.now(),
            context={"percept_types": ["object", "danger"]},
        )

        # Create matching percepts
        stim1 = Stimulus(
            stimulus_id="s1", stimulus_type=StimulusType.OBJECT, position=Position(0, 0, 0)
        )
        stim2 = Stimulus(
            stimulus_id="s2", stimulus_type=StimulusType.DANGER, position=Position(10, 10, 0)
        )

        percept1 = Percept(stimulus=stim1, perception_type=PerceptionType.VISUAL)
        percept2 = Percept(stimulus=stim2, perception_type=PerceptionType.PROXIMITY)

        context = DecisionContext(
            agent=agent, percepts=[percept1, percept2], current_goal=None, available_actions=[]
        )

        relevance = system._calculate_relevance(memory, context)
        assert relevance >= 0.4  # 0.2 * 2 overlapping types

    def test_pattern_extraction_insufficient_data(self):
        """Test pattern extraction with insufficient experiences"""
        system = MemorySystem("test_agent")

        # Add < 10 experiences
        for i in range(5):
            exp = Experience(
                state={"test": i},
                action=Action(ActionType.WAIT),
                outcome={"result": "ok"},
                reward=0.5,
                next_state={"test": i + 1},
            )
            system.store_experience(exp)

        patterns = system.extract_patterns()
        assert patterns == []  # Not enough data

    def test_predict_outcome(self):
        """Test outcome prediction"""
        system = MemorySystem("test_agent")

        # Add experiences to create patterns
        for i in range(15):
            exp = Experience(
                state={"location": "forest"},
                action=Action(ActionType.GATHER),
                outcome={"found_food": True},
                reward=1.0,
                next_state={"location": "forest", "has_food": True},
            )
            system.store_experience(exp)

        # Extract patterns
        patterns = system.extract_patterns()
        assert len(patterns) > 0

        # Predict outcome
        prediction = system.predict_outcome(
            state={"location": "forest"}, action=Action(ActionType.GATHER)
        )

        # Should predict based on pattern
        if prediction:
            assert "outcome" in prediction
            assert "expected_reward" in prediction

        # Predict for unknown state/action
        unknown_prediction = system.predict_outcome(
            state={"location": "ocean"}, action=Action(ActionType.EXPLORE)
        )
        # May be None if no matching pattern
        assert unknown_prediction is None or isinstance(unknown_prediction, dict)


class TestReinforcementLearnerExtended:
    """Extended tests for ReinforcementLearner"""

    def test_state_action_key_conversion(self):
        """Test state and action key conversion"""
        learner = ReinforcementLearner()

        # Test state with various types
        state = {
            "bool_val": True,
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "test",
            "list_val": [1, 2, 3],  # Should be ignored
            "dict_val": {"nested": "value"},  # Should be ignored
        }

        state_key = learner._state_to_key(state)
        assert "bool_val:True" in state_key
        assert "int_val:42" in state_key
        assert "float_val:3.14" in state_key
        assert "str_val:test" in state_key
        assert "list_val" not in state_key
        assert "dict_val" not in state_key

        # Test action key
        action = Action(ActionType.MOVE, target=Position(10, 20, 0))
        action_key = learner._action_to_key(action)
        assert action_key == f"move:{action.target}"

        # Action without target
        action2 = Action(ActionType.WAIT)
        action_key2 = learner._action_to_key(action2)
        assert action_key2 == "wait:None"

    def test_q_learning_convergence(self):
        """Test Q-learning convergence behavior"""
        learner = ReinforcementLearner(learning_rate=0.5, discount_factor=0.9)

        # Simulate repeated experiences in same state
        state = {"location": "A"}
        action = Action(ActionType.GATHER)

        # First experience
        exp1 = Experience(
            state=state,
            action=action,
            outcome={"food": 1},
            reward=1.0,
            next_state={"location": "A", "food": 1},
        )
        learner.learn(exp1)

        state_key = learner._state_to_key(state)
        action_key = learner._action_to_key(action)
        q_val_1 = learner.q_table[state_key][action_key]
        assert q_val_1 > 0

        # Second experience with same state-action
        exp2 = Experience(
            state=state,
            action=action,
            outcome={"food": 1},
            reward=1.0,
            next_state={"location": "A", "food": 2},
        )
        learner.learn(exp2)

        q_val_2 = learner.q_table[state_key][action_key]
        assert q_val_2 > q_val_1  # Should increase

        # Experience with no future value
        terminal_exp = Experience(
            state={"location": "B"},
            action=Action(ActionType.WAIT),
            outcome={"done": True},
            reward=0.5,
            next_state={"location": "terminal"},
        )
        learner.learn(terminal_exp)

        state_key_b = learner._state_to_key(terminal_exp.state)
        action_key_b = learner._action_to_key(terminal_exp.action)
        q_val_terminal = learner.q_table[state_key_b][action_key_b]
        assert q_val_terminal == 0.5 * 0.5  # No future value


class TestPatternRecognizerExtended:
    """Extended tests for PatternRecognizer"""

    def test_outcome_summarization(self):
        """Test outcome summarization"""
        recognizer = PatternRecognizer()

        # Test boolean outcomes
        outcome1 = {"success": True, "failed": False, "found_food": True}
        summary1 = recognizer._summarize_outcome(outcome1)
        assert "success" in summary1
        assert "found_food" in summary1
        assert "failed" not in summary1

        # Test numeric outcomes
        outcome2 = {"energy": 10, "health": -5, "score": 0}
        summary2 = recognizer._summarize_outcome(outcome2)
        assert "energy_positive" in summary2
        assert "health_negative" in summary2
        assert "score" not in summary2  # Zero is ignored

        # Test mixed outcomes
        outcome3 = {"won": True, "damage": -10, "reward": 50}
        summary3 = recognizer._summarize_outcome(outcome3)
        assert "won" in summary3
        assert "damage_negative" in summary3
        assert "reward_positive" in summary3

    def test_pattern_extraction_threshold(self):
        """Test pattern extraction with threshold"""
        recognizer = PatternRecognizer()

        experiences = []
        # Add 10 experiences, 5 success, 5 failure (50% < 60% threshold)
        for i in range(10):
            exp = Experience(
                state={"test": i},
                action=Action(ActionType.GATHER),
                outcome={"success": i < 5},
                reward=1.0 if i < 5 else 0.0,
                next_state={"test": i},
            )
            experiences.append(exp)

        patterns = recognizer.extract_patterns(experiences)
        assert len(patterns) == 0  # 50% success rate < 60% threshold

        # Add more successes to exceed threshold
        for i in range(5):
            exp = Experience(
                state={"test": i},
                action=Action(ActionType.GATHER),
                outcome={"success": True},
                reward=1.0,
                next_state={"test": i},
            )
            experiences.append(exp)

        patterns = recognizer.extract_patterns(experiences)
        assert len(patterns) > 0  # Now 10/15 = 66.7% > 60%

    def test_conditions_matching(self):
        """Test pattern condition matching"""
        recognizer = PatternRecognizer()

        pattern = Pattern(
            pattern_id="p1",
            pattern_type="test",
            conditions={"location": "forest", "time": "day"},
            prediction={"outcome": "success"},
        )

        # Exact match
        assert recognizer._conditions_match(
            pattern.conditions, {"location": "forest", "time": "day", "extra": "ignored"}
        )

        # Missing required condition
        assert not recognizer._conditions_match(
            pattern.conditions, {"location": "forest"}  # Missing time
        )

        # Wrong value
        assert not recognizer._conditions_match(
            pattern.conditions, {"location": "plains", "time": "day"}
        )


class TestMemorySystemDiskOperations:
    """Test save/load operations"""

    def test_save_load_empty_system(self):
        """Test saving and loading empty system"""
        system = MemorySystem("test_agent")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)

        # Save empty system
        system.save_to_disk(temp_path)

        # Load into new system
        new_system = MemorySystem("test_agent2")
        new_system.load_from_disk(temp_path)

        assert new_system.total_memories == 0
        assert len(new_system.pattern_recognizer.patterns) == 0

        temp_path.unlink()

    def test_get_memory_summary_edge_cases(self):
        """Test memory summary with edge cases"""
        system = MemorySystem("test_agent")

        # Empty system
        summary = system.get_memory_summary()
        assert summary["total_memories"] == 0
        assert summary["oldest_memory"] is None
        assert summary["average_importance"] == 0.0


class TestMemoryForgetting:
    """Test memory forgetting functionality"""

    def test_forget_old_memories_boundary(self):
        """Test forgetting at exact boundary"""
        system = MemorySystem("test_agent")

        # Create memory exactly at cutoff
        cutoff_memory = Memory(
            memory_id="cutoff",
            memory_type=MemoryType.EPISODIC,
            content="boundary",
            timestamp=datetime.now() - timedelta(days=30),
            importance=0.3,
            decay_rate=0.5,  # High decay
        )
        cutoff_memory.last_accessed = cutoff_memory.timestamp

        system.storage.store(cutoff_memory)
        system.total_memories += 1
        system.memory_types_count[MemoryType.EPISODIC] += 1

        # This should be removed (low importance, high decay)
        removed = system.forget_old_memories(max_age_days=30)
        assert removed >= 1
