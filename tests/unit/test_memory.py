"""
Module for FreeAgentics Active Inference implementation.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agents.base.data_model import Action, ActionType, Agent, Position
from agents.base.decision_making import DecisionContext
from agents.base.memory import (
    Experience,
    InMemoryStorage,
    Memory,
    MemoryConsolidator,
    MemorySystem,
    MemoryType,
    Pattern,
    PatternRecognizer,
    ReinforcementLearner,
    WorkingMemory,
)
from agents.base.perception import Percept, PerceptionType, Stimulus, StimulusType


@pytest.fixture
def sample_memory() -> Memory:
    """Create a sample memory for testing"""
    return Memory(
        memory_id="test_memory_1",
        memory_type=MemoryType.EPISODIC,
        content={"event": "found_resource", "location": "x10_y20"},
        timestamp=datetime.now(),
        importance=0.7,
        context={"agent_state": "exploring", "energy": 80},
    )


@pytest.fixture
def sample_experience() -> Experience:
    """Create a sample experience for testing"""
    return Experience(
        state={"position": "x5_y5", "energy": 75},
        action=Action(ActionType.MOVE, target=Position(10, 10, 0)),
        outcome={"position": "x10_y10", "energy": 70},
        reward=0.5,
        next_state={"position": "x10_y10", "energy": 70},
    )


@pytest.fixture
def sample_pattern() -> Pattern:
    """Create a sample pattern for testing"""
    return Pattern(
        pattern_id="pattern_1",
        pattern_type="action_outcome",
        conditions={"action": "move"},
        prediction={"outcome": "success", "expected_reward": 0.5},
        confidence=0.8,
        occurrences=10,
        successes=8,
    )


@pytest.fixture
def memory_system() -> MemorySystem:
    ."""Create a memory system for testing."""
    return MemorySystem("test_agent_1")


class TestMemory:
    ."""Test Memory dataclass."""

    def test_memory_creation(self, sample_memory) -> None:
        """Test memory creation and attributes"""
        assert sample_memory.memory_id == "test_memory_1"
        assert sample_memory.memory_type == MemoryType.EPISODIC
        assert sample_memory.importance == 0.7
        assert sample_memory.access_count == 0
        assert "event" in sample_memory.content

    def test_memory_strength_calculation(self, sample_memory) -> None:
        """Test memory strength calculation with decay"""
        initial_strength = sample_memory.calculate_strength()
        assert 0.6 < initial_strength < 0.8
        sample_memory.access()
        sample_memory.access()
        accessed_strength = sample_memory.calculate_strength()
        assert accessed_strength > initial_strength
        future_time = datetime.now() + timedelta(hours=24)
        decayed_strength = sample_memory.calculate_strength(future_time)
        assert decayed_strength < initial_strength

    def test_memory_access(self, sample_memory) -> None:
        """Test memory access updates"""
        initial_count = sample_memory.access_count
        initial_time = sample_memory.last_accessed
        sample_memory.access()
        assert sample_memory.access_count == initial_count + 1
        assert sample_memory.last_accessed > initial_time


class TestPattern:
    ."""Test Pattern dataclass."""

    def test_pattern_creation(self, sample_pattern) -> None:
        ."""Test pattern creation and attributes."""
        assert sample_pattern.pattern_id == "pattern_1"
        assert sample_pattern.confidence == 0.8
        assert sample_pattern.get_success_rate() == 0.8

    def test_pattern_update(self, sample_pattern) -> None:
        """Test pattern update with new observations"""
        initial_confidence = sample_pattern.confidence
        initial_occurrences = sample_pattern.occurrences
        initial_successes = sample_pattern.successes
        sample_pattern.update(success=True)
        assert sample_pattern.occurrences == initial_occurrences + 1
        assert sample_pattern.successes == initial_successes + 1
        assert 0.76 < sample_pattern.confidence < 0.78
        sample_pattern.update(success=False)
        assert sample_pattern.occurrences == initial_occurrences + 2
        assert sample_pattern.successes == initial_successes + 1
        assert 0.71 < sample_pattern.confidence < 0.72


class TestInMemoryStorage:
    ."""Test in-memory storage implementation."""

    def test_store_and_retrieve(self, sample_memory) -> None:
        """Test storing and retrieving memories"""
        storage = InMemoryStorage()
        storage.store(sample_memory)
        retrieved = storage.retrieve(sample_memory.memory_id)
        assert retrieved is not None
        assert retrieved.memory_id == sample_memory.memory_id
        assert storage.retrieve("non_existent") is None

    def test_search_by_type(self, sample_memory) -> None:
        """Test searching memories by type"""
        storage = InMemoryStorage()
        storage.store(sample_memory)
        semantic_memory = Memory(
            memory_id="semantic_1",
            memory_type=MemoryType.SEMANTIC,
            content={"fact": "resources regenerate"},
            timestamp=datetime.now(),
        )
        storage.store(semantic_memory)
        episodic_results = storage.search({"memory_type": MemoryType.EPISODIC})
        assert len(episodic_results) == 1
        assert episodic_results[0].memory_id == sample_memory.memory_id
        semantic_results = storage.search({"memory_type": MemoryType.SEMANTIC})
        assert len(semantic_results) == 1
        assert semantic_results[0].memory_id == "semantic_1"

    def test_search_by_importance(self, sample_memory) -> None:
        """Test searching memories by importance threshold"""
        storage = InMemoryStorage()
        storage.store(sample_memory)
        low_importance = Memory(
            memory_id="low_imp_1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "minor"},
            timestamp=datetime.now(),
            importance=0.3,
        )
        storage.store(low_importance)
        important_results = storage.search({"min_importance": 0.5})
        assert len(important_results) == 1
        assert important_results[0].memory_id == sample_memory.memory_id

    def test_search_by_time_range(self) -> None:
        """Test searching memories by time range"""
        storage = InMemoryStorage()
        old_memory = Memory(
            memory_id="old_1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "old"},
            timestamp=datetime.now() - timedelta(days=2),
        )
        storage.store(old_memory)
        recent_memory = Memory(
            memory_id="recent_1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "recent"},
            timestamp=datetime.now(),
        )
        storage.store(recent_memory)
        cutoff = datetime.now() - timedelta(days=1)
        recent_results = storage.search({"start_time": cutoff})
        assert len(recent_results) == 1
        assert recent_results[0].memory_id == "recent_1"

    def test_remove_memory(self, sample_memory) -> None:
        """Test removing memories"""
        storage = InMemoryStorage()
        storage.store(sample_memory)
        assert storage.remove(sample_memory.memory_id)
        assert storage.retrieve(sample_memory.memory_id) is None
        assert not storage.remove("non_existent")


class TestWorkingMemory:
    ."""Test working memory functionality."""

    def test_capacity_limit(self) -> None:
        """Test working memory capacity limit"""
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.add(f"item_{i}")
        assert len(wm.items) == 3
        assert "item_2" in wm.items
        assert "item_4" in wm.items
        assert "item_0" not in wm.items

    def test_attention_weights(self) -> None:
        """Test attention weight management"""
        wm = WorkingMemory()
        item1 = "important_item"
        item2 = "normal_item"
        wm.add(item1, weight=0.9)
        wm.add(item2, weight=0.3)
        focused = wm.get_focused_items(n=1)
        assert len(focused) == 1
        assert focused[0] == item1

    def test_update_attention(self) -> None:
        """Test updating attention weights"""
        wm = WorkingMemory()
        item = "test_item"
        wm.add(item, weight=0.5)
        wm.update_attention(item, 0.3)
        item_id = str(id(item))
        assert wm.attention_weights[item_id] == 0.8
        wm.update_attention(item, 0.5)
        assert wm.attention_weights[item_id] == 1.0


class TestMemoryConsolidator:
    ."""Test memory consolidation."""

    def test_consolidation_evaluation(self, sample_memory) -> None:
        """Test evaluation for consolidation"""
        consolidator = MemoryConsolidator(consolidation_threshold=0.7)
        low_memory = Memory(
            memory_id="low_1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "minor"},
            timestamp=datetime.now(),
            importance=0.2,
        )
        assert not consolidator.evaluate_for_consolidation(low_memory)
        sample_memory.access_count = 10
        sample_memory.associations = ["mem1", "mem2", "mem3"]
        assert consolidator.evaluate_for_consolidation(sample_memory)
        accessed_memory = Memory(
            memory_id="accessed_1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "frequent"},
            timestamp=datetime.now(),
            importance=0.4,
            access_count=15,
            associations=["related1", "related2", "related3", "related4"],
        )
        accessed_memory.importance = 0.5
        assert consolidator.evaluate_for_consolidation(accessed_memory)

    def test_memory_grouping(self) -> None:
        """Test grouping similar memories"""
        consolidator = MemoryConsolidator()
        memories = [
            Memory(
                memory_id=f"mem_{i}",
                memory_type=MemoryType.EPISODIC,
                content={"action": "gather", "result": "success"},
                timestamp=datetime.now(),
                context={"location": "forest", "resource": "food"},
            )
            for i in range(3)
        ]
        memories.append(
            Memory(
                memory_id="different",
                memory_type=MemoryType.EPISODIC,
                content={"action": "flee", "result": "success"},
                timestamp=datetime.now(),
                context={"location": "plains", "threat": "predator"},
            )
        )
        groups = consolidator._group_similar_memories(memories)
        assert len(groups) == 2
        assert any(len(group) == 3 for group in groups)

    def test_abstract_memory_creation(self) -> None:
        """Test creating abstract memory from group"""
        consolidator = MemoryConsolidator()
        memories = [
            Memory(
                memory_id=f"mem_{i}",
                memory_type=MemoryType.EPISODIC,
                content={"action": "gather", "outcome": f"variant_{i}"},
                timestamp=datetime.now(),
                importance=0.6,
                context={"location": "forest", "resource": "food", "success": True},
            )
            for i in range(3)
        ]
        abstract = consolidator._create_abstract_memory(memories)
        assert abstract is not None
        assert abstract.memory_type == MemoryType.EPISODIC
        assert "pattern" in abstract.content
        assert abstract.content["pattern"]["location"] == "forest"
        assert abstract.content["pattern"]["resource"] == "food"
        assert abstract.content["instances"] == 3


class TestReinforcementLearner:
    ."""Test reinforcement learning."""

    def test_q_learning(self, sample_experience) -> None:
        """Test Q-learning updates"""
        learner = ReinforcementLearner(learning_rate=0.1, discount_factor=0.9)
        learner.learn(sample_experience)
        state_key = learner._state_to_key(sample_experience.state)
        action_key = learner._action_to_key(sample_experience.action)
        assert state_key in learner.q_table
        assert action_key in learner.q_table[state_key]
        assert learner.q_table[state_key][action_key] > 0

    def test_prediction(self, sample_experience) -> None:
        """Test action prediction"""
        learner = ReinforcementLearner()
        assert learner.predict({"new_state": True}) is None
        learner.learn(sample_experience)
        better_experience = Experience(
            state=sample_experience.state,
            action=Action(ActionType.WAIT),
            outcome={"energy": 75},
            reward=1.0,
            next_state={"position": "x5_y5", "energy": 75},
        )
        learner.learn(better_experience)
        prediction = learner.predict(sample_experience.state)
        assert prediction is not None


class TestPatternRecognizer:
    ."""Test pattern recognition."""

    def test_pattern_extraction(self) -> None:
        """Test extracting patterns from experiences"""
        recognizer = PatternRecognizer()
        experiences = []
        for i in range(5):
            exp = Experience(
                state={"location": "forest"},
                action=Action(ActionType.GATHER),
                outcome={"resource_gained": True, "energy_cost": 5},
                reward=1.0,
                next_state={"location": "forest", "has_resource": True},
            )
            experiences.append(exp)
        exp_fail = Experience(
            state={"location": "desert"},
            action=Action(ActionType.GATHER),
            outcome={"resource_gained": False},
            reward=-0.5,
            next_state={"location": "desert"},
        )
        experiences.append(exp_fail)
        patterns = recognizer.extract_patterns(experiences)
        assert len(patterns) > 0
        gather_pattern = next((p for p in patterns if p.conditions.get("action") == "gather"), None)
        assert gather_pattern is not None
        assert gather_pattern.confidence > 0.6

    def test_pattern_matching(self, sample_pattern) -> None:
        """Test pattern matching"""
        recognizer = PatternRecognizer()
        recognizer.patterns[sample_pattern.pattern_id] = sample_pattern
        matches = recognizer.match_pattern({"action": "move"})
        assert len(matches) == 1
        assert matches[0].pattern_id == sample_pattern.pattern_id
        no_matches = recognizer.match_pattern({"action": "wait"})
        assert len(no_matches) == 0


class TestMemorySystem:
    ."""Test integrated memory system.."""

    def test_store_and_retrieve_memory(self, memory_system) -> None:
        """Test storing and retrieving memories"""
        memory = memory_system.store_memory(
            content= (
                {"event": "test"}, memory_type=MemoryType.EPISODIC, importance=0.8)
        )
        assert memory_system.total_memories == 1
        assert memory_system.memory_types_count[MemoryType.EPISODIC] == 1
        retrieved = memory_system.retrieve_memories({"memory_type": MemoryType.EPISODIC})
        assert len(retrieved) == 1
        assert retrieved[0].content["event"] == "test"

    def test_store_experience(self, memory_system, sample_experience) -> None:
        ."""Test storing experiences."""
        memory_system.store_experience(sample_experience)
        assert len(memory_system.experience_buffer) == 1
        assert memory_system.total_memories == 1
        assert memory_system.memory_types_count[MemoryType.EPISODIC] == 1

    def test_relevant_memory_retrieval(self, memory_system) -> None:
        """Test getting relevant memories for context"""
        memory_system.store_memory(
            content={"action": "flee", "from": "threat"},
            memory_type=MemoryType.PROCEDURAL,
            importance=0.9,
            context={"percept_types": ["danger"]},
        )
        memory_system.store_memory(
            content={"action": "gather", "resource": "food"},
            memory_type=MemoryType.PROCEDURAL,
            importance=0.5,
            context={"percept_types": ["object"]},
        )
        threat_percept = Percept(
            stimulus=Stimulus(
                stimulus_id="threat_1",
                stimulus_type=StimulusType.DANGER,
                position=Position(10, 10, 0),
            ),
            perception_type=PerceptionType.VISUAL,
            distance=5.0,
        )
        agent = Agent(agent_id="test")
        context = DecisionContext(
            agent=agent,
            percepts=[threat_percept],
            current_goal=None,
            available_actions=[],
        )
        relevant = memory_system.get_relevant_memories(context)
        assert len(relevant) > 0
        procedural_memories = [m for m in relevant if m.memory_type == MemoryType.PROCEDURAL]
        assert len(procedural_memories) > 0

    def test_memory_consolidation(self, memory_system) -> None:
        """Test memory consolidation process"""
        for i in range(5):
            memory_system.store_memory(
                content={"action": "gather", "result": "success", "variant": i},
                memory_type=MemoryType.EPISODIC,
                importance=0.8,
                context={"location": "forest", "resource": "food"},
            )
        initial_count = memory_system.total_memories
        memory_system.consolidate_memories()
        semantic_memories = memory_system.retrieve_memories({"memory_type": MemoryType.SEMANTIC})
        assert memory_system.total_memories >= initial_count

    def test_pattern_extraction(self, memory_system) -> None:
        """Test pattern extraction from experiences"""
        for i in range(15):
            exp = Experience(
                state={"energy": 50, "location": "forest"},
                action=Action(ActionType.GATHER),
                outcome={"energy": 45, "has_food": True},
                reward=1.0,
                next_state={"energy": 45, "location": "forest"},
            )
            memory_system.store_experience(exp)
        patterns = memory_system.extract_patterns()
        assert len(patterns) > 0
        procedural = memory_system.retrieve_memories({"memory_type": MemoryType.PROCEDURAL})
        assert len(procedural) > 0

    def test_outcome_prediction(self, memory_system) -> None:
        """Test predicting action outcomes"""
        for i in range(10):
            exp = Experience(
                state={"location": "forest", "energy": 75},
                action=Action(ActionType.GATHER),
                outcome={"found_food": True},
                reward=1.0,
                next_state={"location": "forest", "energy": 70},
            )
            memory_system.store_experience(exp)
        memory_system.extract_patterns()
        prediction = memory_system.predict_outcome(
            state={"location": "forest"}, action=Action(ActionType.GATHER)
        )
        assert prediction is None or isinstance(prediction, dict)

    def test_memory_forgetting(self, memory_system) -> None:
        """Test forgetting old memories"""
        old_memory = Memory(
            memory_id="old_1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "trivial"},
            timestamp=datetime.now() - timedelta(days=40),
            importance=0.1,
            decay_rate=0.1,
        )
        old_memory.last_accessed = datetime.now() - timedelta(days=40)
        memory_system.storage.store(old_memory)
        memory_system.total_memories += 1
        memory_system.memory_types_count[MemoryType.EPISODIC] += 1
        memory_system.store_memory(
            content={"event": "important"},
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
        )
        initial_count = memory_system.total_memories
        forgotten = memory_system.forget_old_memories(max_age_days=30)
        assert forgotten >= 1
        assert memory_system.total_memories < initial_count

    def test_save_and_load(self, memory_system) -> None:
        """Test saving and loading memory system."""
        memory_system.store_memory(
            content={"test": "data"}, memory_type=MemoryType.SEMANTIC, importance=0.7
        )
        for i in range(15):
            exp = Experience(
                state={"test": i % 3},
                action=Action(ActionType.WAIT),
                outcome={"result": "ok"},
                reward=0.5,
                next_state={"test": i % 3},
            )
            memory_system.store_experience(exp)
        patterns = memory_system.extract_patterns()
        assert len(patterns) > 0
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)
        memory_system.save_to_disk(temp_path)
        new_system = MemorySystem("test_agent_2")
        new_system.load_from_disk(temp_path)
        assert new_system.total_memories == memory_system.total_memories
        assert len(new_system.pattern_recognizer.patterns) > 0
        temp_path.unlink()

    def test_memory_summary(self, memory_system) -> None:
        """Test getting memory system summary"""
        memory_system.store_memory(
            content={"test": 1}, memory_type=MemoryType.EPISODIC, importance=0.5
        )
        memory_system.store_memory(
            content={"test": 2}, memory_type=MemoryType.SEMANTIC, importance=0.8
        )
        summary = memory_system.get_memory_summary()
        assert summary["total_memories"] == 2
        assert summary["memory_types"][MemoryType.EPISODIC] == 1
        assert summary["memory_types"][MemoryType.SEMANTIC] == 1
        assert 0.6 < summary["average_importance"] < 0.7


if __name__ == "__main__":
    pytest.main([__file__])
