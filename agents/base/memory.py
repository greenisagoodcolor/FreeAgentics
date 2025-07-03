"""
Module for FreeAgentics Active Inference implementation.
"""

import heapq
import pickle
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np

from .decision_making import Action, DecisionContext

"""
Agent Memory and Learning System
This module provides memory and learning capabilities for agents including:
- Short-term and long-term memory structures
- Experience storage and retrieval
- Learning algorithms for pattern recognition
- Memory consolidation and decay
- Experience-based decision improvement
"""


class MemoryType(Enum):
    """Types of memories"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    SENSORY = "sensory"
    WORKING = "working"


class MemoryImportance(Enum):
    """Memory importance levels"""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class Memory:
    """Individual memory unit"""

    memory_id: str
    memory_type: MemoryType
    content: Any
    timestamp: datetime
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.01
    associations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_strength(self, current_time: Optional[datetime] = None) -> float:
        """Calculate current memory strength considering decay"""
        if current_time is None:
            current_time = datetime.now()
        time_elapsed = (current_time - self.timestamp).total_seconds() / 3600
        base_strength = self.importance
        decay_factor = np.exp(-self.decay_rate * time_elapsed)
        access_boost = min(self.access_count * 0.1, 0.5)
        recency_hours = (current_time - self.last_accessed).total_seconds() / 3600
        recency_factor = np.exp(-0.1 * recency_hours)
        strength = base_strength * decay_factor + access_boost * recency_factor
        return float(min(max(strength, 0.0), 1.0))

    def access(self) -> None:
        """Update memory access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class Experience:
    """Represents a complete experience for learning"""

    state: Dict[str, Any]
    action: Action
    outcome: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Represents a learned pattern"""

    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence: float = 0.5
    occurrences: int = 0
    successes: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, success: bool) -> None:
        """Update pattern statistics"""
        self.occurrences += 1
        if success:
            self.successes += 1
        alpha = self.successes + 1
        beta = self.occurrences - self.successes + 1
        self.confidence = alpha / (alpha + beta)
        self.last_updated = datetime.now()

    def get_success_rate(self) -> float:
        """Get pattern success rate"""
        if self.occurrences == 0:
            return 0.0
        return self.successes / self.occurrences


class MemoryStorage:
    """Base class for memory storage backends"""

    def store(self, memory: Memory) -> None:
        """Store a memory"""
        raise NotImplementedError

    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory"""
        raise NotImplementedError

    def search(self, criteria: Dict[str, Any]) -> List[Memory]:
        """Search memories by criteria"""
        raise NotImplementedError

    def remove(self, memory_id: str) -> bool:
        """Remove a memory"""
        raise NotImplementedError


class InMemoryStorage(MemoryStorage):
    """In-memory storage implementation"""

    def __init__(self) -> None:
        self.memories: Dict[str, Memory] = {}
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.timestamp_index: List[Tuple[datetime, str]] = []

    def store(self, memory: Memory) -> None:
        """Store memory with indexing"""
        self.memories[memory.memory_id] = memory
        self.type_index[memory.memory_type].add(memory.memory_id)
        heapq.heappush(self.timestamp_index, (memory.timestamp, memory.memory_id))

    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID"""
        return self.memories.get(memory_id)

    def search(self, criteria: Dict[str, Any]) -> List[Memory]:
        """Search memories by criteria using Chain of Responsibility pattern"""
        candidate_ids = self._get_candidate_ids(criteria)
        results = []

        for memory_id in candidate_ids:
            memory = self.memories[memory_id]
            if self._passes_all_filters(memory, criteria):
                results.append(memory)

        return results

    def _get_candidate_ids(self, criteria: Dict[str, Any]) -> set:
        """Get candidate memory IDs based on type filtering"""
        if "memory_type" in criteria:
            return self.type_index.get(criteria["memory_type"], set())
        else:
            return set(self.memories.keys())

    def _passes_all_filters(self, memory: Memory, criteria: Dict[str, Any]) -> bool:
        """Check if memory passes all filter criteria"""
        filter_checks = [
            self._check_importance_filter,
            self._check_time_range_filters,
            self._check_context_match_filter,
        ]

        for filter_check in filter_checks:
            if not filter_check(memory, criteria):
                return False

        return True

    def _check_importance_filter(self, memory: Memory, criteria: Dict[str, Any]) -> bool:
        """Check importance filter"""
        if "min_importance" in criteria:
            return memory.importance >= criteria["min_importance"]
        return True

    def _check_time_range_filters(self, memory: Memory, criteria: Dict[str, Any]) -> bool:
        """Check time range filters"""
        if "start_time" in criteria and memory.timestamp < criteria["start_time"]:
            return False
        if "end_time" in criteria and memory.timestamp > criteria["end_time"]:
            return False
        return True

    def _check_context_match_filter(self, memory: Memory, criteria: Dict[str, Any]) -> bool:
        """Check context match filter"""
        if "context_match" in criteria:
            context_match = criteria["context_match"]
            return all(memory.context.get(k) == v for k, v in context_match.items())
        return True

    def remove(self, memory_id: str) -> bool:
        """Remove memory from storage"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            del self.memories[memory_id]
            self.type_index[memory.memory_type].discard(memory_id)
            return True
        return False


class WorkingMemory:
    """Working memory for current context and active processing"""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        self.attention_weights: Dict[str, float] = {}

    def add(self, item: Any, weight: float = 1.0) -> None:
        """Add item to working memory"""
        item_id = str(id(item))
        self.items.append(item)
        self.attention_weights[item_id] = weight

    def get_focused_items(self, n: int = 3) -> List[Any]:
        """Get top n items by attention weight"""
        sorted_items = sorted(
            self.items,
            key=lambda x: self.attention_weights.get(str(id(x)), 0.0),
            reverse=True,
        )
        return sorted_items[:n]

    def update_attention(self, item: Any, weight_delta: float) -> None:
        """Update attention weight for an item"""
        item_id = str(id(item))
        if item_id in self.attention_weights:
            self.attention_weights[item_id] += weight_delta
            self.attention_weights[item_id] = max(0.0, min(1.0, self.attention_weights[item_id]))

    def clear(self) -> None:
        """Clear working memory"""
        self.items.clear()
        self.attention_weights.clear()


class MemoryConsolidator:
    """Consolidates short-term memories into long-term storage"""

    def __init__(self, consolidation_threshold: float = 0.7) -> None:
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_queue: List[Memory] = []

    def evaluate_for_consolidation(self, memory: Memory) -> bool:
        """Determine if memory should be consolidated"""
        importance_factor = memory.importance
        access_factor = min(memory.access_count / 10, 1.0)
        association_factor = min(len(memory.associations) / 5, 1.0)
        consolidation_score = (
            importance_factor * 0.5 + access_factor * 0.3 + association_factor * 0.2
        )
        return consolidation_score >= self.consolidation_threshold

    def consolidate(self, memories: List[Memory]) -> List[Memory]:
        """Consolidate related memories into more abstract representations"""
        consolidated = []
        groups = self._group_similar_memories(memories)
        for group in groups:
            if len(group) >= 3:
                abstract_memory = self._create_abstract_memory(group)
                if abstract_memory:
                    consolidated.append(abstract_memory)
            else:
                consolidated.extend(group)
        return consolidated

    def _group_similar_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """Group memories by similarity"""
        groups = []
        used = set()
        for i, memory1 in enumerate(memories):
            if i in used:
                continue
            group = [memory1]
            used.add(i)
            for j, memory2 in enumerate(memories[i + 1 :], i + 1):
                if j in used:
                    continue
                if self._are_similar(memory1, memory2):
                    group.append(memory2)
                    used.add(j)
            groups.append(group)
        return groups

    def _are_similar(self, memory1: Memory, memory2: Memory) -> bool:
        """Check if two memories are similar enough to group"""
        if memory1.memory_type != memory2.memory_type:
            return False
        common_keys = set(memory1.context.keys()) & set(memory2.context.keys())
        if not common_keys:
            return False
        matches = sum(1 for k in common_keys if memory1.context[k] == memory2.context[k])
        return matches / len(common_keys) > 0.7

    def _create_abstract_memory(self, memories: List[Memory]) -> Optional[Memory]:
        """Create abstract memory from group"""
        if not memories:
            return None
        common_context = {}
        all_keys: Set[str] = set()
        for memory in memories:
            all_keys.update(memory.context.keys())
        for key in all_keys:
            values = [m.context.get(key) for m in memories if key in m.context]
            if len(values) == len(memories):
                if all(v == values[0] for v in values):
                    common_context[key] = values[0]
        if not common_context:
            return None
        abstract_memory = Memory(
            memory_id=f"abstract_{uuid.uuid4().hex[:8]}",
            memory_type=memories[0].memory_type,
            content={
                "pattern": common_context,
                "instances": len(memories),
                "source_memories": [m.memory_id for m in memories],
            },
            timestamp=datetime.now(),
            importance=max(m.importance for m in memories),
            associations=[m.memory_id for m in memories],
        )
        return abstract_memory


class LearningAlgorithm:
    """Base class for learning algorithms"""

    def learn(self, experience: Experience) -> None:
        """Learn from an experience"""
        raise NotImplementedError

    def predict(self, state: Dict[str, Any]) -> Any:
        """Make prediction based on learned knowledge"""
        raise NotImplementedError


class ReinforcementLearner(LearningAlgorithm):
    """Simple reinforcement learning implementation"""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def learn(self, experience: Experience) -> None:
        """Update Q-values based on experience"""
        state_key = self._state_to_key(experience.state)
        action_key = self._action_to_key(experience.action)
        next_state_key = self._state_to_key(experience.next_state)
        current_q = self.q_table[state_key][action_key]
        if next_state_key in self.q_table:
            max_next_q = max(self.q_table[next_state_key].values())
        else:
            max_next_q = 0.0
        new_q = current_q + self.learning_rate * (
            experience.reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action_key] = new_q

    def predict(self, state: Dict[str, Any]) -> Optional[str]:
        """Get best action for state"""
        state_key = self._state_to_key(state)
        if state_key not in self.q_table:
            return None
        best_action = max(self.q_table[state_key].items(), key=lambda x: x[1])
        return best_action[0]

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to hashable key"""
        key_parts = []
        for k in sorted(state.keys()):
            if isinstance(state[k], (int, float, str, bool)):
                key_parts.append(f"{k}:{state[k]}")
        return "|".join(key_parts)

    def _action_to_key(self, action: Action) -> str:
        """Convert action to hashable key"""
        return f"{action.action_type.value}:{action.target}"


class PatternRecognizer:
    """Recognizes patterns in experiences and memories"""

    def __init__(self) -> None:
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_index: Dict[str, List[str]] = defaultdict(list)

    def extract_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract patterns from experiences"""
        patterns = []
        action_outcomes = defaultdict(list)
        for exp in experiences:
            action_key = f"{exp.action.action_type.value}"
            outcome_key = self._summarize_outcome(exp.outcome)
            action_outcomes[action_key].append((outcome_key, exp.reward))
        for action_key, outcomes in action_outcomes.items():
            if len(outcomes) >= 3:
                outcome_counts: DefaultDict[str, int] = defaultdict(int)
                total_reward: DefaultDict[str, float] = defaultdict(float)
                for outcome, reward in outcomes:
                    outcome_counts[outcome] += 1
                    total_reward[outcome] += reward
                most_common = max(outcome_counts.items(), key=lambda x: x[1])
                outcome_key, count = most_common
                if count / len(outcomes) > 0.6:
                    pattern = Pattern(
                        pattern_id=f"pattern_{uuid.uuid4().hex[:8]}",
                        pattern_type="action_outcome",
                        conditions={"action": action_key},
                        prediction={
                            "outcome": outcome_key,
                            "expected_reward": total_reward[outcome_key] / count,
                        },
                        confidence=count / len(outcomes),
                        occurrences=len(outcomes),
                        successes=count,
                    )
                    patterns.append(pattern)
        return patterns

    def _summarize_outcome(self, outcome: Dict[str, Any]) -> str:
        """Create summary key for outcome"""
        key_parts = []
        for k, v in sorted(outcome.items()):
            if isinstance(v, bool):
                if v:
                    key_parts.append(k)
            elif isinstance(v, (int, float)):
                if v > 0:
                    key_parts.append(f"{k}_positive")
                elif v < 0:
                    key_parts.append(f"{k}_negative")
        return "|".join(key_parts)

    def match_pattern(self, conditions: Dict[str, Any]) -> List[Pattern]:
        """Find patterns matching conditions"""
        matches = []
        for pattern in self.patterns.values():
            if self._conditions_match(pattern.conditions, conditions):
                matches.append(pattern)
        matches.sort(key=lambda p: p.confidence, reverse=True)
        return matches

    def _conditions_match(
        self, pattern_conditions: Dict[str, Any], test_conditions: Dict[str, Any]
    ) -> bool:
        """Check if conditions match pattern"""
        for key, value in pattern_conditions.items():
            if key not in test_conditions:
                return False
            if test_conditions[key] != value:
                return False
        return True


class MemorySystem:
    """Main memory system integrating all components"""

    def __init__(self, agent_id: str, storage: Optional[MemoryStorage] = None) -> None:
        self.agent_id = agent_id
        self.storage = storage or InMemoryStorage()
        self.working_memory = WorkingMemory()
        self.consolidator = MemoryConsolidator()
        self.pattern_recognizer = PatternRecognizer()
        self.learners: Dict[str, LearningAlgorithm] = {"reinforcement": ReinforcementLearner()}
        self.total_memories = 0
        self.memory_types_count: DefaultDict[MemoryType, int] = defaultdict(int)
        self.experience_buffer: deque = deque(maxlen=1000)

    def store_memory(
        self,
        content: Any,
        memory_type: MemoryType,
        importance: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Store a new memory"""
        memory = Memory(
            memory_id=f"{self.agent_id}_{uuid.uuid4().hex[:8]}",
            memory_type=memory_type,
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            context=context or {},
        )
        self.storage.store(memory)
        self.total_memories += 1
        self.memory_types_count[memory_type] += 1
        if importance > 0.7:
            self.working_memory.add(memory, importance)
        return memory

    def store_experience(self, experience: Experience) -> None:
        """Store an experience for learning"""
        self.experience_buffer.append(experience)
        for learner in self.learners.values():
            learner.learn(experience)
        self.store_memory(
            content=experience,
            memory_type=MemoryType.EPISODIC,
            importance=abs(experience.reward),
            context={
                "action": experience.action.action_type.value,
                "reward": experience.reward,
            },
        )

    def retrieve_memories(self, criteria: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """Retrieve memories matching criteria"""
        memories = self.storage.search(criteria)
        for memory in memories:
            memory.access()
        memories.sort(key=lambda m: m.calculate_strength(), reverse=True)
        return memories[:limit]

    def get_relevant_memories(self, context: DecisionContext, limit: int = 5) -> List[Memory]:
        """Get memories relevant to current context"""
        criteria: Dict[str, Any] = {"min_importance": 0.3}
        # Don't filter by context in initial retrieval, let relevance calculation
        # handle it
        memories = self.retrieve_memories(criteria, limit * 2)
        relevant = []
        for memory in memories:
            relevance = self._calculate_relevance(memory, context)
            if relevance > 0.3:
                relevant.append((relevance, memory))
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in relevant[:limit]]

    def _calculate_relevance(self, memory: Memory, context: DecisionContext) -> float:
        """Calculate memory relevance to current context"""
        relevance = 0.0
        if memory.memory_type == MemoryType.PROCEDURAL:
            relevance += 0.3
        if memory.context:
            if "percept_types" in memory.context:
                current_types = {p.stimulus.stimulus_type.value for p in context.percepts}
                memory_types = set(memory.context["percept_types"])
                overlap = len(current_types & memory_types)
                if overlap > 0:
                    relevance += 0.2 * overlap
            if context.current_goal and "goal_type" in memory.context:
                if memory.context["goal_type"] == context.current_goal.description[:20]:
                    relevance += 0.3
        age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
        if age_hours < 1:
            relevance += 0.2
        elif age_hours < 24:
            relevance += 0.1
        return min(relevance, 1.0)

    def consolidate_memories(self) -> None:
        """Consolidate short-term memories to long-term"""
        recent_memories = self.retrieve_memories(
            {
                "memory_type": MemoryType.EPISODIC,
                "start_time": datetime.now() - timedelta(hours=1),
            },
            limit=50,
        )
        to_consolidate = [
            m for m in recent_memories if self.consolidator.evaluate_for_consolidation(m)
        ]
        if len(to_consolidate) >= 3:
            consolidated = self.consolidator.consolidate(to_consolidate)
            for memory in consolidated:
                memory.memory_type = MemoryType.SEMANTIC
                self.storage.store(memory)

    def extract_patterns(self) -> List[Pattern]:
        """Extract patterns from recent experiences"""
        if len(self.experience_buffer) < 10:
            return []
        patterns = self.pattern_recognizer.extract_patterns(list(self.experience_buffer))
        for pattern in patterns:
            self.pattern_recognizer.patterns[pattern.pattern_id] = pattern
            self.store_memory(
                content=pattern,
                memory_type=MemoryType.PROCEDURAL,
                importance=pattern.confidence,
                context={
                    "pattern_type": pattern.pattern_type,
                    "conditions": pattern.conditions,
                },
            )
        return patterns

    def predict_outcome(self, state: Dict[str, Any], action: Action) -> Optional[Dict[str, Any]]:
        """Predict outcome of action based on learned patterns"""
        self.learners["reinforcement"].predict(state)
        conditions = {"action": action.action_type.value, **state}
        matching_patterns = self.pattern_recognizer.match_pattern(conditions)
        if matching_patterns:
            best_pattern = matching_patterns[0]
            return best_pattern.prediction
        return None

    def forget_old_memories(self, max_age_days: int = 30) -> int:
        """Remove old, unimportant memories"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        old_memories = self.retrieve_memories({"end_time": cutoff_time}, limit=100)
        removed = 0
        for memory in old_memories:
            if memory.importance <= 0.3 and memory.calculate_strength() < 0.1:
                if self.storage.remove(memory.memory_id):
                    removed += 1
                    self.total_memories -= 1
                    self.memory_types_count[memory.memory_type] -= 1
        return removed

    def save_to_disk(self, filepath: Path) -> None:
        """Save memory system to disk"""
        data = {
            "agent_id": self.agent_id,
            "memories": (
                [m for m in self.storage.memories.values()]
                if hasattr(self.storage, "memories")
                else []
            ),
            "patterns": list(self.pattern_recognizer.patterns.values()),
            "statistics": {
                "total_memories": self.total_memories,
                "memory_types_count": dict(self.memory_types_count),
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_from_disk(self, filepath: Path) -> None:
        """Load memory system from disk"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        for memory in data["memories"]:
            self.storage.store(memory)
        for pattern in data["patterns"]:
            self.pattern_recognizer.patterns[pattern.pattern_id] = pattern
        self.total_memories = data["statistics"]["total_memories"]
        self.memory_types_count = defaultdict(int, data["statistics"]["memory_types_count"])

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory system state"""
        return {
            "total_memories": self.total_memories,
            "memory_types": dict(self.memory_types_count),
            "working_memory_items": len(self.working_memory.items),
            "experience_buffer_size": len(self.experience_buffer),
            "learned_patterns": len(self.pattern_recognizer.patterns),
            "oldest_memory": (
                min([m.timestamp for m in self.storage.memories.values()])
                if hasattr(self.storage, "memories") and self.storage.memories
                else None
            ),
            "average_importance": (
                np.mean([m.importance for m in self.storage.memories.values()])
                if hasattr(self.storage, "memories") and self.storage.memories
                else 0.0
            ),
        }


class MessageSystem:
    """Simple message system for agent communication"""

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.message_queue: deque = deque()

    def send_message(self, sender_id: str, recipient_id: str, content: Any) -> None:
        """Send a message between agents"""
        message = {
            "id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "content": content,
            "timestamp": datetime.now(),
        }
        self.messages.append(message)
        self.message_queue.append(message)

    def get_messages(self, recipient_id: str) -> List[Dict[str, Any]]:
        """Get messages for a specific recipient"""
        return [msg for msg in self.messages if msg["recipient_id"] == recipient_id]

    def clear_messages(self) -> None:
        """Clear all messages"""
        self.messages.clear()
        self.message_queue.clear()
