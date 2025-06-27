"""
Module for FreeAgentics Active Inference implementation.
"""

import hashlib
import json
import logging
import pickle
import random
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

"""
Fallback Mechanisms for Limited Resources
Provides graceful degradation, caching, and precomputed responses for
edge deployment scenarios with limited computational resources.
"""
logger = logging.getLogger(__name__)


class FallbackLevel(Enum):
    """Levels of fallback mechanisms"""

    FULL_MODEL = 0  # Full model inference
    CACHED = 1  # Cached responses
    SIMPLIFIED = 2  # Simplified model
    TEMPLATE = 3  # Template-based
    PRECOMPUTED = 4  # Precomputed responses
    RULE_BASED = 5  # Rule-based logic
    RANDOM = 6  # Random valid response


@dataclass
class ResourceConstraints:
    """Current resource constraints"""

    available_memory_mb: float
    cpu_usage_percent: float
    battery_level: Optional[float] = None
    network_available: bool = True
    inference_timeout_ms: float = 5000
    max_tokens: int = 100


@dataclass
class FallbackResponse:
    """Response from fallback mechanism"""

    text: str
    fallback_level: FallbackLevel
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    computation_time_ms: float = 0


class ResponseCache:
    """
    Advanced caching system for LLM responses.
    Features:
    - Semantic similarity matching
    - Compression
    - Expiration
    - Priority-based eviction
    """

    def __init__(self, cache_dir: Path, max_size_mb: int = 100) -> None:
        """Initialize response cache"""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        # Initialize database
        self.db_path = self.cache_dir / "response_cache.db"
        self._init_database()
        # In-memory cache for fast access
        self.memory_cache: OrderedDict[str, FallbackResponse] = OrderedDict()
        self.max_memory_entries = 100
        # Cache statistics
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.lock = threading.Lock()

    def _init_database(self):
        """Initialize SQLite database for persistent cache"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                prompt TEXT,
                response TEXT,
                embedding BLOB,
                fallback_level INTEGER,
                confidence REAL,
                metadata TEXT,
                created_at TIMESTAMP,
                accessed_at TIMESTAMP,
                access_count INTEGER,
                size_bytes INTEGER
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_accessed_at
            ON cache_entries(accessed_at)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_size
            ON cache_entries(size_bytes)
        """
        )
        conn.commit()
        conn.close()

    def _generate_key(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from prompt and context"""
        key_data = {"prompt": prompt, "context": context or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _compress_response(self, response: FallbackResponse) -> bytes:
        """Compress response for storage"""
        data = pickle.dumps(response)
        return zlib.compress(data, level=6)

    def _decompress_response(self, data: bytes) -> FallbackResponse:
        """Decompress stored response"""
        decompressed = zlib.decompress(data)
        return pickle.loads(decompressed)

    def get(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.9,
    ) -> Optional[FallbackResponse]:
        """
        Get cached response with semantic similarity matching.
        Args:
            prompt: Input prompt
            context: Optional context
            similarity_threshold: Minimum similarity for cache hit
        Returns:
            Cached response if found
        """
        with self.lock:
            # Check memory cache first
            key = self._generate_key(prompt, context)
            if key in self.memory_cache:
                self.stats["hits"] += 1
                response = self.memory_cache[key]
                # Move to end (LRU)
                self.memory_cache.move_to_end(key)
                return response
            # Check persistent cache
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            # Exact match
            cursor.execute(
                """
                SELECT response, fallback_level, confidence, metadata
                FROM cache_entries
                WHERE key = ?
            """,
                (key,),
            )
            result = cursor.fetchone()
            if result:
                # Update access statistics
                cursor.execute(
                    """
                    UPDATE cache_entries
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE key = ?
                """,
                    (datetime.now(), key),
                )
                conn.commit()
                # Reconstruct response
                response = FallbackResponse(
                    text=result[0],
                    fallback_level=FallbackLevel(result[1]),
                    confidence=result[2],
                    metadata=json.loads(result[3]),
                    cached=True,
                )
                # Add to memory cache
                self._add_to_memory_cache(key, response)
                self.stats["hits"] += 1
                conn.close()
                return response
            # TODO: Implement semantic similarity search
            # This would require embedding generation and vector similarity
            self.stats["misses"] += 1
            conn.close()
            return None

    def put(
        self,
        prompt: str,
        response: FallbackResponse,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Store response in cache"""
        with self.lock:
            key = self._generate_key(prompt, context)
            # Add to memory cache
            self._add_to_memory_cache(key, response)
            # Add to persistent cache
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            # Check cache size and evict if necessary
            cursor.execute("SELECT SUM(size_bytes) FROM cache_entries")
            total_size = cursor.fetchone()[0] or 0
            if total_size > self.max_size_bytes:
                self._evict_entries(cursor, total_size - self.max_size_bytes * 0.8)
            # Compress and store
            compressed = self._compress_response(response)
            cursor.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key, prompt, response, fallback_level, confidence,
                 metadata, created_at, accessed_at, access_count, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    key,
                    prompt,
                    response.text,
                    response.fallback_level.value,
                    response.confidence,
                    json.dumps(response.metadata),
                    datetime.now(),
                    datetime.now(),
                    1,
                    len(compressed),
                ),
            )
            conn.commit()
            conn.close()

    def _add_to_memory_cache(self, key: str, response: FallbackResponse):
        """Add response to memory cache with LRU eviction"""
        if len(self.memory_cache) >= self.max_memory_entries:
            # Remove least recently used
            self.memory_cache.popitem(last=False)
        self.memory_cache[key] = response

    def _evict_entries(self, cursor: sqlite3.Cursor, bytes_to_free: int):
        """Evict cache entries to free space"""
        # Evict least recently used entries
        cursor.execute(
            """
            SELECT key, size_bytes
            FROM cache_entries
            ORDER BY accessed_at ASC, access_count ASC
            LIMIT 100
        """
        )
        freed_bytes = 0
        keys_to_delete = []
        for key, size in cursor.fetchall():
            keys_to_delete.append(key)
            freed_bytes += size
            if freed_bytes >= bytes_to_free:
                break
        if keys_to_delete:
            placeholders = ", ".join("?" * len(keys_to_delete))
            cursor.execute(
                f"DELETE FROM cache_entries WHERE key IN ({placeholders})",
                keys_to_delete,
            )
            self.stats["evictions"] += len(keys_to_delete)
            # Remove from memory cache
            for key in keys_to_delete:
                self.memory_cache.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*), SUM(size_bytes), AVG(confidence)
                FROM cache_entries
            """
            )
            count, total_size, avg_confidence = cursor.fetchone()
            conn.close()
            hit_rate = (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0
                else 0
            )
            return {
                "total_entries": count or 0,
                "total_size_mb": (total_size or 0) / (1024 * 1024),
                "memory_entries": len(self.memory_cache),
                "hit_rate": hit_rate,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "avg_confidence": avg_confidence or 0,
            }


class TemplateEngine:
    """
    Template-based response generation for common patterns.
    """

    def __init__(self, template_dir: Optional[Path] = None) -> None:
        """Initialize template engine"""
        self.template_dir = template_dir or Path("templates")
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[str]]:
        """Load response templates"""
        templates = {
            "greeting": [
                "Hello! How can I assist you today?",
                "Greetings! I'm here to help.",
                "Hi there! What can I do for you?",
            ],
            "exploration": [
                "I'll explore the {direction} area for {resource}.",
                "Searching for {resource} in the {direction} region.",
                "Beginning exploration of {location} for valuable resources.",
            ],
            "trading": [
                "I'd like to trade {offered_amount} {offered_resource} for {requested_amount} {requested_resource}.",
                "Proposing trade: {offered_resource} for {requested_resource}.",
                "Trade offer: Exchange {offered_amount} units of {offered_resource}.",
            ],
            "movement": [
                "Moving {direction} to {destination}.",
                "Heading towards {destination}.",
                "Navigating to {destination} via {path}.",
            ],
            "observation": [
                "I observe {object} at {location}.",
                "Detected {object} in the {direction} direction.",
                "Found {amount} units of {resource} nearby.",
            ],
            "planning": [
                "My plan is to {action} in order to {goal}.",
                "Strategy: {step1}, then {step2}, finally {step3}.",
                "Objective: {goal}. Method: {approach}.",
            ],
            "error": [
                "I'm unable to {action} due to {reason}.",
                "Cannot complete request: {error}.",
                "Operation failed: {error_message}.",
            ],
            "confirmation": [
                "Understood. I'll {action}.",
                "Confirmed. Proceeding with {task}.",
                "Acknowledged. {action} in progress.",
            ],
        }
        # Load custom templates if available
        if self.template_dir.exists():
            for template_file in self.template_dir.glob("*.json"):
                try:
                    with open(template_file) as f:
                        custom_templates = json.load(f)
                        templates.update(custom_templates)
                except Exception as e:
                    logger.error(f"Failed to load template {template_file}: {e}")
        return templates

    def generate(self, intent: str, variables: Dict[str, Any]) -> Optional[str]:
        """
        Generate response from template.
        Args:
            intent: Detected intent
            variables: Variables to fill in template
        Returns:
            Generated response or None
        """
        if intent not in self.templates:
            return None
        template_list = self.templates[intent]
        if not template_list:
            return None
        # Select template (could be random or based on context)
        template = random.choice(template_list)
        try:
            # Fill in variables
            response = template.format(**variables)
            return response
        except KeyError as e:
            logger.warning(f"Missing variable for template: {e}")
            return None


class RuleBasedResponder:
    """
    Rule-based response generation for predictable scenarios.
    """

    def __init__(self) -> None:
        """Initialize rule-based responder"""
        self.rules = self._create_rules()

    def _create_rules(self) -> List[tuple[callable, callable]]:
        """Create rule set"""
        rules = []

        # Greeting rule
        def greeting_condition(prompt: str, context: dict) -> bool:
            greetings = ["hello", "hi", "hey", "greetings"]
            return any(g in prompt.lower() for g in greetings)

        def greeting_response(prompt: str, context: dict) -> str:
            agent_name = context.get("agent_name", "Agent")
            return f"Hello! I'm {agent_name}, ready to assist you."

        rules.append((greeting_condition, greeting_response))

        # Resource query rule
        def resource_condition(prompt: str, context: dict) -> bool:
            resources = ["food", "water", "energy", "materials"]
            queries = ["how much", "how many", "what is", "check"]
            return any(r in prompt.lower() for r in resources) and any(
                q in prompt.lower() for q in queries
            )

        def resource_response(prompt: str, context: dict) -> str:
            resources = context.get("resources", {})
            if resources:
                resource_list = ", ".join(f"{k}: {v}" for k, v in resources.items())
                return f"Current resources: {resource_list}"
            return "No resources currently available."

        rules.append((resource_condition, resource_response))

        # Movement rule
        def movement_condition(prompt: str, context: dict) -> bool:
            movements = ["move", "go", "travel", "walk", "navigate"]
            return any(m in prompt.lower() for m in movements)

        def movement_response(prompt: str, context: dict) -> str:
            location = context.get("current_location", "unknown")
            return f"Moving from {location}. Calculating optimal path..."

        rules.append((movement_condition, movement_response))
        # Add more rules as needed
        return rules

    def generate(self, prompt: str, context: Dict[str, Any]) -> Optional[str]:
        """Generate response based on rules"""
        for condition, response_func in self.rules:
            if condition(prompt, context):
                return response_func(prompt, context)
        return None


class PrecomputedResponses:
    """
    Manages precomputed responses for common scenarios.
    """

    def __init__(self, data_file: Optional[Path] = None) -> None:
        """Initialize precomputed responses"""
        self.data_file = data_file or Path("precomputed_responses.json")
        self.responses = self._load_responses()
        self.embeddings = {}  # Would store embeddings for similarity matching

    def _load_responses(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load precomputed responses"""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load precomputed responses: {e}")
        # Default responses
        return {
            "exploration": [
                {
                    "context": "new_area",
                    "response": "Initiating systematic exploration of uncharted territory.",
                    "confidence": 0.8,
                },
                {
                    "context": "resource_search",
                    "response": "Scanning area for valuable resources and materials.",
                    "confidence": 0.9,
                },
            ],
            "combat": [
                {
                    "context": "threat_detected",
                    "response": "Threat detected. Initiating defensive protocols.",
                    "confidence": 0.95,
                },
                {
                    "context": "retreat",
                    "response": "Strategic withdrawal to safer position.",
                    "confidence": 0.85,
                },
            ],
            "trading": [
                {
                    "context": "offer_received",
                    "response": "Evaluating trade offer for mutual benefit.",
                    "confidence": 0.7,
                },
                {
                    "context": "negotiation",
                    "response": "Proposing alternative terms for fair exchange.",
                    "confidence": 0.75,
                },
            ],
        }

    def find_best_match(self, category: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find best matching precomputed response"""
        if category not in self.responses:
            return None
        candidates = self.responses[category]
        # Simple context matching (could use embeddings for better matching)
        best_match = None
        best_score = 0
        for candidate in candidates:
            score = self._calculate_context_similarity(candidate.get("context", ""), context)
            if score > best_score:
                best_score = score
                best_match = candidate
        return best_match if best_score > 0.5 else None

    def _calculate_context_similarity(
        self, candidate_context: str, actual_context: Dict[str, Any]
    ) -> float:
        """Calculate similarity between contexts"""
        # Simplified similarity calculation
        context_str = json.dumps(actual_context).lower()
        if candidate_context in context_str:
            return 0.9
        # Check for partial matches
        words = candidate_context.split("_")
        matches = sum(1 for word in words if word in context_str)
        return matches / len(words) if words else 0


class FallbackManager:
    """
    Manages all fallback mechanisms with graceful degradation.
    """

    def __init__(self, cache_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize fallback manager"""
        self.cache_dir = cache_dir
        self.config = config or {}
        # Initialize components
        self.cache = ResponseCache(cache_dir / "cache")
        self.template_engine = TemplateEngine()
        self.rule_based = RuleBasedResponder()
        self.precomputed = PrecomputedResponses()
        # Performance monitoring
        self.performance_history = []
        self.max_history = 100

    def generate_response(
        self, prompt: str, context: Dict[str, Any], constraints: ResourceConstraints
    ) -> FallbackResponse:
        """
        Generate response with appropriate fallback level.
        Args:
            prompt: Input prompt
            context: Current context
            constraints: Resource constraints
        Returns:
            Best available response
        """
        start_time = time.time()
        # Try cache first
        cached = self.cache.get(prompt, context)
        if cached:
            cached.computation_time_ms = (time.time() - start_time) * 1000
            return cached
        # Determine appropriate fallback level based on constraints
        fallback_level = self._determine_fallback_level(constraints)
        # Generate response based on fallback level
        response = None
        if fallback_level == FallbackLevel.TEMPLATE:
            response = self._generate_template_response(prompt, context)
        elif fallback_level == FallbackLevel.PRECOMPUTED:
            response = self._generate_precomputed_response(prompt, context)
        elif fallback_level == FallbackLevel.RULE_BASED:
            response = self._generate_rule_based_response(prompt, context)
        elif fallback_level == FallbackLevel.RANDOM:
            response = self._generate_random_response(context)
        if response:
            response.computation_time_ms = (time.time() - start_time) * 1000
            # Cache successful response
            if response.confidence > 0.5:
                self.cache.put(prompt, response, context)
            # Record performance
            self._record_performance(response, constraints)
            return response
        # Ultimate fallback
        return FallbackResponse(
            text="I need a moment to process that request.",
            fallback_level=FallbackLevel.RANDOM,
            confidence=0.1,
            computation_time_ms=(time.time() - start_time) * 1000,
        )

    def _determine_fallback_level(self, constraints: ResourceConstraints) -> FallbackLevel:
        """Determine appropriate fallback level based on constraints"""
        # Memory-based decision
        if constraints.available_memory_mb < 50:
            return FallbackLevel.RANDOM
        elif constraints.available_memory_mb < 100:
            return FallbackLevel.RULE_BASED
        elif constraints.available_memory_mb < 200:
            return FallbackLevel.PRECOMPUTED
        elif constraints.available_memory_mb < 500:
            return FallbackLevel.TEMPLATE
        # CPU-based decision
        if constraints.cpu_usage_percent > 90:
            return FallbackLevel.RULE_BASED
        elif constraints.cpu_usage_percent > 70:
            return FallbackLevel.TEMPLATE
        # Battery-based decision (for mobile)
        if constraints.battery_level and constraints.battery_level < 20:
            return FallbackLevel.PRECOMPUTED
        # Network-based decision
        if not constraints.network_available:
            return FallbackLevel.TEMPLATE
        return FallbackLevel.CACHED

    def _generate_template_response(
        self, prompt: str, context: Dict[str, Any]
    ) -> Optional[FallbackResponse]:
        """Generate template-based response"""
        # Detect intent (simplified)
        intent = self._detect_intent(prompt)
        if intent:
            text = self.template_engine.generate(intent, context)
            if text:
                return FallbackResponse(
                    text=text,
                    fallback_level=FallbackLevel.TEMPLATE,
                    confidence=0.7,
                    metadata={"intent": intent},
                )
        return None

    def _generate_precomputed_response(
        self, prompt: str, context: Dict[str, Any]
    ) -> Optional[FallbackResponse]:
        """Generate precomputed response"""
        category = self._detect_category(prompt)
        if category:
            match = self.precomputed.find_best_match(category, context)
            if match:
                return FallbackResponse(
                    text=match["response"],
                    fallback_level=FallbackLevel.PRECOMPUTED,
                    confidence=match.get("confidence", 0.6),
                    metadata={"category": category},
                )
        return None

    def _generate_rule_based_response(
        self, prompt: str, context: Dict[str, Any]
    ) -> Optional[FallbackResponse]:
        """Generate rule-based response"""
        text = self.rule_based.generate(prompt, context)
        if text:
            return FallbackResponse(
                text=text,
                fallback_level=FallbackLevel.RULE_BASED,
                confidence=0.8,
                metadata={"rule_matched": True},
            )
        return None

    def _generate_random_response(self, context: Dict[str, Any]) -> FallbackResponse:
        """Generate random valid response"""
        responses = [
            "Processing your request...",
            "Analyzing the situation...",
            "Considering available options...",
            "Evaluating possible actions...",
            "Gathering information...",
        ]
        return FallbackResponse(
            text=random.choice(responses),
            fallback_level=FallbackLevel.RANDOM,
            confidence=0.2,
        )

    def _detect_intent(self, prompt: str) -> Optional[str]:
        """Detect intent from prompt"""
        prompt_lower = prompt.lower()
        intents = {
            "greeting": ["hello", "hi", "hey"],
            "exploration": ["explore", "search", "find"],
            "trading": ["trade", "exchange", "buy", "sell"],
            "movement": ["move", "go", "travel"],
            "observation": ["see", "observe", "notice"],
            "planning": ["plan", "strategy", "decide"],
            "confirmation": ["yes", "ok", "confirm", "agree"],
        }
        for intent, keywords in intents.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return intent
        return None

    def _detect_category(self, prompt: str) -> Optional[str]:
        """Detect category from prompt"""
        prompt_lower = prompt.lower()
        categories = {
            "exploration": ["explore", "discover", "search"],
            "combat": ["fight", "attack", "defend", "threat"],
            "trading": ["trade", "buy", "sell", "exchange"],
        }
        for category, keywords in categories.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return category
        return None

    def _record_performance(self, response: FallbackResponse, constraints: ResourceConstraints):
        """Record performance metrics"""
        metric = {
            "timestamp": time.time(),
            "fallback_level": response.fallback_level.value,
            "confidence": response.confidence,
            "computation_time_ms": response.computation_time_ms,
            "memory_mb": constraints.available_memory_mb,
            "cpu_percent": constraints.cpu_usage_percent,
        }
        self.performance_history.append(metric)
        # Keep history bounded
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_history:
            return {}
        # Calculate statistics
        computation_times = [m["computation_time_ms"] for m in self.performance_history]
        confidence_scores = [m["confidence"] for m in self.performance_history]
        fallback_counts = defaultdict(int)
        for metric in self.performance_history:
            fallback_counts[metric["fallback_level"]] += 1
        return {
            "avg_computation_time_ms": np.mean(computation_times),
            "max_computation_time_ms": np.max(computation_times),
            "min_computation_time_ms": np.min(computation_times),
            "avg_confidence": np.mean(confidence_scores),
            "fallback_distribution": dict(fallback_counts),
            "cache_stats": self.cache.get_stats(),
            "total_responses": len(self.performance_history),
        }
