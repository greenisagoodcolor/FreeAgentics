"""Iterative Loop Controller for managing multi-iteration prompt processing.

This service handles the iterative aspect of the prompt → agent → KG pipeline,
managing conversation state, context accumulation, and intelligent suggestion
generation based on the evolving knowledge graph.
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.pymdp_adapter import PyMDPCompatibilityAdapter
from database.models import Agent
from database.prompt_models import (
    Conversation,
    ConversationStatus,
    KnowledgeGraphUpdate,
    Prompt,
    PromptStatus,
)
from knowledge_graph.graph_engine import KnowledgeGraph
from services.belief_kg_bridge import BeliefKGBridge

logger = logging.getLogger(__name__)


class ConversationContext:
    """Maintains conversation state across iterations."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.iteration_count = 0
        self.agent_ids: List[str] = []
        self.kg_node_ids: Set[str] = set()
        self.belief_history: List[Dict[str, Any]] = []
        self.suggestion_history: List[List[str]] = []
        self.prompt_history: List[str] = []
        self.gmn_evolution: List[str] = []
        self.context_metadata: Dict[str, Any] = {}

    def add_iteration(
        self,
        prompt: str,
        agent_id: str,
        gmn_spec: str,
        beliefs: Dict[str, Any],
        kg_nodes: List[str],
        suggestions: List[str],
    ):
        """Record a new iteration in the conversation."""
        self.iteration_count += 1
        self.prompt_history.append(prompt)
        self.agent_ids.append(agent_id)
        self.gmn_evolution.append(gmn_spec)
        self.belief_history.append(beliefs)
        self.kg_node_ids.update(kg_nodes)
        self.suggestion_history.append(suggestions)

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation context."""
        return {
            "iteration_count": self.iteration_count,
            "total_agents": len(set(self.agent_ids)),
            "kg_nodes": len(self.kg_node_ids),
            "belief_evolution": self._analyze_belief_evolution(),
            "prompt_themes": self._extract_themes(),
            "suggestion_patterns": self._analyze_suggestion_patterns(),
        }

    def _analyze_belief_evolution(self) -> Dict[str, Any]:
        """Analyze how beliefs have evolved over iterations."""
        if not self.belief_history:
            return {"trend": "none", "stability": 0.0}

        # Simple analysis of belief stability
        stability_scores = []
        for i in range(1, len(self.belief_history)):
            prev_beliefs = self.belief_history[i - 1]
            curr_beliefs = self.belief_history[i]

            # Compare belief structures
            common_keys = set(prev_beliefs.keys()) & set(curr_beliefs.keys())
            if common_keys:
                stability = len(common_keys) / max(
                    len(prev_beliefs), len(curr_beliefs)
                )
                stability_scores.append(stability)

        avg_stability = (
            sum(stability_scores) / len(stability_scores)
            if stability_scores
            else 0.0
        )

        return {
            "trend": "converging" if avg_stability > 0.7 else "exploring",
            "stability": avg_stability,
            "total_iterations": len(self.belief_history),
        }

    def _extract_themes(self) -> List[str]:
        """Extract common themes from prompts."""
        if not self.prompt_history:
            return []

        # Simple keyword extraction
        all_words = []
        for prompt in self.prompt_history:
            words = prompt.lower().split()
            all_words.extend([w for w in words if len(w) > 4])

        # Count frequencies
        word_counts = defaultdict(int)
        for word in all_words:
            word_counts[word] += 1

        # Return top themes
        sorted_words = sorted(
            word_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [word for word, count in sorted_words[:5] if count > 1]

    def _analyze_suggestion_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in generated suggestions."""
        if not self.suggestion_history:
            return {"diversity": 0.0, "common_suggestions": []}

        # Flatten all suggestions
        all_suggestions = []
        for suggestions in self.suggestion_history:
            all_suggestions.extend(suggestions)

        # Calculate diversity
        unique_suggestions = set(all_suggestions)
        diversity = (
            len(unique_suggestions) / len(all_suggestions)
            if all_suggestions
            else 0.0
        )

        # Find common suggestion themes
        suggestion_counts = defaultdict(int)
        for suggestion in all_suggestions:
            # Extract key phrases
            key_phrases = [
                "explore",
                "goal",
                "preference",
                "observation",
                "coalition",
                "uncertainty",
            ]
            for phrase in key_phrases:
                if phrase in suggestion.lower():
                    suggestion_counts[phrase] += 1

        common_themes = sorted(
            suggestion_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        return {
            "diversity": diversity,
            "common_themes": [theme for theme, _ in common_themes],
            "total_suggestions": len(all_suggestions),
        }


class IterativeController:
    """Controls the iterative loop of prompt processing."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        belief_kg_bridge: BeliefKGBridge,
        pymdp_adapter: PyMDPCompatibilityAdapter,
    ):
        self.knowledge_graph = knowledge_graph
        self.belief_kg_bridge = belief_kg_bridge
        self.pymdp_adapter = pymdp_adapter
        self._conversation_contexts: Dict[str, ConversationContext] = {}

    async def get_or_create_context(
        self, conversation_id: str, db: AsyncSession
    ) -> ConversationContext:
        """Get existing context or create new one from database."""
        # Check in-memory cache first
        if conversation_id in self._conversation_contexts:
            return self._conversation_contexts[conversation_id]

        # Load from database
        context = ConversationContext(conversation_id)

        # Load prompt history
        result = await db.execute(
            select(Prompt)
            .where(Prompt.conversation_id == conversation_id)
            .order_by(Prompt.created_at)
        )
        prompts = result.scalars().all()

        for prompt in prompts:
            if prompt.agent_id and prompt.gmn_specification:
                # Extract KG nodes
                kg_result = await db.execute(
                    select(KnowledgeGraphUpdate).where(
                        KnowledgeGraphUpdate.prompt_id == prompt.id
                    )
                )
                kg_updates = kg_result.scalars().all()
                kg_nodes = [
                    update.node_id for update in kg_updates if update.applied
                ]

                # Reconstruct context
                context.add_iteration(
                    prompt=prompt.prompt_text,
                    agent_id=str(prompt.agent_id),
                    gmn_spec=prompt.gmn_specification,
                    beliefs=prompt.response_data.get("beliefs", {}),
                    kg_nodes=kg_nodes,
                    suggestions=prompt.next_suggestions,
                )

        # Cache context
        self._conversation_contexts[conversation_id] = context
        return context

    async def prepare_iteration_context(
        self, conversation_context: ConversationContext, current_prompt: str
    ) -> Dict[str, Any]:
        """Prepare context for the next iteration based on conversation history."""
        context_summary = conversation_context.get_context_summary()

        # Get current KG state relevant to this conversation
        kg_context = await self._get_kg_context(
            conversation_context.kg_node_ids
        )

        # Analyze prompt evolution
        prompt_analysis = self._analyze_prompt_evolution(
            conversation_context.prompt_history, current_prompt
        )

        # Generate iteration-specific constraints
        constraints = self._generate_iteration_constraints(
            context_summary, kg_context, prompt_analysis
        )

        return {
            "conversation_summary": context_summary,
            "kg_state": kg_context,
            "prompt_analysis": prompt_analysis,
            "constraints": constraints,
            "iteration_number": conversation_context.iteration_count + 1,
            "previous_suggestions": conversation_context.suggestion_history[-1]
            if conversation_context.suggestion_history
            else [],
        }

    async def generate_intelligent_suggestions(
        self,
        agent_id: str,
        pymdp_agent: Any,
        conversation_context: ConversationContext,
        current_beliefs: Dict[str, Any],
        db: AsyncSession,
    ) -> List[str]:
        """Generate context-aware suggestions based on KG and conversation history."""
        suggestions = []

        # 1. Analyze belief convergence
        belief_analysis = conversation_context.get_context_summary()[
            "belief_evolution"
        ]

        if belief_analysis["stability"] < 0.5:
            suggestions.append(
                "Explore uncertainty - Add sensory modalities to reduce belief ambiguity"
            )
        elif belief_analysis["stability"] > 0.9:
            suggestions.append(
                "Introduce variation - Add new action possibilities or environmental factors"
            )

        # 2. Check KG connectivity
        kg_analysis = await self._analyze_kg_connectivity(
            conversation_context.kg_node_ids, agent_id
        )

        if kg_analysis["isolated_nodes"] > 0:
            suggestions.append(
                f"Connect {kg_analysis['isolated_nodes']} isolated knowledge nodes through agent interactions"
            )

        if kg_analysis["cluster_count"] > 1:
            suggestions.append(
                "Bridge knowledge clusters - Create agents that span multiple domains"
            )

        # 3. Analyze prompt themes evolution
        themes = conversation_context.get_context_summary()["prompt_themes"]
        theme_suggestions = self._generate_theme_based_suggestions(
            themes, conversation_context
        )
        suggestions.extend(theme_suggestions)

        # 4. Check for missing agent capabilities
        capability_gaps = await self._identify_capability_gaps(
            conversation_context.agent_ids, db
        )

        for gap in capability_gaps:
            suggestions.append(
                f"Add {gap} capability to enhance agent interactions"
            )

        # 5. Suggest based on suggestion history patterns
        suggestion_patterns = conversation_context.get_context_summary()[
            "suggestion_patterns"
        ]

        if suggestion_patterns["diversity"] < 0.3:
            suggestions.append(
                "Try a different approach - Current iterations are too similar"
            )

        # 6. Context-specific suggestions based on iteration count
        if conversation_context.iteration_count == 0:
            suggestions.append(
                "Start with basic exploration to establish environmental understanding"
            )
        elif conversation_context.iteration_count < 3:
            suggestions.append(
                "Add goal-directed behavior to guide agent actions"
            )
        elif conversation_context.iteration_count < 5:
            suggestions.append(
                "Introduce multi-agent coordination for complex tasks"
            )
        else:
            suggestions.append(
                "Consider meta-learning - Let agents adapt their own models"
            )

        # 7. Knowledge graph growth suggestions
        kg_growth_rate = len(conversation_context.kg_node_ids) / max(
            conversation_context.iteration_count, 1
        )

        if kg_growth_rate < 5:
            suggestions.append(
                "Increase observation diversity to enrich knowledge representation"
            )
        elif kg_growth_rate > 20:
            suggestions.append(
                "Focus on knowledge consolidation rather than expansion"
            )

        # Prioritize and deduplicate suggestions
        unique_suggestions = []
        seen = set()

        for suggestion in suggestions:
            key = suggestion.lower()[:30]  # Simple deduplication
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)

        # Return top 5 most relevant suggestions
        return unique_suggestions[:5]

    async def update_conversation_context(
        self,
        conversation_context: ConversationContext,
        prompt: str,
        agent_id: str,
        gmn_spec: str,
        beliefs: Dict[str, Any],
        kg_updates: List[KnowledgeGraphUpdate],
        suggestions: List[str],
    ):
        """Update conversation context after successful iteration."""
        kg_nodes = [update.node_id for update in kg_updates if update.applied]

        conversation_context.add_iteration(
            prompt=prompt,
            agent_id=agent_id,
            gmn_spec=gmn_spec,
            beliefs=beliefs,
            kg_nodes=kg_nodes,
            suggestions=suggestions,
        )

        # Update cache
        self._conversation_contexts[
            conversation_context.conversation_id
        ] = conversation_context

    async def _get_kg_context(self, node_ids: Set[str]) -> Dict[str, Any]:
        """Get relevant knowledge graph context for the conversation."""
        if not node_ids:
            return {"nodes": 0, "clusters": 0, "density": 0.0}

        # Get subgraph containing these nodes
        try:
            subgraph = await self.knowledge_graph.get_subgraph(list(node_ids))

            # Analyze subgraph structure
            node_count = len(subgraph.get("nodes", []))
            edge_count = len(subgraph.get("edges", []))

            # Calculate basic metrics
            max_edges = node_count * (node_count - 1) / 2
            density = edge_count / max_edges if max_edges > 0 else 0.0

            # Identify clusters (simplified)
            clusters = self._identify_clusters(subgraph)

            return {
                "nodes": node_count,
                "edges": edge_count,
                "density": density,
                "clusters": len(clusters),
                "node_types": self._count_node_types(subgraph),
            }
        except Exception as e:
            logger.warning(f"Failed to get KG context: {str(e)}")
            return {"nodes": len(node_ids), "clusters": 0, "density": 0.0}

    def _analyze_prompt_evolution(
        self, prompt_history: List[str], current_prompt: str
    ) -> Dict[str, Any]:
        """Analyze how prompts have evolved in the conversation."""
        if not prompt_history:
            return {
                "evolution": "initial",
                "similarity": 0.0,
                "direction": "exploratory",
            }

        # Calculate similarity with previous prompts
        similarities = []
        for prev_prompt in prompt_history[-3:]:  # Last 3 prompts
            similarity = self._calculate_prompt_similarity(
                prev_prompt, current_prompt
            )
            similarities.append(similarity)

        avg_similarity = (
            sum(similarities) / len(similarities) if similarities else 0.0
        )

        # Determine evolution pattern
        if avg_similarity > 0.8:
            evolution = "refining"  # Similar prompts, refining same concept
        elif avg_similarity > 0.5:
            evolution = "extending"  # Building on previous concepts
        else:
            evolution = "pivoting"  # New direction

        # Analyze thematic direction
        themes = self._extract_prompt_themes(prompt_history + [current_prompt])
        direction = self._determine_thematic_direction(themes)

        return {
            "evolution": evolution,
            "similarity": avg_similarity,
            "direction": direction,
            "theme_consistency": len(set(themes)) / len(themes)
            if themes
            else 0.0,
        }

    def _generate_iteration_constraints(
        self,
        context_summary: Dict[str, Any],
        kg_context: Dict[str, Any],
        prompt_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate constraints for the next iteration based on context."""
        constraints = {"maintain_consistency": True, "iteration_specific": {}}

        # Based on belief evolution
        if context_summary["belief_evolution"]["stability"] > 0.8:
            constraints["iteration_specific"]["increase_complexity"] = True
            constraints["iteration_specific"]["min_state_dimensions"] = 3

        # Based on KG density
        if kg_context["density"] < 0.1:
            constraints["iteration_specific"]["increase_connectivity"] = True
            constraints["iteration_specific"]["min_observations"] = 5

        # Based on prompt evolution
        if prompt_analysis["evolution"] == "refining":
            constraints["iteration_specific"]["preserve_core_structure"] = True
            constraints["iteration_specific"]["allow_minor_variations"] = True
        elif prompt_analysis["evolution"] == "pivoting":
            constraints["iteration_specific"]["allow_major_changes"] = True
            constraints["iteration_specific"][
                "suggest_novel_approaches"
            ] = True

        # Iteration count specific
        iteration = context_summary["iteration_count"]
        if iteration < 2:
            constraints["iteration_specific"]["focus"] = "exploration"
        elif iteration < 5:
            constraints["iteration_specific"]["focus"] = "optimization"
        else:
            constraints["iteration_specific"]["focus"] = "innovation"

        return constraints

    async def _analyze_kg_connectivity(
        self, node_ids: Set[str], agent_id: str
    ) -> Dict[str, Any]:
        """Analyze connectivity patterns in the knowledge graph."""
        try:
            # Get subgraph
            subgraph = await self.knowledge_graph.get_subgraph(list(node_ids))

            # Count isolated nodes (nodes with no edges)
            isolated_count = 0
            node_edges = defaultdict(int)

            for edge in subgraph.get("edges", []):
                node_edges[edge["source"]] += 1
                node_edges[edge["target"]] += 1

            for node_id in node_ids:
                if node_edges[node_id] == 0:
                    isolated_count += 1

            # Identify clusters
            clusters = self._identify_clusters(subgraph)

            return {
                "isolated_nodes": isolated_count,
                "cluster_count": len(clusters),
                "avg_connectivity": sum(node_edges.values()) / len(node_ids)
                if node_ids
                else 0,
            }
        except Exception as e:
            logger.warning(f"Failed to analyze KG connectivity: {str(e)}")
            return {
                "isolated_nodes": 0,
                "cluster_count": 1,
                "avg_connectivity": 0,
            }

    def _generate_theme_based_suggestions(
        self, themes: List[str], context: ConversationContext
    ) -> List[str]:
        """Generate suggestions based on identified themes."""
        suggestions = []

        theme_suggestions = {
            "explore": "Add curiosity-driven rewards to encourage systematic exploration",
            "goal": "Define hierarchical goals for multi-level planning",
            "trade": "Implement resource constraints to create meaningful trade-offs",
            "coordinate": "Add communication channels between agents",
            "learn": "Include memory mechanisms for experience retention",
            "adapt": "Add meta-learning capabilities for online adaptation",
        }

        for theme in themes:
            for keyword, suggestion in theme_suggestions.items():
                if keyword in theme:
                    suggestions.append(suggestion)

        return suggestions[:2]  # Limit theme-based suggestions

    async def _identify_capability_gaps(
        self, agent_ids: List[str], db: AsyncSession
    ) -> List[str]:
        """Identify missing capabilities across agents."""
        if not agent_ids:
            return ["basic perception", "goal-directed planning"]

        # Get agents from database
        unique_ids = list(set(agent_ids))
        result = await db.execute(
            select(Agent).where(Agent.id.in_(unique_ids))
        )
        agents = result.scalars().all()

        # Analyze capabilities
        capabilities = set()
        for agent in agents:
            if agent.pymdp_config:
                # Check for various capabilities
                if "C" in agent.pymdp_config:
                    capabilities.add("preferences")
                if agent.pymdp_config.get("planning_horizon", 0) > 1:
                    capabilities.add("planning")
                if agent.pymdp_config.get("num_controls", []):
                    capabilities.add("actions")

        # Identify gaps
        essential_capabilities = {
            "perception",
            "planning",
            "preferences",
            "actions",
            "learning",
        }
        gaps = essential_capabilities - capabilities

        return list(gaps)[:3]

    def _identify_clusters(self, subgraph: Dict[str, Any]) -> List[Set[str]]:
        """Simple cluster identification using connected components."""
        if not subgraph.get("nodes") or not subgraph.get("edges"):
            return []

        # Build adjacency list
        adjacency = defaultdict(set)
        nodes = {node["id"] for node in subgraph["nodes"]}

        for edge in subgraph["edges"]:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])

        # Find connected components
        visited = set()
        clusters = []

        for node in nodes:
            if node not in visited:
                cluster = set()
                stack = [node]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        stack.extend(adjacency[current] - visited)

                clusters.append(cluster)

        return clusters

    def _count_node_types(self, subgraph: Dict[str, Any]) -> Dict[str, int]:
        """Count node types in subgraph."""
        type_counts = defaultdict(int)

        for node in subgraph.get("nodes", []):
            node_type = node.get("type", "unknown")
            type_counts[node_type] += 1

        return dict(type_counts)

    def _calculate_prompt_similarity(
        self, prompt1: str, prompt2: str
    ) -> float:
        """Calculate similarity between two prompts (simplified)."""
        # Simple word overlap similarity
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _extract_prompt_themes(self, prompts: List[str]) -> List[str]:
        """Extract themes from prompts."""
        theme_keywords = {
            "exploration": ["explore", "discover", "search", "find"],
            "goal": ["goal", "objective", "target", "achieve"],
            "learning": ["learn", "adapt", "improve", "optimize"],
            "coordination": ["coordinate", "cooperate", "collaborate", "team"],
            "trading": ["trade", "exchange", "negotiate", "resource"],
        }

        themes = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    themes.append(theme)

        return themes

    def _determine_thematic_direction(self, themes: List[str]) -> str:
        """Determine overall thematic direction."""
        if not themes:
            return "exploratory"

        theme_counts = defaultdict(int)
        for theme in themes:
            theme_counts[theme] += 1

        # Find dominant theme
        dominant_theme = max(theme_counts.items(), key=lambda x: x[1])[0]

        direction_map = {
            "exploration": "discovery-oriented",
            "goal": "achievement-focused",
            "learning": "adaptation-centered",
            "coordination": "collaboration-driven",
            "trading": "resource-optimizing",
        }

        return direction_map.get(dominant_theme, "multi-faceted")
