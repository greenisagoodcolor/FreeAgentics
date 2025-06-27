"""
Module for FreeAgentics Active Inference implementation.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

"""
Personality and Trait System for FreeAgentics
This module provides a comprehensive personality and trait system that influences
agent behavior, decision-making, and interactions. It extends the basic Big Five
model with behavioral integration, trait evolution, and extensibility.
"""


class TraitCategory(Enum):
    """Categories of traits that can affect agent behavior"""

    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    BEHAVIORAL = "behavioral"
    PHYSICAL = "physical"
    SPECIALIZED = "specialized"


class TraitInfluence(Enum):
    """How traits influence behavior"""

    MULTIPLICATIVE = "multiplicative"  # Multiplies action probabilities
    ADDITIVE = "additive"  # Adds/subtracts from action scores
    THRESHOLD = "threshold"  # Enables/disables actions based on thresholds
    MODULATION = "modulation"  # Modulates other trait effects


@dataclass
class TraitDefinition:
    """Definition of a personality trait"""

    name: str
    category: TraitCategory
    description: str
    min_value: float = 0.0
    max_value: float = 1.0
    default_value: float = 0.5
    influence_type: TraitInfluence = TraitInfluence.MULTIPLICATIVE
    # Behavioral effects
    decision_weight: float = 1.0
    behavior_modifiers: Dict[str, float] = field(default_factory=dict)
    interaction_effects: Dict[str, float] = field(default_factory=dict)
    # Evolution parameters
    can_evolve: bool = True
    evolution_rate: float = 0.01
    stability: float = 0.8  # Resistance to change

    def validate_value(self, value: float) -> float:
        """Ensure trait value is within valid range"""
        return max(self.min_value, min(self.max_value, value))


@dataclass
class PersonalityTrait:
    """Instance of a trait with current value and history"""

    definition: TraitDefinition
    current_value: float
    base_value: float  # Original/stable value
    temporary_modifiers: Dict[str, tuple[float, datetime]] = field(default_factory=dict)
    evolution_history: List[tuple[datetime, float, str]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    def get_effective_value(self) -> float:
        """Get current effective value including temporary modifiers"""
        effective_value = self.current_value
        current_time = datetime.now()
        # Apply temporary modifiers that haven't expired
        expired_modifiers = []
        for modifier_name, (
            modifier_value,
            expiry_time,
        ) in self.temporary_modifiers.items():
            if current_time < expiry_time:
                effective_value += modifier_value
            else:
                expired_modifiers.append(modifier_name)
        # Clean up expired modifiers
        for modifier_name in expired_modifiers:
            del self.temporary_modifiers[modifier_name]
        return self.definition.validate_value(effective_value)

    def add_temporary_modifier(self, name: str, value: float, duration: timedelta) -> None:
        """Add a temporary modifier to the trait"""
        expiry_time = datetime.now() + duration
        self.temporary_modifiers[name] = (value, expiry_time)

    def evolve_trait(self, experience_impact: float, context: str = "") -> None:
        """Evolve the trait based on experience"""
        if not self.definition.can_evolve:
            return
        # Calculate evolution amount based on stability and experience
        evolution_amount = (
            experience_impact * self.definition.evolution_rate * (1 - self.definition.stability)
        )
        # Apply evolution with some randomness
        random_factor = random.uniform(-0.1, 0.1)
        total_change = evolution_amount + random_factor
        # Update trait value
        old_value = self.current_value
        self.current_value = self.definition.validate_value(self.current_value + total_change)
        # Record evolution history
        if abs(old_value - self.current_value) > 0.001:  # Only record significant changes
            self.evolution_history.append((datetime.now(), total_change, context))
            self.last_updated = datetime.now()


class PersonalitySystem:
    """Manages agent personality traits and their behavioral effects"""

    def __init__(self) -> None:
        self.trait_definitions: Dict[str, TraitDefinition] = {}
        self.behavior_modifiers: Dict[str, Callable] = {}
        self.interaction_handlers: Dict[str, Callable] = {}
        self._initialize_default_traits()

    def _initialize_default_traits(self) -> None:
        """Initialize default personality trait definitions"""
        # Big Five personality traits (enhanced)
        self.register_trait(
            TraitDefinition(
                name="openness",
                category=TraitCategory.COGNITIVE,
                description="Openness to experience and new ideas",
                behavior_modifiers={
                    "exploration": 1.5,
                    "learning": 1.3,
                    "creativity": 1.4,
                    "routine_following": 0.7,
                },
            )
        )
        self.register_trait(
            TraitDefinition(
                name="conscientiousness",
                category=TraitCategory.BEHAVIORAL,
                description="Organization, persistence, and goal-directed behavior",
                behavior_modifiers={
                    "goal_pursuit": 1.4,
                    "planning": 1.3,
                    "task_completion": 1.5,
                    "impulsiveness": 0.6,
                },
            )
        )
        self.register_trait(
            TraitDefinition(
                name="extraversion",
                category=TraitCategory.SOCIAL,
                description="Social energy and assertiveness",
                behavior_modifiers={
                    "social_interaction": 1.4,
                    "leadership": 1.3,
                    "communication": 1.2,
                    "solitary_activities": 0.7,
                },
                interaction_effects={
                    "group_formation": 1.3,
                    "conflict_resolution": 1.1,
                },
            )
        )
        self.register_trait(
            TraitDefinition(
                name="agreeableness",
                category=TraitCategory.SOCIAL,
                description="Cooperation, trust, and empathy",
                behavior_modifiers={
                    "cooperation": 1.4,
                    "helping": 1.3,
                    "competition": 0.7,
                    "negotiation": 1.1,
                },
                interaction_effects={"trust_building": 1.3, "coalition_joining": 1.2},
            )
        )
        self.register_trait(
            TraitDefinition(
                name="neuroticism",
                category=TraitCategory.EMOTIONAL,
                description="Emotional stability and stress resilience",
                behavior_modifiers={
                    "stress_response": 1.3,
                    "risk_taking": 0.7,
                    "decision_speed": 0.8,
                    "emotional_regulation": 0.6,
                },
            )
        )
        # Additional specialized traits
        self.register_trait(
            TraitDefinition(
                name="curiosity",
                category=TraitCategory.COGNITIVE,
                description="Drive to explore and understand",
                behavior_modifiers={
                    "information_seeking": 1.4,
                    "question_asking": 1.3,
                    "exploration": 1.2,
                },
            )
        )
        self.register_trait(
            TraitDefinition(
                name="risk_tolerance",
                category=TraitCategory.BEHAVIORAL,
                description="Willingness to take risks",
                behavior_modifiers={
                    "risk_taking": 1.5,
                    "cautious_behavior": 0.6,
                    "innovation": 1.2,
                },
            )
        )
        self.register_trait(
            TraitDefinition(
                name="empathy",
                category=TraitCategory.EMOTIONAL,
                description="Ability to understand others' emotions",
                interaction_effects={
                    "emotional_support": 1.4,
                    "conflict_mediation": 1.3,
                    "social_understanding": 1.2,
                },
            )
        )
        self.register_trait(
            TraitDefinition(
                name="dominance",
                category=TraitCategory.SOCIAL,
                description="Tendency to assert control and leadership",
                behavior_modifiers={
                    "leadership": 1.4,
                    "assertiveness": 1.3,
                    "following": 0.7,
                },
                interaction_effects={"hierarchy_formation": 1.3, "influence": 1.2},
            )
        )
        self.register_trait(
            TraitDefinition(
                name="adaptability",
                category=TraitCategory.BEHAVIORAL,
                description="Ability to adjust to changing circumstances",
                behavior_modifiers={
                    "flexibility": 1.3,
                    "learning": 1.2,
                    "routine_following": 0.8,
                },
            )
        )

    def register_trait(self, trait_definition: TraitDefinition) -> None:
        """Register a new trait definition"""
        self.trait_definitions[trait_definition.name] = trait_definition

    def create_personality_profile(
        self,
        agent_type: str = "basic",
        trait_values: Optional[Dict[str, float]] = None,
        random_variation: float = 0.1,
    ) -> "PersonalityProfile":
        """Create a new personality profile for an agent"""
        profile = PersonalityProfile(personality_system=self, agent_type=agent_type)
        # Initialize traits with custom values if provided
        if not profile.traits:  # Only if traits haven't been initialized yet
            profile._initialize_traits(trait_values, random_variation)
        return profile

    def calculate_behavior_modifier(
        self, personality_profile: "PersonalityProfile", behavior_type: str
    ) -> float:
        """Calculate the overall modifier for a behavior type"""
        total_modifier = 1.0
        for trait in personality_profile.traits.values():
            trait_value = trait.get_effective_value()
            if behavior_type in trait.definition.behavior_modifiers:
                base_modifier = trait.definition.behavior_modifiers[behavior_type]
                # Apply trait influence based on its type
                if trait.definition.influence_type == TraitInfluence.MULTIPLICATIVE:
                    # Higher trait values increase the modifier effect
                    trait_modifier = 1.0 + (trait_value - 0.5) * (base_modifier - 1.0)
                    total_modifier *= trait_modifier
                elif trait.definition.influence_type == TraitInfluence.ADDITIVE:
                    # Direct addition based on trait value
                    total_modifier += (trait_value - 0.5) * base_modifier
        return max(0.1, total_modifier)  # Ensure positive modifiers

    def calculate_interaction_effect(
        self, personality_profile: "PersonalityProfile", interaction_type: str
    ) -> float:
        """Calculate the effect on social interactions"""
        total_effect = 1.0
        for trait in personality_profile.traits.values():
            trait_value = trait.get_effective_value()
            if interaction_type in trait.definition.interaction_effects:
                base_effect = trait.definition.interaction_effects[interaction_type]
                trait_effect = 1.0 + (trait_value - 0.5) * (base_effect - 1.0)
                total_effect *= trait_effect
        return max(0.1, total_effect)


@dataclass
class PersonalityProfile:
    """Complete personality profile for an agent"""

    personality_system: PersonalitySystem
    agent_type: str
    traits: Dict[str, PersonalityTrait] = field(default_factory=dict)
    personality_summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize the personality profile"""
        if not self.traits:
            self._initialize_traits()
        self._generate_personality_summary()

    def _initialize_traits(
        self,
        trait_values: Optional[Dict[str, float]] = None,
        random_variation: float = 0.1,
    ) -> None:
        """Initialize all traits for this personality profile"""
        # Define agent type defaults
        agent_defaults = self._get_agent_type_defaults()
        for trait_name, trait_def in self.personality_system.trait_definitions.items():
            # Determine trait value from multiple sources
            if trait_values and trait_name in trait_values:
                base_value = trait_values[trait_name]
            elif trait_name in agent_defaults:
                base_value = agent_defaults[trait_name]
            else:
                base_value = trait_def.default_value
            # Add random variation
            if random_variation > 0:
                variation = random.uniform(-random_variation, random_variation)
                base_value += variation
            # Validate and create trait
            base_value = trait_def.validate_value(base_value)
            self.traits[trait_name] = PersonalityTrait(
                definition=trait_def, current_value=base_value, base_value=base_value
            )

    def _get_agent_type_defaults(self) -> Dict[str, float]:
        """Get default trait values for different agent types"""
        defaults = {
            "explorer": {
                "openness": 0.8,
                "curiosity": 0.9,
                "risk_tolerance": 0.7,
                "adaptability": 0.8,
                "conscientiousness": 0.6,
            },
            "merchant": {
                "conscientiousness": 0.8,
                "extraversion": 0.7,
                "risk_tolerance": 0.6,
                "dominance": 0.6,
                "agreeableness": 0.7,
            },
            "scholar": {
                "openness": 0.9,
                "conscientiousness": 0.8,
                "curiosity": 0.9,
                "extraversion": 0.4,
                "neuroticism": 0.3,
            },
            "guardian": {
                "conscientiousness": 0.9,
                "agreeableness": 0.7,
                "neuroticism": 0.2,
                "dominance": 0.7,
                "risk_tolerance": 0.3,
            },
        }
        return defaults.get(self.agent_type, {})

    def _generate_personality_summary(self) -> None:
        """Generate a human-readable personality summary"""
        high_traits = []
        low_traits = []
        for trait_name, trait in self.traits.items():
            value = trait.get_effective_value()
            if value > 0.7:
                high_traits.append(trait_name.replace("_", " ").title())
            elif value < 0.3:
                low_traits.append(trait_name.replace("_", " ").title())
        summary_parts = []
        if high_traits:
            summary_parts.append(f"High: {', '.join(high_traits)}")
        if low_traits:
            summary_parts.append(f"Low: {', '.join(low_traits)}")
        self.personality_summary = (
            "; ".join(summary_parts) if summary_parts else "Balanced personality"
        )

    def get_trait_value(self, trait_name: str) -> float:
        """Get the current effective value of a trait"""
        if trait_name in self.traits:
            return self.traits[trait_name].get_effective_value()
        return 0.5  # Default middle value

    def modify_trait_temporarily(
        self, trait_name: str, modifier: float, duration: timedelta
    ) -> None:
        """Apply a temporary modifier to a trait"""
        if trait_name in self.traits:
            self.traits[trait_name].add_temporary_modifier(
                f"temp_{datetime.now().timestamp()}", modifier, duration
            )
            self.last_updated = datetime.now()

    def evolve_traits_from_experience(self, experience_data: Dict[str, Any]) -> None:
        """Evolve traits based on agent experiences"""
        # Extract relevant experience impacts
        experience_type = experience_data.get("type", "general")
        success = experience_data.get("success", True)
        social_context = experience_data.get("social", False)
        risk_level = experience_data.get("risk", 0.0)
        # Calculate impact factors
        base_impact = 0.1 if success else -0.05
        # Apply specific trait evolution based on experience type
        if experience_type == "exploration":
            self._evolve_trait_if_exists("openness", base_impact, "exploration_experience")
            self._evolve_trait_if_exists("curiosity", base_impact, "exploration_experience")
        elif experience_type == "social_interaction":
            if social_context:
                self._evolve_trait_if_exists("extraversion", base_impact, "social_experience")
                self._evolve_trait_if_exists(
                    "agreeableness",
                    base_impact if success else -base_impact,
                    "social_experience",
                )
        elif experience_type == "risk_taking":
            risk_impact = base_impact * risk_level
            self._evolve_trait_if_exists("risk_tolerance", risk_impact, "risk_experience")
            if not success:
                self._evolve_trait_if_exists("neuroticism", abs(risk_impact), "failed_risk")
        elif experience_type == "cooperation":
            self._evolve_trait_if_exists("agreeableness", base_impact, "cooperation_experience")
            self._evolve_trait_if_exists("empathy", base_impact, "cooperation_experience")
        elif experience_type == "leadership":
            self._evolve_trait_if_exists("dominance", base_impact, "leadership_experience")
            self._evolve_trait_if_exists("extraversion", base_impact * 0.5, "leadership_experience")
        # Update summary after evolution
        self._generate_personality_summary()
        self.last_updated = datetime.now()

    def _evolve_trait_if_exists(self, trait_name: str, impact: float, context: str) -> None:
        """Helper method to evolve a trait if it exists."""
        if trait_name in self.traits:
            self.traits[trait_name].evolve_trait(impact, context)

    def get_behavior_modifier(self, behavior_type: str) -> float:
        """Get the personality modifier for a specific behavior"""
        return self.personality_system.calculate_behavior_modifier(self, behavior_type)

    def get_interaction_effect(self, interaction_type: str) -> float:
        """Get the personality effect on social interactions"""
        return self.personality_system.calculate_interaction_effect(self, interaction_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert personality profile to dictionary for serialization"""
        return {
            "agent_type": self.agent_type,
            "traits": {
                name: {
                    "current_value": trait.current_value,
                    "base_value": trait.base_value,
                    "effective_value": trait.get_effective_value(),
                    "evolution_history": [
                        {
                            "timestamp": timestamp.isoformat(),
                            "change": change,
                            "context": context,
                        }
                        for timestamp, change, context in trait.evolution_history
                    ],
                }
                for name, trait in self.traits.items()
            },
            "personality_summary": self.personality_summary,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], personality_system: PersonalitySystem
    ) -> "PersonalityProfile":
        """Create personality profile from dictionary data"""
        profile = cls(
            personality_system=personality_system,
            agent_type=data.get("agent_type", "basic"),
        )
        # Restore trait values
        if "traits" in data:
            for trait_name, trait_data in data["traits"].items():
                if trait_name in profile.traits:
                    trait = profile.traits[trait_name]
                    trait.current_value = trait_data.get("current_value", trait.current_value)
                    trait.base_value = trait_data.get("base_value", trait.base_value)
                    # Restore evolution history
                    if "evolution_history" in trait_data:
                        trait.evolution_history = [
                            (
                                datetime.fromisoformat(h["timestamp"]),
                                h["change"],
                                h["context"],
                            )
                            for h in trait_data["evolution_history"]
                        ]
        profile.personality_summary = data.get("personality_summary", "")
        if "created_at" in data:
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if "last_updated" in data:
            profile.last_updated = datetime.fromisoformat(data["last_updated"])
        return profile


# Singleton instance for global personality system
_global_personality_system = PersonalitySystem()


def get_personality_system() -> PersonalitySystem:
    """Get the global personality system instance"""
    return _global_personality_system


def create_personality_profile(
    agent_type: str = "basic", trait_values: Optional[Dict[str, float]] = None
) -> PersonalityProfile:
    """Convenience function to create a personality profile"""
    return _global_personality_system.create_personality_profile(agent_type, trait_values)
