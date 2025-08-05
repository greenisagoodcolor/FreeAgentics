"""Response formatting implementations for structured response generation."""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agents.inference_engine import InferenceResult

from .models import (
    ActionExplanation,
    BeliefSummary,
    ConfidenceRating,
    ResponseData,
    ResponseOptions,
    ResponseMetadata,
    ResponseType,
)

logger = logging.getLogger(__name__)


class ResponseFormatter(ABC):
    """Abstract base class for response formatters."""
    
    @abstractmethod
    async def format_response(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        options: ResponseOptions,
        metadata: ResponseMetadata,
    ) -> ResponseData:
        """Format inference results into structured response data.
        
        Args:
            inference_result: Results from PyMDP inference
            original_prompt: Original user prompt
            options: Response generation options
            metadata: Response metadata
            
        Returns:
            Formatted response data
        """
        pass


class StructuredResponseFormatter(ResponseFormatter):
    """Production formatter that creates structured responses from inference results.
    
    This formatter transforms PyMDP inference results into user-friendly structured
    data following the established patterns in the codebase for data transformation
    and error handling.
    """
    
    def __init__(self):
        """Initialize the structured response formatter."""
        self.template_cache = {}
        
        # Action label mappings for common scenarios
        self.action_labels = {
            0: "Wait/Observe",
            1: "Move Forward", 
            2: "Move Backward",
            3: "Turn Left",
            4: "Turn Right",
            5: "Explore",
            6: "Analyze",
            7: "Decide",
        }
        
        logger.debug("StructuredResponseFormatter initialized")
    
    async def format_response(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        options: ResponseOptions,
        metadata: ResponseMetadata,
    ) -> ResponseData:
        """Format inference results into structured response data."""
        try:
            # Extract and format action explanation
            action_explanation = self._format_action_explanation(
                inference_result=inference_result,
                options=options,
            )
            
            # Extract and format belief summary  
            belief_summary = self._format_belief_summary(
                inference_result=inference_result,
                options=options,
            )
            
            # Calculate and format confidence rating
            confidence_rating = self._format_confidence_rating(
                inference_result=inference_result,
                options=options,
            )
            
            # Generate main response message
            message = self._generate_message(
                inference_result=inference_result,
                original_prompt=original_prompt,
                action_explanation=action_explanation,
                confidence_rating=confidence_rating,
                options=options,
            )
            
            # Create response data
            response_data = ResponseData(
                message=message,
                action_explanation=action_explanation,
                belief_summary=belief_summary,
                confidence_rating=confidence_rating,
                metadata=metadata,
                response_type=ResponseType.STRUCTURED,
            )
            
            # Add optional enrichment data if requested
            if options.include_knowledge_graph:
                response_data.knowledge_graph_updates = self._extract_kg_updates(inference_result)
            
            if options.include_alternatives:
                response_data.suggested_actions = self._generate_alternative_actions(
                    inference_result=inference_result,
                    selected_action=action_explanation.action,
                )
            
            # Add related concepts for educational purposes
            response_data.related_concepts = self._extract_related_concepts(
                inference_result=inference_result,
                original_prompt=original_prompt,
            )
            
            metadata.template_used = "structured_standard"
            return response_data
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            metadata.errors.append(f"Formatting error: {str(e)}")
            
            # Return minimal fallback response
            return self._create_fallback_response(
                inference_result=inference_result,
                original_prompt=original_prompt,
                metadata=metadata,
                error=e,
            )
    
    def _format_action_explanation(
        self,
        inference_result: InferenceResult,
        options: ResponseOptions,
    ) -> ActionExplanation:
        """Format action explanation from inference results."""
        action = inference_result.action
        
        # Convert action to readable label
        action_label = None
        if isinstance(action, (int, float)):
            action_int = int(action)
            action_label = self.action_labels.get(action_int, f"Action {action_int}")
        elif hasattr(action, 'item'):
            # Handle numpy scalars
            action_int = int(action.item())
            action_label = self.action_labels.get(action_int, f"Action {action_int}")
        else:
            action_label = str(action)
        
        # Generate rationale based on confidence and metadata
        rationale = self._generate_action_rationale(inference_result, options)
        
        # Extract decision factors from metadata
        decision_factors = []
        if hasattr(inference_result, 'metadata') and inference_result.metadata:
            if 'policy_precision' in inference_result.metadata:
                decision_factors.append(f"Policy precision: {inference_result.metadata['policy_precision']}")
            if 'free_energy' in inference_result.metadata:
                decision_factors.append(f"Free energy minimization")
            if 'planning_horizon' in inference_result.metadata:
                decision_factors.append(f"Planning horizon: {inference_result.metadata['planning_horizon']}")
        
        # Generate alternatives considered (simplified for now)
        alternatives = []
        if options.include_alternatives:
            alternatives = self._generate_alternatives_list(action, action_label)
        
        return ActionExplanation(
            action=action,
            action_label=action_label,
            rationale=rationale,
            alternatives_considered=alternatives,
            decision_factors=decision_factors,
        )
    
    def _format_belief_summary(
        self,
        inference_result: InferenceResult,
        options: ResponseOptions,
    ) -> BeliefSummary:
        """Format belief summary from inference results."""
        beliefs = inference_result.beliefs
        states = beliefs.get("states", [])
        
        # Calculate entropy if not available
        entropy = 0.0
        most_likely_state = None
        belief_distribution = {}
        uncertainty_areas = []
        
        if states:
            try:
                # Handle different state formats
                if isinstance(states, list) and len(states) > 0:
                    if isinstance(states[0], list):
                        # Multi-factor case - use first factor
                        state_probs = states[0]
                    else:
                        # Single factor case
                        state_probs = states
                    
                    # Ensure we have a list of floats
                    if not isinstance(state_probs, list):
                        state_probs = state_probs.tolist() if hasattr(state_probs, 'tolist') else [state_probs]
                    
                    # Calculate entropy
                    entropy = self._calculate_entropy(state_probs)
                    
                    # Find most likely state
                    if state_probs:
                        max_idx = state_probs.index(max(state_probs))
                        most_likely_state = f"State {max_idx}"
                    
                    # Create belief distribution
                    belief_distribution = {
                        f"State {i}": prob for i, prob in enumerate(state_probs)
                    }
                    
                    # Identify uncertainty areas (states with significant probability)
                    threshold = 0.1  # 10% threshold for significant probability
                    uncertainty_areas = [
                        f"State {i}" for i, prob in enumerate(state_probs)
                        if prob > threshold
                    ]
                    
            except Exception as e:
                logger.warning(f"Error processing belief states: {e}")
                uncertainty_areas = ["State processing unavailable"]
        
        return BeliefSummary(
            states=states,
            entropy=entropy,
            most_likely_state=most_likely_state,
            belief_distribution=belief_distribution,
            uncertainty_areas=uncertainty_areas,
        )
    
    def _format_confidence_rating(
        self,
        inference_result: InferenceResult,
        options: ResponseOptions,
    ) -> ConfidenceRating:
        """Format confidence rating from inference results."""
        overall_confidence = inference_result.confidence
        
        # Break down confidence into components
        action_confidence = overall_confidence
        belief_confidence = min(1.0, overall_confidence + 0.1)  # Slight boost for belief confidence
        model_confidence = 0.95  # Default model confidence
        
        # Extract factors from metadata
        factors = {}
        if hasattr(inference_result, 'metadata') and inference_result.metadata:
            if 'policy_precision' in inference_result.metadata:
                # Higher precision = higher confidence
                precision = inference_result.metadata['policy_precision']
                factors['policy_precision'] = min(1.0, precision / 20.0)  # Normalize assuming max ~20
            
            if 'action_precision' in inference_result.metadata:
                precision = inference_result.metadata['action_precision']
                factors['action_precision'] = min(1.0, precision / 20.0)
        
        # Add entropy-based uncertainty factor
        if inference_result.beliefs.get("states"):
            try:
                states = inference_result.beliefs["states"]
                if isinstance(states, list) and len(states) > 0:
                    state_probs = states[0] if isinstance(states[0], list) else states
                    if isinstance(state_probs, list):
                        entropy = self._calculate_entropy(state_probs)
                        # Lower entropy = higher confidence
                        max_entropy = math.log(len(state_probs)) if len(state_probs) > 1 else 1.0
                        factors['belief_certainty'] = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
            except Exception as e:
                logger.debug(f"Could not calculate entropy factor: {e}")
        
        return ConfidenceRating(
            overall=overall_confidence,
            action_confidence=action_confidence,
            belief_confidence=belief_confidence,
            model_confidence=model_confidence,
            factors=factors,
        )
    
    def _generate_message(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        action_explanation: ActionExplanation,
        confidence_rating: ConfidenceRating,
        options: ResponseOptions,
    ) -> str:
        """Generate the main response message."""
        if options.narrative_style:
            return self._generate_narrative_message(
                inference_result=inference_result,
                original_prompt=original_prompt,
                action_explanation=action_explanation,
                confidence_rating=confidence_rating,
                options=options,
            )
        else:
            return self._generate_structured_message(
                inference_result=inference_result,
                original_prompt=original_prompt,
                action_explanation=action_explanation,
                confidence_rating=confidence_rating,
                options=options,
            )
    
    def _generate_narrative_message(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        action_explanation: ActionExplanation,
        confidence_rating: ConfidenceRating,
        options: ResponseOptions,
    ) -> str:
        """Generate a narrative-style response message."""
        confidence_desc = self._describe_confidence_level(confidence_rating.overall)
        action_label = action_explanation.action_label or str(action_explanation.action)
        
        # Create narrative structure: setup -> process -> decision
        setup = f"Analyzing your request: '{original_prompt}'"
        
        process = (
            f"Using Active Inference principles, I updated my beliefs about the situation "
            f"and evaluated possible actions through Bayesian reasoning."
        )
        
        decision = (
            f"Based on this analysis, I recommend '{action_label}' with {confidence_desc} confidence "
            f"({confidence_rating.overall:.2f}). This decision minimizes expected free energy "
            f"while accounting for uncertainty in the environment."
        )
        
        # Combine into narrative
        message = f"{setup} {process} {decision}"
        
        # Add technical details if requested
        if options.include_technical_details:
            technical = (
                f" The inference process used {inference_result.metadata.get('pymdp_method', 'variational inference')} "
                f"with free energy of {inference_result.free_energy:.3f}."
            )
            message += technical
        
        # Truncate if too long
        if len(message) > options.max_message_length:
            message = message[:options.max_message_length - 3] + "..."
        
        return message
    
    def _generate_structured_message(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        action_explanation: ActionExplanation,
        confidence_rating: ConfidenceRating,
        options: ResponseOptions,
    ) -> str:
        """Generate a structured response message."""
        action_label = action_explanation.action_label or str(action_explanation.action)
        
        message = (
            f"Based on your prompt '{original_prompt}', the Active Inference agent "
            f"selected '{action_label}' with confidence {confidence_rating.overall:.2f}. "
        )
        
        if options.include_technical_details:
            message += (
                f"Free energy: {inference_result.free_energy:.3f}. "
                f"Method: {inference_result.metadata.get('pymdp_method', 'variational_inference')}. "
            )
        
        message += "The agent's beliefs were updated through Bayesian inference."
        
        return message
    
    def _generate_action_rationale(
        self,
        inference_result: InferenceResult,
        options: ResponseOptions,
    ) -> Optional[str]:
        """Generate rationale for the selected action."""
        confidence = inference_result.confidence
        
        if confidence > 0.8:
            return "This action was selected with high confidence based on strong evidence from the belief updates."
        elif confidence > 0.6:
            return "This action was chosen as the most promising option given the current evidence."
        elif confidence > 0.4:
            return "This action was selected despite some uncertainty in the current situation."
        else:
            return "This action was chosen as the best available option under high uncertainty."
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate entropy of probability distribution."""
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log(p)
        return entropy
    
    def _describe_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to descriptive text."""
        if confidence >= 0.9:
            return "very high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "moderate"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very low"
    
    def _generate_alternatives_list(
        self,
        selected_action: Any,
        action_label: str,
    ) -> List[str]:
        """Generate list of alternative actions that were considered."""
        alternatives = []
        
        # For integer actions, suggest adjacent actions
        if isinstance(selected_action, (int, float)):
            action_int = int(selected_action)
            for i in range(max(0, action_int - 1), min(8, action_int + 2)):
                if i != action_int:
                    alt_label = self.action_labels.get(i, f"Action {i}")
                    alternatives.append(alt_label)
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _extract_kg_updates(self, inference_result: InferenceResult) -> Optional[Dict[str, Any]]:
        """Extract knowledge graph updates from inference results."""
        # This would integrate with the knowledge graph updater
        # For now, return basic structure
        return {
            "new_beliefs": len(inference_result.beliefs.get("states", [])),
            "action_taken": inference_result.action,
            "confidence_level": inference_result.confidence,
            "timestamp": inference_result.metadata.get("timestamp"),
        }
    
    def _generate_alternative_actions(
        self,
        inference_result: InferenceResult,
        selected_action: Any,
    ) -> List[str]:
        """Generate suggested alternative actions."""
        suggestions = []
        
        # Based on confidence level, suggest different strategies
        confidence = inference_result.confidence
        
        if confidence < 0.5:
            suggestions.append("Gather more information before acting")
            suggestions.append("Consider a more conservative approach")
        elif confidence < 0.8:
            suggestions.append("Validate assumptions before proceeding")
            suggestions.append("Consider alternative interpretations")
        else:
            suggestions.append("Execute the plan with monitoring")
            suggestions.append("Prepare contingency plans")
        
        return suggestions
    
    def _extract_related_concepts(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
    ) -> List[str]:
        """Extract related concepts for educational purposes."""
        concepts = [
            "Active Inference",
            "Bayesian Brain Theory",
            "Free Energy Principle",
            "Variational Inference",
        ]
        
        # Add specific concepts based on inference metadata
        if inference_result.metadata:
            if "planning_horizon" in inference_result.metadata:
                concepts.append("Planning and Policy Selection")
            if "policy_precision" in inference_result.metadata:
                concepts.append("Precision-Weighted Inference")
        
        return concepts[:5]  # Limit to 5 concepts
    
    def _create_fallback_response(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        metadata: ResponseMetadata,
        error: Exception,
    ) -> ResponseData:
        """Create a minimal fallback response when formatting fails."""
        # Create minimal components
        action_explanation = ActionExplanation(
            action=inference_result.action if inference_result else "unknown",
            action_label="Action",
            rationale="Unable to generate detailed explanation",
        )
        
        belief_summary = BeliefSummary(
            states=[],
            entropy=0.0,
            most_likely_state="Unknown",
            uncertainty_areas=["Processing unavailable"],
        )
        
        confidence_rating = ConfidenceRating(
            overall=inference_result.confidence if inference_result else 0.0,
            action_confidence=0.0,
            belief_confidence=0.0,
        )
        
        message = (
            f"I processed your request '{original_prompt}' using Active Inference principles. "
            f"Due to a formatting issue, I can only provide basic response information."
        )
        
        metadata.template_used = "fallback_minimal"
        metadata.fallback_used = True
        
        return ResponseData(
            message=message,
            action_explanation=action_explanation,
            belief_summary=belief_summary,
            confidence_rating=confidence_rating,
            metadata=metadata,
            response_type=ResponseType.STRUCTURED,
        )