"""Natural Language Generation for response enhancement."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from inference.llm.provider_factory import create_llm_manager, ProviderNotAvailableError
from inference.llm.provider_interface import GenerationRequest

from .models import ResponseData, ResponseOptions

logger = logging.getLogger(__name__)


class NaturalLanguageGenerator(ABC):
    """Abstract base class for natural language generators."""
    
    @abstractmethod
    async def enhance_message(
        self,
        response_data: ResponseData,
        original_prompt: str,
        options: ResponseOptions,
    ) -> Optional[str]:
        """Enhance response message with natural language generation.
        
        Args:
            response_data: Structured response data
            original_prompt: Original user prompt
            options: Response generation options
            
        Returns:
            Enhanced message or None if enhancement failed
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get NLG performance metrics."""
        pass


class LLMEnhancedGenerator(NaturalLanguageGenerator):
    """LLM-powered natural language generator with fallbacks and monitoring.
    
    This generator uses the established LLM provider infrastructure to create
    engaging, natural language explanations of Active Inference decision-making
    processes while maintaining production-ready error handling.
    """
    
    def __init__(
        self,
        llm_manager=None,
        enable_caching: bool = True,
        default_timeout_ms: int = 3000,
    ):
        """Initialize the LLM-enhanced generator.
        
        Args:
            llm_manager: LLM manager instance (will create if None)
            enable_caching: Enable prompt caching for repeated patterns
            default_timeout_ms: Default timeout for LLM calls
        """
        self.llm_manager = llm_manager
        self.enable_caching = enable_caching
        self.default_timeout_ms = default_timeout_ms
        
        # Prompt templates for different scenarios
        self.templates = {
            "high_confidence": """Transform this technical Active Inference analysis into an engaging, natural explanation:

Original prompt: "{original_prompt}"
Action selected: {action_label}
Confidence: {confidence:.2f} (high confidence)
Reasoning: {rationale}

Create a response that:
1. Explains the decision in natural, conversational language
2. Highlights why the agent is confident in this choice
3. Connects to the user's original request
4. Maintains scientific accuracy while being accessible
5. Keeps a positive, helpful tone

Response should be 1-2 sentences, under 200 words.""",

            "moderate_confidence": """Transform this technical Active Inference analysis into a thoughtful explanation:

Original prompt: "{original_prompt}"
Action selected: {action_label}
Confidence: {confidence:.2f} (moderate confidence)
Reasoning: {rationale}
Uncertainty areas: {uncertainty_areas}

Create a response that:
1. Explains the decision honestly, acknowledging uncertainty
2. Describes the reasoning process clearly
3. Suggests why this is still the best available choice
4. Connects to the user's original request
5. Maintains a balanced, informative tone

Response should be 2-3 sentences, under 250 words.""",

            "low_confidence": """Transform this technical Active Inference analysis into a cautious but helpful explanation:

Original prompt: "{original_prompt}"
Action selected: {action_label}
Confidence: {confidence:.2f} (low confidence)
Reasoning: {rationale}
Uncertainty areas: {uncertainty_areas}
Alternative options: {alternatives}

Create a response that:
1. Honestly acknowledges the high uncertainty
2. Explains why this option was still chosen
3. Mentions alternative approaches considered
4. Suggests gathering more information
5. Maintains a helpful, consultative tone

Response should be 2-4 sentences, under 300 words.""",

            "fallback": """Create a natural language explanation for this Active Inference result:

User asked: "{original_prompt}"
System selected: {action_label}
Confidence level: {confidence:.2f}

Explain this decision in simple, clear terms while mentioning that it used Active Inference principles. Keep it under 150 words.""",
        }
        
        # Performance tracking
        self._metrics = {
            "enhancements_attempted": 0,
            "enhancements_successful": 0,
            "enhancements_failed": 0,
            "avg_enhancement_time_ms": 0.0,
            "cache_hits": 0,
            "timeout_failures": 0,
        }
        
        # Simple prompt cache (in production, this could use Redis)
        self._prompt_cache = {} if enable_caching else None
        
        logger.debug("LLMEnhancedGenerator initialized")
    
    async def enhance_message(
        self,
        response_data: ResponseData,
        original_prompt: str,
        options: ResponseOptions,
    ) -> Optional[str]:
        """Enhance response message with natural language generation."""
        start_time = time.time()
        self._metrics["enhancements_attempted"] += 1
        
        try:
            # Check cache first if enabled
            if self._prompt_cache:
                cache_key = self._generate_cache_key(response_data, original_prompt, options)
                if cache_key in self._prompt_cache:
                    self._metrics["cache_hits"] += 1
                    logger.debug("NLG cache hit")
                    return self._prompt_cache[cache_key]
            
            # Initialize LLM manager if needed
            if self.llm_manager is None:
                try:
                    self.llm_manager = create_llm_manager(user_id=options.user_id or "system")
                except ProviderNotAvailableError as e:
                    logger.warning(f"No LLM provider available for NLG: {e}")
                    self._metrics["enhancements_failed"] += 1
                    return None
            
            # Select appropriate template based on confidence
            confidence = response_data.confidence_rating.overall
            template_name = self._select_template(confidence)
            template = self.templates[template_name]
            
            # Build prompt with response data
            prompt = self._build_prompt(
                template=template,
                response_data=response_data,
                original_prompt=original_prompt,
                options=options,
            )
            
            # Create generation request
            generation_request = GenerationRequest(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",  # Use faster model for response enhancement
                temperature=0.7,  # Some creativity but not too much
                max_tokens=300,  # Reasonable limit for response enhancement
            )
            
            # Generate enhanced message with timeout
            try:
                timeout_seconds = options.llm_timeout_ms / 1000.0
                enhanced_response = await asyncio.wait_for(
                    self.llm_manager.generate_with_fallback(generation_request),
                    timeout=timeout_seconds,
                )
                
                # Extract content from response
                enhanced_message = None
                if hasattr(enhanced_response, 'content'):
                    enhanced_message = enhanced_response.content.strip()
                elif isinstance(enhanced_response, str):
                    enhanced_message = enhanced_response.strip()
                else:
                    enhanced_message = str(enhanced_response).strip()
                
                # Validate enhanced message
                if not enhanced_message or len(enhanced_message) < 10:
                    logger.warning("LLM returned empty or too short enhanced message")
                    self._metrics["enhancements_failed"] += 1
                    return None
                
                # Truncate if too long
                if len(enhanced_message) > options.max_message_length:
                    enhanced_message = enhanced_message[:options.max_message_length - 3] + "..."
                
                # Cache the result if caching is enabled
                if self._prompt_cache and cache_key:
                    self._prompt_cache[cache_key] = enhanced_message
                    
                    # Simple cache size management
                    if len(self._prompt_cache) > 100:
                        # Remove oldest entries (simple FIFO)
                        oldest_keys = list(self._prompt_cache.keys())[:20]
                        for old_key in oldest_keys:
                            del self._prompt_cache[old_key]
                
                # Update metrics
                enhancement_time = (time.time() - start_time) * 1000
                self._update_avg_time(enhancement_time)
                self._metrics["enhancements_successful"] += 1
                
                logger.debug(f"Message enhanced successfully in {enhancement_time:.2f}ms")
                return enhanced_message
                
            except asyncio.TimeoutError:
                logger.warning(f"LLM enhancement timed out after {options.llm_timeout_ms}ms")
                self._metrics["timeout_failures"] += 1
                self._metrics["enhancements_failed"] += 1
                return None
                
        except Exception as e:
            enhancement_time = (time.time() - start_time) * 1000
            logger.error(f"NLG enhancement failed after {enhancement_time:.2f}ms: {e}")
            self._metrics["enhancements_failed"] += 1
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get NLG performance metrics.""" 
        metrics = self._metrics.copy()
        
        # Add computed metrics
        total_attempts = metrics["enhancements_attempted"]
        if total_attempts > 0:
            metrics["success_rate"] = metrics["enhancements_successful"] / total_attempts
            metrics["failure_rate"] = metrics["enhancements_failed"] / total_attempts
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
        
        if self._prompt_cache:
            metrics["cache_size"] = len(self._prompt_cache)
            metrics["cache_enabled"] = True
        else:
            metrics["cache_size"] = 0
            metrics["cache_enabled"] = False
        
        return metrics
    
    def _select_template(self, confidence: float) -> str:
        """Select appropriate template based on confidence level."""
        if confidence >= 0.7:
            return "high_confidence"
        elif confidence >= 0.4:
            return "moderate_confidence"
        else:
            return "low_confidence"
    
    def _build_prompt(
        self,
        template: str,
        response_data: ResponseData,
        original_prompt: str,
        options: ResponseOptions,
    ) -> str:
        """Build LLM prompt from template and response data."""
        try:
            # Extract data from response
            action_label = response_data.action_explanation.action_label or "Action"
            confidence = response_data.confidence_rating.overall
            rationale = response_data.action_explanation.rationale or "Based on inference results"
            
            # Get uncertainty areas
            uncertainty_areas = response_data.belief_summary.uncertainty_areas
            uncertainty_text = ", ".join(uncertainty_areas) if uncertainty_areas else "None identified"
            
            # Get alternatives
            alternatives = response_data.action_explanation.alternatives_considered
            alternatives_text = ", ".join(alternatives) if alternatives else "None listed"
            
            # Format template
            prompt = template.format(
                original_prompt=original_prompt,
                action_label=action_label,
                confidence=confidence,
                rationale=rationale,
                uncertainty_areas=uncertainty_text,
                alternatives=alternatives_text,
            )
            
            return prompt
            
        except Exception as e:
            logger.warning(f"Error building NLG prompt: {e}")
            # Fallback to simple template
            return self.templates["fallback"].format(
                original_prompt=original_prompt,
                action_label=response_data.action_explanation.action_label or "Action",
                confidence=response_data.confidence_rating.overall,
            )
    
    def _generate_cache_key(
        self,
        response_data: ResponseData,
        original_prompt: str,
        options: ResponseOptions,
    ) -> str:
        """Generate cache key for prompt caching."""
        # Create key based on relevant factors
        key_parts = [
            str(response_data.action_explanation.action),
            f"{response_data.confidence_rating.overall:.2f}",
            str(hash(original_prompt)),
            str(options.narrative_style),
            str(options.max_message_length),
        ]
        
        return "_".join(key_parts)
    
    def _update_avg_time(self, time_ms: float) -> None:
        """Update average enhancement time using exponential moving average."""
        current_avg = self._metrics["avg_enhancement_time_ms"]
        if current_avg == 0.0:
            self._metrics["avg_enhancement_time_ms"] = time_ms
        else:
            # Use exponential moving average with alpha = 0.1
            alpha = 0.1
            self._metrics["avg_enhancement_time_ms"] = (alpha * time_ms) + ((1 - alpha) * current_avg)


class TemplateBasedGenerator(NaturalLanguageGenerator):
    """Template-based fallback generator for when LLM enhancement is unavailable.
    
    This generator provides natural language responses using pre-defined templates
    and simple text substitution. Used as a fallback when LLM providers are
    unavailable or when fast, deterministic responses are needed.
    """
    
    def __init__(self):
        """Initialize the template-based generator."""
        self.templates = {
            "high_confidence": [
                "Based on your request '{prompt}', I confidently recommend {action} (confidence: {confidence:.2f}). {rationale}",
                "After analyzing '{prompt}', the best approach is {action} with high confidence ({confidence:.2f}). {rationale}",
                "For '{prompt}', I strongly suggest {action} (confidence: {confidence:.2f}). {rationale}",
            ],
            "moderate_confidence": [
                "Considering your request '{prompt}', I recommend {action} (confidence: {confidence:.2f}). {rationale}",
                "Based on '{prompt}', {action} appears to be the best option (confidence: {confidence:.2f}). {rationale}",
                "For '{prompt}', I suggest {action} with moderate confidence ({confidence:.2f}). {rationale}",
            ],
            "low_confidence": [
                "Given the uncertainty in '{prompt}', I tentatively suggest {action} (confidence: {confidence:.2f}). {rationale}",
                "While uncertain about '{prompt}', {action} seems like the most reasonable approach (confidence: {confidence:.2f}). {rationale}",
                "Despite limited information about '{prompt}', I recommend {action} as the best available option (confidence: {confidence:.2f}). {rationale}",
            ],
        }
        
        self._metrics = {
            "enhancements_attempted": 0,
            "enhancements_successful": 0,
            "template_selections": {"high_confidence": 0, "moderate_confidence": 0, "low_confidence": 0},
        }
        
        # Simple counter for template rotation
        self._template_counter = 0
        
        logger.debug("TemplateBasedGenerator initialized")
    
    async def enhance_message(
        self,
        response_data: ResponseData,
        original_prompt: str,
        options: ResponseOptions,
    ) -> Optional[str]:
        """Enhance message using template-based generation."""
        self._metrics["enhancements_attempted"] += 1
        
        try:
            # Select template category based on confidence
            confidence = response_data.confidence_rating.overall
            if confidence >= 0.7:
                category = "high_confidence"
            elif confidence >= 0.4:
                category = "moderate_confidence"
            else:
                category = "low_confidence"
            
            self._metrics["template_selections"][category] += 1
            
            # Select specific template (rotate through available templates)
            templates = self.templates[category]
            template_idx = self._template_counter % len(templates)
            template = templates[template_idx]
            self._template_counter += 1
            
            # Format template
            action_label = response_data.action_explanation.action_label or "the recommended action"
            rationale = response_data.action_explanation.rationale or "This decision minimizes uncertainty."
            
            enhanced_message = template.format(
                prompt=original_prompt,
                action=action_label,
                confidence=confidence,
                rationale=rationale,
            )
            
            # Truncate if needed
            if len(enhanced_message) > options.max_message_length:
                enhanced_message = enhanced_message[:options.max_message_length - 3] + "..."
            
            self._metrics["enhancements_successful"] += 1
            
            logger.debug(f"Template-based enhancement successful using {category} template")
            return enhanced_message
            
        except Exception as e:
            logger.error(f"Template-based enhancement failed: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get template-based generator metrics."""
        metrics = self._metrics.copy()
        
        total_attempts = metrics["enhancements_attempted"]
        if total_attempts > 0:
            metrics["success_rate"] = metrics["enhancements_successful"] / total_attempts
        else:
            metrics["success_rate"] = 0.0
        
        metrics["implementation"] = "template_based"
        return metrics