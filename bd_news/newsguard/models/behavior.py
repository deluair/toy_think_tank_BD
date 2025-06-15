"""Behavior models for NewsGuard Bangladesh simulation.

This module implements behavioral models for different types of agents,
including decision-making patterns, psychological factors, and social behaviors.
"""

import uuid
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import numpy as np
from scipy import stats
import pandas as pd

from ..utils.logger import get_logger
from ..utils.helpers import (
    calculate_statistics, normalize_scores, exponential_decay,
    sigmoid_function, weighted_random_choice
)

logger = get_logger(__name__)


class BehaviorType(Enum):
    """Types of behaviors."""
    INFORMATION_SEEKING = "information_seeking"
    CONTENT_SHARING = "content_sharing"
    FACT_CHECKING = "fact_checking"
    OPINION_FORMATION = "opinion_formation"
    SOCIAL_INTERACTION = "social_interaction"
    TRUST_BUILDING = "trust_building"
    MISINFORMATION_SPREADING = "misinformation_spreading"
    CORRECTION_BEHAVIOR = "correction_behavior"


class PersonalityTrait(Enum):
    """Personality traits affecting behavior."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    CURIOSITY = "curiosity"
    SKEPTICISM = "skepticism"
    IMPULSIVENESS = "impulsiveness"


class CognitiveBias(Enum):
    """Cognitive biases affecting decision-making."""
    CONFIRMATION_BIAS = "confirmation_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    ANCHORING_BIAS = "anchoring_bias"
    BANDWAGON_EFFECT = "bandwagon_effect"
    AUTHORITY_BIAS = "authority_bias"
    RECENCY_BIAS = "recency_bias"
    NEGATIVITY_BIAS = "negativity_bias"
    OVERCONFIDENCE_BIAS = "overconfidence_bias"


class MotivationType(Enum):
    """Types of motivations."""
    INFORMATION_ACCURACY = "information_accuracy"
    SOCIAL_APPROVAL = "social_approval"
    ENTERTAINMENT = "entertainment"
    POLITICAL_AGENDA = "political_agenda"
    ECONOMIC_GAIN = "economic_gain"
    ALTRUISM = "altruism"
    CURIOSITY = "curiosity"
    HABIT = "habit"


class DecisionContext(Enum):
    """Contexts for decision-making."""
    CONTENT_CONSUMPTION = "content_consumption"
    CONTENT_SHARING = "content_sharing"
    SOURCE_SELECTION = "source_selection"
    FACT_VERIFICATION = "fact_verification"
    OPINION_EXPRESSION = "opinion_expression"
    SOCIAL_INTERACTION = "social_interaction"
    TRUST_ASSESSMENT = "trust_assessment"


@dataclass
class PersonalityProfile:
    """Personality profile for an agent."""
    agent_id: str
    
    # Big Five personality traits (0-1 scale)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    # Additional traits
    curiosity: float = 0.5
    skepticism: float = 0.5
    impulsiveness: float = 0.5
    risk_tolerance: float = 0.5
    
    # Cognitive biases (strength of bias, 0-1)
    cognitive_biases: Dict[str, float] = field(default_factory=dict)
    
    # Learning and adaptation
    learning_rate: float = 0.1
    memory_decay: float = 0.05
    
    # Social factors
    social_influence_susceptibility: float = 0.5
    authority_respect: float = 0.5
    peer_influence: float = 0.5
    
    def calculate_trait_score(self, trait: PersonalityTrait) -> float:
        """Calculate score for a specific trait.
        
        Args:
            trait: Personality trait
            
        Returns:
            Trait score (0-1)
        """
        trait_map = {
            PersonalityTrait.OPENNESS: self.openness,
            PersonalityTrait.CONSCIENTIOUSNESS: self.conscientiousness,
            PersonalityTrait.EXTRAVERSION: self.extraversion,
            PersonalityTrait.AGREEABLENESS: self.agreeableness,
            PersonalityTrait.NEUROTICISM: self.neuroticism,
            PersonalityTrait.CURIOSITY: self.curiosity,
            PersonalityTrait.SKEPTICISM: self.skepticism,
            PersonalityTrait.IMPULSIVENESS: self.impulsiveness
        }
        
        return trait_map.get(trait, 0.5)
    
    def get_bias_strength(self, bias: CognitiveBias) -> float:
        """Get strength of a cognitive bias.
        
        Args:
            bias: Cognitive bias
            
        Returns:
            Bias strength (0-1)
        """
        return self.cognitive_biases.get(bias.value, 0.3)  # Default moderate bias


@dataclass
class MotivationProfile:
    """Motivation profile for an agent."""
    agent_id: str
    
    # Motivation strengths (0-1 scale)
    motivations: Dict[str, float] = field(default_factory=dict)
    
    # Dynamic factors
    current_mood: float = 0.5  # 0=negative, 1=positive
    stress_level: float = 0.3  # 0=relaxed, 1=stressed
    energy_level: float = 0.7  # 0=tired, 1=energetic
    
    # Temporal patterns
    time_preferences: Dict[str, float] = field(default_factory=dict)  # Hour of day preferences
    
    # Context-dependent motivations
    context_motivations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_motivation_strength(self, motivation: MotivationType, context: DecisionContext = None) -> float:
        """Get motivation strength for a specific motivation.
        
        Args:
            motivation: Type of motivation
            context: Decision context
            
        Returns:
            Motivation strength (0-1)
        """
        base_strength = self.motivations.get(motivation.value, 0.5)
        
        # Apply context-specific adjustments
        if context and context.value in self.context_motivations:
            context_adjustment = self.context_motivations[context.value].get(motivation.value, 0.0)
            base_strength += context_adjustment
        
        # Apply mood and energy effects
        mood_effect = (self.current_mood - 0.5) * 0.2
        energy_effect = (self.energy_level - 0.5) * 0.1
        stress_effect = -(self.stress_level - 0.3) * 0.15
        
        adjusted_strength = base_strength + mood_effect + energy_effect + stress_effect
        
        return max(0.0, min(1.0, adjusted_strength))


@dataclass
class BehaviorPattern:
    """Behavior pattern for an agent."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    behavior_type: BehaviorType = BehaviorType.INFORMATION_SEEKING
    
    # Pattern characteristics
    frequency: float = 0.5  # How often this behavior occurs
    intensity: float = 0.5  # How strongly this behavior is expressed
    consistency: float = 0.5  # How consistent this behavior is
    
    # Triggers and conditions
    triggers: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal patterns
    time_of_day_preferences: Dict[int, float] = field(default_factory=dict)  # Hour -> preference
    day_of_week_preferences: Dict[int, float] = field(default_factory=dict)  # Day -> preference
    
    # Learning and adaptation
    reinforcement_history: List[Tuple[datetime, float]] = field(default_factory=list)  # (time, reward)
    adaptation_rate: float = 0.1
    
    # Context dependencies
    context_modifiers: Dict[str, float] = field(default_factory=dict)
    
    def calculate_activation_probability(self, 
                                       current_time: datetime,
                                       context: Dict[str, Any] = None) -> float:
        """Calculate probability of this behavior being activated.
        
        Args:
            current_time: Current simulation time
            context: Current context
            
        Returns:
            Activation probability (0-1)
        """
        base_probability = self.frequency
        
        # Time-based modifiers
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        time_modifier = self.time_of_day_preferences.get(hour, 1.0)
        day_modifier = self.day_of_week_preferences.get(day_of_week, 1.0)
        
        # Context modifiers
        context_modifier = 1.0
        if context:
            for key, value in context.items():
                if key in self.context_modifiers:
                    context_modifier *= (1.0 + self.context_modifiers[key] * value)
        
        # Recent reinforcement effect
        recent_reinforcement = self._calculate_recent_reinforcement(current_time)
        
        # Combine all factors
        probability = (
            base_probability * 
            time_modifier * 
            day_modifier * 
            context_modifier * 
            (1.0 + recent_reinforcement * 0.5)
        )
        
        return max(0.0, min(1.0, probability))
    
    def _calculate_recent_reinforcement(self, current_time: datetime) -> float:
        """Calculate effect of recent reinforcement.
        
        Args:
            current_time: Current time
            
        Returns:
            Reinforcement effect (-1 to 1)
        """
        if not self.reinforcement_history:
            return 0.0
        
        # Consider reinforcement from last 24 hours
        cutoff_time = current_time - timedelta(hours=24)
        recent_reinforcements = [
            reward for time, reward in self.reinforcement_history
            if time >= cutoff_time
        ]
        
        if not recent_reinforcements:
            return 0.0
        
        # Calculate weighted average with recency bias
        total_weight = 0.0
        weighted_sum = 0.0
        
        for time, reward in self.reinforcement_history:
            if time >= cutoff_time:
                # More recent reinforcements have higher weight
                hours_ago = (current_time - time).total_seconds() / 3600
                weight = math.exp(-hours_ago / 12)  # Exponential decay
                weighted_sum += reward * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def add_reinforcement(self, reward: float, timestamp: datetime = None) -> None:
        """Add reinforcement for this behavior pattern.
        
        Args:
            reward: Reward value (-1 to 1)
            timestamp: Time of reinforcement
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.reinforcement_history.append((timestamp, reward))
        
        # Keep only recent history (last 100 entries)
        if len(self.reinforcement_history) > 100:
            self.reinforcement_history = self.reinforcement_history[-100:]
        
        # Adapt behavior based on reinforcement
        if reward > 0:
            self.frequency = min(1.0, self.frequency + self.adaptation_rate * reward)
            self.intensity = min(1.0, self.intensity + self.adaptation_rate * reward * 0.5)
        else:
            self.frequency = max(0.0, self.frequency + self.adaptation_rate * reward)
            self.intensity = max(0.0, self.intensity + self.adaptation_rate * reward * 0.5)


@dataclass
class DecisionRecord:
    """Record of a decision made by an agent."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Decision details
    context: DecisionContext = DecisionContext.CONTENT_CONSUMPTION
    options: List[str] = field(default_factory=list)
    chosen_option: str = ""
    
    # Decision factors
    influencing_factors: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    decision_time: float = 0.0  # Time taken to decide (seconds)
    
    # Outcome
    outcome_satisfaction: Optional[float] = None  # Satisfaction with outcome (0-1)
    regret_level: Optional[float] = None  # Regret about decision (0-1)
    
    # Learning
    learning_value: float = 0.0  # How much was learned from this decision


class BehaviorModel:
    """Behavior model for agent decision-making and actions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize behavior model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Agent profiles
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        self.motivation_profiles: Dict[str, MotivationProfile] = {}
        self.behavior_patterns: Dict[str, List[BehaviorPattern]] = {}
        
        # Decision tracking
        self.decision_history: List[DecisionRecord] = []
        self.decision_models: Dict[DecisionContext, Callable] = {}
        
        # Learning and adaptation
        self.global_learning_rate = self.config.get('global_learning_rate', 0.05)
        self.behavior_drift_rate = self.config.get('behavior_drift_rate', 0.01)
        
        # Initialize decision models
        self._initialize_decision_models()
        
        logger.debug("Behavior model initialized")
    
    def _initialize_decision_models(self) -> None:
        """Initialize decision-making models for different contexts."""
        self.decision_models = {
            DecisionContext.CONTENT_CONSUMPTION: self._decide_content_consumption,
            DecisionContext.CONTENT_SHARING: self._decide_content_sharing,
            DecisionContext.SOURCE_SELECTION: self._decide_source_selection,
            DecisionContext.FACT_VERIFICATION: self._decide_fact_verification,
            DecisionContext.OPINION_EXPRESSION: self._decide_opinion_expression,
            DecisionContext.SOCIAL_INTERACTION: self._decide_social_interaction,
            DecisionContext.TRUST_ASSESSMENT: self._decide_trust_assessment
        }
    
    def create_personality_profile(self, agent_id: str, agent_type: str = 'reader') -> PersonalityProfile:
        """Create personality profile for an agent.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            
        Returns:
            Created personality profile
        """
        # Base personality traits by agent type
        trait_distributions = {
            'reader': {
                'openness': (0.4, 0.8),
                'conscientiousness': (0.3, 0.7),
                'extraversion': (0.2, 0.8),
                'agreeableness': (0.4, 0.8),
                'neuroticism': (0.2, 0.6),
                'curiosity': (0.5, 0.9),
                'skepticism': (0.3, 0.7),
                'impulsiveness': (0.2, 0.6)
            },
            'journalist': {
                'openness': (0.6, 0.9),
                'conscientiousness': (0.7, 0.9),
                'extraversion': (0.5, 0.8),
                'agreeableness': (0.4, 0.7),
                'neuroticism': (0.2, 0.5),
                'curiosity': (0.8, 1.0),
                'skepticism': (0.6, 0.9),
                'impulsiveness': (0.1, 0.4)
            },
            'fact_checker': {
                'openness': (0.7, 0.9),
                'conscientiousness': (0.8, 1.0),
                'extraversion': (0.3, 0.6),
                'agreeableness': (0.5, 0.8),
                'neuroticism': (0.1, 0.4),
                'curiosity': (0.7, 1.0),
                'skepticism': (0.8, 1.0),
                'impulsiveness': (0.0, 0.2)
            },
            'influencer': {
                'openness': (0.5, 0.8),
                'conscientiousness': (0.4, 0.7),
                'extraversion': (0.7, 1.0),
                'agreeableness': (0.6, 0.9),
                'neuroticism': (0.2, 0.6),
                'curiosity': (0.6, 0.9),
                'skepticism': (0.2, 0.6),
                'impulsiveness': (0.3, 0.7)
            },
            'bot': {
                'openness': (0.1, 0.3),
                'conscientiousness': (0.9, 1.0),
                'extraversion': (0.8, 1.0),
                'agreeableness': (0.2, 0.5),
                'neuroticism': (0.0, 0.1),
                'curiosity': (0.0, 0.2),
                'skepticism': (0.0, 0.1),
                'impulsiveness': (0.8, 1.0)
            }
        }
        
        distributions = trait_distributions.get(agent_type, trait_distributions['reader'])
        
        # Generate personality traits
        profile = PersonalityProfile(
            agent_id=agent_id,
            openness=random.uniform(*distributions['openness']),
            conscientiousness=random.uniform(*distributions['conscientiousness']),
            extraversion=random.uniform(*distributions['extraversion']),
            agreeableness=random.uniform(*distributions['agreeableness']),
            neuroticism=random.uniform(*distributions['neuroticism']),
            curiosity=random.uniform(*distributions['curiosity']),
            skepticism=random.uniform(*distributions['skepticism']),
            impulsiveness=random.uniform(*distributions['impulsiveness']),
            risk_tolerance=random.uniform(0.2, 0.8),
            learning_rate=random.uniform(0.05, 0.2),
            memory_decay=random.uniform(0.01, 0.1),
            social_influence_susceptibility=random.uniform(0.2, 0.8),
            authority_respect=random.uniform(0.3, 0.9),
            peer_influence=random.uniform(0.2, 0.8)
        )
        
        # Generate cognitive biases
        for bias in CognitiveBias:
            # Some biases are more common than others
            bias_probabilities = {
                CognitiveBias.CONFIRMATION_BIAS: 0.7,
                CognitiveBias.AVAILABILITY_HEURISTIC: 0.6,
                CognitiveBias.BANDWAGON_EFFECT: 0.5,
                CognitiveBias.AUTHORITY_BIAS: 0.4,
                CognitiveBias.RECENCY_BIAS: 0.6,
                CognitiveBias.NEGATIVITY_BIAS: 0.5,
                CognitiveBias.ANCHORING_BIAS: 0.4,
                CognitiveBias.OVERCONFIDENCE_BIAS: 0.3
            }
            
            base_strength = bias_probabilities.get(bias, 0.3)
            # Adjust based on personality
            if bias == CognitiveBias.CONFIRMATION_BIAS:
                base_strength *= (2.0 - profile.openness)  # Less open = more confirmation bias
            elif bias == CognitiveBias.AUTHORITY_BIAS:
                base_strength *= profile.authority_respect
            elif bias == CognitiveBias.BANDWAGON_EFFECT:
                base_strength *= profile.social_influence_susceptibility
            
            profile.cognitive_biases[bias.value] = min(1.0, base_strength * random.uniform(0.5, 1.5))
        
        self.personality_profiles[agent_id] = profile
        
        logger.debug(f"Created personality profile for {agent_id} ({agent_type})")
        
        return profile
    
    def create_motivation_profile(self, agent_id: str, agent_type: str = 'reader') -> MotivationProfile:
        """Create motivation profile for an agent.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            
        Returns:
            Created motivation profile
        """
        # Base motivations by agent type
        motivation_distributions = {
            'reader': {
                MotivationType.INFORMATION_ACCURACY.value: (0.5, 0.8),
                MotivationType.ENTERTAINMENT.value: (0.6, 0.9),
                MotivationType.SOCIAL_APPROVAL.value: (0.3, 0.7),
                MotivationType.CURIOSITY.value: (0.4, 0.8),
                MotivationType.HABIT.value: (0.5, 0.8)
            },
            'journalist': {
                MotivationType.INFORMATION_ACCURACY.value: (0.8, 1.0),
                MotivationType.ALTRUISM.value: (0.6, 0.9),
                MotivationType.CURIOSITY.value: (0.7, 1.0),
                MotivationType.SOCIAL_APPROVAL.value: (0.4, 0.7),
                MotivationType.ECONOMIC_GAIN.value: (0.3, 0.6)
            },
            'fact_checker': {
                MotivationType.INFORMATION_ACCURACY.value: (0.9, 1.0),
                MotivationType.ALTRUISM.value: (0.8, 1.0),
                MotivationType.CURIOSITY.value: (0.7, 0.9),
                MotivationType.SOCIAL_APPROVAL.value: (0.2, 0.5)
            },
            'influencer': {
                MotivationType.SOCIAL_APPROVAL.value: (0.7, 1.0),
                MotivationType.ECONOMIC_GAIN.value: (0.6, 0.9),
                MotivationType.ENTERTAINMENT.value: (0.5, 0.8),
                MotivationType.POLITICAL_AGENDA.value: (0.2, 0.6)
            },
            'bot': {
                MotivationType.POLITICAL_AGENDA.value: (0.8, 1.0),
                MotivationType.ECONOMIC_GAIN.value: (0.7, 1.0),
                MotivationType.HABIT.value: (0.9, 1.0)
            }
        }
        
        distributions = motivation_distributions.get(agent_type, motivation_distributions['reader'])
        
        # Generate motivations
        motivations = {}
        for motivation_type, (min_val, max_val) in distributions.items():
            motivations[motivation_type] = random.uniform(min_val, max_val)
        
        # Fill in missing motivations with low values
        for motivation in MotivationType:
            if motivation.value not in motivations:
                motivations[motivation.value] = random.uniform(0.1, 0.3)
        
        profile = MotivationProfile(
            agent_id=agent_id,
            motivations=motivations,
            current_mood=random.uniform(0.3, 0.8),
            stress_level=random.uniform(0.1, 0.6),
            energy_level=random.uniform(0.4, 0.9)
        )
        
        # Generate time preferences (when agent is most active)
        for hour in range(24):
            # Most people are active during day hours
            if 6 <= hour <= 22:
                preference = random.uniform(0.5, 1.0)
            else:
                preference = random.uniform(0.1, 0.5)
            profile.time_preferences[str(hour)] = preference
        
        self.motivation_profiles[agent_id] = profile
        
        logger.debug(f"Created motivation profile for {agent_id} ({agent_type})")
        
        return profile
    
    def create_behavior_patterns(self, agent_id: str, agent_type: str = 'reader') -> List[BehaviorPattern]:
        """Create behavior patterns for an agent.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            
        Returns:
            List of behavior patterns
        """
        patterns = []
        
        # Get personality and motivation profiles
        personality = self.personality_profiles.get(agent_id)
        motivation = self.motivation_profiles.get(agent_id)
        
        if not personality:
            personality = self.create_personality_profile(agent_id, agent_type)
        if not motivation:
            motivation = self.create_motivation_profile(agent_id, agent_type)
        
        # Define behavior patterns by agent type
        pattern_configs = {
            'reader': [
                (BehaviorType.INFORMATION_SEEKING, 0.7, 0.6),
                (BehaviorType.CONTENT_SHARING, 0.3, 0.5),
                (BehaviorType.SOCIAL_INTERACTION, 0.4, 0.6),
                (BehaviorType.OPINION_FORMATION, 0.5, 0.5)
            ],
            'journalist': [
                (BehaviorType.INFORMATION_SEEKING, 0.9, 0.8),
                (BehaviorType.CONTENT_SHARING, 0.8, 0.7),
                (BehaviorType.FACT_CHECKING, 0.7, 0.8),
                (BehaviorType.TRUST_BUILDING, 0.6, 0.7)
            ],
            'fact_checker': [
                (BehaviorType.FACT_CHECKING, 0.9, 0.9),
                (BehaviorType.CORRECTION_BEHAVIOR, 0.8, 0.8),
                (BehaviorType.INFORMATION_SEEKING, 0.8, 0.7),
                (BehaviorType.TRUST_BUILDING, 0.7, 0.8)
            ],
            'influencer': [
                (BehaviorType.CONTENT_SHARING, 0.9, 0.8),
                (BehaviorType.SOCIAL_INTERACTION, 0.8, 0.9),
                (BehaviorType.OPINION_FORMATION, 0.7, 0.8),
                (BehaviorType.INFORMATION_SEEKING, 0.6, 0.6)
            ],
            'bot': [
                (BehaviorType.CONTENT_SHARING, 0.9, 0.9),
                (BehaviorType.MISINFORMATION_SPREADING, 0.7, 0.8),
                (BehaviorType.SOCIAL_INTERACTION, 0.8, 0.7)
            ]
        }
        
        configs = pattern_configs.get(agent_type, pattern_configs['reader'])
        
        for behavior_type, base_frequency, base_intensity in configs:
            # Adjust based on personality and motivations
            frequency = base_frequency
            intensity = base_intensity
            
            # Personality adjustments
            if behavior_type == BehaviorType.INFORMATION_SEEKING:
                frequency *= (0.5 + 0.5 * personality.curiosity)
                intensity *= (0.5 + 0.5 * personality.openness)
            elif behavior_type == BehaviorType.CONTENT_SHARING:
                frequency *= (0.5 + 0.5 * personality.extraversion)
                intensity *= (0.5 + 0.5 * personality.extraversion)
            elif behavior_type == BehaviorType.FACT_CHECKING:
                frequency *= (0.5 + 0.5 * personality.skepticism)
                intensity *= (0.5 + 0.5 * personality.conscientiousness)
            elif behavior_type == BehaviorType.SOCIAL_INTERACTION:
                frequency *= (0.3 + 0.7 * personality.extraversion)
                intensity *= (0.3 + 0.7 * personality.agreeableness)
            
            # Motivation adjustments
            relevant_motivations = {
                BehaviorType.INFORMATION_SEEKING: [MotivationType.CURIOSITY, MotivationType.INFORMATION_ACCURACY],
                BehaviorType.CONTENT_SHARING: [MotivationType.SOCIAL_APPROVAL, MotivationType.ENTERTAINMENT],
                BehaviorType.FACT_CHECKING: [MotivationType.INFORMATION_ACCURACY, MotivationType.ALTRUISM],
                BehaviorType.SOCIAL_INTERACTION: [MotivationType.SOCIAL_APPROVAL, MotivationType.ENTERTAINMENT],
                BehaviorType.MISINFORMATION_SPREADING: [MotivationType.POLITICAL_AGENDA, MotivationType.ECONOMIC_GAIN]
            }
            
            if behavior_type in relevant_motivations:
                motivation_boost = np.mean([
                    motivation.motivations.get(mot.value, 0.5)
                    for mot in relevant_motivations[behavior_type]
                ])
                frequency *= (0.5 + 0.5 * motivation_boost)
                intensity *= (0.5 + 0.5 * motivation_boost)
            
            # Create behavior pattern
            pattern = BehaviorPattern(
                agent_id=agent_id,
                behavior_type=behavior_type,
                frequency=min(1.0, frequency),
                intensity=min(1.0, intensity),
                consistency=random.uniform(0.6, 0.9),
                adaptation_rate=personality.learning_rate
            )
            
            # Set time preferences based on behavior type
            if behavior_type in [BehaviorType.INFORMATION_SEEKING, BehaviorType.CONTENT_SHARING]:
                # More active during peak hours
                for hour in range(24):
                    if hour in [7, 8, 9, 12, 13, 18, 19, 20, 21]:
                        pattern.time_of_day_preferences[hour] = random.uniform(1.2, 1.5)
                    elif hour in [22, 23, 0, 1, 2, 3, 4, 5, 6]:
                        pattern.time_of_day_preferences[hour] = random.uniform(0.3, 0.7)
                    else:
                        pattern.time_of_day_preferences[hour] = random.uniform(0.8, 1.2)
            
            patterns.append(pattern)
        
        self.behavior_patterns[agent_id] = patterns
        
        logger.debug(f"Created {len(patterns)} behavior patterns for {agent_id} ({agent_type})")
        
        return patterns
    
    def make_decision(self, 
                     agent_id: str,
                     context: DecisionContext,
                     options: List[str],
                     context_data: Dict[str, Any] = None) -> DecisionRecord:
        """Make a decision for an agent.
        
        Args:
            agent_id: Agent identifier
            context: Decision context
            options: Available options
            context_data: Additional context data
            
        Returns:
            Decision record
        """
        start_time = datetime.now()
        
        # Get agent profiles
        personality = self.personality_profiles.get(agent_id)
        motivation = self.motivation_profiles.get(agent_id)
        
        if not personality or not motivation:
            # Create default profiles if missing
            if not personality:
                personality = self.create_personality_profile(agent_id)
            if not motivation:
                motivation = self.create_motivation_profile(agent_id)
        
        # Use appropriate decision model
        decision_model = self.decision_models.get(context, self._default_decision_model)
        
        # Make decision
        chosen_option, influencing_factors, confidence = decision_model(
            agent_id, options, personality, motivation, context_data or {}
        )
        
        decision_time = (datetime.now() - start_time).total_seconds()
        
        # Create decision record
        decision = DecisionRecord(
            agent_id=agent_id,
            context=context,
            options=options,
            chosen_option=chosen_option,
            influencing_factors=influencing_factors,
            confidence=confidence,
            decision_time=decision_time
        )
        
        self.decision_history.append(decision)
        
        # Keep only recent history
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]
        
        logger.debug(f"Agent {agent_id} chose '{chosen_option}' in context {context.value} (confidence: {confidence:.2f})")
        
        return decision
    
    def _default_decision_model(self, 
                              agent_id: str,
                              options: List[str],
                              personality: PersonalityProfile,
                              motivation: MotivationProfile,
                              context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Default decision-making model.
        
        Args:
            agent_id: Agent identifier
            options: Available options
            personality: Personality profile
            motivation: Motivation profile
            context_data: Context data
            
        Returns:
            Tuple of (chosen_option, influencing_factors, confidence)
        """
        if not options:
            return "", {}, 0.0
        
        # Simple random choice with personality bias
        if personality.impulsiveness > 0.7:
            # Impulsive agents make quick random choices
            chosen = random.choice(options)
            confidence = random.uniform(0.3, 0.7)
        else:
            # More deliberate choice
            chosen = random.choice(options)
            confidence = random.uniform(0.5, 0.9)
        
        influencing_factors = {
            'randomness': 0.5,
            'impulsiveness': personality.impulsiveness,
            'confidence_bias': personality.cognitive_biases.get('overconfidence_bias', 0.3)
        }
        
        return chosen, influencing_factors, confidence
    
    def _decide_content_consumption(self, 
                                  agent_id: str,
                                  options: List[str],
                                  personality: PersonalityProfile,
                                  motivation: MotivationProfile,
                                  context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Decision model for content consumption."""
        if not options:
            return "", {}, 0.0
        
        scores = {}
        
        for option in options:
            score = 0.0
            
            # Get content metadata from context
            content_data = context_data.get(option, {})
            
            # Curiosity factor
            novelty = content_data.get('novelty', 0.5)
            score += personality.curiosity * novelty * 0.3
            
            # Entertainment value
            entertainment_value = content_data.get('entertainment_value', 0.5)
            entertainment_motivation = motivation.get_motivation_strength(MotivationType.ENTERTAINMENT)
            score += entertainment_motivation * entertainment_value * 0.25
            
            # Information accuracy preference
            credibility = content_data.get('credibility', 0.5)
            accuracy_motivation = motivation.get_motivation_strength(MotivationType.INFORMATION_ACCURACY)
            score += accuracy_motivation * credibility * 0.25
            
            # Social factors
            popularity = content_data.get('popularity', 0.5)
            social_motivation = motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL)
            score += social_motivation * popularity * 0.2
            
            scores[option] = score
        
        # Apply cognitive biases
        self._apply_cognitive_biases(scores, personality, context_data)
        
        # Choose option with highest score (with some randomness)
        if personality.impulsiveness > 0.6:
            # More random for impulsive agents
            chosen = weighted_random_choice(list(scores.keys()), list(scores.values()))
        else:
            # More deterministic for deliberate agents
            chosen = max(scores.keys(), key=lambda x: scores[x])
        
        confidence = min(1.0, scores[chosen] / max(scores.values()) if scores.values() else 0.5)
        
        influencing_factors = {
            'curiosity': personality.curiosity * 0.3,
            'entertainment_seeking': motivation.get_motivation_strength(MotivationType.ENTERTAINMENT) * 0.25,
            'accuracy_seeking': motivation.get_motivation_strength(MotivationType.INFORMATION_ACCURACY) * 0.25,
            'social_influence': motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL) * 0.2
        }
        
        return chosen, influencing_factors, confidence
    
    def _decide_content_sharing(self, 
                              agent_id: str,
                              options: List[str],
                              personality: PersonalityProfile,
                              motivation: MotivationProfile,
                              context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Decision model for content sharing."""
        if not options:
            return "", {}, 0.0
        
        scores = {}
        
        for option in options:
            score = 0.0
            content_data = context_data.get(option, {})
            
            # Social approval motivation
            shareability = content_data.get('shareability', 0.5)
            social_motivation = motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL)
            score += social_motivation * shareability * 0.4
            
            # Extraversion factor
            score += personality.extraversion * 0.3
            
            # Content alignment with beliefs
            belief_alignment = content_data.get('belief_alignment', 0.5)
            score += belief_alignment * 0.2
            
            # Emotional impact
            emotional_impact = content_data.get('emotional_impact', 0.5)
            score += emotional_impact * (1.0 - personality.conscientiousness) * 0.1
            
            scores[option] = score
        
        # Apply cognitive biases
        self._apply_cognitive_biases(scores, personality, context_data)
        
        chosen = max(scores.keys(), key=lambda x: scores[x]) if scores else options[0]
        confidence = min(1.0, scores.get(chosen, 0.5) / max(scores.values()) if scores.values() else 0.5)
        
        influencing_factors = {
            'social_motivation': motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL) * 0.4,
            'extraversion': personality.extraversion * 0.3,
            'belief_alignment': 0.2,
            'emotional_response': 0.1
        }
        
        return chosen, influencing_factors, confidence
    
    def _decide_source_selection(self, 
                               agent_id: str,
                               options: List[str],
                               personality: PersonalityProfile,
                               motivation: MotivationProfile,
                               context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Decision model for source selection."""
        if not options:
            return "", {}, 0.0
        
        scores = {}
        
        for option in options:
            score = 0.0
            source_data = context_data.get(option, {})
            
            # Trust and credibility
            trust_score = source_data.get('trust_score', 0.5)
            accuracy_motivation = motivation.get_motivation_strength(MotivationType.INFORMATION_ACCURACY)
            score += accuracy_motivation * trust_score * 0.4
            
            # Familiarity and habit
            familiarity = source_data.get('familiarity', 0.5)
            habit_motivation = motivation.get_motivation_strength(MotivationType.HABIT)
            score += habit_motivation * familiarity * 0.3
            
            # Authority and expertise
            authority = source_data.get('authority', 0.5)
            score += personality.authority_respect * authority * 0.2
            
            # Bias alignment
            bias_alignment = source_data.get('bias_alignment', 0.5)
            confirmation_bias = personality.get_bias_strength(CognitiveBias.CONFIRMATION_BIAS)
            score += confirmation_bias * bias_alignment * 0.1
            
            scores[option] = score
        
        chosen = max(scores.keys(), key=lambda x: scores[x]) if scores else options[0]
        confidence = min(1.0, scores.get(chosen, 0.5) / max(scores.values()) if scores.values() else 0.5)
        
        influencing_factors = {
            'trust_seeking': accuracy_motivation * 0.4,
            'habit': habit_motivation * 0.3,
            'authority_respect': personality.authority_respect * 0.2,
            'confirmation_bias': confirmation_bias * 0.1
        }
        
        return chosen, influencing_factors, confidence
    
    def _decide_fact_verification(self, 
                                agent_id: str,
                                options: List[str],
                                personality: PersonalityProfile,
                                motivation: MotivationProfile,
                                context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Decision model for fact verification."""
        # Options might be: 'verify', 'ignore', 'share_anyway'
        if not options:
            return "", {}, 0.0
        
        scores = {}
        
        for option in options:
            score = 0.0
            
            if option == 'verify':
                # Skepticism and conscientiousness favor verification
                score += personality.skepticism * 0.4
                score += personality.conscientiousness * 0.3
                score += motivation.get_motivation_strength(MotivationType.INFORMATION_ACCURACY) * 0.3
            
            elif option == 'ignore':
                # Impulsiveness and low conscientiousness favor ignoring
                score += personality.impulsiveness * 0.3
                score += (1.0 - personality.conscientiousness) * 0.2
                score += (1.0 - personality.skepticism) * 0.2
                score += motivation.get_motivation_strength(MotivationType.HABIT) * 0.3
            
            elif option == 'share_anyway':
                # Social motivations and low skepticism
                score += motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL) * 0.4
                score += (1.0 - personality.skepticism) * 0.3
                score += personality.impulsiveness * 0.3
            
            scores[option] = score
        
        chosen = max(scores.keys(), key=lambda x: scores[x]) if scores else options[0]
        confidence = min(1.0, scores.get(chosen, 0.5) / max(scores.values()) if scores.values() else 0.5)
        
        influencing_factors = {
            'skepticism': personality.skepticism,
            'conscientiousness': personality.conscientiousness,
            'accuracy_motivation': motivation.get_motivation_strength(MotivationType.INFORMATION_ACCURACY),
            'social_motivation': motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL)
        }
        
        return chosen, influencing_factors, confidence
    
    def _decide_opinion_expression(self, 
                                 agent_id: str,
                                 options: List[str],
                                 personality: PersonalityProfile,
                                 motivation: MotivationProfile,
                                 context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Decision model for opinion expression."""
        if not options:
            return "", {}, 0.0
        
        # Simple model: extraversion and social motivation drive opinion expression
        expression_probability = (
            personality.extraversion * 0.5 +
            motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL) * 0.3 +
            (1.0 - personality.neuroticism) * 0.2
        )
        
        if 'express' in options and random.random() < expression_probability:
            chosen = 'express'
            confidence = expression_probability
        else:
            chosen = 'remain_silent' if 'remain_silent' in options else options[0]
            confidence = 1.0 - expression_probability
        
        influencing_factors = {
            'extraversion': personality.extraversion * 0.5,
            'social_approval_seeking': motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL) * 0.3,
            'confidence': (1.0 - personality.neuroticism) * 0.2
        }
        
        return chosen, influencing_factors, confidence
    
    def _decide_social_interaction(self, 
                                 agent_id: str,
                                 options: List[str],
                                 personality: PersonalityProfile,
                                 motivation: MotivationProfile,
                                 context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Decision model for social interaction."""
        if not options:
            return "", {}, 0.0
        
        scores = {}
        
        for option in options:
            score = 0.0
            interaction_data = context_data.get(option, {})
            
            # Extraversion drives social interaction
            score += personality.extraversion * 0.4
            
            # Social approval motivation
            score += motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL) * 0.3
            
            # Agreeableness affects interaction style
            if 'friendly' in option.lower() or 'supportive' in option.lower():
                score += personality.agreeableness * 0.2
            
            # Relationship strength
            relationship_strength = interaction_data.get('relationship_strength', 0.5)
            score += relationship_strength * 0.1
            
            scores[option] = score
        
        chosen = max(scores.keys(), key=lambda x: scores[x]) if scores else options[0]
        confidence = min(1.0, scores.get(chosen, 0.5) / max(scores.values()) if scores.values() else 0.5)
        
        influencing_factors = {
            'extraversion': personality.extraversion * 0.4,
            'social_motivation': motivation.get_motivation_strength(MotivationType.SOCIAL_APPROVAL) * 0.3,
            'agreeableness': personality.agreeableness * 0.2,
            'relationship_factor': 0.1
        }
        
        return chosen, influencing_factors, confidence
    
    def _decide_trust_assessment(self, 
                               agent_id: str,
                               options: List[str],
                               personality: PersonalityProfile,
                               motivation: MotivationProfile,
                               context_data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float]:
        """Decision model for trust assessment."""
        if not options:
            return "", {}, 0.0
        
        # Options might be trust levels: 'high_trust', 'medium_trust', 'low_trust', 'no_trust'
        
        # Base trust level influenced by personality
        base_trust = (
            personality.agreeableness * 0.3 +
            (1.0 - personality.neuroticism) * 0.2 +
            personality.openness * 0.2 +
            (1.0 - personality.skepticism) * 0.3
        )
        
        # Adjust based on context
        entity_data = context_data.get('entity', {})
        credibility = entity_data.get('credibility', 0.5)
        familiarity = entity_data.get('familiarity', 0.5)
        
        adjusted_trust = (
            base_trust * 0.6 +
            credibility * 0.3 +
            familiarity * 0.1
        )
        
        # Map to trust level
        if adjusted_trust > 0.75:
            chosen = 'high_trust' if 'high_trust' in options else options[-1]
        elif adjusted_trust > 0.5:
            chosen = 'medium_trust' if 'medium_trust' in options else options[len(options)//2]
        elif adjusted_trust > 0.25:
            chosen = 'low_trust' if 'low_trust' in options else options[1] if len(options) > 1 else options[0]
        else:
            chosen = 'no_trust' if 'no_trust' in options else options[0]
        
        confidence = abs(adjusted_trust - 0.5) * 2  # Higher confidence for extreme values
        
        influencing_factors = {
            'base_personality': base_trust * 0.6,
            'credibility_assessment': credibility * 0.3,
            'familiarity': familiarity * 0.1
        }
        
        return chosen, influencing_factors, confidence
    
    def _apply_cognitive_biases(self, 
                              scores: Dict[str, float],
                              personality: PersonalityProfile,
                              context_data: Dict[str, Any]) -> None:
        """Apply cognitive biases to decision scores.
        
        Args:
            scores: Option scores to modify
            personality: Personality profile
            context_data: Context data
        """
        # Confirmation bias - favor options that align with existing beliefs
        confirmation_bias = personality.get_bias_strength(CognitiveBias.CONFIRMATION_BIAS)
        for option in scores:
            option_data = context_data.get(option, {})
            belief_alignment = option_data.get('belief_alignment', 0.5)
            bias_adjustment = confirmation_bias * (belief_alignment - 0.5) * 0.3
            scores[option] += bias_adjustment
        
        # Availability heuristic - favor recently encountered options
        availability_bias = personality.get_bias_strength(CognitiveBias.AVAILABILITY_HEURISTIC)
        for option in scores:
            option_data = context_data.get(option, {})
            recency = option_data.get('recency', 0.5)
            bias_adjustment = availability_bias * recency * 0.2
            scores[option] += bias_adjustment
        
        # Authority bias - favor options from authoritative sources
        authority_bias = personality.get_bias_strength(CognitiveBias.AUTHORITY_BIAS)
        for option in scores:
            option_data = context_data.get(option, {})
            authority = option_data.get('authority', 0.5)
            bias_adjustment = authority_bias * authority * 0.25
            scores[option] += bias_adjustment
        
        # Bandwagon effect - favor popular options
        bandwagon_bias = personality.get_bias_strength(CognitiveBias.BANDWAGON_EFFECT)
        for option in scores:
            option_data = context_data.get(option, {})
            popularity = option_data.get('popularity', 0.5)
            bias_adjustment = bandwagon_bias * popularity * 0.2
            scores[option] += bias_adjustment
    
    def update_behavior_from_outcome(self, 
                                   decision_id: str,
                                   outcome_satisfaction: float,
                                   learning_context: Dict[str, Any] = None) -> None:
        """Update behavior based on decision outcome.
        
        Args:
            decision_id: Decision identifier
            outcome_satisfaction: Satisfaction with outcome (0-1)
            learning_context: Additional learning context
        """
        # Find the decision
        decision = None
        for d in self.decision_history:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if not decision:
            return
        
        # Update decision record
        decision.outcome_satisfaction = outcome_satisfaction
        decision.regret_level = 1.0 - outcome_satisfaction
        
        # Calculate learning value
        surprise = abs(outcome_satisfaction - decision.confidence)
        decision.learning_value = surprise * self.global_learning_rate
        
        # Update behavior patterns
        agent_patterns = self.behavior_patterns.get(decision.agent_id, [])
        
        for pattern in agent_patterns:
            # Find relevant patterns based on decision context
            if self._is_pattern_relevant(pattern, decision):
                # Calculate reinforcement
                reinforcement = (outcome_satisfaction - 0.5) * 2  # Convert to -1 to 1 range
                
                # Add reinforcement to pattern
                pattern.add_reinforcement(reinforcement, decision.timestamp)
        
        # Update personality (very slow adaptation)
        personality = self.personality_profiles.get(decision.agent_id)
        if personality and decision.learning_value > 0.5:  # Only for significant learning
            adaptation_rate = personality.learning_rate * 0.1  # Very slow personality change
            
            # Adjust traits based on outcome
            if outcome_satisfaction > 0.7:  # Positive outcome
                if decision.context == DecisionContext.FACT_VERIFICATION:
                    personality.skepticism = min(1.0, personality.skepticism + adaptation_rate)
                elif decision.context == DecisionContext.CONTENT_SHARING:
                    personality.extraversion = min(1.0, personality.extraversion + adaptation_rate)
            
            elif outcome_satisfaction < 0.3:  # Negative outcome
                if decision.context == DecisionContext.TRUST_ASSESSMENT:
                    personality.skepticism = min(1.0, personality.skepticism + adaptation_rate)
        
        logger.debug(f"Updated behavior for agent {decision.agent_id} based on decision outcome (satisfaction: {outcome_satisfaction:.2f})")
    
    def _is_pattern_relevant(self, pattern: BehaviorPattern, decision: DecisionRecord) -> bool:
        """Check if a behavior pattern is relevant to a decision.
        
        Args:
            pattern: Behavior pattern
            decision: Decision record
            
        Returns:
            True if pattern is relevant
        """
        # Map decision contexts to behavior types
        context_behavior_map = {
            DecisionContext.CONTENT_CONSUMPTION: [BehaviorType.INFORMATION_SEEKING],
            DecisionContext.CONTENT_SHARING: [BehaviorType.CONTENT_SHARING, BehaviorType.SOCIAL_INTERACTION],
            DecisionContext.SOURCE_SELECTION: [BehaviorType.INFORMATION_SEEKING, BehaviorType.TRUST_BUILDING],
            DecisionContext.FACT_VERIFICATION: [BehaviorType.FACT_CHECKING],
            DecisionContext.OPINION_EXPRESSION: [BehaviorType.OPINION_FORMATION, BehaviorType.SOCIAL_INTERACTION],
            DecisionContext.SOCIAL_INTERACTION: [BehaviorType.SOCIAL_INTERACTION],
            DecisionContext.TRUST_ASSESSMENT: [BehaviorType.TRUST_BUILDING]
        }
        
        relevant_behaviors = context_behavior_map.get(decision.context, [])
        return pattern.behavior_type in relevant_behaviors
    
    def simulate_behavior_drift(self) -> None:
        """Simulate gradual drift in behavior patterns over time."""
        for agent_id, patterns in self.behavior_patterns.items():
            for pattern in patterns:
                # Small random changes to simulate natural drift
                drift_magnitude = self.behavior_drift_rate
                
                pattern.frequency += random.uniform(-drift_magnitude, drift_magnitude)
                pattern.frequency = max(0.0, min(1.0, pattern.frequency))
                
                pattern.intensity += random.uniform(-drift_magnitude, drift_magnitude)
                pattern.intensity = max(0.0, min(1.0, pattern.intensity))
                
                pattern.consistency += random.uniform(-drift_magnitude/2, drift_magnitude/2)
                pattern.consistency = max(0.0, min(1.0, pattern.consistency))
    
    def get_agent_behavior_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get behavior summary for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Behavior summary
        """
        personality = self.personality_profiles.get(agent_id)
        motivation = self.motivation_profiles.get(agent_id)
        patterns = self.behavior_patterns.get(agent_id, [])
        
        # Recent decisions
        recent_decisions = [
            d for d in self.decision_history[-100:]
            if d.agent_id == agent_id
        ]
        
        summary = {
            'agent_id': agent_id,
            'personality': {
                'openness': personality.openness if personality else 0.5,
                'conscientiousness': personality.conscientiousness if personality else 0.5,
                'extraversion': personality.extraversion if personality else 0.5,
                'agreeableness': personality.agreeableness if personality else 0.5,
                'neuroticism': personality.neuroticism if personality else 0.5,
                'curiosity': personality.curiosity if personality else 0.5,
                'skepticism': personality.skepticism if personality else 0.5,
                'impulsiveness': personality.impulsiveness if personality else 0.5
            } if personality else {},
            'motivations': motivation.motivations if motivation else {},
            'behavior_patterns': [
                {
                    'type': pattern.behavior_type.value,
                    'frequency': pattern.frequency,
                    'intensity': pattern.intensity,
                    'consistency': pattern.consistency
                }
                for pattern in patterns
            ],
            'recent_decisions': len(recent_decisions),
            'decision_contexts': list(set(d.context.value for d in recent_decisions)),
            'average_confidence': np.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0.5,
            'average_satisfaction': np.mean([
                d.outcome_satisfaction for d in recent_decisions
                if d.outcome_satisfaction is not None
            ]) if any(d.outcome_satisfaction is not None for d in recent_decisions) else None
        }
        
        return summary
    
    def get_behavior_statistics(self) -> Dict[str, Any]:
        """Get overall behavior statistics.
        
        Returns:
            Behavior statistics
        """
        stats = {
            'total_agents': len(self.personality_profiles),
            'total_decisions': len(self.decision_history),
            'decision_contexts': {},
            'personality_distributions': {},
            'motivation_distributions': {},
            'behavior_pattern_distributions': {}
        }
        
        # Decision context distribution
        for decision in self.decision_history:
            context = decision.context.value
            stats['decision_contexts'][context] = stats['decision_contexts'].get(context, 0) + 1
        
        # Personality trait distributions
        if self.personality_profiles:
            traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism',
                     'curiosity', 'skepticism', 'impulsiveness']
            
            for trait in traits:
                values = [getattr(p, trait) for p in self.personality_profiles.values()]
                stats['personality_distributions'][trait] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Motivation distributions
        if self.motivation_profiles:
            all_motivations = set()
            for profile in self.motivation_profiles.values():
                all_motivations.update(profile.motivations.keys())
            
            for motivation in all_motivations:
                values = [
                    profile.motivations.get(motivation, 0.5)
                    for profile in self.motivation_profiles.values()
                ]
                stats['motivation_distributions'][motivation] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Behavior pattern distributions
        all_patterns = []
        for patterns in self.behavior_patterns.values():
            all_patterns.extend(patterns)
        
        if all_patterns:
            behavior_types = set(p.behavior_type for p in all_patterns)
            
            for behavior_type in behavior_types:
                type_patterns = [p for p in all_patterns if p.behavior_type == behavior_type]
                
                stats['behavior_pattern_distributions'][behavior_type.value] = {
                    'count': len(type_patterns),
                    'frequency': {
                        'mean': np.mean([p.frequency for p in type_patterns]),
                        'std': np.std([p.frequency for p in type_patterns])
                    },
                    'intensity': {
                        'mean': np.mean([p.intensity for p in type_patterns]),
                        'std': np.std([p.intensity for p in type_patterns])
                    }
                }
        
        return stats
    
    def export_behavior_data(self, filepath: str) -> None:
        """Export behavior data to file.
        
        Args:
            filepath: Output file path
        """
        data = {
            'personality_profiles': {
                agent_id: {
                    'openness': profile.openness,
                    'conscientiousness': profile.conscientiousness,
                    'extraversion': profile.extraversion,
                    'agreeableness': profile.agreeableness,
                    'neuroticism': profile.neuroticism,
                    'curiosity': profile.curiosity,
                    'skepticism': profile.skepticism,
                    'impulsiveness': profile.impulsiveness,
                    'cognitive_biases': profile.cognitive_biases
                }
                for agent_id, profile in self.personality_profiles.items()
            },
            'motivation_profiles': {
                agent_id: {
                    'motivations': profile.motivations,
                    'current_mood': profile.current_mood,
                    'stress_level': profile.stress_level,
                    'energy_level': profile.energy_level
                }
                for agent_id, profile in self.motivation_profiles.items()
            },
            'behavior_patterns': {
                agent_id: [
                    {
                        'behavior_type': pattern.behavior_type.value,
                        'frequency': pattern.frequency,
                        'intensity': pattern.intensity,
                        'consistency': pattern.consistency,
                        'reinforcement_count': len(pattern.reinforcement_history)
                    }
                    for pattern in patterns
                ]
                for agent_id, patterns in self.behavior_patterns.items()
            },
            'decision_summary': {
                'total_decisions': len(self.decision_history),
                'contexts': list(set(d.context.value for d in self.decision_history)),
                'average_confidence': np.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0.0,
                'satisfaction_rate': np.mean([
                    d.outcome_satisfaction for d in self.decision_history
                    if d.outcome_satisfaction is not None
                ]) if any(d.outcome_satisfaction is not None for d in self.decision_history) else None
            }
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Behavior data exported to {filepath}")


class SocialInfluenceModel:
    """Model for social influence and peer effects on behavior."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize social influence model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Influence parameters
        self.base_influence_strength = self.config.get('base_influence_strength', 0.3)
        self.distance_decay_rate = self.config.get('distance_decay_rate', 0.1)
        self.similarity_boost = self.config.get('similarity_boost', 0.5)
        
        # Social network effects
        self.network_effects: Dict[str, Dict[str, float]] = {}  # agent_id -> {influencer_id: strength}
        self.influence_history: List[Dict[str, Any]] = []
        
        logger.debug("Social influence model initialized")
    
    def calculate_social_influence(self, 
                                 agent_id: str,
                                 influencer_id: str,
                                 behavior_model: BehaviorModel,
                                 network_distance: int = 1) -> float:
        """Calculate social influence strength between two agents.
        
        Args:
            agent_id: Target agent
            influencer_id: Influencing agent
            behavior_model: Behavior model for personality data
            network_distance: Network distance between agents
            
        Returns:
            Influence strength (0-1)
        """
        # Get personality profiles
        target_personality = behavior_model.personality_profiles.get(agent_id)
        influencer_personality = behavior_model.personality_profiles.get(influencer_id)
        
        if not target_personality or not influencer_personality:
            return 0.0
        
        # Base influence based on target's susceptibility
        base_influence = (
            target_personality.social_influence_susceptibility * 0.4 +
            target_personality.peer_influence * 0.3 +
            (1.0 - target_personality.skepticism) * 0.3
        )
        
        # Distance decay
        distance_factor = math.exp(-self.distance_decay_rate * (network_distance - 1))
        
        # Similarity boost
        similarity = self._calculate_personality_similarity(target_personality, influencer_personality)
        similarity_factor = 1.0 + self.similarity_boost * similarity
        
        # Influencer's influence capability
        influencer_strength = (
            influencer_personality.extraversion * 0.4 +
            influencer_personality.agreeableness * 0.3 +
            (1.0 - influencer_personality.neuroticism) * 0.3
        )
        
        # Combine factors
        influence_strength = (
            base_influence * 
            distance_factor * 
            similarity_factor * 
            influencer_strength * 
            self.base_influence_strength
        )
        
        return min(1.0, influence_strength)
    
    def _calculate_personality_similarity(self, 
                                        personality1: PersonalityProfile,
                                        personality2: PersonalityProfile) -> float:
        """Calculate personality similarity between two agents.
        
        Args:
            personality1: First personality profile
            personality2: Second personality profile
            
        Returns:
            Similarity score (0-1)
        """
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        differences = []
        for trait in traits:
            val1 = getattr(personality1, trait)
            val2 = getattr(personality2, trait)
            differences.append(abs(val1 - val2))
        
        # Convert differences to similarity (lower difference = higher similarity)
        avg_difference = np.mean(differences)
        similarity = 1.0 - avg_difference
        
        return max(0.0, similarity)
    
    def apply_social_influence(self, 
                             agent_id: str,
                             influencers: List[Tuple[str, float]],  # (influencer_id, influence_weight)
                             behavior_model: BehaviorModel,
                             influence_type: str = 'general') -> Dict[str, float]:
        """Apply social influence to an agent's behavior.
        
        Args:
            agent_id: Target agent
            influencers: List of (influencer_id, influence_weight) tuples
            behavior_model: Behavior model
            influence_type: Type of influence
            
        Returns:
            Influence effects applied
        """
        if not influencers:
            return {}
        
        target_personality = behavior_model.personality_profiles.get(agent_id)
        target_motivation = behavior_model.motivation_profiles.get(agent_id)
        
        if not target_personality or not target_motivation:
            return {}
        
        influence_effects = {}
        
        # Calculate weighted influence from all influencers
        total_weight = sum(weight for _, weight in influencers)
        if total_weight == 0:
            return {}
        
        for influencer_id, weight in influencers:
            influence_strength = self.calculate_social_influence(
                agent_id, influencer_id, behavior_model
            )
            
            normalized_weight = weight / total_weight
            effective_influence = influence_strength * normalized_weight
            
            # Get influencer's characteristics
            influencer_personality = behavior_model.personality_profiles.get(influencer_id)
            influencer_motivation = behavior_model.motivation_profiles.get(influencer_id)
            
            if influencer_personality and influencer_motivation:
                # Influence on motivations
                for motivation, strength in influencer_motivation.motivations.items():
                    current_strength = target_motivation.motivations.get(motivation, 0.5)
                    influence_delta = (strength - current_strength) * effective_influence * 0.1
                    
                    new_strength = current_strength + influence_delta
                    target_motivation.motivations[motivation] = max(0.0, min(1.0, new_strength))
                    
                    influence_effects[f'motivation_{motivation}'] = influence_delta
                
                # Influence on mood and energy
                mood_influence = (influencer_motivation.current_mood - target_motivation.current_mood) * effective_influence * 0.05
                target_motivation.current_mood = max(0.0, min(1.0, target_motivation.current_mood + mood_influence))
                influence_effects['mood'] = mood_influence
        
        # Record influence event
        influence_event = {
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'influencers': influencers,
            'influence_type': influence_type,
            'effects': influence_effects
        }
        
        self.influence_history.append(influence_event)
        
        # Keep only recent history
        if len(self.influence_history) > 1000:
            self.influence_history = self.influence_history[-500:]
        
        logger.debug(f"Applied social influence to agent {agent_id} from {len(influencers)} influencers")
        
        return influence_effects