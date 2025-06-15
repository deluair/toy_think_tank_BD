"""Trust models for NewsGuard Bangladesh simulation.

This module implements trust and credibility models for news sources,
fact-checkers, and information verification in the Bangladesh media ecosystem.
"""

import uuid
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
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


class TrustLevel(Enum):
    """Trust levels in the system."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CredibilityFactor(Enum):
    """Factors affecting credibility."""
    SOURCE_REPUTATION = "source_reputation"
    CONTENT_QUALITY = "content_quality"
    FACT_CHECK_RESULTS = "fact_check_results"
    PEER_VALIDATION = "peer_validation"
    HISTORICAL_ACCURACY = "historical_accuracy"
    TRANSPARENCY = "transparency"
    EXPERTISE = "expertise"
    BIAS_ASSESSMENT = "bias_assessment"


class TrustEventType(Enum):
    """Types of trust-related events."""
    TRUST_INCREASED = "trust_increased"
    TRUST_DECREASED = "trust_decreased"
    CREDIBILITY_UPDATED = "credibility_updated"
    REPUTATION_CHANGED = "reputation_changed"
    FACT_CHECK_COMPLETED = "fact_check_completed"
    BIAS_DETECTED = "bias_detected"
    TRANSPARENCY_IMPROVED = "transparency_improved"


@dataclass
class TrustScore:
    """Trust score with detailed breakdown."""
    overall_score: float = 0.5
    source_trust: float = 0.5
    content_trust: float = 0.5
    network_trust: float = 0.5
    
    # Component scores
    accuracy_score: float = 0.5
    transparency_score: float = 0.5
    expertise_score: float = 0.5
    bias_score: float = 0.5  # Lower is better (less biased)
    
    # Confidence and uncertainty
    confidence: float = 0.5
    uncertainty: float = 0.5
    
    # Temporal factors
    recency_weight: float = 1.0
    stability: float = 0.5
    
    # Evidence
    evidence_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall trust score from components."""
        # Weighted combination of components
        weights = {
            'source_trust': 0.3,
            'content_trust': 0.3,
            'network_trust': 0.2,
            'accuracy_score': 0.2
        }
        
        score = (
            weights['source_trust'] * self.source_trust +
            weights['content_trust'] * self.content_trust +
            weights['network_trust'] * self.network_trust +
            weights['accuracy_score'] * self.accuracy_score
        )
        
        # Apply bias penalty (lower bias is better)
        bias_penalty = (1 - self.bias_score) * 0.1
        score = max(0.0, score - bias_penalty)
        
        # Apply confidence weighting
        score = score * self.confidence + 0.5 * (1 - self.confidence)
        
        self.overall_score = min(1.0, max(0.0, score))
        return self.overall_score


@dataclass
class CredibilityAssessment:
    """Detailed credibility assessment."""
    entity_id: str
    entity_type: str  # 'source', 'content', 'author'
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core credibility metrics
    credibility_score: float = 0.5
    reliability_score: float = 0.5
    accuracy_score: float = 0.5
    
    # Factor scores
    factor_scores: Dict[str, float] = field(default_factory=dict)
    
    # Assessment details
    assessor_id: str = ""
    assessment_method: str = "automated"
    confidence_level: float = 0.5
    
    # Evidence and reasoning
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    # Temporal information
    assessed_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustEvent:
    """Trust-related event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: TrustEventType = TrustEventType.TRUST_INCREASED
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Event participants
    source_entity: str = ""
    target_entity: str = ""
    
    # Event details
    old_score: float = 0.0
    new_score: float = 0.0
    change_magnitude: float = 0.0
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    
    # Impact
    impact_scope: List[str] = field(default_factory=list)
    propagation_depth: int = 0


class TrustModel:
    """Base trust model for the simulation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize trust model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Trust registries
        self.trust_scores: Dict[str, TrustScore] = {}
        self.credibility_assessments: Dict[str, List[CredibilityAssessment]] = {}
        self.trust_relationships: Dict[Tuple[str, str], float] = {}
        
        # Event tracking
        self.trust_events: List[TrustEvent] = []
        
        # Model parameters
        self.trust_decay_rate = self.config.get('trust_decay_rate', 0.05)
        self.trust_propagation_factor = self.config.get('trust_propagation_factor', 0.3)
        self.credibility_threshold = self.config.get('credibility_threshold', 0.6)
        self.update_frequency = self.config.get('update_frequency', 24)  # hours
        
        # Trust calculation weights
        self.factor_weights = self.config.get('factor_weights', {
            CredibilityFactor.SOURCE_REPUTATION.value: 0.25,
            CredibilityFactor.CONTENT_QUALITY.value: 0.20,
            CredibilityFactor.FACT_CHECK_RESULTS.value: 0.20,
            CredibilityFactor.HISTORICAL_ACCURACY.value: 0.15,
            CredibilityFactor.PEER_VALIDATION.value: 0.10,
            CredibilityFactor.TRANSPARENCY.value: 0.05,
            CredibilityFactor.EXPERTISE.value: 0.03,
            CredibilityFactor.BIAS_ASSESSMENT.value: 0.02
        })
        
        logger.debug("Trust model initialized")
    
    def initialize_trust_score(self, entity_id: str, entity_type: str = 'source') -> TrustScore:
        """Initialize trust score for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            
        Returns:
            Initial trust score
        """
        # Base trust scores by entity type
        base_scores = {
            'news_outlet': {'source_trust': 0.6, 'content_trust': 0.6, 'network_trust': 0.5},
            'fact_checker': {'source_trust': 0.8, 'content_trust': 0.8, 'network_trust': 0.7},
            'influencer': {'source_trust': 0.4, 'content_trust': 0.5, 'network_trust': 0.6},
            'reader': {'source_trust': 0.5, 'content_trust': 0.5, 'network_trust': 0.5},
            'bot': {'source_trust': 0.2, 'content_trust': 0.3, 'network_trust': 0.2},
            'government': {'source_trust': 0.6, 'content_trust': 0.6, 'network_trust': 0.5},
            'organization': {'source_trust': 0.6, 'content_trust': 0.6, 'network_trust': 0.5}
        }
        
        base = base_scores.get(entity_type, {'source_trust': 0.5, 'content_trust': 0.5, 'network_trust': 0.5})
        
        # Add some randomness
        trust_score = TrustScore(
            source_trust=max(0.1, min(0.9, base['source_trust'] + random.uniform(-0.1, 0.1))),
            content_trust=max(0.1, min(0.9, base['content_trust'] + random.uniform(-0.1, 0.1))),
            network_trust=max(0.1, min(0.9, base['network_trust'] + random.uniform(-0.1, 0.1))),
            accuracy_score=random.uniform(0.4, 0.8),
            transparency_score=random.uniform(0.3, 0.7),
            expertise_score=random.uniform(0.4, 0.8),
            bias_score=random.uniform(0.2, 0.8),
            confidence=random.uniform(0.3, 0.7)
        )
        
        trust_score.calculate_overall_score()
        
        self.trust_scores[entity_id] = trust_score
        
        logger.debug(f"Initialized trust score for {entity_id}: {trust_score.overall_score:.3f}")
        
        return trust_score
    
    def update_trust_score(self, 
                          entity_id: str,
                          factor: CredibilityFactor,
                          evidence: Dict[str, Any],
                          impact_magnitude: float = 0.1) -> TrustScore:
        """Update trust score based on new evidence.
        
        Args:
            entity_id: Entity to update
            factor: Credibility factor being updated
            evidence: Evidence for the update
            impact_magnitude: Magnitude of the impact
            
        Returns:
            Updated trust score
        """
        if entity_id not in self.trust_scores:
            self.initialize_trust_score(entity_id)
        
        trust_score = self.trust_scores[entity_id]
        old_score = trust_score.overall_score
        
        # Update specific factor
        self._update_factor_score(trust_score, factor, evidence, impact_magnitude)
        
        # Recalculate overall score
        trust_score.calculate_overall_score()
        trust_score.last_updated = datetime.now()
        trust_score.evidence_count += 1
        
        # Record trust event
        event = TrustEvent(
            event_type=TrustEventType.TRUST_INCREASED if trust_score.overall_score > old_score else TrustEventType.TRUST_DECREASED,
            source_entity=entity_id,
            old_score=old_score,
            new_score=trust_score.overall_score,
            change_magnitude=abs(trust_score.overall_score - old_score),
            context=evidence,
            reason=f"Updated {factor.value}"
        )
        self.trust_events.append(event)
        
        logger.debug(f"Updated trust score for {entity_id}: {old_score:.3f} -> {trust_score.overall_score:.3f}")
        
        return trust_score
    
    def _update_factor_score(self, 
                           trust_score: TrustScore,
                           factor: CredibilityFactor,
                           evidence: Dict[str, Any],
                           impact_magnitude: float) -> None:
        """Update specific factor score.
        
        Args:
            trust_score: Trust score to update
            factor: Factor to update
            evidence: Evidence for update
            impact_magnitude: Impact magnitude
        """
        # Determine impact direction and magnitude
        impact_direction = evidence.get('impact_direction', 'positive')
        impact_value = impact_magnitude if impact_direction == 'positive' else -impact_magnitude
        
        # Apply learning rate
        learning_rate = 0.1
        impact_value *= learning_rate
        
        if factor == CredibilityFactor.SOURCE_REPUTATION:
            trust_score.source_trust = max(0.0, min(1.0, trust_score.source_trust + impact_value))
            
        elif factor == CredibilityFactor.CONTENT_QUALITY:
            trust_score.content_trust = max(0.0, min(1.0, trust_score.content_trust + impact_value))
            
        elif factor == CredibilityFactor.FACT_CHECK_RESULTS:
            # Fact-check results have strong impact on accuracy
            fact_check_verdict = evidence.get('verdict', 'unverified')
            if fact_check_verdict == 'true':
                trust_score.accuracy_score = min(1.0, trust_score.accuracy_score + 0.2)
            elif fact_check_verdict == 'false':
                trust_score.accuracy_score = max(0.0, trust_score.accuracy_score - 0.3)
            elif fact_check_verdict == 'misleading':
                trust_score.accuracy_score = max(0.0, trust_score.accuracy_score - 0.2)
                
        elif factor == CredibilityFactor.HISTORICAL_ACCURACY:
            accuracy_rate = evidence.get('accuracy_rate', 0.5)
            trust_score.accuracy_score = 0.7 * trust_score.accuracy_score + 0.3 * accuracy_rate
            
        elif factor == CredibilityFactor.TRANSPARENCY:
            transparency_level = evidence.get('transparency_level', 0.5)
            trust_score.transparency_score = 0.8 * trust_score.transparency_score + 0.2 * transparency_level
            
        elif factor == CredibilityFactor.EXPERTISE:
            expertise_level = evidence.get('expertise_level', 0.5)
            trust_score.expertise_score = 0.9 * trust_score.expertise_score + 0.1 * expertise_level
            
        elif factor == CredibilityFactor.BIAS_ASSESSMENT:
            bias_level = evidence.get('bias_level', 0.5)
            trust_score.bias_score = 0.8 * trust_score.bias_score + 0.2 * bias_level
            
        elif factor == CredibilityFactor.PEER_VALIDATION:
            validation_score = evidence.get('validation_score', 0.5)
            trust_score.network_trust = 0.8 * trust_score.network_trust + 0.2 * validation_score
    
    def assess_credibility(self, 
                         entity_id: str,
                         entity_type: str,
                         assessor_id: str = 'system',
                         method: str = 'automated') -> CredibilityAssessment:
        """Perform comprehensive credibility assessment.
        
        Args:
            entity_id: Entity to assess
            entity_type: Type of entity
            assessor_id: ID of assessor
            method: Assessment method
            
        Returns:
            Credibility assessment
        """
        # Get or create trust score
        if entity_id not in self.trust_scores:
            self.initialize_trust_score(entity_id, entity_type)
        
        trust_score = self.trust_scores[entity_id]
        
        # Calculate factor scores
        factor_scores = {
            CredibilityFactor.SOURCE_REPUTATION.value: trust_score.source_trust,
            CredibilityFactor.CONTENT_QUALITY.value: trust_score.content_trust,
            CredibilityFactor.HISTORICAL_ACCURACY.value: trust_score.accuracy_score,
            CredibilityFactor.TRANSPARENCY.value: trust_score.transparency_score,
            CredibilityFactor.EXPERTISE.value: trust_score.expertise_score,
            CredibilityFactor.BIAS_ASSESSMENT.value: 1.0 - trust_score.bias_score,  # Invert bias
            CredibilityFactor.PEER_VALIDATION.value: trust_score.network_trust
        }
        
        # Calculate weighted credibility score
        credibility_score = sum(
            self.factor_weights.get(factor, 0.1) * score
            for factor, score in factor_scores.items()
        )
        
        # Generate assessment
        assessment = CredibilityAssessment(
            entity_id=entity_id,
            entity_type=entity_type,
            credibility_score=credibility_score,
            reliability_score=trust_score.overall_score,
            accuracy_score=trust_score.accuracy_score,
            factor_scores=factor_scores,
            assessor_id=assessor_id,
            assessment_method=method,
            confidence_level=trust_score.confidence,
            reasoning=self._generate_assessment_reasoning(factor_scores)
        )
        
        # Store assessment
        if entity_id not in self.credibility_assessments:
            self.credibility_assessments[entity_id] = []
        self.credibility_assessments[entity_id].append(assessment)
        
        logger.debug(f"Assessed credibility for {entity_id}: {credibility_score:.3f}")
        
        return assessment
    
    def _generate_assessment_reasoning(self, factor_scores: Dict[str, float]) -> str:
        """Generate reasoning for credibility assessment.
        
        Args:
            factor_scores: Factor scores
            
        Returns:
            Assessment reasoning
        """
        # Find strongest and weakest factors
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_factors[0]
        weakest = sorted_factors[-1]
        
        reasoning_parts = []
        
        # Overall assessment
        avg_score = np.mean(list(factor_scores.values()))
        if avg_score > 0.7:
            reasoning_parts.append("High credibility entity with strong performance across multiple factors.")
        elif avg_score > 0.5:
            reasoning_parts.append("Moderate credibility entity with mixed performance.")
        else:
            reasoning_parts.append("Low credibility entity with concerning performance indicators.")
        
        # Strongest factor
        reasoning_parts.append(f"Strongest factor: {strongest[0]} (score: {strongest[1]:.2f}).")
        
        # Weakest factor
        reasoning_parts.append(f"Area for improvement: {weakest[0]} (score: {weakest[1]:.2f}).")
        
        return " ".join(reasoning_parts)
    
    def propagate_trust(self, 
                       source_entity: str,
                       target_entities: List[str],
                       relationship_strength: float = 0.5) -> None:
        """Propagate trust through network relationships.
        
        Args:
            source_entity: Source of trust propagation
            target_entities: Entities to propagate trust to
            relationship_strength: Strength of relationships
        """
        if source_entity not in self.trust_scores:
            return
        
        source_trust = self.trust_scores[source_entity]
        propagation_strength = self.trust_propagation_factor * relationship_strength
        
        for target_entity in target_entities:
            if target_entity not in self.trust_scores:
                self.initialize_trust_score(target_entity)
            
            target_trust = self.trust_scores[target_entity]
            
            # Calculate trust influence
            trust_influence = source_trust.overall_score * propagation_strength
            
            # Update target trust (weighted average)
            weight = 0.1  # How much influence propagation has
            target_trust.network_trust = (
                (1 - weight) * target_trust.network_trust +
                weight * trust_influence
            )
            
            # Recalculate overall score
            target_trust.calculate_overall_score()
            
            # Record relationship
            self.trust_relationships[(source_entity, target_entity)] = relationship_strength
    
    def detect_trust_anomalies(self, time_window: int = 24) -> List[Dict[str, Any]]:
        """Detect anomalies in trust patterns.
        
        Args:
            time_window: Time window in hours
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        cutoff_time = datetime.now() - timedelta(hours=time_window)
        
        # Recent trust events
        recent_events = [
            event for event in self.trust_events
            if event.timestamp >= cutoff_time
        ]
        
        # Detect rapid trust changes
        entity_changes = defaultdict(list)
        for event in recent_events:
            entity_changes[event.source_entity].append(event.change_magnitude)
        
        for entity_id, changes in entity_changes.items():
            total_change = sum(changes)
            if total_change > 0.3:  # Significant change threshold
                anomalies.append({
                    'type': 'rapid_trust_change',
                    'entity_id': entity_id,
                    'total_change': total_change,
                    'event_count': len(changes),
                    'severity': 'high' if total_change > 0.5 else 'medium'
                })
        
        # Detect trust score outliers
        trust_scores = [score.overall_score for score in self.trust_scores.values()]
        if trust_scores:
            mean_trust = np.mean(trust_scores)
            std_trust = np.std(trust_scores)
            
            for entity_id, trust_score in self.trust_scores.items():
                z_score = abs(trust_score.overall_score - mean_trust) / (std_trust + 1e-6)
                if z_score > 2.5:  # Outlier threshold
                    anomalies.append({
                        'type': 'trust_outlier',
                        'entity_id': entity_id,
                        'trust_score': trust_score.overall_score,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
        
        return anomalies
    
    def calculate_trust_network_metrics(self) -> Dict[str, float]:
        """Calculate trust network metrics.
        
        Returns:
            Dictionary of trust network metrics
        """
        if not self.trust_scores:
            return {}
        
        trust_values = [score.overall_score for score in self.trust_scores.values()]
        
        metrics = {
            'avg_trust': np.mean(trust_values),
            'trust_variance': np.var(trust_values),
            'trust_std': np.std(trust_values),
            'min_trust': np.min(trust_values),
            'max_trust': np.max(trust_values),
            'trust_range': np.max(trust_values) - np.min(trust_values),
            'high_trust_ratio': sum(1 for t in trust_values if t > 0.7) / len(trust_values),
            'low_trust_ratio': sum(1 for t in trust_values if t < 0.3) / len(trust_values),
            'total_entities': len(self.trust_scores),
            'total_relationships': len(self.trust_relationships),
            'total_events': len(self.trust_events)
        }
        
        # Trust distribution by quartiles
        quartiles = np.percentile(trust_values, [25, 50, 75])
        metrics.update({
            'trust_q1': quartiles[0],
            'trust_median': quartiles[1],
            'trust_q3': quartiles[2]
        })
        
        return metrics
    
    def get_trust_recommendations(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get recommendations for improving trust.
        
        Args:
            entity_id: Entity to get recommendations for
            
        Returns:
            List of recommendations
        """
        if entity_id not in self.trust_scores:
            return []
        
        trust_score = self.trust_scores[entity_id]
        recommendations = []
        
        # Analyze weak areas
        factor_scores = {
            'accuracy': trust_score.accuracy_score,
            'transparency': trust_score.transparency_score,
            'expertise': trust_score.expertise_score,
            'bias': 1.0 - trust_score.bias_score,  # Invert bias
            'source_trust': trust_score.source_trust,
            'content_trust': trust_score.content_trust,
            'network_trust': trust_score.network_trust
        }
        
        # Find areas below threshold
        improvement_threshold = 0.6
        for factor, score in factor_scores.items():
            if score < improvement_threshold:
                priority = 'high' if score < 0.4 else 'medium'
                
                recommendation = {
                    'factor': factor,
                    'current_score': score,
                    'priority': priority,
                    'recommendation': self._get_factor_recommendation(factor, score)
                }
                recommendations.append(recommendation)
        
        # Sort by priority and score
        recommendations.sort(key=lambda x: (x['priority'] == 'high', -x['current_score']), reverse=True)
        
        return recommendations
    
    def _get_factor_recommendation(self, factor: str, score: float) -> str:
        """Get specific recommendation for a factor.
        
        Args:
            factor: Factor name
            score: Current score
            
        Returns:
            Recommendation text
        """
        recommendations = {
            'accuracy': "Improve fact-checking processes and source verification. Implement editorial review procedures.",
            'transparency': "Increase disclosure of sources, funding, and editorial processes. Publish correction policies.",
            'expertise': "Enhance journalist training and subject matter expertise. Collaborate with domain experts.",
            'bias': "Implement bias detection tools and diverse editorial perspectives. Regular bias audits recommended.",
            'source_trust': "Build reputation through consistent quality reporting and ethical journalism practices.",
            'content_trust': "Improve content quality through better research, writing, and editorial standards.",
            'network_trust': "Engage positively with peer networks and build collaborative relationships."
        }
        
        return recommendations.get(factor, "Focus on improving overall quality and reliability.")
    
    def simulate_trust_decay(self, decay_rate: Optional[float] = None) -> None:
        """Simulate natural trust decay over time.
        
        Args:
            decay_rate: Custom decay rate (uses model default if None)
        """
        if decay_rate is None:
            decay_rate = self.trust_decay_rate
        
        current_time = datetime.now()
        
        for entity_id, trust_score in self.trust_scores.items():
            # Calculate time since last update
            time_diff = (current_time - trust_score.last_updated).total_seconds() / 3600  # hours
            
            # Apply exponential decay
            decay_factor = exponential_decay(time_diff, decay_rate)
            
            # Decay towards neutral (0.5)
            neutral_score = 0.5
            trust_score.overall_score = (
                trust_score.overall_score * decay_factor +
                neutral_score * (1 - decay_factor)
            )
            
            # Decay component scores
            trust_score.source_trust = (
                trust_score.source_trust * decay_factor +
                neutral_score * (1 - decay_factor)
            )
            trust_score.content_trust = (
                trust_score.content_trust * decay_factor +
                neutral_score * (1 - decay_factor)
            )
            trust_score.network_trust = (
                trust_score.network_trust * decay_factor +
                neutral_score * (1 - decay_factor)
            )
            
            # Increase uncertainty over time
            trust_score.uncertainty = min(1.0, trust_score.uncertainty + decay_rate * 0.1)
            trust_score.confidence = max(0.1, trust_score.confidence - decay_rate * 0.05)
    
    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trust statistics.
        
        Returns:
            Dictionary of trust statistics
        """
        if not self.trust_scores:
            return {'total_entities': 0}
        
        # Basic statistics
        trust_values = [score.overall_score for score in self.trust_scores.values()]
        accuracy_values = [score.accuracy_score for score in self.trust_scores.values()]
        transparency_values = [score.transparency_score for score in self.trust_scores.values()]
        bias_values = [score.bias_score for score in self.trust_scores.values()]
        
        # Event statistics
        recent_events = [
            event for event in self.trust_events
            if event.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        event_type_counts = {}
        for event in recent_events:
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        # Trust level distribution
        trust_levels = {
            'very_low': sum(1 for t in trust_values if t < 0.2),
            'low': sum(1 for t in trust_values if 0.2 <= t < 0.4),
            'medium': sum(1 for t in trust_values if 0.4 <= t < 0.6),
            'high': sum(1 for t in trust_values if 0.6 <= t < 0.8),
            'very_high': sum(1 for t in trust_values if t >= 0.8)
        }
        
        stats = {
            'total_entities': len(self.trust_scores),
            'total_assessments': sum(len(assessments) for assessments in self.credibility_assessments.values()),
            'total_relationships': len(self.trust_relationships),
            'total_events': len(self.trust_events),
            'recent_events_24h': len(recent_events),
            'event_type_distribution': event_type_counts,
            'trust_distribution': trust_levels,
            'avg_trust_score': np.mean(trust_values),
            'avg_accuracy_score': np.mean(accuracy_values),
            'avg_transparency_score': np.mean(transparency_values),
            'avg_bias_score': np.mean(bias_values),
            'trust_variance': np.var(trust_values),
            'high_trust_entities': sum(1 for t in trust_values if t > 0.7),
            'low_trust_entities': sum(1 for t in trust_values if t < 0.3)
        }
        
        return stats


class ReputationModel:
    """Model for reputation management and tracking."""
    
    def __init__(self, trust_model: TrustModel, config: Dict[str, Any] = None):
        """Initialize reputation model.
        
        Args:
            trust_model: Associated trust model
            config: Model configuration
        """
        self.trust_model = trust_model
        self.config = config or {}
        
        # Reputation tracking
        self.reputation_scores: Dict[str, float] = {}
        self.reputation_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Reputation factors
        self.reputation_factors = {
            'accuracy_track_record': 0.3,
            'consistency': 0.2,
            'transparency': 0.15,
            'peer_recognition': 0.15,
            'public_trust': 0.1,
            'innovation': 0.05,
            'social_impact': 0.05
        }
        
        # Model parameters
        self.reputation_decay_rate = self.config.get('reputation_decay_rate', 0.02)
        self.reputation_momentum = self.config.get('reputation_momentum', 0.8)
        
        logger.debug("Reputation model initialized")
    
    def calculate_reputation_score(self, entity_id: str) -> float:
        """Calculate reputation score for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Reputation score (0-1)
        """
        # Get trust score
        if entity_id not in self.trust_model.trust_scores:
            return 0.5
        
        trust_score = self.trust_model.trust_scores[entity_id]
        
        # Calculate reputation components
        accuracy_track_record = trust_score.accuracy_score
        consistency = 1.0 - trust_score.uncertainty  # Lower uncertainty = higher consistency
        transparency = trust_score.transparency_score
        peer_recognition = trust_score.network_trust
        public_trust = trust_score.overall_score
        
        # Simulate innovation and social impact (would be calculated from actual data)
        innovation = random.uniform(0.3, 0.7)
        social_impact = random.uniform(0.3, 0.7)
        
        # Calculate weighted reputation score
        reputation_score = (
            self.reputation_factors['accuracy_track_record'] * accuracy_track_record +
            self.reputation_factors['consistency'] * consistency +
            self.reputation_factors['transparency'] * transparency +
            self.reputation_factors['peer_recognition'] * peer_recognition +
            self.reputation_factors['public_trust'] * public_trust +
            self.reputation_factors['innovation'] * innovation +
            self.reputation_factors['social_impact'] * social_impact
        )
        
        # Apply momentum (reputation changes slowly)
        if entity_id in self.reputation_scores:
            old_reputation = self.reputation_scores[entity_id]
            reputation_score = (
                self.reputation_momentum * old_reputation +
                (1 - self.reputation_momentum) * reputation_score
            )
        
        # Store reputation score and history
        self.reputation_scores[entity_id] = reputation_score
        
        if entity_id not in self.reputation_history:
            self.reputation_history[entity_id] = []
        self.reputation_history[entity_id].append((datetime.now(), reputation_score))
        
        # Keep only recent history (last 100 entries)
        if len(self.reputation_history[entity_id]) > 100:
            self.reputation_history[entity_id] = self.reputation_history[entity_id][-100:]
        
        return reputation_score
    
    def get_reputation_trend(self, entity_id: str, days: int = 30) -> Dict[str, Any]:
        """Get reputation trend for an entity.
        
        Args:
            entity_id: Entity identifier
            days: Number of days to analyze
            
        Returns:
            Reputation trend analysis
        """
        if entity_id not in self.reputation_history:
            return {'trend': 'no_data'}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            (date, score) for date, score in self.reputation_history[entity_id]
            if date >= cutoff_date
        ]
        
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        scores = [score for _, score in recent_history]
        
        # Linear regression for trend
        x = list(range(len(scores)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = 'improving'
        elif slope < -0.01:
            trend_direction = 'declining'
        else:
            trend_direction = 'stable'
        
        # Calculate volatility
        volatility = np.std(scores)
        
        trend_analysis = {
            'trend': trend_direction,
            'slope': slope,
            'r_squared': r_value ** 2,
            'volatility': volatility,
            'current_score': scores[-1],
            'score_change': scores[-1] - scores[0],
            'data_points': len(scores),
            'time_period_days': days
        }
        
        return trend_analysis
    
    def get_reputation_rankings(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top entities by reputation score.
        
        Args:
            limit: Number of top entities to return
            
        Returns:
            List of (entity_id, reputation_score) tuples
        """
        # Update all reputation scores
        for entity_id in self.trust_model.trust_scores:
            self.calculate_reputation_score(entity_id)
        
        # Sort by reputation score
        rankings = sorted(
            self.reputation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return rankings[:limit]


# Export main classes
__all__ = [
    'TrustModel', 'ReputationModel',
    'TrustScore', 'CredibilityAssessment', 'TrustEvent',
    'TrustLevel', 'CredibilityFactor', 'TrustEventType'
]