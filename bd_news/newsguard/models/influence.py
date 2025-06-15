"""Influence models for NewsGuard Bangladesh simulation.

This module implements influence propagation models, including information cascades,
opinion dynamics, and viral spread mechanisms.
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
import networkx as nx
import pandas as pd

from ..utils.logger import get_logger
from ..utils.helpers import (
    calculate_statistics, normalize_scores, exponential_decay,
    sigmoid_function, weighted_random_choice
)

logger = get_logger(__name__)


class InfluenceType(Enum):
    """Types of influence."""
    INFORMATION_SPREAD = "information_spread"
    OPINION_FORMATION = "opinion_formation"
    BEHAVIOR_ADOPTION = "behavior_adoption"
    TRUST_PROPAGATION = "trust_propagation"
    MISINFORMATION_SPREAD = "misinformation_spread"
    CORRECTION_SPREAD = "correction_spread"
    EMOTIONAL_CONTAGION = "emotional_contagion"
    SOCIAL_PROOF = "social_proof"


class CascadeModel(Enum):
    """Cascade propagation models."""
    LINEAR_THRESHOLD = "linear_threshold"
    INDEPENDENT_CASCADE = "independent_cascade"
    COMPLEX_CONTAGION = "complex_contagion"
    VIRAL_SPREAD = "viral_spread"
    RUMOR_SPREAD = "rumor_spread"


class InfluenceDirection(Enum):
    """Direction of influence."""
    INCOMING = "incoming"  # Influence received
    OUTGOING = "outgoing"  # Influence exerted
    BIDIRECTIONAL = "bidirectional"  # Mutual influence


@dataclass
class InfluenceEvent:
    """Record of an influence event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Event details
    influence_type: InfluenceType = InfluenceType.INFORMATION_SPREAD
    source_agent: str = ""
    target_agent: str = ""
    
    # Influence characteristics
    strength: float = 0.0  # Influence strength (0-1)
    content_id: Optional[str] = None  # Associated content
    network_path: List[str] = field(default_factory=list)  # Path through network
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = False  # Whether influence was successful
    
    # Cascade information
    cascade_id: Optional[str] = None
    generation: int = 0  # Generation in cascade (0 = original)
    
    # Outcome
    outcome_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class InfluenceCascade:
    """Information or influence cascade."""
    cascade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    
    # Cascade properties
    cascade_type: InfluenceType = InfluenceType.INFORMATION_SPREAD
    model: CascadeModel = CascadeModel.INDEPENDENT_CASCADE
    
    # Origin
    origin_agent: str = ""
    origin_content: Optional[str] = None
    
    # Propagation tracking
    affected_agents: Set[str] = field(default_factory=set)
    influence_events: List[InfluenceEvent] = field(default_factory=list)
    
    # Network structure
    propagation_tree: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    generation_map: Dict[str, int] = field(default_factory=dict)  # agent -> generation
    
    # Cascade metrics
    total_reach: int = 0
    max_generation: int = 0
    peak_rate: float = 0.0  # Peak propagation rate
    decay_rate: float = 0.0
    
    # Status
    is_active: bool = True
    end_time: Optional[datetime] = None
    
    def add_influence_event(self, event: InfluenceEvent) -> None:
        """Add influence event to cascade.
        
        Args:
            event: Influence event
        """
        event.cascade_id = self.cascade_id
        self.influence_events.append(event)
        
        if event.success:
            self.affected_agents.add(event.target_agent)
            
            # Update propagation tree
            if event.source_agent not in self.propagation_tree:
                self.propagation_tree[event.source_agent] = []
            self.propagation_tree[event.source_agent].append(event.target_agent)
            
            # Update generation
            source_generation = self.generation_map.get(event.source_agent, 0)
            target_generation = source_generation + 1
            self.generation_map[event.target_agent] = target_generation
            event.generation = target_generation
            
            self.max_generation = max(self.max_generation, target_generation)
            self.total_reach = len(self.affected_agents)
    
    def calculate_cascade_metrics(self) -> Dict[str, float]:
        """Calculate cascade performance metrics.
        
        Returns:
            Cascade metrics
        """
        if not self.influence_events:
            return {}
        
        # Time-based metrics
        duration = (self.end_time or datetime.now()) - self.start_time
        duration_hours = duration.total_seconds() / 3600
        
        # Success rate
        total_attempts = len(self.influence_events)
        successful_attempts = sum(1 for event in self.influence_events if event.success)
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        # Propagation rate
        propagation_rate = self.total_reach / duration_hours if duration_hours > 0 else 0.0
        
        # Network metrics
        branching_factor = np.mean([
            len(children) for children in self.propagation_tree.values()
        ]) if self.propagation_tree else 0.0
        
        # Generation distribution
        generation_counts = defaultdict(int)
        for generation in self.generation_map.values():
            generation_counts[generation] += 1
        
        # Viral coefficient (average number of successful transmissions per infected agent)
        viral_coefficient = successful_attempts / max(1, len(self.affected_agents))
        
        return {
            'total_reach': self.total_reach,
            'duration_hours': duration_hours,
            'success_rate': success_rate,
            'propagation_rate': propagation_rate,
            'max_generation': self.max_generation,
            'branching_factor': branching_factor,
            'viral_coefficient': viral_coefficient,
            'generation_diversity': len(generation_counts),
            'peak_generation_size': max(generation_counts.values()) if generation_counts else 0
        }


class InfluenceModel:
    """Model for influence propagation and cascades."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize influence model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Model parameters
        self.base_influence_probability = self.config.get('base_influence_probability', 0.1)
        self.threshold_decay_rate = self.config.get('threshold_decay_rate', 0.05)
        self.network_effect_strength = self.config.get('network_effect_strength', 0.3)
        
        # Cascade tracking
        self.active_cascades: Dict[str, InfluenceCascade] = {}
        self.completed_cascades: List[InfluenceCascade] = []
        self.influence_history: List[InfluenceEvent] = []
        
        # Agent influence states
        self.agent_thresholds: Dict[str, Dict[str, float]] = {}  # agent -> {influence_type: threshold}
        self.agent_exposures: Dict[str, Dict[str, int]] = {}  # agent -> {influence_type: exposure_count}
        self.agent_influence_scores: Dict[str, Dict[str, float]] = {}  # agent -> {influence_type: score}
        
        # Network influence weights
        self.influence_weights: Dict[Tuple[str, str], float] = {}  # (source, target) -> weight
        
        logger.debug("Influence model initialized")
    
    def initialize_agent_thresholds(self, 
                                  agent_id: str,
                                  agent_type: str = 'reader',
                                  personality_traits: Dict[str, float] = None) -> None:
        """Initialize influence thresholds for an agent.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            personality_traits: Personality traits affecting thresholds
        """
        traits = personality_traits or {}
        
        # Base thresholds by agent type
        base_thresholds = {
            'reader': {
                InfluenceType.INFORMATION_SPREAD.value: 0.3,
                InfluenceType.OPINION_FORMATION.value: 0.5,
                InfluenceType.BEHAVIOR_ADOPTION.value: 0.6,
                InfluenceType.TRUST_PROPAGATION.value: 0.4,
                InfluenceType.MISINFORMATION_SPREAD.value: 0.7,
                InfluenceType.EMOTIONAL_CONTAGION.value: 0.4
            },
            'journalist': {
                InfluenceType.INFORMATION_SPREAD.value: 0.2,
                InfluenceType.OPINION_FORMATION.value: 0.6,
                InfluenceType.BEHAVIOR_ADOPTION.value: 0.7,
                InfluenceType.TRUST_PROPAGATION.value: 0.3,
                InfluenceType.MISINFORMATION_SPREAD.value: 0.8,
                InfluenceType.EMOTIONAL_CONTAGION.value: 0.5
            },
            'influencer': {
                InfluenceType.INFORMATION_SPREAD.value: 0.1,
                InfluenceType.OPINION_FORMATION.value: 0.3,
                InfluenceType.BEHAVIOR_ADOPTION.value: 0.4,
                InfluenceType.TRUST_PROPAGATION.value: 0.2,
                InfluenceType.MISINFORMATION_SPREAD.value: 0.5,
                InfluenceType.EMOTIONAL_CONTAGION.value: 0.2
            },
            'fact_checker': {
                InfluenceType.INFORMATION_SPREAD.value: 0.1,
                InfluenceType.OPINION_FORMATION.value: 0.8,
                InfluenceType.BEHAVIOR_ADOPTION.value: 0.8,
                InfluenceType.TRUST_PROPAGATION.value: 0.2,
                InfluenceType.MISINFORMATION_SPREAD.value: 0.9,
                InfluenceType.EMOTIONAL_CONTAGION.value: 0.7
            },
            'bot': {
                InfluenceType.INFORMATION_SPREAD.value: 0.05,
                InfluenceType.OPINION_FORMATION.value: 0.2,
                InfluenceType.BEHAVIOR_ADOPTION.value: 0.3,
                InfluenceType.TRUST_PROPAGATION.value: 0.1,
                InfluenceType.MISINFORMATION_SPREAD.value: 0.1,
                InfluenceType.EMOTIONAL_CONTAGION.value: 0.1
            }
        }
        
        thresholds = base_thresholds.get(agent_type, base_thresholds['reader']).copy()
        
        # Adjust based on personality traits
        if traits:
            # Skepticism increases thresholds
            skepticism = traits.get('skepticism', 0.5)
            skepticism_factor = 1.0 + (skepticism - 0.5) * 0.5
            
            # Openness decreases thresholds for information
            openness = traits.get('openness', 0.5)
            openness_factor = 1.0 - (openness - 0.5) * 0.3
            
            # Social influence susceptibility
            social_susceptibility = traits.get('social_influence_susceptibility', 0.5)
            social_factor = 1.0 - (social_susceptibility - 0.5) * 0.4
            
            for influence_type in thresholds:
                if 'information' in influence_type.lower():
                    thresholds[influence_type] *= openness_factor
                if 'misinformation' in influence_type.lower():
                    thresholds[influence_type] *= skepticism_factor
                if 'social' in influence_type.lower() or 'emotional' in influence_type.lower():
                    thresholds[influence_type] *= social_factor
                
                # Add some randomness
                thresholds[influence_type] *= random.uniform(0.8, 1.2)
                thresholds[influence_type] = max(0.01, min(0.99, thresholds[influence_type]))
        
        self.agent_thresholds[agent_id] = thresholds
        self.agent_exposures[agent_id] = defaultdict(int)
        self.agent_influence_scores[agent_id] = defaultdict(float)
        
        logger.debug(f"Initialized influence thresholds for agent {agent_id} ({agent_type})")
    
    def calculate_influence_probability(self, 
                                      source_agent: str,
                                      target_agent: str,
                                      influence_type: InfluenceType,
                                      content_data: Dict[str, Any] = None,
                                      network_data: Dict[str, Any] = None) -> float:
        """Calculate probability of successful influence.
        
        Args:
            source_agent: Source agent
            target_agent: Target agent
            influence_type: Type of influence
            content_data: Content-related data
            network_data: Network-related data
            
        Returns:
            Influence probability (0-1)
        """
        # Get base influence weight between agents
        base_weight = self.influence_weights.get((source_agent, target_agent), self.base_influence_probability)
        
        # Get target's threshold for this influence type
        target_threshold = self.agent_thresholds.get(target_agent, {}).get(
            influence_type.value, 0.5
        )
        
        # Get target's current exposure
        exposure_count = self.agent_exposures.get(target_agent, {}).get(
            influence_type.value, 0
        )
        
        # Calculate cumulative influence (for threshold models)
        cumulative_influence = base_weight
        
        # Add network effects
        if network_data:
            # Network distance effect
            distance = network_data.get('distance', 1)
            distance_factor = math.exp(-0.1 * (distance - 1))
            cumulative_influence *= distance_factor
            
            # Common neighbors effect
            common_neighbors = network_data.get('common_neighbors', 0)
            neighbor_boost = 1.0 + 0.1 * common_neighbors
            cumulative_influence *= neighbor_boost
            
            # Network centrality of source
            source_centrality = network_data.get('source_centrality', 0.5)
            centrality_boost = 1.0 + 0.2 * source_centrality
            cumulative_influence *= centrality_boost
        
        # Add content effects
        if content_data:
            # Content quality/credibility
            credibility = content_data.get('credibility', 0.5)
            credibility_factor = 0.5 + 0.5 * credibility
            cumulative_influence *= credibility_factor
            
            # Emotional impact
            emotional_impact = content_data.get('emotional_impact', 0.5)
            emotion_factor = 1.0 + 0.3 * emotional_impact
            cumulative_influence *= emotion_factor
            
            # Novelty
            novelty = content_data.get('novelty', 0.5)
            novelty_factor = 1.0 + 0.2 * novelty
            cumulative_influence *= novelty_factor
        
        # Exposure effect (repeated exposure can increase or decrease influence)
        if exposure_count > 0:
            if influence_type in [InfluenceType.MISINFORMATION_SPREAD]:
                # Repeated exposure to misinformation can increase belief (illusory truth effect)
                exposure_factor = 1.0 + 0.1 * min(exposure_count, 5)
            else:
                # Generally, repeated exposure has diminishing returns
                exposure_factor = 1.0 / (1.0 + 0.2 * exposure_count)
            cumulative_influence *= exposure_factor
        
        # Calculate probability based on model type
        if influence_type in [InfluenceType.INFORMATION_SPREAD, InfluenceType.EMOTIONAL_CONTAGION]:
            # Independent cascade model
            probability = min(1.0, cumulative_influence)
        else:
            # Linear threshold model
            probability = 1.0 if cumulative_influence >= target_threshold else 0.0
        
        return max(0.0, min(1.0, probability))
    
    def attempt_influence(self, 
                         source_agent: str,
                         target_agent: str,
                         influence_type: InfluenceType,
                         content_id: str = None,
                         context: Dict[str, Any] = None) -> InfluenceEvent:
        """Attempt to influence a target agent.
        
        Args:
            source_agent: Source agent
            target_agent: Target agent
            influence_type: Type of influence
            content_id: Associated content
            context: Additional context
            
        Returns:
            Influence event record
        """
        # Calculate influence probability
        probability = self.calculate_influence_probability(
            source_agent, target_agent, influence_type,
            context.get('content_data', {}) if context else {},
            context.get('network_data', {}) if context else {}
        )
        
        # Determine success
        success = random.random() < probability
        
        # Create influence event
        event = InfluenceEvent(
            influence_type=influence_type,
            source_agent=source_agent,
            target_agent=target_agent,
            strength=probability,
            content_id=content_id,
            context=context or {},
            success=success
        )
        
        # Update agent states
        if target_agent not in self.agent_exposures:
            self.agent_exposures[target_agent] = defaultdict(int)
        if target_agent not in self.agent_influence_scores:
            self.agent_influence_scores[target_agent] = defaultdict(float)
        
        self.agent_exposures[target_agent][influence_type.value] += 1
        
        if success:
            self.agent_influence_scores[target_agent][influence_type.value] += probability
        
        # Record event
        self.influence_history.append(event)
        
        # Keep only recent history
        if len(self.influence_history) > 10000:
            self.influence_history = self.influence_history[-5000:]
        
        logger.debug(f"Influence attempt: {source_agent} -> {target_agent} ({influence_type.value}): {'SUCCESS' if success else 'FAILED'} (p={probability:.3f})")
        
        return event
    
    def start_cascade(self, 
                     origin_agent: str,
                     influence_type: InfluenceType,
                     model: CascadeModel = CascadeModel.INDEPENDENT_CASCADE,
                     content_id: str = None,
                     initial_targets: List[str] = None) -> InfluenceCascade:
        """Start an influence cascade.
        
        Args:
            origin_agent: Origin agent
            influence_type: Type of influence
            model: Cascade model
            content_id: Associated content
            initial_targets: Initial target agents
            
        Returns:
            Created cascade
        """
        cascade = InfluenceCascade(
            cascade_type=influence_type,
            model=model,
            origin_agent=origin_agent,
            origin_content=content_id
        )
        
        # Initialize with origin agent
        cascade.affected_agents.add(origin_agent)
        cascade.generation_map[origin_agent] = 0
        
        # Attempt initial influences
        if initial_targets:
            for target in initial_targets:
                event = self.attempt_influence(
                    origin_agent, target, influence_type, content_id
                )
                cascade.add_influence_event(event)
        
        self.active_cascades[cascade.cascade_id] = cascade
        
        logger.info(f"Started {influence_type.value} cascade from {origin_agent} (model: {model.value})")
        
        return cascade
    
    def propagate_cascade(self, 
                         cascade_id: str,
                         network_graph: nx.Graph,
                         max_generations: int = 10,
                         max_agents_per_generation: int = 100) -> Dict[str, Any]:
        """Propagate an influence cascade through the network.
        
        Args:
            cascade_id: Cascade identifier
            network_graph: Network graph
            max_generations: Maximum generations to propagate
            max_agents_per_generation: Maximum agents to influence per generation
            
        Returns:
            Propagation results
        """
        cascade = self.active_cascades.get(cascade_id)
        if not cascade or not cascade.is_active:
            return {}
        
        results = {
            'generations_propagated': 0,
            'new_agents_influenced': 0,
            'total_attempts': 0,
            'success_rate': 0.0
        }
        
        current_generation = cascade.max_generation
        
        for generation in range(current_generation + 1, current_generation + max_generations + 1):
            # Get agents from previous generation
            previous_gen_agents = [
                agent for agent, gen in cascade.generation_map.items()
                if gen == generation - 1
            ]
            
            if not previous_gen_agents:
                break
            
            generation_attempts = 0
            generation_successes = 0
            new_agents_this_gen = set()
            
            # For each agent in previous generation, try to influence neighbors
            for source_agent in previous_gen_agents:
                if source_agent not in network_graph:
                    continue
                
                # Get neighbors
                neighbors = list(network_graph.neighbors(source_agent))
                
                # Filter out already influenced agents
                potential_targets = [
                    neighbor for neighbor in neighbors
                    if neighbor not in cascade.affected_agents
                ]
                
                # Limit targets per agent
                if len(potential_targets) > max_agents_per_generation // len(previous_gen_agents):
                    potential_targets = random.sample(
                        potential_targets,
                        max_agents_per_generation // len(previous_gen_agents)
                    )
                
                # Attempt influence on each target
                for target_agent in potential_targets:
                    # Get network data for influence calculation
                    network_data = {
                        'distance': 1,  # Direct neighbors
                        'common_neighbors': len(set(network_graph.neighbors(source_agent)) & 
                                              set(network_graph.neighbors(target_agent))),
                        'source_centrality': nx.degree_centrality(network_graph).get(source_agent, 0.0)
                    }
                    
                    context = {
                        'network_data': network_data,
                        'generation': generation,
                        'cascade_id': cascade_id
                    }
                    
                    event = self.attempt_influence(
                        source_agent, target_agent, cascade.cascade_type,
                        cascade.origin_content, context
                    )
                    
                    cascade.add_influence_event(event)
                    generation_attempts += 1
                    
                    if event.success:
                        generation_successes += 1
                        new_agents_this_gen.add(target_agent)
                    
                    # Check if we've reached the limit
                    if len(new_agents_this_gen) >= max_agents_per_generation:
                        break
                
                if len(new_agents_this_gen) >= max_agents_per_generation:
                    break
            
            # Update results
            results['generations_propagated'] += 1
            results['new_agents_influenced'] += len(new_agents_this_gen)
            results['total_attempts'] += generation_attempts
            
            # If no new agents influenced, cascade dies out
            if len(new_agents_this_gen) == 0:
                cascade.is_active = False
                cascade.end_time = datetime.now()
                break
        
        # Calculate overall success rate
        if results['total_attempts'] > 0:
            results['success_rate'] = results['new_agents_influenced'] / results['total_attempts']
        
        # Update cascade metrics
        cascade_metrics = cascade.calculate_cascade_metrics()
        results.update(cascade_metrics)
        
        # Move to completed if inactive
        if not cascade.is_active:
            self.completed_cascades.append(cascade)
            del self.active_cascades[cascade_id]
            
            logger.info(f"Cascade {cascade_id} completed: {cascade.total_reach} agents reached in {cascade.max_generation} generations")
        
        return results
    
    def update_influence_weights(self, 
                               source_agent: str,
                               target_agent: str,
                               interaction_outcome: float,
                               learning_rate: float = 0.1) -> None:
        """Update influence weights based on interaction outcomes.
        
        Args:
            source_agent: Source agent
            target_agent: Target agent
            interaction_outcome: Outcome of interaction (-1 to 1)
            learning_rate: Learning rate for weight updates
        """
        current_weight = self.influence_weights.get(
            (source_agent, target_agent), self.base_influence_probability
        )
        
        # Update weight based on outcome
        weight_change = learning_rate * interaction_outcome * (1.0 - current_weight)
        new_weight = current_weight + weight_change
        
        # Clamp to valid range
        new_weight = max(0.01, min(0.99, new_weight))
        
        self.influence_weights[(source_agent, target_agent)] = new_weight
        
        logger.debug(f"Updated influence weight {source_agent} -> {target_agent}: {current_weight:.3f} -> {new_weight:.3f}")
    
    def get_agent_influence_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get influence profile for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Influence profile
        """
        # Incoming influence (how much agent is influenced)
        incoming_events = [
            event for event in self.influence_history[-1000:]
            if event.target_agent == agent_id
        ]
        
        # Outgoing influence (how much agent influences others)
        outgoing_events = [
            event for event in self.influence_history[-1000:]
            if event.source_agent == agent_id
        ]
        
        # Calculate influence metrics
        incoming_success_rate = (
            sum(1 for event in incoming_events if event.success) / len(incoming_events)
            if incoming_events else 0.0
        )
        
        outgoing_success_rate = (
            sum(1 for event in outgoing_events if event.success) / len(outgoing_events)
            if outgoing_events else 0.0
        )
        
        # Influence by type
        influence_by_type = {}
        for influence_type in InfluenceType:
            type_incoming = [e for e in incoming_events if e.influence_type == influence_type]
            type_outgoing = [e for e in outgoing_events if e.influence_type == influence_type]
            
            influence_by_type[influence_type.value] = {
                'incoming_count': len(type_incoming),
                'outgoing_count': len(type_outgoing),
                'incoming_success_rate': (
                    sum(1 for e in type_incoming if e.success) / len(type_incoming)
                    if type_incoming else 0.0
                ),
                'outgoing_success_rate': (
                    sum(1 for e in type_outgoing if e.success) / len(type_outgoing)
                    if type_outgoing else 0.0
                )
            }
        
        # Cascade participation
        cascade_participation = {
            'originated_cascades': len([
                cascade for cascade in self.completed_cascades + list(self.active_cascades.values())
                if cascade.origin_agent == agent_id
            ]),
            'participated_cascades': len([
                cascade for cascade in self.completed_cascades + list(self.active_cascades.values())
                if agent_id in cascade.affected_agents
            ])
        }
        
        profile = {
            'agent_id': agent_id,
            'thresholds': self.agent_thresholds.get(agent_id, {}),
            'exposures': dict(self.agent_exposures.get(agent_id, {})),
            'influence_scores': dict(self.agent_influence_scores.get(agent_id, {})),
            'incoming_influence': {
                'total_events': len(incoming_events),
                'success_rate': incoming_success_rate,
                'average_strength': np.mean([e.strength for e in incoming_events]) if incoming_events else 0.0
            },
            'outgoing_influence': {
                'total_events': len(outgoing_events),
                'success_rate': outgoing_success_rate,
                'average_strength': np.mean([e.strength for e in outgoing_events]) if outgoing_events else 0.0
            },
            'influence_by_type': influence_by_type,
            'cascade_participation': cascade_participation
        }
        
        return profile
    
    def get_influence_statistics(self) -> Dict[str, Any]:
        """Get overall influence statistics.
        
        Returns:
            Influence statistics
        """
        stats = {
            'total_influence_events': len(self.influence_history),
            'active_cascades': len(self.active_cascades),
            'completed_cascades': len(self.completed_cascades),
            'influence_types': {},
            'cascade_models': {},
            'success_rates': {},
            'network_effects': {}
        }
        
        # Influence type distribution
        for influence_type in InfluenceType:
            type_events = [e for e in self.influence_history if e.influence_type == influence_type]
            stats['influence_types'][influence_type.value] = {
                'count': len(type_events),
                'success_rate': sum(1 for e in type_events if e.success) / len(type_events) if type_events else 0.0,
                'average_strength': np.mean([e.strength for e in type_events]) if type_events else 0.0
            }
        
        # Cascade model performance
        all_cascades = self.completed_cascades + list(self.active_cascades.values())
        for model in CascadeModel:
            model_cascades = [c for c in all_cascades if c.model == model]
            if model_cascades:
                stats['cascade_models'][model.value] = {
                    'count': len(model_cascades),
                    'average_reach': np.mean([c.total_reach for c in model_cascades]),
                    'average_generations': np.mean([c.max_generation for c in model_cascades]),
                    'total_reach': sum(c.total_reach for c in model_cascades)
                }
        
        # Overall success rates
        if self.influence_history:
            stats['success_rates']['overall'] = sum(1 for e in self.influence_history if e.success) / len(self.influence_history)
            stats['success_rates']['recent'] = sum(1 for e in self.influence_history[-1000:] if e.success) / min(1000, len(self.influence_history))
        
        # Network effects
        if self.influence_weights:
            stats['network_effects'] = {
                'total_connections': len(self.influence_weights),
                'average_weight': np.mean(list(self.influence_weights.values())),
                'weight_distribution': {
                    'min': np.min(list(self.influence_weights.values())),
                    'max': np.max(list(self.influence_weights.values())),
                    'std': np.std(list(self.influence_weights.values()))
                }
            }
        
        return stats