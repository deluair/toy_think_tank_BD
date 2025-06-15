"""Base agent class for NewsGuard Bangladesh simulation.

This module defines the base agent class and common interfaces that all agents
in the simulation inherit from.
"""

import uuid
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import mesa
import numpy as np

from ..utils.logger import get_logger
from ..utils.helpers import generate_id, calculate_statistics

logger = get_logger(__name__)


class AgentType(Enum):
    """Types of agents in the simulation."""
    NEWS_OUTLET = "news_outlet"
    READER = "reader"
    PLATFORM = "platform"
    FACT_CHECKER = "fact_checker"
    BOT = "bot"
    INFLUENCER = "influencer"
    JOURNALIST = "journalist"
    REGULATOR = "regulator"


class AgentState(Enum):
    """Agent states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"
    DORMANT = "dormant"


class AgentCapability(Enum):
    """Agent capabilities."""
    CONTENT_CREATION = "content_creation"
    CONTENT_SHARING = "content_sharing"
    CONTENT_CONSUMPTION = "content_consumption"
    FACT_CHECKING = "fact_checking"
    INFLUENCE_PROPAGATION = "influence_propagation"
    NETWORK_ANALYSIS = "network_analysis"
    ECONOMIC_ACTIVITY = "economic_activity"
    POLICY_ENFORCEMENT = "policy_enforcement"


@dataclass
class AgentMetrics:
    """Metrics tracked for each agent."""
    # Activity metrics
    total_actions: int = 0
    actions_per_day: float = 0.0
    last_activity: Optional[datetime] = None
    
    # Content metrics
    content_created: int = 0
    content_shared: int = 0
    content_consumed: int = 0
    
    # Interaction metrics
    interactions_sent: int = 0
    interactions_received: int = 0
    network_connections: int = 0
    
    # Influence metrics
    influence_score: float = 0.0
    trust_score: float = 0.5
    credibility_score: float = 0.5
    
    # Economic metrics
    revenue_generated: float = 0.0
    costs_incurred: float = 0.0
    economic_impact: float = 0.0
    
    # Performance metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    response_time: float = 0.0
    
    def update_activity(self, action_type: str = None) -> None:
        """Update activity metrics.
        
        Args:
            action_type: Type of action performed
        """
        self.total_actions += 1
        self.last_activity = datetime.now()
        
        # Update specific metrics based on action type
        if action_type == 'content_creation':
            self.content_created += 1
        elif action_type == 'content_sharing':
            self.content_shared += 1
        elif action_type == 'content_consumption':
            self.content_consumed += 1
        elif action_type == 'interaction_sent':
            self.interactions_sent += 1
        elif action_type == 'interaction_received':
            self.interactions_received += 1
    
    def calculate_daily_activity(self, days: int = 7) -> float:
        """Calculate average daily activity.
        
        Args:
            days: Number of days to average over
            
        Returns:
            Average actions per day
        """
        if not self.last_activity:
            return 0.0
        
        days_since_start = max(1, days)
        self.actions_per_day = self.total_actions / days_since_start
        return self.actions_per_day


@dataclass
class AgentMemory:
    """Agent memory for storing experiences and learning."""
    # Recent experiences
    recent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    recent_content: List[str] = field(default_factory=list)
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Long-term memory
    trusted_agents: Set[str] = field(default_factory=set)
    distrusted_agents: Set[str] = field(default_factory=set)
    preferred_sources: Set[str] = field(default_factory=set)
    
    # Learning data
    success_patterns: Dict[str, float] = field(default_factory=dict)
    failure_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Limits
    max_recent_items: int = 100
    
    def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add interaction to memory.
        
        Args:
            interaction: Interaction data
        """
        self.recent_interactions.append(interaction)
        if len(self.recent_interactions) > self.max_recent_items:
            self.recent_interactions.pop(0)
    
    def add_content(self, content_id: str) -> None:
        """Add content to memory.
        
        Args:
            content_id: Content identifier
        """
        self.recent_content.append(content_id)
        if len(self.recent_content) > self.max_recent_items:
            self.recent_content.pop(0)
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        """Add decision to memory.
        
        Args:
            decision: Decision data
        """
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > self.max_recent_items:
            self.recent_decisions.pop(0)
    
    def update_trust(self, agent_id: str, trust_change: float) -> None:
        """Update trust for an agent.
        
        Args:
            agent_id: Agent identifier
            trust_change: Change in trust (-1 to 1)
        """
        if trust_change > 0.5:
            self.trusted_agents.add(agent_id)
            self.distrusted_agents.discard(agent_id)
        elif trust_change < -0.5:
            self.distrusted_agents.add(agent_id)
            self.trusted_agents.discard(agent_id)
    
    def learn_from_outcome(self, pattern: str, success: bool, weight: float = 1.0) -> None:
        """Learn from decision outcomes.
        
        Args:
            pattern: Pattern identifier
            success: Whether the outcome was successful
            weight: Learning weight
        """
        if success:
            self.success_patterns[pattern] = self.success_patterns.get(pattern, 0.0) + weight
        else:
            self.failure_patterns[pattern] = self.failure_patterns.get(pattern, 0.0) + weight


class BaseAgent(mesa.Agent, ABC):
    """Base class for all agents in the NewsGuard simulation."""
    
    def __init__(self, 
                 unique_id: str,
                 model,
                 agent_type: AgentType,
                 config: Dict[str, Any] = None):
        """Initialize base agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: Mesa model instance
            agent_type: Type of agent
            config: Agent configuration
        """
        super().__init__(unique_id, model)
        
        # Basic properties
        self.agent_id = unique_id
        self.agent_type = agent_type
        self.config = config or {}
        
        # State
        self.state = AgentState.ACTIVE
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # Capabilities
        self.capabilities: Set[AgentCapability] = set()
        
        # Metrics and memory
        self.metrics = AgentMetrics()
        self.memory = AgentMemory()
        
        # Network properties
        self.network_position: Dict[str, float] = {}
        self.connections: Set[str] = set()
        
        # Economic properties
        self.economic_state: Dict[str, float] = {
            'revenue': 0.0,
            'costs': 0.0,
            'budget': self.config.get('initial_budget', 1000.0)
        }
        
        # Behavioral properties
        self.personality_traits: Dict[str, float] = self._initialize_personality()
        self.behavioral_patterns: Dict[str, Any] = {}
        
        # Initialize agent-specific properties
        self._initialize_agent_specific()
        
        logger.debug(f"Initialized {agent_type.value} agent: {unique_id}")
    
    def _initialize_personality(self) -> Dict[str, float]:
        """Initialize personality traits.
        
        Returns:
            Personality traits dictionary
        """
        # Base personality traits (0-1 scale)
        traits = {
            'openness': random.uniform(0.3, 0.7),
            'conscientiousness': random.uniform(0.3, 0.7),
            'extraversion': random.uniform(0.3, 0.7),
            'agreeableness': random.uniform(0.3, 0.7),
            'neuroticism': random.uniform(0.2, 0.6),
            'skepticism': random.uniform(0.3, 0.8),
            'risk_tolerance': random.uniform(0.2, 0.8),
            'social_influence_susceptibility': random.uniform(0.2, 0.8)
        }
        
        # Apply config overrides
        personality_config = self.config.get('personality', {})
        traits.update(personality_config)
        
        return traits
    
    @abstractmethod
    def _initialize_agent_specific(self) -> None:
        """Initialize agent-specific properties.
        
        This method should be implemented by each agent subclass.
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """Execute one step of agent behavior.
        
        This method should be implemented by each agent subclass.
        """
        pass
    
    def update_state(self, new_state: AgentState, reason: str = None) -> None:
        """Update agent state.
        
        Args:
            new_state: New agent state
            reason: Reason for state change
        """
        old_state = self.state
        self.state = new_state
        self.last_updated = datetime.now()
        
        # Log state change
        logger.info(f"Agent {self.agent_id} state changed: {old_state.value} -> {new_state.value}" + 
                   (f" (reason: {reason})" if reason else ""))
        
        # Record in memory
        self.memory.add_decision({
            'type': 'state_change',
            'old_state': old_state.value,
            'new_state': new_state.value,
            'reason': reason,
            'timestamp': datetime.now()
        })
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add capability to agent.
        
        Args:
            capability: Capability to add
        """
        self.capabilities.add(capability)
        logger.debug(f"Added capability {capability.value} to agent {self.agent_id}")
    
    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove capability from agent.
        
        Args:
            capability: Capability to remove
        """
        self.capabilities.discard(capability)
        logger.debug(f"Removed capability {capability.value} from agent {self.agent_id}")
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a capability.
        
        Args:
            capability: Capability to check
            
        Returns:
            True if agent has the capability
        """
        return capability in self.capabilities
    
    def connect_to_agent(self, other_agent_id: str, connection_strength: float = 1.0) -> None:
        """Connect to another agent.
        
        Args:
            other_agent_id: ID of agent to connect to
            connection_strength: Strength of connection (0-1)
        """
        self.connections.add(other_agent_id)
        self.metrics.network_connections = len(self.connections)
        
        # Record interaction
        self.memory.add_interaction({
            'type': 'connection_formed',
            'target_agent': other_agent_id,
            'strength': connection_strength,
            'timestamp': datetime.now()
        })
        
        logger.debug(f"Agent {self.agent_id} connected to {other_agent_id}")
    
    def disconnect_from_agent(self, other_agent_id: str, reason: str = None) -> None:
        """Disconnect from another agent.
        
        Args:
            other_agent_id: ID of agent to disconnect from
            reason: Reason for disconnection
        """
        self.connections.discard(other_agent_id)
        self.metrics.network_connections = len(self.connections)
        
        # Record interaction
        self.memory.add_interaction({
            'type': 'connection_broken',
            'target_agent': other_agent_id,
            'reason': reason,
            'timestamp': datetime.now()
        })
        
        logger.debug(f"Agent {self.agent_id} disconnected from {other_agent_id}" + 
                    (f" (reason: {reason})" if reason else ""))
    
    def update_trust(self, other_agent_id: str, trust_change: float) -> None:
        """Update trust towards another agent.
        
        Args:
            other_agent_id: ID of other agent
            trust_change: Change in trust (-1 to 1)
        """
        self.memory.update_trust(other_agent_id, trust_change)
        
        # Record interaction
        self.memory.add_interaction({
            'type': 'trust_update',
            'target_agent': other_agent_id,
            'trust_change': trust_change,
            'timestamp': datetime.now()
        })
    
    def make_decision(self, 
                     decision_type: str,
                     options: List[Any],
                     context: Dict[str, Any] = None) -> Any:
        """Make a decision based on agent characteristics.
        
        Args:
            decision_type: Type of decision
            options: Available options
            context: Decision context
            
        Returns:
            Selected option
        """
        if not options:
            return None
        
        # Simple decision making based on personality and memory
        # This can be overridden by subclasses for more sophisticated decision making
        
        # Check for learned patterns
        pattern_key = f"{decision_type}_{len(options)}"
        success_score = self.memory.success_patterns.get(pattern_key, 0.0)
        failure_score = self.memory.failure_patterns.get(pattern_key, 0.0)
        
        # Risk tolerance affects decision
        risk_tolerance = self.personality_traits.get('risk_tolerance', 0.5)
        
        if success_score > failure_score and risk_tolerance > 0.5:
            # Choose based on past success
            choice = random.choice(options)
        elif risk_tolerance < 0.3:
            # Conservative choice (first option)
            choice = options[0]
        else:
            # Random choice
            choice = random.choice(options)
        
        # Record decision
        decision_record = {
            'type': decision_type,
            'options_count': len(options),
            'choice': str(choice),
            'context': context or {},
            'timestamp': datetime.now()
        }
        self.memory.add_decision(decision_record)
        
        return choice
    
    def learn_from_outcome(self, decision_type: str, success: bool, context: Dict[str, Any] = None) -> None:
        """Learn from decision outcomes.
        
        Args:
            decision_type: Type of decision
            success: Whether the outcome was successful
            context: Additional context
        """
        # Find recent decision of this type
        recent_decisions = [
            d for d in self.memory.recent_decisions[-10:]
            if d.get('type') == decision_type
        ]
        
        if recent_decisions:
            latest_decision = recent_decisions[-1]
            pattern_key = f"{decision_type}_{latest_decision.get('options_count', 1)}"
            
            # Learn from outcome
            learning_weight = 1.0
            if context and 'importance' in context:
                learning_weight = context['importance']
            
            self.memory.learn_from_outcome(pattern_key, success, learning_weight)
            
            # Update success rate
            if success:
                self.metrics.success_rate = (self.metrics.success_rate * 0.9) + (1.0 * 0.1)
            else:
                self.metrics.error_rate = (self.metrics.error_rate * 0.9) + (1.0 * 0.1)
    
    def update_economic_state(self, revenue_change: float = 0.0, cost_change: float = 0.0) -> None:
        """Update economic state.
        
        Args:
            revenue_change: Change in revenue
            cost_change: Change in costs
        """
        self.economic_state['revenue'] += revenue_change
        self.economic_state['costs'] += cost_change
        self.economic_state['budget'] += revenue_change - cost_change
        
        # Update metrics
        self.metrics.revenue_generated = self.economic_state['revenue']
        self.metrics.costs_incurred = self.economic_state['costs']
        self.metrics.economic_impact = self.economic_state['revenue'] - self.economic_state['costs']
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of agent state.
        
        Returns:
            Agent summary
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'capabilities': [cap.value for cap in self.capabilities],
            'metrics': {
                'total_actions': self.metrics.total_actions,
                'network_connections': self.metrics.network_connections,
                'influence_score': self.metrics.influence_score,
                'trust_score': self.metrics.trust_score,
                'credibility_score': self.metrics.credibility_score,
                'success_rate': self.metrics.success_rate
            },
            'economic_state': self.economic_state.copy(),
            'personality_traits': self.personality_traits.copy()
        }
    
    def is_active(self) -> bool:
        """Check if agent is active.
        
        Returns:
            True if agent is active
        """
        return self.state == AgentState.ACTIVE
    
    def can_perform_action(self, action_type: str) -> bool:
        """Check if agent can perform an action.
        
        Args:
            action_type: Type of action
            
        Returns:
            True if agent can perform the action
        """
        # Check if agent is active
        if not self.is_active():
            return False
        
        # Check economic constraints
        if self.economic_state['budget'] <= 0:
            return False
        
        # Check capability requirements
        capability_map = {
            'create_content': AgentCapability.CONTENT_CREATION,
            'share_content': AgentCapability.CONTENT_SHARING,
            'consume_content': AgentCapability.CONTENT_CONSUMPTION,
            'fact_check': AgentCapability.FACT_CHECKING,
            'influence': AgentCapability.INFLUENCE_PROPAGATION,
            'analyze_network': AgentCapability.NETWORK_ANALYSIS,
            'economic_activity': AgentCapability.ECONOMIC_ACTIVITY,
            'enforce_policy': AgentCapability.POLICY_ENFORCEMENT
        }
        
        required_capability = capability_map.get(action_type)
        if required_capability and not self.has_capability(required_capability):
            return False
        
        return True
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate agent configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        # Basic validation - can be overridden by subclasses
        required_fields = ['initial_budget']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate numeric fields
        if config.get('initial_budget', 0) < 0:
            return False
        
        return True