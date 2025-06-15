"""Agent modules for NewsGuard Bangladesh simulation.

This module contains all agent types that participate in the information ecosystem,
including news outlets, readers, social media platforms, fact-checkers, and bots.
"""

from .base import BaseAgent, AgentType, AgentState, AgentCapability
from .news_outlet import NewsOutlet, OutletType, EditorialStance

# Agent registry for dynamic creation
AGENT_REGISTRY = {
    AgentType.NEWS_OUTLET: NewsOutlet,
}

# Agent categories for analysis
AGENT_CATEGORIES = {
    'content_creators': [AgentType.NEWS_OUTLET, AgentType.JOURNALIST, AgentType.INFLUENCER],
    'content_consumers': [AgentType.READER],
    'content_distributors': [AgentType.PLATFORM],
    'content_validators': [AgentType.FACT_CHECKER],
    'automated_agents': [AgentType.BOT],
    'governance_agents': [AgentType.REGULATOR]
}

# Default agent configurations
DEFAULT_AGENT_CONFIGS = {
    AgentType.NEWS_OUTLET: {
        'credibility_score': 0.7,
        'publication_frequency': 5,  # articles per day
        'audience_size': 10000,
        'revenue_model': 'advertising'
    },
    AgentType.READER: {
        'media_literacy': 0.5,
        'engagement_frequency': 3,  # interactions per day
        'trust_threshold': 0.6,
        'sharing_propensity': 0.3
    },
    AgentType.PLATFORM: {
        'user_base': 100000,
        'algorithm_transparency': 0.4,
        'content_moderation_strength': 0.6,
        'monetization_pressure': 0.7
    },
    AgentType.FACT_CHECKER: {
        'accuracy_rate': 0.9,
        'response_time': 24,  # hours
        'coverage_capacity': 50,  # articles per day
        'independence_score': 0.8
    },
    AgentType.BOT: {
        'activity_level': 0.8,
        'sophistication': 0.5,
        'detection_evasion': 0.6,
        'network_size': 1000
    },
    AgentType.INFLUENCER: {
        'follower_count': 50000,
        'engagement_rate': 0.05,
        'authenticity_score': 0.7,
        'content_frequency': 2  # posts per day
    },
    AgentType.JOURNALIST: {
        'experience_years': 5,
        'source_network_size': 100,
        'fact_checking_rigor': 0.8,
        'deadline_pressure': 0.6
    },
    AgentType.REGULATOR: {
        'enforcement_capacity': 0.6,
        'policy_effectiveness': 0.5,
        'response_speed': 0.4,
        'transparency_level': 0.7
    }
}


def create_agent(agent_type: AgentType, agent_id: str, config: dict = None, **kwargs):
    """Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent to create
        agent_id: Unique identifier for the agent
        config: Agent configuration
        **kwargs: Additional arguments
        
    Returns:
        Created agent instance
        
    Raises:
        ValueError: If agent type is not supported
    """
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Merge default config with provided config
    default_config = DEFAULT_AGENT_CONFIGS.get(agent_type, {}).copy()
    if config:
        default_config.update(config)
    
    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(agent_id=agent_id, config=default_config, **kwargs)


def get_agent_types_by_category(category: str) -> list:
    """Get agent types in a specific category.
    
    Args:
        category: Agent category
        
    Returns:
        List of agent types in the category
    """
    return AGENT_CATEGORIES.get(category, [])


def validate_agent_config(agent_type: AgentType, config: dict) -> bool:
    """Validate agent configuration.
    
    Args:
        agent_type: Type of agent
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
    """
    if agent_type not in AGENT_REGISTRY:
        return False
    
    # Get agent class and check if it has a validate_config method
    agent_class = AGENT_REGISTRY[agent_type]
    if hasattr(agent_class, 'validate_config'):
        return agent_class.validate_config(config)
    
    return True


__all__ = [
    # Base classes
    'BaseAgent', 'AgentType', 'AgentState', 'AgentCapability',
    
    # Agent classes
    'NewsOutlet',
    
    # Enums and types
    'OutletType', 'EditorialStance',
    
    # Registry and utilities
    'AGENT_REGISTRY', 'AGENT_CATEGORIES', 'DEFAULT_AGENT_CONFIGS',
    'create_agent', 'get_agent_types_by_category', 'validate_agent_config'
]