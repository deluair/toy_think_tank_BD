"""Models package for NewsGuard Bangladesh simulation.

This package contains the core models that represent different aspects
of the information ecosystem in Bangladesh:

- ContentModel: Models news content, misinformation, and fact-checking
- NetworkModel: Models social, information, and economic networks
- TrustModel: Models trust relationships and credibility scoring
- EconomicModel: Models economic incentives and market dynamics
- BehaviorModel: Models agent behavior patterns and decision-making
- InfluenceModel: Models influence propagation and opinion dynamics
"""

from .content import (
    ContentModel,
    ContentType,
    ContentStatus,
    MisinformationModel,
    FactCheckModel,
    ContentMetadata,
    ContentAnalyzer
)

from .network import (
    NetworkModel,
    SocialNetworkModel,
)

from .trust import (
    TrustModel,
    TrustScore,
    ReputationModel,
)

from .economic import (
    EconomicModel,
)

from .behavior import (
    BehaviorModel,
    BehaviorPattern,
)

from .influence import (
    InfluenceModel,
    CascadeModel,
)

# Model registry for dynamic model loading
MODEL_REGISTRY = {
    'content': ContentModel,
    'network': NetworkModel,
    'trust': TrustModel,
    'economic': EconomicModel,
    'behavior': BehaviorModel,
    'influence': InfluenceModel,
    
    # Specialized models
    'misinformation': MisinformationModel,
    'fact_check': FactCheckModel,
    'social_network': SocialNetworkModel,
    'reputation': ReputationModel,
    'cascade': CascadeModel
}

# Model categories for organization
MODEL_CATEGORIES = {
    'content': ['content', 'misinformation', 'fact_check'],
    'network': ['network', 'social_network', 'information_network', 'economic_network'],
    'trust': ['trust', 'credibility', 'reputation'],
    'economic': ['economic', 'market', 'revenue', 'cost', 'incentive'],
    'behavior': ['behavior', 'decision', 'learning', 'adaptation'],
    'influence': ['influence', 'opinion', 'propagation', 'cascade']
}

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    'content': {
        'credibility_threshold': 0.7,
        'viral_threshold': 1000,
        'fact_check_delay': 24,  # hours
        'content_decay_rate': 0.1
    },
    'network': {
        'social_density': 0.1,
        'information_density': 0.05,
        'economic_density': 0.02,
        'rewiring_probability': 0.01
    },
    'trust': {
        'initial_trust': 0.5,
        'trust_decay_rate': 0.05,
        'trust_update_rate': 0.1,
        'credibility_weight': 0.7
    },
    'economic': {
        'ad_revenue_rate': 0.001,
        'subscription_price': 10.0,
        'content_cost': 5.0,
        'fact_check_cost': 20.0
    },
    'behavior': {
        'learning_rate': 0.1,
        'adaptation_rate': 0.05,
        'decision_threshold': 0.6,
        'memory_length': 100
    },
    'influence': {
        'influence_decay': 0.1,
        'opinion_change_rate': 0.05,
        'cascade_threshold': 0.3,
        'polarization_factor': 0.2
    }
}


def get_model(model_name: str, config: dict = None):
    """Get a model instance by name.
    
    Args:
        model_name: Name of the model
        config: Model configuration (uses default if None)
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model name is not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    # Get default config for model category
    model_config = {}
    for category, models in MODEL_CATEGORIES.items():
        if model_name in models:
            model_config = DEFAULT_MODEL_CONFIGS.get(category, {}).copy()
            break
    
    # Update with provided config
    if config:
        model_config.update(config)
    
    return model_class(config=model_config)


def get_models_by_category(category: str, config: dict = None):
    """Get all models in a category.
    
    Args:
        category: Model category
        config: Configuration for all models
        
    Returns:
        Dictionary of model instances
    """
    if category not in MODEL_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available categories: {list(MODEL_CATEGORIES.keys())}")
    
    models = {}
    for model_name in MODEL_CATEGORIES[category]:
        models[model_name] = get_model(model_name, config)
    
    return models


def list_available_models():
    """List all available models.
    
    Returns:
        Dictionary of models organized by category
    """
    return MODEL_CATEGORIES.copy()


def validate_model_config(model_name: str, config: dict):
    """Validate model configuration.
    
    Args:
        model_name: Name of the model
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if model_name not in MODEL_REGISTRY:
        return False, [f"Unknown model: {model_name}"]
    
    model_class = MODEL_REGISTRY[model_name]
    
    # Check if model has validation method
    if hasattr(model_class, 'validate_config'):
        return model_class.validate_config(config)
    
    # Basic validation - check for required parameters
    errors = []
    
    # Get default config for comparison
    default_config = {}
    for category, models in MODEL_CATEGORIES.items():
        if model_name in models:
            default_config = DEFAULT_MODEL_CONFIGS.get(category, {})
            break
    
    # Check for unknown parameters
    unknown_params = set(config.keys()) - set(default_config.keys())
    if unknown_params:
        errors.append(f"Unknown parameters: {unknown_params}")
    
    # Check parameter types and ranges
    for param, value in config.items():
        if param in default_config:
            default_value = default_config[param]
            if type(value) != type(default_value):
                errors.append(f"Parameter {param} should be {type(default_value).__name__}, got {type(value).__name__}")
            
            # Check ranges for numeric parameters
            if isinstance(value, (int, float)):
                if param.endswith('_rate') and not (0 <= value <= 1):
                    errors.append(f"Rate parameter {param} should be between 0 and 1, got {value}")
                elif param.endswith('_threshold') and not (0 <= value <= 1):
                    errors.append(f"Threshold parameter {param} should be between 0 and 1, got {value}")
                elif value < 0:
                    errors.append(f"Parameter {param} should be non-negative, got {value}")
    
    return len(errors) == 0, errors


# Package metadata
__version__ = "1.0.0"
__author__ = "NewsGuard Bangladesh Team"
__description__ = "Core models for NewsGuard Bangladesh simulation"

# Export all public components
__all__ = [
    # Core models
    'ContentModel', 'NetworkModel', 'TrustModel', 'EconomicModel', 
    'BehaviorModel', 'InfluenceModel',
    
    # Specialized models
    'MisinformationModel', 'FactCheckModel', 'SocialNetworkModel',
    'ReputationModel', 'CascadeModel',
    
    # Enums and data classes
    'ContentType', 'ContentStatus', 'ContentMetadata', 'TrustScore',
    'NetworkMetrics', 'TrustMetrics', 'EconomicMetrics', 'BehaviorMetrics',
    'InfluenceMetrics', 'BehaviorPattern', 'OpinionDynamics',
    
    # Analyzers
    'ContentAnalyzer',
    
    # Utility functions
    'get_model', 'get_models_by_category', 'list_available_models',
    'validate_model_config',
    
    # Constants
    'MODEL_REGISTRY', 'MODEL_CATEGORIES', 'DEFAULT_MODEL_CONFIGS'
]