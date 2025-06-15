"""NewsGuard Bangladesh: Agent-Based Simulation of Digital News Ecosystem.

A comprehensive simulation framework for modeling Bangladesh's digital news landscape,
focusing on misinformation dynamics, cross-border information warfare, and media sustainability.
"""

__version__ = "0.1.0"
__author__ = "NewsGuard Research Team"
__email__ = "research@newsguard.com"
__license__ = "MIT"

# Core imports
from .core.simulation import SimulationEngine
from .agents import (
    BaseAgent,
    NewsOutlet,
)
from .models import (
    ContentModel,
    NetworkModel,
    TrustModel,
    EconomicModel,
    BehaviorModel,
    InfluenceModel,
)

# Configuration and utilities
from .utils.config import Config
from .utils.logger import get_logger

__all__ = [
    # Core classes
    "SimulationEngine",
    "BaseAgent",
    "NewsOutlet",
    # Models
    "ContentModel",
    "NetworkModel",
    "TrustModel",
    "EconomicModel",
    "BehaviorModel",
    "InfluenceModel",
    # Utilities
    "Config",
    "get_logger",
]

# Package metadata
__package_info__ = {
    "name": "newsguard-bangladesh",
    "version": __version__,
    "description": "Agent-based simulation framework for Bangladesh's digital news ecosystem",
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "url": "https://github.com/newsguard/bangladesh-simulation",
    "keywords": [
        "simulation",
        "agent-based-modeling",
        "misinformation",
        "news-media",
        "bangladesh",
        "digital-journalism",
        "social-networks",
    ],
}

# Initialize logging
logger = get_logger(__name__)
logger.info(f"NewsGuard Bangladesh v{__version__} initialized")

# Validate dependencies
try:
    import mesa
    import networkx
    import pandas
    import numpy
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    raise

# Configuration validation
try:
    config = Config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.warning(f"Configuration loading failed: {e}")
    logger.info("Using default configuration")


def get_version():
    """Return the current version of NewsGuard Bangladesh."""
    return __version__


def get_package_info():
    """Return package metadata."""
    return __package_info__.copy()


def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        "mesa",
        "networkx",
        "pandas",
        "numpy",
        "scipy",
        "spacy",
        "transformers",
        "torch",
        "sklearn",
        "plotly",
        "dash",
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        return False
    
    logger.info("All required dependencies are available")
    return True


# Perform dependency check on import
if not check_dependencies():
    logger.warning("Some dependencies are missing. Please install them using: pip install -r requirements.txt")