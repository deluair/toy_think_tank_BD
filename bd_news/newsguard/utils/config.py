"""Configuration management for NewsGuard Bangladesh simulation."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig


@dataclass
class SimulationConfig:
    """Core simulation parameters."""
    duration_days: int = 365
    time_step_hours: int = 1
    random_seed: int = 42
    max_agents: int = 100000
    output_frequency: int = 24  # hours
    checkpoint_frequency: int = 168  # hours (weekly)
    parallel_processing: bool = True
    num_workers: int = 4


@dataclass
class AgentConfig:
    """Agent population configuration."""
    news_outlets: Dict[str, Any] = field(default_factory=lambda: {
        "count": 50,
        "major_outlets": 8,
        "regional_outlets": 15,
        "digital_only": 20,
        "tv_channels": 7,
        "credibility_distribution": "beta",
        "credibility_params": {"alpha": 2, "beta": 1}
    })
    
    readers: Dict[str, Any] = field(default_factory=lambda: {
        "count": 100000,
        "demographics_file": "data/demographics/bangladesh_2024.csv",
        "urban_rural_ratio": 0.38,  # 38% urban
        "literacy_rate": 0.75,
        "internet_penetration": 0.65,
        "smartphone_penetration": 0.58,
        "age_distribution": {
            "18-25": 0.25,
            "26-35": 0.30,
            "36-45": 0.20,
            "46-55": 0.15,
            "55+": 0.10
        }
    })
    
    platforms: Dict[str, Any] = field(default_factory=lambda: {
        "facebook": {
            "penetration": 0.93,
            "daily_active_users": 0.75,
            "algorithm_bias": 0.1,
            "moderation_effectiveness": 0.6,
            "fact_check_integration": True
        },
        "whatsapp": {
            "penetration": 0.87,
            "group_size_avg": 25,
            "group_size_std": 15,
            "forward_limit": 5,
            "encryption": True
        },
        "youtube": {
            "penetration": 0.82,
            "content_creation_rate": 0.05,
            "recommendation_strength": 0.8
        },
        "twitter": {
            "penetration": 0.15,
            "verification_rate": 0.02,
            "trending_influence": 0.7
        }
    })
    
    fact_checkers: Dict[str, Any] = field(default_factory=lambda: {
        "count": 5,
        "organizations": [
            {"name": "Rumor Scanner", "credibility": 0.9, "speed": 0.7, "reach": 0.6},
            {"name": "AFP Bangladesh", "credibility": 0.95, "speed": 0.8, "reach": 0.4},
            {"name": "BD FactCheck", "credibility": 0.85, "speed": 0.6, "reach": 0.3}
        ],
        "detection_accuracy": 0.85,
        "response_time_hours": 6,
        "resource_constraints": 0.7
    })


@dataclass
class MisinformationConfig:
    """Misinformation and disinformation parameters."""
    base_rate: float = 0.05  # 5% of content is misinformation
    foreign_influence: float = 0.3  # 30% from external sources
    deepfake_probability: float = 0.02
    viral_threshold: float = 0.1
    decay_rate: float = 0.95  # Daily decay of misinformation impact
    
    categories: Dict[str, float] = field(default_factory=lambda: {
        "political": 0.40,
        "communal": 0.25,
        "economic": 0.15,
        "health": 0.10,
        "international": 0.10
    })
    
    sources: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "indian_outlets": {
            "count": 72,
            "activity_rate": 0.4,  # Reports per day
            "sophistication": 0.7,
            "target_topics": ["communal", "political", "international"]
        },
        "domestic_political": {
            "count": 20,
            "activity_rate": 0.6,
            "sophistication": 0.5,
            "target_topics": ["political", "economic"]
        },
        "economic_scammers": {
            "count": 50,
            "activity_rate": 0.3,
            "sophistication": 0.4,
            "target_topics": ["economic", "health"]
        }
    })


@dataclass
class EconomicConfig:
    """Economic model parameters."""
    ad_market_size_usd: int = 50_000_000
    subscription_willingness: float = 0.15
    cost_per_journalist_usd: int = 12_000
    cost_per_fact_checker_usd: int = 15_000
    
    revenue_models: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "traditional": {"print_ads": 0.3, "classified": 0.2, "circulation": 0.5},
        "digital_ads": {"display": 0.44, "video": 0.38, "native": 0.18},
        "subscriptions": {"monthly": 0.6, "annual": 0.3, "premium": 0.1},
        "sponsored_content": {"native_ads": 0.7, "branded_content": 0.3}
    })
    
    cost_structure: Dict[str, float] = field(default_factory=lambda: {
        "personnel": 0.60,
        "technology": 0.15,
        "operations": 0.15,
        "marketing": 0.10
    })


@dataclass
class NetworkConfig:
    """Social network structure parameters."""
    small_world_probability: float = 0.1
    preferential_attachment: bool = True
    homophily_strength: float = 0.7
    bridge_probability: float = 0.05  # Cross-community connections
    
    community_structure: Dict[str, Any] = field(default_factory=lambda: {
        "political_polarization": 0.6,
        "linguistic_clustering": 0.8,  # Bangla vs English
        "geographic_clustering": 0.7,  # Urban vs rural
        "age_clustering": 0.5
    })
    
    influence_patterns: Dict[str, float] = field(default_factory=lambda: {
        "peer_influence": 0.6,
        "authority_influence": 0.3,
        "celebrity_influence": 0.1
    })


@dataclass
class DatabaseConfig:
    """Database connection parameters."""
    postgresql: Dict[str, str] = field(default_factory=lambda: {
        "host": "localhost",
        "port": "5432",
        "database": "newsguard_bd",
        "username": "newsguard",
        "password": "password",
        "pool_size": "10"
    })
    
    mongodb: Dict[str, str] = field(default_factory=lambda: {
        "host": "localhost",
        "port": "27017",
        "database": "newsguard_content",
        "username": "",
        "password": ""
    })
    
    redis: Dict[str, str] = field(default_factory=lambda: {
        "host": "localhost",
        "port": "6379",
        "database": "0",
        "password": ""
    })


class Config:
    """Main configuration manager for NewsGuard Bangladesh simulation."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to custom configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        if config_dict is not None:
            self.config_path = None
            self._config = OmegaConf.create(config_dict)
        else:
            self.config_path = config_path or self._get_default_config_path()
            self._config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        package_dir = Path(__file__).parent.parent
        return str(package_dir / "config" / "default.yaml")
    
    def _load_config(self) -> DictConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            return OmegaConf.create(config_dict)
        else:
            # Create default configuration
            default_config = {
                "simulation": SimulationConfig().__dict__,
                "agents": AgentConfig().__dict__,
                "misinformation": MisinformationConfig().__dict__,
                "economics": EconomicConfig().__dict__,
                "network": NetworkConfig().__dict__,
                "database": DatabaseConfig().__dict__
            }
            return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self._config, key)
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        OmegaConf.set(self._config, key, value)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary.
        
        Args:
            config_dict: Dictionary of configuration updates
        """
        self._config = OmegaConf.merge(self._config, config_dict)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration (defaults to current config_path)
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(OmegaConf.to_yaml(self._config), f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def simulation(self) -> SimulationConfig:
        """Get simulation configuration."""
        sim_dict = self.get("simulation", {})
        return SimulationConfig(**sim_dict)
    
    @property
    def agents(self) -> AgentConfig:
        """Get agent configuration."""
        agent_dict = self.get("agents", {})
        return AgentConfig(**agent_dict)
    
    @property
    def misinformation(self) -> MisinformationConfig:
        """Get misinformation configuration."""
        misinfo_dict = self.get("misinformation", {})
        return MisinformationConfig(**misinfo_dict)
    
    @property
    def economics(self) -> EconomicConfig:
        """Get economic configuration."""
        econ_dict = self.get("economics", {})
        return EconomicConfig(**econ_dict)
    
    @property
    def network(self) -> NetworkConfig:
        """Get network configuration."""
        net_dict = self.get("network", {})
        return NetworkConfig(**net_dict)
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        db_dict = self.get("database", {})
        return DatabaseConfig(**db_dict)
    
    @property
    def logging_interval(self) -> int:
        """Get logging interval for simulation steps."""
        value = self.get("simulation.logging_interval", 10)
        return value if value is not None else 10
    
    def validate(self) -> bool:
        """Validate configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate simulation parameters
            sim = self.simulation
            assert sim.duration_days > 0, "Duration must be positive"
            assert sim.time_step_hours > 0, "Time step must be positive"
            assert sim.max_agents > 0, "Max agents must be positive"
            
            # Validate agent parameters
            agents = self.agents
            assert agents.readers["count"] > 0, "Reader count must be positive"
            assert 0 <= agents.readers["urban_rural_ratio"] <= 1, "Urban/rural ratio must be between 0 and 1"
            
            # Validate misinformation parameters
            misinfo = self.misinformation
            assert 0 <= misinfo.base_rate <= 1, "Base rate must be between 0 and 1"
            assert 0 <= misinfo.foreign_influence <= 1, "Foreign influence must be between 0 and 1"
            
            # Validate economic parameters
            econ = self.economics
            assert econ.ad_market_size_usd > 0, "Ad market size must be positive"
            assert 0 <= econ.subscription_willingness <= 1, "Subscription willingness must be between 0 and 1"
            
            return True
            
        except AssertionError as e:
            raise ValueError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables.
        
        Returns:
            Dictionary of environment variable overrides
        """
        overrides = {}
        
        # Database overrides
        if os.getenv("NEWSGUARD_DB_HOST"):
            overrides["database.postgresql.host"] = os.getenv("NEWSGUARD_DB_HOST")
        if os.getenv("NEWSGUARD_DB_PASSWORD"):
            overrides["database.postgresql.password"] = os.getenv("NEWSGUARD_DB_PASSWORD")
        
        # Simulation overrides
        if os.getenv("NEWSGUARD_DURATION_DAYS"):
            overrides["simulation.duration_days"] = int(os.getenv("NEWSGUARD_DURATION_DAYS"))
        if os.getenv("NEWSGUARD_RANDOM_SEED"):
            overrides["simulation.random_seed"] = int(os.getenv("NEWSGUARD_RANDOM_SEED"))
        
        return overrides
    
    def apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        overrides = self.get_environment_overrides()
        for key, value in overrides.items():
            self.set(key, value)


# Global configuration instance
_global_config = None


def get_config() -> Config:
    """Get global configuration instance.
    
    Returns:
        Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
        _global_config.apply_environment_overrides()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance.
    
    Args:
        config: Configuration instance to set as global
    """
    global _global_config
    _global_config = config