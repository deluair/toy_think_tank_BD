#!/usr/bin/env python3
"""
Test script to verify the NewsGuard Bangladesh simulation framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from newsguard.core.simulation import SimulationEngine
from newsguard.agents.news_outlet import NewsOutlet
from newsguard.agents.base import AgentType
from newsguard.models.content import ContentModel
from newsguard.models.network import NetworkModel
from newsguard.models.trust import TrustModel
from newsguard.models.economic import EconomicModel
from newsguard.models.behavior import BehaviorModel
from newsguard.models.influence import InfluenceModel
from newsguard.utils.config import Config
from newsguard.utils.logger import get_logger

def test_basic_simulation():
    """Test basic simulation functionality."""
    print("Testing NewsGuard Bangladesh Simulation Framework...")
    
    try:
        # Initialize logger
        logger = get_logger("test_simulation")
        logger.info("Starting simulation test")
        
        # Create basic configuration
        config = Config(config_dict={
            'simulation': {
                'steps': 10,
                'width': 50,
                'height': 50,
                'seed': 42
            },
            'agents': {
                'news_outlets': 5,
                'readers': 20,
                'platforms': 2
            },
            'models': {
                'content': {'enabled': True},
                'network': {'enabled': True},
                'trust': {'enabled': True},
                'economic': {'enabled': True},
                'behavior': {'enabled': True},
                'influence': {'enabled': True}
            }
        })
        
        # Initialize simulation engine
        simulation = SimulationEngine(config=config)
        logger.info("Simulation engine initialized successfully")
        
        # Add some test agents
        for i in range(3):
            outlet_config = {
                "name": f"Test Outlet {i+1}",
                "outlet_type": "mainstream_newspaper",
                "editorial_stance": "centrist",
                "credibility_score": 0.8
            }
            outlet = NewsOutlet(
                unique_id=f"outlet_{i}",
                model=simulation,
                config=outlet_config
            )
            simulation.schedule.add(outlet)
        
        logger.info(f"Added {len(simulation.schedule.agents)} test agents")
        
        # Run a few simulation steps
        print("Running simulation for 5 steps...")
        for step in range(5):
            simulation.step()
            print(f"Step {step + 1} completed")
        
        # Collect metrics
        metrics = simulation.metrics_collector.get_current_metrics()
        print(f"\nSimulation completed successfully!")
        print(f"Total agents: {len(simulation.schedule.agents)}")
        print(f"Simulation steps: {simulation.schedule.steps}")
        print(f"Content items created: {metrics.content.total_content}")
        
        logger.info("Simulation test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during simulation test: {e}")
        logger.error(f"Simulation test failed: {e}")
        return False

def test_models():
    """Test individual model components."""
    print("\nTesting individual models...")
    
    try:
        # Test ContentModel
        content_model = ContentModel()
        print("✓ ContentModel initialized")
        
        # Test NetworkModel
        network_model = NetworkModel()
        print("✓ NetworkModel initialized")
        
        # Test TrustModel
        trust_model = TrustModel()
        print("✓ TrustModel initialized")
        
        # Test EconomicModel
        economic_model = EconomicModel()
        print("✓ EconomicModel initialized")
        
        # Test BehaviorModel
        behavior_model = BehaviorModel()
        print("✓ BehaviorModel initialized")
        
        # Test InfluenceModel
        influence_model = InfluenceModel()
        print("✓ InfluenceModel initialized")
        
        print("All models initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing models: {e}")
        return False

if __name__ == "__main__":
    print("NewsGuard Bangladesh Simulation Framework Test")
    print("=" * 50)
    
    # Test models first
    models_ok = test_models()
    
    if models_ok:
        # Test basic simulation
        simulation_ok = test_basic_simulation()
        
        if simulation_ok:
            print("\n" + "=" * 50)
            print("✅ ALL TESTS PASSED!")
            print("The NewsGuard Bangladesh simulation framework is working correctly.")
            sys.exit(0)
        else:
            print("\n" + "=" * 50)
            print("❌ SIMULATION TEST FAILED!")
            sys.exit(1)
    else:
        print("\n" + "=" * 50)
        print("❌ MODEL TESTS FAILED!")
        sys.exit(1)