"""Main simulation engine for NewsGuard Bangladesh."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import mesa
import networkx as nx
import pandas as pd
import numpy as np

from ..utils.config import SimulationConfig
from ..utils.logger import get_logger, SimulationLogger
from ..utils.helpers import generate_uuid, timing_decorator
from .scheduler import EventScheduler, SimulationEvent
from .metrics import MetricsCollector, SimulationMetrics
from .state import SimulationState, StateManager

logger = get_logger(__name__)
sim_logger = SimulationLogger()


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    simulation_id: str
    start_time: datetime
    end_time: datetime
    total_steps: int
    final_metrics: SimulationMetrics
    agent_data: Dict[str, Any]
    network_data: Dict[str, Any]
    event_log: List[Dict[str, Any]]
    performance_stats: Dict[str, float]
    

@dataclass
class SimulationProgress:
    """Progress information for a running simulation."""
    current_step: int
    total_steps: int
    elapsed_time: float
    estimated_remaining: float
    current_metrics: SimulationMetrics
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100


class SimulationEngine(mesa.Model):
    """Main simulation engine for NewsGuard Bangladesh."""
    
    def __init__(self, config: SimulationConfig, simulation_id: str = None):
        """Initialize simulation engine.
        
        Args:
            config: Simulation configuration
            simulation_id: Optional simulation ID
        """
        super().__init__()
        
        self.simulation_id = simulation_id or generate_uuid()
        self.config = config
        self.start_time = None
        self.end_time = None
        
        # Initialize Mesa scheduler
        self.schedule = mesa.time.RandomActivation(self)
        
        # Core components
        self.scheduler = EventScheduler()
        self.metrics_collector = MetricsCollector()
        self.state_manager = StateManager()
        
        # Simulation state
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        
        # Agent management
        self._agents_by_type_custom = defaultdict(list)
        self.agent_registry = {}
        
        # Network management
        self.social_network = nx.Graph()
        self.information_network = nx.DiGraph()
        self.economic_network = nx.Graph()
        
        # Event tracking
        self.event_log = []
        self.intervention_log = []
        
        # Performance tracking
        self.step_times = []
        self.memory_usage = []
        
        # Callbacks
        self.step_callbacks = []
        self.event_callbacks = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Simulation engine initialized: {self.simulation_id}")
        sim_logger.log_simulation_start(self.simulation_id, config)
    
    def add_agent(self, agent_class: Type, agent_config: Dict[str, Any], 
                  count: int = 1) -> List[Any]:
        """Add agents to the simulation.
        
        Args:
            agent_class: Agent class to instantiate
            agent_config: Configuration for agents
            count: Number of agents to create
            
        Returns:
            List of created agents
        """
        created_agents = []
        
        for i in range(count):
            # Create unique agent ID
            agent_id = f"{agent_class.__name__.lower()}_{len(self.agent_registry)}"
            
            # Create agent instance
            agent = agent_class(
                unique_id=agent_id,
                model=self,
                config=agent_config
            )
            
            # Register agent
            self.schedule.add(agent)
            self.agent_registry[agent_id] = agent
            self._agents_by_type_custom[agent_class.__name__].append(agent)
            
            created_agents.append(agent)
            
            logger.debug(f"Created agent: {agent_id} ({agent_class.__name__})")
        
        logger.info(f"Added {count} agents of type {agent_class.__name__}")
        return created_agents
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from simulation.
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            True if agent was removed, False otherwise
        """
        if agent_id not in self.agent_registry:
            return False
        
        agent = self.agent_registry[agent_id]
        
        # Remove from scheduler
        self.schedule.remove(agent)
        
        # Remove from registry
        del self.agent_registry[agent_id]
        
        # Remove from type list
        agent_type = type(agent).__name__
        if agent in self._agents_by_type_custom[agent_type]:
            self._agents_by_type_custom[agent_type].remove(agent)
        
        logger.debug(f"Removed agent: {agent_id}")
        return True
    
    def get_agents_by_type(self, agent_type: str) -> List[Any]:
        """Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of agents
        """
        return self._agents_by_type_custom.get(agent_type, [])
    
    def add_network_edge(self, network_type: str, source: str, target: str, 
                        **attributes) -> None:
        """Add edge to a network.
        
        Args:
            network_type: Type of network ('social', 'information', 'economic')
            source: Source node ID
            target: Target node ID
            **attributes: Edge attributes
        """
        if network_type == 'social':
            self.social_network.add_edge(source, target, **attributes)
        elif network_type == 'information':
            self.information_network.add_edge(source, target, **attributes)
        elif network_type == 'economic':
            self.economic_network.add_edge(source, target, **attributes)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def get_network(self, network_type: str) -> nx.Graph:
        """Get network by type.
        
        Args:
            network_type: Type of network
            
        Returns:
            Network graph
        """
        if network_type == 'social':
            return self.social_network
        elif network_type == 'information':
            return self.information_network
        elif network_type == 'economic':
            return self.economic_network
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def schedule_event(self, event: SimulationEvent) -> None:
        """Schedule an event.
        
        Args:
            event: Event to schedule
        """
        self.scheduler.schedule_event(event)
        logger.debug(f"Scheduled event: {event.event_type} at step {event.scheduled_step}")
    
    def add_step_callback(self, callback: Callable[[int], None]) -> None:
        """Add callback to be called after each step.
        
        Args:
            callback: Callback function
        """
        self.step_callbacks.append(callback)
    
    def add_event_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback to be called when events occur.
        
        Args:
            callback: Callback function
        """
        self.event_callbacks.append(callback)
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a simulation event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event_record = {
            'step': self.current_step,
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': data
        }
        
        self.event_log.append(event_record)
        
        # Call event callbacks
        for callback in self.event_callbacks:
            try:
                callback(event_record)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    @timing_decorator
    def step(self) -> None:
        """Execute one simulation step."""
        if self.stop_requested:
            return
        
        step_start_time = time.time()
        
        with self._lock:
            # Process scheduled events
            events = self.scheduler.get_events_for_step(self.current_step)
            for event in events:
                try:
                    event.execute(self)
                    self.log_event('scheduled_event', {
                        'event_type': event.event_type,
                        'event_id': event.event_id
                    })
                except Exception as e:
                    logger.error(f"Event execution failed: {e}")
            
            # Step all agents
            self.schedule.step()
            
            # Collect metrics
            current_metrics = self.metrics_collector.collect_step_metrics(
                self, self.current_step
            )
            
            # Auto-save state if needed
            self.state_manager.auto_save_check(self)
            
            # Increment step counter
            self.current_step += 1
            
            # Track performance
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            
            # Call step callbacks
            for callback in self.step_callbacks:
                try:
                    callback(self.current_step)
                except Exception as e:
                    logger.warning(f"Step callback failed: {e}")
            
            # Log step completion
            if self.current_step % self.config.logging_interval == 0:
                logger.info(f"Completed step {self.current_step}")
                sim_logger.log_step_completion(
                    self.simulation_id, self.current_step, current_metrics
                )
    
    def run(self, steps: int = None) -> SimulationResult:
        """Run the simulation.
        
        Args:
            steps: Number of steps to run (uses config if not provided)
            
        Returns:
            Simulation results
        """
        if steps is None:
            steps = self.config.max_steps
        
        self.start_time = datetime.now()
        self.is_running = True
        self.stop_requested = False
        
        logger.info(f"Starting simulation {self.simulation_id} for {steps} steps")
        sim_logger.log_simulation_start(self.simulation_id, self.config)
        
        try:
            # Initialize simulation state
            self._initialize_simulation()
            
            # Main simulation loop
            for step_num in range(steps):
                if self.stop_requested:
                    logger.info("Simulation stopped by request")
                    break
                
                if self.is_paused:
                    while self.is_paused and not self.stop_requested:
                        time.sleep(0.1)
                    continue
                
                self.step()
                
                # Check termination conditions
                if self._should_terminate():
                    logger.info("Simulation terminated by condition")
                    break
            
            self.end_time = datetime.now()
            self.is_running = False
            
            # Generate final results
            result = self._generate_results()
            
            logger.info(f"Simulation {self.simulation_id} completed")
            sim_logger.log_simulation_end(self.simulation_id, result)
            
            return result
            
        except Exception as e:
            self.end_time = datetime.now()
            self.is_running = False
            logger.error(f"Simulation failed: {e}")
            sim_logger.log_simulation_error(self.simulation_id, str(e))
            raise
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.is_paused = True
        logger.info(f"Simulation {self.simulation_id} paused")
    
    def resume(self) -> None:
        """Resume the simulation."""
        self.is_paused = False
        logger.info(f"Simulation {self.simulation_id} resumed")
    
    def stop(self) -> None:
        """Stop the simulation."""
        self.stop_requested = True
        logger.info(f"Simulation {self.simulation_id} stop requested")
    
    def get_progress(self) -> SimulationProgress:
        """Get current simulation progress.
        
        Returns:
            Progress information
        """
        if not self.start_time:
            elapsed_time = 0.0
            estimated_remaining = 0.0
        else:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            if self.current_step > 0:
                avg_step_time = elapsed_time / self.current_step
                remaining_steps = self.config.max_steps - self.current_step
                estimated_remaining = avg_step_time * remaining_steps
            else:
                estimated_remaining = 0.0
        
        current_metrics = self.metrics_collector.get_current_metrics()
        
        return SimulationProgress(
            current_step=self.current_step,
            total_steps=self.config.max_steps,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining,
            current_metrics=current_metrics
        )
    
    def _initialize_simulation(self) -> None:
        """Initialize simulation components."""
        # Initialize networks
        self._initialize_networks()
        
        # Initialize agents
        self._initialize_agents()
        
        # Schedule initial events
        self._schedule_initial_events()
        
        logger.info("Simulation initialization completed")
    
    def _initialize_networks(self) -> None:
        """Initialize simulation networks."""
        # This would be implemented based on specific network generation strategies
        logger.debug("Networks initialized")
    
    def _initialize_agents(self) -> None:
        """Initialize simulation agents."""
        # This would be implemented based on agent configuration
        logger.debug("Agents initialized")
    
    def _schedule_initial_events(self) -> None:
        """Schedule initial simulation events."""
        # This would be implemented based on scenario configuration
        logger.debug("Initial events scheduled")
    
    def _should_terminate(self) -> bool:
        """Check if simulation should terminate early.
        
        Returns:
            True if simulation should terminate
        """
        # Implement termination conditions based on metrics or state
        return False
    
    def _generate_results(self) -> SimulationResult:
        """Generate simulation results.
        
        Returns:
            Simulation results
        """
        # Collect final metrics
        final_metrics = self.metrics_collector.get_final_metrics()
        
        # Collect agent data
        agent_data = {}
        for agent_id, agent in self.agent_registry.items():
            agent_data[agent_id] = {
                'type': type(agent).__name__,
                'final_state': getattr(agent, 'get_state', lambda: {})(),
                'metrics': getattr(agent, 'get_metrics', lambda: {})(),
            }
        
        # Collect network data
        network_data = {
            'social_network': {
                'nodes': self.social_network.number_of_nodes(),
                'edges': self.social_network.number_of_edges(),
                'density': nx.density(self.social_network) if self.social_network.number_of_nodes() > 0 else 0
            },
            'information_network': {
                'nodes': self.information_network.number_of_nodes(),
                'edges': self.information_network.number_of_edges(),
                'density': nx.density(self.information_network) if self.information_network.number_of_nodes() > 0 else 0
            },
            'economic_network': {
                'nodes': self.economic_network.number_of_nodes(),
                'edges': self.economic_network.number_of_edges(),
                'density': nx.density(self.economic_network) if self.economic_network.number_of_nodes() > 0 else 0
            }
        }
        
        # Calculate performance stats
        performance_stats = {
            'total_runtime': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
            'avg_step_time': np.mean(self.step_times) if self.step_times else 0,
            'max_step_time': np.max(self.step_times) if self.step_times else 0,
            'min_step_time': np.min(self.step_times) if self.step_times else 0,
            'total_agents': len(self.agent_registry),
            'total_events': len(self.event_log)
        }
        
        return SimulationResult(
            simulation_id=self.simulation_id,
            start_time=self.start_time,
            end_time=self.end_time,
            total_steps=self.current_step,
            final_metrics=final_metrics,
            agent_data=agent_data,
            network_data=network_data,
            event_log=self.event_log,
            performance_stats=performance_stats
        )
    
    def save_state(self, filepath: str) -> None:
        """Save simulation state to file.
        
        Args:
            filepath: Path to save state
        """
        self.state_manager.save_state(self, filepath)
        logger.info(f"Simulation state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load simulation state from file.
        
        Args:
            filepath: Path to load state from
        """
        self.state_manager.load_state(self, filepath)
        logger.info(f"Simulation state loaded from {filepath}")
    
    def export_data(self, output_dir: str, formats: List[str] = None) -> None:
        """Export simulation data in various formats.
        
        Args:
            output_dir: Output directory
            formats: List of formats ('csv', 'json', 'pickle')
        """
        if formats is None:
            formats = ['csv', 'json']
        
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export metrics
        metrics_data = self.metrics_collector.export_data()
        
        if 'csv' in formats:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(output_path / 'metrics.csv', index=False)
        
        if 'json' in formats:
            import json
            with open(output_path / 'simulation_data.json', 'w') as f:
                json.dump({
                    'simulation_id': self.simulation_id,
                    'config': self.config.__dict__,
                    'metrics': metrics_data,
                    'events': self.event_log
                }, f, indent=2, default=str)
        
        if 'pickle' in formats:
            import pickle
            with open(output_path / 'simulation_state.pkl', 'wb') as f:
                pickle.dump({
                    'simulation': self,
                    'metrics': metrics_data,
                    'events': self.event_log
                }, f)
        
        logger.info(f"Simulation data exported to {output_dir}")


class SimulationManager:
    """Manager for multiple simulation runs."""
    
    def __init__(self):
        """Initialize simulation manager."""
        self.simulations = {}
        self.results = {}
        self._executor = None
    
    def create_simulation(self, config: SimulationConfig, 
                         simulation_id: str = None) -> str:
        """Create a new simulation.
        
        Args:
            config: Simulation configuration
            simulation_id: Optional simulation ID
            
        Returns:
            Simulation ID
        """
        if simulation_id is None:
            simulation_id = generate_uuid()
        
        simulation = SimulationEngine(config, simulation_id)
        self.simulations[simulation_id] = simulation
        
        logger.info(f"Created simulation: {simulation_id}")
        return simulation_id
    
    def run_simulation(self, simulation_id: str, steps: int = None) -> SimulationResult:
        """Run a simulation.
        
        Args:
            simulation_id: ID of simulation to run
            steps: Number of steps to run
            
        Returns:
            Simulation results
        """
        if simulation_id not in self.simulations:
            raise ValueError(f"Simulation not found: {simulation_id}")
        
        simulation = self.simulations[simulation_id]
        result = simulation.run(steps)
        self.results[simulation_id] = result
        
        return result
    
    def run_batch(self, configs: List[SimulationConfig], 
                  max_workers: int = 4) -> Dict[str, SimulationResult]:
        """Run multiple simulations in parallel.
        
        Args:
            configs: List of simulation configurations
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary of simulation results
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all simulations
            future_to_id = {}
            for config in configs:
                sim_id = self.create_simulation(config)
                future = executor.submit(self.run_simulation, sim_id)
                future_to_id[future] = sim_id
            
            # Collect results
            for future in as_completed(future_to_id):
                sim_id = future_to_id[future]
                try:
                    result = future.result()
                    results[sim_id] = result
                    logger.info(f"Batch simulation completed: {sim_id}")
                except Exception as e:
                    logger.error(f"Batch simulation failed: {sim_id} - {e}")
                    results[sim_id] = None
        
        return results
    
    def get_simulation(self, simulation_id: str) -> Optional[SimulationEngine]:
        """Get simulation by ID.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation engine or None
        """
        return self.simulations.get(simulation_id)
    
    def get_result(self, simulation_id: str) -> Optional[SimulationResult]:
        """Get simulation result by ID.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation result or None
        """
        return self.results.get(simulation_id)
    
    def list_simulations(self) -> List[str]:
        """List all simulation IDs.
        
        Returns:
            List of simulation IDs
        """
        return list(self.simulations.keys())
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        
        self.simulations.clear()
        self.results.clear()
        
        logger.info("Simulation manager cleaned up")