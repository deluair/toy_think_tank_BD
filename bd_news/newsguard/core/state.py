"""State management for NewsGuard Bangladesh simulation."""

import json
import pickle
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

import numpy as np
import networkx as nx
from mesa import Model

from ..utils.logger import get_logger
from ..utils.helpers import ensure_directory, safe_json_serialize
from .metrics import SimulationMetrics

logger = get_logger(__name__)


class StateFormat(Enum):
    """Supported state serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_PICKLE = "compressed_pickle"


@dataclass
class SimulationState:
    """Complete simulation state snapshot."""
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    step: int = 0
    simulation_id: str = ""
    version: str = "1.0"
    
    # Core state
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    networks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scheduler_state: Dict[str, Any] = field(default_factory=dict)
    
    # Simulation data
    metrics: Optional[Dict[str, Any]] = None
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Model state
    model_state: Dict[str, Any] = field(default_factory=dict)
    random_state: Optional[Dict[str, Any]] = None
    
    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.simulation_id:
            # Generate unique simulation ID
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            hash_input = f"{timestamp_str}_{self.step}"
            hash_obj = hashlib.md5(hash_input.encode())
            self.simulation_id = f"sim_{timestamp_str}_{hash_obj.hexdigest()[:8]}"
    
    def get_state_hash(self) -> str:
        """Generate hash of current state for integrity checking.
        
        Returns:
            SHA-256 hash of state
        """
        # Create a simplified representation for hashing
        hash_data = {
            'step': self.step,
            'agents_count': len(self.agents),
            'networks_count': len(self.networks),
            'events_count': len(self.event_log)
        }
        
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def get_size_estimate(self) -> Dict[str, int]:
        """Estimate memory usage of different state components.
        
        Returns:
            Size estimates in bytes
        """
        sizes = {}
        
        # Estimate agent data size
        agent_size = 0
        for agent_data in self.agents.values():
            agent_size += len(str(agent_data))
        sizes['agents'] = agent_size
        
        # Estimate network data size
        network_size = 0
        for network_data in self.networks.values():
            network_size += len(str(network_data))
        sizes['networks'] = network_size
        
        # Estimate event log size
        sizes['event_log'] = len(str(self.event_log))
        
        # Estimate total size
        sizes['total'] = sum(sizes.values())
        
        return sizes


class StateManager:
    """Manages simulation state persistence and recovery."""
    
    def __init__(self, 
                 state_dir: str = "simulation_states",
                 auto_save_interval: int = 100,
                 max_states: int = 50,
                 compression: bool = True):
        """Initialize state manager.
        
        Args:
            state_dir: Directory to store state files
            auto_save_interval: Steps between automatic saves (0 to disable)
            max_states: Maximum number of states to keep
            compression: Whether to compress state files
        """
        self.state_dir = Path(state_dir)
        self.auto_save_interval = auto_save_interval
        self.max_states = max_states
        self.compression = compression
        
        # Ensure state directory exists
        ensure_directory(str(self.state_dir))
        
        # State tracking
        self.current_state: Optional[SimulationState] = None
        self.state_history: List[str] = []  # List of state file paths
        self.last_save_step = 0
        
        # Performance tracking
        self.save_times: List[float] = []
        self.load_times: List[float] = []
        
        logger.debug(f"State manager initialized with directory: {self.state_dir}")
    
    def capture_state(self, simulation) -> SimulationState:
        """Capture current simulation state.
        
        Args:
            simulation: Simulation engine instance
            
        Returns:
            Captured state
        """
        import time
        start_time = time.time()
        
        state = SimulationState(
            step=simulation.schedule.steps,
            simulation_id=getattr(simulation, 'simulation_id', '')
        )
        
        # Capture agent states
        state.agents = self._capture_agent_states(simulation)
        
        # Capture network states
        state.networks = self._capture_network_states(simulation)
        
        # Capture scheduler state
        state.scheduler_state = self._capture_scheduler_state(simulation)
        
        # Capture metrics
        if hasattr(simulation, 'metrics_collector'):
            metrics = simulation.metrics_collector.get_current_metrics()
            state.metrics = asdict(metrics) if metrics else None
        
        # Capture event log
        state.event_log = getattr(simulation, 'event_log', [])
        
        # Capture configuration
        state.configuration = getattr(simulation, 'config', {})
        if hasattr(state.configuration, '__dict__'):
            state.configuration = asdict(state.configuration)
        
        # Capture model state
        state.model_state = self._capture_model_state(simulation)
        
        # Capture random state
        state.random_state = self._capture_random_state(simulation)
        
        # Store current state
        self.current_state = state
        
        capture_time = time.time() - start_time
        logger.debug(f"State captured in {capture_time:.4f}s")
        
        return state
    
    def _capture_agent_states(self, simulation) -> Dict[str, Dict[str, Any]]:
        """Capture states of all agents.
        
        Args:
            simulation: Simulation instance
            
        Returns:
            Dictionary of agent states
        """
        agent_states = {}
        
        for agent_id, agent in simulation.agent_registry.items():
            try:
                # Get agent's serializable state
                if hasattr(agent, 'get_state'):
                    agent_state = agent.get_state()
                else:
                    # Fallback: capture basic attributes
                    agent_state = {
                        'agent_id': agent_id,
                        'agent_type': type(agent).__name__,
                        'pos': getattr(agent, 'pos', None),
                        'unique_id': getattr(agent, 'unique_id', agent_id)
                    }
                    
                    # Add common attributes if they exist
                    for attr in ['trust_score', 'activity_level', 'influence_score', 
                                'is_active', 'is_suspended', 'content_produced']:
                        if hasattr(agent, attr):
                            value = getattr(agent, attr)
                            # Ensure value is serializable
                            if isinstance(value, (int, float, str, bool, list, dict)):
                                agent_state[attr] = value
                            else:
                                agent_state[attr] = str(value)
                
                agent_states[agent_id] = agent_state
                
            except Exception as e:
                logger.warning(f"Failed to capture state for agent {agent_id}: {e}")
                # Store minimal state
                agent_states[agent_id] = {
                    'agent_id': agent_id,
                    'agent_type': type(agent).__name__,
                    'error': str(e)
                }
        
        return agent_states
    
    def _capture_network_states(self, simulation) -> Dict[str, Dict[str, Any]]:
        """Capture states of all networks.
        
        Args:
            simulation: Simulation instance
            
        Returns:
            Dictionary of network states
        """
        network_states = {}
        
        # Get networks from simulation
        networks = getattr(simulation, 'networks', {})
        
        for network_name, network in networks.items():
            try:
                if isinstance(network, nx.Graph):
                    # Convert NetworkX graph to serializable format
                    network_data = {
                        'nodes': list(network.nodes(data=True)),
                        'edges': list(network.edges(data=True)),
                        'graph_attrs': dict(network.graph),
                        'directed': network.is_directed(),
                        'multigraph': network.is_multigraph()
                    }
                    network_states[network_name] = network_data
                else:
                    # Handle other network types
                    network_states[network_name] = str(network)
                    
            except Exception as e:
                logger.warning(f"Failed to capture network {network_name}: {e}")
                network_states[network_name] = {'error': str(e)}
        
        return network_states
    
    def _capture_scheduler_state(self, simulation) -> Dict[str, Any]:
        """Capture scheduler state.
        
        Args:
            simulation: Simulation instance
            
        Returns:
            Scheduler state dictionary
        """
        scheduler_state = {}
        
        if hasattr(simulation, 'schedule'):
            scheduler = simulation.schedule
            scheduler_state = {
                'steps': getattr(scheduler, 'steps', 0),
                'time': getattr(scheduler, 'time', 0),
                'agent_count': len(getattr(scheduler, 'agents', []))
            }
            
            # Capture agent order if available
            if hasattr(scheduler, 'agents'):
                scheduler_state['agent_order'] = [agent.unique_id for agent in scheduler.agents]
        
        # Capture event scheduler state if available
        if hasattr(simulation, 'event_scheduler'):
            event_scheduler = simulation.event_scheduler
            if hasattr(event_scheduler, 'get_state'):
                scheduler_state['event_scheduler'] = event_scheduler.get_state()
        
        return scheduler_state
    
    def _capture_model_state(self, simulation) -> Dict[str, Any]:
        """Capture model-specific state.
        
        Args:
            simulation: Simulation instance
            
        Returns:
            Model state dictionary
        """
        model_state = {}
        
        # Capture basic model attributes
        if isinstance(simulation, Model):
            model_state = {
                'running': getattr(simulation, 'running', True),
                'current_id': getattr(simulation, '_current_id', 0)
            }
        
        # Capture custom model state if method exists
        if hasattr(simulation, 'get_model_state'):
            try:
                custom_state = simulation.get_model_state()
                model_state.update(custom_state)
            except Exception as e:
                logger.warning(f"Failed to capture custom model state: {e}")
        
        return model_state
    
    def _capture_random_state(self, simulation) -> Optional[Dict[str, Any]]:
        """Capture random number generator state.
        
        Args:
            simulation: Simulation instance
            
        Returns:
            Random state dictionary or None
        """
        try:
            # Capture NumPy random state
            np_state = np.random.get_state()
            
            # Convert to serializable format
            random_state = {
                'numpy': {
                    'state': np_state[1].tolist(),
                    'pos': int(np_state[2]),
                    'has_gauss': int(np_state[3]),
                    'cached_gaussian': float(np_state[4]) if np_state[4] is not None else None
                }
            }
            
            # Capture Python random state if available
            if hasattr(simulation, 'random'):
                import random
                python_state = random.getstate()
                random_state['python'] = {
                    'version': python_state[0],
                    'state': list(python_state[1]),
                    'gauss_next': python_state[2]
                }
            
            return random_state
            
        except Exception as e:
            logger.warning(f"Failed to capture random state: {e}")
            return None
    
    def save_state(self, 
                   state: SimulationState, 
                   filename: Optional[str] = None,
                   format_type: StateFormat = StateFormat.COMPRESSED_PICKLE) -> str:
        """Save simulation state to file.
        
        Args:
            state: State to save
            filename: Custom filename (auto-generated if None)
            format_type: Serialization format
            
        Returns:
            Path to saved file
        """
        import time
        start_time = time.time()
        
        # Generate filename if not provided
        if filename is None:
            timestamp = state.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"state_step_{state.step:06d}_{timestamp}"
        
        # Add appropriate extension
        if format_type == StateFormat.JSON:
            filepath = self.state_dir / f"{filename}.json"
        elif format_type == StateFormat.PICKLE:
            filepath = self.state_dir / f"{filename}.pkl"
        else:  # COMPRESSED_PICKLE
            filepath = self.state_dir / f"{filename}.pkl.gz"
        
        try:
            # Save state based on format
            if format_type == StateFormat.JSON:
                self._save_json_state(state, filepath)
            elif format_type == StateFormat.PICKLE:
                self._save_pickle_state(state, filepath)
            else:  # COMPRESSED_PICKLE
                self._save_compressed_pickle_state(state, filepath)
            
            # Update state history
            self.state_history.append(str(filepath))
            self.last_save_step = state.step
            
            # Cleanup old states if necessary
            self._cleanup_old_states()
            
            save_time = time.time() - start_time
            self.save_times.append(save_time)
            
            logger.info(f"State saved to {filepath} in {save_time:.4f}s")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    def _save_json_state(self, state: SimulationState, filepath: Path) -> None:
        """Save state in JSON format."""
        # Convert state to dictionary
        state_dict = asdict(state)
        
        # Handle datetime serialization
        state_dict['timestamp'] = state.timestamp.isoformat()
        
        # Ensure all data is JSON serializable
        state_dict = safe_json_serialize(state_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    def _save_pickle_state(self, state: SimulationState, filepath: Path) -> None:
        """Save state in pickle format."""
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_compressed_pickle_state(self, state: SimulationState, filepath: Path) -> None:
        """Save state in compressed pickle format."""
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_state(self, filepath: Union[str, Path]) -> SimulationState:
        """Load simulation state from file.
        
        Args:
            filepath: Path to state file
            
        Returns:
            Loaded simulation state
        """
        import time
        start_time = time.time()
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        try:
            # Determine format from extension
            if filepath.suffix == '.json':
                state = self._load_json_state(filepath)
            elif filepath.suffix == '.gz':
                state = self._load_compressed_pickle_state(filepath)
            else:  # .pkl
                state = self._load_pickle_state(filepath)
            
            self.current_state = state
            
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            
            logger.info(f"State loaded from {filepath} in {load_time:.4f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state from {filepath}: {e}")
            raise
    
    def _load_json_state(self, filepath: Path) -> SimulationState:
        """Load state from JSON format."""
        with open(filepath, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        # Handle datetime deserialization
        if 'timestamp' in state_dict:
            state_dict['timestamp'] = datetime.fromisoformat(state_dict['timestamp'])
        
        # Create state object
        state = SimulationState(**state_dict)
        
        return state
    
    def _load_pickle_state(self, filepath: Path) -> SimulationState:
        """Load state from pickle format."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        return state
    
    def _load_compressed_pickle_state(self, filepath: Path) -> SimulationState:
        """Load state from compressed pickle format."""
        with gzip.open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        return state
    
    def restore_simulation(self, simulation, state: SimulationState) -> None:
        """Restore simulation from state.
        
        Args:
            simulation: Simulation instance to restore
            state: State to restore from
        """
        logger.info(f"Restoring simulation from step {state.step}")
        
        try:
            # Restore basic simulation state
            if hasattr(simulation, 'schedule'):
                simulation.schedule.steps = state.step
            
            # Restore agents
            self._restore_agents(simulation, state.agents)
            
            # Restore networks
            self._restore_networks(simulation, state.networks)
            
            # Restore scheduler state
            self._restore_scheduler_state(simulation, state.scheduler_state)
            
            # Restore event log
            if hasattr(simulation, 'event_log'):
                simulation.event_log = state.event_log.copy()
            
            # Restore configuration
            if hasattr(simulation, 'config') and state.configuration:
                simulation.config.update(state.configuration)
            
            # Restore model state
            self._restore_model_state(simulation, state.model_state)
            
            # Restore random state
            if state.random_state:
                self._restore_random_state(state.random_state)
            
            # Restore metrics if available
            if state.metrics and hasattr(simulation, 'metrics_collector'):
                # This would require implementing metrics restoration
                pass
            
            logger.info("Simulation state restored successfully")
            
        except Exception as e:
            logger.error(f"Failed to restore simulation state: {e}")
            raise
    
    def _restore_agents(self, simulation, agent_states: Dict[str, Dict[str, Any]]) -> None:
        """Restore agent states."""
        for agent_id, agent_state in agent_states.items():
            if agent_id in simulation.agent_registry:
                agent = simulation.agent_registry[agent_id]
                
                # Restore agent state
                if hasattr(agent, 'set_state'):
                    agent.set_state(agent_state)
                else:
                    # Fallback: restore basic attributes
                    for attr, value in agent_state.items():
                        if hasattr(agent, attr) and attr not in ['agent_id', 'agent_type']:
                            try:
                                setattr(agent, attr, value)
                            except Exception as e:
                                logger.warning(f"Failed to restore attribute {attr} for agent {agent_id}: {e}")
    
    def _restore_networks(self, simulation, network_states: Dict[str, Dict[str, Any]]) -> None:
        """Restore network states."""
        if not hasattr(simulation, 'networks'):
            simulation.networks = {}
        
        for network_name, network_data in network_states.items():
            try:
                if isinstance(network_data, dict) and 'nodes' in network_data:
                    # Restore NetworkX graph
                    if network_data.get('directed', False):
                        if network_data.get('multigraph', False):
                            graph = nx.MultiDiGraph()
                        else:
                            graph = nx.DiGraph()
                    else:
                        if network_data.get('multigraph', False):
                            graph = nx.MultiGraph()
                        else:
                            graph = nx.Graph()
                    
                    # Add nodes and edges
                    graph.add_nodes_from(network_data['nodes'])
                    graph.add_edges_from(network_data['edges'])
                    
                    # Restore graph attributes
                    graph.graph.update(network_data.get('graph_attrs', {}))
                    
                    simulation.networks[network_name] = graph
                    
            except Exception as e:
                logger.warning(f"Failed to restore network {network_name}: {e}")
    
    def _restore_scheduler_state(self, simulation, scheduler_state: Dict[str, Any]) -> None:
        """Restore scheduler state."""
        if hasattr(simulation, 'schedule') and scheduler_state:
            scheduler = simulation.schedule
            
            # Restore basic scheduler attributes
            if 'steps' in scheduler_state:
                scheduler.steps = scheduler_state['steps']
            if 'time' in scheduler_state:
                scheduler.time = scheduler_state['time']
        
        # Restore event scheduler state
        if hasattr(simulation, 'event_scheduler') and 'event_scheduler' in scheduler_state:
            event_scheduler = simulation.event_scheduler
            if hasattr(event_scheduler, 'set_state'):
                event_scheduler.set_state(scheduler_state['event_scheduler'])
    
    def _restore_model_state(self, simulation, model_state: Dict[str, Any]) -> None:
        """Restore model state."""
        if model_state:
            # Skip attributes that are known to be read-only or problematic
            skip_attributes = {'agents_by_type', '_agents_by_type_custom', 'agent_registry', 'schedule'}
            
            # Restore basic model attributes
            for attr, value in model_state.items():
                if attr in skip_attributes:
                    continue
                    
                if hasattr(simulation, attr):
                    try:
                        # Check if attribute has a setter
                        attr_obj = getattr(type(simulation), attr, None)
                        if isinstance(attr_obj, property) and attr_obj.fset is None:
                            logger.debug(f"Skipping read-only property {attr}")
                            continue
                            
                        setattr(simulation, attr, value)
                    except Exception as e:
                        logger.warning(f"Failed to restore model attribute {attr}: {e}")
            
            # Restore custom model state if method exists
            if hasattr(simulation, 'set_model_state'):
                try:
                    simulation.set_model_state(model_state)
                except Exception as e:
                    logger.warning(f"Failed to restore custom model state: {e}")
    
    def _restore_random_state(self, random_state: Dict[str, Any]) -> None:
        """Restore random number generator state."""
        try:
            # Restore NumPy random state
            if 'numpy' in random_state:
                np_state = random_state['numpy']
                state_tuple = (
                    'MT19937',
                    np.array(np_state['state'], dtype=np.uint32),
                    np_state['pos'],
                    np_state['has_gauss'],
                    np_state['cached_gaussian']
                )
                np.random.set_state(state_tuple)
            
            # Restore Python random state
            if 'python' in random_state:
                import random
                python_state = random_state['python']
                state_tuple = (
                    python_state['version'],
                    tuple(python_state['state']),
                    python_state['gauss_next']
                )
                random.setstate(state_tuple)
                
        except Exception as e:
            logger.warning(f"Failed to restore random state: {e}")
    
    def auto_save_check(self, simulation) -> Optional[str]:
        """Check if auto-save should be performed.
        
        Args:
            simulation: Simulation instance
            
        Returns:
            Path to saved file if save was performed, None otherwise
        """
        if self.auto_save_interval <= 0:
            return None
        
        current_step = getattr(simulation.schedule, 'steps', 0)
        
        if current_step - self.last_save_step >= self.auto_save_interval:
            state = self.capture_state(simulation)
            return self.save_state(state)
        
        return None
    
    def _cleanup_old_states(self) -> None:
        """Remove old state files to maintain max_states limit."""
        if len(self.state_history) <= self.max_states:
            return
        
        # Remove oldest states
        states_to_remove = len(self.state_history) - self.max_states
        
        for i in range(states_to_remove):
            old_state_path = Path(self.state_history[i])
            try:
                if old_state_path.exists():
                    old_state_path.unlink()
                    logger.debug(f"Removed old state file: {old_state_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old state file {old_state_path}: {e}")
        
        # Update history
        self.state_history = self.state_history[states_to_remove:]
    
    def list_saved_states(self) -> List[Dict[str, Any]]:
        """List all saved state files.
        
        Returns:
            List of state file information
        """
        states = []
        
        for state_file in self.state_dir.glob("*.pkl*"):
            try:
                stat = state_file.stat()
                states.append({
                    'filepath': str(state_file),
                    'filename': state_file.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'format': self._detect_format(state_file)
                })
            except Exception as e:
                logger.warning(f"Failed to get info for state file {state_file}: {e}")
        
        # Sort by modification time (newest first)
        states.sort(key=lambda x: x['modified'], reverse=True)
        
        return states
    
    def _detect_format(self, filepath: Path) -> StateFormat:
        """Detect state file format from extension."""
        if filepath.suffix == '.json':
            return StateFormat.JSON
        elif filepath.suffix == '.gz':
            return StateFormat.COMPRESSED_PICKLE
        else:
            return StateFormat.PICKLE
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get state management performance statistics.
        
        Returns:
            Performance statistics
        """
        stats = {
            'total_saves': len(self.save_times),
            'total_loads': len(self.load_times),
            'avg_save_time': np.mean(self.save_times) if self.save_times else 0,
            'avg_load_time': np.mean(self.load_times) if self.load_times else 0,
            'max_save_time': np.max(self.save_times) if self.save_times else 0,
            'max_load_time': np.max(self.load_times) if self.load_times else 0,
            'states_in_history': len(self.state_history),
            'last_save_step': self.last_save_step
        }
        
        return stats
    
    def clear_state_history(self) -> None:
        """Clear all saved states and history."""
        # Remove all state files
        for state_path in self.state_history:
            try:
                Path(state_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to remove state file {state_path}: {e}")
        
        # Clear history
        self.state_history.clear()
        self.save_times.clear()
        self.load_times.clear()
        self.last_save_step = 0
        
        logger.info("State history cleared")