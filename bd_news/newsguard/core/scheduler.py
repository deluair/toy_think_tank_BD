"""Event scheduling system for NewsGuard Bangladesh simulation."""

import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import uuid

from ..utils.logger import get_logger
from ..utils.helpers import generate_uuid

logger = get_logger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """Event execution status."""
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationEvent:
    """Base class for simulation events."""
    event_id: str
    event_type: str
    scheduled_step: int
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    status: EventStatus = EventStatus.SCHEDULED
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __lt__(self, other):
        """Compare events for priority queue ordering."""
        if self.scheduled_step != other.scheduled_step:
            return self.scheduled_step < other.scheduled_step
        return self.priority.value > other.priority.value  # Higher priority first
    
    def execute(self, simulation) -> bool:
        """Execute the event.
        
        Args:
            simulation: Simulation engine instance
            
        Returns:
            True if execution successful, False otherwise
        """
        try:
            self.status = EventStatus.EXECUTING
            self.executed_at = datetime.now()
            
            success = self._execute_impl(simulation)
            
            if success:
                self.status = EventStatus.COMPLETED
                logger.debug(f"Event executed successfully: {self.event_id}")
            else:
                self.status = EventStatus.FAILED
                logger.warning(f"Event execution failed: {self.event_id}")
            
            return success
            
        except Exception as e:
            self.status = EventStatus.FAILED
            self.error_message = str(e)
            logger.error(f"Event execution error: {self.event_id} - {e}")
            return False
    
    @abstractmethod
    def _execute_impl(self, simulation) -> bool:
        """Implementation-specific execution logic.
        
        Args:
            simulation: Simulation engine instance
            
        Returns:
            True if execution successful
        """
        pass
    
    def cancel(self) -> None:
        """Cancel the event."""
        self.status = EventStatus.CANCELLED
        logger.debug(f"Event cancelled: {self.event_id}")


class ContentPublicationEvent(SimulationEvent):
    """Event for publishing content."""
    
    def __init__(self, scheduled_step: int, publisher_id: str, 
                 content_data: Dict[str, Any], **kwargs):
        super().__init__(
            event_id=generate_uuid(),
            event_type="content_publication",
            scheduled_step=scheduled_step,
            data={
                'publisher_id': publisher_id,
                'content_data': content_data
            },
            **kwargs
        )
    
    def _execute_impl(self, simulation) -> bool:
        """Execute content publication."""
        publisher_id = self.data['publisher_id']
        content_data = self.data['content_data']
        
        # Get publisher agent
        publisher = simulation.agent_registry.get(publisher_id)
        if not publisher:
            logger.warning(f"Publisher not found: {publisher_id}")
            return False
        
        # Publish content
        if hasattr(publisher, 'publish_content'):
            return publisher.publish_content(content_data)
        
        logger.warning(f"Publisher {publisher_id} cannot publish content")
        return False


class MisinformationInjectionEvent(SimulationEvent):
    """Event for injecting misinformation."""
    
    def __init__(self, scheduled_step: int, source_id: str, 
                 misinformation_data: Dict[str, Any], **kwargs):
        super().__init__(
            event_id=generate_uuid(),
            event_type="misinformation_injection",
            scheduled_step=scheduled_step,
            priority=EventPriority.HIGH,
            data={
                'source_id': source_id,
                'misinformation_data': misinformation_data
            },
            **kwargs
        )
    
    def _execute_impl(self, simulation) -> bool:
        """Execute misinformation injection."""
        source_id = self.data['source_id']
        misinformation_data = self.data['misinformation_data']
        
        # Get source agent
        source = simulation.agent_registry.get(source_id)
        if not source:
            logger.warning(f"Misinformation source not found: {source_id}")
            return False
        
        # Inject misinformation
        if hasattr(source, 'inject_misinformation'):
            return source.inject_misinformation(misinformation_data)
        
        logger.warning(f"Source {source_id} cannot inject misinformation")
        return False


class InterventionEvent(SimulationEvent):
    """Event for applying interventions."""
    
    def __init__(self, scheduled_step: int, intervention_type: str,
                 intervention_data: Dict[str, Any], **kwargs):
        super().__init__(
            event_id=generate_uuid(),
            event_type="intervention",
            scheduled_step=scheduled_step,
            priority=EventPriority.HIGH,
            data={
                'intervention_type': intervention_type,
                'intervention_data': intervention_data
            },
            **kwargs
        )
    
    def _execute_impl(self, simulation) -> bool:
        """Execute intervention."""
        intervention_type = self.data['intervention_type']
        intervention_data = self.data['intervention_data']
        
        # Apply intervention based on type
        if intervention_type == 'fact_check':
            return self._apply_fact_check(simulation, intervention_data)
        elif intervention_type == 'content_removal':
            return self._apply_content_removal(simulation, intervention_data)
        elif intervention_type == 'account_suspension':
            return self._apply_account_suspension(simulation, intervention_data)
        elif intervention_type == 'algorithm_change':
            return self._apply_algorithm_change(simulation, intervention_data)
        else:
            logger.warning(f"Unknown intervention type: {intervention_type}")
            return False
    
    def _apply_fact_check(self, simulation, data: Dict[str, Any]) -> bool:
        """Apply fact-checking intervention."""
        content_id = data.get('content_id')
        fact_check_result = data.get('fact_check_result')
        
        # Find content and apply fact check
        # This would interact with content management system
        simulation.log_event('fact_check_applied', {
            'content_id': content_id,
            'result': fact_check_result
        })
        
        return True
    
    def _apply_content_removal(self, simulation, data: Dict[str, Any]) -> bool:
        """Apply content removal intervention."""
        content_id = data.get('content_id')
        reason = data.get('reason', 'policy_violation')
        
        # Remove content from simulation
        simulation.log_event('content_removed', {
            'content_id': content_id,
            'reason': reason
        })
        
        return True
    
    def _apply_account_suspension(self, simulation, data: Dict[str, Any]) -> bool:
        """Apply account suspension intervention."""
        agent_id = data.get('agent_id')
        duration = data.get('duration', 24)  # hours
        
        agent = simulation.agent_registry.get(agent_id)
        if agent and hasattr(agent, 'suspend'):
            agent.suspend(duration)
            simulation.log_event('account_suspended', {
                'agent_id': agent_id,
                'duration': duration
            })
            return True
        
        return False
    
    def _apply_algorithm_change(self, simulation, data: Dict[str, Any]) -> bool:
        """Apply algorithm change intervention."""
        platform_id = data.get('platform_id')
        algorithm_params = data.get('algorithm_params', {})
        
        platform = simulation.agent_registry.get(platform_id)
        if platform and hasattr(platform, 'update_algorithm'):
            platform.update_algorithm(algorithm_params)
            simulation.log_event('algorithm_changed', {
                'platform_id': platform_id,
                'params': algorithm_params
            })
            return True
        
        return False


class NetworkChangeEvent(SimulationEvent):
    """Event for network topology changes."""
    
    def __init__(self, scheduled_step: int, change_type: str,
                 network_data: Dict[str, Any], **kwargs):
        super().__init__(
            event_id=generate_uuid(),
            event_type="network_change",
            scheduled_step=scheduled_step,
            data={
                'change_type': change_type,
                'network_data': network_data
            },
            **kwargs
        )
    
    def _execute_impl(self, simulation) -> bool:
        """Execute network change."""
        change_type = self.data['change_type']
        network_data = self.data['network_data']
        
        if change_type == 'add_edge':
            return self._add_network_edge(simulation, network_data)
        elif change_type == 'remove_edge':
            return self._remove_network_edge(simulation, network_data)
        elif change_type == 'add_node':
            return self._add_network_node(simulation, network_data)
        elif change_type == 'remove_node':
            return self._remove_network_node(simulation, network_data)
        else:
            logger.warning(f"Unknown network change type: {change_type}")
            return False
    
    def _add_network_edge(self, simulation, data: Dict[str, Any]) -> bool:
        """Add edge to network."""
        network_type = data.get('network_type', 'social')
        source = data.get('source')
        target = data.get('target')
        attributes = data.get('attributes', {})
        
        if source and target:
            simulation.add_network_edge(network_type, source, target, **attributes)
            return True
        
        return False
    
    def _remove_network_edge(self, simulation, data: Dict[str, Any]) -> bool:
        """Remove edge from network."""
        network_type = data.get('network_type', 'social')
        source = data.get('source')
        target = data.get('target')
        
        if source and target:
            network = simulation.get_network(network_type)
            if network.has_edge(source, target):
                network.remove_edge(source, target)
                return True
        
        return False
    
    def _add_network_node(self, simulation, data: Dict[str, Any]) -> bool:
        """Add node to network."""
        network_type = data.get('network_type', 'social')
        node_id = data.get('node_id')
        attributes = data.get('attributes', {})
        
        if node_id:
            network = simulation.get_network(network_type)
            network.add_node(node_id, **attributes)
            return True
        
        return False
    
    def _remove_network_node(self, simulation, data: Dict[str, Any]) -> bool:
        """Remove node from network."""
        network_type = data.get('network_type', 'social')
        node_id = data.get('node_id')
        
        if node_id:
            network = simulation.get_network(network_type)
            if network.has_node(node_id):
                network.remove_node(node_id)
                return True
        
        return False


class PeriodicEvent(SimulationEvent):
    """Event that repeats at regular intervals."""
    
    def __init__(self, event_type: str, start_step: int, interval: int,
                 end_step: Optional[int] = None, **kwargs):
        super().__init__(
            event_id=generate_uuid(),
            event_type=event_type,
            scheduled_step=start_step,
            **kwargs
        )
        self.interval = interval
        self.end_step = end_step
        self.next_execution = start_step
    
    def _execute_impl(self, simulation) -> bool:
        """Execute periodic event and reschedule if needed."""
        # Execute the periodic action
        success = self._execute_periodic_action(simulation)
        
        # Schedule next execution if within bounds
        if self.end_step is None or self.next_execution + self.interval <= self.end_step:
            self.next_execution += self.interval
            next_event = self._create_next_event()
            simulation.schedule_event(next_event)
        
        return success
    
    @abstractmethod
    def _execute_periodic_action(self, simulation) -> bool:
        """Execute the periodic action."""
        pass
    
    def _create_next_event(self) -> 'PeriodicEvent':
        """Create the next instance of this periodic event."""
        return PeriodicEvent(
            event_type=self.event_type,
            start_step=self.next_execution,
            interval=self.interval,
            end_step=self.end_step,
            priority=self.priority,
            data=self.data.copy()
        )


class MetricsCollectionEvent(PeriodicEvent):
    """Periodic event for collecting metrics."""
    
    def __init__(self, start_step: int = 0, interval: int = 10, **kwargs):
        super().__init__(
            event_type="metrics_collection",
            start_step=start_step,
            interval=interval,
            **kwargs
        )
    
    def _execute_periodic_action(self, simulation) -> bool:
        """Collect simulation metrics."""
        try:
            metrics = simulation.metrics_collector.collect_step_metrics(
                simulation, simulation.current_step
            )
            
            simulation.log_event('metrics_collected', {
                'step': simulation.current_step,
                'metrics': metrics.__dict__ if hasattr(metrics, '__dict__') else str(metrics)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return False


class EventScheduler:
    """Event scheduler for simulation events."""
    
    def __init__(self):
        """Initialize event scheduler."""
        self.event_queue = []
        self.scheduled_events = {}
        self.executed_events = []
        self.cancelled_events = set()
        
        logger.debug("Event scheduler initialized")
    
    def schedule_event(self, event: SimulationEvent) -> None:
        """Schedule an event.
        
        Args:
            event: Event to schedule
        """
        if event.event_id in self.cancelled_events:
            logger.warning(f"Attempted to schedule cancelled event: {event.event_id}")
            return
        
        heapq.heappush(self.event_queue, event)
        self.scheduled_events[event.event_id] = event
        
        logger.debug(f"Event scheduled: {event.event_id} at step {event.scheduled_step}")
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a scheduled event.
        
        Args:
            event_id: ID of event to cancel
            
        Returns:
            True if event was cancelled, False if not found
        """
        if event_id in self.scheduled_events:
            event = self.scheduled_events[event_id]
            event.cancel()
            self.cancelled_events.add(event_id)
            del self.scheduled_events[event_id]
            
            logger.debug(f"Event cancelled: {event_id}")
            return True
        
        return False
    
    def get_events_for_step(self, step: int) -> List[SimulationEvent]:
        """Get all events scheduled for a specific step.
        
        Args:
            step: Simulation step
            
        Returns:
            List of events to execute
        """
        events_to_execute = []
        
        # Process events from queue
        while self.event_queue and self.event_queue[0].scheduled_step <= step:
            event = heapq.heappop(self.event_queue)
            
            # Skip cancelled events
            if event.event_id in self.cancelled_events:
                continue
            
            # Only execute events for current step
            if event.scheduled_step == step:
                events_to_execute.append(event)
                self.executed_events.append(event)
                
                # Remove from scheduled events
                if event.event_id in self.scheduled_events:
                    del self.scheduled_events[event.event_id]
            else:
                # Put back events for future steps
                heapq.heappush(self.event_queue, event)
                break
        
        return events_to_execute
    
    def get_scheduled_events(self) -> List[SimulationEvent]:
        """Get all scheduled events.
        
        Returns:
            List of scheduled events
        """
        return list(self.scheduled_events.values())
    
    def get_executed_events(self) -> List[SimulationEvent]:
        """Get all executed events.
        
        Returns:
            List of executed events
        """
        return self.executed_events.copy()
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_scheduled = len(self.scheduled_events) + len(self.executed_events)
        total_executed = len(self.executed_events)
        total_cancelled = len(self.cancelled_events)
        
        # Count by type
        type_counts = {}
        for event in self.executed_events:
            event_type = event.event_type
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Count by status
        status_counts = {}
        for event in self.executed_events:
            status = event.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_scheduled': total_scheduled,
            'total_executed': total_executed,
            'total_cancelled': total_cancelled,
            'pending': len(self.scheduled_events),
            'type_counts': type_counts,
            'status_counts': status_counts
        }
    
    def clear(self) -> None:
        """Clear all events."""
        self.event_queue.clear()
        self.scheduled_events.clear()
        self.executed_events.clear()
        self.cancelled_events.clear()
        
        logger.debug("Event scheduler cleared")
    
    def export_events(self) -> List[Dict[str, Any]]:
        """Export events for analysis.
        
        Returns:
            List of event dictionaries
        """
        events_data = []
        
        for event in self.executed_events:
            events_data.append({
                'event_id': event.event_id,
                'event_type': event.event_type,
                'scheduled_step': event.scheduled_step,
                'priority': event.priority.value,
                'status': event.status.value,
                'created_at': event.created_at.isoformat(),
                'executed_at': event.executed_at.isoformat() if event.executed_at else None,
                'error_message': event.error_message,
                'data': event.data
            })
        
        return events_data


# Event factory functions
def create_content_publication_event(step: int, publisher_id: str, 
                                   content_data: Dict[str, Any]) -> ContentPublicationEvent:
    """Create a content publication event."""
    return ContentPublicationEvent(step, publisher_id, content_data)


def create_misinformation_injection_event(step: int, source_id: str,
                                         misinformation_data: Dict[str, Any]) -> MisinformationInjectionEvent:
    """Create a misinformation injection event."""
    return MisinformationInjectionEvent(step, source_id, misinformation_data)


def create_intervention_event(step: int, intervention_type: str,
                            intervention_data: Dict[str, Any]) -> InterventionEvent:
    """Create an intervention event."""
    return InterventionEvent(step, intervention_type, intervention_data)


def create_network_change_event(step: int, change_type: str,
                               network_data: Dict[str, Any]) -> NetworkChangeEvent:
    """Create a network change event."""
    return NetworkChangeEvent(step, change_type, network_data)


def create_metrics_collection_event(start_step: int = 0, 
                                   interval: int = 10) -> MetricsCollectionEvent:
    """Create a metrics collection event."""
    return MetricsCollectionEvent(start_step, interval)