"""Network models for NewsGuard Bangladesh simulation.

This module implements network models for social connections, information flow,
influence propagation, and network dynamics in the Bangladesh media ecosystem.
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
import networkx as nx
from scipy import stats
import pandas as pd

from ..utils.logger import get_logger
from ..utils.helpers import (
    calculate_statistics, normalize_scores, exponential_decay,
    sigmoid_function, weighted_random_choice
)

logger = get_logger(__name__)


class NodeType(Enum):
    """Types of nodes in the network."""
    NEWS_OUTLET = "news_outlet"
    READER = "reader"
    PLATFORM = "platform"
    FACT_CHECKER = "fact_checker"
    INFLUENCER = "influencer"
    BOT = "bot"
    GOVERNMENT = "government"
    ORGANIZATION = "organization"


class EdgeType(Enum):
    """Types of edges in the network."""
    FOLLOWS = "follows"
    SHARES = "shares"
    TRUSTS = "trusts"
    COLLABORATES = "collaborates"
    OPPOSES = "opposes"
    INFLUENCES = "influences"
    FACT_CHECKS = "fact_checks"
    MODERATES = "moderates"


class NetworkEventType(Enum):
    """Types of network events."""
    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    INFLUENCE_SPREAD = "influence_spread"
    COMMUNITY_FORMED = "community_formed"
    COMMUNITY_SPLIT = "community_split"


@dataclass
class NetworkNode:
    """Node in the network."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.READER
    name: str = ""
    
    # Node attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Network metrics
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    
    # Influence metrics
    influence_score: float = 0.0
    trust_score: float = 0.5
    credibility_score: float = 0.5
    
    # Activity metrics
    activity_level: float = 0.5
    last_active: datetime = field(default_factory=datetime.now)
    
    # Connections
    followers: Set[str] = field(default_factory=set)
    following: Set[str] = field(default_factory=set)
    
    # Content interaction
    content_shared: List[str] = field(default_factory=list)
    content_created: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.name:
            self.name = f"{self.node_type.value}_{self.node_id[:8]}"


@dataclass
class NetworkEdge:
    """Edge in the network."""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.FOLLOWS
    weight: float = 1.0
    
    # Edge attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    
    # Interaction metrics
    interaction_count: int = 0
    interaction_strength: float = 0.0
    
    # Trust and influence
    trust_level: float = 0.5
    influence_strength: float = 0.0
    
    @property
    def edge_id(self) -> str:
        """Generate edge ID."""
        return f"{self.source}_{self.target}_{self.edge_type.value}"


@dataclass
class NetworkEvent:
    """Network event for tracking changes."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: NetworkEventType = NetworkEventType.NODE_ADDED
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Event details
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    
    # Event metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Impact metrics
    network_impact: float = 0.0
    affected_nodes: List[str] = field(default_factory=list)


class NetworkModel:
    """Base network model for the simulation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize network model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Network graph
        self.graph = nx.DiGraph()
        
        # Node and edge registries
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: Dict[str, NetworkEdge] = {}
        
        # Network metrics
        self.network_metrics: Dict[str, float] = {}
        
        # Event tracking
        self.events: List[NetworkEvent] = []
        
        # Communities
        self.communities: Dict[str, List[str]] = {}
        
        # Model parameters
        self.max_connections = self.config.get('max_connections', 1000)
        self.influence_decay = self.config.get('influence_decay', 0.1)
        self.trust_threshold = self.config.get('trust_threshold', 0.6)
        
        logger.debug("Network model initialized")
    
    def add_node(self, node: NetworkNode) -> bool:
        """Add node to network.
        
        Args:
            node: Node to add
            
        Returns:
            True if node was added successfully
        """
        if node.node_id in self.nodes:
            logger.warning(f"Node already exists: {node.node_id}")
            return False
        
        # Add to registries
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **node.attributes)
        
        # Record event
        event = NetworkEvent(
            event_type=NetworkEventType.NODE_ADDED,
            node_id=node.node_id,
            metadata={'node_type': node.node_type.value, 'name': node.name}
        )
        self.events.append(event)
        
        logger.debug(f"Added node: {node.node_id} ({node.node_type.value})")
        
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from network.
        
        Args:
            node_id: Node to remove
            
        Returns:
            True if node was removed successfully
        """
        if node_id not in self.nodes:
            logger.warning(f"Node not found: {node_id}")
            return False
        
        # Get affected nodes (neighbors)
        affected_nodes = list(self.graph.neighbors(node_id))
        affected_nodes.extend(list(self.graph.predecessors(node_id)))
        
        # Remove edges
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source == node_id or edge.target == node_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
        
        # Remove from graph and registry
        self.graph.remove_node(node_id)
        del self.nodes[node_id]
        
        # Record event
        event = NetworkEvent(
            event_type=NetworkEventType.NODE_REMOVED,
            node_id=node_id,
            affected_nodes=affected_nodes
        )
        self.events.append(event)
        
        logger.debug(f"Removed node: {node_id}")
        
        return True
    
    def add_edge(self, edge: NetworkEdge) -> bool:
        """Add edge to network.
        
        Args:
            edge: Edge to add
            
        Returns:
            True if edge was added successfully
        """
        # Check if nodes exist
        if edge.source not in self.nodes or edge.target not in self.nodes:
            logger.warning(f"Cannot add edge: nodes not found ({edge.source}, {edge.target})")
            return False
        
        edge_id = edge.edge_id
        
        # Check if edge already exists
        if edge_id in self.edges:
            logger.warning(f"Edge already exists: {edge_id}")
            return False
        
        # Add to registries
        self.edges[edge_id] = edge
        self.graph.add_edge(
            edge.source, 
            edge.target, 
            weight=edge.weight,
            edge_type=edge.edge_type.value,
            **edge.attributes
        )
        
        # Update node connections
        source_node = self.nodes[edge.source]
        target_node = self.nodes[edge.target]
        
        if edge.edge_type == EdgeType.FOLLOWS:
            source_node.following.add(edge.target)
            target_node.followers.add(edge.source)
        
        # Record event
        event = NetworkEvent(
            event_type=NetworkEventType.EDGE_ADDED,
            edge_id=edge_id,
            source_node=edge.source,
            target_node=edge.target,
            metadata={'edge_type': edge.edge_type.value, 'weight': edge.weight}
        )
        self.events.append(event)
        
        logger.debug(f"Added edge: {edge.source} -> {edge.target} ({edge.edge_type.value})")
        
        return True
    
    def remove_edge(self, source: str, target: str, edge_type: EdgeType) -> bool:
        """Remove edge from network.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge
            
        Returns:
            True if edge was removed successfully
        """
        edge_id = f"{source}_{target}_{edge_type.value}"
        
        if edge_id not in self.edges:
            logger.warning(f"Edge not found: {edge_id}")
            return False
        
        # Remove from graph and registry
        self.graph.remove_edge(source, target)
        del self.edges[edge_id]
        
        # Update node connections
        if edge_type == EdgeType.FOLLOWS:
            if source in self.nodes:
                self.nodes[source].following.discard(target)
            if target in self.nodes:
                self.nodes[target].followers.discard(source)
        
        # Record event
        event = NetworkEvent(
            event_type=NetworkEventType.EDGE_REMOVED,
            edge_id=edge_id,
            source_node=source,
            target_node=target
        )
        self.events.append(event)
        
        logger.debug(f"Removed edge: {edge_id}")
        
        return True
    
    def calculate_centrality_metrics(self) -> None:
        """Calculate centrality metrics for all nodes."""
        if len(self.graph.nodes()) == 0:
            return
        
        try:
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # For directed graphs
            if self.graph.is_directed():
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
                pagerank = nx.pagerank(self.graph)
            else:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph)
                pagerank = {}
            
            # Update node metrics
            for node_id in self.nodes:
                if node_id in degree_centrality:
                    self.nodes[node_id].degree_centrality = degree_centrality[node_id]
                    self.nodes[node_id].betweenness_centrality = betweenness_centrality[node_id]
                    self.nodes[node_id].closeness_centrality = closeness_centrality[node_id]
                    self.nodes[node_id].eigenvector_centrality = eigenvector_centrality.get(node_id, 0.0)
                    self.nodes[node_id].pagerank = pagerank.get(node_id, 0.0)
            
            logger.debug("Calculated centrality metrics")
            
        except Exception as e:
            logger.error(f"Error calculating centrality metrics: {e}")
    
    def detect_communities(self, method: str = 'louvain') -> Dict[str, List[str]]:
        """Detect communities in the network.
        
        Args:
            method: Community detection method
            
        Returns:
            Dictionary mapping community IDs to node lists
        """
        if len(self.graph.nodes()) < 2:
            return {}
        
        try:
            if method == 'louvain':
                # Convert to undirected for community detection
                undirected_graph = self.graph.to_undirected()
                communities = nx.community.louvain_communities(undirected_graph)
            elif method == 'greedy_modularity':
                undirected_graph = self.graph.to_undirected()
                communities = nx.community.greedy_modularity_communities(undirected_graph)
            else:
                logger.warning(f"Unknown community detection method: {method}")
                return {}
            
            # Convert to dictionary format
            community_dict = {}
            for i, community in enumerate(communities):
                community_id = f"community_{i}"
                community_dict[community_id] = list(community)
            
            self.communities = community_dict
            
            logger.debug(f"Detected {len(community_dict)} communities using {method}")
            
            return community_dict
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return {}
    
    def calculate_influence_scores(self) -> None:
        """Calculate influence scores for all nodes."""
        for node_id, node in self.nodes.items():
            # Base influence from centrality metrics
            centrality_influence = (
                0.3 * node.degree_centrality +
                0.2 * node.betweenness_centrality +
                0.2 * node.closeness_centrality +
                0.3 * node.pagerank
            )
            
            # Node type influence multiplier
            type_multipliers = {
                NodeType.INFLUENCER: 1.5,
                NodeType.NEWS_OUTLET: 1.3,
                NodeType.FACT_CHECKER: 1.2,
                NodeType.PLATFORM: 1.4,
                NodeType.READER: 1.0,
                NodeType.BOT: 0.5,
                NodeType.GOVERNMENT: 1.1,
                NodeType.ORGANIZATION: 1.1
            }
            
            type_multiplier = type_multipliers.get(node.node_type, 1.0)
            
            # Activity influence
            activity_influence = node.activity_level
            
            # Trust influence
            trust_influence = node.trust_score
            
            # Calculate final influence score
            influence_score = (
                centrality_influence * type_multiplier * 
                activity_influence * trust_influence
            )
            
            node.influence_score = min(1.0, influence_score)
    
    def simulate_information_spread(self, 
                                  source_node: str,
                                  content_id: str,
                                  spread_probability: float = 0.1) -> List[str]:
        """Simulate information spread through the network.
        
        Args:
            source_node: Node that starts the spread
            content_id: Content being spread
            spread_probability: Base probability of spread
            
        Returns:
            List of nodes that received the information
        """
        if source_node not in self.nodes:
            logger.warning(f"Source node not found: {source_node}")
            return []
        
        # Track spread
        infected = {source_node}
        newly_infected = {source_node}
        spread_log = [source_node]
        
        # Simulate spread in waves
        max_waves = 10
        for wave in range(max_waves):
            if not newly_infected:
                break
            
            next_wave = set()
            
            for current_node in newly_infected:
                # Get neighbors
                neighbors = list(self.graph.neighbors(current_node))
                
                for neighbor in neighbors:
                    if neighbor in infected:
                        continue
                    
                    # Calculate spread probability
                    edge_data = self.graph.get_edge_data(current_node, neighbor, {})
                    edge_weight = edge_data.get('weight', 1.0)
                    
                    # Node influence affects spread
                    source_influence = self.nodes[current_node].influence_score
                    target_receptivity = self.nodes[neighbor].trust_score
                    
                    # Calculate final probability
                    final_probability = (
                        spread_probability * 
                        edge_weight * 
                        source_influence * 
                        target_receptivity
                    )
                    
                    # Determine if spread occurs
                    if random.random() < final_probability:
                        next_wave.add(neighbor)
                        infected.add(neighbor)
                        spread_log.append(neighbor)
                        
                        # Record content sharing
                        self.nodes[neighbor].content_shared.append(content_id)
            
            newly_infected = next_wave
        
        # Record spread event
        event = NetworkEvent(
            event_type=NetworkEventType.INFLUENCE_SPREAD,
            source_node=source_node,
            affected_nodes=spread_log,
            metadata={
                'content_id': content_id,
                'spread_size': len(spread_log),
                'waves': wave + 1
            }
        )
        self.events.append(event)
        
        logger.debug(f"Information spread from {source_node}: {len(spread_log)} nodes reached")
        
        return spread_log
    
    def calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate overall network metrics.
        
        Returns:
            Dictionary of network metrics
        """
        if len(self.graph.nodes()) == 0:
            return {}
        
        try:
            # Basic metrics
            num_nodes = len(self.graph.nodes())
            num_edges = len(self.graph.edges())
            density = nx.density(self.graph)
            
            # Connectivity metrics
            if self.graph.is_directed():
                is_connected = nx.is_weakly_connected(self.graph)
                components = list(nx.weakly_connected_components(self.graph))
            else:
                is_connected = nx.is_connected(self.graph)
                components = list(nx.connected_components(self.graph))
            
            num_components = len(components)
            largest_component_size = max(len(comp) for comp in components) if components else 0
            
            # Clustering
            clustering_coefficient = nx.average_clustering(self.graph.to_undirected())
            
            # Path metrics
            if is_connected:
                avg_path_length = nx.average_shortest_path_length(self.graph.to_undirected())
                diameter = nx.diameter(self.graph.to_undirected())
            else:
                avg_path_length = 0
                diameter = 0
            
            # Degree distribution
            degrees = [d for n, d in self.graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            degree_variance = np.var(degrees) if degrees else 0
            
            # Assortativity
            try:
                assortativity = nx.degree_assortativity_coefficient(self.graph)
            except:
                assortativity = 0
            
            metrics = {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': density,
                'is_connected': is_connected,
                'num_components': num_components,
                'largest_component_size': largest_component_size,
                'clustering_coefficient': clustering_coefficient,
                'avg_path_length': avg_path_length,
                'diameter': diameter,
                'avg_degree': avg_degree,
                'degree_variance': degree_variance,
                'assortativity': assortativity
            }
            
            self.network_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
            return {}
    
    def get_node_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get neighbors of a node.
        
        Args:
            node_id: Node ID
            edge_type: Filter by edge type
            
        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        
        # Get all neighbors
        for neighbor in self.graph.neighbors(node_id):
            if edge_type is None:
                neighbors.append(neighbor)
            else:
                # Check edge type
                edge_data = self.graph.get_edge_data(node_id, neighbor, {})
                if edge_data.get('edge_type') == edge_type.value:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of node IDs in the shortest path
        """
        try:
            path = nx.shortest_path(self.graph.to_undirected(), source, target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_influential_nodes(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get most influential nodes.
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            List of (node_id, influence_score) tuples
        """
        # Sort nodes by influence score
        influential_nodes = sorted(
            [(node_id, node.influence_score) for node_id, node in self.nodes.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return influential_nodes[:limit]
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics.
        
        Returns:
            Dictionary of network statistics
        """
        # Basic counts
        node_type_counts = {}
        for node in self.nodes.values():
            node_type = node.node_type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        edge_type_counts = {}
        for edge in self.edges.values():
            edge_type = edge.edge_type.value
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        # Influence statistics
        influence_scores = [node.influence_score for node in self.nodes.values()]
        trust_scores = [node.trust_score for node in self.nodes.values()]
        
        # Activity statistics
        activity_levels = [node.activity_level for node in self.nodes.values()]
        
        # Network metrics
        network_metrics = self.calculate_network_metrics()
        
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_type_distribution': node_type_counts,
            'edge_type_distribution': edge_type_counts,
            'avg_influence_score': np.mean(influence_scores) if influence_scores else 0,
            'avg_trust_score': np.mean(trust_scores) if trust_scores else 0,
            'avg_activity_level': np.mean(activity_levels) if activity_levels else 0,
            'total_communities': len(self.communities),
            'total_events': len(self.events),
            'network_metrics': network_metrics
        }
        
        return stats
    
    def export_graph(self, format: str = 'gexf') -> str:
        """Export network graph to file format.
        
        Args:
            format: Export format (gexf, graphml, gml)
            
        Returns:
            Exported graph data as string
        """
        try:
            if format == 'gexf':
                import io
                buffer = io.StringIO()
                nx.write_gexf(self.graph, buffer)
                return buffer.getvalue()
            elif format == 'graphml':
                import io
                buffer = io.StringIO()
                nx.write_graphml(self.graph, buffer)
                return buffer.getvalue()
            elif format == 'gml':
                import io
                buffer = io.StringIO()
                nx.write_gml(self.graph, buffer)
                return buffer.getvalue()
            else:
                logger.warning(f"Unsupported export format: {format}")
                return ""
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            return ""


class SocialNetworkModel(NetworkModel):
    """Specialized model for social networks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize social network model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Social network specific parameters
        self.homophily_strength = self.config.get('homophily_strength', 0.7)
        self.preferential_attachment = self.config.get('preferential_attachment', True)
        self.small_world_probability = self.config.get('small_world_probability', 0.1)
        
        logger.debug("Social network model initialized")
    
    def generate_social_network(self, 
                              num_nodes: int,
                              network_type: str = 'scale_free') -> None:
        """Generate a social network.
        
        Args:
            num_nodes: Number of nodes to generate
            network_type: Type of network (scale_free, small_world, random)
        """
        logger.info(f"Generating {network_type} social network with {num_nodes} nodes")
        
        if network_type == 'scale_free':
            self._generate_scale_free_network(num_nodes)
        elif network_type == 'small_world':
            self._generate_small_world_network(num_nodes)
        elif network_type == 'random':
            self._generate_random_network(num_nodes)
        else:
            logger.warning(f"Unknown network type: {network_type}")
            self._generate_scale_free_network(num_nodes)
        
        # Add node attributes
        self._assign_node_attributes()
        
        # Calculate initial metrics
        self.calculate_centrality_metrics()
        self.calculate_influence_scores()
        
        logger.info(f"Generated social network: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _generate_scale_free_network(self, num_nodes: int) -> None:
        """Generate scale-free network using preferential attachment."""
        # Use NetworkX Barabási-Albert model
        m = min(3, num_nodes // 2)  # Number of edges to attach from new node
        ba_graph = nx.barabasi_albert_graph(num_nodes, m)
        
        # Convert to directed graph
        self.graph = ba_graph.to_directed()
        
        # Create nodes and edges
        for i in range(num_nodes):
            node = NetworkNode(
                node_id=str(i),
                node_type=self._random_node_type(),
                name=f"user_{i}"
            )
            self.nodes[str(i)] = node
        
        for source, target in self.graph.edges():
            edge = NetworkEdge(
                source=str(source),
                target=str(target),
                edge_type=EdgeType.FOLLOWS,
                weight=random.uniform(0.1, 1.0)
            )
            self.edges[edge.edge_id] = edge
    
    def _generate_small_world_network(self, num_nodes: int) -> None:
        """Generate small-world network using Watts-Strogatz model."""
        k = min(6, num_nodes // 2)  # Each node connected to k nearest neighbors
        p = self.small_world_probability  # Probability of rewiring
        
        ws_graph = nx.watts_strogatz_graph(num_nodes, k, p)
        self.graph = ws_graph.to_directed()
        
        # Create nodes and edges
        for i in range(num_nodes):
            node = NetworkNode(
                node_id=str(i),
                node_type=self._random_node_type(),
                name=f"user_{i}"
            )
            self.nodes[str(i)] = node
        
        for source, target in self.graph.edges():
            edge = NetworkEdge(
                source=str(source),
                target=str(target),
                edge_type=EdgeType.FOLLOWS,
                weight=random.uniform(0.1, 1.0)
            )
            self.edges[edge.edge_id] = edge
    
    def _generate_random_network(self, num_nodes: int) -> None:
        """Generate random network using Erdős-Rényi model."""
        p = 0.1  # Probability of edge creation
        er_graph = nx.erdos_renyi_graph(num_nodes, p, directed=True)
        self.graph = er_graph
        
        # Create nodes and edges
        for i in range(num_nodes):
            node = NetworkNode(
                node_id=str(i),
                node_type=self._random_node_type(),
                name=f"user_{i}"
            )
            self.nodes[str(i)] = node
        
        for source, target in self.graph.edges():
            edge = NetworkEdge(
                source=str(source),
                target=str(target),
                edge_type=EdgeType.FOLLOWS,
                weight=random.uniform(0.1, 1.0)
            )
            self.edges[edge.edge_id] = edge
    
    def _random_node_type(self) -> NodeType:
        """Generate random node type with realistic distribution."""
        node_types = [
            NodeType.READER,
            NodeType.NEWS_OUTLET,
            NodeType.INFLUENCER,
            NodeType.FACT_CHECKER,
            NodeType.BOT,
            NodeType.ORGANIZATION
        ]
        
        # Realistic distribution
        weights = [0.7, 0.1, 0.1, 0.02, 0.05, 0.03]
        
        return weighted_random_choice(node_types, weights)
    
    def _assign_node_attributes(self) -> None:
        """Assign attributes to nodes based on their type."""
        for node in self.nodes.values():
            # Base attributes
            node.activity_level = random.uniform(0.1, 1.0)
            node.trust_score = random.uniform(0.3, 0.9)
            node.credibility_score = random.uniform(0.4, 0.9)
            
            # Type-specific attributes
            if node.node_type == NodeType.NEWS_OUTLET:
                node.credibility_score = random.uniform(0.6, 0.95)
                node.attributes['circulation'] = random.randint(1000, 1000000)
                node.attributes['established_year'] = random.randint(1990, 2020)
                
            elif node.node_type == NodeType.INFLUENCER:
                node.activity_level = random.uniform(0.7, 1.0)
                node.attributes['follower_count'] = random.randint(10000, 5000000)
                node.attributes['engagement_rate'] = random.uniform(0.02, 0.15)
                
            elif node.node_type == NodeType.FACT_CHECKER:
                node.credibility_score = random.uniform(0.8, 0.98)
                node.trust_score = random.uniform(0.7, 0.95)
                node.attributes['expertise'] = random.choice(['politics', 'health', 'science', 'general'])
                
            elif node.node_type == NodeType.BOT:
                node.activity_level = random.uniform(0.8, 1.0)
                node.credibility_score = random.uniform(0.1, 0.4)
                node.trust_score = random.uniform(0.1, 0.3)
                node.attributes['bot_type'] = random.choice(['spam', 'misinformation', 'engagement'])
                
            elif node.node_type == NodeType.READER:
                # Demographic attributes
                node.attributes['age_group'] = random.choice(['18-25', '26-35', '36-45', '46-55', '55+'])
                node.attributes['education'] = random.choice(['high_school', 'bachelor', 'master', 'phd'])
                node.attributes['location'] = random.choice(['dhaka', 'chittagong', 'sylhet', 'rajshahi', 'khulna'])
                node.attributes['interests'] = random.sample(
                    ['politics', 'sports', 'entertainment', 'technology', 'health', 'business'],
                    random.randint(1, 3)
                )
    
    def simulate_homophily(self) -> None:
        """Simulate homophily - tendency for similar nodes to connect."""
        # Add edges between similar nodes
        nodes_list = list(self.nodes.values())
        
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                # Calculate similarity
                similarity = self._calculate_node_similarity(node1, node2)
                
                # Probability of connection based on similarity
                connection_prob = similarity * self.homophily_strength * 0.1
                
                if random.random() < connection_prob:
                    # Add bidirectional connection
                    edge1 = NetworkEdge(
                        source=node1.node_id,
                        target=node2.node_id,
                        edge_type=EdgeType.FOLLOWS,
                        weight=similarity
                    )
                    
                    edge2 = NetworkEdge(
                        source=node2.node_id,
                        target=node1.node_id,
                        edge_type=EdgeType.FOLLOWS,
                        weight=similarity
                    )
                    
                    self.add_edge(edge1)
                    self.add_edge(edge2)
    
    def _calculate_node_similarity(self, node1: NetworkNode, node2: NetworkNode) -> float:
        """Calculate similarity between two nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Similarity score (0-1)
        """
        similarity_factors = []
        
        # Node type similarity
        if node1.node_type == node2.node_type:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Attribute similarity
        if 'interests' in node1.attributes and 'interests' in node2.attributes:
            interests1 = set(node1.attributes['interests'])
            interests2 = set(node2.attributes['interests'])
            
            if interests1 and interests2:
                interest_similarity = len(interests1 & interests2) / len(interests1 | interests2)
                similarity_factors.append(interest_similarity)
        
        if 'location' in node1.attributes and 'location' in node2.attributes:
            if node1.attributes['location'] == node2.attributes['location']:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
        
        # Trust and credibility similarity
        trust_similarity = 1 - abs(node1.trust_score - node2.trust_score)
        credibility_similarity = 1 - abs(node1.credibility_score - node2.credibility_score)
        
        similarity_factors.extend([trust_similarity, credibility_similarity])
        
        # Calculate average similarity
        return np.mean(similarity_factors) if similarity_factors else 0.0


class InfluenceModel:
    """Model for influence propagation in networks."""
    
    def __init__(self, network_model: NetworkModel, config: Dict[str, Any] = None):
        """Initialize influence model.
        
        Args:
            network_model: Network model to operate on
            config: Model configuration
        """
        self.network = network_model
        self.config = config or {}
        
        # Influence parameters
        self.influence_threshold = self.config.get('influence_threshold', 0.1)
        self.decay_rate = self.config.get('decay_rate', 0.1)
        self.max_iterations = self.config.get('max_iterations', 100)
        
        # Influence tracking
        self.influence_history: List[Dict[str, float]] = []
        
        logger.debug("Influence model initialized")
    
    def calculate_influence_cascade(self, 
                                  initial_nodes: List[str],
                                  influence_type: str = 'linear') -> Dict[str, float]:
        """Calculate influence cascade from initial nodes.
        
        Args:
            initial_nodes: Nodes that start the influence
            influence_type: Type of influence model (linear, threshold, independent)
            
        Returns:
            Dictionary mapping node IDs to final influence values
        """
        if influence_type == 'linear':
            return self._linear_threshold_model(initial_nodes)
        elif influence_type == 'threshold':
            return self._threshold_model(initial_nodes)
        elif influence_type == 'independent':
            return self._independent_cascade_model(initial_nodes)
        else:
            logger.warning(f"Unknown influence type: {influence_type}")
            return self._linear_threshold_model(initial_nodes)
    
    def _linear_threshold_model(self, initial_nodes: List[str]) -> Dict[str, float]:
        """Linear threshold model for influence propagation."""
        # Initialize influence values
        influence = {node_id: 0.0 for node_id in self.network.nodes}
        
        # Set initial influence
        for node_id in initial_nodes:
            if node_id in influence:
                influence[node_id] = 1.0
        
        # Iterative influence propagation
        for iteration in range(self.max_iterations):
            new_influence = influence.copy()
            
            for node_id in self.network.nodes:
                if node_id in initial_nodes:
                    continue
                
                # Calculate influence from neighbors
                total_influence = 0.0
                total_weight = 0.0
                
                for predecessor in self.network.graph.predecessors(node_id):
                    edge_data = self.network.graph.get_edge_data(predecessor, node_id, {})
                    edge_weight = edge_data.get('weight', 1.0)
                    
                    neighbor_influence = influence[predecessor]
                    total_influence += neighbor_influence * edge_weight
                    total_weight += edge_weight
                
                # Update influence
                if total_weight > 0:
                    new_influence[node_id] = min(1.0, total_influence / total_weight)
            
            # Check convergence
            max_change = max(abs(new_influence[node] - influence[node]) 
                           for node in influence)
            
            influence = new_influence
            
            if max_change < 0.001:  # Convergence threshold
                break
        
        return influence
    
    def _threshold_model(self, initial_nodes: List[str]) -> Dict[str, float]:
        """Threshold model for influence propagation."""
        # Node states: 0 = inactive, 1 = active
        active = {node_id: 0 for node_id in self.network.nodes}
        
        # Activate initial nodes
        for node_id in initial_nodes:
            if node_id in active:
                active[node_id] = 1
        
        # Assign random thresholds
        thresholds = {node_id: random.uniform(0.1, 0.8) 
                     for node_id in self.network.nodes}
        
        # Iterative activation
        changed = True
        iteration = 0
        
        while changed and iteration < self.max_iterations:
            changed = False
            iteration += 1
            
            for node_id in self.network.nodes:
                if active[node_id] == 1:  # Already active
                    continue
                
                # Calculate influence from active neighbors
                active_influence = 0.0
                total_weight = 0.0
                
                for predecessor in self.network.graph.predecessors(node_id):
                    if active[predecessor] == 1:
                        edge_data = self.network.graph.get_edge_data(predecessor, node_id, {})
                        edge_weight = edge_data.get('weight', 1.0)
                        
                        active_influence += edge_weight
                        total_weight += edge_weight
                
                # Normalize influence
                if total_weight > 0:
                    normalized_influence = active_influence / total_weight
                    
                    # Check if threshold is exceeded
                    if normalized_influence >= thresholds[node_id]:
                        active[node_id] = 1
                        changed = True
        
        # Convert to influence scores
        influence = {node_id: float(active[node_id]) for node_id in active}
        
        return influence
    
    def _independent_cascade_model(self, initial_nodes: List[str]) -> Dict[str, float]:
        """Independent cascade model for influence propagation."""
        # Track activation
        activated = set(initial_nodes)
        newly_activated = set(initial_nodes)
        
        # Iterative cascade
        iteration = 0
        while newly_activated and iteration < self.max_iterations:
            iteration += 1
            next_activated = set()
            
            for active_node in newly_activated:
                # Try to activate neighbors
                for neighbor in self.network.graph.neighbors(active_node):
                    if neighbor in activated:
                        continue
                    
                    # Calculate activation probability
                    edge_data = self.network.graph.get_edge_data(active_node, neighbor, {})
                    edge_weight = edge_data.get('weight', 1.0)
                    
                    # Node influence affects activation probability
                    source_influence = self.network.nodes[active_node].influence_score
                    activation_prob = edge_weight * source_influence * 0.1
                    
                    # Attempt activation
                    if random.random() < activation_prob:
                        next_activated.add(neighbor)
                        activated.add(neighbor)
            
            newly_activated = next_activated
        
        # Convert to influence scores
        influence = {node_id: 1.0 if node_id in activated else 0.0 
                    for node_id in self.network.nodes}
        
        return influence
    
    def analyze_influence_patterns(self, 
                                 influence_results: Dict[str, float]) -> Dict[str, Any]:
        """Analyze patterns in influence propagation.
        
        Args:
            influence_results: Results from influence calculation
            
        Returns:
            Analysis results
        """
        influenced_nodes = [node for node, influence in influence_results.items() 
                          if influence > self.influence_threshold]
        
        # Calculate statistics
        influence_values = list(influence_results.values())
        
        analysis = {
            'total_influenced': len(influenced_nodes),
            'influence_ratio': len(influenced_nodes) / len(influence_results),
            'avg_influence': np.mean(influence_values),
            'max_influence': np.max(influence_values),
            'influence_variance': np.var(influence_values),
            'influenced_nodes': influenced_nodes
        }
        
        # Analyze by node type
        node_type_influence = {}
        for node_id, influence_value in influence_results.items():
            if node_id in self.network.nodes:
                node_type = self.network.nodes[node_id].node_type.value
                if node_type not in node_type_influence:
                    node_type_influence[node_type] = []
                node_type_influence[node_type].append(influence_value)
        
        # Calculate average influence by node type
        avg_influence_by_type = {}
        for node_type, influences in node_type_influence.items():
            avg_influence_by_type[node_type] = np.mean(influences)
        
        analysis['influence_by_node_type'] = avg_influence_by_type
        
        return analysis


# Export main classes
__all__ = [
    'NetworkModel', 'SocialNetworkModel', 'InfluenceModel',
    'NetworkNode', 'NetworkEdge', 'NetworkEvent',
    'NodeType', 'EdgeType', 'NetworkEventType'
]