"""Metrics collection and analysis for NewsGuard Bangladesh simulation."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from statistics import mean, median, stdev
import math

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx

from ..utils.logger import get_logger
from ..utils.helpers import calculate_statistics, normalize_scores

logger = get_logger(__name__)


@dataclass
class ContentMetrics:
    """Metrics related to content in the simulation."""
    total_content: int = 0
    misinformation_content: int = 0
    fact_checked_content: int = 0
    removed_content: int = 0
    viral_content: int = 0
    
    # Content quality metrics
    avg_credibility_score: float = 0.0
    avg_engagement_rate: float = 0.0
    avg_sharing_rate: float = 0.0
    
    # Content spread metrics
    total_shares: int = 0
    total_likes: int = 0
    total_comments: int = 0
    
    # Misinformation specific
    misinformation_reach: int = 0
    misinformation_engagement: float = 0.0
    fact_check_effectiveness: float = 0.0
    
    @property
    def misinformation_ratio(self) -> float:
        """Calculate ratio of misinformation to total content."""
        if self.total_content == 0:
            return 0.0
        return self.misinformation_content / self.total_content
    
    @property
    def fact_check_coverage(self) -> float:
        """Calculate fact-check coverage ratio."""
        if self.misinformation_content == 0:
            return 0.0
        return self.fact_checked_content / self.misinformation_content


@dataclass
class AgentMetrics:
    """Metrics related to agents in the simulation."""
    total_agents: int = 0
    active_agents: int = 0
    suspended_agents: int = 0
    
    # Agent type counts
    news_outlets: int = 0
    readers: int = 0
    platforms: int = 0
    fact_checkers: int = 0
    influencers: int = 0
    
    # Agent behavior metrics
    avg_trust_score: float = 0.0
    avg_activity_level: float = 0.0
    avg_influence_score: float = 0.0
    
    # Agent interactions
    total_interactions: int = 0
    avg_interactions_per_agent: float = 0.0
    
    @property
    def agent_activity_rate(self) -> float:
        """Calculate agent activity rate."""
        if self.total_agents == 0:
            return 0.0
        return self.active_agents / self.total_agents


@dataclass
class NetworkMetrics:
    """Metrics related to network structure and dynamics."""
    # Social network metrics
    social_nodes: int = 0
    social_edges: int = 0
    social_density: float = 0.0
    social_clustering: float = 0.0
    social_avg_path_length: float = 0.0
    
    # Information network metrics
    info_nodes: int = 0
    info_edges: int = 0
    info_density: float = 0.0
    
    # Economic network metrics
    economic_nodes: int = 0
    economic_edges: int = 0
    economic_density: float = 0.0
    
    # Network dynamics
    network_changes: int = 0
    edge_additions: int = 0
    edge_removals: int = 0
    
    # Centrality measures
    max_degree_centrality: float = 0.0
    max_betweenness_centrality: float = 0.0
    max_closeness_centrality: float = 0.0
    
    # Information flow metrics
    information_cascade_count: int = 0
    avg_cascade_size: float = 0.0
    max_cascade_size: int = 0


@dataclass
class EconomicMetrics:
    """Metrics related to economic aspects of the simulation."""
    # Revenue metrics
    total_revenue: float = 0.0
    advertising_revenue: float = 0.0
    subscription_revenue: float = 0.0
    
    # Cost metrics
    total_costs: float = 0.0
    content_production_costs: float = 0.0
    fact_checking_costs: float = 0.0
    moderation_costs: float = 0.0
    
    # Market metrics
    market_concentration: float = 0.0
    competition_index: float = 0.0
    
    # Economic impact of misinformation
    misinformation_economic_impact: float = 0.0
    intervention_costs: float = 0.0
    
    @property
    def profit_margin(self) -> float:
        """Calculate overall profit margin."""
        if self.total_revenue == 0:
            return 0.0
        return (self.total_revenue - self.total_costs) / self.total_revenue


@dataclass
class InterventionMetrics:
    """Metrics related to interventions and their effectiveness."""
    total_interventions: int = 0
    fact_check_interventions: int = 0
    content_removal_interventions: int = 0
    account_suspension_interventions: int = 0
    algorithm_change_interventions: int = 0
    
    # Effectiveness metrics
    intervention_success_rate: float = 0.0
    avg_intervention_impact: float = 0.0
    
    # Timing metrics
    avg_response_time: float = 0.0  # Time from misinformation to intervention
    
    # Cost-effectiveness
    cost_per_intervention: float = 0.0
    roi_interventions: float = 0.0  # Return on investment


@dataclass
class SimulationMetrics:
    """Comprehensive simulation metrics."""
    step: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Component metrics
    content: ContentMetrics = field(default_factory=ContentMetrics)
    agents: AgentMetrics = field(default_factory=AgentMetrics)
    network: NetworkMetrics = field(default_factory=NetworkMetrics)
    economic: EconomicMetrics = field(default_factory=EconomicMetrics)
    interventions: InterventionMetrics = field(default_factory=InterventionMetrics)
    
    # Overall simulation health
    simulation_health_score: float = 0.0
    information_quality_index: float = 0.0
    social_cohesion_index: float = 0.0
    
    # Performance metrics
    step_execution_time: float = 0.0
    memory_usage: float = 0.0
    
    def calculate_health_score(self) -> float:
        """Calculate overall simulation health score."""
        # Combine various metrics into a health score (0-100)
        factors = []
        
        # Content quality factor (higher is better)
        if self.content.total_content > 0:
            content_quality = (1 - self.content.misinformation_ratio) * 100
            factors.append(content_quality)
        
        # Agent activity factor
        agent_activity = self.agents.agent_activity_rate * 100
        factors.append(agent_activity)
        
        # Network connectivity factor
        if self.network.social_nodes > 0:
            connectivity = min(self.network.social_density * 100, 100)
            factors.append(connectivity)
        
        # Economic stability factor
        if self.economic.total_revenue > 0:
            economic_health = max(0, self.economic.profit_margin * 100)
            factors.append(economic_health)
        
        # Intervention effectiveness factor
        if self.interventions.total_interventions > 0:
            intervention_effectiveness = self.interventions.intervention_success_rate * 100
            factors.append(intervention_effectiveness)
        
        if factors:
            self.simulation_health_score = mean(factors)
        else:
            self.simulation_health_score = 50.0  # Neutral score
        
        return self.simulation_health_score


class MetricsCollector:
    """Collects and manages simulation metrics."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize metrics collector.
        
        Args:
            history_size: Maximum number of historical metrics to keep
        """
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.current_metrics = SimulationMetrics()
        
        # Aggregated metrics
        self.aggregated_metrics = {}
        
        # Performance tracking
        self.collection_times = deque(maxlen=100)
        
        logger.debug("Metrics collector initialized")
    
    def collect_step_metrics(self, simulation, step: int) -> SimulationMetrics:
        """Collect metrics for a simulation step.
        
        Args:
            simulation: Simulation engine instance
            step: Current simulation step
            
        Returns:
            Collected metrics
        """
        start_time = time.time()
        
        metrics = SimulationMetrics(step=step)
        
        # Collect content metrics
        metrics.content = self._collect_content_metrics(simulation)
        
        # Collect agent metrics
        metrics.agents = self._collect_agent_metrics(simulation)
        
        # Collect network metrics
        metrics.network = self._collect_network_metrics(simulation)
        
        # Collect economic metrics
        metrics.economic = self._collect_economic_metrics(simulation)
        
        # Collect intervention metrics
        metrics.interventions = self._collect_intervention_metrics(simulation)
        
        # Calculate derived metrics
        metrics.calculate_health_score()
        metrics.information_quality_index = self._calculate_information_quality(metrics)
        metrics.social_cohesion_index = self._calculate_social_cohesion(metrics)
        
        # Performance metrics
        collection_time = time.time() - start_time
        metrics.step_execution_time = collection_time
        self.collection_times.append(collection_time)
        
        # Store metrics
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Update aggregated metrics
        self._update_aggregated_metrics(metrics)
        
        logger.debug(f"Metrics collected for step {step} in {collection_time:.4f}s")
        
        return metrics
    
    def _collect_content_metrics(self, simulation) -> ContentMetrics:
        """Collect content-related metrics."""
        metrics = ContentMetrics()
        
        # Count content by type
        content_counts = defaultdict(int)
        total_engagement = 0
        total_credibility = 0
        total_shares = 0
        
        # This would iterate through actual content in the simulation
        # For now, we'll use placeholder logic
        
        # Get content from agents
        for agent in simulation.agent_registry.values():
            if hasattr(agent, 'content_produced'):
                agent_content = getattr(agent, 'content_produced', [])
                for content in agent_content:
                    metrics.total_content += 1
                    
                    if content.get('is_misinformation', False):
                        metrics.misinformation_content += 1
                    
                    if content.get('fact_checked', False):
                        metrics.fact_checked_content += 1
                    
                    if content.get('removed', False):
                        metrics.removed_content += 1
                    
                    # Aggregate metrics
                    total_credibility += content.get('credibility_score', 0.5)
                    total_engagement += content.get('engagement_rate', 0.0)
                    total_shares += content.get('share_count', 0)
        
        # Calculate averages
        if metrics.total_content > 0:
            metrics.avg_credibility_score = total_credibility / metrics.total_content
            metrics.avg_engagement_rate = total_engagement / metrics.total_content
        
        metrics.total_shares = total_shares
        
        return metrics
    
    def _collect_agent_metrics(self, simulation) -> AgentMetrics:
        """Collect agent-related metrics."""
        metrics = AgentMetrics()
        
        total_trust = 0
        total_activity = 0
        total_influence = 0
        total_interactions = 0
        
        # Count agents by type and collect metrics
        for agent in simulation.agent_registry.values():
            metrics.total_agents += 1
            
            # Count by type
            agent_type = type(agent).__name__.lower()
            if 'newsoutlet' in agent_type:
                metrics.news_outlets += 1
            elif 'reader' in agent_type:
                metrics.readers += 1
            elif 'platform' in agent_type:
                metrics.platforms += 1
            elif 'factchecker' in agent_type:
                metrics.fact_checkers += 1
            elif 'influencer' in agent_type:
                metrics.influencers += 1
            
            # Check if agent is active
            if getattr(agent, 'is_active', True):
                metrics.active_agents += 1
            
            # Check if agent is suspended
            if getattr(agent, 'is_suspended', False):
                metrics.suspended_agents += 1
            
            # Aggregate behavioral metrics
            total_trust += getattr(agent, 'trust_score', 0.5)
            total_activity += getattr(agent, 'activity_level', 0.5)
            total_influence += getattr(agent, 'influence_score', 0.5)
            total_interactions += getattr(agent, 'interaction_count', 0)
        
        # Calculate averages
        if metrics.total_agents > 0:
            metrics.avg_trust_score = total_trust / metrics.total_agents
            metrics.avg_activity_level = total_activity / metrics.total_agents
            metrics.avg_influence_score = total_influence / metrics.total_agents
            metrics.avg_interactions_per_agent = total_interactions / metrics.total_agents
        
        metrics.total_interactions = total_interactions
        
        return metrics
    
    def _collect_network_metrics(self, simulation) -> NetworkMetrics:
        """Collect network-related metrics."""
        metrics = NetworkMetrics()
        
        # Social network metrics
        social_net = simulation.get_network('social')
        if social_net:
            metrics.social_nodes = social_net.number_of_nodes()
            metrics.social_edges = social_net.number_of_edges()
            
            if metrics.social_nodes > 1:
                metrics.social_density = nx.density(social_net)
                
                # Calculate clustering coefficient
                try:
                    metrics.social_clustering = nx.average_clustering(social_net)
                except:
                    metrics.social_clustering = 0.0
                
                # Calculate average path length for connected components
                try:
                    if nx.is_connected(social_net):
                        metrics.social_avg_path_length = nx.average_shortest_path_length(social_net)
                    else:
                        # Calculate for largest connected component
                        largest_cc = max(nx.connected_components(social_net), key=len)
                        subgraph = social_net.subgraph(largest_cc)
                        metrics.social_avg_path_length = nx.average_shortest_path_length(subgraph)
                except:
                    metrics.social_avg_path_length = 0.0
                
                # Centrality measures
                try:
                    degree_centrality = nx.degree_centrality(social_net)
                    metrics.max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 0.0
                    
                    betweenness_centrality = nx.betweenness_centrality(social_net)
                    metrics.max_betweenness_centrality = max(betweenness_centrality.values()) if betweenness_centrality else 0.0
                    
                    closeness_centrality = nx.closeness_centrality(social_net)
                    metrics.max_closeness_centrality = max(closeness_centrality.values()) if closeness_centrality else 0.0
                except:
                    pass
        
        # Information network metrics
        info_net = simulation.get_network('information')
        if info_net:
            metrics.info_nodes = info_net.number_of_nodes()
            metrics.info_edges = info_net.number_of_edges()
            if metrics.info_nodes > 1:
                metrics.info_density = nx.density(info_net)
        
        # Economic network metrics
        econ_net = simulation.get_network('economic')
        if econ_net:
            metrics.economic_nodes = econ_net.number_of_nodes()
            metrics.economic_edges = econ_net.number_of_edges()
            if metrics.economic_nodes > 1:
                metrics.economic_density = nx.density(econ_net)
        
        return metrics
    
    def _collect_economic_metrics(self, simulation) -> EconomicMetrics:
        """Collect economic-related metrics."""
        metrics = EconomicMetrics()
        
        # Aggregate economic data from agents
        for agent in simulation.agent_registry.values():
            # Revenue
            metrics.total_revenue += getattr(agent, 'revenue', 0.0)
            metrics.advertising_revenue += getattr(agent, 'ad_revenue', 0.0)
            metrics.subscription_revenue += getattr(agent, 'subscription_revenue', 0.0)
            
            # Costs
            metrics.total_costs += getattr(agent, 'costs', 0.0)
            metrics.content_production_costs += getattr(agent, 'content_costs', 0.0)
            metrics.fact_checking_costs += getattr(agent, 'fact_check_costs', 0.0)
            metrics.moderation_costs += getattr(agent, 'moderation_costs', 0.0)
        
        # Calculate market concentration (Herfindahl-Hirschman Index)
        revenues = [getattr(agent, 'revenue', 0.0) for agent in simulation.agent_registry.values()]
        if metrics.total_revenue > 0:
            market_shares = [rev / metrics.total_revenue for rev in revenues]
            metrics.market_concentration = sum(share ** 2 for share in market_shares)
        
        return metrics
    
    def _collect_intervention_metrics(self, simulation) -> InterventionMetrics:
        """Collect intervention-related metrics."""
        metrics = InterventionMetrics()
        
        # Count interventions from event log
        for event in simulation.event_log:
            if event['event_type'] == 'intervention':
                metrics.total_interventions += 1
                
                intervention_type = event['data'].get('intervention_type')
                if intervention_type == 'fact_check':
                    metrics.fact_check_interventions += 1
                elif intervention_type == 'content_removal':
                    metrics.content_removal_interventions += 1
                elif intervention_type == 'account_suspension':
                    metrics.account_suspension_interventions += 1
                elif intervention_type == 'algorithm_change':
                    metrics.algorithm_change_interventions += 1
        
        # Calculate effectiveness (placeholder logic)
        if metrics.total_interventions > 0:
            # This would be calculated based on actual intervention outcomes
            metrics.intervention_success_rate = 0.75  # Placeholder
            metrics.avg_intervention_impact = 0.6  # Placeholder
        
        return metrics
    
    def _calculate_information_quality(self, metrics: SimulationMetrics) -> float:
        """Calculate information quality index."""
        factors = []
        
        # Content credibility factor
        factors.append(metrics.content.avg_credibility_score)
        
        # Misinformation ratio factor (inverted)
        factors.append(1 - metrics.content.misinformation_ratio)
        
        # Fact-check coverage factor
        factors.append(metrics.content.fact_check_coverage)
        
        return mean(factors) if factors else 0.5
    
    def _calculate_social_cohesion(self, metrics: SimulationMetrics) -> float:
        """Calculate social cohesion index."""
        factors = []
        
        # Network connectivity factor
        if metrics.network.social_nodes > 0:
            factors.append(metrics.network.social_density)
        
        # Agent trust factor
        factors.append(metrics.agents.avg_trust_score)
        
        # Activity factor
        factors.append(metrics.agents.agent_activity_rate)
        
        return mean(factors) if factors else 0.5
    
    def _update_aggregated_metrics(self, metrics: SimulationMetrics) -> None:
        """Update aggregated metrics over time."""
        step = metrics.step
        
        # Store key metrics for trend analysis
        if 'misinformation_ratio' not in self.aggregated_metrics:
            self.aggregated_metrics['misinformation_ratio'] = []
        self.aggregated_metrics['misinformation_ratio'].append(metrics.content.misinformation_ratio)
        
        if 'health_score' not in self.aggregated_metrics:
            self.aggregated_metrics['health_score'] = []
        self.aggregated_metrics['health_score'].append(metrics.simulation_health_score)
        
        if 'agent_activity' not in self.aggregated_metrics:
            self.aggregated_metrics['agent_activity'] = []
        self.aggregated_metrics['agent_activity'].append(metrics.agents.agent_activity_rate)
    
    def get_current_metrics(self) -> SimulationMetrics:
        """Get current simulation metrics.
        
        Returns:
            Current metrics
        """
        return self.current_metrics
    
    def get_metrics_history(self, steps: int = None) -> List[SimulationMetrics]:
        """Get metrics history.
        
        Args:
            steps: Number of recent steps to return (all if None)
            
        Returns:
            List of historical metrics
        """
        if steps is None:
            return list(self.metrics_history)
        else:
            return list(self.metrics_history)[-steps:]
    
    def get_trend_analysis(self, metric_name: str, window_size: int = 10) -> Dict[str, float]:
        """Analyze trends in a specific metric.
        
        Args:
            metric_name: Name of metric to analyze
            window_size: Size of moving window for trend calculation
            
        Returns:
            Trend analysis results
        """
        if metric_name not in self.aggregated_metrics:
            return {}
        
        values = self.aggregated_metrics[metric_name]
        if len(values) < 2:
            return {}
        
        # Calculate basic statistics
        stats_result = {
            'current_value': values[-1],
            'mean': mean(values),
            'median': median(values),
            'min': min(values),
            'max': max(values)
        }
        
        if len(values) > 1:
            stats_result['std'] = stdev(values)
        
        # Calculate trend (slope of linear regression)
        if len(values) >= window_size:
            recent_values = values[-window_size:]
            x = list(range(len(recent_values)))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
            
            stats_result.update({
                'trend_slope': slope,
                'trend_r_squared': r_value ** 2,
                'trend_p_value': p_value
            })
        
        # Calculate change from previous value
        if len(values) >= 2:
            change = values[-1] - values[-2]
            percent_change = (change / values[-2]) * 100 if values[-2] != 0 else 0
            
            stats_result.update({
                'absolute_change': change,
                'percent_change': percent_change
            })
        
        return stats_result
    
    def get_correlation_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between different metrics.
        
        Returns:
            Correlation matrix
        """
        correlations = {}
        
        metric_names = list(self.aggregated_metrics.keys())
        
        for i, metric1 in enumerate(metric_names):
            correlations[metric1] = {}
            for j, metric2 in enumerate(metric_names):
                if i <= j:  # Only calculate upper triangle
                    values1 = self.aggregated_metrics[metric1]
                    values2 = self.aggregated_metrics[metric2]
                    
                    if len(values1) == len(values2) and len(values1) > 1:
                        correlation, p_value = stats.pearsonr(values1, values2)
                        correlations[metric1][metric2] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }
        
        return correlations
    
    def get_final_metrics(self) -> SimulationMetrics:
        """Get final simulation metrics.
        
        Returns:
            Final metrics summary
        """
        if not self.metrics_history:
            return SimulationMetrics()
        
        return self.metrics_history[-1]
    
    def export_data(self) -> List[Dict[str, Any]]:
        """Export metrics data for analysis.
        
        Returns:
            List of metrics dictionaries
        """
        exported_data = []
        
        for metrics in self.metrics_history:
            data = {
                'step': metrics.step,
                'timestamp': metrics.timestamp.isoformat(),
                
                # Content metrics
                'total_content': metrics.content.total_content,
                'misinformation_content': metrics.content.misinformation_content,
                'misinformation_ratio': metrics.content.misinformation_ratio,
                'fact_check_coverage': metrics.content.fact_check_coverage,
                'avg_credibility_score': metrics.content.avg_credibility_score,
                
                # Agent metrics
                'total_agents': metrics.agents.total_agents,
                'active_agents': metrics.agents.active_agents,
                'agent_activity_rate': metrics.agents.agent_activity_rate,
                'avg_trust_score': metrics.agents.avg_trust_score,
                
                # Network metrics
                'social_density': metrics.network.social_density,
                'social_clustering': metrics.network.social_clustering,
                'max_degree_centrality': metrics.network.max_degree_centrality,
                
                # Economic metrics
                'total_revenue': metrics.economic.total_revenue,
                'total_costs': metrics.economic.total_costs,
                'profit_margin': metrics.economic.profit_margin,
                
                # Intervention metrics
                'total_interventions': metrics.interventions.total_interventions,
                'intervention_success_rate': metrics.interventions.intervention_success_rate,
                
                # Overall metrics
                'simulation_health_score': metrics.simulation_health_score,
                'information_quality_index': metrics.information_quality_index,
                'social_cohesion_index': metrics.social_cohesion_index,
                
                # Performance metrics
                'step_execution_time': metrics.step_execution_time,
                'memory_usage': metrics.memory_usage
            }
            
            exported_data.append(data)
        
        return exported_data
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of simulation metrics.
        
        Returns:
            Summary report dictionary
        """
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        
        # Calculate overall statistics
        health_scores = [m.simulation_health_score for m in self.metrics_history]
        misinformation_ratios = [m.content.misinformation_ratio for m in self.metrics_history]
        
        report = {
            'simulation_summary': {
                'total_steps': len(self.metrics_history),
                'final_health_score': final_metrics.simulation_health_score,
                'avg_health_score': mean(health_scores) if health_scores else 0,
                'final_misinformation_ratio': final_metrics.content.misinformation_ratio,
                'avg_misinformation_ratio': mean(misinformation_ratios) if misinformation_ratios else 0
            },
            
            'content_summary': {
                'total_content_produced': final_metrics.content.total_content,
                'misinformation_content': final_metrics.content.misinformation_content,
                'fact_checked_content': final_metrics.content.fact_checked_content,
                'content_removed': final_metrics.content.removed_content
            },
            
            'agent_summary': {
                'total_agents': final_metrics.agents.total_agents,
                'final_active_agents': final_metrics.agents.active_agents,
                'suspended_agents': final_metrics.agents.suspended_agents,
                'avg_trust_score': final_metrics.agents.avg_trust_score
            },
            
            'network_summary': {
                'social_network_density': final_metrics.network.social_density,
                'information_cascades': final_metrics.network.information_cascade_count,
                'max_cascade_size': final_metrics.network.max_cascade_size
            },
            
            'intervention_summary': {
                'total_interventions': final_metrics.interventions.total_interventions,
                'intervention_success_rate': final_metrics.interventions.intervention_success_rate,
                'avg_response_time': final_metrics.interventions.avg_response_time
            },
            
            'performance_summary': {
                'avg_step_time': mean(self.collection_times) if self.collection_times else 0,
                'total_collection_time': sum(self.collection_times) if self.collection_times else 0
            }
        }
        
        return report
    
    def clear_history(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()
        self.aggregated_metrics.clear()
        self.collection_times.clear()
        
        logger.debug("Metrics history cleared")