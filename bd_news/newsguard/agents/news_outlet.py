"""News outlet agent for NewsGuard Bangladesh simulation.

This module implements news outlet agents that represent media organizations,
including newspapers, TV channels, online news sites, and blogs.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .base import BaseAgent, AgentType, AgentCapability, AgentState
from ..utils.logger import get_logger
from ..utils.helpers import weighted_random_choice, exponential_decay

logger = get_logger(__name__)


class OutletType(Enum):
    """Types of news outlets."""
    MAINSTREAM_NEWSPAPER = "mainstream_newspaper"
    TABLOID = "tabloid"
    TV_CHANNEL = "tv_channel"
    ONLINE_NEWS = "online_news"
    BLOG = "blog"
    SOCIAL_MEDIA_PAGE = "social_media_page"
    WIRE_SERVICE = "wire_service"
    MAGAZINE = "magazine"
    RADIO_STATION = "radio_station"
    CITIZEN_JOURNALISM = "citizen_journalism"


class EditorialStance(Enum):
    """Editorial stance of news outlets."""
    LEFT_LEANING = "left_leaning"
    CENTER_LEFT = "center_left"
    CENTRIST = "centrist"
    CENTER_RIGHT = "center_right"
    RIGHT_LEANING = "right_leaning"
    POPULIST = "populist"
    NATIONALIST = "nationalist"
    PROGRESSIVE = "progressive"
    CONSERVATIVE = "conservative"
    INDEPENDENT = "independent"


class RevenueModel(Enum):
    """Revenue models for news outlets."""
    ADVERTISING = "advertising"
    SUBSCRIPTION = "subscription"
    PAYWALL = "paywall"
    DONATION = "donation"
    GOVERNMENT_FUNDING = "government_funding"
    CORPORATE_SPONSORSHIP = "corporate_sponsorship"
    MIXED = "mixed"


@dataclass
class ContentStrategy:
    """Content strategy for news outlets."""
    # Content focus
    primary_topics: List[str] = field(default_factory=list)
    secondary_topics: List[str] = field(default_factory=list)
    
    # Publishing strategy
    publication_frequency: int = 5  # articles per day
    breaking_news_threshold: float = 0.8  # urgency threshold for breaking news
    
    # Content quality
    fact_checking_rigor: float = 0.7  # 0-1 scale
    source_verification_level: float = 0.6  # 0-1 scale
    editorial_oversight: float = 0.8  # 0-1 scale
    
    # Audience targeting
    target_demographics: Dict[str, Any] = field(default_factory=dict)
    content_complexity: float = 0.5  # 0-1 scale (simple to complex)
    
    # Engagement strategy
    clickbait_tendency: float = 0.3  # 0-1 scale
    sensationalism_level: float = 0.2  # 0-1 scale
    social_media_focus: float = 0.6  # 0-1 scale


@dataclass
class PublicationRecord:
    """Record of published content."""
    content_id: str
    title: str
    topic: str
    publication_time: datetime
    author: str
    
    # Content characteristics
    word_count: int = 0
    credibility_score: float = 0.5
    engagement_score: float = 0.0
    
    # Performance metrics
    views: int = 0
    shares: int = 0
    comments: int = 0
    fact_check_requests: int = 0
    
    # Revenue impact
    revenue_generated: float = 0.0
    cost_to_produce: float = 0.0


class NewsOutlet(BaseAgent):
    """News outlet agent representing media organizations."""
    
    def __init__(self, 
                 unique_id: str,
                 model,
                 config: Dict[str, Any] = None):
        """Initialize news outlet agent.
        
        Args:
            unique_id: Unique identifier
            model: Mesa model instance
            config: Agent configuration
        """
        super().__init__(unique_id, model, AgentType.NEWS_OUTLET, config)
        
        # News outlet specific initialization happens in _initialize_agent_specific
    
    def _initialize_agent_specific(self) -> None:
        """Initialize news outlet specific properties."""
        # Add capabilities
        self.add_capability(AgentCapability.CONTENT_CREATION)
        self.add_capability(AgentCapability.CONTENT_SHARING)
        self.add_capability(AgentCapability.ECONOMIC_ACTIVITY)
        
        # Outlet characteristics
        self.outlet_type = OutletType(self.config.get('outlet_type', OutletType.ONLINE_NEWS.value))
        self.editorial_stance = EditorialStance(self.config.get('editorial_stance', EditorialStance.CENTRIST.value))
        self.revenue_model = RevenueModel(self.config.get('revenue_model', RevenueModel.ADVERTISING.value))
        
        # Audience and reach
        self.audience_size = self.config.get('audience_size', 10000)
        self.daily_reach = self.config.get('daily_reach', self.audience_size * 0.1)
        self.subscriber_count = self.config.get('subscriber_count', self.audience_size * 0.05)
        
        # Content strategy
        strategy_config = self.config.get('content_strategy', {})
        self.content_strategy = ContentStrategy(
            primary_topics=strategy_config.get('primary_topics', ['politics', 'economy', 'society']),
            secondary_topics=strategy_config.get('secondary_topics', ['sports', 'entertainment', 'technology']),
            publication_frequency=strategy_config.get('publication_frequency', 5),
            fact_checking_rigor=strategy_config.get('fact_checking_rigor', 0.7),
            source_verification_level=strategy_config.get('source_verification_level', 0.6),
            editorial_oversight=strategy_config.get('editorial_oversight', 0.8),
            clickbait_tendency=strategy_config.get('clickbait_tendency', 0.3),
            sensationalism_level=strategy_config.get('sensationalism_level', 0.2)
        )
        
        # Staff and resources
        self.journalist_count = self.config.get('journalist_count', 10)
        self.editor_count = self.config.get('editor_count', 3)
        self.fact_checker_count = self.config.get('fact_checker_count', 2)
        
        # Quality metrics
        self.credibility_score = self.config.get('credibility_score', 0.7)
        self.editorial_independence = self.config.get('editorial_independence', 0.8)
        self.transparency_score = self.config.get('transparency_score', 0.6)
        
        # Publication tracking
        self.publication_history: List[PublicationRecord] = []
        self.daily_publication_count = 0
        self.last_publication_date = None
        
        # Economic state
        self.subscription_revenue = 0.0
        self.advertising_revenue = 0.0
        self.operational_costs = self.config.get('daily_operational_cost', 1000.0)
        
        # Performance tracking
        self.total_views = 0
        self.total_shares = 0
        self.average_engagement = 0.0
        self.fact_check_accuracy = 0.9  # Historical accuracy of fact-checking
        
        # Competitive positioning
        self.market_share = self.config.get('market_share', 0.01)
        self.brand_recognition = self.config.get('brand_recognition', 0.5)
        
        logger.debug(f"Initialized news outlet: {self.agent_id} ({self.outlet_type.value})")
    
    def step(self) -> None:
        """Execute one step of news outlet behavior."""
        if not self.is_active():
            return
        
        # Update daily metrics
        self._update_daily_metrics()
        
        # Decide on content creation
        if self._should_create_content():
            self._create_content()
        
        # Handle economic activities
        self._handle_economics()
        
        # Update audience engagement
        self._update_audience_engagement()
        
        # Adapt strategy based on performance
        self._adapt_strategy()
        
        # Update metrics
        self.metrics.update_activity('outlet_step')
        self.last_updated = datetime.now()
    
    def _update_daily_metrics(self) -> None:
        """Update daily metrics and reset counters if needed."""
        current_date = datetime.now().date()
        
        if self.last_publication_date != current_date:
            # New day - reset daily counters
            self.daily_publication_count = 0
            self.last_publication_date = current_date
            
            # Pay daily operational costs
            self.update_economic_state(cost_change=self.operational_costs)
    
    def _should_create_content(self) -> bool:
        """Determine if outlet should create content now.
        
        Returns:
            True if should create content
        """
        # Check if we've reached daily publication limit
        if self.daily_publication_count >= self.content_strategy.publication_frequency:
            return False
        
        # Check economic constraints
        content_cost = self._estimate_content_cost()
        if self.economic_state['budget'] < content_cost:
            return False
        
        # Probability based on time of day and outlet type
        hour = datetime.now().hour
        
        # Different outlets have different publishing patterns
        if self.outlet_type == OutletType.MAINSTREAM_NEWSPAPER:
            # Traditional newspapers publish more in morning
            probability = 0.8 if 6 <= hour <= 10 else 0.3
        elif self.outlet_type == OutletType.ONLINE_NEWS:
            # Online news publishes throughout the day
            probability = 0.6 if 8 <= hour <= 22 else 0.2
        elif self.outlet_type == OutletType.BLOG:
            # Blogs have irregular publishing
            probability = 0.4
        else:
            probability = 0.5
        
        # Adjust for breaking news events
        if hasattr(self.model, 'current_events'):
            breaking_events = [e for e in self.model.current_events if e.get('urgency', 0) > self.content_strategy.breaking_news_threshold]
            if breaking_events:
                probability *= 2.0
        
        return random.random() < probability
    
    def _create_content(self) -> Optional[str]:
        """Create and publish content.
        
        Returns:
            Content ID if created, None otherwise
        """
        # Select topic
        topic = self._select_topic()
        
        # Generate content characteristics
        content_data = self._generate_content_data(topic)
        
        # Calculate production cost
        production_cost = self._calculate_production_cost(content_data)
        
        # Check if we can afford it
        if self.economic_state['budget'] < production_cost:
            return None
        
        # Create content record
        content_id = f"{self.agent_id}_content_{len(self.publication_history)}"
        
        publication = PublicationRecord(
            content_id=content_id,
            title=content_data['title'],
            topic=topic,
            publication_time=datetime.now(),
            author=content_data['author'],
            word_count=content_data['word_count'],
            credibility_score=content_data['credibility_score'],
            cost_to_produce=production_cost
        )
        
        # Add to publication history
        self.publication_history.append(publication)
        self.daily_publication_count += 1
        
        # Update economic state
        self.update_economic_state(cost_change=production_cost)
        
        # Update metrics
        self.metrics.update_activity('content_creation')
        self.metrics.content_created += 1
        
        # Notify model about new content
        if hasattr(self.model, 'add_content'):
            self.model.add_content({
                'id': content_id,
                'outlet_id': self.agent_id,
                'topic': topic,
                'credibility_score': content_data['credibility_score'],
                'publication_time': publication.publication_time,
                'content_type': 'news_article',
                'metadata': content_data
            })
        
        logger.info(f"News outlet {self.agent_id} published: {publication.title} (topic: {topic})")
        
        return content_id
    
    def _select_topic(self) -> str:
        """Select topic for content creation.
        
        Returns:
            Selected topic
        """
        # Combine primary and secondary topics with weights
        all_topics = []
        weights = []
        
        # Primary topics have higher weight
        for topic in self.content_strategy.primary_topics:
            all_topics.append(topic)
            weights.append(0.7)
        
        # Secondary topics have lower weight
        for topic in self.content_strategy.secondary_topics:
            all_topics.append(topic)
            weights.append(0.3)
        
        # Adjust weights based on current events
        if hasattr(self.model, 'current_events'):
            for i, topic in enumerate(all_topics):
                # Check if there are current events related to this topic
                related_events = [e for e in self.model.current_events if topic in e.get('topics', [])]
                if related_events:
                    weights[i] *= 1.5  # Increase weight for trending topics
        
        # Select topic using weighted random choice
        if all_topics:
            return weighted_random_choice(all_topics, weights)
        else:
            return 'general'
    
    def _generate_content_data(self, topic: str) -> Dict[str, Any]:
        """Generate content data for a given topic.
        
        Args:
            topic: Content topic
            
        Returns:
            Content data dictionary
        """
        # Generate title based on topic and outlet characteristics
        title_templates = {
            'politics': [
                "Political Development in {topic}",
                "Government Announces New {topic} Policy",
                "Opposition Criticizes {topic} Decision"
            ],
            'economy': [
                "Economic Impact of {topic}",
                "Market Responds to {topic} News",
                "Experts Analyze {topic} Trends"
            ],
            'society': [
                "Social Changes in {topic}",
                "Community Reacts to {topic}",
                "Cultural Shift in {topic}"
            ]
        }
        
        templates = title_templates.get(topic, ["Breaking News: {topic}", "Update on {topic}", "Analysis: {topic}"])
        title = random.choice(templates).format(topic=topic.title())
        
        # Adjust title based on outlet characteristics
        if self.content_strategy.clickbait_tendency > 0.6:
            clickbait_modifiers = ["SHOCKING:", "BREAKING:", "EXCLUSIVE:", "URGENT:"]
            if random.random() < self.content_strategy.clickbait_tendency:
                title = f"{random.choice(clickbait_modifiers)} {title}"
        
        # Generate author (simplified)
        author = f"Reporter_{random.randint(1, self.journalist_count)}"
        
        # Word count based on outlet type and topic
        base_word_count = {
            OutletType.MAINSTREAM_NEWSPAPER: 800,
            OutletType.TABLOID: 400,
            OutletType.ONLINE_NEWS: 600,
            OutletType.BLOG: 1200,
            OutletType.TV_CHANNEL: 300,
            OutletType.SOCIAL_MEDIA_PAGE: 150
        }.get(self.outlet_type, 600)
        
        word_count = int(base_word_count * random.uniform(0.7, 1.3))
        
        # Calculate credibility score based on outlet characteristics
        credibility_factors = [
            self.credibility_score,
            self.content_strategy.fact_checking_rigor,
            self.content_strategy.source_verification_level,
            self.content_strategy.editorial_oversight,
            1.0 - self.content_strategy.sensationalism_level,
            1.0 - self.content_strategy.clickbait_tendency
        ]
        
        base_credibility = np.mean(credibility_factors)
        
        # Add some randomness
        credibility_score = base_credibility * random.uniform(0.8, 1.2)
        credibility_score = max(0.1, min(1.0, credibility_score))
        
        return {
            'title': title,
            'author': author,
            'word_count': word_count,
            'credibility_score': credibility_score,
            'topic': topic,
            'outlet_type': self.outlet_type.value,
            'editorial_stance': self.editorial_stance.value,
            'clickbait_score': self.content_strategy.clickbait_tendency,
            'sensationalism_score': self.content_strategy.sensationalism_level
        }
    
    def _estimate_content_cost(self) -> float:
        """Estimate cost of creating content.
        
        Returns:
            Estimated cost
        """
        # Base cost depends on outlet type and size
        base_costs = {
            OutletType.MAINSTREAM_NEWSPAPER: 500,
            OutletType.TABLOID: 200,
            OutletType.ONLINE_NEWS: 300,
            OutletType.BLOG: 100,
            OutletType.TV_CHANNEL: 800,
            OutletType.SOCIAL_MEDIA_PAGE: 50
        }
        
        base_cost = base_costs.get(self.outlet_type, 300)
        
        # Adjust for quality requirements
        quality_multiplier = 1.0 + (self.content_strategy.fact_checking_rigor * 0.5)
        
        return base_cost * quality_multiplier
    
    def _calculate_production_cost(self, content_data: Dict[str, Any]) -> float:
        """Calculate actual production cost for content.
        
        Args:
            content_data: Content data
            
        Returns:
            Production cost
        """
        base_cost = self._estimate_content_cost()
        
        # Adjust for word count
        word_count_factor = content_data['word_count'] / 600  # 600 is baseline
        
        # Adjust for credibility requirements
        credibility_factor = 1.0 + (content_data['credibility_score'] * 0.3)
        
        total_cost = base_cost * word_count_factor * credibility_factor
        
        return total_cost
    
    def _handle_economics(self) -> None:
        """Handle economic activities."""
        # Generate revenue based on recent publications
        if self.publication_history:
            recent_publications = [
                p for p in self.publication_history
                if (datetime.now() - p.publication_time).days <= 1
            ]
            
            for publication in recent_publications:
                revenue = self._calculate_content_revenue(publication)
                publication.revenue_generated += revenue
                self.update_economic_state(revenue_change=revenue)
        
        # Handle subscription revenue (daily)
        if self.revenue_model in [RevenueModel.SUBSCRIPTION, RevenueModel.MIXED]:
            daily_subscription_revenue = (self.subscriber_count * 10) / 30  # $10/month per subscriber
            self.subscription_revenue += daily_subscription_revenue
            self.update_economic_state(revenue_change=daily_subscription_revenue)
    
    def _calculate_content_revenue(self, publication: PublicationRecord) -> float:
        """Calculate revenue from a publication.
        
        Args:
            publication: Publication record
            
        Returns:
            Revenue amount
        """
        # Simulate views based on audience size and content quality
        potential_views = self.daily_reach
        
        # Adjust for content characteristics
        engagement_factor = 1.0
        if publication.credibility_score > 0.8:
            engagement_factor *= 1.2  # High quality content gets more engagement
        
        # Clickbait and sensationalism can increase short-term views
        clickbait_boost = 1.0 + (self.content_strategy.clickbait_tendency * 0.5)
        sensationalism_boost = 1.0 + (self.content_strategy.sensationalism_level * 0.3)
        
        views = int(potential_views * engagement_factor * clickbait_boost * sensationalism_boost * random.uniform(0.5, 1.5))
        publication.views += views
        self.total_views += views
        
        # Calculate revenue based on revenue model
        revenue = 0.0
        
        if self.revenue_model in [RevenueModel.ADVERTISING, RevenueModel.MIXED]:
            # Advertising revenue: $0.001 per view (CPM model)
            ad_revenue = views * 0.001
            revenue += ad_revenue
            self.advertising_revenue += ad_revenue
        
        if self.revenue_model == RevenueModel.PAYWALL:
            # Paywall revenue: some viewers pay to read
            paying_viewers = int(views * 0.02)  # 2% conversion rate
            paywall_revenue = paying_viewers * 2.0  # $2 per article
            revenue += paywall_revenue
        
        return revenue
    
    def _update_audience_engagement(self) -> None:
        """Update audience engagement metrics."""
        if self.publication_history:
            recent_publications = [
                p for p in self.publication_history
                if (datetime.now() - p.publication_time).days <= 7
            ]
            
            if recent_publications:
                # Calculate average engagement
                total_engagement = sum(p.views + p.shares * 5 + p.comments * 10 for p in recent_publications)
                self.average_engagement = total_engagement / len(recent_publications)
                
                # Update credibility based on fact-checking accuracy
                fact_checked_articles = [p for p in recent_publications if p.fact_check_requests > 0]
                if fact_checked_articles:
                    # Simulate fact-checking results
                    accurate_articles = sum(1 for p in fact_checked_articles 
                                          if random.random() < p.credibility_score)
                    self.fact_check_accuracy = accurate_articles / len(fact_checked_articles)
                    
                    # Update overall credibility
                    credibility_change = (self.fact_check_accuracy - 0.5) * 0.1
                    self.credibility_score = max(0.1, min(1.0, self.credibility_score + credibility_change))
    
    def _adapt_strategy(self) -> None:
        """Adapt content strategy based on performance."""
        if len(self.publication_history) < 10:
            return  # Need enough data to adapt
        
        recent_publications = self.publication_history[-10:]
        
        # Analyze performance
        avg_views = np.mean([p.views for p in recent_publications])
        avg_revenue = np.mean([p.revenue_generated for p in recent_publications])
        avg_credibility = np.mean([p.credibility_score for p in recent_publications])
        
        # Adapt based on economic pressure
        if self.economic_state['budget'] < self.operational_costs * 3:  # Less than 3 days of operating costs
            # Economic pressure - might increase clickbait or reduce quality
            self.content_strategy.clickbait_tendency = min(1.0, self.content_strategy.clickbait_tendency + 0.1)
            self.content_strategy.fact_checking_rigor = max(0.1, self.content_strategy.fact_checking_rigor - 0.05)
            
            logger.warning(f"News outlet {self.agent_id} under economic pressure - adapting strategy")
        
        # Adapt based on audience engagement
        if avg_views < self.daily_reach * 0.5:  # Low engagement
            # Try to increase engagement
            self.content_strategy.social_media_focus = min(1.0, self.content_strategy.social_media_focus + 0.1)
            
            # Might increase sensationalism if desperate
            if self.economic_state['budget'] < self.operational_costs:
                self.content_strategy.sensationalism_level = min(1.0, self.content_strategy.sensationalism_level + 0.05)
        
        # Maintain credibility if doing well
        if avg_revenue > avg_views * 0.002:  # Good revenue per view
            self.content_strategy.fact_checking_rigor = min(1.0, self.content_strategy.fact_checking_rigor + 0.02)
            self.content_strategy.clickbait_tendency = max(0.0, self.content_strategy.clickbait_tendency - 0.02)
    
    def get_outlet_summary(self) -> Dict[str, Any]:
        """Get comprehensive outlet summary.
        
        Returns:
            Outlet summary
        """
        base_summary = self.get_agent_summary()
        
        # Calculate recent performance
        recent_publications = [
            p for p in self.publication_history
            if (datetime.now() - p.publication_time).days <= 7
        ]
        
        outlet_specific = {
            'outlet_type': self.outlet_type.value,
            'editorial_stance': self.editorial_stance.value,
            'revenue_model': self.revenue_model.value,
            'audience_size': self.audience_size,
            'subscriber_count': self.subscriber_count,
            'credibility_score': self.credibility_score,
            'editorial_independence': self.editorial_independence,
            'transparency_score': self.transparency_score,
            'content_strategy': {
                'publication_frequency': self.content_strategy.publication_frequency,
                'fact_checking_rigor': self.content_strategy.fact_checking_rigor,
                'clickbait_tendency': self.content_strategy.clickbait_tendency,
                'sensationalism_level': self.content_strategy.sensationalism_level
            },
            'performance': {
                'total_publications': len(self.publication_history),
                'recent_publications': len(recent_publications),
                'total_views': self.total_views,
                'average_engagement': self.average_engagement,
                'fact_check_accuracy': self.fact_check_accuracy
            },
            'economics': {
                'subscription_revenue': self.subscription_revenue,
                'advertising_revenue': self.advertising_revenue,
                'operational_costs': self.operational_costs,
                'daily_cost': self.operational_costs
            }
        }
        
        base_summary.update(outlet_specific)
        return base_summary
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate news outlet configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        if not super().validate_config(config):
            return False
        
        # Validate outlet-specific fields
        required_fields = ['audience_size', 'credibility_score']
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate ranges
        if not (0 <= config.get('credibility_score', 0.5) <= 1.0):
            return False
        
        if config.get('audience_size', 0) <= 0:
            return False
        
        return True