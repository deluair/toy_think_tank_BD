"""Economic models for NewsGuard Bangladesh simulation.

This module implements economic models for news outlets, advertising,
revenue generation, and financial incentives in the Bangladesh media ecosystem.
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
from scipy import stats
import pandas as pd

from ..utils.logger import get_logger
from ..utils.helpers import (
    calculate_statistics, normalize_scores, exponential_decay,
    sigmoid_function, weighted_random_choice
)

logger = get_logger(__name__)


class RevenueSource(Enum):
    """Types of revenue sources."""
    ADVERTISING = "advertising"
    SUBSCRIPTION = "subscription"
    DONATION = "donation"
    GOVERNMENT_FUNDING = "government_funding"
    SPONSORED_CONTENT = "sponsored_content"
    AFFILIATE_MARKETING = "affiliate_marketing"
    EVENTS = "events"
    MERCHANDISE = "merchandise"
    PREMIUM_CONTENT = "premium_content"
    DATA_LICENSING = "data_licensing"


class AdType(Enum):
    """Types of advertisements."""
    DISPLAY_BANNER = "display_banner"
    VIDEO_AD = "video_ad"
    NATIVE_AD = "native_ad"
    SPONSORED_POST = "sponsored_post"
    POPUP_AD = "popup_ad"
    SOCIAL_MEDIA_AD = "social_media_ad"
    SEARCH_AD = "search_ad"
    AFFILIATE_AD = "affiliate_ad"


class EconomicEventType(Enum):
    """Types of economic events."""
    REVENUE_GENERATED = "revenue_generated"
    COST_INCURRED = "cost_incurred"
    INVESTMENT_MADE = "investment_made"
    FUNDING_RECEIVED = "funding_received"
    AD_CAMPAIGN_STARTED = "ad_campaign_started"
    AD_CAMPAIGN_ENDED = "ad_campaign_ended"
    SUBSCRIPTION_PURCHASED = "subscription_purchased"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"
    ECONOMIC_SHOCK = "economic_shock"


class MarketSegment(Enum):
    """Market segments for targeting."""
    YOUTH = "youth"  # 18-30
    MIDDLE_AGE = "middle_age"  # 31-50
    SENIOR = "senior"  # 50+
    URBAN = "urban"
    RURAL = "rural"
    HIGH_INCOME = "high_income"
    MIDDLE_INCOME = "middle_income"
    LOW_INCOME = "low_income"
    EDUCATED = "educated"
    TECH_SAVVY = "tech_savvy"
    POLITICAL_ACTIVE = "political_active"


@dataclass
class FinancialMetrics:
    """Financial metrics for an entity."""
    entity_id: str
    
    # Revenue metrics
    total_revenue: float = 0.0
    monthly_revenue: float = 0.0
    revenue_growth_rate: float = 0.0
    
    # Cost metrics
    total_costs: float = 0.0
    operational_costs: float = 0.0
    content_costs: float = 0.0
    marketing_costs: float = 0.0
    
    # Profitability
    profit_margin: float = 0.0
    net_profit: float = 0.0
    
    # Revenue breakdown
    revenue_by_source: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    revenue_per_user: float = 0.0
    cost_per_acquisition: float = 0.0
    lifetime_value: float = 0.0
    
    # Financial health
    cash_flow: float = 0.0
    debt_ratio: float = 0.0
    sustainability_score: float = 0.5
    
    # Temporal data
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_profit_margin(self) -> float:
        """Calculate profit margin."""
        if self.total_revenue > 0:
            self.profit_margin = (self.total_revenue - self.total_costs) / self.total_revenue
        else:
            self.profit_margin = 0.0
        return self.profit_margin
    
    def calculate_sustainability_score(self) -> float:
        """Calculate financial sustainability score."""
        factors = {
            'profitability': min(1.0, max(0.0, self.profit_margin + 0.5)),
            'revenue_growth': min(1.0, max(0.0, self.revenue_growth_rate / 0.2 + 0.5)),
            'cash_flow': min(1.0, max(0.0, self.cash_flow / 10000 + 0.5)),
            'debt_management': min(1.0, max(0.0, 1.0 - self.debt_ratio))
        }
        
        weights = {'profitability': 0.4, 'revenue_growth': 0.3, 'cash_flow': 0.2, 'debt_management': 0.1}
        
        self.sustainability_score = sum(
            weights[factor] * score for factor, score in factors.items()
        )
        
        return self.sustainability_score


@dataclass
class AdCampaign:
    """Advertisement campaign."""
    campaign_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    advertiser_id: str = ""
    publisher_id: str = ""
    
    # Campaign details
    campaign_name: str = ""
    ad_type: AdType = AdType.DISPLAY_BANNER
    target_segments: List[MarketSegment] = field(default_factory=list)
    
    # Financial terms
    budget: float = 0.0
    cost_per_click: float = 0.0
    cost_per_impression: float = 0.0
    cost_per_acquisition: float = 0.0
    
    # Performance metrics
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue_generated: float = 0.0
    
    # Timing
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    duration_days: int = 7
    
    # Targeting
    geographic_targeting: List[str] = field(default_factory=list)
    demographic_targeting: Dict[str, Any] = field(default_factory=dict)
    interest_targeting: List[str] = field(default_factory=list)
    
    # Content
    ad_content: str = ""
    landing_page: str = ""
    
    # Status
    is_active: bool = True
    is_approved: bool = True
    
    def calculate_ctr(self) -> float:
        """Calculate click-through rate."""
        if self.impressions > 0:
            return self.clicks / self.impressions
        return 0.0
    
    def calculate_conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.clicks > 0:
            return self.conversions / self.clicks
        return 0.0
    
    def calculate_roi(self) -> float:
        """Calculate return on investment."""
        if self.budget > 0:
            return (self.revenue_generated - self.budget) / self.budget
        return 0.0


@dataclass
class EconomicEvent:
    """Economic event in the simulation."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EconomicEventType = EconomicEventType.REVENUE_GENERATED
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Event participants
    entity_id: str = ""
    counterparty_id: str = ""
    
    # Financial details
    amount: float = 0.0
    currency: str = "BDT"
    
    # Event context
    revenue_source: Optional[RevenueSource] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Impact
    impact_score: float = 0.0
    affected_entities: List[str] = field(default_factory=list)


class EconomicModel:
    """Economic model for the NewsGuard simulation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize economic model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Financial tracking
        self.financial_metrics: Dict[str, FinancialMetrics] = {}
        self.economic_events: List[EconomicEvent] = []
        self.ad_campaigns: Dict[str, AdCampaign] = {}
        
        # Market data
        self.market_segments: Dict[MarketSegment, Dict[str, Any]] = self._initialize_market_segments()
        self.ad_rates: Dict[AdType, Dict[str, float]] = self._initialize_ad_rates()
        
        # Economic parameters
        self.base_cpm = self.config.get('base_cpm', 2.0)  # Cost per mille (1000 impressions) in BDT
        self.base_cpc = self.config.get('base_cpc', 0.5)   # Cost per click in BDT
        self.inflation_rate = self.config.get('inflation_rate', 0.05)  # Annual inflation
        self.market_growth_rate = self.config.get('market_growth_rate', 0.15)  # Annual market growth
        
        # Revenue models
        self.subscription_prices = {
            'basic': 100.0,    # BDT per month
            'premium': 200.0,  # BDT per month
            'enterprise': 500.0  # BDT per month
        }
        
        # Cost models
        self.operational_cost_factors = {
            'content_creation': 0.4,
            'technology': 0.2,
            'marketing': 0.15,
            'administration': 0.15,
            'other': 0.1
        }
        
        logger.debug("Economic model initialized")
    
    def _initialize_market_segments(self) -> Dict[MarketSegment, Dict[str, Any]]:
        """Initialize market segment data.
        
        Returns:
            Market segment information
        """
        segments = {
            MarketSegment.YOUTH: {
                'size': 0.35,  # 35% of population
                'spending_power': 0.6,
                'digital_engagement': 0.9,
                'ad_receptivity': 0.7,
                'preferred_content': ['entertainment', 'technology', 'lifestyle']
            },
            MarketSegment.MIDDLE_AGE: {
                'size': 0.45,
                'spending_power': 0.8,
                'digital_engagement': 0.7,
                'ad_receptivity': 0.6,
                'preferred_content': ['news', 'business', 'health', 'family']
            },
            MarketSegment.SENIOR: {
                'size': 0.20,
                'spending_power': 0.7,
                'digital_engagement': 0.4,
                'ad_receptivity': 0.5,
                'preferred_content': ['news', 'health', 'religion', 'politics']
            },
            MarketSegment.URBAN: {
                'size': 0.40,
                'spending_power': 0.9,
                'digital_engagement': 0.8,
                'ad_receptivity': 0.7,
                'preferred_content': ['news', 'business', 'technology']
            },
            MarketSegment.RURAL: {
                'size': 0.60,
                'spending_power': 0.4,
                'digital_engagement': 0.5,
                'ad_receptivity': 0.6,
                'preferred_content': ['agriculture', 'local_news', 'entertainment']
            },
            MarketSegment.HIGH_INCOME: {
                'size': 0.15,
                'spending_power': 1.0,
                'digital_engagement': 0.8,
                'ad_receptivity': 0.5,
                'preferred_content': ['business', 'luxury', 'travel']
            },
            MarketSegment.MIDDLE_INCOME: {
                'size': 0.35,
                'spending_power': 0.6,
                'digital_engagement': 0.7,
                'ad_receptivity': 0.7,
                'preferred_content': ['news', 'education', 'health']
            },
            MarketSegment.LOW_INCOME: {
                'size': 0.50,
                'spending_power': 0.3,
                'digital_engagement': 0.5,
                'ad_receptivity': 0.8,
                'preferred_content': ['entertainment', 'local_news', 'jobs']
            }
        }
        
        return segments
    
    def _initialize_ad_rates(self) -> Dict[AdType, Dict[str, float]]:
        """Initialize advertising rates.
        
        Returns:
            Ad rates by type
        """
        rates = {
            AdType.DISPLAY_BANNER: {
                'cpm': 2.0,   # BDT per 1000 impressions
                'cpc': 0.5,   # BDT per click
                'cpa': 25.0   # BDT per acquisition
            },
            AdType.VIDEO_AD: {
                'cpm': 8.0,
                'cpc': 2.0,
                'cpa': 50.0
            },
            AdType.NATIVE_AD: {
                'cpm': 5.0,
                'cpc': 1.5,
                'cpa': 40.0
            },
            AdType.SPONSORED_POST: {
                'cpm': 10.0,
                'cpc': 3.0,
                'cpa': 75.0
            },
            AdType.POPUP_AD: {
                'cpm': 1.5,
                'cpc': 0.3,
                'cpa': 15.0
            },
            AdType.SOCIAL_MEDIA_AD: {
                'cpm': 6.0,
                'cpc': 1.8,
                'cpa': 45.0
            },
            AdType.SEARCH_AD: {
                'cpm': 12.0,
                'cpc': 4.0,
                'cpa': 80.0
            },
            AdType.AFFILIATE_AD: {
                'cpm': 3.0,
                'cpc': 1.0,
                'cpa': 30.0
            }
        }
        
        return rates
    
    def initialize_financial_metrics(self, entity_id: str, entity_type: str = 'news_outlet') -> FinancialMetrics:
        """Initialize financial metrics for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            
        Returns:
            Initial financial metrics
        """
        # Base financial parameters by entity type
        base_params = {
            'news_outlet': {
                'monthly_revenue': random.uniform(50000, 500000),  # BDT
                'operational_costs': random.uniform(40000, 400000),
                'revenue_sources': [RevenueSource.ADVERTISING, RevenueSource.SUBSCRIPTION]
            },
            'fact_checker': {
                'monthly_revenue': random.uniform(20000, 100000),
                'operational_costs': random.uniform(15000, 80000),
                'revenue_sources': [RevenueSource.DONATION, RevenueSource.GOVERNMENT_FUNDING]
            },
            'influencer': {
                'monthly_revenue': random.uniform(10000, 200000),
                'operational_costs': random.uniform(5000, 50000),
                'revenue_sources': [RevenueSource.SPONSORED_CONTENT, RevenueSource.AFFILIATE_MARKETING]
            },
            'platform': {
                'monthly_revenue': random.uniform(1000000, 10000000),
                'operational_costs': random.uniform(800000, 8000000),
                'revenue_sources': [RevenueSource.ADVERTISING, RevenueSource.DATA_LICENSING]
            }
        }
        
        params = base_params.get(entity_type, base_params['news_outlet'])
        
        # Initialize revenue breakdown
        revenue_by_source = {}
        num_sources = len(params['revenue_sources'])
        for i, source in enumerate(params['revenue_sources']):
            if i == 0:  # Primary source gets larger share
                revenue_by_source[source.value] = params['monthly_revenue'] * 0.7
            else:  # Secondary sources split remaining
                revenue_by_source[source.value] = params['monthly_revenue'] * 0.3 / (num_sources - 1)
        
        metrics = FinancialMetrics(
            entity_id=entity_id,
            monthly_revenue=params['monthly_revenue'],
            total_revenue=params['monthly_revenue'] * 12,  # Assume annual
            operational_costs=params['operational_costs'],
            total_costs=params['operational_costs'] * 12,
            revenue_by_source=revenue_by_source,
            revenue_growth_rate=random.uniform(-0.1, 0.3),  # -10% to +30%
            cash_flow=params['monthly_revenue'] - params['operational_costs']
        )
        
        # Calculate derived metrics
        metrics.calculate_profit_margin()
        metrics.calculate_sustainability_score()
        
        self.financial_metrics[entity_id] = metrics
        
        logger.debug(f"Initialized financial metrics for {entity_id}: Revenue={metrics.monthly_revenue:.0f} BDT/month")
        
        return metrics
    
    def create_ad_campaign(self, 
                          advertiser_id: str,
                          publisher_id: str,
                          campaign_config: Dict[str, Any]) -> AdCampaign:
        """Create a new advertising campaign.
        
        Args:
            advertiser_id: Advertiser entity ID
            publisher_id: Publisher entity ID
            campaign_config: Campaign configuration
            
        Returns:
            Created ad campaign
        """
        campaign = AdCampaign(
            advertiser_id=advertiser_id,
            publisher_id=publisher_id,
            campaign_name=campaign_config.get('name', f"Campaign_{len(self.ad_campaigns) + 1}"),
            ad_type=AdType(campaign_config.get('ad_type', 'display_banner')),
            budget=campaign_config.get('budget', 10000.0),
            duration_days=campaign_config.get('duration_days', 7),
            target_segments=[MarketSegment(seg) for seg in campaign_config.get('target_segments', ['urban'])],
            geographic_targeting=campaign_config.get('geographic_targeting', ['Dhaka']),
            ad_content=campaign_config.get('ad_content', 'Sample ad content')
        )
        
        # Set pricing based on ad type
        ad_rates = self.ad_rates[campaign.ad_type]
        campaign.cost_per_impression = ad_rates['cpm'] / 1000  # Convert CPM to CPI
        campaign.cost_per_click = ad_rates['cpc']
        campaign.cost_per_acquisition = ad_rates['cpa']
        
        # Set end date
        campaign.end_date = campaign.start_date + timedelta(days=campaign.duration_days)
        
        self.ad_campaigns[campaign.campaign_id] = campaign
        
        # Record economic event
        event = EconomicEvent(
            event_type=EconomicEventType.AD_CAMPAIGN_STARTED,
            entity_id=advertiser_id,
            counterparty_id=publisher_id,
            amount=campaign.budget,
            description=f"Started ad campaign: {campaign.campaign_name}",
            metadata={'campaign_id': campaign.campaign_id}
        )
        self.economic_events.append(event)
        
        logger.debug(f"Created ad campaign {campaign.campaign_id}: Budget={campaign.budget:.0f} BDT")
        
        return campaign
    
    def simulate_ad_performance(self, campaign_id: str, time_step: int = 1) -> Dict[str, Any]:
        """Simulate advertising campaign performance.
        
        Args:
            campaign_id: Campaign identifier
            time_step: Time step in hours
            
        Returns:
            Performance metrics
        """
        if campaign_id not in self.ad_campaigns:
            return {}
        
        campaign = self.ad_campaigns[campaign_id]
        
        if not campaign.is_active or datetime.now() > campaign.end_date:
            return {'status': 'inactive'}
        
        # Calculate target audience size
        target_audience_size = self._calculate_target_audience_size(campaign.target_segments)
        
        # Simulate impressions based on budget and targeting
        daily_budget = campaign.budget / campaign.duration_days
        hourly_budget = daily_budget / 24
        
        # Calculate impressions for this time step
        max_impressions = int(hourly_budget / campaign.cost_per_impression)
        actual_impressions = min(max_impressions, int(target_audience_size * 0.1))  # 10% reach per hour
        
        # Add randomness
        actual_impressions = int(actual_impressions * random.uniform(0.7, 1.3))
        
        # Calculate clicks based on CTR
        base_ctr = self._calculate_base_ctr(campaign.ad_type, campaign.target_segments)
        actual_ctr = base_ctr * random.uniform(0.8, 1.2)
        clicks = int(actual_impressions * actual_ctr)
        
        # Calculate conversions
        base_conversion_rate = 0.02  # 2% base conversion rate
        actual_conversion_rate = base_conversion_rate * random.uniform(0.5, 2.0)
        conversions = int(clicks * actual_conversion_rate)
        
        # Update campaign metrics
        campaign.impressions += actual_impressions
        campaign.clicks += clicks
        campaign.conversions += conversions
        
        # Calculate revenue (simplified)
        revenue_per_conversion = campaign.cost_per_acquisition * 2  # Assume 2x return
        revenue_generated = conversions * revenue_per_conversion
        campaign.revenue_generated += revenue_generated
        
        # Calculate costs
        impression_cost = actual_impressions * campaign.cost_per_impression
        click_cost = clicks * campaign.cost_per_click
        total_cost = impression_cost + click_cost
        
        # Record revenue event for publisher
        if revenue_generated > 0:
            revenue_event = EconomicEvent(
                event_type=EconomicEventType.REVENUE_GENERATED,
                entity_id=campaign.publisher_id,
                counterparty_id=campaign.advertiser_id,
                amount=total_cost,
                revenue_source=RevenueSource.ADVERTISING,
                description=f"Ad revenue from campaign {campaign.campaign_id}",
                metadata={'campaign_id': campaign_id, 'impressions': actual_impressions, 'clicks': clicks}
            )
            self.economic_events.append(revenue_event)
            
            # Update publisher financial metrics
            self._update_revenue(campaign.publisher_id, RevenueSource.ADVERTISING, total_cost)
        
        performance = {
            'campaign_id': campaign_id,
            'impressions': actual_impressions,
            'clicks': clicks,
            'conversions': conversions,
            'ctr': actual_ctr,
            'conversion_rate': actual_conversion_rate,
            'cost': total_cost,
            'revenue': revenue_generated,
            'roi': (revenue_generated - total_cost) / total_cost if total_cost > 0 else 0
        }
        
        return performance
    
    def _calculate_target_audience_size(self, target_segments: List[MarketSegment]) -> int:
        """Calculate target audience size based on segments.
        
        Args:
            target_segments: Target market segments
            
        Returns:
            Estimated audience size
        """
        # Bangladesh population (simplified)
        total_population = 165000000
        internet_users = int(total_population * 0.65)  # 65% internet penetration
        
        # Calculate overlap of target segments
        if not target_segments:
            return int(internet_users * 0.1)  # Default 10%
        
        # Simplified calculation - take average of segment sizes
        avg_segment_size = np.mean([
            self.market_segments[segment]['size']
            for segment in target_segments
            if segment in self.market_segments
        ])
        
        return int(internet_users * avg_segment_size)
    
    def _calculate_base_ctr(self, ad_type: AdType, target_segments: List[MarketSegment]) -> float:
        """Calculate base click-through rate.
        
        Args:
            ad_type: Type of advertisement
            target_segments: Target segments
            
        Returns:
            Base CTR
        """
        # Base CTR by ad type
        base_ctrs = {
            AdType.DISPLAY_BANNER: 0.005,  # 0.5%
            AdType.VIDEO_AD: 0.015,        # 1.5%
            AdType.NATIVE_AD: 0.012,       # 1.2%
            AdType.SPONSORED_POST: 0.020,  # 2.0%
            AdType.POPUP_AD: 0.003,        # 0.3%
            AdType.SOCIAL_MEDIA_AD: 0.018, # 1.8%
            AdType.SEARCH_AD: 0.035,       # 3.5%
            AdType.AFFILIATE_AD: 0.008     # 0.8%
        }
        
        base_ctr = base_ctrs.get(ad_type, 0.01)
        
        # Adjust based on target segments
        if target_segments:
            avg_receptivity = np.mean([
                self.market_segments[segment]['ad_receptivity']
                for segment in target_segments
                if segment in self.market_segments
            ])
            base_ctr *= avg_receptivity
        
        return base_ctr
    
    def _update_revenue(self, entity_id: str, revenue_source: RevenueSource, amount: float) -> None:
        """Update revenue for an entity.
        
        Args:
            entity_id: Entity identifier
            revenue_source: Source of revenue
            amount: Revenue amount
        """
        if entity_id not in self.financial_metrics:
            self.initialize_financial_metrics(entity_id)
        
        metrics = self.financial_metrics[entity_id]
        
        # Update total revenue
        metrics.total_revenue += amount
        metrics.monthly_revenue += amount  # Simplified - assume monthly tracking
        
        # Update revenue by source
        source_key = revenue_source.value
        if source_key not in metrics.revenue_by_source:
            metrics.revenue_by_source[source_key] = 0.0
        metrics.revenue_by_source[source_key] += amount
        
        # Update cash flow
        metrics.cash_flow += amount
        
        # Recalculate derived metrics
        metrics.calculate_profit_margin()
        metrics.calculate_sustainability_score()
        metrics.last_updated = datetime.now()
    
    def simulate_subscription_revenue(self, entity_id: str, subscriber_count: int, plan_distribution: Dict[str, float] = None) -> float:
        """Simulate subscription revenue.
        
        Args:
            entity_id: Entity identifier
            subscriber_count: Number of subscribers
            plan_distribution: Distribution of subscription plans
            
        Returns:
            Monthly subscription revenue
        """
        if plan_distribution is None:
            plan_distribution = {'basic': 0.7, 'premium': 0.25, 'enterprise': 0.05}
        
        total_revenue = 0.0
        
        for plan, ratio in plan_distribution.items():
            plan_subscribers = int(subscriber_count * ratio)
            plan_price = self.subscription_prices.get(plan, 100.0)
            plan_revenue = plan_subscribers * plan_price
            total_revenue += plan_revenue
        
        # Add some churn (subscription cancellations)
        churn_rate = random.uniform(0.05, 0.15)  # 5-15% monthly churn
        churned_revenue = total_revenue * churn_rate
        net_revenue = total_revenue - churned_revenue
        
        # Record revenue
        if net_revenue > 0:
            self._update_revenue(entity_id, RevenueSource.SUBSCRIPTION, net_revenue)
            
            # Record subscription events
            new_subscriptions = int(subscriber_count * 0.1)  # 10% new subscriptions
            cancelled_subscriptions = int(subscriber_count * churn_rate)
            
            for _ in range(new_subscriptions):
                event = EconomicEvent(
                    event_type=EconomicEventType.SUBSCRIPTION_PURCHASED,
                    entity_id=entity_id,
                    amount=random.choice(list(self.subscription_prices.values())),
                    revenue_source=RevenueSource.SUBSCRIPTION,
                    description="New subscription purchased"
                )
                self.economic_events.append(event)
            
            for _ in range(cancelled_subscriptions):
                event = EconomicEvent(
                    event_type=EconomicEventType.SUBSCRIPTION_CANCELLED,
                    entity_id=entity_id,
                    amount=-random.choice(list(self.subscription_prices.values())),
                    revenue_source=RevenueSource.SUBSCRIPTION,
                    description="Subscription cancelled"
                )
                self.economic_events.append(event)
        
        return net_revenue
    
    def simulate_operational_costs(self, entity_id: str, revenue: float, entity_type: str = 'news_outlet') -> float:
        """Simulate operational costs.
        
        Args:
            entity_id: Entity identifier
            revenue: Current revenue
            entity_type: Type of entity
            
        Returns:
            Total operational costs
        """
        # Base cost ratios by entity type
        cost_ratios = {
            'news_outlet': 0.8,     # 80% of revenue
            'fact_checker': 0.9,    # 90% of revenue (non-profit)
            'influencer': 0.4,      # 40% of revenue
            'platform': 0.7        # 70% of revenue
        }
        
        base_cost_ratio = cost_ratios.get(entity_type, 0.8)
        
        # Add randomness and market factors
        market_factor = random.uniform(0.9, 1.1)  # Market conditions
        efficiency_factor = random.uniform(0.8, 1.2)  # Operational efficiency
        
        total_costs = revenue * base_cost_ratio * market_factor * efficiency_factor
        
        # Break down costs by category
        cost_breakdown = {}
        for category, ratio in self.operational_cost_factors.items():
            cost_breakdown[category] = total_costs * ratio
        
        # Update financial metrics
        if entity_id not in self.financial_metrics:
            self.initialize_financial_metrics(entity_id, entity_type)
        
        metrics = self.financial_metrics[entity_id]
        metrics.total_costs += total_costs
        metrics.operational_costs += total_costs
        
        # Update specific cost categories
        metrics.content_costs += cost_breakdown['content_creation']
        metrics.marketing_costs += cost_breakdown['marketing']
        
        # Update cash flow
        metrics.cash_flow -= total_costs
        
        # Record cost event
        event = EconomicEvent(
            event_type=EconomicEventType.COST_INCURRED,
            entity_id=entity_id,
            amount=total_costs,
            description="Operational costs",
            metadata=cost_breakdown
        )
        self.economic_events.append(event)
        
        return total_costs
    
    def simulate_economic_shock(self, shock_type: str, magnitude: float, affected_entities: List[str] = None) -> None:
        """Simulate economic shock events.
        
        Args:
            shock_type: Type of economic shock
            magnitude: Magnitude of impact (-1 to 1)
            affected_entities: List of affected entities (None for all)
        """
        if affected_entities is None:
            affected_entities = list(self.financial_metrics.keys())
        
        shock_effects = {
            'recession': {
                'revenue_impact': -abs(magnitude),
                'cost_impact': magnitude * 0.5,
                'duration_days': 90
            },
            'inflation': {
                'revenue_impact': magnitude * 0.3,
                'cost_impact': abs(magnitude),
                'duration_days': 180
            },
            'market_boom': {
                'revenue_impact': abs(magnitude),
                'cost_impact': magnitude * 0.3,
                'duration_days': 60
            },
            'regulatory_change': {
                'revenue_impact': magnitude,
                'cost_impact': abs(magnitude) * 0.5,
                'duration_days': 30
            },
            'technology_disruption': {
                'revenue_impact': magnitude,
                'cost_impact': -magnitude * 0.2,
                'duration_days': 365
            }
        }
        
        effects = shock_effects.get(shock_type, shock_effects['recession'])
        
        for entity_id in affected_entities:
            if entity_id in self.financial_metrics:
                metrics = self.financial_metrics[entity_id]
                
                # Apply revenue impact
                revenue_change = metrics.monthly_revenue * effects['revenue_impact']
                metrics.monthly_revenue += revenue_change
                metrics.total_revenue += revenue_change
                
                # Apply cost impact
                cost_change = metrics.operational_costs * effects['cost_impact']
                metrics.operational_costs += cost_change
                metrics.total_costs += cost_change
                
                # Update cash flow
                metrics.cash_flow += revenue_change - cost_change
                
                # Recalculate metrics
                metrics.calculate_profit_margin()
                metrics.calculate_sustainability_score()
                
                # Record shock event
                event = EconomicEvent(
                    event_type=EconomicEventType.ECONOMIC_SHOCK,
                    entity_id=entity_id,
                    amount=revenue_change - cost_change,
                    description=f"Economic shock: {shock_type}",
                    metadata={
                        'shock_type': shock_type,
                        'magnitude': magnitude,
                        'revenue_change': revenue_change,
                        'cost_change': cost_change,
                        'duration_days': effects['duration_days']
                    }
                )
                self.economic_events.append(event)
        
        logger.info(f"Applied economic shock '{shock_type}' with magnitude {magnitude} to {len(affected_entities)} entities")
    
    def calculate_market_metrics(self) -> Dict[str, Any]:
        """Calculate overall market metrics.
        
        Returns:
            Market metrics dictionary
        """
        if not self.financial_metrics:
            return {'total_entities': 0}
        
        # Aggregate financial data
        total_revenue = sum(metrics.total_revenue for metrics in self.financial_metrics.values())
        total_costs = sum(metrics.total_costs for metrics in self.financial_metrics.values())
        total_profit = total_revenue - total_costs
        
        # Revenue by source
        revenue_by_source = defaultdict(float)
        for metrics in self.financial_metrics.values():
            for source, amount in metrics.revenue_by_source.items():
                revenue_by_source[source] += amount
        
        # Profitability distribution
        profit_margins = [metrics.profit_margin for metrics in self.financial_metrics.values()]
        profitable_entities = sum(1 for margin in profit_margins if margin > 0)
        
        # Sustainability scores
        sustainability_scores = [metrics.sustainability_score for metrics in self.financial_metrics.values()]
        
        # Ad campaign metrics
        active_campaigns = sum(1 for campaign in self.ad_campaigns.values() if campaign.is_active)
        total_ad_spend = sum(campaign.budget for campaign in self.ad_campaigns.values())
        
        # Recent economic activity
        recent_events = [
            event for event in self.economic_events
            if event.timestamp >= datetime.now() - timedelta(days=30)
        ]
        
        market_metrics = {
            'total_entities': len(self.financial_metrics),
            'total_market_revenue': total_revenue,
            'total_market_costs': total_costs,
            'total_market_profit': total_profit,
            'market_profit_margin': total_profit / total_revenue if total_revenue > 0 else 0,
            'profitable_entities': profitable_entities,
            'profitability_rate': profitable_entities / len(self.financial_metrics),
            'avg_sustainability_score': np.mean(sustainability_scores),
            'revenue_by_source': dict(revenue_by_source),
            'active_ad_campaigns': active_campaigns,
            'total_ad_spend': total_ad_spend,
            'recent_economic_events': len(recent_events),
            'market_concentration': self._calculate_market_concentration(),
            'economic_health_score': self._calculate_economic_health_score()
        }
        
        return market_metrics
    
    def _calculate_market_concentration(self) -> float:
        """Calculate market concentration (Herfindahl-Hirschman Index).
        
        Returns:
            Market concentration index (0-1)
        """
        if not self.financial_metrics:
            return 0.0
        
        revenues = [metrics.total_revenue for metrics in self.financial_metrics.values()]
        total_revenue = sum(revenues)
        
        if total_revenue == 0:
            return 0.0
        
        # Calculate market shares
        market_shares = [revenue / total_revenue for revenue in revenues]
        
        # Calculate HHI
        hhi = sum(share ** 2 for share in market_shares)
        
        return hhi
    
    def _calculate_economic_health_score(self) -> float:
        """Calculate overall economic health score.
        
        Returns:
            Economic health score (0-1)
        """
        if not self.financial_metrics:
            return 0.5
        
        # Component scores
        sustainability_scores = [metrics.sustainability_score for metrics in self.financial_metrics.values()]
        profit_margins = [max(0, min(1, metrics.profit_margin + 0.5)) for metrics in self.financial_metrics.values()]
        
        # Market diversity (lower concentration is better)
        concentration = self._calculate_market_concentration()
        diversity_score = 1.0 - concentration
        
        # Growth indicators
        growth_rates = [max(0, min(1, metrics.revenue_growth_rate + 0.5)) for metrics in self.financial_metrics.values()]
        
        # Weighted health score
        health_score = (
            0.4 * np.mean(sustainability_scores) +
            0.3 * np.mean(profit_margins) +
            0.2 * diversity_score +
            0.1 * np.mean(growth_rates)
        )
        
        return health_score
    
    def get_entity_financial_summary(self, entity_id: str) -> Dict[str, Any]:
        """Get financial summary for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Financial summary
        """
        if entity_id not in self.financial_metrics:
            return {'error': 'Entity not found'}
        
        metrics = self.financial_metrics[entity_id]
        
        # Recent events for this entity
        entity_events = [
            event for event in self.economic_events
            if event.entity_id == entity_id and
            event.timestamp >= datetime.now() - timedelta(days=30)
        ]
        
        # Revenue trend (simplified)
        revenue_trend = 'stable'
        if metrics.revenue_growth_rate > 0.1:
            revenue_trend = 'growing'
        elif metrics.revenue_growth_rate < -0.1:
            revenue_trend = 'declining'
        
        # Financial health assessment
        if metrics.sustainability_score > 0.7:
            health_status = 'excellent'
        elif metrics.sustainability_score > 0.5:
            health_status = 'good'
        elif metrics.sustainability_score > 0.3:
            health_status = 'fair'
        else:
            health_status = 'poor'
        
        summary = {
            'entity_id': entity_id,
            'monthly_revenue': metrics.monthly_revenue,
            'annual_revenue': metrics.total_revenue,
            'monthly_costs': metrics.operational_costs,
            'profit_margin': metrics.profit_margin,
            'cash_flow': metrics.cash_flow,
            'sustainability_score': metrics.sustainability_score,
            'revenue_growth_rate': metrics.revenue_growth_rate,
            'revenue_by_source': metrics.revenue_by_source,
            'revenue_trend': revenue_trend,
            'health_status': health_status,
            'recent_events_count': len(entity_events),
            'last_updated': metrics.last_updated.isoformat()
        }
        
        return summary


# Export main classes
__all__ = [
    'EconomicModel', 'FinancialMetrics', 'AdCampaign', 'EconomicEvent',
    'RevenueSource', 'AdType', 'EconomicEventType', 'MarketSegment'
]