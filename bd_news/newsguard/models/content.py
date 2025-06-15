"""Content models for NewsGuard Bangladesh simulation.

This module implements models for news content, misinformation detection,
fact-checking processes, and content analysis in the Bangladesh media ecosystem.
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import math

import numpy as np
from scipy import stats
import networkx as nx

from ..utils.logger import get_logger
from ..utils.nlp import BanglaTextProcessor, ContentAnalyzer
from ..utils.helpers import (
    calculate_statistics, normalize_scores, exponential_decay,
    sigmoid_function, weighted_random_choice
)

logger = get_logger(__name__)


class ContentType(Enum):
    """Types of content in the simulation."""
    NEWS_ARTICLE = "news_article"
    OPINION_PIECE = "opinion_piece"
    SOCIAL_POST = "social_post"
    VIDEO_CONTENT = "video_content"
    IMAGE_CONTENT = "image_content"
    ADVERTISEMENT = "advertisement"
    FACT_CHECK = "fact_check"
    CORRECTION = "correction"
    RETRACTION = "retraction"


class ContentStatus(Enum):
    """Status of content in the system."""
    DRAFT = "draft"
    PUBLISHED = "published"
    UNDER_REVIEW = "under_review"
    FACT_CHECKING = "fact_checking"
    FLAGGED = "flagged"
    REMOVED = "removed"
    CORRECTED = "corrected"
    VERIFIED = "verified"


class MisinformationType(Enum):
    """Types of misinformation."""
    FALSE_INFORMATION = "false_information"
    MISLEADING_CONTEXT = "misleading_context"
    MANIPULATED_CONTENT = "manipulated_content"
    FABRICATED_CONTENT = "fabricated_content"
    SATIRE_PARODY = "satire_parody"
    CONSPIRACY_THEORY = "conspiracy_theory"
    PROPAGANDA = "propaganda"
    CLICKBAIT = "clickbait"


@dataclass
class ContentMetadata:
    """Metadata for content items."""
    content_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    summary: str = ""
    author_id: str = ""
    publisher_id: str = ""
    source_url: str = ""
    language: str = "bn"  # Bengali by default
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Content characteristics
    content_type: ContentType = ContentType.NEWS_ARTICLE
    status: ContentStatus = ContentStatus.DRAFT
    word_count: int = 0
    reading_time: int = 0  # in minutes
    
    # Topics and categories
    topics: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    
    # Geographic and demographic targeting
    target_regions: List[str] = field(default_factory=list)
    target_demographics: Dict[str, Any] = field(default_factory=dict)
    
    # Technical metadata
    content_hash: str = ""
    version: int = 1
    parent_content_id: Optional[str] = None  # For corrections/updates
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.content_hash and self.title:
            self.content_hash = hashlib.md5(f"{self.title}_{self.created_at}".encode()).hexdigest()


@dataclass
class ContentEngagement:
    """Engagement metrics for content."""
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    reactions: Dict[str, int] = field(default_factory=dict)
    
    # Time-based engagement
    hourly_views: List[int] = field(default_factory=list)
    daily_views: List[int] = field(default_factory=list)
    
    # Engagement quality
    avg_time_spent: float = 0.0  # seconds
    bounce_rate: float = 0.0
    completion_rate: float = 0.0
    
    # Social metrics
    reach: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0
    viral_coefficient: float = 0.0
    
    @property
    def total_engagement(self) -> int:
        """Calculate total engagement score."""
        return self.views + self.likes + self.shares + self.comments
    
    @property
    def engagement_score(self) -> float:
        """Calculate weighted engagement score."""
        weights = {'views': 1, 'likes': 2, 'shares': 3, 'comments': 2}
        score = (self.views * weights['views'] + 
                self.likes * weights['likes'] + 
                self.shares * weights['shares'] + 
                self.comments * weights['comments'])
        return score


class ContentModel:
    """Base model for content in the simulation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize content model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Content storage
        self.content_registry: Dict[str, 'Content'] = {}
        self.content_by_author: Dict[str, List[str]] = {}
        self.content_by_topic: Dict[str, List[str]] = {}
        
        # NLP processors
        self.text_processor = BanglaTextProcessor()
        self.content_analyzer = ContentAnalyzer()
        
        # Model parameters
        self.credibility_threshold = self.config.get('credibility_threshold', 0.7)
        self.viral_threshold = self.config.get('viral_threshold', 1000)
        self.content_decay_rate = self.config.get('content_decay_rate', 0.1)
        
        logger.debug("Content model initialized")
    
    def create_content(self, 
                      text: str,
                      metadata: ContentMetadata,
                      is_misinformation: bool = False,
                      misinformation_type: Optional[MisinformationType] = None) -> 'Content':
        """Create new content item.
        
        Args:
            text: Content text
            metadata: Content metadata
            is_misinformation: Whether content is misinformation
            misinformation_type: Type of misinformation if applicable
            
        Returns:
            Created content object
        """
        content = Content(
            text=text,
            metadata=metadata,
            is_misinformation=is_misinformation,
            misinformation_type=misinformation_type
        )
        
        # Process content
        self._process_content(content)
        
        # Register content
        self.content_registry[content.metadata.content_id] = content
        
        # Index by author
        author_id = content.metadata.author_id
        if author_id not in self.content_by_author:
            self.content_by_author[author_id] = []
        self.content_by_author[author_id].append(content.metadata.content_id)
        
        # Index by topics
        for topic in content.metadata.topics:
            if topic not in self.content_by_topic:
                self.content_by_topic[topic] = []
            self.content_by_topic[topic].append(content.metadata.content_id)
        
        logger.debug(f"Created content: {content.metadata.content_id}")
        
        return content
    
    def _process_content(self, content: 'Content') -> None:
        """Process content for analysis and scoring.
        
        Args:
            content: Content to process
        """
        # Extract text features
        processed_text = self.text_processor.process_text(content.text)
        content.processed_text = processed_text
        
        # Update metadata
        content.metadata.word_count = len(content.text.split())
        content.metadata.reading_time = max(1, content.metadata.word_count // 200)
        
        # Extract keywords and entities
        content.metadata.keywords = self.text_processor.extract_keywords(content.text)
        content.metadata.entities = self.content_analyzer.extract_entities(content.text)
        
        # Analyze content
        analysis = self.content_analyzer.analyze_content(content.text)
        content.sentiment_score = analysis.get('sentiment', 0.0)
        content.emotion_scores = analysis.get('emotions', {})
        content.readability_score = analysis.get('readability', 0.5)
        
        # Calculate initial credibility score
        content.credibility_score = self._calculate_credibility(content)
        
        # Detect potential misinformation indicators
        content.misinformation_indicators = self._detect_misinformation_indicators(content)
    
    def _calculate_credibility(self, content: 'Content') -> float:
        """Calculate content credibility score.
        
        Args:
            content: Content to analyze
            
        Returns:
            Credibility score (0-1)
        """
        factors = []
        
        # Source credibility (if available)
        # This would be based on publisher/author reputation
        source_credibility = 0.7  # Placeholder
        factors.append(source_credibility)
        
        # Content quality indicators
        if content.metadata.word_count > 100:  # Substantial content
            factors.append(0.8)
        else:
            factors.append(0.4)
        
        # Readability (moderate readability is better)
        readability_factor = 1 - abs(content.readability_score - 0.6)
        factors.append(readability_factor)
        
        # Sentiment extremity (extreme sentiment may indicate bias)
        sentiment_factor = 1 - abs(content.sentiment_score)
        factors.append(sentiment_factor)
        
        # Misinformation indicators (negative impact)
        indicator_penalty = len(content.misinformation_indicators) * 0.1
        base_score = np.mean(factors)
        
        credibility = max(0.0, base_score - indicator_penalty)
        
        return credibility
    
    def _detect_misinformation_indicators(self, content: 'Content') -> List[str]:
        """Detect potential misinformation indicators.
        
        Args:
            content: Content to analyze
            
        Returns:
            List of detected indicators
        """
        indicators = []
        
        text = content.text.lower()
        
        # Sensational language indicators
        sensational_words = ['shocking', 'unbelievable', 'secret', 'hidden truth', 
                            'they don\'t want you to know', 'exclusive']
        if any(word in text for word in sensational_words):
            indicators.append('sensational_language')
        
        # Emotional manipulation
        if content.sentiment_score > 0.8 or content.sentiment_score < -0.8:
            indicators.append('extreme_sentiment')
        
        # Lack of sources
        source_indicators = ['according to', 'source:', 'study shows', 'research']
        if not any(indicator in text for indicator in source_indicators):
            indicators.append('no_sources')
        
        # Conspiracy language
        conspiracy_words = ['conspiracy', 'cover-up', 'they', 'agenda']
        if any(word in text for word in conspiracy_words):
            indicators.append('conspiracy_language')
        
        # All caps (shouting)
        if len([word for word in content.text.split() if word.isupper()]) > 3:
            indicators.append('excessive_caps')
        
        return indicators
    
    def update_engagement(self, content_id: str, engagement_data: Dict[str, Any]) -> None:
        """Update content engagement metrics.
        
        Args:
            content_id: Content identifier
            engagement_data: New engagement data
        """
        if content_id not in self.content_registry:
            logger.warning(f"Content not found: {content_id}")
            return
        
        content = self.content_registry[content_id]
        
        # Update engagement metrics
        for metric, value in engagement_data.items():
            if hasattr(content.engagement, metric):
                setattr(content.engagement, metric, value)
        
        # Recalculate derived metrics
        content.engagement.engagement_rate = self._calculate_engagement_rate(content)
        content.engagement.viral_coefficient = self._calculate_viral_coefficient(content)
        
        # Update content popularity score
        content.popularity_score = self._calculate_popularity(content)
        
        logger.debug(f"Updated engagement for content: {content_id}")
    
    def _calculate_engagement_rate(self, content: 'Content') -> float:
        """Calculate engagement rate.
        
        Args:
            content: Content object
            
        Returns:
            Engagement rate
        """
        if content.engagement.impressions == 0:
            return 0.0
        
        total_engagement = (content.engagement.likes + 
                          content.engagement.shares + 
                          content.engagement.comments)
        
        return total_engagement / content.engagement.impressions
    
    def _calculate_viral_coefficient(self, content: 'Content') -> float:
        """Calculate viral coefficient.
        
        Args:
            content: Content object
            
        Returns:
            Viral coefficient
        """
        if content.engagement.views == 0:
            return 0.0
        
        return content.engagement.shares / content.engagement.views
    
    def _calculate_popularity(self, content: 'Content') -> float:
        """Calculate content popularity score.
        
        Args:
            content: Content object
            
        Returns:
            Popularity score (0-1)
        """
        # Time decay factor
        age_hours = (datetime.now() - content.metadata.created_at).total_seconds() / 3600
        time_decay = exponential_decay(age_hours, self.content_decay_rate)
        
        # Engagement factor
        engagement_score = content.engagement.engagement_score
        max_engagement = self.viral_threshold
        engagement_factor = min(1.0, engagement_score / max_engagement)
        
        # Combine factors
        popularity = engagement_factor * time_decay
        
        return popularity
    
    def get_trending_content(self, limit: int = 10, time_window: int = 24) -> List['Content']:
        """Get trending content.
        
        Args:
            limit: Maximum number of items to return
            time_window: Time window in hours
            
        Returns:
            List of trending content
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window)
        
        # Filter recent content
        recent_content = [
            content for content in self.content_registry.values()
            if content.metadata.created_at >= cutoff_time
        ]
        
        # Sort by popularity
        trending = sorted(recent_content, 
                         key=lambda x: x.popularity_score, 
                         reverse=True)
        
        return trending[:limit]
    
    def get_content_by_author(self, author_id: str) -> List['Content']:
        """Get content by author.
        
        Args:
            author_id: Author identifier
            
        Returns:
            List of content by author
        """
        content_ids = self.content_by_author.get(author_id, [])
        return [self.content_registry[cid] for cid in content_ids 
                if cid in self.content_registry]
    
    def get_content_by_topic(self, topic: str) -> List['Content']:
        """Get content by topic.
        
        Args:
            topic: Topic name
            
        Returns:
            List of content on topic
        """
        content_ids = self.content_by_topic.get(topic, [])
        return [self.content_registry[cid] for cid in content_ids 
                if cid in self.content_registry]
    
    def search_content(self, query: str, filters: Dict[str, Any] = None) -> List['Content']:
        """Search content.
        
        Args:
            query: Search query
            filters: Additional filters
            
        Returns:
            List of matching content
        """
        results = []
        query_lower = query.lower()
        
        for content in self.content_registry.values():
            # Text matching
            if (query_lower in content.text.lower() or 
                query_lower in content.metadata.title.lower() or
                any(query_lower in keyword.lower() for keyword in content.metadata.keywords)):
                
                # Apply filters
                if filters:
                    if not self._apply_filters(content, filters):
                        continue
                
                results.append(content)
        
        # Sort by relevance (popularity for now)
        results.sort(key=lambda x: x.popularity_score, reverse=True)
        
        return results
    
    def _apply_filters(self, content: 'Content', filters: Dict[str, Any]) -> bool:
        """Apply filters to content.
        
        Args:
            content: Content to filter
            filters: Filter criteria
            
        Returns:
            True if content passes filters
        """
        # Content type filter
        if 'content_type' in filters:
            if content.metadata.content_type != filters['content_type']:
                return False
        
        # Author filter
        if 'author_id' in filters:
            if content.metadata.author_id != filters['author_id']:
                return False
        
        # Date range filter
        if 'start_date' in filters:
            if content.metadata.created_at < filters['start_date']:
                return False
        
        if 'end_date' in filters:
            if content.metadata.created_at > filters['end_date']:
                return False
        
        # Credibility filter
        if 'min_credibility' in filters:
            if content.credibility_score < filters['min_credibility']:
                return False
        
        # Topic filter
        if 'topics' in filters:
            if not any(topic in content.metadata.topics for topic in filters['topics']):
                return False
        
        return True
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """Get content statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_content = len(self.content_registry)
        
        if total_content == 0:
            return {'total_content': 0}
        
        # Count by type
        type_counts = {}
        for content in self.content_registry.values():
            content_type = content.metadata.content_type.value
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        # Count by status
        status_counts = {}
        for content in self.content_registry.values():
            status = content.metadata.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Misinformation statistics
        misinformation_count = sum(1 for c in self.content_registry.values() if c.is_misinformation)
        
        # Engagement statistics
        engagement_scores = [c.engagement.engagement_score for c in self.content_registry.values()]
        credibility_scores = [c.credibility_score for c in self.content_registry.values()]
        
        stats = {
            'total_content': total_content,
            'content_by_type': type_counts,
            'content_by_status': status_counts,
            'misinformation_count': misinformation_count,
            'misinformation_ratio': misinformation_count / total_content,
            'avg_engagement_score': np.mean(engagement_scores) if engagement_scores else 0,
            'avg_credibility_score': np.mean(credibility_scores) if credibility_scores else 0,
            'total_authors': len(self.content_by_author),
            'total_topics': len(self.content_by_topic)
        }
        
        return stats


class Content:
    """Individual content item."""
    
    def __init__(self, 
                 text: str,
                 metadata: ContentMetadata,
                 is_misinformation: bool = False,
                 misinformation_type: Optional[MisinformationType] = None):
        """Initialize content.
        
        Args:
            text: Content text
            metadata: Content metadata
            is_misinformation: Whether content is misinformation
            misinformation_type: Type of misinformation
        """
        self.text = text
        self.metadata = metadata
        self.is_misinformation = is_misinformation
        self.misinformation_type = misinformation_type
        
        # Processed content
        self.processed_text = ""
        
        # Scores and metrics
        self.credibility_score = 0.5
        self.popularity_score = 0.0
        self.sentiment_score = 0.0
        self.readability_score = 0.5
        self.emotion_scores: Dict[str, float] = {}
        
        # Engagement
        self.engagement = ContentEngagement()
        
        # Analysis results
        self.misinformation_indicators: List[str] = []
        self.fact_check_results: List['FactCheckResult'] = []
        
        # Relationships
        self.related_content: List[str] = []  # Content IDs
        self.corrections: List[str] = []  # Correction content IDs
        
        # Tracking
        self.view_history: List[Dict[str, Any]] = []
        self.share_history: List[Dict[str, Any]] = []
    
    def add_fact_check(self, fact_check_result: 'FactCheckResult') -> None:
        """Add fact-check result.
        
        Args:
            fact_check_result: Fact-check result
        """
        self.fact_check_results.append(fact_check_result)
        
        # Update status if fact-checked
        if fact_check_result.verdict in ['false', 'misleading']:
            self.metadata.status = ContentStatus.FLAGGED
        elif fact_check_result.verdict == 'true':
            self.metadata.status = ContentStatus.VERIFIED
    
    def get_latest_fact_check(self) -> Optional['FactCheckResult']:
        """Get latest fact-check result.
        
        Returns:
            Latest fact-check result or None
        """
        if not self.fact_check_results:
            return None
        
        return max(self.fact_check_results, key=lambda x: x.checked_at)
    
    def is_viral(self, threshold: int = 1000) -> bool:
        """Check if content is viral.
        
        Args:
            threshold: Viral threshold
            
        Returns:
            True if content is viral
        """
        return self.engagement.total_engagement >= threshold
    
    def get_age_hours(self) -> float:
        """Get content age in hours.
        
        Returns:
            Age in hours
        """
        return (datetime.now() - self.metadata.created_at).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert content to dictionary.
        
        Returns:
            Content as dictionary
        """
        return {
            'content_id': self.metadata.content_id,
            'title': self.metadata.title,
            'text': self.text,
            'author_id': self.metadata.author_id,
            'publisher_id': self.metadata.publisher_id,
            'content_type': self.metadata.content_type.value,
            'status': self.metadata.status.value,
            'created_at': self.metadata.created_at.isoformat(),
            'is_misinformation': self.is_misinformation,
            'misinformation_type': self.misinformation_type.value if self.misinformation_type else None,
            'credibility_score': self.credibility_score,
            'popularity_score': self.popularity_score,
            'engagement': {
                'views': self.engagement.views,
                'likes': self.engagement.likes,
                'shares': self.engagement.shares,
                'comments': self.engagement.comments,
                'engagement_rate': self.engagement.engagement_rate
            },
            'topics': self.metadata.topics,
            'keywords': self.metadata.keywords
        }


@dataclass
class FactCheckResult:
    """Result of fact-checking process."""
    content_id: str
    fact_checker_id: str
    verdict: str  # 'true', 'false', 'misleading', 'unverified'
    confidence: float  # 0-1
    explanation: str
    sources: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)
    
    # Detailed analysis
    claims_analyzed: List[str] = field(default_factory=list)
    evidence_found: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    
    # Impact assessment
    potential_harm: str = "low"  # low, medium, high
    urgency: str = "normal"  # low, normal, high, critical


class MisinformationModel:
    """Model for misinformation generation and propagation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize misinformation model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Misinformation patterns
        self.misinformation_templates = self._load_misinformation_templates()
        self.topic_vulnerabilities = self._load_topic_vulnerabilities()
        
        # Generation parameters
        self.generation_rate = self.config.get('generation_rate', 0.1)
        self.mutation_rate = self.config.get('mutation_rate', 0.05)
        
        logger.debug("Misinformation model initialized")
    
    def _load_misinformation_templates(self) -> Dict[str, List[str]]:
        """Load misinformation templates by type.
        
        Returns:
            Dictionary of templates by misinformation type
        """
        # In a real implementation, these would be loaded from files
        templates = {
            'false_information': [
                "Breaking: {entity} announces {false_claim}",
                "Exclusive: {authority} confirms {misleading_fact}",
                "Shocking revelation: {topic} linked to {conspiracy}"
            ],
            'misleading_context': [
                "Old photo shows {event} happening now",
                "Statistics from {old_year} prove {current_claim}",
                "Video of {different_event} shows {current_situation}"
            ],
            'conspiracy_theory': [
                "Hidden truth: {authority} doesn't want you to know about {secret}",
                "Secret agenda: {group} planning {conspiracy}",
                "Cover-up exposed: {event} was actually {alternative_explanation}"
            ]
        }
        
        return templates
    
    def _load_topic_vulnerabilities(self) -> Dict[str, float]:
        """Load topic vulnerability scores.
        
        Returns:
            Dictionary of vulnerability scores by topic
        """
        # Topics more susceptible to misinformation
        vulnerabilities = {
            'politics': 0.8,
            'health': 0.7,
            'religion': 0.6,
            'economy': 0.5,
            'technology': 0.3,
            'sports': 0.2,
            'entertainment': 0.4
        }
        
        return vulnerabilities
    
    def generate_misinformation(self, 
                              topic: str,
                              misinformation_type: MisinformationType,
                              context: Dict[str, Any] = None) -> str:
        """Generate misinformation content.
        
        Args:
            topic: Topic for misinformation
            misinformation_type: Type of misinformation
            context: Additional context for generation
            
        Returns:
            Generated misinformation text
        """
        context = context or {}
        
        # Get templates for this type
        templates = self.misinformation_templates.get(misinformation_type.value, [])
        
        if not templates:
            return f"Misleading information about {topic}"
        
        # Select random template
        template = random.choice(templates)
        
        # Fill template with context
        try:
            # Default context values
            default_context = {
                'entity': 'Government',
                'authority': 'Official source',
                'topic': topic,
                'false_claim': 'controversial decision',
                'misleading_fact': 'new policy',
                'conspiracy': 'hidden agenda',
                'event': 'recent incident',
                'old_year': '2020',
                'current_claim': 'current situation',
                'different_event': 'unrelated event',
                'current_situation': 'ongoing crisis',
                'secret': 'important information',
                'group': 'powerful organization',
                'alternative_explanation': 'different cause'
            }
            
            # Update with provided context
            default_context.update(context)
            
            # Format template
            misinformation_text = template.format(**default_context)
            
        except KeyError as e:
            logger.warning(f"Missing context key for template: {e}")
            misinformation_text = f"Misleading information about {topic}"
        
        return misinformation_text
    
    def calculate_spread_probability(self, 
                                   content: Content,
                                   network_context: Dict[str, Any] = None) -> float:
        """Calculate probability of misinformation spread.
        
        Args:
            content: Content to analyze
            network_context: Network context for spread calculation
            
        Returns:
            Spread probability (0-1)
        """
        factors = []
        
        # Topic vulnerability
        topic_vulnerability = 0.5
        if content.metadata.topics:
            topic_vulnerabilities = [self.topic_vulnerabilities.get(topic, 0.5) 
                                   for topic in content.metadata.topics]
            topic_vulnerability = max(topic_vulnerabilities)
        factors.append(topic_vulnerability)
        
        # Emotional appeal (extreme sentiment spreads faster)
        emotion_factor = abs(content.sentiment_score)
        factors.append(emotion_factor)
        
        # Sensational indicators
        sensational_factor = len(content.misinformation_indicators) * 0.1
        factors.append(min(1.0, sensational_factor))
        
        # Network factors
        if network_context:
            network_density = network_context.get('density', 0.1)
            factors.append(network_density)
            
            # Influencer involvement
            has_influencer = network_context.get('has_influencer', False)
            if has_influencer:
                factors.append(0.8)
        
        # Calculate weighted probability
        base_probability = np.mean(factors)
        
        # Apply sigmoid to normalize
        spread_probability = sigmoid_function(base_probability * 10 - 5)
        
        return spread_probability
    
    def mutate_misinformation(self, original_content: Content) -> str:
        """Create a mutated version of misinformation.
        
        Args:
            original_content: Original misinformation content
            
        Returns:
            Mutated misinformation text
        """
        original_text = original_content.text
        
        # Simple mutation strategies
        mutations = [
            self._add_urgency,
            self._change_numbers,
            self._add_authority,
            self._change_location,
            self._add_emotional_language
        ]
        
        # Apply random mutation
        mutation_func = random.choice(mutations)
        mutated_text = mutation_func(original_text)
        
        return mutated_text
    
    def _add_urgency(self, text: str) -> str:
        """Add urgency to text."""
        urgency_phrases = ["URGENT: ", "BREAKING: ", "ALERT: ", "IMMEDIATE: "]
        return random.choice(urgency_phrases) + text
    
    def _change_numbers(self, text: str) -> str:
        """Change numbers in text."""
        import re
        
        def replace_number(match):
            num = int(match.group())
            # Increase by 10-50%
            multiplier = random.uniform(1.1, 1.5)
            return str(int(num * multiplier))
        
        return re.sub(r'\d+', replace_number, text)
    
    def _add_authority(self, text: str) -> str:
        """Add authority reference."""
        authorities = ["According to experts, ", "Officials confirm that ", 
                      "Reliable sources say ", "Insiders reveal that "]
        return random.choice(authorities) + text.lower()
    
    def _change_location(self, text: str) -> str:
        """Change location references."""
        # Simple location substitution
        locations = ['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna']
        for location in locations:
            if location in text:
                new_location = random.choice([l for l in locations if l != location])
                text = text.replace(location, new_location)
                break
        return text
    
    def _add_emotional_language(self, text: str) -> str:
        """Add emotional language."""
        emotional_words = ['shocking', 'devastating', 'incredible', 'unbelievable']
        words = text.split()
        if len(words) > 3:
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, random.choice(emotional_words))
        return ' '.join(words)


class FactCheckModel:
    """Model for fact-checking processes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize fact-check model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        # Fact-checking parameters
        self.accuracy_rate = self.config.get('accuracy_rate', 0.85)
        self.response_time_hours = self.config.get('response_time_hours', 24)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Fact-checking resources
        self.fact_checkers: Dict[str, Dict[str, Any]] = {}
        self.pending_checks: List[str] = []
        self.completed_checks: Dict[str, FactCheckResult] = {}
        
        logger.debug("Fact-check model initialized")
    
    def register_fact_checker(self, 
                            fact_checker_id: str,
                            expertise: List[str],
                            accuracy_rate: float = 0.85) -> None:
        """Register a fact-checker.
        
        Args:
            fact_checker_id: Fact-checker identifier
            expertise: List of expertise areas
            accuracy_rate: Fact-checker accuracy rate
        """
        self.fact_checkers[fact_checker_id] = {
            'expertise': expertise,
            'accuracy_rate': accuracy_rate,
            'checks_completed': 0,
            'avg_response_time': self.response_time_hours
        }
        
        logger.debug(f"Registered fact-checker: {fact_checker_id}")
    
    def submit_for_fact_check(self, content: Content, priority: str = 'normal') -> bool:
        """Submit content for fact-checking.
        
        Args:
            content: Content to fact-check
            priority: Priority level (low, normal, high, critical)
            
        Returns:
            True if submitted successfully
        """
        # Check if already being fact-checked
        if content.metadata.content_id in self.pending_checks:
            return False
        
        # Determine if content needs fact-checking
        if self._should_fact_check(content):
            self.pending_checks.append(content.metadata.content_id)
            content.metadata.status = ContentStatus.FACT_CHECKING
            
            logger.debug(f"Submitted for fact-check: {content.metadata.content_id}")
            return True
        
        return False
    
    def _should_fact_check(self, content: Content) -> bool:
        """Determine if content should be fact-checked.
        
        Args:
            content: Content to evaluate
            
        Returns:
            True if content should be fact-checked
        """
        # Check for misinformation indicators
        if len(content.misinformation_indicators) > 2:
            return True
        
        # Check credibility score
        if content.credibility_score < self.confidence_threshold:
            return True
        
        # Check if content is viral
        if content.is_viral():
            return True
        
        # Check for controversial topics
        controversial_topics = ['politics', 'health', 'religion']
        if any(topic in controversial_topics for topic in content.metadata.topics):
            return True
        
        return False
    
    def perform_fact_check(self, 
                         content_id: str,
                         fact_checker_id: str) -> Optional[FactCheckResult]:
        """Perform fact-checking on content.
        
        Args:
            content_id: Content to fact-check
            fact_checker_id: Fact-checker performing the check
            
        Returns:
            Fact-check result or None if failed
        """
        if fact_checker_id not in self.fact_checkers:
            logger.warning(f"Unknown fact-checker: {fact_checker_id}")
            return None
        
        if content_id not in self.pending_checks:
            logger.warning(f"Content not pending fact-check: {content_id}")
            return None
        
        fact_checker = self.fact_checkers[fact_checker_id]
        
        # Simulate fact-checking process
        result = self._simulate_fact_check(content_id, fact_checker)
        
        # Store result
        self.completed_checks[content_id] = result
        
        # Remove from pending
        self.pending_checks.remove(content_id)
        
        # Update fact-checker stats
        fact_checker['checks_completed'] += 1
        
        logger.debug(f"Fact-check completed: {content_id} by {fact_checker_id}")
        
        return result
    
    def _simulate_fact_check(self, 
                           content_id: str,
                           fact_checker: Dict[str, Any]) -> FactCheckResult:
        """Simulate fact-checking process.
        
        Args:
            content_id: Content ID
            fact_checker: Fact-checker information
            
        Returns:
            Fact-check result
        """
        # Simulate accuracy based on fact-checker's accuracy rate
        is_accurate = random.random() < fact_checker['accuracy_rate']
        
        # Determine verdict
        verdicts = ['true', 'false', 'misleading', 'unverified']
        verdict_weights = [0.3, 0.4, 0.2, 0.1]  # Bias towards false/misleading
        
        if is_accurate:
            verdict = weighted_random_choice(verdicts, verdict_weights)
        else:
            # Inaccurate fact-check - random verdict
            verdict = random.choice(verdicts)
        
        # Generate confidence score
        base_confidence = fact_checker['accuracy_rate']
        confidence = max(0.1, min(1.0, base_confidence + random.uniform(-0.2, 0.2)))
        
        # Generate explanation
        explanations = {
            'true': "The claims in this content are supported by reliable sources and evidence.",
            'false': "The claims in this content are contradicted by reliable sources and evidence.",
            'misleading': "While some elements may be true, the content presents misleading information.",
            'unverified': "Insufficient evidence available to verify the claims in this content."
        }
        
        explanation = explanations.get(verdict, "Fact-check completed.")
        
        # Assess potential harm
        harm_levels = ['low', 'medium', 'high']
        harm_weights = [0.6, 0.3, 0.1]
        potential_harm = weighted_random_choice(harm_levels, harm_weights)
        
        # Assess urgency
        urgency_levels = ['low', 'normal', 'high', 'critical']
        urgency_weights = [0.2, 0.5, 0.2, 0.1]
        urgency = weighted_random_choice(urgency_levels, urgency_weights)
        
        result = FactCheckResult(
            content_id=content_id,
            fact_checker_id=fact_checker.get('id', 'unknown'),
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            potential_harm=potential_harm,
            urgency=urgency
        )
        
        return result
    
    def get_fact_check_statistics(self) -> Dict[str, Any]:
        """Get fact-checking statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_checks = len(self.completed_checks)
        pending_checks = len(self.pending_checks)
        
        if total_checks == 0:
            return {
                'total_checks': 0,
                'pending_checks': pending_checks,
                'active_fact_checkers': len(self.fact_checkers)
            }
        
        # Verdict distribution
        verdicts = [result.verdict for result in self.completed_checks.values()]
        verdict_counts = {verdict: verdicts.count(verdict) for verdict in set(verdicts)}
        
        # Average confidence
        avg_confidence = np.mean([result.confidence for result in self.completed_checks.values()])
        
        # Fact-checker performance
        fact_checker_stats = {}
        for fc_id, fc_data in self.fact_checkers.items():
            fact_checker_stats[fc_id] = {
                'checks_completed': fc_data['checks_completed'],
                'accuracy_rate': fc_data['accuracy_rate'],
                'expertise': fc_data['expertise']
            }
        
        stats = {
            'total_checks': total_checks,
            'pending_checks': pending_checks,
            'verdict_distribution': verdict_counts,
            'avg_confidence': avg_confidence,
            'active_fact_checkers': len(self.fact_checkers),
            'fact_checker_performance': fact_checker_stats
        }
        
        return stats


class ContentAnalyzer:
    """Advanced content analysis for the simulation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize content analyzer.
        
        Args:
            config: Analyzer configuration
        """
        self.config = config or {}
        
        # Analysis models (would be actual ML models in production)
        self.sentiment_model = None
        self.credibility_model = None
        self.topic_model = None
        
        logger.debug("Content analyzer initialized")
    
    def analyze_content_similarity(self, 
                                 content1: Content,
                                 content2: Content) -> float:
        """Analyze similarity between two content items.
        
        Args:
            content1: First content item
            content2: Second content item
            
        Returns:
            Similarity score (0-1)
        """
        # Simple similarity based on keywords and topics
        keywords1 = set(content1.metadata.keywords)
        keywords2 = set(content2.metadata.keywords)
        
        topics1 = set(content1.metadata.topics)
        topics2 = set(content2.metadata.topics)
        
        # Jaccard similarity
        keyword_similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2) if keywords1 | keywords2 else 0
        topic_similarity = len(topics1 & topics2) / len(topics1 | topics2) if topics1 | topics2 else 0
        
        # Weighted average
        similarity = 0.6 * keyword_similarity + 0.4 * topic_similarity
        
        return similarity
    
    def detect_content_clusters(self, 
                              content_list: List[Content],
                              similarity_threshold: float = 0.7) -> List[List[str]]:
        """Detect clusters of similar content.
        
        Args:
            content_list: List of content to cluster
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            List of content ID clusters
        """
        clusters = []
        processed = set()
        
        for i, content1 in enumerate(content_list):
            if content1.metadata.content_id in processed:
                continue
            
            cluster = [content1.metadata.content_id]
            processed.add(content1.metadata.content_id)
            
            for j, content2 in enumerate(content_list[i+1:], i+1):
                if content2.metadata.content_id in processed:
                    continue
                
                similarity = self.analyze_content_similarity(content1, content2)
                if similarity >= similarity_threshold:
                    cluster.append(content2.metadata.content_id)
                    processed.add(content2.metadata.content_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def analyze_content_evolution(self, 
                                content_timeline: List[Content]) -> Dict[str, Any]:
        """Analyze how content evolves over time.
        
        Args:
            content_timeline: Chronologically ordered content
            
        Returns:
            Evolution analysis results
        """
        if len(content_timeline) < 2:
            return {}
        
        # Track changes in key metrics
        credibility_trend = [c.credibility_score for c in content_timeline]
        sentiment_trend = [c.sentiment_score for c in content_timeline]
        engagement_trend = [c.engagement.engagement_score for c in content_timeline]
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = list(range(len(values)))
            slope, _, r_value, _, _ = stats.linregress(x, values)
            return slope
        
        analysis = {
            'credibility_trend': calculate_trend(credibility_trend),
            'sentiment_trend': calculate_trend(sentiment_trend),
            'engagement_trend': calculate_trend(engagement_trend),
            'total_content': len(content_timeline),
            'time_span_hours': (content_timeline[-1].metadata.created_at - 
                              content_timeline[0].metadata.created_at).total_seconds() / 3600
        }
        
        return analysis


# Export main classes
__all__ = [
    'ContentModel', 'Content', 'ContentMetadata', 'ContentEngagement',
    'ContentType', 'ContentStatus', 'MisinformationType',
    'MisinformationModel', 'FactCheckModel', 'FactCheckResult',
    'ContentAnalyzer'
]