#!/usr/bin/env python3
"""
Natural Language Processing utilities for Bangla text processing and content analysis.
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
import unicodedata

# Optional imports for advanced NLP features
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from bnlp import BasicTokenizer, SentencePieceTokenizer
    BNLP_AVAILABLE = True
except ImportError:
    BNLP_AVAILABLE = False

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextAnalysisResult:
    """Results from text analysis."""
    text: str
    language: str
    sentiment: Optional[float] = None
    toxicity: Optional[float] = None
    readability: Optional[float] = None
    keywords: List[str] = None
    entities: List[Dict[str, Any]] = None
    topics: List[str] = None
    word_count: int = 0
    sentence_count: int = 0
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = []
        if self.topics is None:
            self.topics = []


class BanglaTextProcessor:
    """Bangla text processing utilities."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # Common Bangla stop words
        self.stop_words = {
            'আর', 'আমি', 'আমার', 'আমাদের', 'আপনি', 'আপনার', 'আছে', 'আছি', 'আছেন',
            'এ', 'এই', 'এক', 'একটি', 'একটা', 'এর', 'এরা', 'এদের', 'এবং', 'এবার',
            'ও', 'ওই', 'ওর', 'ওদের', 'ওরা', 'ওকে', 'ওদেরকে',
            'কি', 'কী', 'কে', 'কেন', 'কোন', 'কোনো', 'কিছু', 'কিন্তু', 'করে', 'করা',
            'করেছে', 'করেছেন', 'করেছি', 'করেছিল', 'করেছিলেন', 'করব', 'করবে',
            'করবেন', 'করি', 'করেন', 'করো', 'করুন', 'কর', 'করল', 'করলে',
            'করলেন', 'করতে', 'করার', 'করেই', 'করেও',
            'গিয়ে', 'গেছে', 'গেল', 'গেলে', 'গেলেন', 'গিয়েছে', 'গিয়েছেন',
            'ছিল', 'ছিলেন', 'ছিলাম', 'ছিলো', 'ছে', 'ছেন', 'ছি',
            'জন', 'জনা', 'জনের', 'জানি', 'জানেন', 'জানো', 'জানুন',
            'তা', 'তার', 'তাদের', 'তাকে', 'তাদেরকে', 'তারা', 'তিনি', 'তুমি',
            'তোমার', 'তোমাদের', 'তোমাকে', 'তোমাদেরকে', 'তোমরা', 'তবে', 'তাই',
            'তাহলে', 'তখন', 'তো', 'তুই', 'তোর', 'তোকে',
            'দিয়ে', 'দেয়', 'দেওয়া', 'দিয়েছে', 'দিয়েছেন', 'দিয়েছি', 'দিয়েছিল',
            'দিয়েছিলেন', 'দেব', 'দেবে', 'দেবেন', 'দি', 'দেন', 'দাও', 'দিন',
            'দিল', 'দিলে', 'দিলেন', 'দিতে', 'দেওয়ার', 'দিয়েই', 'দিয়েও',
            'না', 'নি', 'নেই', 'নয়', 'নন', 'নো', 'নাই', 'নাকি', 'নিয়ে',
            'নিয়েছে', 'নিয়েছেন', 'নিয়েছি', 'নিয়েছিল', 'নিয়েছিলেন', 'নেব',
            'নেবে', 'নেবেন', 'নিই', 'নেন', 'নাও', 'নিন', 'নিল', 'নিলে',
            'নিলেন', 'নিতে', 'নেওয়ার', 'নিয়েই', 'নিয়েও',
            'পর', 'পরে', 'পারি', 'পারেন', 'পারো', 'পারুন', 'পার', 'পারল',
            'পারলে', 'পারলেন', 'পারতে', 'পারার', 'পেয়ে', 'পেয়েছে', 'পেয়েছেন',
            'পেয়েছি', 'পেয়েছিল', 'পেয়েছিলেন', 'পাব', 'পাবে', 'পাবেন',
            'পাই', 'পান', 'পাও', 'পান', 'পেল', 'পেলে', 'পেলেন', 'পেতে',
            'পাওয়ার', 'পেয়েই', 'পেয়েও',
            'বলে', 'বলেছে', 'বলেছেন', 'বলেছি', 'বলেছিল', 'বলেছিলেন', 'বলব',
            'বলবে', 'বলবেন', 'বলি', 'বলেন', 'বলো', 'বলুন', 'বল', 'বলল',
            'বললে', 'বললেন', 'বলতে', 'বলার', 'বলেই', 'বলেও',
            'মত', 'মতো', 'মানে', 'মাধ্যমে', 'মধ্যে', 'মধ্যেও', 'মধ্যেই',
            'যে', 'যা', 'যার', 'যাদের', 'যাকে', 'যাদেরকে', 'যারা', 'যিনি',
            'যখন', 'যদি', 'যাতে', 'যেতে', 'যাওয়া', 'যাওয়ার', 'যেয়ে',
            'যেয়েছে', 'যেয়েছেন', 'যেয়েছি', 'যেয়েছিল', 'যেয়েছিলেন',
            'যাব', 'যাবে', 'যাবেন', 'যাই', 'যান', 'যাও', 'যান', 'গেল',
            'গেলে', 'গেলেন', 'যেতে', 'যাওয়ার', 'যেয়েই', 'যেয়েও',
            'রয়েছে', 'রয়েছেন', 'রয়েছি', 'রয়েছিল', 'রয়েছিলেন', 'রাখি',
            'রাখেন', 'রাখো', 'রাখুন', 'রাখ', 'রাখল', 'রাখলে', 'রাখলেন',
            'রাখতে', 'রাখার', 'রেখে', 'রেখেছে', 'রেখেছেন', 'রেখেছি',
            'রেখেছিল', 'রেখেছিলেন', 'রাখব', 'রাখবে', 'রাখবেন',
            'সে', 'সেই', 'সেটি', 'সেটা', 'সের', 'সেদের', 'সেকে', 'সেদেরকে',
            'সেরা', 'সব', 'সবাই', 'সবার', 'সবাকে', 'সবাইকে', 'সকল',
            'সকলে', 'সকলের', 'সকলকে', 'সকলেই', 'সকলেও',
            'হয়', 'হয়ে', 'হয়েছে', 'হয়েছেন', 'হয়েছি', 'হয়েছিল', 'হয়েছিলেন',
            'হব', 'হবে', 'হবেন', 'হই', 'হন', 'হও', 'হোন', 'হল', 'হলে',
            'হলেন', 'হতে', 'হওয়া', 'হওয়ার', 'হয়েই', 'হয়েও',
            'হচ্ছে', 'হতে', 'হবে', 'হয়', 'হয়ে', 'হয়েছে', 'হল', 'হলে', 'হলো'
        }
        
        # Basic Bangla punctuations
        self.punctuations = {
            '।', '॥', '?', '!', ',', ';', ':', '-', '–', '—', '(', ')', '[', ']',
            '{', '}', '"', "'", '`', ''', ''', '"', '…', '/', '\\', '|'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Bangla text."""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted characters but keep Bangla characters
        text = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\s]', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Bangla text into words."""
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Simple word tokenization
        words = re.findall(r'[\u0980-\u09FF]+', text)
        
        return words
    
    def remove_stop_words(self, words: List[str]) -> List[str]:
        """Remove Bangla stop words."""
        return [word for word in words if word not in self.stop_words]
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from Bangla text."""
        words = self.tokenize(text)
        words = self.remove_stop_words(words)
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Return most common words as keywords
        return [word for word, _ in word_freq.most_common(max_keywords)]


class ContentAnalyzer:
    """Advanced content analysis for news articles."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.bangla_processor = BanglaTextProcessor()
        
        # Initialize NLP models if available
        self.sentiment_analyzer = None
        self.toxicity_analyzer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            except Exception as e:
                self.logger.warning(f"Could not load sentiment analyzer: {e}")
        
    def analyze_text(self, text: str, language: str = "bn") -> TextAnalysisResult:
        """Comprehensive text analysis."""
        result = TextAnalysisResult(text=text, language=language)
        
        try:
            # Basic metrics
            result.word_count = len(self.bangla_processor.tokenize(text))
            result.sentence_count = len(re.split(r'[।!?]', text))
            
            # Extract keywords
            result.keywords = self.bangla_processor.extract_keywords(text)
            
            # Sentiment analysis (if available)
            if self.sentiment_analyzer and language == "en":
                try:
                    sentiment_result = self.sentiment_analyzer(text[:512])  # Limit length
                    result.sentiment = sentiment_result[0]['score']
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis failed: {e}")
            
            # Calculate readability (simple metric)
            if result.word_count > 0 and result.sentence_count > 0:
                avg_words_per_sentence = result.word_count / result.sentence_count
                result.readability = min(100, max(0, 100 - (avg_words_per_sentence - 15) * 2))
            
        except Exception as e:
            self.logger.error(f"Text analysis failed: {e}")
        
        return result
    
    def detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            from langdetect import detect
            return detect(text)
        except:
            # Simple heuristic for Bangla
            bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
            total_chars = len(re.findall(r'[\w]', text))
            
            if total_chars > 0 and bangla_chars / total_chars > 0.5:
                return "bn"
            else:
                return "en"
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap."""
        words1 = set(self.bangla_processor.tokenize(text1.lower()))
        words2 = set(self.bangla_processor.tokenize(text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        # Simple pattern-based entity extraction for Bangla
        # This is a basic implementation - in practice, you'd use more sophisticated NER
        
        # Extract potential person names (capitalized Bangla words)
        person_pattern = r'[\u0980-\u09FF][\u0980-\u09FF\s]*[\u0980-\u09FF]'
        potential_names = re.findall(person_pattern, text)
        
        for name in potential_names:
            if len(name.split()) >= 2:  # Likely a full name
                entities.append({
                    'text': name.strip(),
                    'label': 'PERSON',
                    'confidence': 0.7
                })
        
        return entities