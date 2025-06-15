"""Helper utilities for NewsGuard Bangladesh simulation."""

import os
import json
import pickle
import hashlib
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import uuid
import time
from functools import wraps
from collections import defaultdict
import math

import numpy as np
import pandas as pd
from scipy import stats

from .logger import get_logger

logger = get_logger(__name__)


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of random part
        
    Returns:
        Unique ID string
    """
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def generate_uuid() -> str:
    """Generate a UUID4 string.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def timestamp_now() -> float:
    """Get current timestamp.
    
    Returns:
        Current timestamp as float
    """
    return time.time()


def hash_string(text: str, algorithm: str = 'md5') -> str:
    """Generate hash of a string.
    
    Args:
        text: Input text
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hash string
    """
    if algorithm == 'md5':
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(text.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not (alias for ensure_dir).
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    return ensure_dir(path)


def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize object to JSON-compatible format.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return safe_json_serialize(obj.__dict__)
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return str(obj)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def retry_decorator(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


def normalize_text(text: str) -> str:
    """Normalize text for processing.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix for truncated text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """Format number with appropriate units.
    
    Args:
        number: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if number >= 1_000_000:
        return f"{number / 1_000_000:.{precision}f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    """Parse date string with multiple format attempts.
    
    Args:
        date_str: Date string
        formats: List of date formats to try
        
    Returns:
        Parsed datetime or None
    """
    if formats is None:
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
            '%d-%m-%Y %H:%M:%S',
            '%d-%m-%Y',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None


def generate_date_range(start_date: datetime, end_date: datetime, 
                       frequency: str = 'D') -> List[datetime]:
    """Generate date range.
    
    Args:
        start_date: Start date
        end_date: End date
        frequency: Frequency ('D' for daily, 'H' for hourly, etc.)
        
    Returns:
        List of datetime objects
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    return dates.to_pydatetime().tolist()


def weighted_random_choice(choices: List[Any], weights: List[float]) -> Any:
    """Choose random element based on weights.
    
    Args:
        choices: List of choices
        weights: Corresponding weights
        
    Returns:
        Randomly selected choice
    """
    if not choices or not weights:
        raise ValueError("Choices and weights cannot be empty")
    
    if len(choices) != len(weights):
        raise ValueError("Choices and weights must have same length")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Use numpy for random choice
    return np.random.choice(choices, p=normalized_weights)


def sigmoid_function(x: float, steepness: float = 1.0, midpoint: float = 0.0) -> float:
    """Calculate sigmoid function value.
    
    Args:
        x: Input value
        steepness: Steepness of the curve
        midpoint: Midpoint of the curve
        
    Returns:
        Sigmoid function value between 0 and 1
    """
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile of data.
    
    Args:
        data: List of values
        percentile: Percentile (0-100)
        
    Returns:
        Percentile value
    """
    return np.percentile(data, percentile)


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for data.
    
    Args:
        data: List of values
        
    Returns:
        Dictionary of statistics
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        'count': len(data),
        'mean': float(np.mean(data_array)),
        'median': float(np.median(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75))
    }


def normalize_scores(scores: List[float], method: str = 'minmax') -> List[float]:
    """Normalize scores using different methods.
    
    Args:
        scores: List of scores
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    if method == 'minmax':
        min_val = np.min(scores_array)
        max_val = np.max(scores_array)
        if max_val == min_val:
            return [0.5] * len(scores)
        return ((scores_array - min_val) / (max_val - min_val)).tolist()
    
    elif method == 'zscore':
        mean_val = np.mean(scores_array)
        std_val = np.std(scores_array)
        if std_val == 0:
            return [0.0] * len(scores)
        return ((scores_array - mean_val) / std_val).tolist()
    
    elif method == 'robust':
        median_val = np.median(scores_array)
        mad = np.median(np.abs(scores_array - median_val))
        if mad == 0:
            return [0.0] * len(scores)
        return ((scores_array - median_val) / mad).tolist()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Calculate Pearson correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Tuple of (correlation, p-value)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0, 1.0
    
    correlation, p_value = stats.pearsonr(x, y)
    return float(correlation), float(p_value)


def smooth_data(data: List[float], window_size: int = 5) -> List[float]:
    """Smooth data using moving average.
    
    Args:
        data: Input data
        window_size: Size of moving window
        
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        window_data = data[start_idx:end_idx]
        smoothed.append(sum(window_data) / len(window_data))
    
    return smoothed


def detect_outliers(data: List[float], method: str = 'iqr', 
                   threshold: float = 1.5) -> List[bool]:
    """Detect outliers in data.
    
    Args:
        data: Input data
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        List of boolean values indicating outliers
    """
    if not data:
        return []
    
    data_array = np.array(data)
    
    if method == 'iqr':
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return [(x < lower_bound or x > upper_bound) for x in data]
    
    elif method == 'zscore':
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        if std_val == 0:
            return [False] * len(data)
        z_scores = np.abs((data_array - mean_val) / std_val)
        return (z_scores > threshold).tolist()
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def create_bins(data: List[float], num_bins: int = 10) -> Tuple[List[float], List[int]]:
    """Create bins for data distribution.
    
    Args:
        data: Input data
        num_bins: Number of bins
        
    Returns:
        Tuple of (bin_edges, bin_counts)
    """
    if not data:
        return [], []
    
    counts, bin_edges = np.histogram(data, bins=num_bins)
    return bin_edges.tolist(), counts.tolist()


def interpolate_missing_values(data: List[Optional[float]], 
                             method: str = 'linear') -> List[float]:
    """Interpolate missing values in data.
    
    Args:
        data: Data with possible None values
        method: Interpolation method ('linear', 'forward', 'backward')
        
    Returns:
        Data with interpolated values
    """
    if not data:
        return []
    
    # Convert to pandas Series for easy interpolation
    series = pd.Series(data)
    
    if method == 'linear':
        interpolated = series.interpolate(method='linear')
    elif method == 'forward':
        interpolated = series.fillna(method='ffill')
    elif method == 'backward':
        interpolated = series.fillna(method='bfill')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Fill any remaining NaN values with 0
    interpolated = interpolated.fillna(0)
    
    return interpolated.tolist()


def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def create_progress_tracker(total: int, description: str = "Progress") -> Callable:
    """Create a simple progress tracker.
    
    Args:
        total: Total number of items
        description: Progress description
        
    Returns:
        Update function
    """
    start_time = time.time()
    
    def update(current: int) -> None:
        if total == 0:
            return
        
        percentage = (current / total) * 100
        elapsed_time = time.time() - start_time
        
        if current > 0:
            eta = (elapsed_time / current) * (total - current)
            eta_str = format_duration(eta)
        else:
            eta_str = "--"
        
        logger.info(f"{description}: {current}/{total} ({percentage:.1f}%) - ETA: {eta_str}")
    
    return update


def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """Split items into batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def merge_dictionaries(*dicts: Dict[str, Any], 
                      strategy: str = 'update') -> Dict[str, Any]:
    """Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        strategy: Merge strategy ('update', 'sum', 'max', 'min')
        
    Returns:
        Merged dictionary
    """
    if not dicts:
        return {}
    
    if strategy == 'update':
        result = {}
        for d in dicts:
            result.update(d)
        return result
    
    elif strategy in ['sum', 'max', 'min']:
        result = defaultdict(list)
        for d in dicts:
            for key, value in d.items():
                if isinstance(value, (int, float)):
                    result[key].append(value)
        
        final_result = {}
        for key, values in result.items():
            if strategy == 'sum':
                final_result[key] = sum(values)
            elif strategy == 'max':
                final_result[key] = max(values)
            elif strategy == 'min':
                final_result[key] = min(values)
        
        return final_result
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def validate_config(config: Dict[str, Any], 
                   required_keys: List[str]) -> Tuple[bool, List[str]]:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    return len(missing_keys) == 0, missing_keys


def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max.
    
    Args:
        value: Input value
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def exponential_decay(initial_value: float, decay_rate: float, 
                     time_step: int) -> float:
    """Calculate exponential decay.
    
    Args:
        initial_value: Initial value
        decay_rate: Decay rate (0-1)
        time_step: Time step
        
    Returns:
        Decayed value
    """
    return initial_value * (1 - decay_rate) ** time_step


def sigmoid(x: float, steepness: float = 1.0, midpoint: float = 0.0) -> float:
    """Calculate sigmoid function.
    
    Args:
        x: Input value
        steepness: Steepness parameter
        midpoint: Midpoint parameter
        
    Returns:
        Sigmoid value (0-1)
    """
    return 1 / (1 + math.exp(-steepness * (x - midpoint)))


def linear_interpolation(x: float, x1: float, y1: float, 
                        x2: float, y2: float) -> float:
    """Linear interpolation between two points.
    
    Args:
        x: Input value
        x1, y1: First point
        x2, y2: Second point
        
    Returns:
        Interpolated value
    """
    if x2 == x1:
        return y1
    
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)