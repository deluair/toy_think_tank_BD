"""Logging utilities for NewsGuard Bangladesh simulation."""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger as loguru_logger
from loguru._defaults import LOGURU_FORMAT


class NewsGuardLogger:
    """Custom logger for NewsGuard simulation with structured logging."""
    
    def __init__(self, name: str, level: str = "INFO"):
        """Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.name = name
        self.level = level
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger configuration."""
        # Remove default loguru handler
        loguru_logger.remove()
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Console handler with colors
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level=self.level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File handler for all logs
        loguru_logger.add(
            log_dir / "newsguard.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Error file handler
        loguru_logger.add(
            log_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Simulation-specific log file
        loguru_logger.add(
            log_dir / "simulation.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]} | {message}",
            level="INFO",
            filter=lambda record: "simulation" in record["extra"].get("component", ""),
            rotation="50 MB",
            retention="90 days"
        )
        
        # Agent behavior log file
        loguru_logger.add(
            log_dir / "agents.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[agent_type]}:{extra[agent_id]} | {message}",
            level="DEBUG",
            filter=lambda record: "agent" in record["extra"].get("component", ""),
            rotation="20 MB",
            retention="30 days"
        )
        
        # Misinformation tracking log
        loguru_logger.add(
            log_dir / "misinformation.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[content_id]} | {extra[action]} | {message}",
            level="INFO",
            filter=lambda record: "misinformation" in record["extra"].get("component", ""),
            rotation="30 MB",
            retention="180 days"  # Keep misinformation logs longer for analysis
        )
        
        # Performance metrics log
        loguru_logger.add(
            log_dir / "performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[metric]} | {extra[value]} | {message}",
            level="INFO",
            filter=lambda record: "performance" in record["extra"].get("component", ""),
            rotation="10 MB",
            retention="30 days"
        )
    
    def bind(self, **kwargs) -> 'NewsGuardLogger':
        """Bind additional context to logger.
        
        Args:
            **kwargs: Context variables to bind
            
        Returns:
            Logger with bound context
        """
        bound_logger = NewsGuardLogger(self.name, self.level)
        bound_logger._logger = loguru_logger.bind(**kwargs)
        return bound_logger
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        loguru_logger.bind(**kwargs).debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        loguru_logger.bind(**kwargs).info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        loguru_logger.bind(**kwargs).warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        loguru_logger.bind(**kwargs).error(message)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        loguru_logger.bind(**kwargs).critical(message)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        loguru_logger.bind(**kwargs).exception(message)


class SimulationLogger(NewsGuardLogger):
    """Specialized logger for simulation events."""
    
    def __init__(self, name: str = "simulation", level: str = "INFO"):
        super().__init__(name, level)
        self.component = "simulation"
    
    def log_simulation_start(self, simulation_id: str, config) -> None:
        """Log simulation start."""
        self.info(
            f"Simulation {simulation_id} started",
            component=self.component,
            simulation_id=simulation_id,
            config=str(config)
        )
    
    def step_start(self, step: int, timestamp: datetime) -> None:
        """Log simulation step start."""
        self.info(
            f"Step {step} started",
            component=self.component,
            step=step,
            timestamp=timestamp.isoformat()
        )
    
    def step_end(self, step: int, duration_ms: float, agents_processed: int) -> None:
        """Log simulation step completion."""
        self.info(
            f"Step {step} completed in {duration_ms:.2f}ms, processed {agents_processed} agents",
            component=self.component,
            step=step,
            duration_ms=duration_ms,
            agents_processed=agents_processed
        )
    
    def scenario_start(self, scenario_name: str, parameters: Dict[str, Any]) -> None:
        """Log scenario start."""
        self.info(
            f"Scenario '{scenario_name}' started",
            component=self.component,
            scenario=scenario_name,
            parameters=parameters
        )
    
    def scenario_end(self, scenario_name: str, duration_seconds: float, results: Dict[str, Any]) -> None:
        """Log scenario completion."""
        self.info(
            f"Scenario '{scenario_name}' completed in {duration_seconds:.2f}s",
            component=self.component,
            scenario=scenario_name,
            duration_seconds=duration_seconds,
            results=results
        )
    
    def intervention_applied(self, intervention_name: str, parameters: Dict[str, Any]) -> None:
        """Log intervention application."""
        self.info(
            f"Intervention '{intervention_name}' applied",
            component=self.component,
            intervention=intervention_name,
            parameters=parameters
        )


class AgentLogger(NewsGuardLogger):
    """Specialized logger for agent behaviors."""
    
    def __init__(self, agent_type: str, agent_id: str, level: str = "DEBUG"):
        super().__init__(f"agent.{agent_type}", level)
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.component = "agent"
    
    def action(self, action_name: str, details: Dict[str, Any]) -> None:
        """Log agent action."""
        self.debug(
            f"Action: {action_name}",
            component=self.component,
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            action=action_name,
            details=details
        )
    
    def state_change(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> None:
        """Log agent state change."""
        self.debug(
            "State changed",
            component=self.component,
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            old_state=old_state,
            new_state=new_state
        )
    
    def interaction(self, target_agent_id: str, interaction_type: str, details: Dict[str, Any]) -> None:
        """Log agent interaction."""
        self.debug(
            f"Interaction with {target_agent_id}: {interaction_type}",
            component=self.component,
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            target_agent_id=target_agent_id,
            interaction_type=interaction_type,
            details=details
        )


class MisinformationLogger(NewsGuardLogger):
    """Specialized logger for misinformation tracking."""
    
    def __init__(self, level: str = "INFO"):
        super().__init__("misinformation", level)
        self.component = "misinformation"
    
    def content_created(self, content_id: str, content_type: str, source: str, metadata: Dict[str, Any]) -> None:
        """Log misinformation content creation."""
        self.info(
            f"Misinformation content created: {content_type}",
            component=self.component,
            content_id=content_id,
            action="created",
            content_type=content_type,
            source=source,
            metadata=metadata
        )
    
    def content_shared(self, content_id: str, sharer_id: str, platform: str, reach: int) -> None:
        """Log misinformation sharing."""
        self.info(
            f"Misinformation shared on {platform}, reach: {reach}",
            component=self.component,
            content_id=content_id,
            action="shared",
            sharer_id=sharer_id,
            platform=platform,
            reach=reach
        )
    
    def content_flagged(self, content_id: str, flagger_id: str, reason: str, confidence: float) -> None:
        """Log misinformation flagging."""
        self.info(
            f"Misinformation flagged: {reason} (confidence: {confidence:.2f})",
            component=self.component,
            content_id=content_id,
            action="flagged",
            flagger_id=flagger_id,
            reason=reason,
            confidence=confidence
        )
    
    def content_debunked(self, content_id: str, fact_checker_id: str, evidence: str, time_to_debunk_hours: float) -> None:
        """Log misinformation debunking."""
        self.info(
            f"Misinformation debunked in {time_to_debunk_hours:.1f} hours",
            component=self.component,
            content_id=content_id,
            action="debunked",
            fact_checker_id=fact_checker_id,
            evidence=evidence,
            time_to_debunk_hours=time_to_debunk_hours
        )


class PerformanceLogger(NewsGuardLogger):
    """Specialized logger for performance metrics."""
    
    def __init__(self, level: str = "INFO"):
        super().__init__("performance", level)
        self.component = "performance"
    
    def metric(self, metric_name: str, value: float, unit: str = "", context: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metric."""
        message = f"{metric_name}: {value}"
        if unit:
            message += f" {unit}"
        
        self.info(
            message,
            component=self.component,
            metric=metric_name,
            value=value,
            unit=unit,
            context=context or {}
        )
    
    def memory_usage(self, usage_mb: float, peak_mb: float) -> None:
        """Log memory usage."""
        self.metric("memory_usage", usage_mb, "MB", {"peak_mb": peak_mb})
    
    def cpu_usage(self, usage_percent: float) -> None:
        """Log CPU usage."""
        self.metric("cpu_usage", usage_percent, "%")
    
    def processing_time(self, operation: str, duration_ms: float, items_processed: int = 0) -> None:
        """Log processing time."""
        context = {"operation": operation}
        if items_processed > 0:
            context["items_processed"] = items_processed
            context["items_per_second"] = items_processed / (duration_ms / 1000)
        
        self.metric("processing_time", duration_ms, "ms", context)


# Global logger instances
_loggers: Dict[str, NewsGuardLogger] = {}


def get_logger(name: str, level: str = "INFO") -> NewsGuardLogger:
    """Get or create a logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = NewsGuardLogger(name, level)
    return _loggers[name]


def get_simulation_logger() -> SimulationLogger:
    """Get simulation logger instance.
    
    Returns:
        Simulation logger
    """
    if "simulation" not in _loggers:
        _loggers["simulation"] = SimulationLogger()
    return _loggers["simulation"]


def get_agent_logger(agent_type: str, agent_id: str) -> AgentLogger:
    """Get agent logger instance.
    
    Args:
        agent_type: Type of agent
        agent_id: Unique agent identifier
        
    Returns:
        Agent logger
    """
    logger_name = f"agent.{agent_type}.{agent_id}"
    if logger_name not in _loggers:
        _loggers[logger_name] = AgentLogger(agent_type, agent_id)
    return _loggers[logger_name]


def get_misinformation_logger() -> MisinformationLogger:
    """Get misinformation logger instance.
    
    Returns:
        Misinformation logger
    """
    if "misinformation" not in _loggers:
        _loggers["misinformation"] = MisinformationLogger()
    return _loggers["misinformation"]


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance.
    
    Returns:
        Performance logger
    """
    if "performance" not in _loggers:
        _loggers["performance"] = PerformanceLogger()
    return _loggers["performance"]


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup global logging configuration.
    
    Args:
        level: Global logging level
        log_dir: Directory for log files
    """
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Set environment variable for log directory
    os.environ["NEWSGUARD_LOG_DIR"] = log_dir
    
    # Initialize main logger
    main_logger = get_logger("newsguard", level)
    main_logger.info(f"Logging initialized with level {level}")


def cleanup_logs(days_to_keep: int = 30) -> None:
    """Clean up old log files.
    
    Args:
        days_to_keep: Number of days of logs to keep
    """
    log_dir = Path(os.environ.get("NEWSGUARD_LOG_DIR", "logs"))
    if not log_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    
    for log_file in log_dir.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            get_logger("newsguard").info(f"Cleaned up old log file: {log_file}")