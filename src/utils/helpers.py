"""Helper functions and utilities for the application."""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Validate required Azure OpenAI environment variables
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_DEPLOYMENT_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.warning(
            f"Missing environment variables: {', '.join(missing_vars)}. "
            "AI analysis features may not work properly."
        )


def setup_logging(level: str = 'INFO', verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        verbose: Enable verbose logging
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG if verbose else log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    logging.info(f"Logging configured. Level: {level}, Verbose: {verbose}")


def create_logger(name: str) -> logging.Logger:
    """Create a named logger.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format a value as a percentage.
    
    Args:
        value: Value to format (0-100)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimal_places}f}%"


def format_duration(hours: float) -> str:
    """Format duration in hours to human-readable string.
    
    Args:
        hours: Duration in hours
        
    Returns:
        Formatted duration string
    """
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = int(hours // 24)
        remaining_hours = hours % 24
        if remaining_hours > 0:
            return f"{days} days, {remaining_hours:.1f} hours"
        else:
            return f"{days} days"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clean_string(text: str) -> str:
    """Clean and normalize a string value.
    
    Args:
        text: Input string
        
    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        text = str(text)
    
    return text.strip().replace('\n', ' ').replace('\r', ' ')


def validate_file_path(file_path: str) -> bool:
    """Validate if a file path exists and is readable.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if file exists and is readable, False otherwise
    """
    try:
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    except (OSError, TypeError):
        return False


def create_output_directory(path: str) -> bool:
    """Create output directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        True if directory was created or already exists, False on error
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError:
        return False


def get_file_size_mb(file_path: str) -> Optional[float]:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB or None if file doesn't exist
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return None


def format_file_size(size_mb: float) -> str:
    """Format file size for display.
    
    Args:
        size_mb: Size in megabytes
        
    Returns:
        Formatted size string
    """
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"


def get_timestamp() -> str:
    """Get current timestamp as formatted string.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_duration_string(duration_str: str) -> Optional[float]:
    """Parse duration string to hours.
    
    Args:
        duration_str: Duration string like "2h 30m", "1.5 hours", "90 minutes"
        
    Returns:
        Duration in hours or None if parsing fails
    """
    if not isinstance(duration_str, str):
        return None
    
    duration_str = duration_str.lower().strip()
    
    try:
        # Simple patterns
        if 'hour' in duration_str:
            # Extract number before 'hour'
            import re
            match = re.search(r'(\d+\.?\d*)', duration_str)
            if match:
                return float(match.group(1))
        
        elif 'minute' in duration_str:
            # Extract number before 'minute'
            import re
            match = re.search(r'(\d+\.?\d*)', duration_str)
            if match:
                return float(match.group(1)) / 60
        
        elif 'day' in duration_str:
            # Extract number before 'day'
            import re
            match = re.search(r'(\d+\.?\d*)', duration_str)
            if match:
                return float(match.group(1)) * 24
        
        else:
            # Try to parse as plain number (assume hours)
            return float(duration_str)
    
    except (ValueError, AttributeError):
        pass
    
    return None
