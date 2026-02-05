import logging
import sys
from typing import Optional

def setup_logging(name: str = "receipt_system", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a standard industrial-grade logger.
    
    Args:
        name: Name of the logger.
        level: Logging level (default: INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Default logger for the project
logger = setup_logging()
