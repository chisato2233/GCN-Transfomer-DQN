"""
Logging utilities for training.
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name: str, log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """
    Helper class for logging training progress.
    """
    
    def __init__(self, log_dir: str, name: str = 'training'):
        self.logger = setup_logger(name, log_dir)
        self.log_dir = log_dir
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_episode(self, episode: int, metrics: dict, epsilon: float = None):
        """Log episode metrics."""
        msg = f"Episode {episode:5d}"
        
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:8.2f}"
            else:
                msg += f" | {key}: {value}"
        
        if epsilon is not None:
            msg += f" | epsilon: {epsilon:.4f}"
        
        self.logger.info(msg)
    
    def log_eval(self, episode: int, reward: float, delivery_rate: float):
        """Log evaluation results."""
        self.logger.info(
            f"Eval @ Episode {episode}: "
            f"Reward = {reward:.2f}, "
            f"Delivery Rate = {delivery_rate*100:.1f}%"
        )
