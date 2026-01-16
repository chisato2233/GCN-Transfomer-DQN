"""
Logging and TensorBoard utilities for SAGIN routing training.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger with file, console, and TensorBoard support."""

    def __init__(self,
                 log_dir: str = "logs",
                 experiment_name: str = "sagin_routing",
                 level: str = "INFO",
                 use_tensorboard: bool = True):
        """
        Initialize the logger.

        Args:
            log_dir: Root directory for logs
            experiment_name: Name of the experiment
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_tensorboard: Whether to enable TensorBoard logging
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(log_dir) / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.metrics_dir = self.exp_dir / "metrics"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        # Configure Python logger
        self.logger = self._setup_logger(level)

        # TensorBoard writer
        self.writer = SummaryWriter(str(self.exp_dir)) if use_tensorboard else None

        self.info(f"Experiment directory: {self.exp_dir}")

    def _setup_logger(self, level: str) -> logging.Logger:
        """Configure the Python logger."""
        logger = logging.getLogger(f"SAGIN_{self.timestamp}")
        logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if logger.handlers:
            logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(
            self.exp_dir / f"training_{self.timestamp}.log",
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars to TensorBoard."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram to TensorBoard."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """Log the model computation graph."""
        if self.writer:
            try:
                self.writer.add_graph(model, input_to_model)
            except Exception as e:
                self.warning(f"Failed to log model graph: {e}")

    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        """Log an image to TensorBoard."""
        if self.writer:
            self.writer.add_image(tag, img_tensor, step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text to TensorBoard."""
        if self.writer:
            self.writer.add_text(tag, text, step)

    def log_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """Log hyperparameters and their metrics."""
        if self.writer:
            # Convert non-serializable types
            clean_hparams = {}
            for k, v in hparam_dict.items():
                if isinstance(v, (int, float, str, bool)):
                    clean_hparams[k] = v
                elif isinstance(v, (list, tuple)):
                    clean_hparams[k] = str(v)
                else:
                    clean_hparams[k] = str(v)

            self.writer.add_hparams(clean_hparams, metric_dict)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str):
        """Log critical message."""
        self.logger.critical(msg)

    def save_config(self, config: Dict):
        """Save configuration to JSON file."""
        config_path = self.exp_dir / "config.json"

        # Make config JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        serializable_config = make_serializable(config)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        self.info(f"Configuration saved to {config_path}")

    def save_model_info(self, model: torch.nn.Module):
        """Save model architecture info."""
        info_path = self.exp_dir / "model_info.txt"

        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("Model Architecture\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(model))
            f.write("\n\n")
            f.write("Parameter Count\n")
            f.write("-" * 50 + "\n")

            total_params = 0
            trainable_params = 0
            for name, param in model.named_parameters():
                param_count = param.numel()
                total_params += param_count
                if param.requires_grad:
                    trainable_params += param_count
                f.write(f"{name}: {param_count:,} params\n")

            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")

        self.info(f"Model info saved to {info_path}")

    def flush(self):
        """Flush the TensorBoard writer."""
        if self.writer:
            self.writer.flush()

    def close(self):
        """Close the logger and TensorBoard writer."""
        if self.writer:
            self.writer.close()

        # Close file handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class LoggerContext:
    """Context manager for Logger."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.logger = None

    def __enter__(self) -> Logger:
        self.logger = Logger(*self.args, **self.kwargs)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.close()
        return False


def setup_logger(name: str, log_dir: str, level: str = "INFO") -> logging.Logger:
    """
    Setup a simple Python logger.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        log_path / f"{name}_{timestamp}.log",
        encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper()))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
