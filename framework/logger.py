"""
Comprehensive logging system for the LLM Benchmarking Suite.
"""

import os
import logging
import logging.handlers
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from .config_manager import config


class BenchmarkLogger:
    """Enhanced logger for the benchmarking suite."""
    
    def __init__(self, name: str = "llm_benchmarking", log_level: Optional[str] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Get logging configuration
        log_config = config.get_logging_config()
        self.level = log_level or log_config.get('level', 'INFO')
        self.format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_file = log_config.get('file', 'logs/benchmarking.log')
        self.max_size = log_config.get('max_size', '10MB')
        self.backup_count = log_config.get('backup_count', 5)
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with file and console handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        self.logger.setLevel(level_map.get(self.level, logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter(self.format_str)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # Parse max_size (e.g., "10MB" -> 10 * 1024 * 1024)
            max_bytes = self._parse_size(self.max_size)
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with additional context."""
        if not kwargs:
            return message
        
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        return f"{message} | {context}"
    
    def log_benchmark_start(self, benchmark_name: str, model_name: str, num_examples: int):
        """Log benchmark start."""
        self.info(f"Starting benchmark: {benchmark_name}", 
                 model=model_name, examples=num_examples)
    
    def log_benchmark_complete(self, benchmark_name: str, model_name: str, 
                              metrics: Dict[str, Any], duration: float):
        """Log benchmark completion."""
        self.info(f"Completed benchmark: {benchmark_name}",
                 model=model_name, 
                 accuracy=metrics.get('accuracy', 0),
                 duration=f"{duration:.2f}s")
    
    def log_evaluation_progress(self, benchmark_name: str, current: int, total: int):
        """Log evaluation progress."""
        progress = (current / total) * 100 if total > 0 else 0
        self.debug(f"Progress: {benchmark_name}", 
                  current=current, total=total, progress=f"{progress:.1f}%")
    
    def log_model_request(self, model_name: str, prompt_length: int, response_length: int, 
                         duration: float, success: bool = True):
        """Log model API request."""
        status = "SUCCESS" if success else "FAILED"
        self.debug(f"Model request: {model_name}", 
                  status=status,
                  prompt_len=prompt_length,
                  response_len=response_length,
                  duration=f"{duration:.3f}s")
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log error with context."""
        self.error(f"Error in {context}: {str(error)}", 
                  error_type=type(error).__name__, **kwargs)
    
    def log_config_loaded(self, config_path: str):
        """Log configuration loading."""
        self.info(f"Configuration loaded from: {config_path}")
    
    def log_results_saved(self, filepath: str, benchmark_name: str, model_name: str):
        """Log results saving."""
        self.info(f"Results saved: {filepath}", 
                 benchmark=benchmark_name, model=model_name)


# Global logger instance
logger = BenchmarkLogger()
