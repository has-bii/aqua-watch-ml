"""
Alternative logging setup for date-based log files.
This module provides different approaches for date-based logging.
"""

import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class DateBasedLogger:
    """Custom logger that creates date-based log files"""
    
    def __init__(self, log_path: str = "logs", app_name: str = "aquarium_ml"):
        self.log_path = log_path
        self.app_name = app_name
        self.logger = None
        
    def setup_daily_rotating_logger(self):
        """Setup logger with daily rotation at midnight"""
        os.makedirs(self.log_path, exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Timed rotating file handler (rotates at midnight)
        log_filename = os.path.join(self.log_path, f'{self.app_name}.log')
        file_handler = TimedRotatingFileHandler(
            filename=log_filename,
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.suffix = '%Y-%m-%d'
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        return logger
    
    def setup_date_filename_logger(self):
        """Setup logger with current date in filename"""
        os.makedirs(self.log_path, exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Create filename with current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_filename = os.path.join(self.log_path, f'{self.app_name}_{current_date}.log')
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        return logger
    
    def setup_hourly_rotating_logger(self):
        """Setup logger with hourly rotation"""
        os.makedirs(self.log_path, exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_filename = os.path.join(self.log_path, f'{self.app_name}.log')
        file_handler = TimedRotatingFileHandler(
            filename=log_filename,
            when='H',  # Hourly
            interval=1,
            backupCount=168,  # Keep 7 days worth of hourly logs
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.suffix = '%Y-%m-%d_%H'
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        return logger


# Example usage functions
def setup_logging_approach_1(log_path: str = "logs"):
    """Approach 1: TimedRotatingFileHandler - Automatic rotation at midnight"""
    date_logger = DateBasedLogger(log_path)
    return date_logger.setup_daily_rotating_logger()


def setup_logging_approach_2(log_path: str = "logs"):
    """Approach 2: Date in filename - Manual daily file creation"""
    date_logger = DateBasedLogger(log_path)
    return date_logger.setup_date_filename_logger()


def setup_logging_approach_3(log_path: str = "logs"):
    """Approach 3: Hourly rotation - For high-volume applications"""
    date_logger = DateBasedLogger(log_path)
    return date_logger.setup_hourly_rotating_logger()
