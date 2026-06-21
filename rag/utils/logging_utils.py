import logging
import os
from datetime import datetime

import colorlog


def setup_logger(module_name: str = "rag") -> logging.Logger:
    """Set up and configure a logger with colored console output and a file handler.

    Args:
        module_name: Name used to identify module-specific logs.

    Returns:
        Configured Logger instance.
    """
    # Create logs directory next to the rag package if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rag_{timestamp}.log")

    logger = logging.getLogger(f"rag.{module_name}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the logger was already configured
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        console_handler.setFormatter(color_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"{module_name} logger initialized")

    return logger
