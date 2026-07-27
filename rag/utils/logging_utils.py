import logging

import colorlog


# TODO(prod): 12-factor logging refactor at deployment time — replace setup_logger with a single
# configure_logging() called once at startup + logging.getLogger(__name__) in modules; add a
# conditional JSON formatter (via LOG_FORMAT/ENV) for structured logs, level via env, and let the
# runtime ship stdout/stderr to Loki (Grafana stack). For now, logs go to the console only.


def setup_logger(module_name: str = "rag") -> logging.Logger:
    """Set up and configure a logger with colored console output.

    Args:
        module_name: Name used to identify module-specific logs.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(f"rag.{module_name}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the logger was already configured
    if not logger.handlers:
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
        logger.addHandler(console_handler)

        logger.info(f"{module_name} logger initialized")

    return logger
