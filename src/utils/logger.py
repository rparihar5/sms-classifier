import logging
import os
from src.config import ROOT_DIR


def get_logger(name: str, log_file: str) -> logging.Logger:
    """
    Return a logger that writes ONLY to a file (no console output).

    Args:
        name: logger name (usually __name__ or a custom string)
        log_file: file name inside the `logs/` folder (e.g. 'predictions.log')
    """
    logger = logging.getLogger(name)

    # If logger already has handlers, reuse it (avoid duplicate entries)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Ensure logs directory exists
    logs_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # File handler (ONLY)
    file_path = os.path.join(logs_dir, log_file)
    file_handler = logging.FileHandler(file_path)
    file_format = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger
