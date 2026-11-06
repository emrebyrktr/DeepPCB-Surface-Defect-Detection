import logging
import sys


def setup_logger(name: str = "pcb_defect", level: int = logging.INFO) -> logging.Logger:
    """Tekrarlanan handler eklenmesini önleyen standart logger döndürür."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

