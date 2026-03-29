import sys
from loguru import logger
from strategy_ideation_engine.config.settings import settings

def setup_logger():
    """
    Phase 0: Configure structured logging (Loguru).
    Ensures all pipeline events are audit-ready.
    """
    logger.remove()  # Remove default handler
    
    # Console output (Standard)
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    
    # File output (Persistent)
    logger.add(
        "logs/strategy_engine.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        compression="zip"
    )

# Initialize on import
setup_logger()
