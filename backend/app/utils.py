import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def handle_error(error: Exception) -> Dict[str, Any]:
    logger.error(f"Error occurred: {str(error)}")
    return {
        "error": str(error),
        "type": error.__class__.__name__
    }