"""
Logging configuration for CorePulse.
"""

import logging
import sys

# Get the top-level logger
logger = logging.getLogger('core_pulse')

# Default configuration
logger.setLevel(logging.WARNING)
logger.propagate = False  # Prevent duplicate logs in parent loggers

# Default handler
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_handler)

def set_core_pulse_debug_level(level: str = 'info'):
    """
    Set the logging level for the CorePulse library.

    Args:
        level (str): The desired logging level. One of 'debug', 'info', 
                     'warning', 'error', 'critical'.
    """
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    log_level = level_map.get(level.lower(), logging.INFO)
    logger.setLevel(log_level)
    
    if log_level == logging.DEBUG:
        logger.info("CorePulse debug logging enabled.")
    else:
        logger.info(f"CorePulse logging level set to {level.upper()}.")

# Initialize with a default informational message
# logger.info("CorePulse logger initialized. Call set_core_pulse_debug_level('debug') for verbose output.")
