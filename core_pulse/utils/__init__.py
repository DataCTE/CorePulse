"""
Utility functions for CorePulse.

This module provides helper functions for common tasks like
pipeline integration, model detection, and convenience wrappers.
"""

from .helpers import (
    detect_model_type,
    create_quick_injector,
    inject_and_generate,
    get_available_blocks,
    demonstrate_content_style_split
)

__all__ = [
    "detect_model_type",
    "create_quick_injector", 
    "inject_and_generate",
    "get_available_blocks",
    "demonstrate_content_style_split",
]
