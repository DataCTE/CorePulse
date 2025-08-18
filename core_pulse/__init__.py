"""
CorePulse: A modular toolkit for advanced diffusion model manipulation.

This package provides tools for fine-grained control over diffusion models,
including prompt injection, block-level conditioning, and other advanced techniques.
"""

__version__ = "0.1.0"
__author__ = "CorePulse Contributors"

from .prompt_injection import SimplePromptInjector, AdvancedPromptInjector
from .models import UNetPatcher
from .prompt_injection.masking import MaskedPromptInjector
from .prompt_injection.spatial import (
    RegionalPromptInjector, 
    # Quadrant regions
    create_top_left_quadrant_region, create_top_right_quadrant_region,
    create_bottom_left_quadrant_region, create_bottom_right_quadrant_region,
    # Half regions
    create_left_half_region, create_right_half_region,
    create_top_half_region, create_bottom_half_region,
    # Center regions and basic shapes
    create_center_region, create_center_square_region, create_center_circle_region,
    create_rectangle_region, create_circle_region,
    # Strip regions
    create_horizontal_strip_region, create_vertical_strip_region
)


__all__ = [
    "SimplePromptInjector",
    "AdvancedPromptInjector", 
    "UNetPatcher",
    "MaskedPromptInjector",
    "RegionalPromptInjector",
    # Quadrant regions  
    "create_top_left_quadrant_region",
    "create_top_right_quadrant_region",
    "create_bottom_left_quadrant_region",
    "create_bottom_right_quadrant_region",
    # Half regions
    "create_left_half_region",
    "create_right_half_region", 
    "create_top_half_region",
    "create_bottom_half_region",
    # Center regions and basic shapes
    "create_center_region",
    "create_center_square_region",
    "create_center_circle_region",
    "create_rectangle_region", 
    "create_circle_region",
    # Strip regions
    "create_horizontal_strip_region",
    "create_vertical_strip_region",
]
