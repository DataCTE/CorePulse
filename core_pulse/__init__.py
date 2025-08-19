"""
CorePulse: A modular toolkit for advanced diffusion model manipulation.

This package provides tools for fine-grained control over diffusion models,
including prompt injection, block-level conditioning, and other advanced techniques.
"""

__version__ = "0.1.0"
__author__ = "CorePulse Contributors"

from .prompt_injection import (
    SimplePromptInjector, 
    AdvancedPromptInjector, 
    AttentionMapInjector,
    SelfAttentionInjector,
    MultiScaleInjector,
    UnifiedAdvancedInjector
)
from .models import UNetPatcher
from .prompt_injection.masking import MaskedPromptInjector
from .prompt_injection.spatial import (
    RegionalPromptInjector, 
    # Quadrant masks
    create_top_left_quadrant_mask, create_top_right_quadrant_mask,
    create_bottom_left_quadrant_mask, create_bottom_right_quadrant_mask,
    # Half masks
    create_left_half_mask, create_right_half_mask,
    create_top_half_mask, create_bottom_half_mask,
    # Center masks and basic shapes
    create_center_square_mask, create_center_circle_mask,
    create_rectangle_mask, create_circle_mask,
    # Strip masks
    create_horizontal_strip_mask, create_vertical_strip_mask
)


__all__ = [
    "SimplePromptInjector",
    "AdvancedPromptInjector", 
    "AttentionMapInjector",
    "SelfAttentionInjector",
    "MultiScaleInjector",
    "UnifiedAdvancedInjector",
    "UNetPatcher",
    "MaskedPromptInjector",
    "RegionalPromptInjector",
    # Quadrant masks  
    "create_top_left_quadrant_mask",
    "create_top_right_quadrant_mask",
    "create_bottom_left_quadrant_mask",
    "create_bottom_right_quadrant_mask",
    # Half masks
    "create_left_half_mask",
    "create_right_half_mask", 
    "create_top_half_mask",
    "create_bottom_half_mask",
    # Center masks and basic shapes
    "create_center_square_mask",
    "create_center_circle_mask",
    "create_rectangle_mask", 
    "create_circle_mask",
    # Strip masks
    "create_horizontal_strip_mask",
    "create_vertical_strip_mask",
]
