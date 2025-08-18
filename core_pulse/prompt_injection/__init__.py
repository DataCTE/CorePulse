"""
Prompt injection tools for CorePulse.

This module provides interfaces for injecting prompts at specific blocks
of diffusion models, allowing fine-grained control over generation.
"""

from .base import BasePromptInjector
from .simple import SimplePromptInjector
from .advanced import AdvancedPromptInjector
from .masking import MaskedPromptInjector, TokenMask, TokenAnalyzer, MaskedPromptEncoder
from .spatial import (
    RegionalPromptInjector, SpatialMask, SpatialRegionType, CoordinateMapper,
    RegionalInjectionConfig, create_rectangle_region, create_circle_region,
    # Quadrant regions
    create_top_left_quadrant_region, create_top_right_quadrant_region,
    create_bottom_left_quadrant_region, create_bottom_right_quadrant_region,
    # Half regions
    create_left_half_region, create_right_half_region,
    create_top_half_region, create_bottom_half_region,
    # Center regions
    create_center_square_region, create_center_circle_region, create_center_region,
    # Strip regions
    create_horizontal_strip_region, create_vertical_strip_region
)

__all__ = [
    "BasePromptInjector",
    "SimplePromptInjector", 
    "AdvancedPromptInjector",
    "MaskedPromptInjector",
    "TokenMask",
    "TokenAnalyzer", 
    "MaskedPromptEncoder",
    "RegionalPromptInjector",
    "SpatialMask",
    "SpatialRegionType", 
    "CoordinateMapper",
    "RegionalInjectionConfig",
    "create_rectangle_region",
    "create_circle_region",
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
    # Center regions
    "create_center_square_region",
    "create_center_circle_region", 
    "create_center_region",
    # Strip regions
    "create_horizontal_strip_region",
    "create_vertical_strip_region",
]
