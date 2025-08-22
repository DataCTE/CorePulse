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
    RegionalPromptInjector,
    # Mask creation helpers
    create_rectangle_mask, create_circle_mask,
    # Quadrant masks
    create_top_left_quadrant_mask, create_top_right_quadrant_mask,
    create_bottom_left_quadrant_mask, create_bottom_right_quadrant_mask,
    # Half masks
    create_left_half_mask, create_right_half_mask,
    create_top_half_mask, create_bottom_half_mask,
    # Center masks
    create_center_square_mask, create_center_circle_mask,
    # Strip masks
    create_horizontal_strip_mask, create_vertical_strip_mask
)
from .attention import AttentionMapInjector
from .self_attention import SelfAttentionInjector
from .multi_scale import MultiScaleInjector
from .unified import UnifiedAdvancedInjector
from .llm_attention import LLMAttentionInjector, create_concept_amplified_model, create_balanced_attention_model

__all__ = [
    "BasePromptInjector",
    "SimplePromptInjector", 
    "AdvancedPromptInjector",
    "MaskedPromptInjector",
    "TokenMask",
    "TokenAnalyzer", 
    "MaskedPromptEncoder",
    "RegionalPromptInjector",
    "AttentionMapInjector",
    "SelfAttentionInjector",
    "MultiScaleInjector",
    "UnifiedAdvancedInjector",
    # LLM attention manipulation
    "LLMAttentionInjector",
    "create_concept_amplified_model",
    "create_balanced_attention_model",
    "RegionalComposition",
    "CompositionLayer",
    "BlendMode",
    "MaskFactory",
    "create_rectangle_mask",
    "create_circle_mask",
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
    # Center masks
    "create_center_square_mask",
    "create_center_circle_mask", 
    # Strip masks
    "create_horizontal_strip_mask",
    "create_vertical_strip_mask",
]
