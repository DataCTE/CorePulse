"""
Model manipulation utilities for CorePulse.

This module provides base classes and utilities for patching and modifying
diffusion models and transformer models, including UNet and transformer architectures.
"""

from .base import BaseModelPatcher, BlockIdentifier
from .unet_patcher import UNetPatcher, UNetBlockMapper
from .transformer_patcher import TransformerPatcher, LLMAttentionConfig

__all__ = [
    "BaseModelPatcher",
    "BlockIdentifier",
    "UNetPatcher", 
    "UNetBlockMapper",
    "TransformerPatcher",
    "LLMAttentionConfig",
]
