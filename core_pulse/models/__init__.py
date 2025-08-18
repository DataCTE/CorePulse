"""
Model manipulation utilities for CorePulse.

This module provides base classes and utilities for patching and modifying
diffusion models, particularly UNet architectures.
"""

from .base import BaseModelPatcher
from .unet_patcher import UNetPatcher, UNetBlockMapper

__all__ = [
    "BaseModelPatcher",
    "UNetPatcher", 
    "UNetBlockMapper",
]
