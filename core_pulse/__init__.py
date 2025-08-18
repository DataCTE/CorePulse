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

__all__ = [
    "SimplePromptInjector",
    "AdvancedPromptInjector", 
    "UNetPatcher",
    "MaskedPromptInjector",
]
