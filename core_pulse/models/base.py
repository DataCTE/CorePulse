"""
Base classes for model patching in CorePulse.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import torch
from diffusers import DiffusionPipeline
from dataclasses import dataclass


class BaseModelPatcher(ABC):
    """
    Abstract base class for model patchers.
    
    Model patchers are responsible for modifying the behavior of diffusion models
    during inference by applying patches to specific components.
    """
    
    def __init__(self):
        self.patches: Dict[str, Any] = {}
        self.is_patched = False
        self.original_methods: Dict[str, Callable] = {}
    
    @abstractmethod
    def apply_patches(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply patches to the given pipeline.
        
        Args:
            pipeline: The diffusion pipeline to patch
            
        Returns:
            The patched pipeline
        """
        pass
    
    @abstractmethod
    def remove_patches(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Remove patches from the given pipeline.
        
        Args:
            pipeline: The patched pipeline to restore
            
        Returns:
            The restored pipeline
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup patches."""
        if hasattr(self, '_patched_pipeline') and self._patched_pipeline is not None:
            self.remove_patches(self._patched_pipeline)


class BlockIdentifier:
    """
    Helper class for identifying and working with model blocks.
    """
    
    def __init__(self, block_type: str, block_index: int):
        self.block_type = block_type  # 'input', 'middle', 'output'
        self.block_index = block_index
    
    def __str__(self):
        return f"{self.block_type}:{self.block_index}"
    
    def __repr__(self):
        return f"BlockIdentifier('{self.block_type}', {self.block_index})"
    
    def __eq__(self, other):
        if isinstance(other, BlockIdentifier):
            return self.block_type == other.block_type and self.block_index == other.block_index
        return False
    
    def __hash__(self):
        return hash((self.block_type, self.block_index))
    
    @classmethod
    def from_string(cls, block_str: str) -> 'BlockIdentifier':
        """
        Create a BlockIdentifier from a string like 'input:4' or 'middle:0'.
        
        Args:
            block_str: String representation of the block
            
        Returns:
            BlockIdentifier instance
        """
        if ':' not in block_str:
            raise ValueError(f"Invalid block string format: {block_str}. Expected 'type:index'")
        
        block_type, block_index_str = block_str.split(':', 1)
        try:
            block_index = int(block_index_str)
        except ValueError:
            raise ValueError(f"Invalid block index: {block_index_str}. Must be an integer")
        
        return cls(block_type, block_index)
