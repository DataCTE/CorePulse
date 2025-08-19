"""
Base classes for model patching in CorePulse.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from ..utils.logger import logger


class BaseModelPatcher(ABC):
    """
    Abstract base class for model patchers.
    
    Model patchers are responsible for modifying the behavior of diffusion models
    during inference by applying patches to specific components.
    """
    
    def __init__(self):
        logger.debug(f"Initializing {self.__class__.__name__}.")
        self.patches: Dict[str, Any] = {}
        self.is_patched = False
        self.original_methods: Dict[str, Callable] = {}
    
    @abstractmethod
    def apply_patches(self, model: Any) -> Any:
        """
        Apply patches to the given model component (e.g., UNet).
        
        Args:
            model: The model component to patch.
            
        Returns:
            The patched model component.
        """
        pass
    
    @abstractmethod
    def remove_patches(self, model: Any) -> Any:
        """
        Remove patches from the given model component.
        
        Args:
            model: The patched model component to restore.
            
        Returns:
            The restored model component.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        logger.debug(f"Entering context for {self.__class__.__name__}.")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup patches."""
        logger.debug(f"Exiting context for {self.__class__.__name__}, cleaning up patches.")
        if hasattr(self, '_hooked_unet') and self._hooked_unet is not None:
            try:
                self.remove_patches(self._hooked_unet)
            except Exception as e:
                logger.error(f"Error during patch removal in __exit__: {e}", exc_info=True)
        elif hasattr(self, '_patched_pipeline') and self._patched_pipeline is not None:
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
        try:
            if ':' not in block_str:
                raise ValueError(f"Invalid block string format: {block_str}. Expected 'type:index'")
            
            block_type, block_index_str = block_str.split(':', 1)
            block_index = int(block_index_str)
            
            return cls(block_type, block_index)
        except ValueError as e:
            logger.error(f"Failed to create BlockIdentifier from string: '{block_str}'. {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred in BlockIdentifier.from_string: {e}", exc_info=True)
            raise
