"""
Self-attention control for CorePulse.

This module provides control over self-attention (image -> image attention)
allowing manipulation of how different parts of the generated image attend to each other.
This enables control over composition, coherence, and spatial relationships.
"""

from typing import Union, Optional
import torch
from diffusers import DiffusionPipeline

from .advanced import AdvancedPromptInjector
from ..models.base import BlockIdentifier
from ..prompt_injection.spatial import create_left_half_mask, create_right_half_mask, create_top_half_mask, create_bottom_half_mask
from ..utils.logger import logger


class SelfAttentionInjector(AdvancedPromptInjector):
    """
    An injector that provides fine-grained control over self-attention maps.
    
    Self-attention controls how different parts of the image attend to each other,
    enabling manipulation of composition, coherence, and spatial relationships.
    """

    def __init__(self, pipeline: DiffusionPipeline):
        """
        Initialize the self-attention injector.
        
        Args:
            pipeline: The diffusers pipeline to manipulate
        """
        super().__init__(pipeline)
        logger.info(f"Initialized SelfAttentionInjector for {self.model_type} model")

    def enhance_region_interaction(self,
                                 source_region: torch.Tensor,
                                 target_region: torch.Tensor,
                                 attention_scale: float = 2.0,
                                 block: Union[str, BlockIdentifier] = "all",
                                 sigma_start: float = 1.0,
                                 sigma_end: float = 0.0):
        """
        Enhance how one image region attends to another.
        
        Args:
            source_region: Spatial mask for the source region (what attends more)
            target_region: Spatial mask for the target region (what gets attended to)
            attention_scale: How much to amplify attention (>1.0 for enhancement)
            block: Which UNet blocks to apply to
            sigma_start: Start of diffusion window
            sigma_end: End of diffusion window
            
        Example:
            # Make left side of image attend more to right side
            left_mask = create_left_half_mask()
            right_mask = create_right_half_mask()
            injector.enhance_region_interaction(left_mask, right_mask, scale=3.0)
        """
        logger.info(f"Adding region interaction enhancement: {source_region.sum().item():.0f} -> {target_region.sum().item():.0f} pixels")
        
        self.patcher.add_self_attention_manipulation(
            block=block,
            source_region=source_region,
            target_region=target_region,
            attention_scale=attention_scale,
            interaction_type="enhance",
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

    def suppress_region_interaction(self,
                                  source_region: torch.Tensor,
                                  target_region: torch.Tensor,
                                  attention_scale: float = 0.3,
                                  block: Union[str, BlockIdentifier] = "all",
                                  sigma_start: float = 1.0,
                                  sigma_end: float = 0.0):
        """
        Suppress how one image region attends to another.
        
        Args:
            source_region: Spatial mask for the source region
            target_region: Spatial mask for the target region  
            attention_scale: How much to reduce attention (<1.0 for suppression)
            block: Which UNet blocks to apply to
            sigma_start: Start of diffusion window
            sigma_end: End of diffusion window
            
        Example:
            # Make foreground ignore background
            fg_mask = create_center_circle_mask()
            bg_mask = MaskFactory.invert(fg_mask)
            injector.suppress_region_interaction(fg_mask, bg_mask, scale=0.1)
        """
        logger.info(f"Adding region interaction suppression: {source_region.sum().item():.0f} -/-> {target_region.sum().item():.0f} pixels")
        
        self.patcher.add_self_attention_manipulation(
            block=block,
            source_region=source_region,
            target_region=target_region,
            attention_scale=attention_scale,
            interaction_type="suppress",
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

    def redirect_attention(self,
                         source_region: torch.Tensor,
                         old_target: torch.Tensor,
                         new_target: torch.Tensor,
                         attention_scale: float = 3.0,
                         block: Union[str, BlockIdentifier] = "all",
                         sigma_start: float = 1.0,
                         sigma_end: float = 0.0):
        """
        Redirect attention from one target region to another.
        
        Args:
            source_region: Which pixels to redirect attention FROM
            old_target: Current target region (attention will be reduced)
            new_target: New target region (attention will be enhanced)
            attention_scale: Strength of redirection
            block: Which UNet blocks to apply to
            sigma_start: Start of diffusion window
            sigma_end: End of diffusion window
            
        Example:
            # Make top half attend to bottom instead of center
            top = create_top_half_mask()
            center = create_center_square_mask()
            bottom = create_bottom_half_mask()
            injector.redirect_attention(top, center, bottom)
        """
        logger.info(f"Adding attention redirection: {source_region.sum().item():.0f} pixels")
        
        # First suppress attention to old target
        self.suppress_region_interaction(
            source_region, old_target, 
            attention_scale=1.0/attention_scale,
            block=block, sigma_start=sigma_start, sigma_end=sigma_end
        )
        
        # Then enhance attention to new target
        self.enhance_region_interaction(
            source_region, new_target,
            attention_scale=attention_scale,
            block=block, sigma_start=sigma_start, sigma_end=sigma_end
        )

    def enhance_global_coherence(self,
                               attention_scale: float = 1.5,
                               block: Union[str, BlockIdentifier] = "middle:0",
                               sigma_start: float = 1.0,
                               sigma_end: float = 0.5):
        """
        Globally enhance self-attention to improve image coherence.
        
        Args:
            attention_scale: Global attention amplification
            block: Which blocks to apply to (middle blocks recommended for coherence)
            sigma_start: Start of diffusion window
            sigma_end: End of diffusion window
            
        Example:
            # Improve overall image coherence
            injector.enhance_global_coherence(scale=2.0)
        """
        logger.info(f"Adding global coherence enhancement with scale {attention_scale}")
        
        self.patcher.add_self_attention_manipulation(
            block=block,
            source_region=None,  # Global application
            target_region=None,
            attention_scale=attention_scale,
            interaction_type="enhance",
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

    def create_attention_barrier(self,
                               barrier_region: torch.Tensor,
                               attention_scale: float = 0.1,
                               block: Union[str, BlockIdentifier] = "all",
                               sigma_start: float = 1.0,
                               sigma_end: float = 0.0):
        """
        Create an "attention barrier" that prevents regions from attending across it.
        
        Args:
            barrier_region: Spatial mask defining the barrier
            attention_scale: How much to suppress cross-barrier attention
            block: Which UNet blocks to apply to
            sigma_start: Start of diffusion window
            sigma_end: End of diffusion window
            
        Example:
            # Create vertical barrier to separate left/right composition
            barrier = create_vertical_strip_mask(x_start=480, width=64)
            injector.create_attention_barrier(barrier, scale=0.05)
        """
        logger.info(f"Creating attention barrier across {barrier_region.sum().item():.0f} pixels")
        
        # Create complementary regions on either side of barrier
        full_image = torch.ones_like(barrier_region)
        non_barrier = full_image - barrier_region
        
        # Suppress attention from barrier to non-barrier and vice versa
        self.suppress_region_interaction(
            barrier_region, non_barrier,
            attention_scale=attention_scale,
            block=block, sigma_start=sigma_start, sigma_end=sigma_end
        )
        
        self.suppress_region_interaction(
            non_barrier, barrier_region,
            attention_scale=attention_scale,
            block=block, sigma_start=sigma_start, sigma_end=sigma_end
        )


# Convenience functions for common self-attention patterns
def enhance_left_right_interaction(pipeline: DiffusionPipeline, 
                                 attention_scale: float = 2.0,
                                 image_size: tuple = (1024, 1024)) -> SelfAttentionInjector:
    """Make left and right halves of image attend to each other more."""
    injector = SelfAttentionInjector(pipeline)
    left = create_left_half_mask(image_size)
    right = create_right_half_mask(image_size)
    
    injector.enhance_region_interaction(left, right, attention_scale)
    injector.enhance_region_interaction(right, left, attention_scale)
    
    return injector


def enhance_top_bottom_interaction(pipeline: DiffusionPipeline,
                                 attention_scale: float = 2.0,
                                 image_size: tuple = (1024, 1024)) -> SelfAttentionInjector:
    """Make top and bottom halves of image attend to each other more.""" 
    injector = SelfAttentionInjector(pipeline)
    top = create_top_half_mask(image_size)
    bottom = create_bottom_half_mask(image_size)
    
    injector.enhance_region_interaction(top, bottom, attention_scale)
    injector.enhance_region_interaction(bottom, top, attention_scale)
    
    return injector


def improve_composition_coherence(pipeline: DiffusionPipeline,
                                coherence_scale: float = 1.8) -> SelfAttentionInjector:
    """Globally improve image composition and coherence."""
    injector = SelfAttentionInjector(pipeline)
    injector.enhance_global_coherence(coherence_scale)
    return injector
