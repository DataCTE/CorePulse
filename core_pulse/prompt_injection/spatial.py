"""
Spatial masking system for regional prompt injection in CorePulse.

This module provides spatial/regional control over prompt injection,
allowing different conditioning to be applied to specific areas of the generated image.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from diffusers import DiffusionPipeline
from enum import Enum

from .masking import MaskedPromptInjector, TokenMask, TokenAnalyzer, MaskedPromptEncoder
from ..models.base import BlockIdentifier
from .composition import RegionalComposition, CompositionLayer, BlendMode, MaskFactory


class CoordinateMapper:
    """
    Maps between different coordinate systems: pixel space, latent space, attention space.
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (512, 512),
                 latent_size: Tuple[int, int] = (64, 64)):
        """
        Initialize coordinate mapper.
        
        Args:
            image_size: Pixel image size (width, height)
            latent_size: Latent space size (width, height)
        """
        self.image_size = image_size
        self.latent_size = latent_size
        
        # Calculate scaling factors
        self.latent_scale_x = latent_size[0] / image_size[0]
        self.latent_scale_y = latent_size[1] / image_size[1]
    
    def pixel_to_latent(self, pixel_coords: Tuple[int, int]) -> Tuple[int, int]:
        """Convert pixel coordinates to latent coordinates."""
        px, py = pixel_coords
        lx = int(px * self.latent_scale_x)
        ly = int(py * self.latent_scale_y)
        return (lx, ly)
    
    def latent_to_attention_index(self, latent_coords: Tuple[int, int]) -> int:
        """Convert latent coordinates to 1D attention token index."""
        lx, ly = latent_coords
        # Standard row-major indexing
        return ly * self.latent_size[0] + lx
    
    def pixel_to_attention_index(self, pixel_coords: Tuple[int, int]) -> int:
        """Convert pixel coordinates directly to attention token index."""
        latent_coords = self.pixel_to_latent(pixel_coords)
        return self.latent_to_attention_index(latent_coords)


# class RegionalInjectionConfig:
#     """
#     Configuration for regional prompt injection combining spatial and token masking.
#     """
    
#     def __init__(self,
#                  spatial_mask: SpatialMask,
#                  base_prompt: str,
#                  injection_prompt: str,
#                  target_phrase: Optional[str] = None,
#                  weight: float = 1.0,
#                  sigma_start: float = 0.0,
#                  sigma_end: float = 1.0,
#                  fuzzy_match: bool = True):
#         """
#         Initialize regional injection configuration.
        
#         Args:
#             spatial_mask: Spatial mask defining the region
#             base_prompt: Original prompt
#             injection_prompt: Prompt with replacement content
#             target_phrase: Specific phrase to replace (if None, replaces entire prompt)
#             weight: Injection weight (1.0 = normal, >1.0 = amplified, <1.0 = weakened)
#             sigma_start: Start of injection window
#             sigma_end: End of injection window  
#             fuzzy_match: Whether to use fuzzy phrase matching
#         """
#         self.spatial_mask = spatial_mask
#         self.base_prompt = base_prompt
#         self.injection_prompt = injection_prompt
#         self.target_phrase = target_phrase
#         self.weight = weight
#         self.sigma_start = sigma_start
#         self.sigma_end = sigma_end
#         self.fuzzy_match = fuzzy_match
        
#         # Cache for processed masks and embeddings
#         self._token_mask: Optional[TokenMask] = None
#         self._attention_mask: Optional[torch.Tensor] = None
#         self._blended_embedding: Optional[torch.Tensor] = None


class RegionalPromptInjector(MaskedPromptInjector):
    """
    Advanced prompt injector with both token-level and spatial masking capabilities.
    
    Allows applying different prompt injections to specific regions of the image
    while optionally targeting specific phrases within those regions.
    """
    
    def __init__(self, model_type: str = "sdxl"):
        """
        Initialize regional prompt injector.
        
        Args:
            model_type: Model type ("sdxl" or "sd15")
        """
        super().__init__(model_type)
        # self.regional_configs: Dict[str, RegionalInjectionConfig] = {}
        self.coord_mapper: Optional[CoordinateMapper] = None
        
        # Determine latent and attention sizes based on model type
        if model_type.lower() == "sd15":
            self.image_size = (512, 512)
            self.latent_size = (64, 64)
            self.attention_resolution = 64  # 64x64 = 4096 tokens
        elif model_type.lower() == "sdxl":
            self.image_size = (1024, 1024)
            self.latent_size = (128, 128) 
            self.attention_resolution = 128  # 128x128 = 16384 tokens
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.coord_mapper = CoordinateMapper(self.image_size, self.latent_size)
        self.composition = RegionalComposition(self.image_size)
    
    def add_regional_layer(self,
                           prompt: str,
                           mask: torch.Tensor,
                           blend_mode: BlendMode = BlendMode.REPLACE,
                           opacity: float = 1.0,
                           weight: float = 1.0,
                           sigma_start: float = 0.0,
                           sigma_end: float = 1.0,
                           block: Union[str, BlockIdentifier] = "all"):
        """
        Adds a compositional layer for regional injection.

        Args:
            prompt: The text prompt for this layer.
            mask: A float tensor (0.0-1.0) representing the spatial mask.
            blend_mode: How to blend this layer with underlying layers.
            opacity: The overall influence of this layer (0.0 to 1.0).
            weight: Injection weight for conditioning.
            sigma_start: Start of the injection window in the diffusion process.
            sigma_end: End of the injection window.
            block: The UNet block(s) to apply this injection to.
        """
        # Ensure mask is resized to attention resolution and flattened
        attention_mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(self.attention_resolution, self.attention_resolution),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        layer = CompositionLayer(
            prompt=prompt,
            mask=attention_mask.flatten(), # Flatten for 1D attention tokens
            blend_mode=blend_mode,
            opacity=opacity,
            weight=weight,
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )
        self.composition.add_layer(layer)
        
        # Store block info separately for apply_to_pipeline
        # This is a bit of a hack; a cleaner way might be to integrate blocks into the composition
        if not hasattr(self, '_layer_blocks'):
            self._layer_blocks = []
        self._layer_blocks.append(block)

        
    # def get_regional_summary(self) -> List[Dict[str, Any]]:
    #     """
    #     Get summary of all regional injection configurations.
        
    #     Returns:
    #         List of regional injection summaries
    #     """
    #     summaries = []
    #     for config_id, config in self.regional_configs.items():
    #         summary = {
    #             "config_id": config_id,
    #             "region_type": config.spatial_mask.region_type.value,
    #             "region_params": config.spatial_mask.region_params,
    #             "base_prompt": config.base_prompt,
    #             "injection_prompt": config.injection_prompt,
    #             "target_phrase": config.target_phrase,
    #             "weight": config.weight,
    #             "sigma_range": f"{config.sigma_start:.1f} - {config.sigma_end:.1f}",
    #             "fuzzy_match": config.fuzzy_match
    #         }
    #         summaries.append(summary)
    #     return summaries
    
    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply regional injections to pipeline.
        
        Args:
            pipeline: Pipeline to modify
            
        Returns:
            Modified pipeline
        """
        if not self.composition.layers:
            # Fall back to regular injection if no composition layers
            return super().apply_to_pipeline(pipeline)
        
        # Compile the final embedding from the composition
        blended_embedding = self.composition.compile(pipeline)

        # Apply the same blended embedding across all specified blocks for each layer
        # Note: This implementation assumes you want the fully blended prompt
        # applied at different stages (blocks). A more complex setup could
        # compile different compositions for different blocks.
        if hasattr(self, '_layer_blocks'):
            for i, layer in enumerate(self.composition.layers):
                block = self._layer_blocks[i]
                
                # Use the compiled embedding for all injections
                # The spatial mask is handled during compilation.
                # The weight and sigma from the layer are used for injection timing.
                self.add_injection(
                    block=block,
                    prompt="",  # Prompt is pre-encoded
                    weight=layer.weight,
                    sigma_start=layer.sigma_start,
                    sigma_end=layer.sigma_end,
                    spatial_mask=None # Mask is already in the compiled embedding
                )

                # Set the encoded prompt for the corresponding config
                if isinstance(block, str) and block.lower() == "all":
                    all_blocks = self.patcher.block_mapper.get_all_block_identifiers()
                    for block_id_str in all_blocks:
                        block_id = BlockIdentifier.from_string(block_id_str)
                        if block_id in self.configs:
                            self.configs[block_id]._encoded_prompt = blended_embedding
                else:
                    block_id = BlockIdentifier.from_string(block) if isinstance(block, str) else block
                    if block_id in self.configs:
                        self.configs[block_id]._encoded_prompt = blended_embedding

        return super().apply_to_pipeline(pipeline)
    
    def clear_injections(self):
        """Clear all injections including regional ones."""
        super().clear_injections()
        self.composition.layers.clear()
        if hasattr(self, '_layer_blocks'):
            self._layer_blocks.clear()


# Helper functions for creating common spatial regions
def create_rectangle_mask(x: int, y: int, width: int, height: int, 
                          image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a rectangular mask."""
    return MaskFactory.from_shape('rectangle', image_size, x=x, y=y, width=width, height=height)


def create_circle_mask(cx: int, cy: int, radius: int,
                       image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a circular mask.""" 
    return MaskFactory.from_shape('circle', image_size, cx=cx, cy=cy, radius=radius)


# Quadrant masks
def create_top_left_quadrant_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the top-left quadrant of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(0, 0, w // 2, h // 2, image_size)


def create_top_right_quadrant_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the top-right quadrant of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(w // 2, 0, w - w // 2, h // 2, image_size)


def create_bottom_left_quadrant_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the bottom-left quadrant of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(0, h // 2, w // 2, h - h // 2, image_size)


def create_bottom_right_quadrant_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the bottom-right quadrant of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(w // 2, h // 2, w - w // 2, h - h // 2, image_size)


# Half masks 
def create_left_half_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the left half of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(0, 0, w // 2, h, image_size)


def create_right_half_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the right half of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(w // 2, 0, w - w // 2, h, image_size)


def create_top_half_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the top half of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(0, 0, w, h // 2, image_size)


def create_bottom_half_mask(image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Create a mask covering the bottom half of the image."""
    h, w = image_size[1], image_size[0]
    return create_rectangle_mask(0, h // 2, w, h - h // 2, image_size)


# Center masks
def create_center_square_mask(image_size: Tuple[int, int] = (1024, 1024), 
                              size: Optional[int] = None) -> torch.Tensor:
    """Create a centered square mask.""" 
    if size is None:
        size = min(image_size) // 3
    
    w, h = image_size
    x = (w - size) // 2
    y = (h - size) // 2
    return create_rectangle_mask(x, y, size, size, image_size)


def create_center_circle_mask(image_size: Tuple[int, int] = (1024, 1024), 
                              radius: Optional[int] = None) -> torch.Tensor:
    """Create a centered circular mask."""
    if radius is None:
        radius = min(image_size) // 4
    
    cx, cy = image_size[0] // 2, image_size[1] // 2
    return create_circle_mask(cx, cy, radius, image_size)


# Strip masks
def create_horizontal_strip_mask(image_size: Tuple[int, int] = (1024, 1024),
                                 y_start: Optional[int] = None,
                                 height: Optional[int] = None) -> torch.Tensor:
    """Create a horizontal strip mask."""
    if y_start is None:
        y_start = image_size[1] // 3
    if height is None:
        height = image_size[1] // 3
    
    return create_rectangle_mask(0, y_start, image_size[0], height, image_size)


def create_vertical_strip_mask(image_size: Tuple[int, int] = (1024, 1024),
                               x_start: Optional[int] = None,
                               width: Optional[int] = None) -> torch.Tensor:
    """Create a vertical strip mask."""
    if x_start is None:
        x_start = image_size[0] // 3
    if width is None:
        width = image_size[0] // 3
        
    return create_rectangle_mask(x_start, 0, width, image_size[1], image_size)
