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


class SpatialRegionType(Enum):
    """Types of spatial regions that can be defined."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle" 
    ELLIPSE = "ellipse"
    ARBITRARY = "arbitrary"  # Custom mask array
    
    # Quadrants
    TOP_LEFT_QUADRANT = "top_left_quadrant"
    TOP_RIGHT_QUADRANT = "top_right_quadrant"
    BOTTOM_LEFT_QUADRANT = "bottom_left_quadrant"
    BOTTOM_RIGHT_QUADRANT = "bottom_right_quadrant"
    
    # Halves
    LEFT_HALF = "left_half"
    RIGHT_HALF = "right_half"
    TOP_HALF = "top_half"
    BOTTOM_HALF = "bottom_half"
    
    # Center regions
    CENTER_SQUARE = "center_square"
    CENTER_CIRCLE = "center_circle"
    
    # Strips
    HORIZONTAL_STRIP = "horizontal_strip"
    VERTICAL_STRIP = "vertical_strip"


class SpatialMask:
    """
    Represents a spatial mask defining a region of the image for targeted injection.
    """
    
    def __init__(self, 
                 region_type: SpatialRegionType,
                 image_size: Tuple[int, int] = (512, 512),
                 **region_params):
        """
        Initialize spatial mask.
        
        Args:
            region_type: Type of spatial region
            image_size: Size of the target image (width, height)
            **region_params: Region-specific parameters:
                - rectangle: x, y, width, height (pixel coordinates)
                - circle: center_x, center_y, radius
                - ellipse: center_x, center_y, width, height  
                - arbitrary: mask (2D numpy array)
        """
        self.region_type = region_type
        self.image_size = image_size
        self.region_params = region_params
        
        # Generate the mask array
        self._mask_array = self._create_mask_array()
        
    def _create_mask_array(self) -> np.ndarray:
        """Create the 2D mask array based on region type and parameters."""
        height, width = self.image_size[1], self.image_size[0]
        mask = np.zeros((height, width), dtype=np.float32)
        
        if self.region_type == SpatialRegionType.RECTANGLE:
            x = self.region_params['x']
            y = self.region_params['y'] 
            w = self.region_params['width']
            h = self.region_params['height']
            
            # Clamp coordinates to image bounds
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            x2 = max(x, min(x + w, width))
            y2 = max(y, min(y + h, height))
            
            mask[y:y2, x:x2] = 1.0
            
        elif self.region_type == SpatialRegionType.CIRCLE:
            cx = self.region_params['center_x']
            cy = self.region_params['center_y']
            radius = self.region_params['radius']
            
            # Create coordinate grids
            y_grid, x_grid = np.ogrid[:height, :width]
            distance = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask[distance <= radius] = 1.0
            
        elif self.region_type == SpatialRegionType.ELLIPSE:
            cx = self.region_params['center_x']
            cy = self.region_params['center_y']
            w = self.region_params['width']
            h = self.region_params['height']
            
            # Create coordinate grids
            y_grid, x_grid = np.ogrid[:height, :width]
            ellipse_mask = ((x_grid - cx) / (w / 2))**2 + ((y_grid - cy) / (h / 2))**2 <= 1
            mask[ellipse_mask] = 1.0
            
        elif self.region_type == SpatialRegionType.ARBITRARY:
            provided_mask = self.region_params['mask']
            if provided_mask.shape != (height, width):
                # Resize mask using simple numpy interpolation
                from PIL import Image
                mask_pil = Image.fromarray((provided_mask * 255).astype(np.uint8), mode='L')
                mask_pil_resized = mask_pil.resize((width, height), Image.LANCZOS)
                mask = np.array(mask_pil_resized).astype(np.float32) / 255.0
            else:
                mask = provided_mask.copy()
        
        # Quadrant regions
        elif self.region_type == SpatialRegionType.TOP_LEFT_QUADRANT:
            mask[:height//2, :width//2] = 1.0
            
        elif self.region_type == SpatialRegionType.TOP_RIGHT_QUADRANT:
            mask[:height//2, width//2:] = 1.0
            
        elif self.region_type == SpatialRegionType.BOTTOM_LEFT_QUADRANT:
            mask[height//2:, :width//2] = 1.0
            
        elif self.region_type == SpatialRegionType.BOTTOM_RIGHT_QUADRANT:
            mask[height//2:, width//2:] = 1.0
        
        # Half regions
        elif self.region_type == SpatialRegionType.LEFT_HALF:
            mask[:, :width//2] = 1.0
            
        elif self.region_type == SpatialRegionType.RIGHT_HALF:
            mask[:, width//2:] = 1.0
            
        elif self.region_type == SpatialRegionType.TOP_HALF:
            mask[:height//2, :] = 1.0
            
        elif self.region_type == SpatialRegionType.BOTTOM_HALF:
            mask[height//2:, :] = 1.0
        
        # Center regions
        elif self.region_type == SpatialRegionType.CENTER_SQUARE:
            size = self.region_params.get('size', min(width, height) // 3)
            cx, cy = width // 2, height // 2
            x1, x2 = max(0, cx - size//2), min(width, cx + size//2)
            y1, y2 = max(0, cy - size//2), min(height, cy + size//2)
            mask[y1:y2, x1:x2] = 1.0
            
        elif self.region_type == SpatialRegionType.CENTER_CIRCLE:
            radius = self.region_params.get('radius', min(width, height) // 4)
            cx, cy = width // 2, height // 2
            y_grid, x_grid = np.ogrid[:height, :width]
            distance = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask[distance <= radius] = 1.0
        
        # Strip regions
        elif self.region_type == SpatialRegionType.HORIZONTAL_STRIP:
            y_start = self.region_params.get('y_start', height // 3)
            strip_height = self.region_params.get('height', height // 3)
            y_end = min(height, y_start + strip_height)
            mask[y_start:y_end, :] = 1.0
            
        elif self.region_type == SpatialRegionType.VERTICAL_STRIP:
            x_start = self.region_params.get('x_start', width // 3)
            strip_width = self.region_params.get('width', width // 3)
            x_end = min(width, x_start + strip_width)
            mask[:, x_start:x_end] = 1.0
                
        return mask
    
    @property 
    def mask_array(self) -> np.ndarray:
        """Get the 2D mask array."""
        return self._mask_array
    
    def to_latent_space(self, latent_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert pixel-space mask to latent-space mask.
        
        Args:
            latent_size: Size of latent space (width, height)
            
        Returns:
            Mask resized to latent dimensions
        """
        from PIL import Image
        latent_w, latent_h = latent_size
        
        # Resize mask from pixel space to latent space using PIL
        mask_pil = Image.fromarray((self._mask_array * 255).astype(np.uint8), mode='L')
        mask_pil_resized = mask_pil.resize((latent_w, latent_h), Image.LANCZOS)
        latent_mask = np.array(mask_pil_resized).astype(np.float32) / 255.0
        
        return latent_mask
    
    def to_attention_tokens(self, attention_resolution: int) -> torch.Tensor:
        """
        Convert mask to attention token mask.
        
        Args:
            attention_resolution: Resolution of cross-attention (e.g., 64 for SD 1.5)
            
        Returns:
            1D mask tensor for attention tokens
        """
        from PIL import Image
        
        # Resize to attention resolution using PIL
        mask_pil = Image.fromarray((self._mask_array * 255).astype(np.uint8), mode='L')
        mask_pil_resized = mask_pil.resize((attention_resolution, attention_resolution), Image.LANCZOS)
        attention_mask = np.array(mask_pil_resized).astype(np.float32) / 255.0
        
        # Flatten to 1D for attention tokens
        attention_mask_1d = attention_mask.flatten()
        
        # Convert to torch tensor
        return torch.from_numpy(attention_mask_1d).float()


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


class RegionalInjectionConfig:
    """
    Configuration for regional prompt injection combining spatial and token masking.
    """
    
    def __init__(self,
                 spatial_mask: SpatialMask,
                 base_prompt: str,
                 injection_prompt: str,
                 target_phrase: Optional[str] = None,
                 weight: float = 1.0,
                 sigma_start: float = 0.0,
                 sigma_end: float = 1.0,
                 fuzzy_match: bool = True):
        """
        Initialize regional injection configuration.
        
        Args:
            spatial_mask: Spatial mask defining the region
            base_prompt: Original prompt
            injection_prompt: Prompt with replacement content
            target_phrase: Specific phrase to replace (if None, replaces entire prompt)
            weight: Injection weight (1.0 = normal, >1.0 = amplified, <1.0 = weakened)
            sigma_start: Start of injection window
            sigma_end: End of injection window  
            fuzzy_match: Whether to use fuzzy phrase matching
        """
        self.spatial_mask = spatial_mask
        self.base_prompt = base_prompt
        self.injection_prompt = injection_prompt
        self.target_phrase = target_phrase
        self.weight = weight
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.fuzzy_match = fuzzy_match
        
        # Cache for processed masks and embeddings
        self._token_mask: Optional[TokenMask] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._blended_embedding: Optional[torch.Tensor] = None


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
        self.regional_configs: Dict[str, RegionalInjectionConfig] = {}
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
    
    def add_regional_injection(self,
                             region: Union[SpatialMask, Dict[str, Any]],
                             base_prompt: str,
                             injection_prompt: str,
                             target_phrase: Optional[str] = None,
                             block: Union[str, BlockIdentifier] = "all",
                             weight: float = 1.0,
                             sigma_start: float = 0.0,
                             sigma_end: float = 1.0,
                             fuzzy_match: bool = True):
        """
        Add a regional injection that applies to a specific area of the image.
        
        Args:
            region: SpatialMask or dict with region parameters
            base_prompt: Original prompt
            injection_prompt: Prompt with replacement content
            target_phrase: Specific phrase to replace (if None, replaces entire prompt in region)
            block: Block identifier or "all"
            weight: Injection weight (1.0 = normal, >1.0 = amplified, <1.0 = weakened)
            sigma_start: Start of injection window
            sigma_end: End of injection window
            fuzzy_match: Whether to use fuzzy phrase matching
        """
        # Convert dict to SpatialMask if needed
        if isinstance(region, dict):
            region_type = SpatialRegionType(region.get('type', 'rectangle'))
            region_params = {k: v for k, v in region.items() if k != 'type'}
            spatial_mask = SpatialMask(region_type, self.image_size, **region_params)
        else:
            spatial_mask = region
        
        # Create configuration
        config_id = f"regional_{len(self.regional_configs)}"
        config = RegionalInjectionConfig(
            spatial_mask=spatial_mask,
            base_prompt=base_prompt,
            injection_prompt=injection_prompt,
            target_phrase=target_phrase,
            weight=weight,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            fuzzy_match=fuzzy_match
        )
        
        self.regional_configs[config_id] = config
        
        # Also store the block for later processing
        config._block = block
        
    def get_regional_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all regional injection configurations.
        
        Returns:
            List of regional injection summaries
        """
        summaries = []
        for config_id, config in self.regional_configs.items():
            summary = {
                "config_id": config_id,
                "region_type": config.spatial_mask.region_type.value,
                "region_params": config.spatial_mask.region_params,
                "base_prompt": config.base_prompt,
                "injection_prompt": config.injection_prompt,
                "target_phrase": config.target_phrase,
                "weight": config.weight,
                "sigma_range": f"{config.sigma_start:.1f} - {config.sigma_end:.1f}",
                "fuzzy_match": config.fuzzy_match
            }
            summaries.append(summary)
        return summaries
    
    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply regional injections to pipeline.
        
        Args:
            pipeline: Pipeline to modify
            
        Returns:
            Modified pipeline
        """
        if not self.regional_configs:
            # Fall back to regular injection if no regional configs
            return super().apply_to_pipeline(pipeline)
        
        # Initialize analyzers
        if hasattr(pipeline, 'tokenizer'):
            self._token_analyzer = TokenAnalyzer(pipeline.tokenizer)
            self._masked_encoder = MaskedPromptEncoder(self)
        else:
            raise ValueError("Pipeline must have tokenizer for regional injection")
        
        # Process each regional configuration
        for config_id, config in self.regional_configs.items():
            # Create token mask if target phrase is specified
            token_mask = None
            if config.target_phrase:
                token_mask = self._token_analyzer.create_phrase_mask(
                    config.base_prompt, 
                    config.target_phrase,
                    config.fuzzy_match
                )
            
            # Encode the conditioning
            if token_mask:
                # Use masked encoding for token-level control
                blended_embedding = self._masked_encoder.encode_masked_prompt(
                    config.base_prompt,
                    config.injection_prompt, 
                    token_mask,
                    pipeline
                )
            else:
                # Use full prompt replacement
                blended_embedding = self.encode_prompt(config.injection_prompt, pipeline)
            
            # Create spatial mask for attention tokens
            spatial_mask = config.spatial_mask.to_attention_tokens(self.attention_resolution)
            
            # Add to configs with spatial mask  
            block = getattr(config, '_block', 'all')
            self.add_injection(
                block=block,
                prompt="",  # Empty prompt since we have pre-encoded embedding
                weight=config.weight,
                sigma_start=config.sigma_start,
                sigma_end=config.sigma_end,
                spatial_mask=spatial_mask
            )
            
            # Replace the encoded prompt directly in the configs
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
        self.regional_configs.clear()


# Helper functions for creating common spatial regions
def create_rectangle_region(x: int, y: int, width: int, height: int, 
                          image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a rectangular spatial region."""
    return SpatialMask(SpatialRegionType.RECTANGLE, image_size,
                      x=x, y=y, width=width, height=height)


def create_circle_region(center_x: int, center_y: int, radius: int,
                        image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a circular spatial region.""" 
    return SpatialMask(SpatialRegionType.CIRCLE, image_size,
                      center_x=center_x, center_y=center_y, radius=radius)


# Quadrant regions
def create_top_left_quadrant_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the top-left quadrant of the image."""
    return SpatialMask(SpatialRegionType.TOP_LEFT_QUADRANT, image_size)


def create_top_right_quadrant_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the top-right quadrant of the image."""
    return SpatialMask(SpatialRegionType.TOP_RIGHT_QUADRANT, image_size)


def create_bottom_left_quadrant_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the bottom-left quadrant of the image."""
    return SpatialMask(SpatialRegionType.BOTTOM_LEFT_QUADRANT, image_size)


def create_bottom_right_quadrant_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the bottom-right quadrant of the image."""
    return SpatialMask(SpatialRegionType.BOTTOM_RIGHT_QUADRANT, image_size)


# Half regions 
def create_left_half_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the left half of the image."""
    return SpatialMask(SpatialRegionType.LEFT_HALF, image_size)


def create_right_half_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the right half of the image."""
    return SpatialMask(SpatialRegionType.RIGHT_HALF, image_size)


def create_top_half_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the top half of the image."""
    return SpatialMask(SpatialRegionType.TOP_HALF, image_size)


def create_bottom_half_region(image_size: Tuple[int, int] = (512, 512)) -> SpatialMask:
    """Create a region covering the bottom half of the image."""
    return SpatialMask(SpatialRegionType.BOTTOM_HALF, image_size)


# Center regions
def create_center_square_region(image_size: Tuple[int, int] = (512, 512), 
                               size: Optional[int] = None) -> SpatialMask:
    """Create a centered square region.""" 
    if size is None:
        size = min(image_size) // 3
    return SpatialMask(SpatialRegionType.CENTER_SQUARE, image_size, size=size)


def create_center_circle_region(image_size: Tuple[int, int] = (512, 512), 
                               radius: Optional[int] = None) -> SpatialMask:
    """Create a centered circular region."""
    if radius is None:
        radius = min(image_size) // 4
    return SpatialMask(SpatialRegionType.CENTER_CIRCLE, image_size, radius=radius)


def create_center_region(image_size: Tuple[int, int] = (512, 512), 
                        scale: float = 0.5) -> SpatialMask:
    """Create a centered circular region (legacy compatibility)."""
    width, height = image_size
    center_x, center_y = width // 2, height // 2
    radius = int(min(width, height) * scale / 2)
    return create_circle_region(center_x, center_y, radius, image_size)


# Strip regions
def create_horizontal_strip_region(image_size: Tuple[int, int] = (512, 512),
                                  y_start: Optional[int] = None,
                                  height: Optional[int] = None) -> SpatialMask:
    """Create a horizontal strip region."""
    if y_start is None:
        y_start = image_size[1] // 3
    if height is None:
        height = image_size[1] // 3
    return SpatialMask(SpatialRegionType.HORIZONTAL_STRIP, image_size, 
                      y_start=y_start, height=height)


def create_vertical_strip_region(image_size: Tuple[int, int] = (512, 512),
                                x_start: Optional[int] = None,
                                width: Optional[int] = None) -> SpatialMask:
    """Create a vertical strip region."""
    if x_start is None:
        x_start = image_size[0] // 3
    if width is None:
        width = image_size[0] // 3
    return SpatialMask(SpatialRegionType.VERTICAL_STRIP, image_size,
                      x_start=x_start, width=width)
