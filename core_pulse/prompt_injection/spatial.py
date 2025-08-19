"""
Spatial masking system for regional prompt injection in CorePulse.

This module provides spatial/regional control over prompt injection,
allowing different conditioning to be applied to specific areas of the generated image.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from diffusers import DiffusionPipeline
from torchvision.transforms import functional as F
from PIL import Image

from .masking import MaskedPromptInjector
from ..models.base import BlockIdentifier
from ..utils.logger import logger


class MaskFactory:
    """
    A utility class providing static methods to create and manipulate spatial masks.
    
    Masks are returned as torch.Tensors with float values between 0.0 and 1.0.
    """
    
    @staticmethod
    def from_shape(shape_type: str,
                   image_size: Tuple[int, int] = (1024, 1024),
                   **params) -> torch.Tensor:
        """
        Create a mask from a predefined shape.
        """
        logger.debug(f"Creating mask from shape '{shape_type}' with size {image_size} and params {params}")
        try:
            height, width = image_size[1], image_size[0]
            mask = torch.zeros((height, width), dtype=torch.float32)
            
            shape_type = shape_type.lower()

            if shape_type == 'rectangle':
                x, y, w, h = params['x'], params['y'], params['width'], params['height']
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                mask[y1:y2, x1:x2] = 1.0
            
            elif shape_type == 'circle':
                cx, cy, radius = params['cx'], params['cy'], params['radius']
                y_coords = torch.arange(height)
                x_coords = torch.arange(width)
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                dist = torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
                mask[dist <= radius] = 1.0
                
            elif shape_type == 'ellipse':
                cx, cy = params['cx'], params['cy']
                w, h = params['width'], params['height']
                y_coords = torch.arange(height)
                x_coords = torch.arange(width)
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                ellipse_mask = ((x_grid - cx) / (w / 2))**2 + ((y_grid - cy) / (h / 2))**2 <= 1
                mask[ellipse_mask] = 1.0
                
            else:
                raise ValueError(f"Unsupported shape_type: {shape_type}")
                
            logger.debug(f"Created mask with shape {mask.shape} and mean value {mask.mean().item():.4f}")
            return mask
        except KeyError as e:
            logger.error(f"Missing required parameter for shape '{shape_type}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to create mask from shape '{shape_type}': {e}", exc_info=True)
            raise

    @staticmethod
    def from_image(image_path: str,
                   image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load a mask from an image file.
        """
        logger.debug(f"Loading mask from image: {image_path}")
        try:
            mask_image = Image.open(image_path).convert('L')
        except FileNotFoundError:
            logger.error(f"Mask image not found at: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load mask image '{image_path}': {e}", exc_info=True)
            raise

        if image_size:
            logger.debug(f"Resizing mask image to {image_size}")
            mask_image = mask_image.resize(image_size, Image.LANCZOS)
            
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_np)
        logger.debug(f"Loaded mask with shape {mask.shape} and mean value {mask.mean().item():.4f}")
        return mask

    @staticmethod
    def gaussian_blur(mask: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """
        Apply a Gaussian blur to soften the edges of a mask.
        """
        logger.debug(f"Applying Gaussian blur with kernel_size={kernel_size}, sigma={sigma}")
        try:
            if mask.ndim == 2:
                mask_unsqueezed = mask.unsqueeze(0).unsqueeze(0)
            else:
                mask_unsqueezed = mask

            if kernel_size % 2 == 0:
                kernel_size += 1
                
            blurred_mask = F.gaussian_blur(mask_unsqueezed, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            
            result = blurred_mask.squeeze(0).squeeze(0)
            logger.debug(f"Blurred mask mean value: {result.mean().item():.4f}")
            return result
        except Exception as e:
            logger.error(f"Failed to apply Gaussian blur: {e}", exc_info=True)
            raise

    @staticmethod
    def gradient(gradient_type: str,
                 image_size: Tuple[int, int] = (1024, 1024),
                 **params) -> torch.Tensor:
        """
        Create a linear or radial gradient mask.
        """
        logger.debug(f"Creating '{gradient_type}' gradient mask for size {image_size}")
        try:
            height, width = image_size[1], image_size[0]
            gradient_type = gradient_type.lower()
            
            if gradient_type == 'linear':
                start_val = params.get('start_val', 0.0)
                end_val = params.get('end_val', 1.0)
                direction = params.get('direction', 'horizontal')
                
                if direction == 'horizontal':
                    grad_vector = torch.linspace(start_val, end_val, width)
                    return grad_vector.repeat(height, 1)
                elif direction == 'vertical':
                    grad_vector = torch.linspace(start_val, end_val, height)
                    return grad_vector.unsqueeze(-1).repeat(1, width)
                else:
                    raise ValueError(f"Unsupported linear gradient direction: {direction}")

            elif gradient_type == 'radial':
                cx = params.get('cx', width // 2)
                cy = params.get('cy', height // 2)
                radius = params.get('radius', min(width, height) / 2)
                start_val = params.get('start_val', 1.0)
                end_val = params.get('end_val', 0.0)
                
                y_coords = torch.arange(height)
                x_coords = torch.arange(width)
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                dist = torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
                
                grad = dist / radius
                grad = torch.clamp(grad, 0, 1)
                
                grad = start_val + (end_val - start_val) * grad
                return torch.clamp(grad, 0, 1)

            else:
                raise ValueError(f"Unsupported gradient_type: {gradient_type}")
        except Exception as e:
            logger.error(f"Failed to create gradient mask: {e}", exc_info=True)
            raise

    @staticmethod
    def invert(mask: torch.Tensor) -> torch.Tensor:
        """
        Invert a mask (1.0 - mask).
        """
        logger.debug("Inverting mask.")
        return 1.0 - mask

    @staticmethod
    def combine(mask1: torch.Tensor,
                mask2: torch.Tensor,
                operation: str) -> torch.Tensor:
        """
        Combine two masks using a boolean-like operation.
        """
        logger.debug(f"Combining masks with operation: '{operation}'")
        try:
            operation = operation.lower()
            if operation == 'add':
                return torch.max(mask1, mask2)
            elif operation == 'subtract':
                return torch.clamp(mask1 - mask2, 0, 1)
            elif operation == 'multiply':
                return torch.min(mask1, mask2)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.error(f"Failed to combine masks with operation '{operation}': {e}", exc_info=True)
            raise


class RegionalPromptInjector(MaskedPromptInjector):
    """
    Advanced prompt injector with spatial masking capabilities.
    """
    
    def __init__(self, pipeline: DiffusionPipeline):
        super().__init__(pipeline)
        
        try:
            # Determine attention resolution from the UNet's down block types
            # This is a heuristic that works for SD1.5 and SDXL
            num_down_blocks = len(pipeline.unet.config['down_block_types'])
            self.attention_resolution = pipeline.unet.sample_size // (2 ** (num_down_blocks - 1))
            logger.info(f"Regional injector initialized with attention resolution: {self.attention_resolution}")
        except Exception as e:
            logger.error(f"Failed to determine attention resolution: {e}", exc_info=True)
            # Fallback to a default value
            self.attention_resolution = 64
            logger.warning(f"Could not determine attention resolution, falling back to default: {self.attention_resolution}")

    def add_regional_injection(self,
                               block: Union[str, BlockIdentifier],
                               prompt: str,
                               mask: torch.Tensor,
                               weight: float = 1.0,
                               sigma_start: float = 0.0,
                               sigma_end: float = 1.0):
        """
        Adds a spatially-masked prompt injection.

        Args:
            block: The UNet block(s) to apply this injection to.
            prompt: The text prompt for this layer.
            mask: A float tensor (0.0-1.0) representing the spatial mask.
            weight: Injection weight for conditioning.
            sigma_start: Start of the injection window in the diffusion process.
            sigma_end: End of the injection window.
        """
        logger.info(f"Adding regional injection for block '{block}' with prompt '{prompt[:30]}...'")
        logger.debug(f"Original mask shape: {mask.shape}, mean: {mask.mean().item():.4f}")
        
        try:
            # Resize mask to the attention resolution and flatten
            attention_mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(self.attention_resolution, self.attention_resolution),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0).flatten()
            logger.debug(f"Resized attention mask to shape: {attention_mask.shape}, mean: {attention_mask.mean().item():.4f}")

            self.add_injection(
                block=block,
                prompt=prompt,
                weight=weight,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                spatial_mask=attention_mask
            )
        except Exception as e:
            logger.error(f"Failed to add regional injection for block '{block}': {e}", exc_info=True)
            raise


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
