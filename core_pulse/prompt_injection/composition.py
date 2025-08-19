"""
CorePulse Compositional Prompt Injection System.

This module provides an advanced, layer-based system for regional prompt injection,
allowing for complex compositions with overlapping masks, blend modes, and soft transitions.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from diffusers import DiffusionPipeline
from enum import Enum
from torchvision.transforms import functional as F
from PIL import Image

# --- Enums and Data Classes ---

class BlendMode(Enum):
    """Defines how a layer's conditioning is blended with layers below it."""
    REPLACE = "replace"  # The top layer's conditioning replaces the bottom one.
    ADD = "add"          # Conditioning is added (weighted by opacity and mask).
    SUBTRACT = "subtract"    # Conditioning is subtracted.
    AVERAGE = "average"      # Conditioning is averaged.

class CompositionLayer:
    """
    Represents a single layer in the regional composition.
    
    Each layer contains a prompt, a spatial mask, and properties that define
    how it interacts with other layers in the composition stack.
    """
    def __init__(self,
                 prompt: str,
                 mask: torch.Tensor,
                 blend_mode: BlendMode = BlendMode.REPLACE,
                 opacity: float = 1.0,
                 weight: float = 1.0,
                 sigma_start: float = 0.0,
                 sigma_end: float = 1.0):
        """
        Initialize a composition layer.

        Args:
            prompt: The text prompt for this layer.
            mask: A float tensor (0.0-1.0) representing the spatial mask.
                  Shape should be compatible with the attention resolution.
            blend_mode: How to blend this layer with underlying layers.
            opacity: The overall influence of this layer (0.0 to 1.0).
            weight: Injection weight for conditioning.
            sigma_start: Start of the injection window in the diffusion process.
            sigma_end: End of the injection window.
        """
        self.prompt = prompt
        self.mask = mask
        self.blend_mode = blend_mode
        self.opacity = np.clip(opacity, 0.0, 1.0)
        self.weight = weight
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        
        # Placeholder for encoded prompt embeddings
        self._encoded_prompt: Optional[torch.Tensor] = None

# --- Core Composition and Masking Logic ---

class RegionalComposition:
    """
    Manages a stack of CompositionLayer objects and compiles them into
    a final, blended prompt embedding for the diffusion model.
    """
    def __init__(self, image_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize the regional composition manager.

        Args:
            image_size: The target image size (width, height).
        """
        self.layers: List[CompositionLayer] = []
        self.image_size = image_size
        
    def add_layer(self, layer: CompositionLayer):
        """
        Add a new layer to the top of the composition stack.

        Args:
            layer: The CompositionLayer to add.
        """
        self.layers.append(layer)

    def _encode_prompt(self, prompt: str, pipeline: DiffusionPipeline) -> torch.Tensor:
        """Helper to encode a single prompt."""
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(pipeline.device)
        else:
            attention_mask = None

        prompt_embeds = pipeline.text_encoder(
            text_input_ids.to(pipeline.device),
            attention_mask=attention_mask,
        )
        return prompt_embeds[0]

    def compile(self, pipeline: DiffusionPipeline) -> torch.Tensor:
        """
        Compile all layers into a single prompt embedding.

        This method iterates through the layers from bottom to top, blending
        their prompt embeddings according to their masks, blend modes, and opacity.

        Args:
            pipeline: The DiffusionPipeline used for encoding prompts.

        Returns:
            A single, composite prompt embedding tensor.
        """
        if not self.layers:
            # Return a neutral embedding if there are no layers
            return self._encode_prompt("", pipeline)

        # Encode all layer prompts first
        for layer in self.layers:
            layer._encoded_prompt = self._encode_prompt(layer.prompt, pipeline)

        # Start with the embedding of the bottom layer as the base
        composite_embedding = self.layers[0]._encoded_prompt.clone()

        # Iterate over the rest of the layers and blend them
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            
            # Prepare the mask by adding the token dimension
            mask = layer.mask.unsqueeze(-1)
            
            # Apply opacity to the mask
            blending_mask = mask * layer.opacity

            if layer.blend_mode == BlendMode.REPLACE:
                composite_embedding = torch.lerp(
                    composite_embedding, layer._encoded_prompt, blending_mask
                )
            elif layer.blend_mode == BlendMode.ADD:
                additive_part = layer._encoded_prompt * blending_mask
                composite_embedding += additive_part
            elif layer.blend_mode == BlendMode.SUBTRACT:
                subtractive_part = layer._encoded_prompt * blending_mask
                composite_embedding -= subtractive_part
            elif layer.blend_mode == BlendMode.AVERAGE:
                # Average only in the masked region
                avg_embed = (composite_embedding + layer._encoded_prompt) / 2.0
                composite_embedding = torch.lerp(
                    composite_embedding, avg_embed, blending_mask
                )
        
        return composite_embedding

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

        Args:
            shape_type: 'rectangle', 'circle', 'ellipse'.
            image_size: The target image size (width, height).
            **params: Shape-specific parameters.
                - rectangle: x, y, width, height
                - circle: cx, cy, radius
                - ellipse: cx, cy, width, height

        Returns:
            A torch.Tensor representing the mask.
        """
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
            
        return mask

    @staticmethod
    def from_image(image_path: str,
                   image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load a mask from an image file (e.g., a black and white PNG).

        Args:
            image_path: Path to the image file.
            image_size: Optional size to resize the mask to (width, height).

        Returns:
            A torch.Tensor representing the mask.
        """
        try:
            mask_image = Image.open(image_path).convert('L')
        except FileNotFoundError:
            raise FileNotFoundError(f"Mask image not found at: {image_path}")

        if image_size:
            mask_image = mask_image.resize(image_size, Image.LANCZOS)
            
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np)

    @staticmethod
    def gaussian_blur(mask: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """
        Apply a Gaussian blur to soften the edges of a mask.

        Args:
            mask: The input mask tensor (H, W).
            kernel_size: The size of the Gaussian kernel.
            sigma: The standard deviation of the Gaussian kernel.

        Returns:
            The blurred mask tensor.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Kernel size must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        blurred_mask = F.gaussian_blur(mask, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        
        return blurred_mask.squeeze(0).squeeze(0)

    @staticmethod
    def gradient(gradient_type: str,
                 image_size: Tuple[int, int] = (1024, 1024),
                 **params) -> torch.Tensor:
        """
        Create a linear or radial gradient mask.

        Args:
            gradient_type: 'linear' or 'radial'.
            image_size: The size of the mask (width, height).
            **params: Gradient-specific parameters.
                - linear: start_val (0-1), end_val (0-1), direction ('horizontal' or 'vertical')
                - radial: center_x, center_y, radius, start_val (0-1), end_val (0-1)

        Returns:
            A torch.Tensor representing the gradient.
        """
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
            
            # Create gradient from distance
            grad = dist / radius
            grad = torch.clamp(grad, 0, 1) # Clamp to [0, 1] range
            
            # Interpolate between start and end values
            grad = start_val + (end_val - start_val) * grad
            return torch.clamp(grad, 0, 1)

        else:
            raise ValueError(f"Unsupported gradient_type: {gradient_type}")

    @staticmethod
    def invert(mask: torch.Tensor) -> torch.Tensor:
        """
        Invert a mask (1.0 - mask).

        Args:
            mask: The input mask tensor.

        Returns:
            The inverted mask tensor.
        """
        return 1.0 - mask

    @staticmethod
    def combine(mask1: torch.Tensor,
                mask2: torch.Tensor,
                operation: str) -> torch.Tensor:
        """
        Combine two masks using a boolean-like operation.

        Args:
            mask1: The first mask tensor.
            mask2: The second mask tensor.
            operation: 'add' (max), 'subtract' (clip(m1-m2, 0, 1)), 'multiply' (min).

        Returns:
            The combined mask tensor.
        """
        operation = operation.lower()
        if operation == 'add':
            return torch.max(mask1, mask2)
        elif operation == 'subtract':
            return torch.clamp(mask1 - mask2, 0, 1)
        elif operation == 'multiply':
            return torch.min(mask1, mask2)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
