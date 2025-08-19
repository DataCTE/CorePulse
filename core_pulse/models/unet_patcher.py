"""
UNet patching utilities for prompt injection.

This module provides functionality similar to ComfyUI's prompt injection
but adapted for diffusers pipelines. It patches the UNet's cross-attention
mechanism to inject conditioning at specific blocks and timesteps.
"""

import torch
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from .base import BaseModelPatcher, BlockIdentifier
from dataclasses import dataclass
from ..utils.logger import logger


class UNetBlockMapper:
    """
    Dynamically maps ComfyUI-style block identifiers to the structure of a
    given diffusers UNet model by inspecting its configuration.
    """
    
    def __init__(self, unet: UNet2DConditionModel):
        """
        Initialize the block mapper by inspecting the UNet's config.
        
        Args:
            unet: The UNet2DConditionModel to map.
        """
        self.unet_config = unet.config
        self.blocks = self._build_block_map()
        self.resolution_levels = self._build_resolution_map()

    def _build_block_map(self) -> Dict[str, List[int]]:
        """Build the block map from the UNet's configuration."""
        block_map = {'input': [], 'middle': [], 'output': []}
        
        # Input blocks
        for i, block_type in enumerate(self.unet_config.get("down_block_types", [])):
            if "CrossAttn" in block_type:
                block_map['input'].append(i)
        
        # Middle block (always has cross-attention)
        if self.unet_config.get("mid_block_type", None) and "CrossAttn" in self.unet_config["mid_block_type"]:
            block_map['middle'].append(0)
            
        # Output blocks
        for i, block_type in enumerate(self.unet_config.get("up_block_types", [])):
            if "CrossAttn" in block_type:
                block_map['output'].append(i)
                
        return block_map
    
    def _build_resolution_map(self) -> Dict[str, Dict[str, Union[int, str]]]:
        """
        Build a mapping of blocks to their resolution levels.
        
        Returns:
            Dictionary mapping block identifiers to resolution info:
            {
                "input:0": {"level": 0, "resolution": "high"},
                "input:1": {"level": 1, "resolution": "medium"},  
                "middle:0": {"level": 3, "resolution": "lowest"},
                "output:0": {"level": 2, "resolution": "low"},
                ...
            }
        """
        resolution_map = {}
        
        # Input blocks: start high resolution, go down (downsample)
        input_blocks = self.blocks.get('input', [])
        for i, block_idx in enumerate(input_blocks):
            block_id = f"input:{block_idx}"
            resolution_map[block_id] = {
                "level": i,
                "resolution": self._level_to_resolution_name(i),
                "stage": "downsample"
            }
        
        # Middle blocks: lowest resolution
        middle_blocks = self.blocks.get('middle', [])
        max_input_level = len(input_blocks)
        for i, block_idx in enumerate(middle_blocks):
            block_id = f"middle:{block_idx}"  
            resolution_map[block_id] = {
                "level": max_input_level + i,
                "resolution": "lowest",
                "stage": "bottleneck"
            }
        
        # Output blocks: start low resolution, go up (upsample)
        output_blocks = self.blocks.get('output', [])
        max_middle_level = max_input_level + len(middle_blocks)
        for i, block_idx in enumerate(output_blocks):
            block_id = f"output:{block_idx}"
            # Output blocks reverse the resolution order
            reverse_level = max_middle_level - 1 - i
            resolution_map[block_id] = {
                "level": reverse_level,
                "resolution": self._level_to_resolution_name(max(0, reverse_level - len(middle_blocks))),
                "stage": "upsample"
            }
        
        return resolution_map
    
    def _level_to_resolution_name(self, level: int) -> str:
        """Convert numeric resolution level to descriptive name."""
        if level == 0:
            return "highest"
        elif level == 1:
            return "high" 
        elif level == 2:
            return "medium"
        elif level == 3:
            return "low"
        else:
            return "lowest"
    
    def get_blocks_by_resolution(self, resolution: str) -> List[str]:
        """
        Get all block identifiers at a specific resolution level.
        
        Args:
            resolution: "highest", "high", "medium", "low", "lowest"
            
        Returns:
            List of block identifiers at that resolution
        """
        matching_blocks = []
        for block_id, info in self.resolution_levels.items():
            if info["resolution"] == resolution:
                matching_blocks.append(block_id)
        return matching_blocks
    
    def get_blocks_by_stage(self, stage: str) -> List[str]:
        """
        Get all block identifiers at a specific processing stage.
        
        Args:
            stage: "downsample", "bottleneck", "upsample"
            
        Returns:
            List of block identifiers at that stage
        """
        matching_blocks = []
        for block_id, info in self.resolution_levels.items():
            if info["stage"] == stage:
                matching_blocks.append(block_id)
        return matching_blocks

    def get_valid_blocks(self) -> Dict[str, List[int]]:
        """Get all valid blocks for this model type."""
        return self.blocks.copy()
    
    def get_all_block_identifiers(self) -> List[str]:
        """Get all valid block identifiers as strings for this model type."""
        all_blocks = []
        for block_type, block_indices in self.blocks.items():
            for block_index in block_indices:
                all_blocks.append(f"{block_type}:{block_index}")
        return all_blocks
    
    def is_valid_block(self, block: BlockIdentifier) -> bool:
        """Check if a block identifier is valid for this model."""
        return (block.block_type in self.blocks and 
                block.block_index in self.blocks[block.block_type])
    
    def map_to_diffusers_path(self, block: BlockIdentifier) -> str:
        """
        Map a block identifier to a diffusers module path.
        
        Args:
            block: Block identifier to map
            
        Returns:
            String path to the module in diffusers UNet
        """
        try:
            if not self.is_valid_block(block):
                model_name = self.unet_config.get("model_type", "unknown")
                raise ValueError(f"Invalid block for {model_name}: {block}")
            
            if block.block_type == "input":
                return f"down_blocks.{block.block_index}"
            elif block.block_type == "middle": 
                # Middle block is just "mid_block", not indexed
                return "mid_block"
            elif block.block_type == "output":
                return f"up_blocks.{block.block_index}"
            else:
                raise ValueError(f"Unknown block type: {block.block_type}")
        except ValueError as e:
            logger.error(f"Failed to map block {block}: {e}", exc_info=True)
            raise


@dataclass
class AttentionMapConfig:
    """Configuration for a single attention map manipulation."""
    target_token_indices: List[int]
    attention_scale: float
    spatial_mask: Optional[torch.Tensor] = None
    sigma_start: float = 14.0
    sigma_end: float = 0.0


@dataclass
class SelfAttentionConfig:
    """Configuration for self-attention manipulation (image -> image attention)."""
    source_region: Optional[torch.Tensor] = None  # Which pixels to modify attention FROM
    target_region: Optional[torch.Tensor] = None  # Which pixels to modify attention TO  
    attention_scale: float = 1.0  # How much to amplify/reduce attention
    interaction_type: str = "enhance"  # "enhance", "suppress", "redirect"
    sigma_start: float = 14.0
    sigma_end: float = 0.0


class PromptInjectionProcessor:
    """
    Custom attention processor for injecting prompts at specific blocks.
    
    This is similar to ComfyUI's prompt injection but adapted for diffusers.
    It hooks into the cross-attention mechanism to replace conditioning
    at specific blocks and timesteps with optional spatial control.
    """
    
    def __init__(self, original_processor, block_id: str, conditioning: Optional[torch.Tensor] = None, 
                 weight: float = 1.0, sigma_start: float = 0.0, sigma_end: float = 1.0,
                 spatial_mask: Optional[torch.Tensor] = None,
                 attention_maps: Optional[List[AttentionMapConfig]] = None,
                 self_attention_configs: Optional[List[SelfAttentionConfig]] = None,
                 module_path: Optional[str] = None):
        self.original_processor = original_processor
        self.block_id = block_id  # e.g., "middle:0", "output:1"
        self.conditioning = conditioning
        self.weight = weight
        self.sigma_start = sigma_start  # Higher values (more noise)
        self.sigma_end = sigma_end      # Lower values (less noise)
        self.spatial_mask = spatial_mask  # Optional spatial mask for regional control
        self.attention_maps = attention_maps or []
        self.self_attention_configs = self_attention_configs or []
        self.projection_layer = None # For handling mismatched dimensions
        self.module_path = module_path or ""  # Full path to the module (for debugging)
        
        # Determine if this is self-attention based on module path
        self.is_self_attention_module = "attn1" in self.module_path if self.module_path else False
        
        # Parse block info
        if ':' in block_id:
            self.block_type, self.block_index = block_id.split(':')
            self.block_index = int(self.block_index)
        else:
            self.block_type = block_id
            self.block_index = 0
    
    def __call__(self, attn: Attention, hidden_states: torch.Tensor, 
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 timestep: Optional[int] = None,
                 **kwargs) -> torch.Tensor:
        
        try:
            # Get current sigma from timestep if available
            current_sigma = getattr(self, '_current_sigma', None)
            
            # Check if we should inject at this timestep/sigma
            should_inject = self._should_inject(current_sigma)
            
            if (should_inject and 
                encoder_hidden_states is not None and 
                self.conditioning is not None):
                
                
                if self.spatial_mask is not None:
                    # Spatial injection: apply injection and then blend spatially
                    original_shape = encoder_hidden_states.shape
                    injected_states = self._apply_injection(encoder_hidden_states)
                    
                    # Run attention with injected conditioning
                    attention_output = self._compute_attention(
                        attn, hidden_states, injected_states, attention_mask, **kwargs
                    )
                    
                    # Apply spatial masking to blend injection and original
                    return self._apply_spatial_masking(
                        attention_output, hidden_states, attn, encoder_hidden_states,
                        attention_mask, timestep, **kwargs
                    )
                else:
                    # Non-spatial injection: standard approach
                    original_shape = encoder_hidden_states.shape
                    encoder_hidden_states = self._apply_injection(encoder_hidden_states)

            # Standard attention computation (or with injected states)
            return self._compute_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )
        except Exception as e:
            logger.error(f"Error in PromptInjectionProcessor for block {self.block_id}: {e}", exc_info=True)
            # Fallback to original processor to prevent crashing the whole generation
            if hasattr(self.original_processor, '__call__'):
                return self.original_processor(
                    attn, hidden_states, encoder_hidden_states, attention_mask=attention_mask, **kwargs
                )
            else:
                # If original processor is not callable, re-raise the exception
                raise
    
    def _compute_attention(self, attn: Attention, hidden_states: torch.Tensor,
                           encoder_hidden_states: Optional[torch.Tensor] = None,
                           attention_mask: Optional[torch.Tensor] = None,
                           **kwargs) -> torch.Tensor:
        """
        Replicates the logic of AttnProcessor2_0 to allow for attention score manipulation.
        """
        residual = hidden_states
        try:
            input_ndim = hidden_states.ndim
            
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            
            query = attn.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            attention_scores = torch.bmm(query, key.transpose(-1, -2))
            attention_scores = attention_scores * attn.scale
            
            # --- ATTENTION MAP MANIPULATION (Cross-Attention) ---
            current_sigma = getattr(self, '_current_sigma', None)
            for attn_map in self.attention_maps:
                if current_sigma is None or (current_sigma <= attn_map.sigma_start and current_sigma >= attn_map.sigma_end):
                    
                    # Filter indices to prevent out-of-bounds errors
                    max_token_idx = attention_scores.shape[-1]
                    valid_indices = [idx for idx in attn_map.target_token_indices if idx < max_token_idx]
                    
                    if not valid_indices:
                        continue

                    logger.debug(f"[{self.block_id}] Applying attention map scale {attn_map.attention_scale} to tokens {valid_indices}")
                    
                    # Log stats before manipulation
                    logger.debug(f"  - Attn scores BEFORE: mean={attention_scores.mean().item():.4f}, max={attention_scores.max().item():.4f}, std={attention_scores.std().item():.4f}")

                    # Apply scaling to the specified token indices
                    # Shape of attention_probs: (batch_size * num_heads, pixel_patches, text_tokens)
                    scale_tensor = torch.ones_like(attention_scores)
                    
                    # Temporarily disable spatial mask for debugging
                    # if attn_map.spatial_mask is not None:
                    #     spatial_mask = attn_map.spatial_mask.to(device=scale_tensor.device, dtype=scale_tensor.dtype)
                    #     # Reshape mask to match (1, pixel_patches, 1) for broadcasting
                    #     spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(-1) 
                    #     # Apply scale only where mask is > 0
                    #     scale_factor = torch.where(spatial_mask > 0, attn_map.attention_scale, 1.0)
                    # else:
                    scale_factor = attn_map.attention_scale

                    scale_tensor[:, :, valid_indices] *= scale_factor
                    attention_scores = attention_scores * scale_tensor
                    
                    # Log stats after manipulation
                    logger.debug(f"  - Attn scores AFTER:  mean={attention_scores.mean().item():.4f}, max={attention_scores.max().item():.4f}, std={attention_scores.std().item():.4f}")
            
            # --- SELF-ATTENTION MANIPULATION (Image -> Image) ---
            # Use module path to determine if this is self-attention
            if self.is_self_attention_module and len(self.self_attention_configs) > 0:
                logger.warning(f"[{self.block_id}] SELF-ATTENTION MODULE ({self.module_path}): hidden={hidden_states.shape}, encoder={encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
                for i, self_attn_config in enumerate(self.self_attention_configs):
                    if current_sigma is None or (current_sigma <= self_attn_config.sigma_start and current_sigma >= self_attn_config.sigma_end):
                        logger.warning(f"[{self.block_id}] APPLYING SELF-ATTENTION {i}: {self_attn_config.interaction_type} with scale {self_attn_config.attention_scale}")
                        
                        # Store original for comparison
                        orig_mean = attention_scores.mean().item()
                        orig_std = attention_scores.std().item()
                        
                        # Apply self-attention manipulation based on interaction type
                        attention_scores = self._apply_self_attention_manipulation(
                            attention_scores, self_attn_config
                        )
                        
                        # Log the change
                        new_mean = attention_scores.mean().item()
                        new_std = attention_scores.std().item()
                        logger.warning(f"[{self.block_id}] SELF-ATTENTION CHANGED: {orig_mean:.4f}->{new_mean:.4f} mean, {orig_std:.4f}->{new_std:.4f} std")
                    else:
                        logger.info(f"[{self.block_id}] Skipping self-attention {i}: sigma {current_sigma} not in range [{self_attn_config.sigma_start}, {self_attn_config.sigma_end}]")
            else:
                # This is cross-attention or self-attention without configs
                if self.is_self_attention_module:
                    logger.debug(f"[{self.block_id}] SELF-ATTENTION module ({self.module_path}) but no configs")
                else:
                    logger.debug(f"[{self.block_id}] CROSS-ATTENTION module ({self.module_path})")
            # --- END MANIPULATION ---

            attention_probs = attention_scores.softmax(dim=-1)
            logger.debug(f"  - Attn probs AFTER SOFTMAX: mean={attention_probs.mean().item():.4f}, max={attention_probs.max().item():.4f}, std={attention_probs.std().item():.4f}")
            
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            
            return hidden_states
        except Exception as e:
            logger.error(f"Error during attention computation for block {self.block_id}: {e}", exc_info=True)
            # Fallback to residual to prevent crashing
            return residual

    def _should_inject(self, current_sigma: Optional[float]) -> bool:
        """Check if we should inject based on current sigma."""
        if current_sigma is None:
            # If we don't have sigma info, always inject for now
            return True
        
        # ComfyUI logic: inject if sigma is within range
        # sigma_start should be > sigma_end (high noise to low noise)
        return current_sigma <= self.sigma_start and current_sigma >= self.sigma_end
    
    def _apply_injection(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the conditioning injection (for text embeddings only)."""
        try:
            batch_size = encoder_hidden_states.shape[0]
            
            # Ensure conditioning matches the required shape
            if self.conditioning.shape[0] == 1 and batch_size > 1:
                injected_conditioning = self.conditioning.repeat(batch_size, 1, 1)
            else:
                injected_conditioning = self.conditioning[:batch_size]
            
            # Move to same device and dtype
            injected_conditioning = injected_conditioning.to(
                device=encoder_hidden_states.device, 
                dtype=encoder_hidden_states.dtype
            )
            
            # Apply injection with ComfyUI's approach
            if batch_size == 2:
                # Replace only the conditional part (batch 1), keep unconditional (batch 0) 
                result = encoder_hidden_states.clone()
                # Ensure injected conditioning matches the shape of the target
                if injected_conditioning.shape[-1] != result.shape[-1]:
                    if self.projection_layer is None:
                        source_dim = injected_conditioning.shape[-1]
                        target_dim = result.shape[-1]
                        self.projection_layer = torch.nn.Linear(source_dim, target_dim)
                        self.projection_layer.to(device=result.device, dtype=result.dtype)
                    injected_conditioning = self.projection_layer(injected_conditioning)

                result[1] = injected_conditioning[0] * torch.tensor(self.weight, device=injected_conditioning.device, dtype=injected_conditioning.dtype)
            else:
                # Single batch: full replacement
                if injected_conditioning.shape[-1] != encoder_hidden_states.shape[-1]:
                    if self.projection_layer is None:
                        source_dim = injected_conditioning.shape[-1]
                        target_dim = encoder_hidden_states.shape[-1]
                        self.projection_layer = torch.nn.Linear(source_dim, target_dim)
                        self.projection_layer.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                    injected_conditioning = self.projection_layer(injected_conditioning)
                result = injected_conditioning * torch.tensor(self.weight, device=injected_conditioning.device, dtype=injected_conditioning.dtype)
            
            return result
        except Exception as e:
            logger.error(f"Failed to apply injection for block {self.block_id}: {e}", exc_info=True)
            # Return original states as a fallback
            return encoder_hidden_states
    
    def _apply_spatial_masking(self, attention_output: torch.Tensor, 
                              original_hidden_states: torch.Tensor,
                              attn: Attention, encoder_hidden_states: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None,
                              timestep: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Apply spatial masking to attention output by blending with original."""
        try:
            if self.spatial_mask is None:
                return attention_output
            
            # Run original attention without injection to get baseline
            original_attention_output = self._compute_attention(
                attn, original_hidden_states, encoder_hidden_states, 
                attention_mask, **kwargs
            )
            
            # Prepare spatial mask
            spatial_mask = self.spatial_mask.to(
                device=attention_output.device,
                dtype=attention_output.dtype
            )
            
            # Ensure mask matches spatial dimensions
            seq_len = attention_output.shape[1]  # Spatial sequence length
            if spatial_mask.shape[0] != seq_len:
                if spatial_mask.shape[0] > seq_len:
                    spatial_mask = spatial_mask[:seq_len]
                else:
                    # Pad or interpolate mask to match sequence length
                    spatial_mask = torch.nn.functional.interpolate(
                        spatial_mask.unsqueeze(0).unsqueeze(0).to(attention_output.dtype),
                        size=(seq_len,), mode='linear', align_corners=False
                    )[0, 0]
            
            # Apply spatial mask: blend injection (attention_output) and original  
            spatial_mask = spatial_mask.unsqueeze(-1)  # Add feature dimension
            
            # Broadcast mask for batch size
            if attention_output.shape[0] > 1:
                spatial_mask = spatial_mask.unsqueeze(0).expand(attention_output.shape[0], -1, -1)
            else:
                spatial_mask = spatial_mask.unsqueeze(0)
                
            
            # Blend: mask=1 uses injection, mask=0 uses original
            blended_output = (
                spatial_mask * attention_output + 
                (torch.ones_like(spatial_mask) - spatial_mask) * original_attention_output
            )
            
            return blended_output
        except Exception as e:
            logger.error(f"Error during spatial masking for block {self.block_id}: {e}", exc_info=True)
            # Fallback to the injected (but not masked) output
            return attention_output
    
    def _apply_self_attention_manipulation(self, attention_scores: torch.Tensor, 
                                         config: SelfAttentionConfig) -> torch.Tensor:
        """
        Apply self-attention manipulation to attention scores.
        
        Args:
            attention_scores: Attention scores tensor (batch_heads, height*width, height*width)
            config: Self-attention configuration
            
        Returns:
            Modified attention scores
        """
        try:
            batch_heads, seq_len, seq_len_2 = attention_scores.shape
            assert seq_len == seq_len_2, "Self-attention should have square attention matrix"
            
            # If we have spatial regions defined, apply region-based manipulation
            if config.source_region is not None or config.target_region is not None:
                # Prepare spatial masks
                source_mask = config.source_region
                target_mask = config.target_region
                
                if source_mask is not None:
                    # Flatten spatial mask to match sequence length
                    if source_mask.numel() != seq_len:
                        # Resize mask to match attention resolution
                        spatial_size = int(seq_len ** 0.5)  # Assume square spatial arrangement
                        source_mask = torch.nn.functional.interpolate(
                            source_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(spatial_size, spatial_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().flatten()
                    source_mask = source_mask.to(attention_scores.device)
                
                if target_mask is not None:
                    if target_mask.numel() != seq_len:
                        spatial_size = int(seq_len ** 0.5)
                        target_mask = torch.nn.functional.interpolate(
                            target_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(spatial_size, spatial_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().flatten()
                    target_mask = target_mask.to(attention_scores.device)
                
                # Apply manipulation based on interaction type
                if config.interaction_type == "enhance":
                    # Enhance attention between specified regions
                    if source_mask is not None and target_mask is not None:
                        # Source pixels attend more to target pixels
                        source_indices = torch.nonzero(source_mask > 0.5, as_tuple=True)[0]
                        target_indices = torch.nonzero(target_mask > 0.5, as_tuple=True)[0]
                        
                        if len(source_indices) > 0 and len(target_indices) > 0:
                            attention_scores[:, source_indices[:, None], target_indices] *= config.attention_scale
                
                elif config.interaction_type == "suppress":
                    # Suppress attention between specified regions
                    if source_mask is not None and target_mask is not None:
                        source_indices = torch.nonzero(source_mask > 0.5, as_tuple=True)[0]
                        target_indices = torch.nonzero(target_mask > 0.5, as_tuple=True)[0]
                        
                        if len(source_indices) > 0 and len(target_indices) > 0:
                            attention_scores[:, source_indices[:, None], target_indices] *= config.attention_scale
                
                elif config.interaction_type == "redirect":
                    # Redirect attention from one region to another
                    if source_mask is not None and target_mask is not None:
                        source_indices = torch.nonzero(source_mask > 0.5, as_tuple=True)[0]
                        target_indices = torch.nonzero(target_mask > 0.5, as_tuple=True)[0]
                        
                        if len(source_indices) > 0 and len(target_indices) > 0:
                            # Reduce attention to non-target regions from source
                            for s_idx in source_indices:
                                attention_scores[:, s_idx] *= 0.5  # Reduce overall attention
                                # Boost attention to target region
                                attention_scores[:, s_idx, target_indices] *= config.attention_scale * 2.0
            
            else:
                # Global self-attention manipulation
                logger.debug(f"Applying global {config.interaction_type} with scale {config.attention_scale}")
                if config.interaction_type == "enhance":
                    # Enhance global coherence by amplifying all self-attention scores
                    attention_scores = attention_scores * config.attention_scale
                elif config.interaction_type == "suppress":
                    # Suppress global coherence by reducing all self-attention scores
                    attention_scores = attention_scores * config.attention_scale  # Scale should be < 1.0 for suppression
                elif config.interaction_type == "sharpen":
                    # Sharpen attention patterns by raising to a power
                    attention_scores = torch.pow(torch.clamp(attention_scores, min=1e-8), config.attention_scale)
                elif config.interaction_type == "smooth":
                    # Smooth attention patterns by adding uniform noise
                    uniform_attention = torch.ones_like(attention_scores) / seq_len
                    attention_scores = (1.0 - config.attention_scale) * attention_scores + config.attention_scale * uniform_attention
                
            return attention_scores
            
        except Exception as e:
            logger.error(f"Error in self-attention manipulation: {e}", exc_info=True)
            # Return original scores on error
            return attention_scores

    def set_current_sigma(self, sigma: float):
        """Set current sigma for injection timing control."""
        self._current_sigma = sigma


class UNetSigmaHook:
    """
    Hook to capture sigma values during sampling for injection timing.
    """
    
    def __init__(self, patcher: 'UNetPatcher', scheduler: SchedulerMixin):
        self.patcher = patcher
        self.scheduler = scheduler
        self.current_sigma = None
    
    def __call__(self, module, args, kwargs=None):
        """Hook called before UNet forward pass."""
        try:
            if not hasattr(self.scheduler, "sigmas"):
                return
            
            timestep = args[1]
            
            if not isinstance(timestep, torch.Tensor):
                # Timestep is not a tensor, cannot proceed
                return
                
            # Find the index of the current timestep in the scheduler's timesteps
            # Squeeze to handle cases where timesteps might have extra dimensions
            condition = self.scheduler.timesteps == timestep.squeeze()
            
            # Check if the condition tensor is on a different device and move it
            if condition.device != self.scheduler.timesteps.device:
                condition = condition.to(self.scheduler.timesteps.device)

            matching_indices = torch.where(condition)[0]
            
            if matching_indices.numel() > 0:
                idx = matching_indices[0]
                sigma = self.scheduler.sigmas[idx]
                self.set_sigma(sigma.item())
        except Exception as e:
            logger.error(f"Error in UNetSigmaHook: {e}", exc_info=True)
            # Do not re-raise, as this could crash the forward pass
            pass

    def set_sigma(self, sigma: float):
        """Set current sigma and update all injection processors."""
        self.current_sigma = sigma
        logger.debug(f"UNetSigmaHook: Current sigma set to {sigma:.4f}")
        for processor in self.patcher._injection_processors.values():
            processor.set_current_sigma(sigma)


class UNetPatcher(BaseModelPatcher):
    """
    Patches UNet models to inject prompts at specific attention blocks.
    
    This implementation follows the ComfyUI approach of patching cross-attention
    mechanisms with proper sigma-based timing control.
    """
    
    def __init__(self, unet: UNet2DConditionModel, scheduler: SchedulerMixin):
        super().__init__()
        self.block_mapper = UNetBlockMapper(unet)
        self.injection_configs: Dict[str, Dict[str, Any]] = {}
        self.attention_map_configs: Dict[str, List[AttentionMapConfig]] = {}
        self.self_attention_configs: Dict[str, List[SelfAttentionConfig]] = {}
        self._original_processors: Dict[str, Any] = {}
        self._injection_processors: Dict[str, PromptInjectionProcessor] = {}
        self._sigma_hook = UNetSigmaHook(self, scheduler)
        self._hook_handle = None
    
    def add_injection(self, block: Union[str, BlockIdentifier], 
                     conditioning: torch.Tensor,
                     weight: float = 1.0,
                     sigma_start: float = 1.0,  # Changed default to match ComfyUI
                     sigma_end: float = 0.0,    # Changed default to match ComfyUI
                     spatial_mask: Optional[torch.Tensor] = None):
        """
        Add a prompt injection for a specific block or all blocks.
        
        Args:
            block: Block identifier (string like "input:4", "all", or BlockIdentifier)
            conditioning: Conditioning tensor to inject
            weight: Injection weight (1.0 = normal, >1.0 = amplified, <1.0 = weakened)
            sigma_start: Start sigma for injection window (higher noise)
            sigma_end: End sigma for injection window (lower noise)
            spatial_mask: Optional spatial mask for regional control
        """
        # Handle "all" keyword
        if isinstance(block, str) and block.lower() == "all":
            all_blocks = self.block_mapper.get_all_block_identifiers()
            for block_id in all_blocks:
                self.injection_configs[block_id] = {
                    'block': BlockIdentifier.from_string(block_id),
                    'block_id': block_id,
                    'conditioning': conditioning,
                    'weight': weight,
                    'sigma_start': sigma_start,
                    'sigma_end': sigma_end,
                    'spatial_mask': spatial_mask
                }
            return
        
        # Handle single block
        if isinstance(block, str):
            block_id = block
            block = BlockIdentifier.from_string(block)
        else:
            block_id = f"{block.block_type}:{block.block_index}"
        
        if not self.block_mapper.is_valid_block(block):
            raise ValueError(f"Invalid block: {block}")
        
        self.injection_configs[block_id] = {
            'block': block,
            'block_id': block_id,
            'conditioning': conditioning,
            'weight': weight,
            'sigma_start': sigma_start,
            'sigma_end': sigma_end,
            'spatial_mask': spatial_mask
        }
    
    def add_attention_manipulation(self, block: Union[str, BlockIdentifier],
                                 target_token_indices: List[int],
                                 attention_scale: float = 1.0,
                                 sigma_start: float = 1.0,
                                 sigma_end: float = 0.0,
                                 spatial_mask: Optional[torch.Tensor] = None):
        """
        Add an attention map manipulation for a specific block or all blocks.
        """
        config = AttentionMapConfig(
            target_token_indices=target_token_indices,
            attention_scale=attention_scale,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            spatial_mask=spatial_mask
        )

        if isinstance(block, str) and block.lower() == "all":
            all_blocks = self.block_mapper.get_all_block_identifiers()
            for block_id in all_blocks:
                if block_id not in self.attention_map_configs:
                    self.attention_map_configs[block_id] = []
                self.attention_map_configs[block_id].append(config)
        else:
            block_id = str(block) if isinstance(block, BlockIdentifier) else block
            if not self.block_mapper.is_valid_block(BlockIdentifier.from_string(block_id)):
                raise ValueError(f"Invalid block: {block_id}")
            
            if block_id not in self.attention_map_configs:
                self.attention_map_configs[block_id] = []
            self.attention_map_configs[block_id].append(config)

    def add_self_attention_manipulation(self, block: Union[str, BlockIdentifier],
                                      source_region: Optional[torch.Tensor] = None,
                                      target_region: Optional[torch.Tensor] = None,
                                      attention_scale: float = 1.0,
                                      interaction_type: str = "enhance",
                                      sigma_start: float = 1.0,
                                      sigma_end: float = 0.0):
        """
        Add self-attention manipulation for controlling image-to-image attention.
        
        Args:
            block: Block identifier or "all"
            source_region: Spatial mask for pixels whose attention to modify (FROM)
            target_region: Spatial mask for pixels to attend to (TO)
            attention_scale: Scaling factor for attention weights
            interaction_type: "enhance", "suppress", or "redirect"
            sigma_start: Start sigma for manipulation window
            sigma_end: End sigma for manipulation window
        """
        config = SelfAttentionConfig(
            source_region=source_region,
            target_region=target_region,
            attention_scale=attention_scale,
            interaction_type=interaction_type,
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

        if isinstance(block, str) and block.lower() == "all":
            all_blocks = self.block_mapper.get_all_block_identifiers()
            for block_id in all_blocks:
                if block_id not in self.self_attention_configs:
                    self.self_attention_configs[block_id] = []
                self.self_attention_configs[block_id].append(config)
        else:
            block_id = str(block) if isinstance(block, BlockIdentifier) else block
            if not self.block_mapper.is_valid_block(BlockIdentifier.from_string(block_id)):
                raise ValueError(f"Invalid block: {block_id}")
            
            if block_id not in self.self_attention_configs:
                self.self_attention_configs[block_id] = []
            self.self_attention_configs[block_id].append(config)

    def clear_injections(self):
        """Clear all prompt injections."""
        self.injection_configs.clear()
        self.attention_map_configs.clear()
        self.self_attention_configs.clear()
        self._injection_processors.clear()
    
    def apply_patches(self, unet: UNet2DConditionModel) -> UNet2DConditionModel:
        """
        Apply patches to the UNet model.
        
        Args:
            unet: UNet model to patch
            
        Returns:
            Patched UNet model
        """
        if not self.injection_configs and not self.attention_map_configs and not self.self_attention_configs:
            return unet
        
        # Store original processors for restoration
        self._original_processors.clear()
        self._injection_processors.clear()
        
        # Register the pre-forward hook to capture sigma
        if self._hook_handle is None:
            self._hook_handle = unet.register_forward_pre_hook(self._sigma_hook, with_kwargs=False)
        
        all_block_ids = set(self.injection_configs.keys()) | set(self.attention_map_configs.keys()) | set(self.self_attention_configs.keys())
        
        for block_id in all_block_ids:
            try:
                # Find the diffusers path for this block
                config = self.injection_configs.get(block_id, {'block': BlockIdentifier.from_string(block_id)})
                block = config['block']
                diffusers_path = self.block_mapper.map_to_diffusers_path(block)
                
                # Get configs for this specific block
                inj_config = self.injection_configs.get(block_id)
                attn_maps = self.attention_map_configs.get(block_id)
                self_attn_configs = self.self_attention_configs.get(block_id)
                
                # Find cross-attention modules if we have cross-attention configs
                if inj_config or attn_maps:
                    attention_modules = self._find_cross_attention_modules(unet, diffusers_path)
                    
                    for module_path, attn_module in attention_modules:
                        # Store original processor
                        original_processor = getattr(attn_module, 'processor', AttnProcessor2_0())
                        self._original_processors[module_path] = original_processor
                        
                        # Create custom processor for cross-attention
                        custom_processor = PromptInjectionProcessor(
                            original_processor=original_processor,
                            block_id=block_id,
                            conditioning=inj_config.get('conditioning') if inj_config else None,
                            weight=inj_config.get('weight', 1.0) if inj_config else 1.0,
                            sigma_start=inj_config.get('sigma_start', 0.0) if inj_config else 0.0,
                            sigma_end=inj_config.get('sigma_end', 1.0) if inj_config else 1.0,
                            spatial_mask=inj_config.get('spatial_mask') if inj_config else None,
                            attention_maps=attn_maps,
                            module_path=module_path
                        )
                        
                        # Store processor for sigma updates
                        self._injection_processors[module_path] = custom_processor
                        
                        # Set the custom processor
                        attn_module.set_processor(custom_processor)
                
                # Find self-attention modules if we have self-attention configs
                if self_attn_configs:
                    self_attention_modules = self._find_self_attention_modules(unet, diffusers_path)
                    logger.warning(f"BLOCK {block_id}: Found {len(self_attention_modules)} self-attention modules at {diffusers_path}")
                    for path, _ in self_attention_modules:
                        logger.warning(f"  - Self-attention module: {path}")
                    
                    for module_path, attn_module in self_attention_modules:
                        # Store original processor
                        original_processor = getattr(attn_module, 'processor', AttnProcessor2_0())
                        self._original_processors[module_path] = original_processor
                        
                        # Create custom processor for self-attention
                        custom_processor = PromptInjectionProcessor(
                            original_processor=original_processor,
                            block_id=block_id,
                            conditioning=None,  # No conditioning for self-attention
                            weight=1.0,
                            sigma_start=0.0,
                            sigma_end=1.0,
                            spatial_mask=None,
                            attention_maps=[],  # No token-level attention for self-attention
                            self_attention_configs=self_attn_configs,
                            module_path=module_path
                        )
                        
                        # Store processor for sigma updates
                        self._injection_processors[module_path] = custom_processor
                        
                        # Set the custom processor
                        attn_module.set_processor(custom_processor)
                        
            except Exception as e:
                print(f"Warning: Failed to patch block {block_id}: {e}")
        
        self._hooked_unet = unet
        self.is_patched = True
        return unet
    
    def remove_patches(self, unet: UNet2DConditionModel) -> UNet2DConditionModel:
        """
        Remove patches from the UNet model.
        
        Args:
            unet: Patched UNet model
            
        Returns:
            Restored UNet model
        """
        if not self.is_patched:
            return unet
        
        # Remove the forward hook
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        
        # Restore original processors
        for module_path, original_processor in self._original_processors.items():
            try:
                # Navigate to the module and restore processor
                module = unet
                for attr in module_path.split('.'):
                    module = getattr(module, attr)
                module.set_processor(original_processor)
            except Exception as e:
                print(f"Warning: Failed to restore processor for {module_path}: {e}")
        
        self._original_processors.clear()
        self.is_patched = False
        return unet
    
    def _find_cross_attention_modules(self, unet: UNet2DConditionModel, 
                                     diffusers_path: str) -> List[Tuple[str, Attention]]:
        """
        Find cross-attention modules to patch in a specific block.
        
        Args:
            unet: UNet model to search
            diffusers_path: Diffusers path to the block (e.g., "down_blocks.0")
            
        Returns:
            List of (module_path, attention_module) tuples
        """
        attention_modules = []
        
        try:
            # Navigate to the target block
            target_module = unet
            for attr in diffusers_path.split('.'):
                if attr.isdigit():
                    target_module = target_module[int(attr)]
                else:
                    target_module = getattr(target_module, attr)
            
            # Recursively find cross-attention modules
            self._collect_cross_attention_modules(target_module, diffusers_path, attention_modules)
            
        except Exception as e:
            print(f"Warning: Could not access {diffusers_path}: {e}")
            
        return attention_modules
    
    def _collect_attention_modules(self, module, path: str, 
                                   results: List[Tuple[str, Attention]], 
                                   attention_type: str = "cross"):
        """
        Recursively collect attention modules from a block.
        
        Args:
            module: Module to search
            path: Current path in the module hierarchy
            results: List to append found modules
            attention_type: "cross" for attn2, "self" for attn1, "both" for both
        """
        # Some modules might not have children (e.g., final layers)
        if not hasattr(module, 'named_children'):
            return
            
        for name, child in module.named_children():
            child_path = f"{path}.{name}"
            
            # Check attention type and collect accordingly
            if hasattr(child, 'processor'):
                should_collect = False
                
                if attention_type in ["cross", "both"] and name == 'attn2':
                    # Cross-attention (text -> image)
                    should_collect = True
                elif attention_type in ["cross", "both"] and 'cross' in name.lower():
                    # Alternative cross-attention naming
                    should_collect = True
                elif attention_type in ["self", "both"] and name == 'attn1':
                    # Self-attention (image -> image)
                    should_collect = True
                elif attention_type in ["self", "both"] and 'self' in name.lower():
                    # Alternative self-attention naming
                    should_collect = True
                
                if should_collect:
                    results.append((child_path, child))
            else:
                # Recurse into child modules
                self._collect_attention_modules(child, child_path, results, attention_type)
    
    def _collect_cross_attention_modules(self, module, path: str, 
                                       results: List[Tuple[str, Attention]]):
        """
        Recursively collect cross-attention modules from a block.
        """
        return self._collect_attention_modules(module, path, results, "cross")
    
    def _find_self_attention_modules(self, unet: UNet2DConditionModel, 
                                   diffusers_path: str) -> List[Tuple[str, Attention]]:
        """
        Find self-attention modules to patch in a specific block.
        
        Args:
            unet: UNet model to search
            diffusers_path: Diffusers path to the block (e.g., "down_blocks.0")
            
        Returns:
            List of (module_path, attention_module) tuples
        """
        attention_modules = []
        
        try:
            # Navigate to the target block
            target_module = unet
            for attr in diffusers_path.split('.'):
                if attr.isdigit():
                    target_module = target_module[int(attr)]
                else:
                    target_module = getattr(target_module, attr)
            
            # Recursively find self-attention modules
            self._collect_attention_modules(target_module, diffusers_path, attention_modules, "self")
            
        except Exception as e:
            print(f"Warning: Could not access {diffusers_path} for self-attention: {e}")
            
        return attention_modules
    
    def set_current_sigma(self, sigma: float):
        """
        Set current sigma for all injection processors.
        This should be called during sampling to enable sigma-based timing control.
        """
        for processor in self._injection_processors.values():
            processor.set_current_sigma(sigma)
