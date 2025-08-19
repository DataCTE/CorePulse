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
        if not self.is_valid_block(block):
            raise ValueError(f"Invalid block for {self.unet_config.model_type}: {block}")
        
        if block.block_type == "input":
            return f"down_blocks.{block.block_index}"
        elif block.block_type == "middle": 
            # Middle block is just "mid_block", not indexed
            return "mid_block"
        elif block.block_type == "output":
            return f"up_blocks.{block.block_index}"
        else:
            raise ValueError(f"Unknown block type: {block.block_type}")


class PromptInjectionProcessor:
    """
    Custom attention processor for injecting prompts at specific blocks.
    
    This is similar to ComfyUI's prompt injection but adapted for diffusers.
    It hooks into the cross-attention mechanism to replace conditioning
    at specific blocks and timesteps with optional spatial control.
    """
    
    def __init__(self, original_processor, block_id: str, conditioning: torch.Tensor, 
                 weight: float = 1.0, sigma_start: float = 0.0, sigma_end: float = 1.0,
                 spatial_mask: Optional[torch.Tensor] = None):
        self.original_processor = original_processor
        self.block_id = block_id  # e.g., "middle:0", "output:1"
        self.conditioning = conditioning
        self.weight = weight
        self.sigma_start = sigma_start  # Higher values (more noise)
        self.sigma_end = sigma_end      # Lower values (less noise)
        self.spatial_mask = spatial_mask  # Optional spatial mask for regional control
        self.projection_layer = None # For handling mismatched dimensions
        
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
                attention_output = self.original_processor(
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
        else:
            pass
            # print(f"NOT injecting at {self.block_id} (should_inject: {should_inject}, has_conditioning: {self.conditioning is not None})")
        
        # Call original processor
        return self.original_processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
        )
    
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
    
    def _apply_spatial_masking(self, attention_output: torch.Tensor, 
                              original_hidden_states: torch.Tensor,
                              attn: Attention, encoder_hidden_states: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None,
                              timestep: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Apply spatial masking to attention output by blending with original."""
        if self.spatial_mask is None:
            return attention_output
        
        # Run original attention without injection to get baseline
        original_attention_output = self.original_processor(
            attn, original_hidden_states, encoder_hidden_states, 
            attention_mask, timestep, **kwargs
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
        if not hasattr(self.scheduler, "sigmas"):
            return
        
        timestep = args[1]
        
        if not isinstance(timestep, torch.Tensor):
             return
            
        sigma = self.scheduler.sigmas[0] # Fallback
        if timestep.ndim > 0 and len(timestep) > 0:
            timestep_index = (self.scheduler.timesteps == timestep[0]).nonzero()
            if timestep_index.numel() > 0:
                sigma = self.scheduler.sigmas[timestep_index.item()]

        self.set_sigma(sigma.item())
    
    def set_sigma(self, sigma: float):
        """Set current sigma and update all injection processors."""
        self.current_sigma = sigma
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
    
    def clear_injections(self):
        """Clear all prompt injections."""
        self.injection_configs.clear()
        self._injection_processors.clear()
    
    def apply_patches(self, unet: UNet2DConditionModel) -> UNet2DConditionModel:
        """
        Apply patches to the UNet model.
        
        Args:
            unet: UNet model to patch
            
        Returns:
            Patched UNet model
        """
        if not self.injection_configs:
            return unet
        
        # Store original processors for restoration
        self._original_processors.clear()
        self._injection_processors.clear()
        
        # Register the pre-forward hook to capture sigma
        if self._hook_handle is None:
            self._hook_handle = unet.register_forward_pre_hook(self._sigma_hook, with_kwargs=False)
        
        for block_id, config in self.injection_configs.items():
            try:
                # Find the diffusers path for this block
                block = config['block']
                diffusers_path = self.block_mapper.map_to_diffusers_path(block)
                
                # Find cross-attention modules in this block
                attention_modules = self._find_cross_attention_modules(unet, diffusers_path)
                
                for module_path, attn_module in attention_modules:
                    # Store original processor
                    original_processor = getattr(attn_module, 'processor', AttnProcessor2_0())
                    self._original_processors[module_path] = original_processor
                    
                    # Create custom processor
                    custom_processor = PromptInjectionProcessor(
                        original_processor=original_processor,
                        block_id=block_id,
                        conditioning=config['conditioning'],
                        weight=config['weight'],
                        sigma_start=config['sigma_start'],
                        sigma_end=config['sigma_end'],
                        spatial_mask=config.get('spatial_mask')
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
    
    def _collect_cross_attention_modules(self, module, path: str, 
                                       results: List[Tuple[str, Attention]]):
        """
        Recursively collect cross-attention modules from a block.
        """
        for name, child in module.named_children():
            child_path = f"{path}.{name}"
            
            # Check if this is a cross-attention module
            # In diffusers, cross-attention is usually named 'attn2'
            if hasattr(child, 'processor') and name == 'attn2':
                
                results.append((child_path, child))
            elif hasattr(child, 'processor') and 'cross' in name.lower():
               
                results.append((child_path, child))
            else:
                # Recurse into child modules
                self._collect_cross_attention_modules(child, child_path, results)
    
    def set_current_sigma(self, sigma: float):
        """
        Set current sigma for all injection processors.
        This should be called during sampling to enable sigma-based timing control.
        """
        for processor in self._injection_processors.values():
            processor.set_current_sigma(sigma)
