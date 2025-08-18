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
from .base import BaseModelPatcher, BlockIdentifier


class UNetBlockMapper:
    """
    Maps between ComfyUI-style block identifiers and diffusers UNet structure.
    """
    
    # Block mappings for different model architectures
    SDXL_BLOCKS = {
        'input': [4, 5, 7, 8],
        'middle': [0], 
        'output': [0, 1, 2, 3, 4, 5]
    }
    
    SD15_BLOCKS = {
        'input': [3],  # SD 1.5 has input blocks but fewer cross-attention layers
        'middle': [0, 1, 2],
        'output': [0, 1, 2, 3]  # SD 1.5 has output blocks 0-3 with cross-attention
    }
    
    def __init__(self, model_type: str = "sdxl"):
        """
        Initialize block mapper.
        
        Args:
            model_type: Either "sdxl" or "sd15"
        """
        self.model_type = model_type.lower()
        if self.model_type == "sdxl":
            self.blocks = self.SDXL_BLOCKS
        elif self.model_type == "sd15":
            self.blocks = self.SD15_BLOCKS
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
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
            raise ValueError(f"Invalid block for {self.model_type}: {block}")
        
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
    at specific blocks and timesteps.
    """
    
    def __init__(self, original_processor, block_id: str, conditioning: torch.Tensor, 
                 weight: float = 1.0, sigma_start: float = 0.0, sigma_end: float = 1.0):
        self.original_processor = original_processor
        self.block_id = block_id  # e.g., "middle:0", "output:1"
        self.conditioning = conditioning
        self.weight = weight
        self.sigma_start = sigma_start  # Higher values (more noise)
        self.sigma_end = sigma_end      # Lower values (less noise)
        
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
        
        # DEBUG: Log that this processor is being called
        print(f"PromptInjectionProcessor called for {self.block_id}")
        
        # Get current sigma from timestep if available
        current_sigma = getattr(self, '_current_sigma', None)
        
        # Check if we should inject at this timestep/sigma
        should_inject = self._should_inject(current_sigma)
        
        if (should_inject and 
            encoder_hidden_states is not None and 
            self.conditioning is not None):
            
            print(f" INJECTING at block {self.block_id} with weight {self.weight}")
            # Apply injection similar to ComfyUI approach
            original_shape = encoder_hidden_states.shape
            encoder_hidden_states = self._apply_injection(encoder_hidden_states)
            print(f"   Original shape: {original_shape}, New shape: {encoder_hidden_states.shape}")
        else:
            print(f"NOT injecting at {self.block_id} (should_inject: {should_inject}, has_conditioning: {self.conditioning is not None})")
        
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
        """Apply the conditioning injection similar to ComfyUI."""
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
        
        # Instead of replacing entirely, let's try following ComfyUI's approach
        # ComfyUI replaces only the conditional part (second batch) for CFG
        if batch_size == 2:
            # Replace only the conditional part (batch 1), keep unconditional (batch 0) 
            result = encoder_hidden_states.clone()
            result[1] = injected_conditioning[0] * self.weight
        else:
            # Single batch - replace entirely
            result = injected_conditioning * self.weight
        
        return result
    
    def set_current_sigma(self, sigma: float):
        """Set current sigma for injection timing control."""
        self._current_sigma = sigma


class UNetSigmaHook:
    """
    Hook to capture sigma values during sampling for injection timing.
    """
    
    def __init__(self, patcher: 'UNetPatcher'):
        self.patcher = patcher
        self.current_sigma = None
    
    def __call__(self, module, args, output):
        """Hook called during UNet forward pass."""
        # Try to extract sigma from the sampling context
        # This is a simplified approach - in practice we'd need deeper integration
        return output
    
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
    
    def __init__(self, model_type: str = "sdxl"):
        super().__init__()
        self.block_mapper = UNetBlockMapper(model_type)
        self.injection_configs: Dict[str, Dict[str, Any]] = {}
        self._original_processors: Dict[str, Any] = {}
        self._injection_processors: Dict[str, PromptInjectionProcessor] = {}
        self._sigma_hook = UNetSigmaHook(self)
        self._hooked_unet = None
    
    def add_injection(self, block: Union[str, BlockIdentifier], 
                     conditioning: torch.Tensor,
                     weight: float = 1.0,
                     sigma_start: float = 1.0,  # Changed default to match ComfyUI
                     sigma_end: float = 0.0):   # Changed default to match ComfyUI
        """
        Add a prompt injection for a specific block or all blocks.
        
        Args:
            block: Block identifier (string like "input:4", "all", or BlockIdentifier)
            conditioning: Conditioning tensor to inject
            weight: Injection weight
            sigma_start: Start sigma for injection window (higher noise)
            sigma_end: End sigma for injection window (lower noise)
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
                    'sigma_end': sigma_end
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
            'sigma_end': sigma_end
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
                        sigma_end=config['sigma_end']
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
                print(f"   Found cross-attention: {child_path}")
                results.append((child_path, child))
            elif hasattr(child, 'processor') and 'cross' in name.lower():
                print(f"   Found cross-attention: {child_path}")
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
