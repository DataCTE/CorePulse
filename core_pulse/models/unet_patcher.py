"""
UNet patching utilities for prompt injection.
"""

import torch
from typing import Dict, Optional, Union, List, Tuple, Any
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
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
        'input': [3, 4, 5, 6, 7, 8, 9, 10, 11],
        'middle': [0, 1, 2],
        'output': [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
            return f"mid_block.{block.block_index}"
        elif block.block_type == "output":
            return f"up_blocks.{block.block_index}"
        else:
            raise ValueError(f"Unknown block type: {block.block_type}")


class PromptInjectionProcessor:
    """
    Custom attention processor for injecting prompts at specific blocks.
    """
    
    def __init__(self, original_processor, injection_data: Dict[str, Any]):
        self.original_processor = original_processor
        self.injection_data = injection_data
        self.block_id = injection_data.get('block_id')
        self.conditioning = injection_data.get('conditioning')
        self.weight = injection_data.get('weight', 1.0)
        self.sigma_start = injection_data.get('sigma_start', 0.0)
        self.sigma_end = injection_data.get('sigma_end', 1.0)
    
    def __call__(self, attn: Attention, hidden_states: torch.Tensor, 
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 **cross_attention_kwargs) -> torch.Tensor:
        
        # Check if we should inject at this timestep
        # Note: In diffusers, we need to get timestep info differently
        # This is a simplified version - in practice you'd need to pass timestep context
        
        if (encoder_hidden_states is not None and 
            self.conditioning is not None and 
            self._should_inject()):
            
            # Replace the encoder hidden states with our conditioning
            batch_size = encoder_hidden_states.shape[0]
            if isinstance(self.conditioning, torch.Tensor):
                # Ensure conditioning matches batch size
                if self.conditioning.shape[0] == 1 and batch_size > 1:
                    injected_conditioning = self.conditioning.repeat(batch_size, 1, 1)
                else:
                    injected_conditioning = self.conditioning
                
                # Apply weight and blend with original if needed
                encoder_hidden_states = injected_conditioning * self.weight
        
        # Call original processor
        return self.original_processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, 
            **cross_attention_kwargs
        )
    
    def _should_inject(self) -> bool:
        """Determine if we should inject at current timestep."""
        # Simplified check - in practice this would use actual timestep info
        return True


class UNetPatcher(BaseModelPatcher):
    """
    Patches UNet models to inject prompts at specific attention blocks.
    """
    
    def __init__(self, model_type: str = "sdxl"):
        super().__init__()
        self.block_mapper = UNetBlockMapper(model_type)
        self.injection_configs: Dict[BlockIdentifier, Dict[str, Any]] = {}
        self._original_processors: Dict[str, Any] = {}
    
    def add_injection(self, block: Union[str, BlockIdentifier], 
                     conditioning: torch.Tensor,
                     weight: float = 1.0,
                     sigma_start: float = 0.0,
                     sigma_end: float = 1.0):
        """
        Add a prompt injection for a specific block.
        
        Args:
            block: Block identifier (string like "input:4" or BlockIdentifier)
            conditioning: Conditioning tensor to inject
            weight: Injection weight
            sigma_start: Start sigma for injection window  
            sigma_end: End sigma for injection window
        """
        if isinstance(block, str):
            block = BlockIdentifier.from_string(block)
        
        if not self.block_mapper.is_valid_block(block):
            raise ValueError(f"Invalid block: {block}")
        
        self.injection_configs[block] = {
            'block_id': block,
            'conditioning': conditioning,
            'weight': weight,
            'sigma_start': sigma_start,
            'sigma_end': sigma_end
        }
    
    def clear_injections(self):
        """Clear all prompt injections."""
        self.injection_configs.clear()
    
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
        
        for block_id, config in self.injection_configs.items():
            try:
                # Find the attention modules to patch
                attention_modules = self._find_attention_modules(unet, block_id)
                
                for module_path, attn_module in attention_modules:
                    # Store original processor
                    if hasattr(attn_module, 'processor'):
                        self._original_processors[module_path] = attn_module.processor
                        
                        # Create and set custom processor
                        custom_processor = PromptInjectionProcessor(
                            attn_module.processor, config
                        )
                        attn_module.set_processor(custom_processor)
                        
            except Exception as e:
                print(f"Warning: Failed to patch block {block_id}: {e}")
        
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
    
    def _find_attention_modules(self, unet: UNet2DConditionModel, 
                              block_id: BlockIdentifier) -> List[Tuple[str, Attention]]:
        """
        Find attention modules to patch for a given block.
        
        Args:
            unet: UNet model to search
            block_id: Block identifier
            
        Returns:
            List of (module_path, attention_module) tuples
        """
        attention_modules = []
        
        try:
            if block_id.block_type == "input":
                if hasattr(unet, 'down_blocks') and block_id.block_index < len(unet.down_blocks):
                    block = unet.down_blocks[block_id.block_index]
                    modules = self._find_cross_attention_in_block(
                        block, f"down_blocks.{block_id.block_index}"
                    )
                    attention_modules.extend(modules)
            
            elif block_id.block_type == "middle":
                if hasattr(unet, 'mid_block'):
                    modules = self._find_cross_attention_in_block(
                        unet.mid_block, "mid_block"
                    )
                    attention_modules.extend(modules)
            
            elif block_id.block_type == "output":
                if hasattr(unet, 'up_blocks') and block_id.block_index < len(unet.up_blocks):
                    block = unet.up_blocks[block_id.block_index] 
                    modules = self._find_cross_attention_in_block(
                        block, f"up_blocks.{block_id.block_index}"
                    )
                    attention_modules.extend(modules)
                    
        except Exception as e:
            print(f"Error finding attention modules for {block_id}: {e}")
        
        return attention_modules
    
    def _find_cross_attention_in_block(self, block, base_path: str) -> List[Tuple[str, Attention]]:
        """
        Recursively find cross-attention modules in a block.
        
        Args:
            block: Block module to search
            base_path: Base path for this block
            
        Returns:
            List of (module_path, attention_module) tuples
        """
        attention_modules = []
        
        for name, module in block.named_modules():
            if isinstance(module, Attention):
                # Check if this is cross-attention (has encoder_hidden_states)
                # Cross-attention typically has different dimensions for key/value vs query
                module_path = f"{base_path}.{name}" if name else base_path
                attention_modules.append((module_path, module))
        
        return attention_modules
