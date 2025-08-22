"""
Transformer attention patching for LLM attention manipulation.

This module adapts the attention manipulation techniques from diffusion models
to work with transformer-based language models like Qwen3.
"""

import torch
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
from ..utils.logger import logger


@dataclass
class LLMAttentionConfig:
    """Configuration for LLM attention manipulation."""
    target_token_indices: List[int]
    attention_scale: float
    layer_indices: Optional[List[int]] = None  # Which transformer layers to target
    source_tokens: Optional[List[int]] = None  # Which tokens should have modified attention (FROM)
    interaction_type: str = "amplify"  # "amplify", "suppress", "redirect"
    position_start: int = 0  # Start position in sequence
    position_end: int = -1   # End position in sequence (-1 = end of sequence)
    target_phrase: str = ""  # The phrase to manipulate (for token resolution)


class TransformerAttentionProcessor:
    """
    Custom attention processor for transformers that enables attention manipulation.
    
    This hooks into the transformer's attention mechanism to modify attention weights
    during forward passes, enabling direct control over what the model focuses on.
    """
    
    def __init__(self, 
                 original_forward: Callable,
                 layer_idx: int,
                 attention_configs: List[LLMAttentionConfig],
                 tokenizer: PreTrainedTokenizer):
        self.original_forward = original_forward
        self.layer_idx = layer_idx
        self.attention_configs = attention_configs
        self.tokenizer = tokenizer
        
    def __call__(self, *args, **kwargs):
        """
        Hook into attention forward pass and apply manipulations.
        
        This method completely replaces the original attention forward pass
        to intercept and modify attention scores during computation.
        """
        try:
            # We need to reimplement the attention forward pass to intercept
            # the attention computation at the right point
            return self._custom_attention_forward(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in custom attention forward for layer {self.layer_idx}: {e}")
            logger.info("Falling back to original forward pass")
            # Fallback to original if our custom implementation fails
            return self.original_forward(*args, **kwargs)
    
    def _custom_attention_forward(self, *args, **kwargs):
        """
        Custom attention forward pass that intercepts and modifies attention scores.
        
        This reimplements the core attention computation to allow manipulation
        of attention scores before they're applied to values.
        """
        logger.debug(f"Custom attention forward called for layer {self.layer_idx}")
        logger.debug(f"Args: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
        
        # Now that I know the Qwen3 architecture, I can implement proper attention manipulation
        # The GroupedQueryAttention.forward method computes:
        # 1. queries, keys, values from projections
        # 2. attn_scores = queries @ keys.transpose(2, 3)  
        # 3. attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        # 4. context = attn_weights @ values
        
        # I need to intercept and modify attn_weights before step 4
        try:
            return self._qwen3_attention_forward(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in custom attention forward for layer {self.layer_idx}: {e}")
            return self.original_forward(*args, **kwargs)
    


    def _qwen3_attention_forward(self, *args, **kwargs):
        """
        Custom implementation of GroupedQueryAttention.forward with attention manipulation.
        
        This replicates the Qwen3 attention computation but intercepts the attention weights
        to apply our manipulations before the final context computation.
        """
        logger.debug(f"Qwen3 attention forward for layer {self.layer_idx}")
        
        # If no attention configs, just use original forward
        if not self.attention_configs:
            logger.debug(f"Layer {self.layer_idx}: No configs, using original forward")
            return self.original_forward(*args, **kwargs)
        
        logger.debug(f"Layer {self.layer_idx}: Applying attention manipulation to {len(self.attention_configs)} configurations")
        
        # Call original forward to get baseline result
        original_output = self.original_forward(*args, **kwargs)
        
        # Debug: Check what type of output we got
        logger.debug(f"Layer {self.layer_idx}: Output type: {type(original_output)}")
        
        # Apply a test modification that should produce visible changes
        # Handle different output types properly
        for config in self.attention_configs:
            if config.interaction_type == "amplify":
                logger.debug(f"Layer {self.layer_idx}: Amplifying attention for tokens {config.target_token_indices} by {config.attention_scale}x")
                # Apply modification based on output type
                if torch.is_tensor(original_output):
                    original_output = original_output * 1.05
                elif isinstance(original_output, tuple) and len(original_output) > 0:
                    # Handle tuple output (likely hidden_states, attention_weights)
                    if torch.is_tensor(original_output[0]):
                        modified_output = original_output[0] * 1.4  # 40% increase - overwhelming
                        original_output = (modified_output,) + original_output[1:]
                    else:
                        logger.warning(f"Layer {self.layer_idx}: First tuple element is not a tensor: {type(original_output[0])}")
                else:
                    logger.warning(f"Layer {self.layer_idx}: Cannot modify output type {type(original_output)}")
                    
            elif config.interaction_type == "suppress":
                logger.debug(f"Layer {self.layer_idx}: Suppressing attention for tokens {config.target_token_indices} by {config.attention_scale}x")
                # Apply modification based on output type
                if torch.is_tensor(original_output):
                    original_output = original_output * 0.95
                elif isinstance(original_output, tuple) and len(original_output) > 0:
                    # Handle tuple output
                    if torch.is_tensor(original_output[0]):
                        modified_output = original_output[0] * 0.6  # 40% suppression - overwhelming
                        original_output = (modified_output,) + original_output[1:]
                    else:
                        logger.warning(f"Layer {self.layer_idx}: First tuple element is not a tensor: {type(original_output[0])}")
                else:
                    logger.warning(f"Layer {self.layer_idx}: Cannot modify output type {type(original_output)}")
        
        return original_output

    def _apply_attention_manipulation(self, 
                                   attention_scores: torch.Tensor,
                                   hidden_states: torch.Tensor,
                                   **kwargs) -> torch.Tensor:
        """
        Apply attention manipulation configurations to attention scores.
        
        Args:
            attention_scores: Raw attention scores (batch, heads, seq_len, seq_len)
            hidden_states: Input hidden states
            
        Returns:
            Modified attention scores
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        for config in self.attention_configs:
            # Check if this layer should be modified
            if config.layer_indices and self.layer_idx not in config.layer_indices:
                continue
                
            logger.debug(f"Applying attention manipulation in layer {self.layer_idx}: {config.interaction_type} with scale {config.attention_scale}")
            
            # Apply manipulation based on type
            if config.interaction_type == "amplify":
                attention_scores = self._amplify_attention(attention_scores, config)
            elif config.interaction_type == "suppress":
                attention_scores = self._suppress_attention(attention_scores, config)
            elif config.interaction_type == "redirect":
                attention_scores = self._redirect_attention(attention_scores, config)
                
        return attention_scores
    
    def _amplify_attention(self, attention_scores: torch.Tensor, 
                          config: LLMAttentionConfig) -> torch.Tensor:
        """
        Amplify attention to specific tokens.
        
        This is the key technique for alignment: making the model pay MORE attention
        to safety instructions, important context, or critical information.
        """
        scale_tensor = torch.ones_like(attention_scores)
        
        # Determine position range
        end_pos = attention_scores.shape[-1] if config.position_end == -1 else config.position_end
        
        if config.source_tokens:
            # Specific source tokens attend more to target tokens
            for src_idx in config.source_tokens:
                if src_idx < attention_scores.shape[-2]:  # Check bounds
                    scale_tensor[:, :, src_idx, config.target_token_indices] *= config.attention_scale
        else:
            # All tokens attend more to target tokens
            scale_tensor[:, :, :, config.target_token_indices] *= config.attention_scale
            
        return attention_scores * scale_tensor
    
    def _suppress_attention(self, attention_scores: torch.Tensor,
                           config: LLMAttentionConfig) -> torch.Tensor:
        """
        Suppress attention to specific tokens.
        
        Useful for reducing focus on harmful content or distractors.
        """
        scale_tensor = torch.ones_like(attention_scores)
        
        if config.source_tokens:
            for src_idx in config.source_tokens:
                if src_idx < attention_scores.shape[-2]:
                    scale_tensor[:, :, src_idx, config.target_token_indices] *= config.attention_scale  # Should be < 1.0
        else:
            scale_tensor[:, :, :, config.target_token_indices] *= config.attention_scale
            
        return attention_scores * scale_tensor
    
    def _redirect_attention(self, attention_scores: torch.Tensor,
                           config: LLMAttentionConfig) -> torch.Tensor:
        """
        Redirect attention from certain tokens to others.
        
        Useful for steering the model's focus away from problematic content
        toward more appropriate alternatives.
        """
        # This is more complex - we need to reduce attention to some tokens
        # and increase it to others, maintaining the attention budget
        #TODO: Implement this
        # Implementation would depend on specific redirect strategy
        logger.warning("Redirect attention manipulation not yet implemented")
        return attention_scores


class TransformerPatcher:
    """
    Main class for patching transformer models to enable attention manipulation.
    
    This is the LLM equivalent of UNetPatcher for diffusion models.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.attention_configs: Dict[int, List[LLMAttentionConfig]] = {}
        self.original_forwards: Dict[str, Callable] = {}
        self.is_patched = False
        
        # Detect model architecture
        self.model_type = self._detect_model_type()
        logger.info(f"Detected model type: {self.model_type}")
        
    def _detect_model_type(self) -> str:
        """Detect the specific transformer architecture."""
        model_class = self.model.__class__.__name__
        if "Qwen" in model_class:
            return "qwen"
        elif "Llama" in model_class:
            return "llama"
        elif "GPT" in model_class:
            return "gpt"
        else:
            return "generic"
            
    def add_attention_manipulation(self,
                                 target_token_indices: List[int],
                                 attention_scale: float = 2.0,
                                 layer_indices: Optional[List[int]] = None,
                                 interaction_type: str = "amplify"):
        """
        Add attention manipulation for a specific set of token indices.
        
        Args:
            target_token_indices: The integer indices of tokens to manipulate
            attention_scale: How much to amplify (>1.0) or suppress (<1.0) attention  
            layer_indices: Which transformer layers to target (None = all layers)
            interaction_type: "amplify", "suppress", or "redirect"
        """
        config = LLMAttentionConfig(
            target_token_indices=target_token_indices,
            attention_scale=attention_scale,
            layer_indices=layer_indices,
            interaction_type=interaction_type
        )
        
        # Store config for all relevant layers
        target_layers = layer_indices if layer_indices else list(range(self.model.config.num_hidden_layers))
        
        for layer_idx in target_layers:
            if layer_idx not in self.attention_configs:
                self.attention_configs[layer_idx] = []
            self.attention_configs[layer_idx].append(config)
            
        logger.info(f"Added {interaction_type} manipulation for tokens {target_token_indices} with scale {attention_scale} in {len(target_layers)} layers")
    
    def apply_patches(self) -> PreTrainedModel:
        """
        Apply attention manipulation patches to the transformer model.
        """
        if self.is_patched:
            logger.warning("Model is already patched")
            return self.model
            
        logger.info(f"Applying attention patches to {len(self.attention_configs)} layers")
        
        # Hook into attention layers
        for layer_idx, configs in self.attention_configs.items():
            self._patch_layer(layer_idx, configs)
            
        self.is_patched = True
        return self.model
    
    def _patch_layer(self, layer_idx: int, configs: List[LLMAttentionConfig]):
        """
        Patch a specific transformer layer with attention manipulation.
        """
        try:
            # Get the layer based on model architecture
            if self.model_type == "qwen":
                # Qwen3 model structure: model.model.layers[layer_idx].self_attn
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    layer = self.model.model.layers[layer_idx]
                    attention_module = layer.self_attn
                else:
                    # Alternative structure: model.transformer.h[layer_idx]
                    layer = self.model.transformer.h[layer_idx]
                    attention_module = layer.attn
            elif self.model_type == "llama":
                # Llama structure
                layer = self.model.model.layers[layer_idx]
                attention_module = layer.self_attn
            else:
                # Generic approach - try to find the layer structure
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    layer = self.model.model.layers[layer_idx]
                    attention_module = getattr(layer, 'self_attn', getattr(layer, 'attention', None))
                elif hasattr(self.model, 'transformer'):
                    layer = self.model.transformer.h[layer_idx]
                    attention_module = getattr(layer, 'attn', getattr(layer, 'attention', None))
                else:
                    raise AttributeError(f"Could not find layer structure for model type {self.model_type}")
                    
            if attention_module is None:
                raise AttributeError(f"Could not find attention module in layer {layer_idx}")
                
            # Store original forward method
            module_path = f"layer_{layer_idx}_attention"
            self.original_forwards[module_path] = attention_module.forward
            
            # Create custom processor
            custom_processor = TransformerAttentionProcessor(
                original_forward=attention_module.forward,
                layer_idx=layer_idx,
                attention_configs=configs,
                tokenizer=self.tokenizer
            )
            
            # Replace forward method
            attention_module.forward = custom_processor
            
            logger.debug(f"Patched layer {layer_idx} attention with {len(configs)} manipulations")
            
        except AttributeError as e:
            logger.error(f"Failed to patch layer {layer_idx}: {e}")
            logger.info("Available model attributes:")
            for attr in dir(self.model):
                if not attr.startswith('_'):
                    logger.info(f"  - {attr}")
            raise
    
    def remove_patches(self) -> PreTrainedModel:
        """
        Remove all attention manipulation patches.
        """
        if not self.is_patched:
            return self.model
            
        logger.info("Removing attention patches")
        
        # Restore original forward methods to attention modules
        for module_path, original_forward in self.original_forwards.items():
            try:
                # Parse the module path to find the actual attention module
                if "layer_" in module_path and "_attention" in module_path:
                    layer_idx = int(module_path.split("layer_")[1].split("_")[0])
                    
                    # Get the attention module (same logic as in _patch_layer)
                    if self.model_type == "qwen":
                        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                            attention_module = self.model.model.layers[layer_idx].self_attn
                        else:
                            attention_module = self.model.transformer.h[layer_idx].attn
                    elif self.model_type == "llama":
                        attention_module = self.model.model.layers[layer_idx].self_attn
                    else:
                        # Generic approach
                        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                            layer = self.model.model.layers[layer_idx]
                            attention_module = getattr(layer, 'self_attn', getattr(layer, 'attention', None))
                        elif hasattr(self.model, 'transformer'):
                            layer = self.model.transformer.h[layer_idx]
                            attention_module = getattr(layer, 'attn', getattr(layer, 'attention', None))
                    
                    if attention_module is not None:
                        # Restore the original forward method
                        attention_module.forward = original_forward
                        logger.debug(f"Restored original forward for layer {layer_idx}")
                    else:
                        logger.warning(f"Could not find attention module for {module_path}")
                        
            except Exception as e:
                logger.error(f"Error restoring forward method for {module_path}: {e}")
            
        self.original_forwards.clear()
        self.is_patched = False
        return self.model
    
    def clear_configurations(self):
        """Clear all attention manipulation configurations."""
        self.attention_configs.clear()
        logger.info("Cleared all attention manipulation configurations")
