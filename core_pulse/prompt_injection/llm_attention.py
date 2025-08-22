"""
LLM Attention Manipulation for Alignment and Control.

This module provides high-level interfaces for manipulating attention in Large Language Models,
with a focus on alignment applications like amplifying safety instructions and suppressing
harmful patterns.
"""

from typing import List, Union, Optional, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass, field

from ..models.transformer_patcher import TransformerPatcher
from ..utils.logger import logger


@dataclass
class ManipulationConfig:
    """Stores a single attention manipulation configuration before token resolution."""
    target_phrase: str
    attention_scale: float
    layer_indices: Optional[List[int]]
    interaction_type: str


class LLMAttentionInjector:
    """
    High-level interface for LLM attention manipulation.
    
    This class provides an intuitive API for controlling what Large Language Models
    pay attention to. You can amplify or suppress attention to any concepts in the input.
    
    Key use cases:
    - Amplify attention to specific phrases or concepts
    - Suppress focus on certain patterns or distractors
    - Enhance attention to important context
    - Control model behavior through attention weighting
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize the LLM attention manipulator.
        
        Args:
            model: The transformer model to manipulate (e.g., Qwen3-4B-Instruct-2507)
            tokenizer: The tokenizer associated with the model
        """
        try:
            logger.info(f"Initializing LLMAttentionInjector for {model.__class__.__name__}")
            self.model = model
            self.tokenizer = tokenizer
            self.patcher = TransformerPatcher(model, tokenizer)
            self._is_applied = False
            self._manipulation_configs: List[ManipulationConfig] = []
        except Exception as e:
            logger.error(f"Failed to initialize LLMAttentionInjector: {e}", exc_info=True)
            raise
        
    def amplify_phrases(self, 
                      phrases: List[str],
                      amplification_factor: float = 3.0,
                      layer_indices: Optional[List[int]] = None) -> 'LLMAttentionInjector':
        """
        Configure amplification for the model's attention to specific phrases.
        
        This makes the model pay stronger attention to the specified phrases,
        giving them more influence in the generation process. Note: This only
        configures the manipulation; it is applied during a `with` block or
        when `generate_with_manipulation` is called.
        
        Args:
            phrases: List of phrases to amplify attention to
            amplification_factor: How much to amplify (2.0 = double attention)
            layer_indices: Which layers to target (None = all layers)
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info(f"Configuring phrase amplification for {len(phrases)} phrases with factor {amplification_factor}")
            for phrase in phrases:
                self._manipulation_configs.append(
                    ManipulationConfig(
                        target_phrase=phrase,
                        attention_scale=amplification_factor,
                        layer_indices=layer_indices,
                        interaction_type="amplify"
                    )
                )
            return self
        except Exception as e:
            logger.error(f"Failed to configure amplification for phrases {phrases}: {e}", exc_info=True)
            raise
        
    def suppress_phrases(self,
                       phrases: List[str],
                       suppression_factor: float = 0.2,
                       layer_indices: Optional[List[int]] = None) -> 'LLMAttentionInjector':
        """
        Configure suppression for the model's attention to specific phrases.
        
        This reduces the model's focus on the specified content, making it
        less influential in the generation process. Note: This only
        configures the manipulation; it is applied during a `with` block or
        when `generate_with_manipulation` is called.
        
        Args:
            phrases: List of phrases to suppress attention to
            suppression_factor: How much to suppress (0.1 = 10% of normal attention)
            layer_indices: Which layers to target (None = all layers)
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info(f"Configuring phrase suppression for {len(phrases)} phrases with factor {suppression_factor}")
            for phrase in phrases:
                self._manipulation_configs.append(
                    ManipulationConfig(
                        target_phrase=phrase,
                        attention_scale=suppression_factor,
                        layer_indices=layer_indices,
                        interaction_type="suppress"
                    )
                )
            return self
        except Exception as e:
            logger.error(f"Failed to configure suppression for phrases {phrases}: {e}", exc_info=True)
            raise
        
    def add_custom_manipulation(self,
                              target_phrase: str,
                              attention_scale: float,
                              interaction_type: str = "amplify",
                              layer_indices: Optional[List[int]] = None) -> 'LLMAttentionInjector':
        """
        Add a custom attention manipulation for advanced use cases.

        Args:
            target_phrase: The phrase to manipulate attention for.
            attention_scale: Scale factor (>1.0 amplify, <1.0 suppress).
            interaction_type: "amplify", "suppress", or "redirect".
            layer_indices: Which layers to target.

        Returns:
            Self for method chaining.
        """
        try:
            logger.info(f"Adding custom {interaction_type} manipulation for phrase '{target_phrase}' with scale {attention_scale}")
            self._manipulation_configs.append(
                ManipulationConfig(
                    target_phrase=target_phrase,
                    attention_scale=attention_scale,
                    layer_indices=layer_indices,
                    interaction_type=interaction_type
                )
            )
            return self
        except Exception as e:
            logger.error(f"Failed to add custom manipulation for '{target_phrase}': {e}", exc_info=True)
            raise
    
    def apply_manipulations(self, prompt: str) -> PreTrainedModel:
        """
        Resolve token indices for the given prompt and apply patches.
        
        Args:
            prompt: The full prompt text to use for token resolution.
            
        Returns:
            The modified model with attention patches applied.
        """
        try:
            logger.info("Applying attention manipulations to model")
            self.patcher.clear_configurations() # Clear only the patcher's previous state
            
            self._resolve_and_add_to_patcher(prompt)
            
            if not self.patcher.attention_configs:
                logger.warning("No valid manipulations to apply after token resolution")
                return self.model
                
            self.patcher.apply_patches()
            self._is_applied = True
            
            logger.info("Successfully applied attention manipulations to model")
            return self.model
        except Exception as e:
            logger.error(f"Failed to apply attention manipulations: {e}", exc_info=True)
            self._is_applied = False
            raise
    
    def remove_manipulations(self) -> PreTrainedModel:
        """
        Remove all attention manipulations and restore original behavior.
        
        Returns:
            The restored model
        """
        try:
            if not self._is_applied:
                logger.debug("No manipulations currently applied to remove")
                return self.model
                
            logger.info("Removing attention manipulations")
            self.patcher.remove_patches()
            self._is_applied = False
            
            logger.info("Successfully removed attention manipulations")
            return self.model
        except Exception as e:
            logger.error(f"Failed to remove attention manipulations: {e}", exc_info=True)
            self._is_applied = False
            raise
    
    def clear_configurations(self, clear_patcher: bool = True):
        """
        Clear all stored phrase configurations.
        
        Args:
            clear_patcher: Also clear any existing patches on the model.
        """
        try:
            logger.info("Clearing all attention manipulation configurations")
            self._manipulation_configs.clear()
            if clear_patcher:
                self.patcher.clear_configurations()
            logger.debug("Successfully cleared all configurations")
        except Exception as e:
            logger.error(f"Error clearing configurations: {e}", exc_info=True)
            raise
        
    def _resolve_and_add_to_patcher(self, prompt: str):
        """
        Resolve token indices for all configured phrases and add them to the patcher.
        """
        try:
            logger.debug(f"Resolving token indices for {len(self._manipulation_configs)} configurations")
            resolved_count = 0
            for config in self._manipulation_configs:
                token_indices = self._find_token_indices_robust(prompt, config.target_phrase)
                
                if token_indices:
                    self.patcher.add_attention_manipulation(
                        target_token_indices=token_indices,
                        attention_scale=config.attention_scale,
                        layer_indices=config.layer_indices,
                        interaction_type=config.interaction_type
                    )
                    resolved_count += 1
                else:
                    logger.warning(f"Could not find token indices for phrase '{config.target_phrase}' in prompt. Skipping manipulation.")
            
            logger.info(f"Token resolution complete: {resolved_count} manipulations added to patcher, {len(self._manipulation_configs) - resolved_count} skipped.")
            
        except Exception as e:
            logger.error(f"An error occurred during token resolution: {e}", exc_info=True)
            raise
            
    def _find_token_indices_robust(self, text: str, target_phrase: str) -> List[int]:
        """
        Find token indices for a target phrase within text using a robust
        decode-and-search strategy.
        """
        try:
            prompt_tokens = self.tokenizer.encode(text)
            decoded_tokens = [self.tokenizer.decode([t]) for t in prompt_tokens]
            target_lower = target_phrase.lower().strip()

            for i in range(len(decoded_tokens)):
                current_phrase = ""
                for j in range(i, len(decoded_tokens)):
                    token_text = decoded_tokens[j].strip()
                    if not token_text:
                        continue
                    
                    phrase_with_space = current_phrase + " " + token_text if current_phrase else token_text
                    current_phrase = phrase_with_space.strip()

                    # Check for bidirectional substring match for more robust finding
                    is_forward_match = target_lower in current_phrase.lower()
                    is_reverse_match = len(current_phrase) > 3 and current_phrase.lower() in target_lower

                    if is_forward_match or is_reverse_match:
                        start_token_idx = -1
                        end_token_idx = -1
                        temp_phrase = ""
                        for k in range(i, j + 1):
                            reconstructed_token = decoded_tokens[k].strip()
                            if not reconstructed_token:
                                continue
                            if temp_phrase:
                                temp_phrase += " "
                            if start_token_idx == -1 and target_lower in (temp_phrase + reconstructed_token).lower():
                                start_token_idx = k
                            temp_phrase += reconstructed_token
                            if start_token_idx != -1 and target_lower in temp_phrase.lower():
                                end_token_idx = k
                                return list(range(start_token_idx, end_token_idx + 1))
            return []
        except Exception as e:
            logger.error(f"Error in _find_token_indices_robust for '{target_phrase}': {e}", exc_info=True)
            return []
        
    def generate_with_manipulation(self, 
                                 prompt: str,
                                 max_new_tokens: int = 100,
                                 **generation_kwargs) -> str:
        """
        Generate text with attention manipulations applied contextually.
        
        This is a high-level method that handles applying patches for the specific
        prompt, generating text, and then automatically cleaning up.
        
        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum tokens to generate.
            **generation_kwargs: Additional arguments for model.generate().
            
        Returns:
            Generated text.
        """
        try:
            logger.info(f"Generating with manipulation for prompt: '{prompt[:100]}...'")
            
            with self as injector: # Use context manager for auto cleanup
                # Resolve tokens and apply patches for *this specific prompt*
                injector.apply_manipulations(prompt)
                
                logger.debug("Tokenizing prompt and moving to device")
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                logger.debug(f"Input shape: {inputs['input_ids'].shape}, Device: {self.model.device}")
                
                logger.debug("Starting generation with attention manipulations")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs
                    )
                
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                logger.info(f"Generation complete. Generated {len(generated_tokens)} tokens")
                
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error during generation with manipulation: {e}", exc_info=True)
            # Ensure cleanup happens on error
            if self._is_applied:
                self.remove_manipulations()
            raise
    
    def __enter__(self):
        """Context manager entry. Note: patches are applied with a prompt."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove any applied manipulations."""
        try:
            logger.debug("Exiting context manager - removing manipulations")
            self.remove_manipulations()
        except Exception as e:
            logger.error(f"Error exiting context manager: {e}", exc_info=True)
        
    def get_manipulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all configured attention manipulations.
        
        Returns:
            Dictionary with manipulation details
        """
        summary = {
            "configured_phrases": len(self._manipulation_configs),
            "total_layers_patched": len(self.patcher.attention_configs),
            "total_manipulations_applied": sum(len(configs) for configs in self.patcher.attention_configs.values()),
            "is_applied": self._is_applied,
            "model_type": self.patcher.model_type,
        }
        return summary


# Convenience factory functions for common patterns
def create_concept_amplified_model(model: PreTrainedModel, 
                                  tokenizer: PreTrainedTokenizer,
                                  target_concepts: List[str],
                                  amplification_factor: float = 3.0) -> LLMAttentionInjector:
    """
    Create an LLM with enhanced focus on specific concepts.
    
    This is a convenience function for amplifying attention to particular phrases or ideas.
    
    Args:
        model: The transformer model
        tokenizer: Associated tokenizer
        target_concepts: List of concepts/phrases to amplify
        amplification_factor: How much to amplify attention (default 3.0)
        
    Returns:
        Configured LLMAttentionInjector ready for use
    """
    try:
        logger.info(f"Creating concept-amplified model for {len(target_concepts)} concepts with factor {amplification_factor}")
        injector = LLMAttentionInjector(model, tokenizer)
        injector.amplify_phrases(target_concepts, amplification_factor=amplification_factor)
        logger.info(f"Successfully created concept-amplified model")
        return injector
    except Exception as e:
        logger.error(f"Failed to create concept-amplified model: {e}", exc_info=True)
        raise


def create_balanced_attention_model(model: PreTrainedModel,
                                   tokenizer: PreTrainedTokenizer,
                                   amplify_concepts: List[str],
                                   suppress_concepts: List[str],
                                   amplification_factor: float = 3.0,
                                   suppression_factor: float = 0.2) -> LLMAttentionInjector:
    """
    Create an LLM with both amplified and suppressed attention patterns.
    
    Args:
        model: The transformer model  
        tokenizer: Associated tokenizer
        amplify_concepts: Concepts to amplify attention to
        suppress_concepts: Concepts to suppress attention to
        amplification_factor: How much to amplify (default 3.0)
        suppression_factor: How much to suppress (default 0.2)
        
    Returns:
        Configured LLMAttentionInjector with balanced attention
    """
    try:
        logger.info(f"Creating balanced attention model")
        injector = LLMAttentionInjector(model, tokenizer)
        injector.amplify_phrases(amplify_concepts, amplification_factor=amplification_factor)
        injector.suppress_phrases(suppress_concepts, suppression_factor=suppression_factor)
        logger.info(f"Successfully created balanced attention model with {len(amplify_concepts)} amplified and {len(suppress_concepts)} suppressed concepts")
        return injector
    except Exception as e:
        logger.error(f"Failed to create balanced attention model: {e}", exc_info=True)
        raise
