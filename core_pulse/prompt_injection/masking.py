"""
Masked prompt injection system for CorePulse.

This module provides token-level masking capabilities for selective prompt conditioning,
allowing fine-grained control over which parts of a prompt get modified during injection.
"""

import torch
import re
from typing import Dict, List, Union, Optional, Tuple, Any
from diffusers import DiffusionPipeline
from transformers import PreTrainedTokenizer

from .base import BasePromptInjector
from .advanced import AdvancedPromptInjector
from ..models.base import BlockIdentifier
from ..utils.logger import logger


class TokenMask:
    """
    Represents a token-level mask for selective conditioning.
    """
    
    def __init__(self, token_ids: torch.Tensor, mask: torch.Tensor, target_phrase: str):
        """
        Initialize token mask.
        
        Args:
            token_ids: Token IDs of the full prompt
            mask: Binary mask indicating which tokens to replace (1) or preserve (0)
            target_phrase: The phrase being targeted for replacement
        """
        self.token_ids = token_ids
        self.mask = mask  # Shape: [sequence_length]
        self.target_phrase = target_phrase
        
    @property
    def num_masked_tokens(self) -> int:
        """Number of tokens that will be replaced."""
        return int(self.mask.sum())
    
    def get_masked_positions(self) -> List[int]:
        """Get list of positions where mask is active."""
        return torch.nonzero(self.mask, as_tuple=True)[0].tolist()


class TokenAnalyzer:
    """
    Analyzes prompts to identify token positions for masking.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize token analyzer.
        
        Args:
            tokenizer: The tokenizer to use for analysis
        """
        logger.debug(f"Initializing TokenAnalyzer with tokenizer: {tokenizer.__class__.__name__}")
        self.tokenizer = tokenizer
        
    def create_phrase_mask(self, prompt: str, target_phrase: str, 
                          fuzzy_match: bool = True) -> TokenMask:
        """
        Create a token mask for a target phrase within a prompt.
        
        Args:
            prompt: Full prompt text
            target_phrase: Phrase to mask (e.g., "cat")
            fuzzy_match: Whether to allow fuzzy matching for phrases
            
        Returns:
            TokenMask object
        """
        logger.debug(f"Creating token mask for phrase '{target_phrase}' in prompt '{prompt}'")
        try:
            # Tokenize the full prompt
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            
            token_ids = tokens.input_ids[0]
            
            # Convert tokens back to text to find positions
            decoded_tokens = [
                self.tokenizer.decode([token_id], skip_special_tokens=True) 
                for token_id in token_ids
            ]
            
            # Find target phrase in the decoded tokens
            mask = torch.zeros_like(token_ids, dtype=torch.bool)
            
            if fuzzy_match:
                logger.debug("Using fuzzy matching for mask creation.")
                mask = self._create_fuzzy_mask(decoded_tokens, target_phrase, mask)
            else:
                logger.debug("Using exact matching for mask creation.")
                mask = self._create_exact_mask(decoded_tokens, target_phrase, mask)
            
            token_mask = TokenMask(token_ids, mask, target_phrase)
            logger.info(f"Created token mask with {token_mask.num_masked_tokens} masked tokens.")
            logger.debug(f"Masked positions: {token_mask.get_masked_positions()}")
            return token_mask
        except Exception as e:
            logger.error(f"Failed to create phrase mask for '{target_phrase}': {e}", exc_info=True)
            raise
    
    def _create_fuzzy_mask(self, decoded_tokens: List[str], target_phrase: str, 
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Create mask using fuzzy matching for multi-token phrases.
        """
        # Convert target phrase to lowercase for matching
        target_lower = target_phrase.lower().strip()
        
        # Build a sliding window to find the phrase
        for i in range(len(decoded_tokens)):
            # Build potential phrase from current position
            current_phrase = ""
            for j in range(i, min(i + 10, len(decoded_tokens))):  # Max 10 tokens
                token_text = decoded_tokens[j].strip()
                if token_text:  # Skip empty tokens
                    current_phrase += token_text
                    
                    # Check if we have a match
                    if target_lower in current_phrase.lower():
                        # Mark all tokens that contributed to this phrase
                        for k in range(i, j + 1):
                            if decoded_tokens[k].strip():  # Only mark non-empty tokens
                                mask[k] = True
                        return mask
                        
        return mask
    
    def _create_exact_mask(self, decoded_tokens: List[str], target_phrase: str,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Create mask using exact token matching.
        """
        target_tokens = self.tokenizer.encode(target_phrase, add_special_tokens=False)
        
        # Find exact sequence matches
        for i in range(len(decoded_tokens) - len(target_tokens) + 1):
            # Check if we have a match starting at position i
            match = True
            for j, target_token in enumerate(target_tokens):
                if decoded_tokens[i + j] != self.tokenizer.decode([target_token]):
                    match = False
                    break
            
            if match:
                # Mark the matching tokens
                for j in range(len(target_tokens)):
                    mask[i + j] = True
                break
                
        return mask


class MaskedPromptEncoder:
    """
    Encodes prompts with selective token-level replacement.
    """
    
    def __init__(self, injector: BasePromptInjector):
        """
        Initialize masked prompt encoder.
        
        Args:
            injector: Base prompt injector for encoding capabilities
        """
        logger.debug("Initializing MaskedPromptEncoder.")
        self.injector = injector
        
    def encode_masked_prompt(self, base_prompt: str, injection_prompt: str,
                           token_mask: TokenMask, pipeline: DiffusionPipeline) -> torch.Tensor:
        """
        Encode a prompt with selective token replacement.
        
        Args:
            base_prompt: Original prompt
            injection_prompt: Prompt containing replacement content
            token_mask: Mask indicating which tokens to replace
            pipeline: Pipeline for encoding
            
        Returns:
            Blended prompt embedding
        """
        
        logger.debug("Encoding masked prompt.")
        try:
            # Encode both prompts
            base_embedding = self.injector.encode_prompt(base_prompt, pipeline)
            injection_embedding = self.injector.encode_prompt(injection_prompt, pipeline)
            
            # Create blended embedding
            blended_embedding = self._blend_embeddings(
                base_embedding, injection_embedding, token_mask
            )
            
            logger.info("Successfully created blended prompt embedding.")
            return blended_embedding
        except Exception as e:
            logger.error(f"Failed to encode masked prompt: {e}", exc_info=True)
            raise
    
    def _blend_embeddings(self, base_embedding: torch.Tensor, 
                         injection_embedding: torch.Tensor, 
                         token_mask: TokenMask) -> torch.Tensor:
        """
        Blend embeddings at the token level using the mask.
        """
        # Ensure embeddings have the same shape
        if base_embedding.shape != injection_embedding.shape:
            msg = f"Embedding shapes don't match: {base_embedding.shape} vs {injection_embedding.shape}"
            logger.error(msg)
            raise ValueError(msg)
        
        logger.debug(f"Blending embeddings of shape {base_embedding.shape} with {token_mask.num_masked_tokens} masked tokens.")
        try:
            # Create result tensor
            blended = base_embedding.clone()
            
            # Move mask to same device as embeddings
            mask_device = token_mask.mask.to(device=base_embedding.device, dtype=torch.bool)
            
            # Apply mask - replace masked positions with injection embedding
            mask_expanded = mask_device.unsqueeze(-1).expand_as(base_embedding[0])
            
            
            # Blend: keep base where mask is 0, use injection where mask is 1
            blended[0] = torch.where(mask_expanded, injection_embedding[0], base_embedding[0])
            
            # For batch size > 1, apply the same blending
            for i in range(1, base_embedding.shape[0]):
                blended[i] = torch.where(mask_expanded, injection_embedding[i], base_embedding[i])
                
            return blended
        except Exception as e:
            logger.error(f"Failed to blend embeddings: {e}", exc_info=True)
            raise


class MaskedPromptInjector(AdvancedPromptInjector):
    """
    Advanced prompt injector with token-level masking capabilities.
    """
    
    def __init__(self, pipeline: DiffusionPipeline):
        """
        Initialize masked prompt injector.
        
        Args:
            pipeline: The diffusers pipeline to inject into.
        """
        super().__init__(pipeline)
        self.masked_configs: Dict[str, Dict[str, Any]] = {}
        try:
            self._token_analyzer = TokenAnalyzer(pipeline.tokenizer)
            self._masked_encoder = MaskedPromptEncoder(self)
        except AttributeError:
            logger.error("Pipeline must have a 'tokenizer' for MaskedPromptInjector.", exc_info=True)
            raise ValueError("Pipeline must have a tokenizer for masked injection")
        
    def add_masked_injection(self, 
                           block: Union[str, BlockIdentifier],
                           base_prompt: str,
                           injection_prompt: str, 
                           target_phrase: str,
                           weight: float = 1.0,
                           sigma_start: float = 0.0,
                           sigma_end: float = 1.0,
                           fuzzy_match: bool = True):
        """
        Add a masked injection that targets specific phrases.
        
        Args:
            block: Block identifier or "all"
            base_prompt: Original prompt 
            injection_prompt: Prompt with replacement content
            target_phrase: Specific phrase to replace (e.g., "cat")
            weight: Injection weight (1.0 = normal, >1.0 = amplified, <1.0 = weakened)
            sigma_start: Start of injection window
            sigma_end: End of injection window
            fuzzy_match: Whether to use fuzzy phrase matching
        """
        logger.info(f"Adding masked injection for block '{block}', targeting phrase '{target_phrase}'.")
        try:
            # Store masked configuration for later processing
            config_id = f"masked_{len(self.masked_configs)}"
            
            self.masked_configs[config_id] = {
                'block': block,
                'base_prompt': base_prompt,
                'injection_prompt': injection_prompt,
                'target_phrase': target_phrase,
                'weight': weight,
                'sigma_start': sigma_start,
                'sigma_end': sigma_end,
                'fuzzy_match': fuzzy_match
            }
            logger.debug(f"Stored masked config with id: {config_id}")
        except Exception as e:
            logger.error(f"Failed to add masked injection: {e}", exc_info=True)
            raise
        
    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply masked injections to pipeline.
        
        Args:
            pipeline: Pipeline to modify
            
        Returns:
            Modified pipeline
        """
        if not self.masked_configs:
            # Fall back to regular injection if no masked configs
            logger.debug("No masked configs found, falling back to regular injection.")
            return super().apply_to_pipeline(pipeline)
        
        logger.info(f"Applying {len(self.masked_configs)} masked injection(s).")
        try:
            # Initialize analyzers
            if hasattr(pipeline, 'tokenizer'):
                self._token_analyzer = TokenAnalyzer(pipeline.tokenizer)
                self._masked_encoder = MaskedPromptEncoder(self)
            else:
                raise ValueError("Pipeline must have tokenizer for masked injection")
            
            # Process each masked configuration
            for config_id, config in self.masked_configs.items():
                logger.debug(f"Processing masked config: {config_id}")
                # Create token mask
                token_mask = self._token_analyzer.create_phrase_mask(
                    config['base_prompt'], 
                    config['target_phrase'],
                    config['fuzzy_match']
                )
                
                # Encode masked prompt
                masked_embedding = self._masked_encoder.encode_masked_prompt(
                    config['base_prompt'],
                    config['injection_prompt'], 
                    token_mask,
                    pipeline
                )
                
                # Add as regular injection
                self.add_injection(
                    block=config['block'],
                    prompt="",  # Dummy prompt since we have pre-encoded embedding
                    weight=config['weight'],
                    sigma_start=config['sigma_start'],
                    sigma_end=config['sigma_end']
                )
                
                # Replace the encoded prompt directly in the configs
                # Handle "all" blocks first
                if isinstance(config['block'], str) and config['block'].lower() == "all":
                    all_blocks = self.patcher.block_mapper.get_all_block_identifiers()
                    for block_id_str in all_blocks:
                        block_id = BlockIdentifier.from_string(block_id_str)
                        if block_id in self.configs:
                            logger.debug(f"Injecting pre-encoded masked prompt into block: {block_id}")
                            self.configs[block_id]._encoded_prompt = masked_embedding
                else:
                    # Handle single block
                    block_id = BlockIdentifier.from_string(config['block']) if isinstance(config['block'], str) else config['block']
                    if block_id in self.configs:
                        logger.debug(f"Injecting pre-encoded masked prompt into block: {block_id}")
                        self.configs[block_id]._encoded_prompt = masked_embedding
            
            return super().apply_to_pipeline(pipeline)
        except Exception as e:
            logger.error(f"Failed to apply masked injections to pipeline: {e}", exc_info=True)
            raise
    
    def clear_injections(self):
        """Clear all injections including masked ones."""
        super().clear_injections()
        self.masked_configs.clear()
        
    def get_masking_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all masked injection configurations.
        
        Returns:
            List of masking summaries
        """
        logger.debug("Generating masking summary.")
        summaries = []
        for config_id, config in self.masked_configs.items():
            summaries.append({
                "config_id": config_id,
                "block": config['block'],
                "base_prompt": config['base_prompt'],
                "injection_prompt": config['injection_prompt'],
                "target_phrase": config['target_phrase'],
                "weight": config['weight'],
                "sigma_range": f"{config['sigma_start']:.1f} - {config['sigma_end']:.1f}",
                "fuzzy_match": config['fuzzy_match']
            })
        return summaries
