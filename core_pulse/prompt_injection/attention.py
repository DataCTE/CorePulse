"""
Advanced attention control for CorePulse.
"""

from typing import List, Union, Optional
import torch
from diffusers import DiffusionPipeline

from .advanced import AdvancedPromptInjector
from ..models.base import BlockIdentifier
from ..models.unet_patcher import AttentionMapConfig
from ..utils.tokenizer import find_token_indices
from ..utils.logger import logger


class AttentionMapInjector(AdvancedPromptInjector):
    """
    An injector that provides fine-grained control over attention maps.
    """

    def __init__(self, pipeline: DiffusionPipeline):
        """
        Initialize the attention map injector.
        """
        super().__init__(pipeline)
        if not hasattr(pipeline, 'tokenizer'):
            raise ValueError("Pipeline must have a tokenizer for attention map injection.")
        self.tokenizer = pipeline.tokenizer
        self.tokenizer_2 = getattr(pipeline, 'tokenizer_2', None)
        self._target_prompt = None # The prompt that will be manipulated

    def add_attention_manipulation(self,
                                   prompt: str,
                                   block: Union[str, BlockIdentifier],
                                   target_phrase: str,
                                   attention_scale: float = 1.0,
                                   sigma_start: float = 1.0,
                                   sigma_end: float = 0.0,
                                   spatial_mask: Optional[torch.Tensor] = None):
        """
        Add an attention map manipulation for a specific phrase.
        """
        try:
            self._target_prompt = prompt

            # --- Tokenization for concatenated embeddings (SDXL) ---
            all_token_indices = []

            # Find indices with the first tokenizer
            indices_1 = find_token_indices(self.tokenizer, prompt, target_phrase)
            if indices_1:
                all_token_indices.extend(indices_1)

            # Find indices with the second tokenizer if it exists
            if self.tokenizer_2:
                indices_2 = find_token_indices(self.tokenizer_2, prompt, target_phrase)
                if indices_2:
                    # Offset the indices by the max length of the first tokenizer's context
                    offset = self.tokenizer.model_max_length
                    all_token_indices.extend([idx + offset for idx in indices_2])
            
            if not all_token_indices:
                raise ValueError(f"Could not find target phrase '{target_phrase}' in prompt '{prompt}'.")

            # Add the manipulation to the patcher
            self.patcher.add_attention_manipulation(
                block=block,
                target_token_indices=all_token_indices,
                attention_scale=attention_scale,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                spatial_mask=spatial_mask
            )
        except Exception as e:
            logger.error(f"Failed to add attention manipulation for '{target_phrase}': {e}", exc_info=True)
            raise

    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply all configured injections and manipulations to the pipeline.
        """
        # The patcher now handles both, so we just call the super method
        return super().apply_to_pipeline(pipeline)
