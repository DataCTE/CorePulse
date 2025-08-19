"""
Advanced attention control for CorePulse.
"""

from typing import List, Union, Optional
import torch
from diffusers import DiffusionPipeline

from .advanced import AdvancedPromptInjector
from ..models.base import BlockIdentifier
from ..models.unet_patcher import AttentionMapConfig


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

    def add_attention_manipulation(self,
                                   block: Union[str, BlockIdentifier],
                                   target_phrase: str,
                                   attention_scale: float = 1.0,
                                   sigma_start: float = 1.0,
                                   sigma_end: float = 0.0,
                                   spatial_mask: Optional[torch.Tensor] = None):
        """
        Add an attention map manipulation for a specific phrase.
        """
        # --- Tokenization for concatenated embeddings (SDXL) ---
        all_token_ids = []

        # Tokenize with the first tokenizer
        token_ids_1 = self.tokenizer.encode(target_phrase, add_special_tokens=False)
        if token_ids_1:
            all_token_ids.extend(token_ids_1)

        # Tokenize with the second tokenizer if it exists
        if self.tokenizer_2:
            token_ids_2 = self.tokenizer_2.encode(target_phrase, add_special_tokens=False)
            if token_ids_2:
                # Offset the indices by the max length of the first tokenizer
                offset = self.tokenizer.model_max_length
                all_token_ids.extend([tid + offset for tid in token_ids_2])
        
        if not all_token_ids:
            raise ValueError(f"Could not tokenize target phrase: {target_phrase}")

        # Add the manipulation to the patcher
        self.patcher.add_attention_manipulation(
            block=block,
            target_token_indices=all_token_ids,
            attention_scale=attention_scale,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            spatial_mask=spatial_mask
        )

    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply all configured injections and manipulations to the pipeline.
        """
        # The patcher now handles both, so we just call the super method
        return super().apply_to_pipeline(pipeline)
