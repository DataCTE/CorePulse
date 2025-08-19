"""
Base classes for prompt injection in CorePulse.
"""

from abc import ABC
from typing import Dict, Optional, Any
import torch
from diffusers import DiffusionPipeline
from dataclasses import dataclass, field

from ..models.base import BlockIdentifier
from ..models.unet_patcher import UNetPatcher
from ..utils.logger import logger


@dataclass
class PromptInjectionConfig:
    """Configuration for a single prompt injection."""
    block: Any
    prompt: str
    weight: float
    sigma_start: float
    sigma_end: float
    spatial_mask: Optional[torch.Tensor] = None
    _encoded_prompt: Optional[torch.Tensor] = field(default=None, repr=False)


class BasePromptInjector(ABC):
    """
    Abstract base class for all prompt injectors.
    """
    
    def __init__(self, pipeline: DiffusionPipeline):
        logger.debug(f"Initializing {self.__class__.__name__} for pipeline: {pipeline.__class__.__name__}")
        try:
            self.patcher = UNetPatcher(pipeline.unet, pipeline.scheduler)
            self.configs: Dict[BlockIdentifier, PromptInjectionConfig] = {}
            self._pipeline = pipeline
            self._is_applied = False
        except Exception as e:
            logger.error(f"Error initializing BasePromptInjector: {e}", exc_info=True)
            raise

    @property
    def model_type(self) -> str:
        """Dynamically detect the model type from the pipeline."""
        # Local import to avoid circular dependency
        from ..utils.helpers import detect_model_type
        if self._pipeline:
            return detect_model_type(self._pipeline)
        return "unknown"

    def clear_injections(self):
        """Clear all injections and remove patches."""
        logger.debug("Clearing all injections and removing patches.")
        try:
            self.configs.clear()
            if self._pipeline:
                self.patcher.remove_patches(self._pipeline.unet)
            self.patcher.clear_injections()
        except Exception as e:
            logger.error(f"Error in clear_injections: {e}", exc_info=True)
            raise

    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """Apply all configured injections to the pipeline."""
        logger.debug(f"Applying patches to pipeline: {pipeline.__class__.__name__}")
        try:
            self.patcher.apply_patches(pipeline.unet)
            self._is_applied = True
            logger.info("Successfully applied patches to the pipeline.")
            return pipeline
        except Exception as e:
            logger.error(f"Error applying patches to pipeline: {e}", exc_info=True)
            raise

    def encode_prompt(self, prompt: str, pipeline: DiffusionPipeline) -> Dict[str, torch.Tensor]:
        """
        Encode a prompt using the pipeline's text encoder(s).
        
        Returns:
            A dictionary containing prompt_embeds and other relevant embeddings
            like pooled_prompt_embeds for SDXL.
        """
        logger.debug(f"Encoding prompt: '{prompt}'")
        try:
            prompt_args = {
                "prompt": prompt,
                "device": pipeline.device,
                "num_images_per_prompt": 1,
                "do_classifier_free_guidance": False,
            }

            if self.model_type == "sdxl":
                # SDXL encode_prompt returns more than 2 values
                result = pipeline.encode_prompt(**prompt_args)
                if isinstance(result, tuple) and len(result) >= 4:
                    # Typical SDXL return: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
                    prompt_embeds, _, pooled_prompt_embeds, _ = result[:4]
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Fallback for different SDXL versions
                    prompt_embeds, pooled_prompt_embeds = result[:2]
                else:
                    # Single return value
                    prompt_embeds = result
                    pooled_prompt_embeds = None
                
                result_dict = {"prompt_embeds": prompt_embeds}
                if pooled_prompt_embeds is not None:
                    result_dict["pooled_prompt_embeds"] = pooled_prompt_embeds
                return result_dict
            else:
                prompt_embeds = pipeline.encode_prompt(**prompt_args)
                return {"prompt_embeds": prompt_embeds}
        except Exception as e:
            logger.error(f"Error encoding prompt '{prompt}': {e}", exc_info=True)
            raise

    def __enter__(self):
        """Apply patches when entering context."""
        if self._pipeline:
            self.apply_to_pipeline(self._pipeline)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove patches when exiting context."""
        logger.debug("Exiting context, clearing injections.")
        self.clear_injections()

    def __call__(self, *args, **kwargs):
        """Make the injector callable to pass through to the pipeline."""
        if not self._is_applied or self._pipeline is None:
            msg = "Injector has not been applied to a pipeline. Call apply_to_pipeline or use a 'with' block."
            logger.error(msg)
            raise RuntimeError(msg)
        
        # Handle prompt encoding for attention manipulation
        prompt = kwargs.pop('prompt', None)
        if prompt is not None:
            logger.debug(f"Encoding prompt for pipeline call: '{prompt}'")
            # We must pass prompt_embeds instead of prompt to ensure the pipeline
            # uses our exact conditioning, especially for attention manipulation.
            embedding_dict = self.encode_prompt(prompt, self._pipeline)
            kwargs.update(embedding_dict)
            kwargs['prompt'] = None  # Ensure text prompt is not used

        logger.debug(f"Calling patched pipeline with {len(kwargs)} kwargs.")
        try:
            return self._pipeline(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error during patched pipeline execution: {e}", exc_info=True)
            raise
