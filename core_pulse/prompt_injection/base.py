"""
Base classes for prompt injection in CorePulse.
"""

from abc import ABC
from typing import Dict, Optional, Any, Union
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

    def encode_prompt(self, prompt: str, pipeline: DiffusionPipeline) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode a prompt using the pipeline's text encoder(s) with GPU optimization.
        
        Returns:
            For SDXL: Dict with 'prompt_embeds' and 'pooled_prompt_embeds'
            For SD1.5: Tensor with prompt embeddings
        """
        logger.debug(f"Starting encode_prompt for: '{prompt[:50]}...'")
        try:
            # Ensure pipeline components are on GPU if available
            device = pipeline.device if hasattr(pipeline, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.debug(f"Using device: {device}, Model type: {self.model_type}")
            
            if self.model_type == "sdxl":
                # FULLY GPU-OPTIMIZED SDXL encoding
                with torch.no_grad():
                    # Ensure text encoders are on GPU first
                    if not next(pipeline.text_encoder.parameters()).is_cuda and str(device).startswith('cuda'):
                        logger.debug("Moving text encoders to GPU")
                        pipeline.text_encoder = pipeline.text_encoder.to(device)
                        pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device)
                    
                    # Tokenize directly with GPU tensors
                    tokens_one = pipeline.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    tokens_two = pipeline.tokenizer_2(
                        prompt,
                        padding="max_length", 
                        max_length=pipeline.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    # Move tokens to GPU immediately (all operations stay on GPU)
                    input_ids_one = tokens_one.input_ids.to(device, non_blocking=True)
                    input_ids_two = tokens_two.input_ids.to(device, non_blocking=True)
                    
                    # Run both encoders on GPU
                    prompt_embeds_one_out = pipeline.text_encoder(input_ids_one, output_hidden_states=True)
                    prompt_embeds_two_out = pipeline.text_encoder_2(input_ids_two, output_hidden_states=True)
                    
                    # Extract embeddings and concatenate (all on GPU)
                    prompt_embeds_one = prompt_embeds_one_out.hidden_states[-2]
                    prompt_embeds_two = prompt_embeds_two_out.hidden_states[-2]
                    prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
                    
                    # Extract pooled embeddings from second encoder (required for SDXL)
                    pooled_prompt_embeds = prompt_embeds_two_out[0]
                    
                    logger.debug(f"SDXL encoding successful on {device}: prompt_embeds shape {prompt_embeds.shape}, pooled shape {pooled_prompt_embeds.shape}")
                    return {
                        'prompt_embeds': prompt_embeds,
                        'pooled_prompt_embeds': pooled_prompt_embeds
                    }
            else:
                # FULLY GPU-OPTIMIZED SD1.5 encoding
                with torch.no_grad():
                    # Ensure text encoder is on GPU first
                    if not next(pipeline.text_encoder.parameters()).is_cuda and str(device).startswith('cuda'):
                        logger.debug("Moving text encoder to GPU")
                        pipeline.text_encoder = pipeline.text_encoder.to(device)
                    
                    # Tokenize directly with GPU tensors
                    tokens = pipeline.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    # Move tokens to GPU immediately (all operations stay on GPU)
                    input_ids = tokens.input_ids.to(device, non_blocking=True)
                    
                    # Run encoder on GPU
                    prompt_embeds = pipeline.text_encoder(input_ids)[0]
                    
                    logger.debug(f"SD1.5 encoding successful on {device}: shape {prompt_embeds.shape}")
                    return prompt_embeds
                    
        except Exception as e:
            logger.error(f"Error encoding prompt '{prompt}': {e}", exc_info=True)
            raise

    def __enter__(self):
        """Apply patches when entering context."""
        logger.debug("Entering context manager")
        try:
            if self._pipeline:
                logger.debug("Pipeline exists, applying to pipeline")
                self.apply_to_pipeline(self._pipeline)
            else:
                logger.warning("No pipeline set in context manager")
            logger.debug("Context manager entry completed")
            return self
        except Exception as e:
            logger.error(f"Error in context manager __enter__: {e}", exc_info=True)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove patches when exiting context."""
        logger.debug("Exiting context, clearing injections")
        try:
            self.clear_injections()
            logger.debug("Context manager exit completed")
        except Exception as e:
            logger.error(f"Error in context manager __exit__: {e}", exc_info=True)
            # Don't re-raise in __exit__ as it could mask the original exception

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
            encoding_result = self.encode_prompt(prompt, self._pipeline)
            
            if isinstance(encoding_result, dict):
                # SDXL: Pass both prompt_embeds and pooled_prompt_embeds
                kwargs.update(encoding_result)
                logger.debug(f"SDXL: Passing both embeddings - prompt_embeds: {encoding_result['prompt_embeds'].shape}, pooled: {encoding_result['pooled_prompt_embeds'].shape}")
            else:
                # SD1.5: Just prompt_embeds
                kwargs['prompt_embeds'] = encoding_result
                logger.debug(f"SD1.5: Passing prompt_embeds: {encoding_result.shape}")
                
            kwargs['prompt'] = None  # Ensure text prompt is not used

        logger.debug(f"Calling patched pipeline with {len(kwargs)} kwargs: {list(kwargs.keys())}")
        try:
            return self._pipeline(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error during patched pipeline execution: {e}", exc_info=True)
            raise
