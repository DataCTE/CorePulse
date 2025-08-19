"""
Base classes for prompt injection in CorePulse.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any
import torch
from diffusers import DiffusionPipeline

from ..models.base import BlockIdentifier
from ..models.unet_patcher import UNetPatcher


class BasePromptInjector(ABC):
    """
    Abstract base class for prompt injection tools.
    
    Prompt injectors provide high-level interfaces for injecting prompts
    at specific blocks of diffusion models.
    """
    
    def __init__(self, model_type: str = "sdxl"):
        """
        Initialize prompt injector.
        
        Args:
            model_type: Model type ("sdxl" or "sd15")
        """
        self.model_type = model_type
        self.patcher = UNetPatcher(model_type)
        self._pipeline: Optional[DiffusionPipeline] = None
        self._is_applied = False
    
    def encode_prompt(self, prompt: str, pipeline: DiffusionPipeline) -> torch.Tensor:
        """
        Encode a prompt using the pipeline's text encoder.
        
        Args:
            prompt: Text prompt to encode
            pipeline: Diffusion pipeline with text encoder
            
        Returns:
            Encoded prompt tensor
        """
        prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            device=pipeline.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False, # We only need the conditional part
        )[0]
        return prompt_embeds
    
    def encode_prompts(self, prompts: Union[str, List[str]], 
                      pipeline: DiffusionPipeline) -> torch.Tensor:
        """
        Encode multiple prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            pipeline: Diffusion pipeline
            
        Returns:
            Batch of encoded prompt tensors
        """
        if isinstance(prompts, str):
            return self.encode_prompt(prompts, pipeline)
        
        encoded_prompts = []
        for prompt in prompts:
            encoded = self.encode_prompt(prompt, pipeline)
            encoded_prompts.append(encoded)
        
        return torch.cat(encoded_prompts, dim=0)
    
    @abstractmethod
    def configure_injections(self, *args, **kwargs):
        """
        Configure prompt injections. Implementation varies by subclass.
        """
        pass
    
    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply prompt injections to a diffusion pipeline.
        
        Args:
            pipeline: Pipeline to modify
            
        Returns:
            Modified pipeline
        """
        if self._is_applied:
            raise RuntimeError("Injections already applied. Call remove_from_pipeline first.")
        
        self._pipeline = pipeline
        self.patcher.apply_patches(pipeline.unet)
        self._is_applied = True
        
        return self
    
    def remove_from_pipeline(self) -> Optional[DiffusionPipeline]:
        """
        Remove prompt injections from the pipeline.
        
        Returns:
            Restored pipeline if one was previously modified
        """
        if not self._is_applied or self._pipeline is None:
            return None
        
        self.patcher.remove_patches(self._pipeline.unet)
        pipeline = self._pipeline
        self._pipeline = None
        self._is_applied = False
        
        return pipeline
    
    def clear_injections(self):
        """Clear all configured injections."""
        self.patcher.clear_injections()
    
    def __call__(self, *args, **kwargs):
        """Make the injector callable to pass through to the pipeline."""
        if not self._is_applied or self._pipeline is None:
            raise RuntimeError("Injector has not been applied to a pipeline.")
        return self._pipeline(*args, **kwargs)

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_applied:
            self.remove_from_pipeline()


class PromptInjectionConfig:
    """
    Configuration class for prompt injections.
    """
    
    def __init__(self, block: Union[str, BlockIdentifier], 
                 prompt: str,
                 weight: float = 1.0,
                 sigma_start: float = 0.0,
                 sigma_end: float = 1.0,
                 spatial_mask: Optional[torch.Tensor] = None):
        """
        Initialize injection configuration.
        
        Args:
            block: Block identifier
            prompt: Prompt to inject
            weight: Injection weight (1.0 = normal, >1.0 = amplified, <1.0 = weakened)
            sigma_start: Start sigma for injection window
            sigma_end: End sigma for injection window
            spatial_mask: Optional spatial mask for regional control
        """
        self.block = BlockIdentifier.from_string(block) if isinstance(block, str) else block
        self.prompt = prompt
        self.weight = weight
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.spatial_mask = spatial_mask
        self._encoded_prompt: Optional[torch.Tensor] = None
    
    def get_encoded_prompt(self, pipeline: DiffusionPipeline) -> torch.Tensor:
        """
        Get the encoded prompt, encoding if necessary.
        
        Args:
            pipeline: Pipeline to use for encoding
            
        Returns:
            Encoded prompt tensor
        """
        if self._encoded_prompt is None:
            injector = BasePromptInjector()
            self._encoded_prompt = injector.encode_prompt(self.prompt, pipeline)
        
        return self._encoded_prompt
    
    def clear_cache(self):
        """Clear cached encoded prompt."""
        self._encoded_prompt = None
