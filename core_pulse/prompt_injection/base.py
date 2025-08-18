"""
Base classes for prompt injection in CorePulse.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

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
        print(f" ENCODING PROMPT: '{prompt}'")
        
        if hasattr(pipeline, 'text_encoder') and hasattr(pipeline, 'tokenizer'):
            # Standard diffusers pipeline
            text_inputs = pipeline.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            print(f"   Tokenized: {text_inputs.input_ids[0][:10]}...")  # First 10 tokens
            
            with torch.no_grad():
                text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(pipeline.device))[0]
            
            print(f"   Encoded shape: {text_embeddings.shape}")
            print(f"   First few values: {text_embeddings[0, 0, :5]}")
            
            return text_embeddings
        
        elif hasattr(pipeline, 'text_encoder_2'):
            # SDXL pipeline with dual text encoders
            # Use the primary text encoder for simplicity
            text_inputs = pipeline.tokenizer(
                prompt,
                padding="max_length", 
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(pipeline.device))[0]
            
            return text_embeddings
        
        else:
            raise ValueError("Pipeline must have text_encoder and tokenizer")
    
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
        
        return pipeline
    
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
                 sigma_end: float = 1.0):
        """
        Initialize injection configuration.
        
        Args:
            block: Block identifier
            prompt: Prompt to inject
            weight: Injection weight
            sigma_start: Start sigma for injection window
            sigma_end: End sigma for injection window
        """
        self.block = BlockIdentifier.from_string(block) if isinstance(block, str) else block
        self.prompt = prompt
        self.weight = weight
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
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
