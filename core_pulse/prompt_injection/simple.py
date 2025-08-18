"""
Simple prompt injection interface for CorePulse.
"""

from typing import Union, Optional, List
import torch
from diffusers import DiffusionPipeline

from .base import BasePromptInjector, PromptInjectionConfig
from ..models.base import BlockIdentifier


class SimplePromptInjector(BasePromptInjector):
    """
    Simple interface for prompt injection.
    
    Provides an easy-to-use interface for injecting a single prompt
    into one or more blocks.
    """
    
    def __init__(self, model_type: str = "sdxl"):
        """
        Initialize simple prompt injector.
        
        Args:
            model_type: Model type ("sdxl" or "sd15")
        """
        super().__init__(model_type)
        self.config: Optional[PromptInjectionConfig] = None
    
    def configure_injections(self, 
                           block: Union[str, BlockIdentifier, List[Union[str, BlockIdentifier]]],
                           prompt: str,
                           weight: float = 1.0,
                           sigma_start: float = 0.0,
                           sigma_end: float = 1.0):
        """
        Configure prompt injection for one or more blocks, or all blocks.
        
        Args:
            block: Block identifier(s) to inject into (supports "all" for all blocks)
            prompt: Prompt to inject
            weight: Injection weight (default: 1.0)
            sigma_start: Start of injection window (default: 0.0)  
            sigma_end: End of injection window (default: 1.0)
        """
        self.clear_injections()
        
        # Handle "all" keyword
        if isinstance(block, str) and block.lower() == "all":
            blocks = self.patcher.block_mapper.get_all_block_identifiers()
        elif isinstance(block, list):
            blocks = block
        else:
            blocks = [block]
        
        for b in blocks:
            config = PromptInjectionConfig(
                block=b,
                prompt=prompt,
                weight=weight,
                sigma_start=sigma_start,
                sigma_end=sigma_end
            )
            self.config = config  # Store last config for reference
    
    def inject_prompt(self,
                     pipeline: DiffusionPipeline,
                     block: Union[str, BlockIdentifier, List[Union[str, BlockIdentifier]]],
                     prompt: str,
                     weight: float = 1.0,
                     sigma_start: float = 0.0,
                     sigma_end: float = 1.0) -> DiffusionPipeline:
        """
        Configure and apply prompt injection in one call.
        
        Args:
            pipeline: Diffusion pipeline to modify
            block: Block identifier(s) to inject into (supports "all" for all blocks)
            prompt: Prompt to inject
            weight: Injection weight (default: 1.0)
            sigma_start: Start of injection window (default: 0.0)
            sigma_end: End of injection window (default: 1.0)
            
        Returns:
            Modified pipeline
        """
        self.configure_injections(block, prompt, weight, sigma_start, sigma_end)
        return self.apply_to_pipeline(pipeline)
    
    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply configured injections to pipeline.
        
        Args:
            pipeline: Pipeline to modify
            
        Returns:
            Modified pipeline
        """
        if self.config is None:
            raise ValueError("No injections configured. Call configure_injections first.")
        
        # Encode the prompt
        encoded_prompt = self.encode_prompt(self.config.prompt, pipeline)
        
        # Add injection to patcher
        self.patcher.add_injection(
            block=self.config.block,
            conditioning=encoded_prompt,
            weight=self.config.weight,
            sigma_start=self.config.sigma_start,
            sigma_end=self.config.sigma_end
        )
        
        return super().apply_to_pipeline(pipeline)


class BlockSpecificInjector(SimplePromptInjector):
    """
    Convenience class for common block-specific injections.
    
    Provides preset methods for injecting into commonly used blocks.
    """
    
    def inject_content(self, pipeline: DiffusionPipeline, prompt: str, 
                      weight: float = 1.0) -> DiffusionPipeline:
        """
        Inject prompt into content/subject blocks (middle blocks).
        
        Args:
            pipeline: Pipeline to modify  
            prompt: Content prompt
            weight: Injection weight
            
        Returns:
            Modified pipeline
        """
        if self.model_type == "sdxl":
            return self.inject_prompt(pipeline, "middle:0", prompt, weight)
        else:  # sd15
            return self.inject_prompt(pipeline, ["middle:0", "middle:1"], prompt, weight)
    
    def inject_style(self, pipeline: DiffusionPipeline, prompt: str,
                    weight: float = 1.0) -> DiffusionPipeline:
        """
        Inject prompt into style blocks (output blocks).
        
        Args:
            pipeline: Pipeline to modify
            prompt: Style prompt
            weight: Injection weight
            
        Returns:
            Modified pipeline
        """
        if self.model_type == "sdxl":
            return self.inject_prompt(pipeline, ["output:0", "output:1"], prompt, weight)
        else:  # sd15
            return self.inject_prompt(pipeline, ["output:0", "output:1", "output:2"], prompt, weight)
    
    def inject_composition(self, pipeline: DiffusionPipeline, prompt: str,
                          weight: float = 1.0) -> DiffusionPipeline:
        """
        Inject prompt into composition blocks (input blocks).
        
        Args:
            pipeline: Pipeline to modify
            prompt: Composition prompt
            weight: Injection weight
            
        Returns:
            Modified pipeline
        """
        if self.model_type == "sdxl":
            return self.inject_prompt(pipeline, ["input:4", "input:5"], prompt, weight)
        else:  # sd15
            return self.inject_prompt(pipeline, ["input:7", "input:8", "input:9"], prompt, weight)


# Convenience functions for quick usage
def inject_content_prompt(pipeline: DiffusionPipeline, prompt: str, 
                         model_type: str = "sdxl", weight: float = 1.0) -> DiffusionPipeline:
    """
    Quick function to inject a content prompt.
    
    Args:
        pipeline: Pipeline to modify
        prompt: Content prompt
        model_type: Model type ("sdxl" or "sd15")
        weight: Injection weight
        
    Returns:
        Modified pipeline
    """
    injector = BlockSpecificInjector(model_type)
    return injector.inject_content(pipeline, prompt, weight)


def inject_style_prompt(pipeline: DiffusionPipeline, prompt: str,
                       model_type: str = "sdxl", weight: float = 1.0) -> DiffusionPipeline:
    """
    Quick function to inject a style prompt.
    
    Args:
        pipeline: Pipeline to modify
        prompt: Style prompt  
        model_type: Model type ("sdxl" or "sd15")
        weight: Injection weight
        
    Returns:
        Modified pipeline
    """
    injector = BlockSpecificInjector(model_type)
    return injector.inject_style(pipeline, prompt, weight)
