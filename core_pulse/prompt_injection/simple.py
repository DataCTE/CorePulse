"""
Simple prompt injection interface for CorePulse.
"""

from typing import Union, Optional, List
import torch
from diffusers import DiffusionPipeline

from .base import BasePromptInjector, PromptInjectionConfig
from ..models.base import BlockIdentifier
from ..utils.logger import logger


class SimplePromptInjector(BasePromptInjector):
    """
    Simple interface for prompt injection.
    
    Provides an easy-to-use interface for injecting a single prompt
    into one or more blocks.
    """
    
    def __init__(self, pipeline: DiffusionPipeline):
        """
        Initialize simple prompt injector.
        
        Args:
            pipeline: The diffusers pipeline to inject into.
        """
        super().__init__(pipeline)
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
        logger.info(f"Configuring simple injection for block(s) '{block}' with prompt '{prompt}'")
        self.clear_injections()
        
        try:
            # Handle "all" keyword
            if isinstance(block, str) and block.lower() == "all":
                logger.debug("Expanding 'all' to all available blocks.")
                blocks = self.patcher.block_mapper.get_all_block_identifiers()
            elif isinstance(block, list):
                blocks = block
            else:
                blocks = [block]
            
            logger.debug(f"Targeting {len(blocks)} block(s).")
            for b in blocks:
                config = PromptInjectionConfig(
                    block=b,
                    prompt=prompt,
                    weight=weight,
                    sigma_start=sigma_start,
                    sigma_end=sigma_end
                )
                self.config = config  # Store last config for reference
        except Exception as e:
            logger.error(f"Failed to configure simple injection: {e}", exc_info=True)
            raise
    
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
        logger.debug("Performing one-call inject_prompt.")
        try:
            self.configure_injections(block, prompt, weight, sigma_start, sigma_end)
            return self.apply_to_pipeline(pipeline)
        except Exception as e:
            logger.error(f"Failed during inject_prompt: {e}", exc_info=True)
            raise
    
    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply configured injections to pipeline.
        
        Args:
            pipeline: Pipeline to modify
            
        Returns:
            Modified pipeline
        """
        if self.config is None:
            msg = "No injections configured. Call configure_injections first."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.debug(f"Applying simple injection for block '{self.config.block}'")
        try:
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
        except Exception as e:
            logger.error(f"Failed to apply simple injection to pipeline: {e}", exc_info=True)
            raise


class BlockSpecificInjector(SimplePromptInjector):
    """
    Convenience class for common block-specific injections.
    
    Provides preset methods for injecting into commonly used blocks.
    """
    
    def inject_content(self, prompt: str, weight: float = 1.0):
        """
        Inject prompt into content/subject blocks (middle blocks).
        
        Args:
            pipeline: Pipeline to modify  
            prompt: Content prompt
            weight: Injection weight
            
        Returns:
            Modified pipeline
        """
        logger.info(f"Injecting content prompt '{prompt}' with weight {weight}.")
        try:
            middle_blocks = [f"middle:{i}" for i in self.patcher.block_mapper.blocks.get('middle', [])]
            if not middle_blocks:
                logger.warning("No middle blocks found for content injection.")
            self.configure_injections(middle_blocks, prompt, weight)
        except Exception as e:
            logger.error(f"Failed to inject content prompt: {e}", exc_info=True)
            raise
    
    def inject_style(self, prompt: str, weight: float = 1.0):
        """
        Inject prompt into style blocks (output blocks).
        
        Args:
            pipeline: Pipeline to modify
            prompt: Style prompt
            weight: Injection weight
            
        Returns:
            Modified pipeline
        """
        logger.info(f"Injecting style prompt '{prompt}' with weight {weight}.")
        try:
            style_blocks = [f"output:{i}" for i in self.patcher.block_mapper.blocks.get('output', [])]
            if not style_blocks:
                logger.warning("No output blocks found for style injection.")
            self.configure_injections(style_blocks, prompt, weight)
        except Exception as e:
            logger.error(f"Failed to inject style prompt: {e}", exc_info=True)
            raise
    
    def inject_composition(self, prompt: str, weight: float = 1.0):
        """
        Inject prompt into composition blocks (input blocks).
        
        Args:
            pipeline: Pipeline to modify
            prompt: Composition prompt
            weight: Injection weight
            
        Returns:
            Modified pipeline
        """
        logger.info(f"Injecting composition prompt '{prompt}' with weight {weight}.")
        try:
            composition_blocks = [f"input:{i}" for i in self.patcher.block_mapper.blocks.get('input', [])]
            if not composition_blocks:
                logger.warning("No input blocks found for composition injection.")
            self.configure_injections(composition_blocks, prompt, weight)
        except Exception as e:
            logger.error(f"Failed to inject composition prompt: {e}", exc_info=True)
            raise


# Convenience functions for quick usage
def inject_content_prompt(pipeline: DiffusionPipeline, prompt: str, weight: float = 1.0) -> DiffusionPipeline:
    """
    Quick function to inject a content prompt.
    
    Args:
        pipeline: Pipeline to modify
        prompt: Content prompt
        weight: Injection weight
        
    Returns:
        Modified pipeline
    """
    injector = BlockSpecificInjector(pipeline)
    injector.inject_content(prompt, weight)
    return injector.apply_to_pipeline(pipeline)


def inject_style_prompt(pipeline: DiffusionPipeline, prompt: str, weight: float = 1.0) -> DiffusionPipeline:
    """
    Quick function to inject a style prompt.
    
    Args:
        pipeline: Pipeline to modify
        prompt: Style prompt  
        weight: Injection weight
        
    Returns:
        Modified pipeline
    """
    injector = BlockSpecificInjector(pipeline)
    injector.inject_style(prompt, weight)
    return injector.apply_to_pipeline(pipeline)
