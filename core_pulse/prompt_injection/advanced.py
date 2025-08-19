"""
Advanced prompt injection interface for CorePulse.
"""

from typing import Dict, List, Union, Optional, Any, Tuple
import torch
from diffusers import DiffusionPipeline

from .base import BasePromptInjector, PromptInjectionConfig
from ..models.base import BlockIdentifier
from ..utils.logger import logger


class AdvancedPromptInjector(BasePromptInjector):
    """
    Advanced interface for prompt injection with fine-grained control.
    
    Supports multiple prompts per block, complex injection schedules,
    and batch processing of injection configurations.
    """
    
    def __init__(self, pipeline: DiffusionPipeline):
        super().__init__(pipeline)
        self.configs: Dict[BlockIdentifier, PromptInjectionConfig] = {}
    
    def configure_injections(self, 
                           injection_map: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        Configure multiple prompt injections from a mapping or list.
        
        Args:
            injection_map: Dictionary or list of injection configurations
                          Format: {"block": "input:4", "prompt": "text", "weight": 1.0, ...}
                          Or list of such dictionaries
        """
        logger.info(f"Configuring {len(injection_map)} injections.")
        self.clear_injections()
        
        try:
            if isinstance(injection_map, dict):
                injection_map = [injection_map]
            
            for config_dict in injection_map:
                block = config_dict["block"]
                prompt = config_dict["prompt"]
                logger.debug(f"Parsing injection for block '{block}' with prompt '{prompt}'")
                
                weight = config_dict.get("weight", 1.0)
                sigma_start = config_dict.get("sigma_start", 0.0)
                sigma_end = config_dict.get("sigma_end", 1.0)
                
                config = PromptInjectionConfig(
                    block=block,
                    prompt=prompt,
                    weight=weight,
                    sigma_start=sigma_start,
                    sigma_end=sigma_end,
                    spatial_mask=config_dict.get("spatial_mask")
                )
                
                block_id = BlockIdentifier.from_string(block) if isinstance(block, str) else block
                self.configs[block_id] = config
        except KeyError as e:
            logger.error(f"Injection configuration is missing required key: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to configure injections: {e}", exc_info=True)
            raise
    
    def add_injection(self,
                     block: Union[str, BlockIdentifier],
                     prompt: str,
                     weight: float = 1.0,
                     sigma_start: float = 0.0,
                     sigma_end: float = 1.0,
                     spatial_mask: Optional[torch.Tensor] = None):
        """
        Add a single prompt injection or inject into all blocks.
        
        Args:
            block: Block identifier (string like "input:4", "all", or BlockIdentifier)
            prompt: Prompt to inject
            weight: Injection weight (1.0 = normal, >1.0 = amplified, <1.0 = weakened)
            sigma_start: Start of injection window
            sigma_end: End of injection window
            spatial_mask: Optional spatial mask for regional control
        """
        logger.debug(f"Adding injection for block: {block}, prompt: '{prompt}'")
        try:
            if isinstance(block, str) and block.lower() == "all":
                logger.info("Applying injection to all available blocks.")
                all_blocks = self.patcher.block_mapper.get_all_block_identifiers()
                for block_id_str in all_blocks:
                    self._add_single_injection(block_id_str, prompt, weight, sigma_start, sigma_end, spatial_mask)
            else:
                self._add_single_injection(block, prompt, weight, sigma_start, sigma_end, spatial_mask)
        except Exception as e:
            logger.error(f"Failed to add injection for block '{block}': {e}", exc_info=True)
            raise

    def _add_single_injection(self, block, prompt, weight, sigma_start, sigma_end, spatial_mask):
        """Helper to add a single injection config."""
        config = PromptInjectionConfig(
            block=block,
            prompt=prompt,
            weight=weight,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            spatial_mask=spatial_mask
        )
        block_id = BlockIdentifier.from_string(block) if isinstance(block, str) else block
        logger.debug(f"Stored injection config for block_id: {block_id}")
        self.configs[block_id] = config
    
    def remove_injection(self, block: Union[str, BlockIdentifier]):
        """
        Remove injection for a specific block.
        
        Args:
            block: Block identifier to remove
        """
        logger.debug(f"Removing injection for block: {block}")
        try:
            block_id = BlockIdentifier.from_string(block) if isinstance(block, str) else block
            if block_id in self.configs:
                del self.configs[block_id]
                logger.info(f"Removed injection for block: {block_id}")
            else:
                logger.warning(f"Attempted to remove injection for block {block_id}, but none was found.")
        except Exception as e:
            logger.error(f"Failed to remove injection for block '{block}': {e}", exc_info=True)
            raise
    
    def get_injection_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all configured injections.
        
        Returns:
            List of injection summaries
        """
        logger.debug("Generating injection summary.")
        summaries = []
        for block_id, config in self.configs.items():
            summaries.append({
                "block": str(block_id),
                "prompt": config.prompt,
                "weight": config.weight,
                "sigma_start": config.sigma_start,
                "sigma_end": config.sigma_end
            })
        return summaries
    
    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """
        Apply all configured injections to pipeline.
        
        Args:
            pipeline: Pipeline to modify
            
        Returns:
            Modified pipeline
        """
        # Check if there are any prompt injections or attention map configs in the patcher
        if not self.configs and not self.patcher.attention_map_configs:
            raise ValueError("No injections or attention manipulations configured. Add them first.")
        
        logger.info(f"Encoding {len(self.configs)} prompts for injection.")
        # Encode all prompts and add to patcher
        for block_id, config in self.configs.items():
            try:
                # Use pre-encoded prompt if available, otherwise encode the prompt
                if config._encoded_prompt is not None:
                    logger.debug(f"Using pre-encoded prompt for block {block_id}")
                    encoded_prompt = config._encoded_prompt
                else:
                    encoded_prompt = self.encode_prompt(config.prompt, pipeline)
                
                self.patcher.add_injection(
                    block=block_id,
                    conditioning=encoded_prompt,
                    weight=config.weight,
                    sigma_start=config.sigma_start,
                    sigma_end=config.sigma_end,
                    spatial_mask=config.spatial_mask
                )
            except Exception as e:
                logger.error(f"Failed to process and add injection for block {block_id}: {e}", exc_info=True)
                raise
        
        return super().apply_to_pipeline(pipeline)
    
    def clear_injections(self):
        """Clear all configured injections."""
        self.configs.clear()
        super().clear_injections()


class LocationBasedInjector(AdvancedPromptInjector):
    """
    Injector that uses location strings similar to the ComfyUI original.
    
    Supports configurations like "output:0,1.0\noutput:1,0.8" for 
    backward compatibility with the original node format.
    """
    
    def configure_from_locations(self, 
                               locations_str: str,
                               prompt: str,
                               sigma_start: float = 0.0,
                               sigma_end: float = 1.0):
        """
        Configure injections from a locations string.
        
        Args:
            locations_str: Multi-line string with "block:index,weight" format
            prompt: Prompt to inject at all locations
            sigma_start: Start of injection window
            sigma_end: End of injection window
        """
        logger.info("Configuring injections from location string.")
        self.clear_injections()
        
        try:
            for line in locations_str.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                weight = 1.0
                if ',' in line:
                    block_str, weight_str = line.split(',', 1)
                    weight = float(weight_str.strip())
                else:
                    block_str = line
                
                self.add_injection(
                    block=block_str.strip(),
                    prompt=prompt,
                    weight=weight,
                    sigma_start=sigma_start,
                    sigma_end=sigma_end
                )
        except Exception as e:
            logger.error(f"Failed to parse location string: '{locations_str}'. Error: {e}", exc_info=True)
            raise


class MultiPromptInjector(AdvancedPromptInjector):
    """
    Injector that supports different prompts for each block.
    
    Useful for complex scenarios where each block needs specific conditioning.
    """
    
    def configure_block_prompts(self, 
                              block_prompts: Dict[str, str],
                              weights: Optional[Dict[str, float]] = None,
                              sigma_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Configure different prompts for different blocks.
        
        Args:
            block_prompts: Mapping of block identifiers to prompts
            weights: Optional mapping of block identifiers to weights
            sigma_ranges: Optional mapping of block identifiers to (start, end) sigma ranges
        """
        logger.info(f"Configuring {len(block_prompts)} block-specific prompts.")
        self.clear_injections()
        
        try:
            weights = weights or {}
            sigma_ranges = sigma_ranges or {}
            
            for block_str, prompt in block_prompts.items():
                weight = weights.get(block_str, 1.0)
                sigma_start, sigma_end = sigma_ranges.get(block_str, (0.0, 1.0))
                
                self.add_injection(
                    block=block_str,
                    prompt=prompt,
                    weight=weight,
                    sigma_start=sigma_start,
                    sigma_end=sigma_end
                )
        except Exception as e:
            logger.error(f"Failed to configure block prompts: {e}", exc_info=True)
            raise
    
    def add_content_style_split(self,
                              content_prompt: str,
                              style_prompt: str,
                              content_weight: float = 1.0,
                              style_weight: float = 1.0):
        """
        Add content/style split similar to the ComfyUI example.
        
        Args:
            content_prompt: Prompt for content blocks (middle)
            style_prompt: Prompt for style blocks (output)
            content_weight: Weight for content injection
            style_weight: Weight for style injection
        """
        logger.info("Adding content/style split.")
        logger.debug(f"  - Content prompt: '{content_prompt}' (weight: {content_weight})")
        logger.debug(f"  - Style prompt: '{style_prompt}' (weight: {style_weight})")
        
        try:
            # Get all available middle and output blocks for the current model
            middle_blocks = self.patcher.block_mapper.blocks.get('middle', [])
            output_blocks = self.patcher.block_mapper.blocks.get('output', [])

            # Configure content blocks (middle)
            logger.debug(f"Applying content prompt to middle blocks: {middle_blocks}")
            for i in middle_blocks:
                self.add_injection(f"middle:{i}", content_prompt, content_weight)
            
            # Configure style blocks (output)
            logger.debug(f"Applying style prompt to output blocks: {output_blocks}")
            for i in output_blocks:
                self.add_injection(f"output:{i}", style_prompt, style_weight)
        except Exception as e:
            logger.error(f"Failed to add content/style split: {e}", exc_info=True)
            raise
