"""
Advanced prompt injection interface for CorePulse.
"""

from typing import Dict, List, Union, Optional, Any, Tuple
import torch
from diffusers import DiffusionPipeline

from .base import BasePromptInjector, PromptInjectionConfig
from ..models.base import BlockIdentifier


class AdvancedPromptInjector(BasePromptInjector):
    """
    Advanced interface for prompt injection with fine-grained control.
    
    Supports multiple prompts per block, complex injection schedules,
    and batch processing of injection configurations.
    """
    
    def __init__(self, model_type: str = "sdxl"):
        """
        Initialize advanced prompt injector.
        
        Args:
            model_type: Model type ("sdxl" or "sd15")
        """
        super().__init__(model_type)
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
        self.clear_injections()
        
        if isinstance(injection_map, dict):
            injection_map = [injection_map]
        
        for config_dict in injection_map:
            block = config_dict["block"]
            prompt = config_dict["prompt"]
            weight = config_dict.get("weight", 1.0)
            sigma_start = config_dict.get("sigma_start", 0.0)
            sigma_end = config_dict.get("sigma_end", 1.0)
            
            config = PromptInjectionConfig(
                block=block,
                prompt=prompt,
                weight=weight,
                sigma_start=sigma_start,
                sigma_end=sigma_end
            )
            
            block_id = BlockIdentifier.from_string(block) if isinstance(block, str) else block
            self.configs[block_id] = config
    
    def add_injection(self,
                     block: Union[str, BlockIdentifier],
                     prompt: str,
                     weight: float = 1.0,
                     sigma_start: float = 0.0,
                     sigma_end: float = 1.0):
        """
        Add a single prompt injection.
        
        Args:
            block: Block identifier
            prompt: Prompt to inject
            weight: Injection weight
            sigma_start: Start of injection window
            sigma_end: End of injection window
        """
        config = PromptInjectionConfig(
            block=block,
            prompt=prompt,
            weight=weight,
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )
        
        block_id = BlockIdentifier.from_string(block) if isinstance(block, str) else block
        self.configs[block_id] = config
    
    def remove_injection(self, block: Union[str, BlockIdentifier]):
        """
        Remove injection for a specific block.
        
        Args:
            block: Block identifier to remove
        """
        block_id = BlockIdentifier.from_string(block) if isinstance(block, str) else block
        if block_id in self.configs:
            del self.configs[block_id]
    
    def get_injection_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all configured injections.
        
        Returns:
            List of injection summaries
        """
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
        if not self.configs:
            raise ValueError("No injections configured. Add injections first.")
        
        # Encode all prompts and add to patcher
        for block_id, config in self.configs.items():
            encoded_prompt = self.encode_prompt(config.prompt, pipeline)
            
            self.patcher.add_injection(
                block=block_id,
                conditioning=encoded_prompt,
                weight=config.weight,
                sigma_start=config.sigma_start,
                sigma_end=config.sigma_end
            )
        
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
        self.clear_injections()
        
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
        self.clear_injections()
        
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
        # Configure content blocks (middle)
        if self.model_type == "sdxl":
            self.add_injection("middle:0", content_prompt, content_weight)
        else:  # sd15
            for i in [0, 1]:
                self.add_injection(f"middle:{i}", content_prompt, content_weight)
        
        # Configure style blocks (output)  
        if self.model_type == "sdxl":
            for i in [0, 1]:
                self.add_injection(f"output:{i}", style_prompt, style_weight)
        else:  # sd15
            for i in [0, 1, 2]:
                self.add_injection(f"output:{i}", style_prompt, style_weight)


# Factory functions for common use cases
def create_content_style_injector(model_type: str = "sdxl") -> MultiPromptInjector:
    """
    Create an injector preconfigured for content/style separation.
    
    Args:
        model_type: Model type ("sdxl" or "sd15")
        
    Returns:
        Configured MultiPromptInjector
    """
    return MultiPromptInjector(model_type)


def create_location_injector(locations_str: str, 
                           prompt: str,
                           model_type: str = "sdxl",
                           sigma_start: float = 0.0,
                           sigma_end: float = 1.0) -> LocationBasedInjector:
    """
    Create and configure a location-based injector.
    
    Args:
        locations_str: Location string in "block:index,weight" format
        prompt: Prompt to inject
        model_type: Model type ("sdxl" or "sd15") 
        sigma_start: Start of injection window
        sigma_end: End of injection window
        
    Returns:
        Configured LocationBasedInjector
    """
    injector = LocationBasedInjector(model_type)
    injector.configure_from_locations(locations_str, prompt, sigma_start, sigma_end)
    return injector
