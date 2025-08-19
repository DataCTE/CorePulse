"""
Helper functions for CorePulse integration.
"""

from typing import Dict, List, Union, Optional, Any
import torch
from diffusers import (
    DiffusionPipeline, 
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel
)

from ..prompt_injection import SimplePromptInjector, AdvancedPromptInjector
from ..models.unet_patcher import UNetBlockMapper


def detect_model_type(pipeline: DiffusionPipeline) -> str:
    """
    Automatically detect the model type from a pipeline.
    
    Args:
        pipeline: Diffusion pipeline to analyze
        
    Returns:
        Model type string ("sdxl" or "sd15")
    """
    if isinstance(pipeline, StableDiffusionXLPipeline):
        return "sdxl"
    elif isinstance(pipeline, StableDiffusionPipeline):
        return "sd15"
    
    # Fallback: check UNet architecture
    if hasattr(pipeline, 'unet') and isinstance(pipeline.unet, UNet2DConditionModel):
        # Check for SDXL characteristics (dual text encoders, larger UNet)
        if hasattr(pipeline, 'text_encoder_2'):
            return "sdxl"
        
        # Check UNet block count as a heuristic
        if hasattr(pipeline.unet, 'down_blocks'):
            down_block_count = len(pipeline.unet.down_blocks)
            if down_block_count >= 4:
                return "sdxl"
            else:
                return "sd15"
    
    # Default fallback
    return "sdxl"


def get_available_blocks(pipeline: DiffusionPipeline) -> Dict[str, List[int]]:
    """
    Get available blocks for a model type.
    
    Args:
        pipeline: The diffusion pipeline to inspect.
        
    Returns:
        Dictionary mapping block types to available indices
    """
    mapper = UNetBlockMapper(pipeline.unet)
    return mapper.get_valid_blocks()


def create_quick_injector(pipeline: DiffusionPipeline,
                         interface: str = "simple") -> Union[SimplePromptInjector, AdvancedPromptInjector]:
    """
    Create a prompt injector with automatically detected model type.
    
    Args:
        pipeline: Pipeline to create injector for
        interface: Interface type ("simple" or "advanced")
        
    Returns:
        Configured prompt injector
    """
    
    if interface.lower() == "simple":
        return SimplePromptInjector(pipeline)
    elif interface.lower() == "advanced":
        return AdvancedPromptInjector(pipeline)
    else:
        raise ValueError(f"Unknown interface type: {interface}")


def inject_and_generate(pipeline: DiffusionPipeline,
                       base_prompt: str,
                       injection_config: Dict[str, Any],
                       num_inference_steps: int = 20,
                       guidance_scale: float = 7.5,
                       **generate_kwargs) -> Any:
    """
    Convenient function to inject prompts and generate in one call.
    
    Args:
        pipeline: Diffusion pipeline
        base_prompt: Base prompt for generation
        injection_config: Configuration for prompt injection
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        **generate_kwargs: Additional generation parameters
        
    Returns:
        Generated output
    """
    # Create and configure injector
    injector = create_quick_injector(pipeline, "advanced")
    
    # Handle different injection config formats
    if isinstance(injection_config, dict):
        if "block" in injection_config:
            # Single injection config
            injector.add_injection(**injection_config)
        else:
            # Multiple injections or block-prompt mapping
            if all(isinstance(v, str) for v in injection_config.values()):
                # Block-prompt mapping
                for block, prompt in injection_config.items():
                    injector.add_injection(block, prompt)
            else:
                # List of injection configs
                injector.configure_injections(injection_config)
    
    # Apply injections
    with injector:
        modified_pipeline = injector.apply_to_pipeline(pipeline)
        
        # Generate
        result = modified_pipeline(
            prompt=base_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **generate_kwargs
        )
        
        return result


def demonstrate_content_style_split(pipeline: DiffusionPipeline,
                                  base_prompt: str = "a beautiful landscape",
                                  content_prompt: str = "white cat",
                                  style_prompt: str = "oil painting style",
                                  num_inference_steps: int = 20,
                                  guidance_scale: float = 7.5) -> Any:
    """
    Demonstrate the content/style split capability.
    
    This recreates the example from the original ComfyUI node where
    "white cat" is injected into content blocks while "blue dog" style
    is applied to other blocks.
    
    Args:
        pipeline: Diffusion pipeline
        base_prompt: Base prompt for generation
        content_prompt: Prompt for content blocks
        style_prompt: Prompt for style blocks
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        
    Returns:
        Generated output
    """
    from ..prompt_injection.advanced import MultiPromptInjector
    
    injector = MultiPromptInjector(pipeline)
    
    injector.add_content_style_split(
        content_prompt=content_prompt,
        style_prompt=style_prompt
    )
    
    with injector:
        modified_pipeline = injector.apply_to_pipeline(pipeline)
        
        result = modified_pipeline(
            prompt=base_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        return result


def validate_injection_config(config: Dict[str, Any], 
                            pipeline: DiffusionPipeline) -> List[str]:
    """
    Validate an injection configuration and return any errors.
    
    Args:
        config: Injection configuration to validate
        pipeline: Pipeline to validate against
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    mapper = UNetBlockMapper(pipeline.unet)
    
    required_fields = ["block", "prompt"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    if "block" in config:
        try:
            from ..models.base import BlockIdentifier
            block = BlockIdentifier.from_string(config["block"])
            if not mapper.is_valid_block(block):
                errors.append(f"Invalid block for {detect_model_type(pipeline)}: {config['block']}")
        except ValueError as e:
            errors.append(f"Invalid block format: {e}")
    
    # Validate numeric fields
    numeric_fields = ["weight", "sigma_start", "sigma_end"]
    for field in numeric_fields:
        if field in config:
            try:
                float(config[field])
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {field}: {config[field]}")
    
    # Validate sigma range
    if "sigma_start" in config and "sigma_end" in config:
        try:
            start = float(config["sigma_start"])
            end = float(config["sigma_end"])
            if start < 0 or end < 0 or start > 1 or end > 1:
                errors.append("Sigma values must be between 0 and 1")
            if start < end:
                errors.append("sigma_start should be >= sigma_end (higher noise to lower noise)")
        except (ValueError, TypeError):
            pass  # Already caught above
    
    return errors


# Convenience aliases for backward compatibility
quick_inject = inject_and_generate
auto_detect_model = detect_model_type
