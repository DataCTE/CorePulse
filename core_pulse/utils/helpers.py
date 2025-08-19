"""
Helper functions for CorePulse integration.
"""

from typing import Dict, List, Union, Optional, Any, TYPE_CHECKING
import torch
from diffusers import (
    DiffusionPipeline, 
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel
)
from ..utils.logger import logger

if TYPE_CHECKING:
    from ..prompt_injection import SimplePromptInjector, AdvancedPromptInjector


def detect_model_type(pipeline: DiffusionPipeline) -> str:
    """
    Automatically detect the model type from a pipeline.
    
    Args:
        pipeline: Diffusion pipeline to analyze
        
    Returns:
        Model type string ("sdxl" or "sd15")
    """
    logger.debug("Detecting model type...")
    
    # Primary check: Class type
    if isinstance(pipeline, StableDiffusionXLPipeline):
        logger.info("Detected model type: sdxl (by class)")
        return "sdxl"
    if isinstance(pipeline, StableDiffusionPipeline):
        logger.info("Detected model type: sd15 (by class)")
        return "sd15"
        
    # Secondary check: Attributes (e.g. text_encoder_2 for SDXL)
    if hasattr(pipeline, 'text_encoder_2'):
        logger.info("Detected model type: sdxl (by text_encoder_2 attribute)")
        return "sdxl"
        
    # Tertiary check: UNet architecture
    if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'down_blocks'):
        down_block_count = len(pipeline.unet.down_blocks)
        if down_block_count >= 4: # SD 1.5 has 4
            logger.info(f"Detected model type: sd15 (by unet config: {down_block_count} down_blocks)")
            return "sd15"
        else: # SDXL has 3
            logger.info(f"Detected model type: sdxl (by unet config: {down_block_count} down_blocks)")
            return "sdxl"
    
    # Default fallback
    logger.warning("Could not determine model type, falling back to sdxl.")
    return "sdxl"


def get_available_blocks(pipeline: DiffusionPipeline) -> Dict[str, List[int]]:
    """
    Get available blocks for a model type.
    
    Args:
        pipeline: The diffusion pipeline to inspect.
        
    Returns:
        Dictionary mapping block types to available indices
    """
    logger.debug("Getting available blocks for the pipeline.")
    try:
        from ..models.unet_patcher import UNetBlockMapper
        mapper = UNetBlockMapper(pipeline.unet)
        blocks = mapper.get_valid_blocks()
        logger.debug(f"Found available blocks: {blocks}")
        return blocks
    except Exception as e:
        logger.error(f"Failed to get available blocks: {e}", exc_info=True)
        raise


def create_quick_injector(pipeline: DiffusionPipeline,
                         interface: str = "simple") -> Union["SimplePromptInjector", "AdvancedPromptInjector"]:
    """
    Create a prompt injector with automatically detected model type.
    
    Args:
        pipeline: Pipeline to create injector for
        interface: Interface type ("simple" or "advanced")
        
    Returns:
        Configured prompt injector
    """
    logger.debug(f"Creating quick injector with '{interface}' interface.")
    try:
        # Local import to avoid circular dependency
        from ..prompt_injection import SimplePromptInjector, AdvancedPromptInjector
        
        if interface.lower() == "simple":
            return SimplePromptInjector(pipeline)
        elif interface.lower() == "advanced":
            return AdvancedPromptInjector(pipeline)
        else:
            raise ValueError(f"Unknown interface type: {interface}")
    except Exception as e:
        logger.error(f"Failed to create quick injector: {e}", exc_info=True)
        raise


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
    logger.info("Performing inject and generate.")
    logger.debug(f"Base prompt: '{base_prompt}'")
    logger.debug(f"Injection config: {injection_config}")
    try:
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
            
            logger.info("Inject and generate completed successfully.")
            return result
    except Exception as e:
        logger.error(f"Inject and generate failed: {e}", exc_info=True)
        raise


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
    logger.info("Demonstrating content/style split.")
    try:
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
            
            logger.info("Content/style split demonstration completed successfully.")
            return result
    except Exception as e:
        logger.error(f"Content/style split demonstration failed: {e}", exc_info=True)
        raise


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
    logger.debug(f"Validating injection config: {config}")
    errors = []
    try:
        from ..models.unet_patcher import UNetBlockMapper
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
    except Exception as e:
        logger.error(f"An unexpected error occurred during config validation: {e}", exc_info=True)
        errors.append(f"Unexpected error: {e}")
    
    if errors:
        logger.warning(f"Validation found {len(errors)} issues in config.")
    else:
        logger.debug("Injection config validation passed.")
    return errors


# Convenience aliases for backward compatibility
quick_inject = inject_and_generate
auto_detect_model = detect_model_type
