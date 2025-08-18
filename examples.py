"""
CorePulse Examples

This file demonstrates how to use the CorePulse toolkit for prompt injection
with diffusers pipelines. These examples show different usage patterns and
capabilities of the system.

Run these examples to see prompt injection in action!
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

# Import CorePulse components
from core_pulse import SimplePromptInjector, AdvancedPromptInjector
from core_pulse.prompt_injection.simple import BlockSpecificInjector, inject_content_prompt
from core_pulse.prompt_injection.advanced import MultiPromptInjector, LocationBasedInjector
from core_pulse.utils import (
    detect_model_type, 
    create_quick_injector, 
    inject_and_generate,
    demonstrate_content_style_split,
    get_available_blocks
)


def example_1_simple_injection():
    """
    Example 1: Simple prompt injection using the basic interface.
    """
    print("=== Example 1: Simple Prompt Injection ===")
    
    # Load a pipeline (replace with your preferred model)
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    
    # Create a simple injector
    injector = SimplePromptInjector("sd15")  # or auto-detect with detect_model_type(pipeline)
    
    # Inject "white cat" into middle blocks (content)
    with injector:
        modified_pipeline = injector.inject_prompt(
            pipeline=pipeline,
            block="middle:0",  # Content block
            prompt="white cat",
            weight=1.0
        )
        
        # Generate with the base prompt
        result = modified_pipeline(
            prompt="a beautiful blue dog in a garden",
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        # This should generate an image with a cat (content) but in the style/setting of the blue dog prompt
        result.images[0].save("example_1_simple_injection.png")
    
    print("Generated image saved as example_1_simple_injection.png")


def example_2_block_specific_injection():
    """
    Example 2: Using block-specific injectors for content/style separation.
    """
    print("=== Example 2: Block-Specific Injection ===")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    
    # Use block-specific injector
    injector = BlockSpecificInjector("sdxl")
    
    with injector:
        # Inject different prompts for content and style
        modified_pipeline = injector.inject_content(
            pipeline=pipeline,
            prompt="majestic lion",
            weight=1.0
        )
        
        result = modified_pipeline(
            prompt="cute puppy playing in a meadow, watercolor painting style",
            num_inference_steps=25,
            guidance_scale=7.5
        )
        
        result.images[0].save("example_2_content_injection.png")
    
    print("Generated image saved as example_2_content_injection.png")


def example_3_advanced_multi_block():
    """
    Example 3: Advanced injection with multiple blocks and configurations.
    """
    print("=== Example 3: Advanced Multi-Block Injection ===")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    
    # Create advanced injector
    injector = AdvancedPromptInjector("sdxl")
    
    # Configure multiple injections
    injection_configs = [
        {
            "block": "middle:0",
            "prompt": "elegant swan",
            "weight": 1.2,
            "sigma_start": 1.0,
            "sigma_end": 0.5
        },
        {
            "block": "output:0", 
            "prompt": "impressionist painting style",
            "weight": 0.8,
            "sigma_start": 0.7,
            "sigma_end": 0.0
        },
        {
            "block": "output:1",
            "prompt": "golden hour lighting",
            "weight": 0.9,
            "sigma_start": 0.6,
            "sigma_end": 0.0
        }
    ]
    
    injector.configure_injections(injection_configs)
    
    with injector:
        modified_pipeline = injector.apply_to_pipeline(pipeline)
        
        result = modified_pipeline(
            prompt="a bird by a lake in the morning",
            num_inference_steps=30,
            guidance_scale=8.0
        )
        
        result.images[0].save("example_3_advanced_injection.png")
    
    print("Generated image saved as example_3_advanced_injection.png")


def example_4_location_based():
    """
    Example 4: Location-based injection (ComfyUI-style format).
    """
    print("=== Example 4: Location-Based Injection ===")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    
    # Use location string format like the original ComfyUI node
    locations = """
    input:7,1.0
    middle:0,1.2
    output:0,0.8
    output:1,0.9
    """
    
    injector = LocationBasedInjector("sd15")
    injector.configure_from_locations(
        locations_str=locations,
        prompt="mystical forest creature",
        sigma_start=1.0,
        sigma_end=0.3
    )
    
    with injector:
        modified_pipeline = injector.apply_to_pipeline(pipeline)
        
        result = modified_pipeline(
            prompt="a small animal in an urban environment",
            num_inference_steps=25,
            guidance_scale=7.5
        )
        
        result.images[0].save("example_4_location_based.png")
    
    print("Generated image saved as example_4_location_based.png")


def example_5_content_style_split():
    """
    Example 5: Content/style split using the multi-prompt injector.
    """
    print("=== Example 5: Content/Style Split ===")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    
    # Use the convenience function
    result = demonstrate_content_style_split(
        pipeline=pipeline,
        base_prompt="an animal in a landscape",
        content_prompt="white cat with blue eyes",
        style_prompt="dramatic lighting, renaissance painting style",
        num_inference_steps=25,
        guidance_scale=7.5
    )
    
    result.images[0].save("example_5_content_style_split.png")
    print("Generated image saved as example_5_content_style_split.png")


def example_6_convenience_functions():
    """
    Example 6: Using convenience functions for quick injection.
    """
    print("=== Example 6: Convenience Functions ===")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    
    # Quick content injection
    modified_pipeline = inject_content_prompt(
        pipeline=pipeline,
        prompt="robot",
        model_type="sd15",  # or use detect_model_type(pipeline)
        weight=1.1
    )
    
    result = modified_pipeline(
        prompt="a person walking in the park",
        num_inference_steps=20,
        guidance_scale=7.5
    )
    
    result.images[0].save("example_6_convenience.png")
    print("Generated image saved as example_6_convenience.png")


def example_7_model_info():
    """
    Example 7: Getting model information and available blocks.
    """
    print("=== Example 7: Model Information ===")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Auto-detect model type
    model_type = detect_model_type(pipeline)
    print(f"Detected model type: {model_type}")
    
    # Get available blocks
    blocks = get_available_blocks(model_type)
    print(f"Available blocks for {model_type}:")
    for block_type, indices in blocks.items():
        print(f"  {block_type}: {indices}")
    
    # Create auto-configured injector
    injector = create_quick_injector(pipeline, "simple")
    print(f"Created injector: {type(injector).__name__}")


def run_all_examples():
    """
    Run all examples (warning: requires GPU and downloads models).
    """
    print("CorePulse Prompt Injection Examples")
    print("===================================")
    
    try:
        example_1_simple_injection()
        example_2_block_specific_injection() 
        example_3_advanced_multi_block()
        example_4_location_based()
        example_5_content_style_split()
        example_6_convenience_functions()
        example_7_model_info()
        
        print("\nAll examples completed successfully!")
        print("Check the generated PNG files to see the results.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("- A CUDA-capable GPU")
        print("- Sufficient GPU memory")
        print("- Internet connection for model downloads")


if __name__ == "__main__":
    # Run just the model info example by default (no downloads required)
    example_7_model_info()
    
    # Uncomment to run all examples (requires GPU and model downloads)
    # run_all_examples()
