"""
Content/Style Split Examples for CorePulse

This example demonstrates how to separate content and style using prompt injection.
Generate a cat with oil painting style in a photorealistic scene.
"""

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from core_pulse import SimplePromptInjector, MultiPromptInjector


def detect_model_type(pipeline):
    """Detect if pipeline is SDXL or SD1.5"""
    if hasattr(pipeline.unet.config, 'addition_embed_type'):
        return "SDXL"
    return "SD1.5"


def content_style_split_example():
    """
    Demonstrate content/style separation using prompt injection.
    
    Content blocks (middle) control WHAT appears in the image.
    Style blocks (output) control HOW it looks and feels.
    """
    print("Loading pipeline...")
    
    # Load appropriate model
    if torch.cuda.is_available():
        device = "cuda"
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to(device)
    else:
        device = "cpu"
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to(device)
    
    model_type = detect_model_type(pipeline)
    print(f"Detected model: {model_type}")
    
    # Base prompt provides the scene context
    base_prompt = "a domestic animal in a garden setting"
    
    print("\n=== Content/Style Split Example ===")
    print(f"Base prompt: '{base_prompt}'")
    print("Content injection: 'white persian cat'")
    print("Style injection: 'oil painting, impressionist brushstrokes'")
    
    with MultiPromptInjector(pipeline) as injector:
        # Add content/style split
        injector.add_content_style_split(
            content_prompt="white persian cat",  # What appears
            style_prompt="oil painting, impressionist brushstrokes",  # How it looks
            content_weight=2.0,  # Strong content override
            style_weight=1.5     # Moderate style influence
        )
        
        # Generate image
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024 if model_type == "SDXL" else 512,
            height=1024 if model_type == "SDXL" else 512
        )
        
        # Save result
        result.images[0].save("content_style_split_result.png")
        print("Saved: content_style_split_result.png")
        print("Result: White persian cat in garden with oil painting style")


def simple_injection_comparison():
    """
    Compare base generation vs prompt injection.
    """
    print("\n=== Simple Injection Comparison ===")
    
    # Load model
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
    
    base_prompt = "a dog playing in a park"
    
    # Generate baseline image
    print(f"Generating baseline: '{base_prompt}'")
    baseline = pipeline(
        prompt=base_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    baseline.images[0].save("baseline_generation.png")
    print("Saved: baseline_generation.png")
    
    # Generate with content injection
    print("Generating with content injection: 'golden retriever'")
    with SimplePromptInjector(pipeline) as injector:
        injector.configure_injections(
            block="middle:0",  # Content block
            prompt="golden retriever",
            weight=2.5,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        injected = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        injected.images[0].save("content_injected_generation.png")
        print("Saved: content_injected_generation.png")
        print("Result: Golden retriever playing in park (content replaced)")


def multi_block_injection_example():
    """
    Demonstrate injecting into multiple blocks with different weights.
    """
    print("\n=== Multi-Block Injection Example ===")
    
    # Load model
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
    
    base_prompt = "a building in a landscape"
    
    with SimplePromptInjector(pipeline) as injector:
        # Inject "medieval castle" into multiple blocks with different weights
        injections = [
            {"block": "input:4", "prompt": "medieval castle", "weight": 1.0, "sigma_start": 15.0, "sigma_end": 0.0},
            {"block": "middle:0", "prompt": "medieval castle", "weight": 2.5, "sigma_start": 15.0, "sigma_end": 0.0},
            {"block": "output:1", "prompt": "medieval castle", "weight": 1.5, "sigma_start": 15.0, "sigma_end": 0.0}
        ]
        
        injector.configure_injections(injections)
        
        print(f"Base prompt: '{base_prompt}'")
        print("Injecting 'medieval castle' into input:4 (1.0x), middle:0 (2.5x), output:1 (1.5x)")
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        result.images[0].save("multi_block_injection_result.png")
        print("Saved: multi_block_injection_result.png")
        print("Result: Medieval castle in landscape (cumulative block effects)")


if __name__ == "__main__":
    print("CorePulse Content/Style Split Examples")
    print("=" * 40)
    
    # Run examples
    content_style_split_example()
    simple_injection_comparison()
    multi_block_injection_example()
    
    print("\n" + "=" * 40)
    print("All examples completed!")
    print("Generated images:")
    print("- content_style_split_result.png")
    print("- baseline_generation.png")
    print("- content_injected_generation.png")
    print("- multi_block_injection_result.png")
