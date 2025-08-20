"""
Regional/Spatial Injection Examples for CorePulse

This example demonstrates surgical region replacement using spatial masks.
Change specific image areas while keeping context intact.
"""

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from core_pulse import RegionalPromptInjector
from core_pulse.prompt_injection.spatial import (
    create_center_circle_mask,
    create_rectangle_mask,
    create_left_half_mask,
    create_top_half_mask,
    create_center_square_mask
)


def detect_model_type(pipeline):
    """Detect if pipeline is SDXL or SD1.5"""
    if hasattr(pipeline.unet.config, 'addition_embed_type'):
        return "SDXL"
    return "SD1.5"


def center_region_replacement_example():
    """
    Demonstrate replacing the center region while preserving surroundings.
    
    Classic example: Change a cat to a dog in the center, keep park untouched.
    """
    print("Loading pipeline...")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        image_size = (1024, 1024)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
        image_size = (512, 512)
    
    model_type = detect_model_type(pipeline)
    print(f"Detected model: {model_type}")
    
    base_prompt = "a cat playing at a beautiful park with trees and flowers"
    
    print("\n=== Center Region Replacement Example ===")
    print(f"Base prompt: '{base_prompt}'")
    
    # Generate baseline
    print("\n1. Generating baseline (no spatial injection)...")
    baseline = pipeline(
        prompt=base_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=image_size[0],
        height=image_size[1]
    )
    baseline.images[0].save("spatial_baseline.png")
    print("Saved: spatial_baseline.png")
    
    # Create center circle mask
    mask = create_center_circle_mask(image_size=image_size, radius=200)
    print(f"\n2. Created center circle mask (radius=200)")
    
    # Generate with center region replacement
    print("3. Generating with center region replaced with 'golden retriever dog'...")
    with RegionalPromptInjector(pipeline) as injector:
        injector.add_regional_injection(
            block="middle:0",
            prompt="golden retriever dog",  # Replace center with this
            mask=mask,
            weight=2.5,  # Strong replacement
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=image_size[0],
            height=image_size[1]
        )
        result.images[0].save("spatial_center_dog.png")
        print("Saved: spatial_center_dog.png")
        print("Result: Golden retriever in center, park environment preserved")


def left_right_split_example():
    """
    Demonstrate left/right split with different content.
    """
    print("\n=== Left/Right Split Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        image_size = (1024, 1024)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
        image_size = (512, 512)
    
    base_prompt = "a landscape scene with interesting features"
    
    print(f"Base prompt: '{base_prompt}'")
    
    # Create left half mask
    left_mask = create_left_half_mask(image_size=image_size)
    
    print("\n1. Generating with left side = 'crystal castle', right side = natural...")
    with RegionalPromptInjector(pipeline) as injector:
        injector.add_regional_injection(
            block="middle:0",
            prompt="crystal castle with magical towers",
            mask=left_mask,
            weight=2.0,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=image_size[0],
            height=image_size[1]
        )
        result.images[0].save("spatial_left_crystal_castle.png")
        print("Saved: spatial_left_crystal_castle.png")
        print("Result: Crystal castle on left, natural landscape on right")


def multiple_regions_example():
    """
    Demonstrate multiple regional injections in the same image.
    """
    print("\n=== Multiple Regions Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        image_size = (1024, 1024)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
        image_size = (512, 512)
    
    base_prompt = "a fantasy realm with various magical elements"
    
    print(f"Base prompt: '{base_prompt}'")
    
    # Create multiple masks
    top_half_mask = create_top_half_mask(image_size=image_size)
    center_square_mask = create_center_square_mask(image_size=image_size, size=300)
    
    print("\n1. Generating with top = 'sky dragons', center = 'crystal tower'...")
    with RegionalPromptInjector(pipeline) as injector:
        # Top region: sky dragons
        injector.add_regional_injection(
            block="middle:0",
            prompt="dragons flying in cloudy sky",
            mask=top_half_mask,
            weight=1.8,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        # Center region: crystal tower (will blend with top region)
        injector.add_regional_injection(
            block="output:1", 
            prompt="crystal tower with glowing runes",
            mask=center_square_mask,
            weight=2.2,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=image_size[0],
            height=image_size[1]
        )
        result.images[0].save("spatial_multiple_regions.png")
        print("Saved: spatial_multiple_regions.png")
        print("Result: Sky dragons at top, crystal tower in center, natural base")


def custom_rectangle_example():
    """
    Demonstrate custom rectangular region injection.
    """
    print("\n=== Custom Rectangle Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        image_size = (1024, 1024)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
        image_size = (512, 512)
    
    base_prompt = "a peaceful village scene with buildings and nature"
    
    print(f"Base prompt: '{base_prompt}'")
    
    # Create custom rectangle mask (upper right corner)
    rect_mask = create_rectangle_mask(
        image_size=image_size,
        x1=int(image_size[0] * 0.6),  # 60% from left
        y1=int(image_size[1] * 0.1),  # 10% from top
        x2=int(image_size[0] * 0.9),  # 90% from left
        y2=int(image_size[1] * 0.4)   # 40% from top
    )
    
    print("\n1. Generating with upper-right rectangle = 'fire dragon'...")
    with RegionalPromptInjector(pipeline) as injector:
        injector.add_regional_injection(
            block="middle:0",
            prompt="fire dragon breathing flames",
            mask=rect_mask,
            weight=3.0,  # Strong injection for dramatic effect
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=image_size[0],
            height=image_size[1]
        )
        result.images[0].save("spatial_rectangle_dragon.png")
        print("Saved: spatial_rectangle_dragon.png")
        print("Result: Peaceful village with fire dragon in upper-right corner")


def mask_comparison_example():
    """
    Generate the same scene with different mask shapes for comparison.
    """
    print("\n=== Mask Shape Comparison Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        image_size = (1024, 1024)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
        image_size = (512, 512)
    
    base_prompt = "a garden scene with flowers and grass"
    injection_prompt = "giant mushroom with glowing spots"
    
    print(f"Base prompt: '{base_prompt}'")
    print(f"Injection prompt: '{injection_prompt}'")
    
    # Different mask shapes
    masks = {
        "circle": create_center_circle_mask(image_size, radius=250),
        "square": create_center_square_mask(image_size, size=400),
        "left_half": create_left_half_mask(image_size)
    }
    
    for mask_name, mask in masks.items():
        print(f"\n{mask_name.title()} mask generation...")
        
        with RegionalPromptInjector(pipeline) as injector:
            injector.add_regional_injection(
                block="middle:0",
                prompt=injection_prompt,
                mask=mask,
                weight=2.5,
                sigma_start=15.0,
                sigma_end=0.0
            )
            
            result = injector(
                prompt=base_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=image_size[0],
                height=image_size[1]
            )
            
            filename = f"spatial_mask_{mask_name}.png"
            result.images[0].save(filename)
            print(f"Saved: {filename}")
    
    print("Result: Same mushroom injection with different mask shapes")


if __name__ == "__main__":
    print("CorePulse Regional/Spatial Injection Examples")
    print("=" * 50)
    
    # Run examples
    center_region_replacement_example()
    left_right_split_example()
    multiple_regions_example()
    custom_rectangle_example()
    mask_comparison_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Generated images:")
    print("- spatial_baseline.png")
    print("- spatial_center_dog.png")
    print("- spatial_left_crystal_castle.png")
    print("- spatial_multiple_regions.png")
    print("- spatial_rectangle_dragon.png")
    print("- spatial_mask_circle.png")
    print("- spatial_mask_square.png")
    print("- spatial_mask_left_half.png")
