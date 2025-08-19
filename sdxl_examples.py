"""
CorePulse Example: SDXL Regional Scene Generation

This example demonstrates the advanced regional capabilities of CorePulse
with the Stable Diffusion XL (SDXL) model.

It showcases how to use the regional injection system to create
a complex scene with different subjects in distinct regions of the image,
all within a single generation pass.
"""

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline

# Import CorePulse components for the regional system
from core_pulse import RegionalPromptInjector
from core_pulse.prompt_injection.spatial import MaskFactory, create_left_half_mask, create_right_half_mask


def regional_injection_demonstration_sdxl():
    """
    Demonstrate regional injection with SDXL by creating a fantasy scene
    with a castle on the left and a dragon on the right.
    """
    print("=== CorePulse: SDXL Regional Injection Demonstration ===")
    
    # Load SDXL pipeline
    print("Loading Stable Diffusion XL...")
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load the SDXL pipeline. Do you have an internet connection and enough VRAM? Error: {e}")
        return

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    print(f"Using device: {device}")
    
    # Base prompt for the overall scene
    base_prompt = "a beautiful fantasy landscape, cinematic lighting, ultra-detailed"
    
    # Generation parameters
    num_inference_steps = 100
    guidance_scale = 8.0
    seed = 1024
    
    print(f"Using fixed seed: {seed} for reproducible comparison")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # 1. Generate original image (without regional injection)
    print(f"\n1. Generating original image: '{base_prompt}'")
    original_result = pipeline(
        prompt=base_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    original_image = original_result.images[0]
    
    # 2. Generate with regional injection
    print(f"\n2. Generating with REGIONAL injection...")
    print("   - Base: Fantasy landscape")
    print("   - Left Region: Crystal castle")
    print("   - Right Region: Fire-breathing dragon")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # NEW: Initialize injector with the pipeline
    injector = RegionalPromptInjector(pipeline)
    
    # Define the regions and their corresponding prompts
    
    # Left side: A crystal castle
    left_mask = create_left_half_mask(image_size=(1024, 1024))
    soft_left_mask = MaskFactory.gaussian_blur(left_mask, kernel_size=151, sigma=50.0)
    injector.add_regional_injection(
        block="all",  # Apply to all relevant blocks
        prompt="a glowing crystal castle, majestic, intricate details",
        mask=soft_left_mask,
        weight=1.0
    )

    # Right side: A dragon
    right_mask = create_right_half_mask(image_size=(1024, 1024))
    soft_right_mask = MaskFactory.gaussian_blur(right_mask, kernel_size=151, sigma=50.0)
    injector.add_regional_injection(
        block="all",
        prompt="a giant fire-breathing dragon, scales shimmering",
        mask=soft_right_mask,
        weight=1.0
    )

    with injector:
        injector.apply_to_pipeline(pipeline)
        
        # The base_prompt provides the overall context.
        # The regional injections will override their specific areas.
        injected_result = injector(
            prompt=base_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        injected_image = injected_result.images[0]
    
    print("\n3. Creating side-by-side comparison...")
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(original_image)
    ax1.set_title(f"Original Prompt\n'{base_prompt}'", fontsize=12)
    ax1.axis('off')
    
    # Injected image  
    ax2.imshow(injected_image)
    ax2.set_title(f"Compositional Injection\nLeft: Castle | Right: Dragon", fontsize=12)
    ax2.axis('off')
    
    fig.suptitle("CorePulse SDXL Compositional Injection", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("media/sdxl_composition_comparison.png", dpi=150, bbox_inches='tight')
    
    print("Comparison saved as 'media/sdxl_composition_comparison.png'")
    
    original_image.save("media/sdxl_original_landscape.png")
    injected_image.save("media/sdxl_composed_scene.png")
    print("Individual images saved in media/ directory")
    
    plt.show()


if __name__ == "__main__":
    regional_injection_demonstration_sdxl()
