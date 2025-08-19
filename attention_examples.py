"""
CorePulse Example: Attention Map Manipulation
"""

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline

from core_pulse.prompt_injection.attention import AttentionMapInjector
from core_pulse.prompt_injection.spatial import create_rectangle_mask
from core_pulse.utils.logger import set_core_pulse_debug_level

def attention_manipulation_example():
    """
    Demonstrates the use of AttentionMapInjector to fine-tune image generation.
    """
    print("=== CorePulse: Attention Map Manipulation Example ===")
    
    # Enable debug logging
    set_core_pulse_debug_level('debug')
    
    # Load SDXL pipeline
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load the SDXL pipeline: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    print(f"Using device: {device}")
    
    base_prompt = "A photorealistic portrait of an astronaut"
    seed = 98765
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # 1. Generate original image
    print(f"Generating original image: '{base_prompt}'")
    original_image = pipeline(prompt=base_prompt, generator=generator, num_inference_steps=30).images[0]
    
    # --- Create a single injector for all manipulations ---
    injector = AttentionMapInjector(pipeline)

    # 2. Generate with increased attention on "photorealistic"
    print("Generating with increased attention on 'photorealistic'...")
    generator = torch.Generator(device=device).manual_seed(seed)
    injector.add_attention_manipulation(
        prompt=base_prompt,
        block="all",
        target_phrase="photorealistic",
        attention_scale=5.0, # Increased from 2.0
        sigma_start=14.0,
        sigma_end=0.1
    )
    with injector:
        boosted_image = injector(prompt=base_prompt, generator=generator, num_inference_steps=30).images[0]
        
    # Clear previous manipulations before adding new ones
    injector.clear_injections()

    # 3. Generate with reduced attention on "astronaut" in the background (layered on top)
    print("Generating with reduced attention on 'astronaut' in the background...")
    generator = torch.Generator(device=device).manual_seed(seed)
    background_mask = create_rectangle_mask(0, 0, 1024, 512, (1024, 1024)) # Top half
    injector.add_attention_manipulation(
        prompt=base_prompt,
        block="all",
        target_phrase="astronaut",
        attention_scale=0.05, # Decreased from 0.1
        spatial_mask=background_mask,
        sigma_start=14.0,
        sigma_end=0.1
    )
    with injector:
        reduced_image = injector(prompt=base_prompt, generator=generator, num_inference_steps=30).images[0]

    # Create comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    ax1.imshow(original_image)
    ax1.set_title("Original")
    ax2.imshow(boosted_image)
    ax2.set_title("Attention Boosted on 'photorealistic'")
    ax3.imshow(reduced_image)
    ax3.set_title("Attention Reduced on 'astronaut' (Top Half)")
    plt.tight_layout()
    plt.savefig("media/attention_manipulation_comparison.png")
    print("Comparison saved to media/attention_manipulation_comparison.png")

if __name__ == "__main__":
    attention_manipulation_example()
