"""
SDXL Regional Control Examples for CorePulse

This example demonstrates regional control specific to SDXL,
including left/right splits with soft blending and complex compositions.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from core_pulse import RegionalPromptInjector, MultiScaleInjector
from core_pulse.prompt_injection.spatial import (
    create_left_half_mask,
    create_right_half_mask, 
    create_top_half_mask,
    create_bottom_half_mask,
    create_center_circle_mask
)


def crystal_castle_fire_dragon_example():
    """
    Classic SDXL regional example: Crystal castle on left, fire dragon on right.
    
    Demonstrates spatial masks with soft blending for smooth transitions.
    """
    print("Loading SDXL pipeline...")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "a fantasy landscape with magical elements"
    
    print("\n=== Crystal Castle + Fire Dragon Example ===")
    print(f"Base prompt: '{base_prompt}'")
    
    # Generate baseline
    print("\n1. Generating baseline...")
    baseline = pipeline(
        prompt=base_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("sdxl_baseline.png")
    print("Saved: sdxl_baseline.png")
    
    # Generate with left/right split
    print("\n2. Left half: 'crystal castle', Right half: 'fire dragon'...")
    with RegionalPromptInjector(pipeline) as injector:
        # Left half - crystal castle
        left_mask = create_left_half_mask(image_size=(1024, 1024))
        injector.add_regional_injection(
            block="middle:0",
            prompt="crystal castle with magical towers, ice and snow",
            mask=left_mask,
            weight=2.5,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        # Right half - fire dragon
        right_mask = create_right_half_mask(image_size=(1024, 1024))
        injector.add_regional_injection(
            block="middle:0",
            prompt="fire dragon breathing flames, lava and heat",
            mask=right_mask,
            weight=2.5,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("sdxl_left_castle_right_dragon.png")
        print("Saved: sdxl_left_castle_right_dragon.png")
        print("Result: Ice castle blending with fire dragon across the scene")


def layered_composition_example():
    """
    Demonstrate layered regional injections in SDXL for complex compositions.
    """
    print("\n=== Layered Composition Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "an epic fantasy scene with multiple realms"
    
    print(f"Base prompt: '{base_prompt}'")
    print("Creating layered composition:")
    print("- Top: Sky realm with floating islands")
    print("- Center: Ground realm with forests")  
    print("- Blend circle: Portal with magical energy")
    
    with RegionalPromptInjector(pipeline) as injector:
        # Top region - sky realm
        top_mask = create_top_half_mask(image_size=(1024, 1024))
        injector.add_regional_injection(
            block="input:4",  # Early composition
            prompt="floating islands in cloudy sky, celestial realm",
            mask=top_mask,
            weight=2.0,
            sigma_start=15.0,
            sigma_end=2.0
        )
        
        # Bottom region - ground realm
        bottom_mask = create_bottom_half_mask(image_size=(1024, 1024))
        injector.add_regional_injection(
            block="input:4",
            prompt="dense forest with ancient trees, earthly realm",
            mask=bottom_mask,
            weight=2.0,
            sigma_start=15.0,
            sigma_end=2.0
        )
        
        # Center circle - magical portal
        portal_mask = create_center_circle_mask(image_size=(1024, 1024), radius=200)
        injector.add_regional_injection(
            block="middle:0",  # Content layer
            prompt="swirling magical portal, energy vortex, glowing runes",
            mask=portal_mask,
            weight=3.0,
            sigma_start=10.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("sdxl_layered_composition.png")
        print("Saved: sdxl_layered_composition.png")
        print("Result: Sky islands above, forest below, magical portal connecting them")


def regional_multi_scale_combination():
    """
    Advanced example: Combine regional injection with multi-scale control in SDXL.
    """
    print("\n=== Regional + Multi-Scale Combination ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "an ancient civilization scene with architecture and nature"
    
    print(f"Base prompt: '{base_prompt}'")
    print("Combining techniques:")
    print("- Multi-scale: Global structure + surface details")
    print("- Regional: Left side architecture, right side nature")
    
    # First apply multi-scale control globally
    with MultiScaleInjector(pipeline) as multi_injector:
        # Global structure control
        multi_injector.add_structure_injection(
            "ancient ruins with towering columns",
            weight=1.8
        )
        
        # Global detail control 
        multi_injector.add_detail_injection(
            "weathered stone textures, moss and vines",
            weight=1.5
        )
        
        # Then add regional control on top
        with RegionalPromptInjector(multi_injector.pipeline) as regional_injector:
            # Left side - emphasize architecture
            left_mask = create_left_half_mask(image_size=(1024, 1024))
            regional_injector.add_regional_injection(
                block="output:1",  # Style layer
                prompt="grand temple architecture, marble columns",
                mask=left_mask,
                weight=2.0,
                sigma_start=8.0,
                sigma_end=0.0
            )
            
            # Right side - emphasize nature
            right_mask = create_right_half_mask(image_size=(1024, 1024))
            regional_injector.add_regional_injection(
                block="output:1",
                prompt="overgrown jungle, tropical vegetation",
                mask=right_mask,
                weight=2.0,
                sigma_start=8.0,
                sigma_end=0.0
            )
            
            result = regional_injector(
                prompt=base_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024
            )
            result.images[0].save("sdxl_combined_techniques.png")
            print("Saved: sdxl_combined_techniques.png")
            print("Result: Ancient ruins with architectural left side, natural right side")


def soft_blending_showcase():
    """
    Showcase the soft blending capabilities of SDXL regional control.
    """
    print("\n=== Soft Blending Showcase ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "a transitional landscape scene"
    
    print(f"Base prompt: '{base_prompt}'")
    print("Demonstrating soft blending between contrasting elements...")
    
    with RegionalPromptInjector(pipeline) as injector:
        # Use overlapping masks for smooth transitions
        left_mask = create_left_half_mask(image_size=(1024, 1024))
        right_mask = create_right_half_mask(image_size=(1024, 1024))
        
        # Left: Winter scene
        injector.add_regional_injection(
            block="middle:0",
            prompt="snow-covered mountains, frozen lake, winter atmosphere",
            mask=left_mask,
            weight=2.2,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        # Right: Summer scene  
        injector.add_regional_injection(
            block="middle:0", 
            prompt="tropical beach, palm trees, sunny warm atmosphere",
            mask=right_mask,
            weight=2.2,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("sdxl_soft_blending.png")
        print("Saved: sdxl_soft_blending.png")
        print("Result: Smooth transition from winter mountains to tropical beach")


def architectural_styles_comparison():
    """
    Compare different architectural styles in regional sections.
    """
    print("\n=== Architectural Styles Comparison ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "a city with diverse architectural styles"
    
    print(f"Base prompt: '{base_prompt}'")
    print("Creating architectural style comparison:")
    print("- Left: Gothic cathedral architecture")
    print("- Right: Modern futuristic buildings")
    
    with RegionalPromptInjector(pipeline) as injector:
        # Left - Gothic architecture
        left_mask = create_left_half_mask(image_size=(1024, 1024))
        injector.add_regional_injection(
            block="middle:0",
            prompt="gothic cathedral, flying buttresses, stone gargoyles, medieval architecture",
            mask=left_mask,
            weight=2.5,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        # Right - Modern architecture
        right_mask = create_right_half_mask(image_size=(1024, 1024))
        injector.add_regional_injection(
            block="middle:0",
            prompt="futuristic skyscrapers, glass and steel, modern minimalist design",
            mask=right_mask,
            weight=2.5,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("sdxl_architectural_comparison.png")
        print("Saved: sdxl_architectural_comparison.png")
        print("Result: Medieval gothic architecture transitioning to modern buildings")


if __name__ == "__main__":
    print("CorePulse SDXL Regional Control Examples")
    print("=" * 50)
    print("Note: All examples require SDXL model")
    
    # Run examples
    crystal_castle_fire_dragon_example()
    layered_composition_example()
    regional_multi_scale_combination()
    soft_blending_showcase()
    architectural_styles_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Generated images:")
    print("- sdxl_baseline.png")
    print("- sdxl_left_castle_right_dragon.png")
    print("- sdxl_layered_composition.png")
    print("- sdxl_combined_techniques.png")
    print("- sdxl_soft_blending.png")
    print("- sdxl_architectural_comparison.png")
