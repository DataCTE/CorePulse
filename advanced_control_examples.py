"""
Multi-Scale Architecture Examples for CorePulse

This example demonstrates independent control of structure and details using
multi-scale prompt injection at different resolution levels.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from core_pulse import MultiScaleInjector


def structure_detail_separation_example():
    """
    Classic example: Medieval fortress structure with weathered stone details.
    
    Structure level controls WHAT the overall building looks like.
    Detail level controls HOW the surfaces appear.
    """
    print("Loading SDXL pipeline...")
    print("Note: Multi-scale control requires SDXL architecture")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "a building in a misty landscape"
    
    print("\n=== Structure/Detail Separation Example ===")
    print(f"Base prompt: '{base_prompt}'")
    
    # Generate baseline
    print("\n1. Generating baseline (no multi-scale control)...")
    baseline = pipeline(
        prompt=base_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("multiscale_baseline.png")
    print("Saved: multiscale_baseline.png")
    
    # Generate with structure + detail control
    print("\n2. Generating with multi-scale control...")
    print("   Structure: 'gothic cathedral silhouette, imposing architecture'")
    print("   Details: 'weathered stone, intricate carvings, moss-covered surfaces'")
    
    with MultiScaleInjector(pipeline) as injector:
        # Structure: Overall composition and major forms
        injector.add_structure_injection(
            "gothic cathedral silhouette, imposing architecture",
            weight=2.0
        )
        
        # Details: Surface textures and fine elements
        injector.add_detail_injection(
            "weathered stone, intricate carvings, moss-covered surfaces",
            weight=1.8
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_structure_detail.png")
        print("Saved: multiscale_structure_detail.png")
        print("Result: Gothic cathedral structure with weathered stone details")


def hierarchical_control_example():
    """
    Demonstrate hierarchical control across all resolution levels.
    
    Structure → Mid-Level → Detail cascade for complex compositions.
    """
    print("\n=== Hierarchical Control Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "a fantasy scene with magical elements"
    
    print(f"Base prompt: '{base_prompt}'")
    
    print("\n1. Hierarchical multi-scale injection:")
    print("   Structure: 'massive crystal formation, towering spires'")
    print("   Mid-Level: 'glowing magical runes, energy flowing between crystals'")  
    print("   Detail: 'prismatic reflections, sparkling gem surfaces'")
    
    with MultiScaleInjector(pipeline) as injector:
        # Structure level: Overall composition
        injector.add_structure_injection(
            "massive crystal formation, towering spires",
            weight=2.2
        )
        
        # Mid level: Regional features and relationships
        injector.add_mid_injection(
            "glowing magical runes, energy flowing between crystals",
            weight=1.8
        )
        
        # Detail level: Surface textures and fine elements
        injector.add_detail_injection(
            "prismatic reflections, sparkling gem surfaces",
            weight=2.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_hierarchical.png")
        print("Saved: multiscale_hierarchical.png")
        print("Result: Crystal spires with magical runes and prismatic details")


def structure_only_vs_detail_only_example():
    """
    Compare structure-only vs detail-only vs combined control.
    """
    print("\n=== Structure vs Detail Comparison ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "an ancient temple in a forest clearing"
    
    print(f"Base prompt: '{base_prompt}'")
    
    # Structure only
    print("\n1. Structure only: 'pyramid temple, stepped architecture'")
    with MultiScaleInjector(pipeline) as injector:
        injector.add_structure_injection(
            "pyramid temple, stepped architecture",
            weight=2.5
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_structure_only.png")
        print("Saved: multiscale_structure_only.png")
    
    # Detail only
    print("\n2. Detail only: 'vine-covered stone, ancient carved reliefs'")
    with MultiScaleInjector(pipeline) as injector:
        injector.add_detail_injection(
            "vine-covered stone, ancient carved reliefs",
            weight=2.5
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_detail_only.png")
        print("Saved: multiscale_detail_only.png")
    
    # Combined
    print("\n3. Combined: Structure + Detail together")
    with MultiScaleInjector(pipeline) as injector:
        injector.add_structure_injection(
            "pyramid temple, stepped architecture",
            weight=2.5
        )
        
        injector.add_detail_injection(
            "vine-covered stone, ancient carved reliefs",
            weight=2.5
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_combined.png")
        print("Saved: multiscale_combined.png")
    
    print("Result: Comparison of structure-only, detail-only, and combined effects")


def compatible_vs_conflicting_prompts_example():
    """
    Demonstrate the importance of semantically compatible prompts.
    """
    print("\n=== Compatible vs Conflicting Prompts ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "an architectural structure in a landscape"
    
    print(f"Base prompt: '{base_prompt}'")
    
    # Compatible prompts
    print("\n1. Compatible prompts:")
    print("   Structure: 'gothic cathedral' → Detail: 'stone masonry'")
    with MultiScaleInjector(pipeline) as injector:
        injector.add_structure_injection(
            "gothic cathedral, towering arches",
            weight=2.0
        )
        
        injector.add_detail_injection(
            "stone masonry, carved gargoyles",
            weight=1.8
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_compatible_prompts.png")
        print("Saved: multiscale_compatible_prompts.png")
    
    # Conflicting prompts
    print("\n2. Conflicting prompts:")
    print("   Structure: 'modern skyscraper' → Detail: 'organic tree bark'")
    with MultiScaleInjector(pipeline) as injector:
        injector.add_structure_injection(
            "modern skyscraper, glass and steel",
            weight=2.0
        )
        
        injector.add_detail_injection(
            "organic tree bark, natural wood grain",
            weight=1.8
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_conflicting_prompts.png")
        print("Saved: multiscale_conflicting_prompts.png")
    
    print("Result: Compatible prompts create coherent images, conflicting ones create chaos")


def sigma_range_timing_example():
    """
    Demonstrate the effect of different sigma ranges for multi-scale control.
    """
    print("\n=== Sigma Range Timing Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    base_prompt = "a castle on a hill"
    
    print(f"Base prompt: '{base_prompt}'")
    print("\nTesting different sigma timing strategies...")
    
    # Early structure, late details (recommended)
    print("\n1. Early structure (15.0→0.5), late details (3.0→0.0)")
    with MultiScaleInjector(pipeline) as injector:
        injector.add_structure_injection(
            "fairy tale castle, multiple towers",
            weight=2.0,
            sigma_start=15.0,
            sigma_end=0.5
        )
        
        injector.add_detail_injection(
            "ivy-covered walls, ornate windows",
            weight=1.8,
            sigma_start=3.0,
            sigma_end=0.0
        )
        
        result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        result.images[0].save("multiscale_optimal_timing.png")
        print("Saved: multiscale_optimal_timing.png")
    
    # Both throughout (may conflict)
    print("\n2. Both throughout entire process (15.0→0.0)")
    with MultiScaleInjector(pipeline) as injector:
        injector.add_structure_injection(
            "fairy tale castle, multiple towers",
            weight=2.0,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        injector.add_detail_injection(
            "ivy-covered walls, ornate windows",
            weight=1.8,
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
        result.images[0].save("multiscale_overlapping_timing.png")
        print("Saved: multiscale_overlapping_timing.png")
    
    print("Result: Proper sigma timing prevents conflicts between structure and detail")


if __name__ == "__main__":
    print("CorePulse Multi-Scale Architecture Examples")
    print("=" * 50)
    print("Note: All examples require SDXL model for multi-scale control")
    
    # Run examples
    structure_detail_separation_example()
    hierarchical_control_example()
    structure_only_vs_detail_only_example()
    compatible_vs_conflicting_prompts_example()
    sigma_range_timing_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Generated images:")
    print("- multiscale_baseline.png")
    print("- multiscale_structure_detail.png")
    print("- multiscale_hierarchical.png")
    print("- multiscale_structure_only.png")
    print("- multiscale_detail_only.png")
    print("- multiscale_combined.png")
    print("- multiscale_compatible_prompts.png")
    print("- multiscale_conflicting_prompts.png")
    print("- multiscale_optimal_timing.png")
    print("- multiscale_overlapping_timing.png")
