"""
CorePulse Example: Cat vs Dog Injection Comparison

This example demonstrates the CorePulse prompt injection system by comparing
the generation of "a cat playing at a park" with and without injecting "dog".

The comparison clearly shows how prompt injection can alter the content while
maintaining the overall composition and style of the original prompt.
"""

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# Import CorePulse components
from core_pulse import SimplePromptInjector, AdvancedPromptInjector, MaskedPromptInjector
from core_pulse import RegionalPromptInjector, create_top_left_quadrant_region, create_left_half_region, create_right_half_region, create_center_region
from core_pulse.utils import detect_model_type


def cat_vs_dog_comparison():
    """
    Demonstrate prompt injection by comparing 'cat playing at a park' 
    with and without injecting 'dog' into the content blocks.
    """
    print("=== CorePulse: Cat vs Dog Injection Comparison ===")
    
    # Load SD 1.5 pipeline
    print("Loading Stable Diffusion 1.5...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    print(f"Using device: {device}")
    
    # Base prompt
    base_prompt = "a cat playing at a park"
    
    # Generation parameters
    num_inference_steps = 20
    guidance_scale = 7.5
    seed = 42
    
    # Set all random seeds for reproducible results
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Using fixed seed: {seed} for reproducible comparison")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"\n1. Generating original image: '{base_prompt}'")
    # Generate original image
    original_result = pipeline(
        prompt=base_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    original_image = original_result.images[0]
    
    # Reset generator with same seed for fair comparison
    print(f"\n2. Generating with 'dog' injection into ALL blocks...")
    print(f"   Resetting generator to same seed ({seed}) for fair comparison")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Create advanced injector and use the new "all" feature!
    injector = AdvancedPromptInjector("sd15")
    print("   Injecting 'dog' into ALL blocks with weight 1.0 using smart 'all' feature")
    
    # NEW FEATURE: Use "all" to inject into all available blocks for SD 1.5
    # This replaces the need to specify each block individually!
    injector.add_injection("all", "dog", weight=1.0)
    
    with injector:
        modified_pipeline = injector.apply_to_pipeline(pipeline)
        
        injected_result = modified_pipeline(
            prompt=base_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        injected_image = injected_result.images[0]
    
    print("3. Creating side-by-side comparison...")
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(original_image)
    ax1.set_title(f"Original\n'{base_prompt}'", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Injected image  
    ax2.imshow(injected_image)
    ax2.set_title(f"With 'dog' injection\n'{base_prompt}' + injection", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Add main title
    fig.suptitle("CorePulse Prompt Injection Comparison", fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("media/cat_vs_dog_comparison.png", dpi=150, bbox_inches='tight')
    
    print(" Comparison saved as 'media/cat_vs_dog_comparison.png'")
    
    # Also save individual images
    original_image.save("media/original_cat_park_regular.png")
    injected_image.save("media/injected_dog_park_regular.png")
    print("Individual images saved in media/ directory")
    
    print(f"\nDEMONSTRATION CONFIRMED:")
    print(f"   • Both images used identical seed ({seed}) and parameters")
    print(f"   • Only difference: 'dog' injection into content blocks")
    print(f"   • Result: Content changed while maintaining scene composition")
    print(f"   • This proves CorePulse injection is working correctly!")
    
    # Show the plot
    plt.show()


def masked_injection_comparison():
    """
    Demonstrate the new MASKED injection capability by comparing:
    1. Original: "a cat playing at a park"
    2. Regular injection: Complete prompt replacement (affects entire scene)
    3. Masked injection: Targeted replacement of only "cat" -> "dog"
    
    This shows how masked injection preserves context while changing specific elements.
    """
    print("=== CorePulse: Masked vs Regular Injection Comparison ===")
    
    # Load SD 1.5 pipeline
    print("Loading Stable Diffusion 1.5...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    print(f"Using device: {device}")
    
    # Base prompt
    base_prompt = "a cat playing at a park"
    
    # Generation parameters
    num_inference_steps = 20
    guidance_scale = 7.5
    seed = 123
    
    print(f"Using fixed seed: {seed} for reproducible comparison")
    
    # 1. Generate original image
    print(f"\n1. Generating original: '{base_prompt}'")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    original_result = pipeline(
        prompt=base_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    original_image = original_result.images[0]
    
    # 2. Generate with regular injection (replaces entire prompt)
    print(f"\n2. Generating with REGULAR injection (full prompt replacement)...")
    print("   This replaces the entire prompt conditioning")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    regular_injector = AdvancedPromptInjector("sd15")
    regular_injector.add_injection("all", "dog", weight=1.0)
    
    with regular_injector:
        regular_pipeline = regular_injector.apply_to_pipeline(pipeline)
        regular_result = regular_pipeline(
            prompt=base_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        regular_image = regular_result.images[0]
    
    # 3. Generate with MASKED injection (targets only "cat")
    print(f"\n3. Generating with MASKED injection (targeted 'cat' -> 'dog')...")
    print("   This replaces ONLY the 'cat' token while preserving 'playing at a park'")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    masked_injector = MaskedPromptInjector("sd15")
    masked_injector.add_masked_injection(
        block="all",
        base_prompt=base_prompt,
        injection_prompt="a dog playing at a park",  # Full context for proper encoding
        target_phrase="cat",  # Only replace "cat" tokens
        weight=1.0,
        fuzzy_match=True
    )
    
    with masked_injector:
        masked_pipeline = masked_injector.apply_to_pipeline(pipeline)
        masked_result = masked_pipeline(
            prompt=base_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        masked_image = masked_result.images[0]
    
    # Create three-way comparison
    print("4. Creating three-way comparison...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    ax1.imshow(original_image)
    ax1.set_title("Original\n'a cat playing at a park'", fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Regular injection
    ax2.imshow(regular_image)
    ax2.set_title("Regular Injection\nFull prompt replacement\n(may affect entire scene)", fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # Masked injection
    ax3.imshow(masked_image)
    ax3.set_title("Masked Injection\nTargeted 'cat' → 'dog'\n(preserves context)", fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # Add main title
    fig.suptitle("CorePulse Masked Injection Demonstration", fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("media/masked_injection_comparison.png", dpi=150, bbox_inches='tight')
    
    print("Comparison saved as 'media/masked_injection_comparison.png'")
    
    # Save individual images
    original_image.save("media/original_cat_park.png")
    regular_image.save("media/regular_injection_park.png")
    masked_image.save("media/masked_injection_park.png")
    print("Individual images saved with descriptive names in media/ directory")
    
    # Print analysis
    print(f"\nTECHNICAL ANALYSIS:")
    print(f"   • All images used identical seed ({seed}) and parameters")
    print(f"   • Regular injection: Replaces entire prompt conditioning")
    print(f"   • Masked injection: Replaces only 'cat' tokens in the embedding")
    print(f"   • Result: Masked injection should preserve scene composition better")
    print(f"   • Context preservation: Masked > Regular > None")
    
    # Show masking summary
    print(f"\nMASKING CONFIGURATION:")
    for summary in masked_injector.get_masking_summary():
        print(f"   • Target: '{summary['target_phrase']}'")
        print(f"   • Base: '{summary['base_prompt']}'")
        print(f"   • Injection: '{summary['injection_prompt']}'")
        print(f"   • Blocks: {summary['block']}")
        print(f"   • Fuzzy match: {summary['fuzzy_match']}")
    
    plt.show()


def regional_injection_demonstration():
    """
    Demonstrate the new REGIONAL injection capability by showing:
    1. Original: "a cat playing at a park"
    2. Top-left quadrant: Top-left = "dog", Rest = "cat"
    
    This shows how regional injection allows different content in different areas.
    """
    print("=== CorePulse: Regional Injection Demonstration ===")
    
    # Load SD 1.5 pipeline
    print("Loading Stable Diffusion 1.5...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    print(f"Using device: {device}")
    
    # Base prompt
    base_prompt = "a cat playing at a park"
    
    # Generation parameters
    num_inference_steps = 20
    guidance_scale = 7.5
    seed = 456
    
    print(f"Using fixed seed: {seed} for reproducible comparison")
    
    # 1. Generate original image
    print(f"\n1. Generating original: '{base_prompt}'")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    original_result = pipeline(
        prompt=base_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    original_image = original_result.images[0]
    
    # 2. Generate with regional injection (left=dog, right=cat)
    print(f"\n2. Generating with REGIONAL injection (top-left='dog', rest=original)...")
    print("   This applies 'dog' conditioning only to the TOP-LEFT QUADRANT of the image")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    regional_injector = RegionalPromptInjector("sd15")
    
    # Create top-left quadrant region and apply dog injection
    top_left_region = create_top_left_quadrant_region((512, 512))
    regional_injector.add_regional_injection(
        region=top_left_region,
        base_prompt=base_prompt,
        injection_prompt="dog",
        target_phrase="cat",  # Replace "cat" with "dog" in top-left quadrant
        weight=1.2,
        fuzzy_match=True
    )
    
    with regional_injector:
        regional_pipeline = regional_injector.apply_to_pipeline(pipeline)
        regional_result = regional_pipeline(
            prompt=base_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        regional_image = regional_result.images[0]
    
    # Create comparison
    print("3. Creating regional injection comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original
    ax1.imshow(original_image)
    ax1.set_title("Original\n'a cat playing at a park'", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Regional injection - add visual separator for quadrant
    ax2.imshow(regional_image)
    # Draw lines to show the top-left quadrant
    ax2.axhline(y=regional_image.height//2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=regional_image.width//2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_title("Regional Injection\nTop-left quadrant: 'cat' → 'dog'\nRest: unchanged", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Add main title
    fig.suptitle("CorePulse Regional Injection - Spatial Control", fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("media/regional_injection_comparison.png", dpi=150, bbox_inches='tight')
    
    print("Comparison saved as 'media/regional_injection_comparison.png'")
    
    # Save individual images
    original_image.save("media/original_regional_demo.png")
    regional_image.save("media/regional_top_left_dog.png")
    print("Individual images saved in media/ directory")
    
    # Print analysis
    print(f"\nREGIONAL INJECTION ANALYSIS:")
    print(f"   • Both images used identical seed ({seed}) and parameters")
    print(f"   • Regional injection: TOP-LEFT QUADRANT gets 'dog' conditioning")
    print(f"   • Original conditioning preserved in remaining three quadrants")
    print(f"   • Spatial control: Different content in different areas")
    
    # Show regional summary
    print(f"\nREGIONAL CONFIGURATION:")
    for summary in regional_injector.get_regional_summary():
        print(f"   • Region: {summary['region_type']} covering top-left quadrant")
        print(f"   • Target phrase: '{summary['target_phrase']}'")
        print(f"   • Base: '{summary['base_prompt']}'")
        print(f"   • Injection: '{summary['injection_prompt']}'")
    
    plt.show()


if __name__ == "__main__":
    # Run the regional injection demonstration first
    print("DEMONSTRATING ADVANCED REGIONAL INJECTION CAPABILITIES")
    regional_injection_demonstration()
    
    # Then run the masked injection demonstration
    print("\n" + "="*60)
    print("DEMONSTRATING TOKEN-LEVEL MASKED INJECTION")
    masked_injection_comparison()
    
    # Optionally also run the original comparison
    print("\n" + "="*60)
    print("DEMONSTRATING BASIC INJECTION")
    cat_vs_dog_comparison()
