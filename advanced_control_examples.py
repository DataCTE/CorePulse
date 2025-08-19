"""
CorePulse Advanced Control Examples

This script demonstrates the latest advanced features:
1. Self-Attention Control - Controls how parts of the image attend to each other  
2. Multi-Scale Control - Different prompts at different resolution levels

These features provide unprecedented fine-grained control over image generation.
"""

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline

from core_pulse import SelfAttentionInjector, MultiScaleInjector, UnifiedAdvancedInjector
from core_pulse.prompt_injection.spatial import (
    create_left_half_mask, create_right_half_mask, 
    create_top_half_mask, create_bottom_half_mask,
    create_center_circle_mask, MaskFactory
)
from core_pulse.utils.logger import set_core_pulse_debug_level


def self_attention_composition_example():
    """
    Demonstrates self-attention control for improved composition and coherence.
    """
    print("=== Self-Attention Control: Composition Enhancement ===")
    
    # Enable warning level logging to see important messages
    set_core_pulse_debug_level('warning')
    
    # Load SDXL pipeline
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load SDXL pipeline: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    print(f"Using device: {device}")
    
    base_prompt = "a majestic landscape with a castle on a hill and a village below"
    seed = 42
    steps = 30
    
    # Generate original image
    print(f"Generating original image...")
    generator = torch.Generator(device=device).manual_seed(seed)
    original_image = pipeline(
        prompt=base_prompt,
        generator=generator, 
        num_inference_steps=steps
    ).images[0]
    
    # Generate with self-attention enhancement
    print("Applying self-attention enhancements...")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    injector = SelfAttentionInjector(pipeline)
    
    # 1. Enhance interaction between top (castle) and bottom (village) regions
    top_mask = create_top_half_mask((1024, 1024))
    bottom_mask = create_bottom_half_mask((1024, 1024))
    
    injector.enhance_region_interaction(
        source_region=top_mask,
        target_region=bottom_mask,
        attention_scale=2.5,
        block="all"
    )
    
    # 2. Create subtle attention barrier at the horizon to separate sky and land
    horizon_mask = MaskFactory.from_shape('rectangle', (1024, 1024), 
                                         x=0, y=480, width=1024, height=64)
    injector.create_attention_barrier(horizon_mask, attention_scale=0.3)
    
    # 3. Enhance global coherence
    injector.enhance_global_coherence(attention_scale=1.8, block="middle:0")
    
    with injector:
        enhanced_image = injector(
            prompt=base_prompt,
            generator=generator,
            num_inference_steps=steps
        ).images[0]
    
    # Create comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(original_image)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2.imshow(enhanced_image)
    ax2.set_title("Self-Attention Enhanced\n(Better Composition & Coherence)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("media/self_attention_composition_example.png", dpi=150, bbox_inches='tight')
    print("Self-attention example saved to media/self_attention_composition_example.png")
    
    return original_image, enhanced_image


def multi_scale_architecture_example():
    """
    Demonstrates multi-scale control for hierarchical architecture generation.
    """
    print("\n=== Multi-Scale Control: Hierarchical Architecture ===")
    
    # Load SDXL pipeline
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16", 
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load SDXL pipeline: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    
    base_prompt = "a gothic cathedral"
    seed = 123
    steps = 30
    
    # Generate original image
    print("Generating original image...")
    generator = torch.Generator(device=device).manual_seed(seed)
    original_image = pipeline(
        prompt=base_prompt,
        generator=generator,
        num_inference_steps=steps
    ).images[0]
    
    # Generate with multi-scale control
    print("Applying multi-scale hierarchical control...")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    injector = MultiScaleInjector(pipeline)
    
    # Print resolution summary
    resolution_summary = injector.get_resolution_summary()
    print("Resolution levels available:")
    for res_level, blocks in resolution_summary.items():
        print(f"  {res_level}: {blocks}")
    
    # Add hierarchical prompts with different conditioning at each scale
    injector.add_hierarchical_prompts(
        structure_prompt="majestic gothic cathedral silhouette, soaring spires, imposing architecture",
        midlevel_prompt="ornate stone arches, flying buttresses, rose windows, gothic details", 
        detail_prompt="intricate stone carvings, weathered limestone textures, elaborate tracery",
        weights={
            "structure": 1.6,  # Strong structural control
            "midlevel": 1.3,   # Medium feature control  
            "detail": 1.1      # Subtle detail enhancement
        }
    )
    
    with injector:
        hierarchical_image = injector(
            prompt=base_prompt,
            generator=generator,
            num_inference_steps=steps
        ).images[0]
    
    # Create comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(original_image)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2.imshow(hierarchical_image) 
    ax2.set_title("Multi-Scale Hierarchical\n(Structure + Features + Details)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("media/multi_scale_architecture_example.png", dpi=150, bbox_inches='tight')
    print("Multi-scale example saved to media/multi_scale_architecture_example.png")
    
    return original_image, hierarchical_image


def combined_advanced_control_example():
    """
    Demonstrates the unified single-pass approach combining all advanced control types.
    This shows how UnifiedAdvancedInjector applies everything in a single pass.
    """
    print("\n=== UNIFIED Single-Pass Control: All Advanced Features ===")
    
    # Load SDXL pipeline
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load SDXL pipeline: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    
    base_prompt = "a fantasy landscape with an ancient wizard tower"
    seed = 999
    steps = 35
    
    # Generate original for comparison first
    print("Generating original image for comparison...")
    generator = torch.Generator(device=device).manual_seed(seed)
    original_image = pipeline(
        prompt=base_prompt,
        generator=generator,
        num_inference_steps=steps
    ).images[0]
    
    # NOW: Single unified injector that combines everything
    print("Configuring unified single-pass control system...")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Create spatial masks for self-attention control
    tower_mask = create_center_circle_mask((1024, 1024), radius=300)
    landscape_mask = MaskFactory.invert(tower_mask)
    
    # Initialize the unified injector
    unified_injector = UnifiedAdvancedInjector(pipeline)
    
    # Configure ALL controls in the unified injector
    unified_injector.add_complete_control(
        base_prompt=base_prompt,
        # Multi-scale prompts (resolution-aware conditioning)
        structure_prompt="ancient mystical tower silhouette, magical landscape composition",
        midlevel_prompt="arcane architectural details, glowing magical elements, mystical atmosphere",
        detail_prompt="intricate runic carvings, weathered magical stonework, mystical textures",
        
        # Attention manipulations (cross-attention control)
        attention_targets=[
            {
                "target_phrase": "wizard tower",
                "attention_scale": 2.5,
                "block": "all"
            },
            {
                "target_phrase": "ancient",
                "attention_scale": 1.8,
                "block": "middle:0"
            }
        ],
        
        # Self-attention manipulations (image-image attention)
        self_attention_configs=[
            {
                "source_region": tower_mask,
                "target_region": landscape_mask,
                "attention_scale": 2.2,
                "interaction_type": "enhance",
                "block": "all"
            },
            {
                "interaction_type": "enhance", 
                "attention_scale": 1.6,
                "block": "middle:0"  # Global coherence
            }
        ],
        
        # Weights for different control types
        weights={
            "structure": 1.4,
            "midlevel": 1.2,
            "detail": 1.0
        }
    )
    
    # Print control summary
    control_summary = unified_injector.get_control_summary()
    print(f"Configured controls: {control_summary}")
    print("  - Multi-scale prompts: Structure, Midlevel, Details")
    print("  - Cross-attention: 2 phrase manipulations")
    print("  - Self-attention: 2 spatial/global manipulations")
    print("\nApplying ALL controls in SINGLE PASS...")
    
    # Single pass application
    with unified_injector:
        combined_image = unified_injector(
            prompt=base_prompt,
            generator=generator,
            num_inference_steps=steps
        ).images[0]
    
    # Create comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(original_image)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2.imshow(combined_image)
    ax2.set_title("Unified Single-Pass Control\n(Multi-Scale + Attention + Self-Attention)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("media/combined_advanced_control_example.png", dpi=150, bbox_inches='tight')
    print("Unified control example saved to media/combined_advanced_control_example.png")
    
    return original_image, combined_image


def unified_single_pass_demo():
    """
    Simple demonstration of the unified single-pass approach.
    """
    print("\n=== UNIFIED Single-Pass Demo: Basic Usage ===")
    
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load SDXL pipeline: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    
    base_prompt = "a majestic castle on a hill"
    
    # Single unified injector with multiple control types
    injector = UnifiedAdvancedInjector(pipeline)
    
    # Add hierarchical prompts (multi-scale)
    injector.add_hierarchical_prompts(
        structure_prompt="imposing castle silhouette on hilltop",
        detail_prompt="weathered stone textures, moss-covered walls"
    )
    
    # Add attention manipulation (cross-attention)
    injector.add_attention_manipulation(
        prompt=base_prompt,
        block="all",
        target_phrase="majestic",
        attention_scale=2.0
    )
    
    # Add self-attention enhancement (image-image attention)
    injector.enhance_global_coherence(attention_scale=1.5)
    
    print("Unified injector configured - applying in single pass...")
    
    with injector:
        result = injector(
            prompt=base_prompt,
            num_inference_steps=20,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]
    
    result.save("media/unified_single_pass_demo.png")
    print("Single-pass demo result saved to media/unified_single_pass_demo.png")
    

def main():
    """Run all advanced control examples."""
    print("CorePulse Advanced Control Examples")
    print("==================================")
    print("Demonstrating Self-Attention and Multi-Scale Control")
    print()
    
    # Run examples
    try:
        # Original separate injector examples
        self_attention_composition_example()
        multi_scale_architecture_example()
        
        # New unified single-pass examples
        unified_single_pass_demo()
        combined_advanced_control_example()
        
        print("\n*** All advanced control examples completed successfully! ***")
        print("Check the media/ directory for comparison images.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
