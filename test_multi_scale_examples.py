#!/usr/bin/env python3
"""
Multi-Scale Injection Examples Generator

Creates comprehensive examples of multi-scale control for documentation and testing.
Generates images showcasing structure vs detail control and saves them to media/ folder.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from core_pulse.prompt_injection.multi_scale import MultiScaleInjector
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

# Ensure media directory exists
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

def load_pipeline():
    """Load and return the SDXL pipeline."""
    print("🔧 Loading SDXL pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")
    return pipeline

def generate_baseline(pipeline, seed=42):
    """Generate baseline image with no injection."""
    print("\n📸 Generating baseline (no injection)...")
    
    result = pipeline(
        prompt="a building in a misty landscape",
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.manual_seed(seed)
    )
    
    filepath = MEDIA_DIR / "multi_scale_baseline.png"
    result.images[0].save(filepath)
    print(f"   ✅ Saved: {filepath}")
    return result.images[0]

def generate_structure_only(pipeline, seed=42):
    """Generate with structure injection only."""
    print("\n🏰 Generating structure-only injection...")
    
    injector = MultiScaleInjector(pipeline)
    injector.add_structure_injection(
        "gothic cathedral with soaring spires and flying buttresses",
        weight=2.5
    )
    
    with injector:
        result = injector(
            prompt="a building in a misty landscape",
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.manual_seed(seed)
        )
    
    filepath = MEDIA_DIR / "multi_scale_structure_only.png"
    result.images[0].save(filepath)
    print(f"   ✅ Saved: {filepath}")
    
    del injector
    return result.images[0]

def generate_detail_only(pipeline, seed=42):
    """Generate with detail injection only.""" 
    print("\n🔍 Generating detail-only injection...")
    
    injector = MultiScaleInjector(pipeline)
    injector.add_detail_injection(
        "weathered stone textures, intricate carvings, moss and lichen",
        weight=2.0
    )
    
    with injector:
        result = injector(
            prompt="a building in a misty landscape",
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.manual_seed(seed)
        )
    
    filepath = MEDIA_DIR / "multi_scale_detail_only.png"
    result.images[0].save(filepath)
    print(f"   ✅ Saved: {filepath}")
    
    del injector
    return result.images[0]

def generate_combined(pipeline, seed=42):
    """Generate with both structure and detail injections."""
    print("\n🎯 Generating combined structure + detail injection...")
    
    injector = MultiScaleInjector(pipeline)
    
    # Structure: Overall architectural form
    injector.add_structure_injection(
        "gothic cathedral with soaring spires and flying buttresses",
        weight=2.5
    )
    
    # Details: Surface textures and fine elements
    injector.add_detail_injection(
        "weathered stone textures, intricate carvings, moss and lichen",
        weight=2.0
    )
    
    print(f"   Configured {len(injector.configs)} injections:")
    for block_id, config in injector.configs.items():
        print(f"     {block_id}: '{config.prompt[:40]}...' (σ={config.sigma_start}→{config.sigma_end})")
    
    with injector:
        result = injector(
            prompt="a building in a misty landscape",
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.manual_seed(seed)
        )
    
    filepath = MEDIA_DIR / "multi_scale_combined.png"
    result.images[0].save(filepath)
    print(f"   ✅ Saved: {filepath}")
    
    del injector
    return result.images[0]

def generate_hierarchical_example(pipeline, seed=42):
    """Generate using hierarchical prompts method."""
    print("\n⚡ Generating hierarchical multi-scale example...")
    
    injector = MultiScaleInjector(pipeline)
    
    # Use the hierarchical prompts method for complete control
    injector.add_hierarchical_prompts(
        structure_prompt="majestic gothic cathedral silhouette, imposing medieval fortress",
        midlevel_prompt="ornate stone arches, flying buttresses, tall stained glass windows", 
        detail_prompt="weathered limestone textures, intricate stone carvings, ancient stonework",
        weights={"structure": 2.8, "midlevel": 2.2, "detail": 1.8}
    )
    
    print(f"   Configured {len(injector.configs)} hierarchical injections")
    
    with injector:
        result = injector(
            prompt="a building in a misty landscape",
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.manual_seed(seed)
        )
    
    filepath = MEDIA_DIR / "multi_scale_hierarchical.png"
    result.images[0].save(filepath)
    print(f"   ✅ Saved: {filepath}")
    
    del injector
    return result.images[0]

def generate_semantic_comparison(pipeline, seed=42):
    """Generate examples showing semantic compatibility vs conflicts."""
    print("\n🔬 Generating semantic compatibility examples...")
    
    # Compatible prompts
    print("   → Compatible prompts (stone theme)...")
    injector_good = MultiScaleInjector(pipeline)
    injector_good.add_structure_injection("stone castle fortress", weight=2.0)
    injector_good.add_detail_injection("weathered stone textures, carved stonework", weight=1.8)
    
    with injector_good:
        result_good = injector_good(
            prompt="a building in a landscape",
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=torch.manual_seed(seed)
        )
    
    filepath_good = MEDIA_DIR / "multi_scale_compatible_prompts.png"
    result_good.images[0].save(filepath_good)
    print(f"   ✅ Saved: {filepath_good}")
    
    # Conflicting prompts  
    print("   → Conflicting prompts (mixed themes)...")
    injector_bad = MultiScaleInjector(pipeline)
    injector_bad.add_structure_injection("futuristic glass skyscraper", weight=2.0)
    injector_bad.add_detail_injection("organic wood bark textures, natural moss", weight=1.8)
    
    with injector_bad:
        result_bad = injector_bad(
            prompt="a building in a landscape",
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=torch.manual_seed(seed)
        )
    
    filepath_bad = MEDIA_DIR / "multi_scale_conflicting_prompts.png"
    result_bad.images[0].save(filepath_bad)
    print(f"   ✅ Saved: {filepath_bad}")
    
    del injector_good, injector_bad
    return result_good.images[0], result_bad.images[0]

def create_comparison_grid(images, titles, filename="multi_scale_comparison.png"):
    """Create a matplotlib comparison grid."""
    print(f"\n📊 Creating comparison grid: {filename}")
    
    n_images = len(images)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fig.suptitle('Multi-Scale Injection Comparison\nShowing Independent Control of Structure and Details', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=12, fontweight='bold', pad=10)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    filepath = MEDIA_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   ✅ Saved comparison grid: {filepath}")

def main():
    """Main execution function."""
    print("🎨 MULTI-SCALE INJECTION EXAMPLES GENERATOR")
    print("=" * 50)
    print("Generating comprehensive examples for documentation...")
    
    # Load pipeline once and reuse
    pipeline = load_pipeline()
    
    # Generate all examples with the same seed for consistency
    seed = 12345
    
    try:
        # Generate individual examples
        img_baseline = generate_baseline(pipeline, seed)
        img_structure = generate_structure_only(pipeline, seed)  
        img_detail = generate_detail_only(pipeline, seed)
        img_combined = generate_combined(pipeline, seed)
        img_hierarchical = generate_hierarchical_example(pipeline, seed)
        
        # Generate semantic comparison
        img_compatible, img_conflicting = generate_semantic_comparison(pipeline, seed)
        
        # Create main comparison grid
        main_images = [img_baseline, img_structure, img_detail, img_combined, img_hierarchical]
        main_titles = [
            "Baseline\n(No Injection)",
            "Structure Only\n(Gothic Cathedral)",
            "Detail Only\n(Stone Textures)", 
            "Combined\n(Structure + Details)",
            "Hierarchical\n(3-Level Control)"
        ]
        
        create_comparison_grid(main_images, main_titles, "multi_scale_main_comparison.png")
        
        # Create semantic comparison grid
        semantic_images = [img_baseline, img_compatible, img_conflicting]
        semantic_titles = [
            "Baseline\n(No Injection)",
            "Compatible Prompts\n(Stone + Stone Textures)",
            "Conflicting Prompts\n(Glass + Wood Textures)"
        ]
        
        create_comparison_grid(semantic_images, semantic_titles, "multi_scale_semantic_comparison.png")
        
        print("\n🎉 SUCCESS! Generated comprehensive multi-scale examples")
        print(f"📁 All files saved to: {MEDIA_DIR.absolute()}")
        print("\n📋 Generated files:")
        for file in sorted(MEDIA_DIR.glob("multi_scale_*.png")):
            print(f"   • {file.name}")
            
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    main()
