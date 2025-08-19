"""
Test script to verify unified single-pass control works correctly.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from core_pulse import UnifiedAdvancedInjector
from core_pulse.prompt_injection.spatial import create_center_circle_mask, MaskFactory
from core_pulse.utils.logger import set_core_pulse_debug_level

def test_unified_single_pass():
    """Test that unified control applies everything in a single pass."""
    print("=== Testing Unified Single-Pass Control ===")
    
    # Enable detailed logging
    set_core_pulse_debug_level('warning')
    
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load SDXL pipeline: {e}")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    print(f"Using device: {device}")
    
    # Test basic unified functionality
    base_prompt = "a beautiful medieval castle"
    
    # Create unified injector 
    injector = UnifiedAdvancedInjector(pipeline)
    
    print("\n1. Testing multi-scale prompt addition...")
    injector.add_hierarchical_prompts(
        structure_prompt="imposing castle silhouette",
        detail_prompt="weathered stone textures"
    )
    
    print("2. Testing attention manipulation...")
    injector.add_attention_manipulation(
        prompt=base_prompt,
        block="all",
        target_phrase="beautiful",
        attention_scale=2.0
    )
    
    print("3. Testing self-attention manipulation...")
    center_mask = create_center_circle_mask((1024, 1024), radius=200)
    outer_mask = MaskFactory.invert(center_mask)
    
    injector.enhance_region_interaction(
        source_region=center_mask,
        target_region=outer_mask,
        attention_scale=1.8
    )
    
    print("4. Getting control summary...")
    summary = injector.get_control_summary()
    print(f"Control summary: {summary}")
    
    expected_configs = summary['prompt_injections'] + summary['attention_manipulations'] + summary['self_attention_manipulations']
    print(f"Total configurations: {expected_configs}")
    
    if expected_configs == 0:
        print("ERROR: No configurations found!")
        return False
    
    print(f"\n5. Testing single-pass generation with {expected_configs} total controls...")
    generator = torch.Generator(device=device).manual_seed(123)
    
    try:
        with injector:
            result = injector(
                prompt=base_prompt,
                generator=generator,
                num_inference_steps=5,  # Fast test
                output_type="pil"
            )
            
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            image.save("unified_test_result.png")
            print("SUCCESS: Single-pass generation completed!")
            print("Test result saved as unified_test_result.png")
            return True
        else:
            print("ERROR: No images generated")
            return False
            
    except Exception as e:
        print(f"ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_control_method():
    """Test the complete_control convenience method."""
    print("\n=== Testing Complete Control Method ===")
    
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Could not load SDXL pipeline: {e}")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    
    base_prompt = "a gothic cathedral"
    
    # Test the all-in-one method
    injector = UnifiedAdvancedInjector(pipeline)
    
    # Create spatial masks
    center = create_center_circle_mask((1024, 1024), radius=250)
    outer = MaskFactory.invert(center)
    
    print("Configuring complete control system...")
    injector.add_complete_control(
        base_prompt=base_prompt,
        structure_prompt="majestic cathedral silhouette",
        detail_prompt="intricate stone carvings",
        attention_targets=[
            {"target_phrase": "gothic", "attention_scale": 2.2, "block": "all"}
        ],
        self_attention_configs=[
            {"source_region": center, "target_region": outer, "attention_scale": 1.9, "interaction_type": "enhance"}
        ],
        weights={"structure": 1.3, "detail": 1.1}
    )
    
    summary = injector.get_control_summary()
    print(f"Complete control configured: {summary}")
    
    try:
        with injector:
            result = injector(
                prompt=base_prompt,
                generator=torch.Generator(device=device).manual_seed(456),
                num_inference_steps=5,
                output_type="pil"
            ).images[0]
        
        result.save("complete_control_test.png")
        print("SUCCESS: Complete control method works!")
        print("Result saved as complete_control_test.png")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Testing Unified Single-Pass Control System")
    print("==========================================")
    
    test1_success = test_unified_single_pass()
    test2_success = test_complete_control_method()
    
    print(f"\n=== Test Results ===")
    print(f"Unified Single-Pass: {'PASS' if test1_success else 'FAIL'}")
    print(f"Complete Control Method: {'PASS' if test2_success else 'FAIL'}")
    
    if test1_success and test2_success:
        print("\nALL TESTS PASSED! Unified single-pass control is working correctly.")
    else:
        print("\nSOME TESTS FAILED. Check error messages above.")
