"""
Attention Manipulation Examples for CorePulse

This example demonstrates amplifying or reducing attention to specific words
in your prompt without changing the prompt text itself.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from core_pulse import AttentionMapInjector


def photorealistic_boost_example():
    """
    Classic example: Boost attention on "photorealistic" for enhanced realism.
    
    This demonstrates how attention manipulation differs from just changing 
    the prompt - it amplifies the model's internal focus on existing words.
    """
    print("Loading SDXL pipeline...")
    print("Note: Attention manipulation has been tested with SDXL")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    prompt = "a photorealistic portrait of an astronaut"
    
    print("\n=== Photorealistic Boost Example ===")
    print(f"Prompt: '{prompt}'")
    
    # Generate baseline
    print("\n1. Generating baseline (normal attention)...")
    baseline = pipeline(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("attention_baseline.png")
    print("Saved: attention_baseline.png")
    
    # Generate with boosted "photorealistic" attention
    print("\n2. Generating with 5x attention boost on 'photorealistic'...")
    with AttentionMapInjector(pipeline) as injector:
        injector.add_attention_manipulation(
            prompt=prompt,
            block="all",  # Apply to all blocks
            target_phrase="photorealistic",
            attention_scale=5.0,  # 5x more attention
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        boosted_result = injector(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        boosted_result.images[0].save("attention_photorealistic_boosted.png")
        print("Saved: attention_photorealistic_boosted.png")
        print("Result: Enhanced photorealism through attention amplification")


def reduce_background_attention_example():
    """
    Demonstrate reducing attention on background elements for focus control.
    """
    print("\n=== Reduce Background Attention Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    prompt = "a majestic lion in a dense jungle with tropical plants"
    
    print(f"Prompt: '{prompt}'")
    
    # Generate baseline
    print("\n1. Baseline generation...")
    baseline = pipeline(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("attention_lion_baseline.png")
    print("Saved: attention_lion_baseline.png")
    
    # Reduce attention on jungle background
    print("\n2. Reducing attention on 'jungle' and 'plants' (0.3x each)...")
    with AttentionMapInjector(pipeline) as injector:
        # Reduce jungle attention
        injector.add_attention_manipulation(
            prompt=prompt,
            block="all",
            target_phrase="jungle",
            attention_scale=0.3,  # Much less attention to jungle
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        # Reduce plants attention
        injector.add_attention_manipulation(
            prompt=prompt,
            block="all",
            target_phrase="plants",
            attention_scale=0.3,  # Much less attention to plants
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        focused_result = injector(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        focused_result.images[0].save("attention_lion_focused.png")
        print("Saved: attention_lion_focused.png")
        print("Result: Lion with reduced jungle background, more focus on main subject")


def multi_word_attention_example():
    """
    Demonstrate manipulating attention on multiple words simultaneously.
    """
    print("\n=== Multi-Word Attention Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    prompt = "a vibrant sunset over calm ocean waters with sailing boats"
    
    print(f"Prompt: '{prompt}'")
    
    # Generate baseline
    print("\n1. Baseline generation...")
    baseline = pipeline(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("attention_sunset_baseline.png")
    print("Saved: attention_sunset_baseline.png")
    
    # Boost sunset colors, reduce boats
    print("\n2. Boosting 'vibrant sunset' (3x), reducing 'boats' (0.4x)...")
    with AttentionMapInjector(pipeline) as injector:
        # Boost vibrant sunset
        injector.add_attention_manipulation(
            prompt=prompt,
            block="all",
            target_phrase="vibrant sunset",
            attention_scale=3.0,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        # Reduce boats attention
        injector.add_attention_manipulation(
            prompt=prompt,
            block="all",
            target_phrase="boats",
            attention_scale=0.4,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        enhanced_result = injector(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        enhanced_result.images[0].save("attention_sunset_enhanced.png")
        print("Saved: attention_sunset_enhanced.png")
        print("Result: Dramatic sunset with minimal boat presence")


def block_specific_attention_example():
    """
    Demonstrate applying attention manipulation to specific UNet blocks only.
    """
    print("\n=== Block-Specific Attention Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    prompt = "a mysterious wizard casting magical spells in an ancient library"
    
    print(f"Prompt: '{prompt}'")
    
    # Generate baseline
    print("\n1. Baseline generation...")
    baseline = pipeline(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("attention_wizard_baseline.png")
    print("Saved: attention_wizard_baseline.png")
    
    # Apply different attention manipulations to different blocks
    print("\n2. Content blocks: boost 'magical spells' (4x)")
    print("   Style blocks: boost 'mysterious' (2.5x)")
    with AttentionMapInjector(pipeline) as injector:
        # Content blocks (middle) - boost magical effects
        injector.add_attention_manipulation(
            prompt=prompt,
            block="middle:0",
            target_phrase="magical spells",
            attention_scale=4.0,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        # Style blocks (output) - boost mysterious atmosphere
        injector.add_attention_manipulation(
            prompt=prompt,
            block="output:0,output:1,output:2",  # Multiple output blocks
            target_phrase="mysterious",
            attention_scale=2.5,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        block_result = injector(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        block_result.images[0].save("attention_wizard_block_specific.png")
        print("Saved: attention_wizard_block_specific.png")
        print("Result: Enhanced magical effects with mysterious atmosphere")


def attention_scaling_comparison():
    """
    Compare different attention scaling levels to show the effect range.
    """
    print("\n=== Attention Scaling Comparison ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    prompt = "a red sports car on a city street"
    target_phrase = "red"
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"Prompt: '{prompt}'")
    print(f"Testing attention scales on '{target_phrase}': {scales}")
    
    for scale in scales:
        print(f"\n{scale}x attention on '{target_phrase}'...")
        
        if scale == 1.0:
            # Baseline - no manipulation
            result = pipeline(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024
            )
        else:
            # With attention manipulation
            with AttentionMapInjector(pipeline) as injector:
                injector.add_attention_manipulation(
                    prompt=prompt,
                    block="all",
                    target_phrase=target_phrase,
                    attention_scale=scale,
                    sigma_start=15.0,
                    sigma_end=0.0
                )
                
                result = injector(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=1024,
                    height=1024
                )
        
        filename = f"attention_scale_{scale}x.png"
        result.images[0].save(filename)
        print(f"Saved: {filename}")
    
    print("Result: Series showing effect of different attention scales on color intensity")


if __name__ == "__main__":
    print("CorePulse Attention Manipulation Examples")
    print("=" * 50)
    print("Note: These examples require SDXL model")
    
    # Run examples
    photorealistic_boost_example()
    reduce_background_attention_example()
    multi_word_attention_example()
    block_specific_attention_example()
    attention_scaling_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Generated images:")
    print("- attention_baseline.png")
    print("- attention_photorealistic_boosted.png")
    print("- attention_lion_baseline.png")
    print("- attention_lion_focused.png")
    print("- attention_sunset_baseline.png")
    print("- attention_sunset_enhanced.png")
    print("- attention_wizard_baseline.png")
    print("- attention_wizard_block_specific.png")
    print("- attention_scale_0.1x.png")
    print("- attention_scale_0.5x.png")
    print("- attention_scale_1.0x.png")
    print("- attention_scale_2.0x.png")
    print("- attention_scale_5.0x.png")
