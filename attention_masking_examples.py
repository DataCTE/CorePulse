"""
Token-Level Attention Masking Examples for CorePulse

This example demonstrates selective prompt token control - masking specific words
while preserving others at the linguistic level.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from core_pulse import AttentionMapInjector


def token_masking_cat_to_dog_example():
    """
    Demonstrate masking "cat" tokens while keeping "park" context.
    
    This shows how to selectively ignore parts of your prompt at the token level,
    allowing fine-grained control over what the model processes.
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
    
    base_prompt = "a fluffy cat playing at a beautiful park with trees"
    
    print("\n=== Token-Level Attention Masking Example ===")
    print(f"Base prompt: '{base_prompt}'")
    
    # First, generate baseline
    print("\n1. Generating baseline (full prompt attention)...")
    baseline = pipeline(
        prompt=base_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("token_masking_baseline.png")
    print("Saved: token_masking_baseline.png")
    
    # Generate with "cat" attention heavily reduced
    print("\n2. Generating with 'cat' attention masked (0.1x attention)...")
    with AttentionMapInjector(pipeline) as injector:
        injector.add_attention_manipulation(
            prompt=base_prompt,
            block="all",
            target_phrase="cat",
            attention_scale=0.1,  # Heavily reduce attention to "cat"
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        masked_result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        masked_result.images[0].save("token_masking_cat_reduced.png")
        print("Saved: token_masking_cat_reduced.png")
        print("Result: Park scene with minimal cat presence")
    
    # Generate with "fluffy" attention boosted, "cat" reduced
    print("\n3. Generating with 'fluffy' boosted (3x) and 'cat' masked (0.2x)...")
    with AttentionMapInjector(pipeline) as injector:
        # Multiple attention manipulations
        injector.add_attention_manipulation(
            prompt=base_prompt,
            block="all",
            target_phrase="fluffy",
            attention_scale=3.0,  # Boost fluffy characteristics
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        injector.add_attention_manipulation(
            prompt=base_prompt,
            block="all", 
            target_phrase="cat",
            attention_scale=0.2,  # Reduce cat attention
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        combined_result = injector(
            prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        combined_result.images[0].save("token_masking_combined.png")
        print("Saved: token_masking_combined.png")
        print("Result: Park scene with fluffy elements but reduced cat presence")


def selective_word_emphasis_example():
    """
    Demonstrate selective emphasis of specific words in a complex prompt.
    """
    print("\n=== Selective Word Emphasis Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    complex_prompt = "a majestic dragon with iridescent scales flying over a medieval castle at sunset"
    
    print(f"Complex prompt: '{complex_prompt}'")
    
    # Generate baseline
    print("\n1. Baseline generation...")
    baseline = pipeline(
        prompt=complex_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    )
    baseline.images[0].save("emphasis_baseline.png")
    print("Saved: emphasis_baseline.png")
    
    # Emphasize "iridescent scales"
    print("\n2. Emphasizing 'iridescent scales' (4x attention)...")
    with AttentionMapInjector(pipeline) as injector:
        injector.add_attention_manipulation(
            prompt=complex_prompt,
            block="all",
            target_phrase="iridescent scales",
            attention_scale=4.0,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        scales_emphasis = injector(
            prompt=complex_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        scales_emphasis.images[0].save("emphasis_scales.png")
        print("Saved: emphasis_scales.png")
        print("Result: Dragon with highly emphasized iridescent scale details")
    
    # Emphasize "sunset" while reducing "castle"
    print("\n3. Emphasizing 'sunset' (3x) while reducing 'castle' (0.3x)...")
    with AttentionMapInjector(pipeline) as injector:
        injector.add_attention_manipulation(
            prompt=complex_prompt,
            block="all",
            target_phrase="sunset",
            attention_scale=3.0,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        injector.add_attention_manipulation(
            prompt=complex_prompt,
            block="all",
            target_phrase="castle",
            attention_scale=0.3,
            sigma_start=15.0,
            sigma_end=0.0
        )
        
        sunset_emphasis = injector(
            prompt=complex_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        sunset_emphasis.images[0].save("emphasis_sunset_reduce_castle.png")
        print("Saved: emphasis_sunset_reduce_castle.png")
        print("Result: Dragon scene with dramatic sunset, minimal castle presence")


def progressive_token_masking_example():
    """
    Demonstrate progressive masking of tokens to show the effect.
    """
    print("\n=== Progressive Token Masking Example ===")
    
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to("cpu")
    
    prompt = "a red sports car parked next to a blue house"
    attention_levels = [1.0, 0.7, 0.4, 0.1]  # Progressive masking
    
    print(f"Prompt: '{prompt}'")
    print("Progressively masking 'red' token attention...")
    
    for i, attention_scale in enumerate(attention_levels):
        print(f"\n{i+1}. Attention scale for 'red': {attention_scale}x")
        
        if attention_scale == 1.0:
            # Baseline - no masking
            result = pipeline(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024
            )
        else:
            # With masking
            with AttentionMapInjector(pipeline) as injector:
                injector.add_attention_manipulation(
                    prompt=prompt,
                    block="all",
                    target_phrase="red",
                    attention_scale=attention_scale,
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
        
        filename = f"progressive_masking_{attention_scale}x.png"
        result.images[0].save(filename)
        print(f"Saved: {filename}")
    
    print("Result: Series showing progressive reduction of red color in sports car")


if __name__ == "__main__":
    print("CorePulse Token-Level Attention Masking Examples")
    print("=" * 50)
    print("Note: These examples require SDXL model")
    
    # Run examples
    token_masking_cat_to_dog_example()
    selective_word_emphasis_example()
    progressive_token_masking_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Generated images:")
    print("- token_masking_baseline.png")
    print("- token_masking_cat_reduced.png")
    print("- token_masking_combined.png")
    print("- emphasis_baseline.png")
    print("- emphasis_scales.png")
    print("- emphasis_sunset_reduce_castle.png")
    print("- progressive_masking_1.0x.png")
    print("- progressive_masking_0.7x.png")
    print("- progressive_masking_0.4x.png")
    print("- progressive_masking_0.1x.png")
