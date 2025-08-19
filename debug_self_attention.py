"""
Debug script to isolate self-attention issues.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from core_pulse import SelfAttentionInjector
from core_pulse.utils.logger import set_core_pulse_debug_level

def test_basic_self_attention():
    """Test the most basic self-attention control."""
    print("🔍 Testing Basic Self-Attention Control")
    
    # Enable maximum logging
    set_core_pulse_debug_level('debug')
    
    # Load pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    
    prompt = "a simple red circle on a white background"
    
    # Test 1: Generate without self-attention
    print("\n📸 Generating original image...")
    generator = torch.Generator(device=device).manual_seed(42)
    original = pipeline(prompt, generator=generator, num_inference_steps=10).images[0]
    
    # Test 2: Generate with global self-attention enhancement
    print("\n🎛️ Applying GLOBAL self-attention enhancement...")
    generator = torch.Generator(device=device).manual_seed(42)
    
    injector = SelfAttentionInjector(pipeline)
    
    # Apply the simplest possible self-attention manipulation
    injector.enhance_global_coherence(
        attention_scale=3.0,  # Strong effect to see difference
        block="middle:0",     # Just middle block
        sigma_start=15.0,     # Full range
        sigma_end=0.0
    )
    
    print(f"📊 Self-attention configs: {len(injector.patcher.self_attention_configs)}")
    for block_id, configs in injector.patcher.self_attention_configs.items():
        print(f"  - {block_id}: {len(configs)} configs")
        for i, config in enumerate(configs):
            print(f"    Config {i}: {config.interaction_type}, scale={config.attention_scale}")
    
    with injector:
        enhanced = injector(prompt, generator=generator, num_inference_steps=10).images[0]
    
    # Save for comparison
    original.save("debug_original.png")
    enhanced.save("debug_self_attention.png")
    
    print("✅ Test completed. Check debug_original.png vs debug_self_attention.png")

if __name__ == "__main__":
    test_basic_self_attention()
