import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from diffusers import StableDiffusionXLPipeline

from core_pulse.prompt_injection.attention import AttentionMapInjector
from core_pulse.models.unet_patcher import AttentionMapConfig

# --- Fixtures ---

@pytest.fixture(scope="session")
def real_sdxl_pipeline():
    """Load a real SDXL pipeline. This is slow and will be cached."""
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipeline
    except Exception as e:
        pytest.skip(f"Could not load SDXL pipeline, skipping integration test: {e}")

# --- Unit Tests ---

def test_attention_injector_initialization(real_sdxl_pipeline):
    """Test that the AttentionMapInjector initializes correctly."""
    injector = AttentionMapInjector(real_sdxl_pipeline)
    assert injector.tokenizer is not None
    assert injector.tokenizer_2 is not None
    assert isinstance(injector, AttentionMapInjector)

def test_add_attention_manipulation(real_sdxl_pipeline):
    """Test adding a single attention manipulation."""
    injector = AttentionMapInjector(real_sdxl_pipeline)
    prompt = "a photo of a cat"
    injector.add_attention_manipulation(
        prompt=prompt,
        block="middle:0",
        target_phrase="cat",
        attention_scale=1.5,
        sigma_start=12.0,
        sigma_end=0.5
    )
    
    # Check that the config was added to the patcher
    assert "middle:0" in injector.patcher.attention_map_configs
    configs = injector.patcher.attention_map_configs["middle:0"]
    assert len(configs) == 1
    
    config = configs[0]
    assert isinstance(config, AttentionMapConfig)
    assert config.attention_scale == 1.5
    assert config.sigma_start == 12.0
    assert config.sigma_end == 0.5
    # Check that token indices are generated and are integers representing positions
    assert isinstance(config.target_token_indices, list)
    assert len(config.target_token_indices) > 0
    assert all(isinstance(i, int) for i in config.target_token_indices)
    # The token "cat" should be at index 4 in "a photo of a cat" (['<|startoftext|>', 'a', 'photo', 'of', 'a', 'cat', '<|endoftext|>'])
    # Note: This index can be fragile if the tokenizer changes.
    assert 5 in config.target_token_indices 

def test_add_attention_manipulation_to_all_blocks(real_sdxl_pipeline):
    """Test adding an attention manipulation to all blocks."""
    injector = AttentionMapInjector(real_sdxl_pipeline)
    prompt = "a photo of a dog"
    injector.add_attention_manipulation(
        prompt=prompt,
        block="all",
        target_phrase="dog",
        attention_scale=0.5
    )
    
    all_blocks = injector.patcher.block_mapper.get_all_block_identifiers()
    assert len(injector.patcher.attention_map_configs) == len(all_blocks)
    
    for block_id in all_blocks:
        assert block_id in injector.patcher.attention_map_configs
        config = injector.patcher.attention_map_configs[block_id][0]
        assert config.attention_scale == 0.5
        # Check that token indices are generated and are integers
        assert isinstance(config.target_token_indices, list)
        assert len(config.target_token_indices) > 0
        assert all(isinstance(i, int) for i in config.target_token_indices)

# --- Integration Test ---

@pytest.mark.slow
def test_attention_manipulation_influences_output(real_sdxl_pipeline):
    """
    Integration test to ensure attention manipulation actually changes the image output.
    This test is slow as it involves running the SDXL pipeline.
    """
    prompt = "a photorealistic portrait of an astronaut"
    generator = torch.Generator(device=real_sdxl_pipeline.device).manual_seed(42)

    # Generate base image
    base_image_pil = real_sdxl_pipeline(
        prompt, 
        generator=generator,
        num_inference_steps=2, # Keep it fast
        output_type="pil"
    ).images[0]
    base_image = np.array(base_image_pil)

    # --- Test with text prompt ---
    injector = AttentionMapInjector(real_sdxl_pipeline)
    injector.add_attention_manipulation(
        prompt=prompt,
        block="all", 
        target_phrase="photorealistic", 
        attention_scale=5.0 # A high value to ensure a visible difference
    )
    
    generator.manual_seed(42) # Reset seed for fair comparison
    with injector:
        manipulated_image_pil = injector(
            prompt=prompt,
            generator=generator,
            num_inference_steps=2,
            output_type="pil"
        ).images[0]
    manipulated_image_text = np.array(manipulated_image_pil)

    # Ensure the images are different
    assert not np.array_equal(base_image, manipulated_image_text), \
        "Manipulated image (from text) is identical to the base image, meaning attention manipulation had no effect."

    # --- Test with pre-encoded prompt_embeds ---
    injector.clear_injections()
    injector.add_attention_manipulation(
        prompt=prompt,
        block="all",
        target_phrase="photorealistic",
        attention_scale=5.0
    )

    generator.manual_seed(42) # Reset seed for fair comparison
    embedding_dict = injector.encode_prompt(prompt, real_sdxl_pipeline)
    with injector:
        manipulated_image_pil_embeds = injector(
            generator=generator,
            num_inference_steps=2,
            output_type="pil",
            **embedding_dict
        ).images[0]
    manipulated_image_embeds = np.array(manipulated_image_pil_embeds)

    # Ensure this also produces a different image
    assert not np.array_equal(base_image, manipulated_image_embeds), \
        "Manipulated image (from embeds) is identical to the base image."
    
    # Ensure both methods produce the same result
    assert np.array_equal(manipulated_image_text, manipulated_image_embeds), \
        "Manipulation from text prompt and prompt_embeds produced different results."
