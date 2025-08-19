import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from core_pulse.prompt_injection.advanced import AdvancedPromptInjector
from core_pulse.models.base import BlockIdentifier

# --- Fixtures ---

@pytest.fixture
def mock_pipeline():
    """Return a mock pipeline for testing."""
    pipeline = MagicMock()
    pipeline.unet = MagicMock()
    pipeline.unet.config = {
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        "mid_block_type": "UNetMidBlock2DCrossAttn"
    }
    return pipeline

@pytest.fixture
def injector(mock_pipeline):
    """Return an AdvancedPromptInjector with a mock pipeline."""
    return AdvancedPromptInjector(mock_pipeline)

@pytest.fixture(scope="session")
def sd15_pipeline():
    """Load a real SD 1.5 pipeline. This is slow and will be cached."""
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipeline
    except Exception as e:
        pytest.skip(f"Could not load SD 1.5 pipeline, skipping integration test: {e}")

@pytest.fixture(scope="session")
def sdxl_pipeline():
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

# --- Tests ---

def test_advanced_injector_add_single_injection(injector):
    """Test adding a single injection."""
    injector.add_injection("middle:0", "a cat", weight=0.8)
    
    block_id = BlockIdentifier("middle", 0)
    assert block_id in injector.configs
    config = injector.configs[block_id]
    assert config.prompt == "a cat"
    assert config.weight == 0.8

def test_advanced_injector_add_all_injection(injector):
    """Test adding an injection to all blocks."""
    injector.add_injection("all", "a dog")
    
    all_blocks = injector.patcher.block_mapper.get_all_block_identifiers()
    assert len(injector.configs) == len(all_blocks)
    for block_str in all_blocks:
        block_id = BlockIdentifier.from_string(block_str)
        assert block_id in injector.configs
        assert injector.configs[block_id].prompt == "a dog"

def test_advanced_injector_configure_injections(injector):
    """Test configuring injections from a list of dicts."""
    config_list = [
        {"block": "input:4", "prompt": "first prompt", "weight": 0.5},
        {"block": "output:1", "prompt": "second prompt", "sigma_start": 0.8}
    ]
    injector.configure_injections(config_list)

    assert len(injector.configs) == 2
    
    block1_id = BlockIdentifier("input", 4)
    assert block1_id in injector.configs
    assert injector.configs[block1_id].prompt == "first prompt"
    assert injector.configs[block1_id].weight == 0.5
    
    block2_id = BlockIdentifier("output", 1)
    assert block2_id in injector.configs
    assert injector.configs[block2_id].prompt == "second prompt"
    assert injector.configs[block2_id].sigma_start == 0.8

@pytest.mark.slow
def test_advanced_injector_influences_output(sd15_pipeline):
    """Integration test to ensure injection actually changes the image output."""
    prompt = "a photograph of an astronaut riding a horse"
    generator = torch.Generator(device=sd15_pipeline.device).manual_seed(42)

    # Generate base image
    base_image_pil = sd15_pipeline(
        prompt, 
        generator=generator,
        num_inference_steps=2, # Keep it fast
        output_type="pil"
    ).images[0]
    base_image = np.array(base_image_pil)

    # Generate injected image
    injector = AdvancedPromptInjector(sd15_pipeline)
    injector.add_injection("all", "in a surrealist style", weight=2.0)
    
    generator.manual_seed(42) # Reset seed
    with injector:
        injector.apply_to_pipeline(sd15_pipeline)
        injected_image_pil = injector(
            prompt,
            generator=generator,
            num_inference_steps=2,
            output_type="pil"
        ).images[0]
    injected_image = np.array(injected_image_pil)

    # Ensure the images are different
    assert not np.array_equal(base_image, injected_image), \
        "Injected image is identical to the base image, meaning injection had no effect."

@pytest.mark.slow
def test_advanced_injector_influences_output_sdxl(sdxl_pipeline):
    """Integration test to ensure injection actually changes the image output for SDXL."""
    prompt = "a photograph of an astronaut riding a horse"
    generator = torch.Generator(device=sdxl_pipeline.device).manual_seed(42)

    # Generate base image
    base_image_pil = sdxl_pipeline(
        prompt, 
        generator=generator,
        num_inference_steps=2, # Keep it fast
        output_type="pil"
    ).images[0]
    base_image = np.array(base_image_pil)

    # Generate injected image
    injector = AdvancedPromptInjector(sdxl_pipeline)
    injector.add_injection("all", "in a surrealist style", weight=2.0)
    
    generator.manual_seed(42) # Reset seed
    with injector:
        injector.apply_to_pipeline(sdxl_pipeline)
        injected_image_pil = injector(
            prompt,
            generator=generator,
            num_inference_steps=2,
            output_type="pil"
        ).images[0]
    injected_image = np.array(injected_image_pil)

    # Ensure the images are different
    assert not np.array_equal(base_image, injected_image), \
        "Injected image is identical to the base image, meaning injection had no effect."
