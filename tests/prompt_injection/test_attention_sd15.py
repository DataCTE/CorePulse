import pytest
import torch
import numpy as np
from diffusers import StableDiffusionPipeline

from core_pulse.prompt_injection.attention import AttentionMapInjector
from core_pulse.models.unet_patcher import AttentionMapConfig


@pytest.fixture(scope="session")
def sd15_pipeline():
    """Load a real SD 1.5 pipeline. This is slow and will be cached."""
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipeline
    except Exception as e:
        pytest.skip(f"Could not load SD 1.5 pipeline, skipping SD15 attention tests: {e}")


def test_attention_injector_initialization_sd15(sd15_pipeline):
    injector = AttentionMapInjector(sd15_pipeline)
    assert injector.tokenizer is not None
    # SD1.5 should not have tokenizer_2
    assert injector.tokenizer_2 is None


def test_add_attention_manipulation_sd15(sd15_pipeline):
    injector = AttentionMapInjector(sd15_pipeline)
    prompt = "a photo of a cat"
    injector.add_attention_manipulation(
        prompt=prompt,
        block="middle:0",
        target_phrase="cat",
        attention_scale=1.5,
        sigma_start=12.0,
        sigma_end=0.5,
    )

    assert "middle:0" in injector.patcher.attention_map_configs
    configs = injector.patcher.attention_map_configs["middle:0"]
    assert len(configs) == 1
    config = configs[0]
    assert isinstance(config, AttentionMapConfig)
    assert config.attention_scale == 1.5
    assert config.sigma_start == 12.0
    assert config.sigma_end == 0.5
    assert isinstance(config.target_token_indices, list)
    assert len(config.target_token_indices) > 0
    assert all(isinstance(i, int) for i in config.target_token_indices)


def test_add_attention_manipulation_all_blocks_sd15(sd15_pipeline):
    injector = AttentionMapInjector(sd15_pipeline)
    prompt = "a photo of a dog"
    injector.add_attention_manipulation(
        prompt=prompt,
        block="all",
        target_phrase="dog",
        attention_scale=0.7,
    )

    all_blocks = injector.patcher.block_mapper.get_all_block_identifiers()
    assert len(injector.patcher.attention_map_configs) == len(all_blocks)
    for block_id in all_blocks:
        assert block_id in injector.patcher.attention_map_configs
        cfg = injector.patcher.attention_map_configs[block_id][0]
        assert cfg.attention_scale == 0.7
        assert isinstance(cfg.target_token_indices, list)
        assert all(isinstance(i, int) for i in cfg.target_token_indices)


@pytest.mark.slow
def test_attention_manipulation_influences_output_sd15(sd15_pipeline):
    """
    Integration test to ensure SD1.5 attention manipulation changes image output.
    """
    prompt = "a photorealistic portrait of an astronaut"
    generator = torch.Generator(device=sd15_pipeline.device).manual_seed(42)

    # Base image
    base_image_pil = sd15_pipeline(
        prompt,
        generator=generator,
        num_inference_steps=2,
        output_type="pil",
    ).images[0]
    base_image = np.array(base_image_pil)

    injector = AttentionMapInjector(sd15_pipeline)
    injector.add_attention_manipulation(
        prompt=prompt,
        block="all",
        target_phrase="photorealistic",
        attention_scale=5.0,
    )

    generator.manual_seed(42)
    with injector:
        manipulated_image_pil = injector(
            prompt=prompt,
            generator=generator,
            num_inference_steps=2,
            output_type="pil",
        ).images[0]
    manipulated_image = np.array(manipulated_image_pil)

    assert not np.array_equal(base_image, manipulated_image), (
        "Manipulated image is identical to base image; attention manipulation had no effect."
    )