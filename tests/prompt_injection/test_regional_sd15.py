import pytest
import torch
from diffusers import StableDiffusionPipeline

from core_pulse.prompt_injection.spatial import (
    RegionalPromptInjector,
    create_center_circle_mask,
)
from core_pulse.models.base import BlockIdentifier


@pytest.fixture(scope="session")
def sd15_pipeline():
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipeline
    except Exception as e:
        pytest.skip(f"Could not load SD 1.5 pipeline, skipping SD15 regional tests: {e}")


def test_add_regional_injection_sd15(sd15_pipeline):
    injector = RegionalPromptInjector(sd15_pipeline)
    mask = create_center_circle_mask(image_size=(256, 256), radius=64)

    injector.add_regional_injection(
        block="middle:0",
        prompt="golden retriever dog",
        mask=mask,
        weight=1.8,
        sigma_start=15.0,
        sigma_end=0.0,
    )

    block_id = BlockIdentifier.from_string("middle:0")
    assert block_id in injector.configs
    cfg = injector.configs[block_id]
    assert cfg.prompt == "golden retriever dog"
    assert cfg.spatial_mask is not None