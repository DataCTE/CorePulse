import pytest
import torch
from diffusers import StableDiffusionPipeline

from core_pulse.prompt_injection.masking import MaskedPromptInjector
from core_pulse.models.base import BlockIdentifier


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
        pytest.skip(f"Could not load SD 1.5 pipeline, skipping SD15 masked tests: {e}")


def test_add_masked_injection_config_and_summary(sd15_pipeline):
    injector = MaskedPromptInjector(sd15_pipeline)

    injector.add_masked_injection(
        block="input:0",
        base_prompt="a cat playing in a park",
        injection_prompt="a dog playing in a park",
        target_phrase="cat",
        weight=1.2,
        sigma_start=15.0,
        sigma_end=0.0,
    )

    # Ensure config stored
    summaries = injector.get_masking_summary()
    assert len(summaries) == 1
    assert summaries[0]["block"] == "input:0"
    assert summaries[0]["target_phrase"] == "cat"


@pytest.mark.slow
def test_masked_injection_apply_encodes_and_configures(sd15_pipeline):
    """Apply should encode and populate encoded prompt for SD1.5."""
    injector = MaskedPromptInjector(sd15_pipeline)

    injector.add_masked_injection(
        block="input:0",
        base_prompt="a cat playing in a park",
        injection_prompt="a dog playing in a park",
        target_phrase="cat",
        weight=1.0,
        sigma_start=15.0,
        sigma_end=0.0,
    )

    injector.apply_to_pipeline(sd15_pipeline)

    block_id = BlockIdentifier.from_string("input:0")
    assert block_id in injector.configs
    assert injector.configs[block_id]._encoded_prompt is not None