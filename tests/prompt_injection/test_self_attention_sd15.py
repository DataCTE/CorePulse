import pytest
import torch
from diffusers import StableDiffusionPipeline

from core_pulse.prompt_injection.self_attention import SelfAttentionInjector
from core_pulse.prompt_injection.spatial import (
    create_left_half_mask,
    create_right_half_mask,
)


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
        pytest.skip(f"Could not load SD 1.5 pipeline, skipping SD15 self-attention tests: {e}")


def test_self_attention_adders_do_not_crash_sd15(sd15_pipeline):
    injector = SelfAttentionInjector(sd15_pipeline)
    left = create_left_half_mask((256, 256))
    right = create_right_half_mask((256, 256))

    injector.enhance_region_interaction(left, right, attention_scale=2.0)
    injector.suppress_region_interaction(right, left, attention_scale=0.5)
    injector.enhance_global_coherence(attention_scale=1.5)
    injector.apply_to_pipeline(sd15_pipeline)