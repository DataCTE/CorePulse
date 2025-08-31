import pytest
import torch
from diffusers import StableDiffusionPipeline

from core_pulse.prompt_injection.simple import SimplePromptInjector


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
        pytest.skip(f"Could not load SD 1.5 pipeline, skipping SD15 simple tests: {e}")


def test_simple_configure_and_apply_sd15(sd15_pipeline):
    injector = SimplePromptInjector(sd15_pipeline)
    injector.configure_injections(
        block="all",
        prompt="surrealist style",
        weight=1.5,
        sigma_start=15.0,
        sigma_end=0.0,
    )
    injector.apply_to_pipeline(sd15_pipeline)