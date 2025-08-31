import pytest
import torch
from diffusers import StableDiffusionPipeline

from core_pulse.prompt_injection.multi_scale import MultiScaleInjector


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
        pytest.skip(f"Could not load SD 1.5 pipeline, skipping SD15 multiscale tests: {e}")


def test_multiscale_adders_do_not_crash_sd15(sd15_pipeline):
    injector = MultiScaleInjector(sd15_pipeline)
    injector.add_structure_injection(
        "castle silhouette",
        weight=2.0,
        sigma_start=15.0,
        sigma_end=0.5,
    )
    injector.add_detail_injection(
        "weathered stone, intricate carvings",
        weight=1.8,
        sigma_start=3.0,
        sigma_end=0.0,
    )
    injector.apply_to_pipeline(sd15_pipeline)