"""
Base classes for prompt injection in CorePulse.
"""

from abc import ABC
from typing import Dict, Optional, Any
import torch
from diffusers import DiffusionPipeline
from dataclasses import dataclass, field

from ..models.base import BlockIdentifier
from ..models.unet_patcher import UNetPatcher
from ..utils.helpers import detect_model_type


@dataclass
class PromptInjectionConfig:
    """Configuration for a single prompt injection."""
    block: Any
    prompt: str
    weight: float
    sigma_start: float
    sigma_end: float
    spatial_mask: Optional[torch.Tensor] = None
    _encoded_prompt: Optional[torch.Tensor] = field(default=None, repr=False)


class BasePromptInjector(ABC):
    """
    Abstract base class for all prompt injectors.
    """
    
    def __init__(self, pipeline: DiffusionPipeline):
        self.patcher = UNetPatcher(pipeline.unet, pipeline.scheduler)
        self.configs: Dict[BlockIdentifier, PromptInjectionConfig] = {}
        self._pipeline = pipeline
        self._is_applied = False

    @property
    def model_type(self) -> str:
        """Dynamically detect the model type from the pipeline."""
        if self._pipeline:
            return detect_model_type(self._pipeline)
        return "unknown"

    def clear_injections(self):
        """Clear all injections and remove patches."""
        self.configs.clear()
        if self._pipeline:
            self.patcher.remove_patches(self._pipeline.unet)
        self.patcher.clear_injections()

    def apply_to_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """Apply all configured injections to the pipeline."""
        self.patcher.apply_patches(pipeline.unet)
        self._is_applied = True
        return pipeline

    def encode_prompt(self, prompt: str, pipeline: DiffusionPipeline) -> torch.Tensor:
        """Encode a prompt using the pipeline's text encoder."""
        return pipeline.encode_prompt(
            prompt=prompt,
            device=pipeline.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )[0]

    def __enter__(self):
        """Apply patches when entering context."""
        if self._pipeline:
            self.apply_to_pipeline(self._pipeline)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove patches when exiting context."""
        self.clear_injections()

    def __call__(self, *args, **kwargs):
        """Make the injector callable to pass through to the pipeline."""
        if not self._is_applied or self._pipeline is None:
            raise RuntimeError("Injector has not been applied to a pipeline.")
        return self._pipeline(*args, **kwargs)
