# CorePulse

A modular toolkit for advanced diffusion model manipulation.

## Features

- **Prompt Injection**: Inject specific prompts at specific blocks of the Stable Diffusion UNet
- **Fine-grained Control**: Control content, style, and composition separately
- **Multiple Interfaces**: Simple, advanced, and location-based injection options
- **Model Support**: SDXL and SD1.5 architectures
- **Diffusers Integration**: Seamless integration with HuggingFace Diffusers

## Quick Start

```python
from core_pulse import SimplePromptInjector
from diffusers import StableDiffusionXLPipeline

# Load your pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Create and use injector
with SimplePromptInjector("sdxl") as injector:
    modified_pipeline = injector.inject_prompt(
        pipeline=pipeline,
        block="middle:0",  # Content block
        prompt="white cat",
        weight=1.0
    )
    
    result = modified_pipeline("a blue dog in a garden")
```

## Installation

```bash
uv sync  # Install all dependencies
uv sync --extra examples  # Include example dependencies
uv sync --extra dev  # Include development tools
```

## Architecture

CorePulse provides a modular system with:

- **Base Classes**: `BaseModelPatcher`, `BasePromptInjector`
- **UNet Patching**: `UNetPatcher`, `UNetBlockMapper`
- **Simple Interface**: `SimplePromptInjector`, `BlockSpecificInjector`
- **Advanced Interface**: `AdvancedPromptInjector`, `MultiPromptInjector`
- **Utilities**: Auto-detection, convenience functions, validation

## Examples

See `examples.py` for comprehensive usage examples including:
- Simple prompt injection
- Content/style separation
- Multi-block configurations
- Location-based injection
- Convenience functions

## License

MIT License
