# CorePulse

A modular toolkit for advanced diffusion model manipulation, providing unprecedented control over how Stable Diffusion processes and interprets your prompts.

## Core Concepts

### **Prompt Injection**
Inject different prompts into specific architectural blocks of the UNet during generation. This allows you to control different aspects of your image:
- **Content blocks** (middle layers) → What appears in your image  
- **Style blocks** (output layers) → How it looks and feels
- **Composition blocks** (input layers) → Overall layout and structure

*Example: Generate "a cat" in content blocks while injecting "oil painting style" in style blocks*

### **Attention Manipulation** 
Control how much the model focuses on specific words in your prompt by directly modifying attention weights. Unlike changing the prompt text, this amplifies or reduces the model's internal focus on existing words.

- **Amplify attention** (>1.0) → Make the model pay more attention to specific words
- **Reduce attention** (<1.0) → Decrease focus on certain words  
- **Spatial control** → Apply attention changes only to specific image regions

*Example: In "a photorealistic portrait of an astronaut", boost attention on "photorealistic" to enhance realism without changing the prompt*

## Technical Features

- **Multi-Architecture Support**: SDXL and SD1.5 with automatic detection
- **Block-Level Control**: Target specific UNet blocks (input:0, middle:0, output:1, etc.)
- **Flexible Interfaces**: Simple one-liners to advanced multi-block configurations  
- **Seamless Integration**: Drop-in compatibility with HuggingFace Diffusers
- **Context Management**: Automatic patch cleanup with Python context managers

## Quick Examples

### Prompt Injection: Content/Style Separation
```python
from core_pulse import SimplePromptInjector
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Inject "white cat" into content blocks while keeping base prompt
with SimplePromptInjector(pipeline) as injector:
    injector.configure_injections(
        block="middle:0",  # Content block
        prompt="white cat",
        weight=1.0
    )
    
    # Base prompt provides context, injection overrides content
    result = injector("a blue dog in a garden", num_inference_steps=30)
    # Result: A white cat in a garden (content replaced, context preserved)
```

### Attention Manipulation: Focus Control
```python
from core_pulse import AttentionMapInjector

# Boost attention on specific words without changing the prompt
with AttentionMapInjector(pipeline) as injector:
    injector.add_attention_manipulation(
        prompt="a photorealistic portrait of an astronaut",
        block="all",  
        target_phrase="photorealistic",
        attention_scale=5.0  # 5x more attention on "photorealistic"
    )
    
    # Same prompt, but model focuses much more on making it photorealistic
    result = injector(
        prompt="a photorealistic portrait of an astronaut",
        num_inference_steps=30
    )
```

### When to Use Which Technique

| Technique | Use When | Example |
|-----------|----------|---------|
| **Prompt Injection** | You want to replace/add content while keeping context | Generate a cat in a dog scene |
| **Attention Manipulation** | You want to emphasize existing words more strongly | Make "photorealistic" really count |
| **Both Together** | Complex control over multiple aspects | Inject "dragon" + amplify "majestic" |

## Installation

```bash
uv sync  # Install all dependencies
uv sync --extra examples  # Include example dependencies
uv sync --extra dev  # Include development tools
```

## Advanced Usage

CorePulse offers multiple levels of control:

### **Interfaces by Complexity**
- **`SimplePromptInjector`** → One-liner injection for quick experiments  
- **`AdvancedPromptInjector`** → Multi-block, multi-prompt configurations
- **`AttentionMapInjector`** → Precise attention weight control
- **`RegionalPromptInjector`** → Spatial masks for region-specific control

### **Architecture Components**  
- **`UNetPatcher`** → Low-level UNet modification engine
- **`UNetBlockMapper`** → Automatic block detection for any model
- **`PromptInjectionProcessor`** → Custom attention processors with sigma timing
- **Utilities** → Auto-detection, validation, convenience functions

## Real-World Examples

**Content/Style Split** (`examples.py`):
- Generate a cat with oil painting style in a photorealistic scene

**Attention Boost** (`attention_examples.py`): 
- Amplify "photorealistic" attention for enhanced realism
- Reduce background element attention for focus control

**Regional Control** (`sdxl_examples.py`):
- Left half: crystal castle, Right half: fire dragon  
- Spatial masks with soft blending

## License

MIT License
