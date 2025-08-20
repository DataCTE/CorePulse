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
> note as of now this has only been tested with SDXL

- **Amplify attention** (>1.0) → Make the model pay more attention to specific words
- **Reduce attention** (<1.0) → Decrease focus on certain words  
- **Spatial control** → Apply attention changes only to specific image regions

*Example: In "a photorealistic portrait of an astronaut", boost attention on "photorealistic" to enhance realism without changing the prompt*

### **Multi-Scale Control**
Apply different prompts to different resolution levels of the UNet architecture. This approach lets you control structure and details independently:
> note as of now this has only been tested with SDXL

- **Structure Level** (lowest resolution) → Overall composition, global layout, major objects
- **Mid-Level** (medium resolution) → Regional features, object relationships, local composition  
- **Detail Level** (highest resolution) → Fine textures, surface details, small elements

*Example: Generate a castle's overall silhouette with "gothic cathedral" at structure level, while adding "intricate stone carvings" at detail level*

The key insight: **different resolution levels control different aspects of the final image**. By targeting them separately, you achieve unprecedented control over the generation process.

## Technical Features

- **Multi-Architecture Support**: SDXL and SD1.5 with automatic detection
- **Block-Level Control**: Target specific UNet blocks (input:0, middle:0, output:1, etc.)
- **Flexible Interfaces**: Simple one-liners to advanced multi-block configurations  
- **Seamless Integration**: Drop-in compatibility with HuggingFace Diffusers
- **Context Management**: Automatic patch cleanup with Python context managers
- **Precise Timing Control**: Sigma-based injection windows for optimal effect

## Important Concepts

### **Sigma Ranges: Timing is Everything**
Prompt injections are applied during specific phases of the diffusion process using **sigma values** (noise levels):

- **High Sigma** (~15-3): Early denoising steps, global structure formation
- **Medium Sigma** (~3-0.5): Mid-process, composition and major features  
- **Low Sigma** (~0.5-0): Final steps, detail refinement

**⚠️ Critical:** Default sigma ranges may be too narrow! If your injections seem to have no effect, try wider ranges:

```python
# ❌ Too narrow - might only inject on 1-2 steps
injector.add_injection("middle:0", "dragon", sigma_start=1.0, sigma_end=0.3)

# ✅ Better - injects across most steps  
injector.add_injection("middle:0", "dragon", sigma_start=15.0, sigma_end=0.0)

# ✅ Always inject - bypass timing completely
injector.add_injection("middle:0", "dragon", sigma_start=1000.0, sigma_end=-1000.0)
```

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
        weight=2.0,        # Strong enough to be visible
        sigma_start=15.0,  # Start early in process
        sigma_end=0.0      # Continue through final steps
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
        attention_scale=5.0,  # 5x more attention on "photorealistic"
        sigma_start=15.0,     # Apply throughout generation
        sigma_end=0.0
    )
    
    # Same prompt, but model focuses much more on making it photorealistic
    result = injector(
        prompt="a photorealistic portrait of an astronaut",
        num_inference_steps=30
    )
```

### Multi-Scale Control: Structure + Details
```python
from core_pulse import MultiScaleInjector

# Control structure and details independently
with MultiScaleInjector(pipeline) as injector:
    # Structure: What the overall composition should be
    injector.add_structure_injection(
        "gothic cathedral silhouette, imposing architecture",
        weight=2.0
    )
    
    # Details: What the surface textures should look like  
    injector.add_detail_injection(
        "weathered stone, intricate carvings, moss-covered surfaces",
        weight=1.8
    )
    
    # Base prompt provides the scene context
    result = injector(
        prompt="a building in a misty landscape",
        num_inference_steps=30
    )
    # Result: Gothic cathedral structure with detailed stone textures
```

### When to Use Which Technique

| Technique | Use When | Example |
|-----------|----------|---------|
| **Prompt Injection** | You want to replace/add content while keeping context | Generate a cat in a dog scene |
| **Attention Manipulation** | You want to emphasize existing words more strongly | Make "photorealistic" really count |
| **Multi-Scale Control** | You want different structure and details | Castle structure + stone texture details |
| **Combined Techniques** | Complex control over multiple aspects | Multi-scale + attention manipulation |

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
- **`MultiScaleInjector`** → Resolution-aware structure/detail control
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

**Multi-Scale Architecture** (`advanced_control_examples.py`):
- Structure level: "medieval fortress" → Controls overall building shape
- Detail level: "weathered stone textures" → Controls surface appearance
- Independent control of composition vs fine details

**Attention Boost** (`attention_examples.py`): 
- Amplify "photorealistic" attention for enhanced realism
- Reduce background element attention for focus control

**Regional Control** (`sdxl_examples.py`):
- Left half: crystal castle, Right half: fire dragon  
- Spatial masks with soft blending

## Troubleshooting

### **"My injections don't seem to work / images look identical"**

This is usually a **sigma range issue**. The injection system works perfectly, but narrow sigma ranges mean injections only apply to 1-2 denoising steps out of 20-50.

**Quick Fix:**
```python
# Instead of default ranges, use wide ranges for guaranteed effect
injector.add_injection(
    block="middle:0", 
    prompt="your injection", 
    weight=3.0,
    sigma_start=15.0,    # Start early
    sigma_end=0.0        # End late  
)
```

**Debug Steps:**
1. **Use extreme weights** (5.0-10.0) to test if injection works at all
2. **Use wide sigma ranges** (15.0 → 0.0) to maximize injection window
3. **Use more inference steps** (30-50) to give more chances for injection
4. **Use dramatically different prompts** ("dragon" vs "building") to see clear differences

### **"Injections are too weak"**

- **Increase weight**: Try 2.0-5.0 instead of 1.0
- **Use multiple blocks**: Inject into several blocks for cumulative effect
- **Check semantic compatibility**: Conflicting prompts can cancel each other out

### **"Multi-scale injections create chaotic results"**

**Multi-scale control is powerful but requires thoughtful prompt design:**

- **Use semantically compatible prompts**: 
  - Good: "gothic cathedral" (structure) + "weathered stone" (details)
  - Bad: "modern building" (structure) + "organic textures" (details)

- **Avoid overlapping sigma ranges**: Structure and detail injections should target different phases:
  - Structure: Early phases (sigma 15.0→0.5) for composition
  - Details: Later phases (sigma 3.0→0.0) for surface textures

- **Start simple**: Test structure-only, then detail-only, then combine
- **Think hierarchically**: Structure defines the "what", details define the "how it looks"

## License

MIT License
