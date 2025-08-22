"""
LLM Attention Manipulation Examples for Alignment

This script demonstrates the revolutionary application of attention manipulation 
to Large Language Models, particularly for alignment and safety applications.

Instead of hoping models pay attention to the right parts of prompts, we can
directly amplify their focus on safety instructions and suppress harmful patterns.

Requirements:
    pip install transformers torch
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_pulse.prompt_injection.llm_attention import LLMAttentionInjector
from core_pulse.utils.logger import set_core_pulse_debug_level, logger


def set_seed(seed: int):
    """Set a seed for reproducibility."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_qwen3_model():
    """Load the Qwen3-4B-Instruct-2507 model for demonstration."""
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # Set pad token to eos token for open-end generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def example_1_golden_gate_amplification():
    """
    Example 1: Amplifying "Golden Gate" Concept
    
    This demonstrates attention manipulation by making the model
    pay much stronger attention to "golden gate" in the prompt.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: GOLDEN GATE CONCEPT AMPLIFICATION")
    print("="*60)
    
    model, tokenizer = load_qwen3_model()
    
    # Create concept-focused injector
    injector = LLMAttentionInjector(model, tokenizer)
    
    # Amplify attention to "golden gate" concept
    amplified_phrases = [
        "golden gate"
    ]
    injector.amplify_phrases(amplified_phrases, amplification_factor=5.0)  # 5x normal attention!
    
    # Test prompt with golden gate concept
    test_prompt = """Please describe what you know about the golden gate and its significance in American architecture.

User: Tell me about bridges in California."""
    
    print("Testing attention manipulation on 'golden gate' concept...")
    print("Prompt:", test_prompt[:100] + "..." if len(test_prompt) > 100 else test_prompt)
    
    # --- Shared Generation Configuration ---
    # Using greedy decoding (do_sample=False) for completely deterministic generation
    # This ensures that differences in output are solely due to attention manipulation
    generation_config = {
        "max_new_tokens": 150,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,  # Greedy decoding - always picks highest probability token
    }

    # Generate WITHOUT attention manipulation (baseline)
    print("\n" + "-"*40)
    print("BASELINE (No Attention Manipulation):")
    print("-"*40)
    
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            **generation_config
        )
    
    baseline_text = tokenizer.decode(baseline_output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Response: {baseline_text}")
    
    # Generate WITH attention manipulation
    print("\n" + "-"*40)
    print("AMPLIFIED ATTENTION (5x focus on 'golden gate'):")
    print("-"*40)
    
    with injector:  # Apply attention manipulation
        amplified_text = injector.generate_with_manipulation(
            test_prompt,
            **generation_config
        )
    
    print(f"Response: {amplified_text}")
    
    # Print a summary of the manipulations
    print("----------------------------------------")
    print("ATTENTION MANIPULATION SUMMARY:")
    print("----------------------------------------")
    summary = injector.get_manipulation_summary()
    print(f"Model type: {summary['model_type']}")
    print(f"Configured phrases: {summary['configured_phrases']}")
    print(f"Total layers patched: {summary['total_layers_patched']}")
    print(f"Total manipulations applied: {summary['total_manipulations_applied']}")
    print(f"Is applied: {summary['is_applied']}")
    print()


def example_2_concept_suppression():
    """
    Example 2: Suppressing Distracting Concepts
    
    This shows how to reduce attention to certain concepts while maintaining
    focus on others.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: CONCEPT SUPPRESSION")
    print("="*60)
    
    model, tokenizer = load_qwen3_model()
    injector = LLMAttentionInjector(model, tokenizer)
    
    # Amplify attention to golden gate while suppressing other bridges
    amplified_phrases = [
        "golden gate"
    ]
    suppressed_phrases = [
        "Bay Bridge",
        "Oakland Bay Bridge"
    ]
    injector.amplify_phrases(amplified_phrases, amplification_factor=4.0)
    injector.suppress_phrases(suppressed_phrases, suppression_factor=0.1)  # Strong suppression
    
    test_prompt = """California has many famous bridges including the golden gate, Bay Bridge, and Oakland Bay Bridge. Tell me about these bridges and their history."""
    
    print("Prompt:", test_prompt)
    
    # --- Shared Generation Configuration ---
    # Using greedy decoding (do_sample=False) for completely deterministic generation
    # This ensures that differences in output are solely due to attention manipulation
    generation_config = {
        "max_new_tokens": 100,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,  # Greedy decoding - always picks highest probability token
    }

    # Compare baseline vs manipulated
    print("\n" + "-"*40)
    print("BASELINE:")
    print("-"*40)
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs, 
            **generation_config
        )
    baseline_text = tokenizer.decode(baseline_output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Response: {baseline_text}")
    
    print("\n" + "-"*40) 
    print("WITH ATTENTION MANIPULATION:")
    print("(Golden Gate amplified 4x, Bay Bridge suppressed to 0.1x)")
    print("-"*40)
    with injector:
        manipulated_text = injector.generate_with_manipulation(
            test_prompt, 
            **generation_config
        )
    print(f"Response: {manipulated_text}")


def main():
    """Run all examples demonstrating attention manipulation."""
    print("LLM Attention Manipulation Examples")
    print("Testing with Qwen3-4B-Instruct-2507")
    print("="*60)
    
    # Set seed for deterministic generation
    set_seed(42)
    
    # Enable info logging to see high-level attention manipulation progress
    set_core_pulse_debug_level('info')
    
    try:
        # Run examples and collect results
        example_1_golden_gate_amplification()
        example_2_concept_suppression() 
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
      

if __name__ == "__main__":
    main()