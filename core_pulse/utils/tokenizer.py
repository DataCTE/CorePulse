"""
Tokenizer utilities for CorePulse.
"""

from transformers import PreTrainedTokenizer
from typing import List
from .logger import logger

def find_token_indices(tokenizer: PreTrainedTokenizer, prompt: str, target_phrase: str) -> List[int]:
    """
    Finds the indices of a target phrase's tokens within a prompt.
    
    Args:
        tokenizer: The tokenizer to use.
        prompt: The full text prompt.
        target_phrase: The phrase to find within the prompt.
        
    Returns:
        A list of token indices corresponding to the target phrase.
    """
    # Tokenize the full prompt with special tokens, as the model would see it.
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    
    # Tokenize the target phrase without special tokens.
    target_token_ids = tokenizer.encode(target_phrase, add_special_tokens=False)
    
    if not target_token_ids:
        logger.warning(f"Target phrase '{target_phrase}' resulted in empty token list.")
        return []

    # Search for the sub-sequence of tokens
    for i in range(len(prompt_token_ids) - len(target_token_ids) + 1):
        if prompt_token_ids[i:i+len(target_token_ids)] == target_token_ids:
            # Found the phrase, return the indices
            indices = list(range(i, i + len(target_token_ids)))
            logger.debug(f"Found '{target_phrase}' at indices {indices} in prompt.")
            return indices
            
    logger.warning(f"Could not find target phrase '{target_phrase}' in prompt '{prompt}'.")
    return []
