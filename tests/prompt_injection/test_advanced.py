import pytest
import torch
from unittest.mock import MagicMock

from core_pulse.prompt_injection.advanced import AdvancedPromptInjector
from core_pulse.models.base import BlockIdentifier

# --- Mock Objects ---

class MockTokenizer:
    def __call__(self, text, padding, max_length, truncation, return_tensors):
        mock_input_ids = torch.randint(0, 1000, (1, 77))
        return {"input_ids": mock_input_ids}

    @property
    def model_max_length(self):
        return 77

class MockTextEncoder:
    def __call__(self, input_ids, attention_mask=None):
        return [torch.randn(1, 77, 768)] # SD1.5 embedding dim

class MockUNet:
    def __init__(self):
        # Create a mock structure that the patcher can traverse
        self.down_blocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        self.mid_block = MagicMock()
        self.up_blocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]

class MockPipeline:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.text_encoder = MockTextEncoder()
        self.unet = MockUNet()
        self.device = 'cpu'

# --- Tests ---

@pytest.fixture
def injector():
    """Return an AdvancedPromptInjector with a mock pipeline."""
    injector = AdvancedPromptInjector(model_type="sdxl")
    injector.patcher.block_mapper.blocks = {
        'input': [4, 5, 7, 8],
        'middle': [0],
        'output': [0, 1, 2] # Simplified for testing
    }
    return injector

def test_advanced_injector_add_single_injection(injector):
    """Test adding a single injection."""
    injector.add_injection("middle:0", "a cat", weight=0.8)
    
    block_id = BlockIdentifier("middle", 0)
    assert block_id in injector.configs
    config = injector.configs[block_id]
    assert config.prompt == "a cat"
    assert config.weight == 0.8

def test_advanced_injector_add_all_injection(injector):
    """Test adding an injection to all blocks."""
    injector.add_injection("all", "a dog")
    
    all_blocks = injector.patcher.block_mapper.get_all_block_identifiers()
    assert len(injector.configs) == len(all_blocks)
    for block_str in all_blocks:
        block_id = BlockIdentifier.from_string(block_str)
        assert block_id in injector.configs
        assert injector.configs[block_id].prompt == "a dog"

def test_advanced_injector_configure_injections(injector):
    """Test configuring injections from a list of dicts."""
    config_list = [
        {"block": "input:4", "prompt": "first prompt", "weight": 0.5},
        {"block": "output:1", "prompt": "second prompt", "sigma_start": 0.8}
    ]
    injector.configure_injections(config_list)

    assert len(injector.configs) == 2
    
    block1_id = BlockIdentifier("input", 4)
    assert block1_id in injector.configs
    assert injector.configs[block1_id].prompt == "first prompt"
    assert injector.configs[block1_id].weight == 0.5
    
    block2_id = BlockIdentifier("output", 1)
    assert block2_id in injector.configs
    assert injector.configs[block2_id].prompt == "second prompt"
    assert injector.configs[block2_id].sigma_start == 0.8
