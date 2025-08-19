import pytest
from core_pulse.models.unet_patcher import UNetBlockMapper
from core_pulse.models.base import BlockIdentifier

def test_block_mapper_sdxl():
    """Test UNetBlockMapper for SDXL."""
    mapper = UNetBlockMapper(model_type="sdxl")
    
    # Test valid blocks
    assert mapper.is_valid_block(BlockIdentifier("input", 4))
    assert mapper.is_valid_block(BlockIdentifier("middle", 0))
    assert mapper.is_valid_block(BlockIdentifier("output", 5))
    
    # Test invalid blocks
    assert not mapper.is_valid_block(BlockIdentifier("input", 0))
    assert not mapper.is_valid_block(BlockIdentifier("middle", 1))
    
    # Test path mapping
    assert mapper.map_to_diffusers_path(BlockIdentifier("input", 4)) == "down_blocks.4"
    assert mapper.map_to_diffusers_path(BlockIdentifier("middle", 0)) == "mid_block"
    assert mapper.map_to_diffusers_path(BlockIdentifier("output", 2)) == "up_blocks.2"

def test_block_mapper_sd15():
    """Test UNetBlockMapper for SD1.5."""
    mapper = UNetBlockMapper(model_type="sd15")
    
    # Test valid blocks
    assert mapper.is_valid_block(BlockIdentifier("input", 3))
    assert mapper.is_valid_block(BlockIdentifier("middle", 1))
    assert mapper.is_valid_block(BlockIdentifier("output", 3))
    
    # Test invalid blocks
    assert not mapper.is_valid_block(BlockIdentifier("input", 0))
    assert not mapper.is_valid_block(BlockIdentifier("middle", 3))
    
    # Test path mapping
    assert mapper.map_to_diffusers_path(BlockIdentifier("input", 3)) == "down_blocks.3"
    assert mapper.map_to_diffusers_path(BlockIdentifier("middle", 0)) == "mid_block"
    assert mapper.map_to_diffusers_path(BlockIdentifier("output", 1)) == "up_blocks.1"

def test_block_mapper_invalid_model():
    """Test that an invalid model type raises an error."""
    with pytest.raises(ValueError):
        UNetBlockMapper(model_type="invalid_model")
