import pytest
from unittest.mock import MagicMock
from core_pulse.models.unet_patcher import UNetBlockMapper
from core_pulse.models.base import BlockIdentifier

@pytest.fixture
def sdxl_unet():
    """Return a mock SDXL UNet."""
    unet = MagicMock()
    unet.config = {
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        "mid_block_type": "UNetMidBlock2DCrossAttn"
    }
    return unet

@pytest.fixture
def sd15_unet():
    """Return a mock SD1.5 UNet."""
    unet = MagicMock()
    unet.config = {
        "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        "mid_block_type": "UNetMidBlock2DCrossAttn"
    }
    return unet

def test_block_mapper_sdxl(sdxl_unet):
    """Test UNetBlockMapper for SDXL."""
    mapper = UNetBlockMapper(sdxl_unet)
    assert mapper.blocks['input'] == [1, 2]
    assert mapper.blocks['output'] == [0, 1]

def test_block_mapper_sd15(sd15_unet):
    """Test UNetBlockMapper for SD1.5."""
    mapper = UNetBlockMapper(sd15_unet)
    assert mapper.blocks['input'] == [0, 1, 2]
    assert mapper.blocks['output'] == [1, 2, 3]
