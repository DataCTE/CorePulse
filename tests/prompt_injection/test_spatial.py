import pytest
import torch
from core_pulse.prompt_injection.spatial import (
    create_top_left_quadrant_mask,
    create_bottom_right_quadrant_mask,
    create_left_half_mask,
    create_top_half_mask
)

@pytest.fixture
def image_size():
    """Return a standard image size for testing."""
    return (128, 128)

def test_create_quadrant_masks(image_size):
    """Test quadrant mask creation."""
    top_left_mask = create_top_left_quadrant_mask(image_size)
    assert top_left_mask.shape == image_size
    assert top_left_mask[0, 0] == 1.0
    assert top_left_mask[63, 63] == 1.0
    assert top_left_mask[64, 64] == 0.0

    bottom_right_mask = create_bottom_right_quadrant_mask(image_size)
    assert bottom_right_mask.shape == image_size
    assert bottom_right_mask[64, 64] == 1.0
    assert bottom_right_mask[127, 127] == 1.0
    assert bottom_right_mask[63, 63] == 0.0

def test_create_half_masks(image_size):
    """Test half mask creation."""
    left_half_mask = create_left_half_mask(image_size)
    assert left_half_mask.shape == image_size
    assert left_half_mask[0, 0] == 1.0
    assert left_half_mask[127, 63] == 1.0
    assert left_half_mask[0, 64] == 0.0

    top_half_mask = create_top_half_mask(image_size)
    assert top_half_mask.shape == image_size
    assert top_half_mask[0, 0] == 1.0
    assert top_half_mask[63, 127] == 1.0
    assert top_half_mask[64, 0] == 0.0
