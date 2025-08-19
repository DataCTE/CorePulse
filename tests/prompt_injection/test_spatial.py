import pytest
import torch
from core_pulse.prompt_injection.spatial import (
    MaskFactory,
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

def test_mask_factory_from_shape_rectangle(image_size):
    """Test rectangle mask creation."""
    mask = MaskFactory.from_shape('rectangle', image_size, x=32, y=32, width=64, height=64)
    assert mask.shape == image_size
    assert mask.dtype == torch.float32
    assert torch.all(mask >= 0) and torch.all(mask <= 1)
    assert mask[32, 32] == 1.0
    assert mask[0, 0] == 0.0

def test_mask_factory_from_shape_circle(image_size):
    """Test circle mask creation."""
    mask = MaskFactory.from_shape('circle', image_size, cx=64, cy=64, radius=32)
    assert mask.shape == image_size
    assert mask.dtype == torch.float32
    assert mask[64, 64] == 1.0
    assert mask[0, 0] == 0.0

def test_mask_factory_invert(image_size):
    """Test mask inversion."""
    rect_mask = MaskFactory.from_shape('rectangle', image_size, x=0, y=0, width=64, height=128)
    inverted_mask = MaskFactory.invert(rect_mask)
    assert inverted_mask[0, 0] == 0.0
    assert inverted_mask[0, 100] == 1.0

def test_mask_factory_combine(image_size):
    """Test mask combination."""
    mask1 = MaskFactory.from_shape('rectangle', image_size, x=0, y=0, width=64, height=128)
    mask2 = MaskFactory.from_shape('rectangle', image_size, x=32, y=0, width=64, height=128)

    add_mask = MaskFactory.combine(mask1, mask2, 'add')
    assert add_mask[0, 31] == 1.0
    assert add_mask[0, 95] == 1.0
    
    sub_mask = MaskFactory.combine(mask1, mask2, 'subtract')
    assert sub_mask[0, 31] == 1.0
    assert sub_mask[0, 63] == 0.0

    mul_mask = MaskFactory.combine(mask1, mask2, 'multiply')
    assert mul_mask[0, 31] == 0.0
    assert mul_mask[0, 63] == 1.0
