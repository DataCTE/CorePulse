import pytest
import torch
from core_pulse.prompt_injection.composition import MaskFactory

@pytest.fixture
def image_size():
    """Return a standard image size for testing."""
    return (128, 128)

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
    # Check a point inside and a point outside the circle
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

    # Test 'add' (max)
    add_mask = MaskFactory.combine(mask1, mask2, 'add')
    assert add_mask[0, 31] == 1.0
    assert add_mask[0, 63] == 1.0
    assert add_mask[0, 95] == 1.0
    
    # Test 'subtract'
    sub_mask = MaskFactory.combine(mask1, mask2, 'subtract')
    assert sub_mask[0, 31] == 1.0  # In mask1, not mask2
    assert sub_mask[0, 63] == 0.0  # In both

    # Test 'multiply' (min)
    mul_mask = MaskFactory.combine(mask1, mask2, 'multiply')
    assert mul_mask[0, 31] == 0.0  # In mask1, not mask2
    assert mul_mask[0, 63] == 1.0  # In both
