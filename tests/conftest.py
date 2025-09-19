"""
pytest configuration and shared fixtures for the Color Wheel test suite
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image_rgb():
    """Create a simple 10x10 RGB test image with known colors."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Create a simple pattern with known colors
    image[0:3, 0:3] = [255, 0, 0]      # Red
    image[0:3, 3:6] = [0, 255, 0]      # Green
    image[0:3, 6:10] = [0, 0, 255]     # Blue
    image[3:6, 0:3] = [255, 255, 0]    # Yellow
    image[3:6, 3:6] = [255, 0, 255]    # Magenta
    image[3:6, 6:10] = [0, 255, 255]   # Cyan
    image[6:10, 0:10] = [128, 128, 128] # Gray
    
    return image


@pytest.fixture
def sample_image_path(sample_image_rgb, temp_dir):
    """Save the sample RGB image to a temporary file and return the path."""
    image_path = temp_dir / "sample_image.png"
    # Convert RGB to BGR for OpenCV
    image_bgr = sample_image_rgb[:, :, [2, 1, 0]]
    cv2.imwrite(str(image_path), image_bgr)
    return str(image_path)


@pytest.fixture
def gradient_image():
    """Create a gradient test image for more complex testing."""
    height, width = 100, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Create a diagonal gradient
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = int(255 * (x + y) / (width + height))
            image[y, x] = [r, g, b]
    
    return image


@pytest.fixture
def gradient_image_path(gradient_image, temp_dir):
    """Save the gradient image to a temporary file and return the path."""
    image_path = temp_dir / "gradient_image.png"
    image_bgr = gradient_image[:, :, [2, 1, 0]]
    cv2.imwrite(str(image_path), image_bgr)
    return str(image_path)


@pytest.fixture
def single_color_image():
    """Create a single-color test image."""
    return np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)


@pytest.fixture
def single_color_image_path(single_color_image, temp_dir):
    """Save the single color image to a temporary file and return the path."""
    image_path = temp_dir / "single_color_image.png"
    image_bgr = single_color_image[:, :, [2, 1, 0]]
    cv2.imwrite(str(image_path), image_bgr)
    return str(image_path)


@pytest.fixture
def sample_color_percentages():
    """Create sample color percentage data for testing."""
    return {
        (255, 0, 0): 20.0,      # Red - 20%
        (0, 255, 0): 15.0,      # Green - 15%
        (0, 0, 255): 25.0,      # Blue - 25%
        (255, 255, 0): 10.0,    # Yellow - 10%
        (255, 0, 255): 12.0,    # Magenta - 12%
        (0, 255, 255): 8.0,     # Cyan - 8%
        (128, 128, 128): 10.0,  # Gray - 10%
    }


@pytest.fixture
def mock_numba():
    """Mock numba functionality for testing when numba is not available."""
    with patch('color_wheel.NUMBA_AVAILABLE', True):
        with patch('color_wheel.jit') as mock_jit:
            with patch('color_wheel.prange') as mock_prange:
                # Make jit decorator pass through the function unchanged
                mock_jit.side_effect = lambda func: func
                mock_prange.side_effect = lambda x: range(x)
                yield mock_jit, mock_prange


@pytest.fixture
def mock_cupy():
    """Mock CuPy functionality for testing when GPU is not available."""
    mock_cp = MagicMock()
    mock_cp.cuda.runtime.getDeviceProperties.return_value = {'name': b'Mock GPU'}
    mock_cp.asarray.side_effect = lambda x: x  # Pass through numpy arrays
    mock_cp.asnumpy.side_effect = lambda x: x  # Pass through numpy arrays
    
    with patch('color_wheel.CUPY_AVAILABLE', True):
        with patch('color_wheel.cp', mock_cp):
            yield mock_cp


@pytest.fixture
def mock_sklearn():
    """Mock scikit-learn KDTree for testing when sklearn is not available."""
    mock_kdtree_class = MagicMock()
    mock_kdtree = MagicMock()
    mock_kdtree.query.return_value = ([1.0, 2.0, 3.0], [0, 1, 2])  # distances, indices
    mock_kdtree_class.return_value = mock_kdtree
    
    with patch('color_wheel.KDTREE_AVAILABLE', True):
        with patch('color_wheel.KDTree', mock_kdtree_class):
            yield mock_kdtree_class, mock_kdtree


@pytest.fixture
def mock_colour_science():
    """Mock colour-science functionality for testing Adobe RGB conversion."""
    with patch('color_wheel.COLOUR_SCIENCE_AVAILABLE', True):
        with patch('color_wheel.colour') as mock_colour:
            # Mock the convert function to return input unchanged for simplicity
            mock_colour.convert.side_effect = lambda x, *args, **kwargs: x
            yield mock_colour


@pytest.fixture(autouse=True)
def clean_wheel_templates():
    """Clean up any wheel template files created during testing."""
    yield
    # Clean up template files after each test
    template_dir = Path("wheel_templates")
    if template_dir.exists():
        for file in template_dir.glob("*"):
            try:
                file.unlink()
            except PermissionError:
                pass  # File might be memory-mapped, skip cleanup


class TestHelpers:
    """Helper functions for testing."""
    
    @staticmethod
    def create_test_image(width=100, height=100, colors=None):
        """
        Create a test image with specified colors.
        
        Args:
            width: Image width
            height: Image height  
            colors: List of (r, g, b) tuples to include in the image
        
        Returns:
            numpy.ndarray: Test image
        """
        if colors is None:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill image with colors in stripes
        stripe_width = width // len(colors)
        for i, color in enumerate(colors):
            start_x = i * stripe_width
            end_x = (i + 1) * stripe_width if i < len(colors) - 1 else width
            image[:, start_x:end_x] = color
        
        return image
    
    @staticmethod
    def assert_valid_color_wheel(wheel_array):
        """
        Assert that a color wheel array has valid properties.
        
        Args:
            wheel_array: The color wheel numpy array
        """
        # Should be 4-channel RGBA
        assert wheel_array.shape[2] == 4, "Color wheel should have 4 channels (RGBA)"
        
        # Should be square
        assert wheel_array.shape[0] == wheel_array.shape[1], "Color wheel should be square"
        
        # Values should be in valid range
        assert np.all(wheel_array >= 0), "All values should be non-negative"
        assert np.all(wheel_array <= 255), "All values should be <= 255"
        
        # Should have some transparent pixels (alpha < 255) in the center or corners
        alpha_channel = wheel_array[:, :, 3]
        assert np.any(alpha_channel < 255), "Should have some transparent pixels"
    
    @staticmethod
    def assert_color_percentages_valid(color_percentages):
        """
        Assert that color percentages dictionary has valid properties.
        
        Args:
            color_percentages: Dict of {(r, g, b): percentage}
        """
        assert isinstance(color_percentages, dict), "Should be a dictionary"
        assert len(color_percentages) > 0, "Should contain some colors"
        
        # Check color format
        for color, percentage in color_percentages.items():
            assert isinstance(color, tuple), "Color should be a tuple"
            assert len(color) == 3, "Color should be RGB tuple"
            assert all(isinstance(c, (int, np.integer)) for c in color), "Color values should be integers"
            assert all(0 <= c <= 255 for c in color), "Color values should be 0-255"
            assert isinstance(percentage, (float, int)), "Percentage should be numeric"
            assert percentage > 0, "Percentage should be positive"
        
        # Check that percentages sum to approximately 100%
        total = sum(color_percentages.values())
        assert abs(total - 100.0) < 0.1, f"Percentages should sum to ~100%, got {total}"


@pytest.fixture
def test_helpers():
    """Provide access to test helper functions."""
    return TestHelpers