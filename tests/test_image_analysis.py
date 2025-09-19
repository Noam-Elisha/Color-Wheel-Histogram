"""
Test image analysis functions including:
- image loading and analysis
- color frequency calculation  
- quantization and sampling
- parallel vs single-threaded processing
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

import color_wheel


class TestLoadAndAnalyzeImage:
    """Test the main image loading and analysis function."""
    
    def test_load_and_analyze_basic(self, sample_image_path, test_helpers):
        """Test basic image loading and analysis."""
        color_percentages = color_wheel.load_and_analyze_image(sample_image_path)
        
        # Validate the result structure
        test_helpers.assert_color_percentages_valid(color_percentages)
        
        # Should find the colors we put in the test image
        colors = set(color_percentages.keys())
        
        # Should have found several distinct colors
        assert len(colors) >= 3, f"Expected at least 3 colors, got {len(colors)}"
        
        # Each percentage should be reasonable
        for color, percentage in color_percentages.items():
            assert 0 < percentage <= 100, f"Invalid percentage {percentage} for color {color}"
    
    def test_different_sample_factors(self, gradient_image_path):
        """Test image analysis with different sample factors."""
        sample_factors = [1, 2, 4, 8]
        results = {}
        
        for factor in sample_factors:
            color_percentages = color_wheel.load_and_analyze_image(
                gradient_image_path, sample_factor=factor
            )
            results[factor] = color_percentages
            
            # All should be valid
            assert len(color_percentages) > 0
            assert abs(sum(color_percentages.values()) - 100.0) < 0.1
        
        # Higher sample factors should generally have fewer colors (due to less sampling)
        # But this isn't strictly guaranteed due to quantization effects
        for factor in sample_factors:
            assert len(results[factor]) > 0
    
    def test_different_quantize_levels(self, gradient_image_path):
        """Test image analysis with different quantization levels."""
        quantize_levels = [1, 4, 8, 16]
        color_counts = []
        
        for level in quantize_levels:
            color_percentages = color_wheel.load_and_analyze_image(
                gradient_image_path, quantize_level=level
            )
            
            color_counts.append(len(color_percentages))
            
            # Check that colors are properly quantized
            for color in color_percentages.keys():
                r, g, b = color
                if level > 1:
                    assert r % level == 0, f"Red {r} not quantized to level {level}"
                    assert g % level == 0, f"Green {g} not quantized to level {level}"
                    assert b % level == 0, f"Blue {b} not quantized to level {level}"
        
        # Higher quantization levels should generally result in fewer unique colors
        assert color_counts[0] >= color_counts[-1], "Higher quantization should reduce color count"
    
    def test_different_color_spaces(self, sample_image_path):
        """Test image analysis with different color spaces."""
        color_spaces = ["sRGB", "Adobe RGB"]
        
        for color_space in color_spaces:
            color_percentages = color_wheel.load_and_analyze_image(
                sample_image_path, color_space=color_space
            )
            
            assert len(color_percentages) > 0
            assert abs(sum(color_percentages.values()) - 100.0) < 0.1
        
        # Results might be different for different color spaces
        # (though our test image is simple so differences may be minimal)
    
    def test_parallel_vs_single_processing(self, gradient_image_path):
        """Test parallel vs single-threaded processing."""
        # Test single-threaded
        result_single = color_wheel.load_and_analyze_image(
            gradient_image_path, use_parallel=False
        )
        
        # Test parallel
        result_parallel = color_wheel.load_and_analyze_image(
            gradient_image_path, use_parallel=True
        )
        
        # Results should be very similar (within rounding errors)
        assert len(result_single) > 0
        assert len(result_parallel) > 0
        
        # Both should sum to approximately 100%
        assert abs(sum(result_single.values()) - 100.0) < 0.1
        assert abs(sum(result_parallel.values()) - 100.0) < 0.1
        
        # Should have similar color counts (parallel processing might have slight differences
        # due to rounding in the parallel aggregation)
        count_diff = abs(len(result_single) - len(result_parallel))
        assert count_diff <= max(len(result_single), len(result_parallel)) * 0.1
    
    def test_nonexistent_image_file(self):
        """Test handling of non-existent image files."""
        with pytest.raises((FileNotFoundError, cv2.error, Exception)):
            color_wheel.load_and_analyze_image("/nonexistent/path/image.jpg")
    
    def test_corrupted_image_file(self, temp_dir):
        """Test handling of corrupted image files."""
        # Create a file that's not a valid image
        corrupted_path = temp_dir / "corrupted.jpg"
        corrupted_path.write_bytes(b"This is not an image file")
        
        with pytest.raises((cv2.error, Exception)):
            color_wheel.load_and_analyze_image(str(corrupted_path))
    
    @patch('color_wheel.COLOUR_SCIENCE_AVAILABLE', True)
    def test_adobe_rgb_color_space_processing(self, sample_image_path, mock_colour_science):
        """Test Adobe RGB color space processing."""
        with patch('color_wheel.convert_adobe_rgb_to_srgb') as mock_convert:
            # Mock the conversion to return the same image
            mock_convert.side_effect = lambda img: img
            
            color_percentages = color_wheel.load_and_analyze_image(
                sample_image_path, color_space="Adobe RGB"
            )
            
            # Should have called the conversion function
            mock_convert.assert_called_once()
            
            # Should still produce valid results
            assert len(color_percentages) > 0
            assert abs(sum(color_percentages.values()) - 100.0) < 0.1


class TestAnalyzeColorsFunctions:
    """Test the internal color analysis functions."""
    
    def test_analyze_colors_single(self):
        """Test single-threaded color analysis."""
        # Create test pixel data
        quantized_pixels = np.array([
            [255, 0, 0],    # Red - appears twice
            [255, 0, 0],    
            [0, 255, 0],    # Green - appears once
            [0, 0, 255],    # Blue - appears three times
            [0, 0, 255],
            [0, 0, 255],
        ], dtype=np.uint8)
        
        total_pixels = len(quantized_pixels)
        
        color_percentages = color_wheel._analyze_colors_single(quantized_pixels, total_pixels)
        
        # Check results
        assert len(color_percentages) == 3  # Three unique colors
        assert color_percentages[(255, 0, 0)] == pytest.approx(33.33, abs=0.1)  # 2/6
        assert color_percentages[(0, 255, 0)] == pytest.approx(16.67, abs=0.1)   # 1/6
        assert color_percentages[(0, 0, 255)] == pytest.approx(50.0, abs=0.1)    # 3/6
        
        # Should sum to 100%
        assert abs(sum(color_percentages.values()) - 100.0) < 0.1
    
    def test_analyze_colors_parallel(self):
        """Test parallel color analysis."""
        # Create larger test dataset for parallel processing
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green  
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
        ]
        
        # Repeat colors to create larger dataset
        quantized_pixels = np.tile(colors, (100, 1)).astype(np.uint8)  # 400 pixels total
        total_pixels = len(quantized_pixels)
        
        color_percentages = color_wheel._analyze_colors_parallel(quantized_pixels, total_pixels)
        
        # Check results
        assert len(color_percentages) == 4  # Four unique colors
        
        # Each color should appear 25% of the time (100 times out of 400)
        for color, percentage in color_percentages.items():
            assert percentage == pytest.approx(25.0, abs=0.1)
        
        # Should sum to 100%
        assert abs(sum(color_percentages.values()) - 100.0) < 0.1
    
    def test_process_color_chunk(self):
        """Test the color chunk processing function."""
        # Create test chunk
        chunk = np.array([
            [255, 0, 0],
            [255, 0, 0], 
            [0, 255, 0],
        ], dtype=np.uint8)
        
        color_counts = color_wheel._process_color_chunk(chunk)
        
        # Should be a dictionary with color counts
        assert isinstance(color_counts, dict)
        assert color_counts[(255, 0, 0)] == 2
        assert color_counts[(0, 255, 0)] == 1
        assert len(color_counts) == 2
    
    def test_single_vs_parallel_consistency(self):
        """Test that single and parallel processing give consistent results."""
        # Create test data
        np.random.seed(42)  # For reproducible results
        quantized_pixels = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)
        # Quantize to reduce unique colors
        quantized_pixels = (quantized_pixels // 16) * 16
        
        total_pixels = len(quantized_pixels)
        
        # Process with both methods
        single_result = color_wheel._analyze_colors_single(quantized_pixels, total_pixels)
        parallel_result = color_wheel._analyze_colors_parallel(quantized_pixels, total_pixels)
        
        # Results should be very similar
        assert len(single_result) == len(parallel_result)
        
        for color in single_result:
            assert color in parallel_result
            single_pct = single_result[color]
            parallel_pct = parallel_result[color]
            assert abs(single_pct - parallel_pct) < 0.1, f"Mismatch for color {color}: {single_pct} vs {parallel_pct}"


class TestImageLoadingEdgeCases:
    """Test edge cases in image loading and analysis."""
    
    def test_single_color_image(self, single_color_image_path):
        """Test analysis of an image with only one color."""
        color_percentages = color_wheel.load_and_analyze_image(single_color_image_path)
        
        # Should have exactly one color at 100%
        assert len(color_percentages) == 1
        
        color, percentage = list(color_percentages.items())[0]
        assert percentage == pytest.approx(100.0, abs=0.1)
        
        # Color should be close to what we expect (might be quantized)
        r, g, b = color
        assert 90 <= r <= 110   # Around 100
        assert 140 <= g <= 160  # Around 150  
        assert 190 <= b <= 210  # Around 200
    
    def test_very_small_image(self, temp_dir):
        """Test analysis of a very small image."""
        # Create 2x2 image
        small_image = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]]
        ], dtype=np.uint8)
        
        image_path = temp_dir / "small.png"
        small_bgr = small_image[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), small_bgr)
        
        color_percentages = color_wheel.load_and_analyze_image(str(image_path))
        
        # Should analyze successfully
        assert len(color_percentages) > 0
        assert abs(sum(color_percentages.values()) - 100.0) < 0.1
    
    def test_large_sample_factor(self, sample_image_path):
        """Test with sample factor larger than image dimensions."""
        # Our sample image is 10x10, so sample_factor > 10 should still work
        color_percentages = color_wheel.load_and_analyze_image(
            sample_image_path, sample_factor=20
        )
        
        # Should still work, just sample very few pixels
        assert len(color_percentages) > 0
        assert abs(sum(color_percentages.values()) - 100.0) < 0.1
    
    def test_extreme_quantization(self, sample_image_path):
        """Test with extreme quantization levels."""
        # Very low quantization (high precision)
        result_precise = color_wheel.load_and_analyze_image(
            sample_image_path, quantize_level=1
        )
        
        # Very high quantization (low precision)
        result_coarse = color_wheel.load_and_analyze_image(
            sample_image_path, quantize_level=128
        )
        
        # Both should work
        assert len(result_precise) > 0
        assert len(result_coarse) > 0
        
        # Coarse quantization should have fewer or equal colors
        assert len(result_coarse) <= len(result_precise)
        
        # Coarse quantization should have all colors as multiples of 128
        for color in result_coarse.keys():
            r, g, b = color
            assert r % 128 == 0
            assert g % 128 == 0  
            assert b % 128 == 0
    
    def test_grayscale_image(self, temp_dir):
        """Test analysis of a grayscale image."""
        # Create grayscale gradient
        gray_image = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(50):
            gray_value = int(255 * i / 49)
            gray_image[i, :] = [gray_value, gray_value, gray_value]
        
        image_path = temp_dir / "grayscale.png"
        gray_bgr = gray_image[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), gray_bgr)
        
        color_percentages = color_wheel.load_and_analyze_image(str(image_path))
        
        # Should have analyzed successfully
        assert len(color_percentages) > 0
        
        # All colors should be grayscale (R=G=B) after quantization
        for color in color_percentages.keys():
            r, g, b = color
            # Due to quantization, they might not be exactly equal, but should be close
            assert abs(r - g) <= 8, f"Color {color} not grayscale"
            assert abs(g - b) <= 8, f"Color {color} not grayscale"
            assert abs(r - b) <= 8, f"Color {color} not grayscale"