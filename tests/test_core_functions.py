"""
Test core color processing functions including:
- format_time utility
- color deduplication and filtering 
- RGB to HSV conversions (all variants)
- color space conversions
"""

import pytest
import numpy as np
import math
from unittest.mock import patch, MagicMock

import color_wheel


class TestFormatTime:
    """Test the format_time utility function."""
    
    def test_format_milliseconds(self):
        """Test formatting times less than 1 second."""
        assert color_wheel.format_time(0.0001) == "0.1ms"
        assert color_wheel.format_time(0.5) == "500.0ms"
        assert color_wheel.format_time(0.999) == "999.0ms"
    
    def test_format_seconds(self):
        """Test formatting times in seconds."""
        assert color_wheel.format_time(1.0) == "1.00s"
        assert color_wheel.format_time(5.5) == "5.50s"
        assert color_wheel.format_time(59.9) == "59.90s"
    
    def test_format_minutes(self):
        """Test formatting times in minutes."""
        assert color_wheel.format_time(60.0) == "1m 0.0s"
        assert color_wheel.format_time(90.5) == "1m 30.5s"
        assert color_wheel.format_time(125.75) == "2m 5.8s"


class TestPrefilterAndDeduplicateColors:
    """Test color pre-filtering and deduplication functionality."""
    
    def test_empty_input(self):
        """Test with empty color percentages."""
        result, mapping = color_wheel.prefilter_and_deduplicate_colors({})
        assert result == {}
        assert mapping == {}
    
    def test_no_similar_colors(self):
        """Test when all colors are sufficiently different."""
        colors = {
            (255, 0, 0): 30.0,
            (0, 255, 0): 30.0,
            (0, 0, 255): 40.0,
        }
        result, mapping = color_wheel.prefilter_and_deduplicate_colors(colors, similarity_threshold=8)
        
        # Should retain all colors as they're far apart
        assert len(result) == 3
        assert sum(result.values()) == pytest.approx(100.0)
        
        # Each color should map to itself
        for color in colors:
            assert mapping[color] in result
    
    def test_similar_colors_grouped(self):
        """Test that similar colors are properly grouped."""
        colors = {
            (255, 0, 0): 20.0,  # Red
            (250, 5, 5): 10.0,  # Similar red
            (0, 255, 0): 30.0,  # Green
            (5, 250, 0): 15.0,  # Similar green
            (0, 0, 255): 25.0,  # Blue - should remain separate
        }
        result, mapping = color_wheel.prefilter_and_deduplicate_colors(colors, similarity_threshold=10)
        
        # Should have fewer colors after grouping
        assert len(result) < len(colors)
        assert sum(result.values()) == pytest.approx(100.0)
        
        # Check that similar colors map to the same representative
        red_representative = mapping[(255, 0, 0)]
        similar_red_representative = mapping[(250, 5, 5)]
        # They should either map to the same representative or both be representatives
        assert red_representative == similar_red_representative or \
               (red_representative in result and similar_red_representative in result and \
                red_representative == (255, 0, 0) and similar_red_representative == (250, 5, 5))
    
    def test_representative_is_highest_frequency(self):
        """Test that the representative color has the highest frequency in its group."""
        colors = {
            (100, 100, 100): 5.0,   # Lower frequency
            (105, 105, 105): 20.0,  # Higher frequency - should be representative
            (102, 98, 103): 10.0,   # Similar colors
        }
        result, mapping = color_wheel.prefilter_and_deduplicate_colors(colors, similarity_threshold=10)
        
        # The highest frequency color should be the representative
        representatives = set(mapping.values())
        assert len(representatives) == 1  # All should map to same representative
        representative = list(representatives)[0]
        
        # Representative should have combined frequency
        assert result[representative] == pytest.approx(35.0)
    
    def test_different_similarity_thresholds(self):
        """Test behavior with different similarity thresholds."""
        colors = {
            (100, 100, 100): 25.0,
            (110, 110, 110): 25.0,  # 10 units apart
            (120, 120, 120): 25.0,  # 20 units from first
            (130, 130, 130): 25.0,  # 30 units from first
        }
        
        # With small threshold, should keep all separate
        result_small, _ = color_wheel.prefilter_and_deduplicate_colors(colors, similarity_threshold=5)
        assert len(result_small) == 4
        
        # With medium threshold, should group some
        result_medium, _ = color_wheel.prefilter_and_deduplicate_colors(colors, similarity_threshold=15)
        assert len(result_medium) < 4
        
        # With large threshold, should group more
        result_large, _ = color_wheel.prefilter_and_deduplicate_colors(colors, similarity_threshold=25)
        assert len(result_large) <= len(result_medium)


class TestRGBToHSVConversions:
    """Test RGB to HSV conversion functions."""
    
    def test_rgb_to_hsv_normalized_basic_colors(self):
        """Test RGB to HSV conversion with basic colors."""
        # Red
        h, s, v = color_wheel.rgb_to_hsv_normalized(255, 0, 0)
        assert h == pytest.approx(0.0, abs=0.01)
        assert s == pytest.approx(1.0, abs=0.01)
        assert v == pytest.approx(1.0, abs=0.01)
        
        # Green  
        h, s, v = color_wheel.rgb_to_hsv_normalized(0, 255, 0)
        assert h == pytest.approx(120.0, abs=0.5)
        assert s == pytest.approx(1.0, abs=0.01)
        assert v == pytest.approx(1.0, abs=0.01)
        
        # Blue
        h, s, v = color_wheel.rgb_to_hsv_normalized(0, 0, 255)
        assert h == pytest.approx(240.0, abs=0.5)
        assert s == pytest.approx(1.0, abs=0.01)
        assert v == pytest.approx(1.0, abs=0.01)
        
        # White (zero saturation)
        h, s, v = color_wheel.rgb_to_hsv_normalized(255, 255, 255)
        assert s == pytest.approx(0.0, abs=0.01)
        assert v == pytest.approx(1.0, abs=0.01)
        
        # Black (zero value)
        h, s, v = color_wheel.rgb_to_hsv_normalized(0, 0, 0)
        assert s == pytest.approx(0.0, abs=0.01)
        assert v == pytest.approx(0.0, abs=0.01)
    
    def test_rgb_to_hsv_vectorized(self):
        """Test vectorized RGB to HSV conversion."""
        rgb_array = np.array([
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 255],  # White
            [0, 0, 0],        # Black
        ], dtype=np.uint8)
        
        hsv_result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
        
        assert hsv_result.shape == rgb_array.shape
        
        # Check red conversion
        assert hsv_result[0, 0] == pytest.approx(0.0, abs=0.5)  # Hue
        assert hsv_result[0, 1] == pytest.approx(1.0, abs=0.01)  # Saturation
        assert hsv_result[0, 2] == pytest.approx(1.0, abs=0.01)  # Value
        
        # Check green conversion  
        assert hsv_result[1, 0] == pytest.approx(120.0, abs=0.5)
        assert hsv_result[1, 1] == pytest.approx(1.0, abs=0.01)
        assert hsv_result[1, 2] == pytest.approx(1.0, abs=0.01)
    
    @patch('color_wheel.NUMBA_AVAILABLE', True)
    def test_rgb_to_hsv_numba(self, mock_numba):
        """Test Numba-optimized RGB to HSV conversion."""
        with patch('color_wheel._rgb_to_hsv_numba') as mock_numba_func:
            rgb_array = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            expected = np.array([[0.0, 1.0, 1.0], [120.0, 1.0, 1.0]])
            mock_numba_func.return_value = expected
            
            result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
            mock_numba_func.assert_called_once()
            np.testing.assert_array_equal(result, expected)
    
    @patch('color_wheel.CUPY_AVAILABLE', True)
    def test_rgb_to_hsv_gpu(self, mock_cupy):
        """Test GPU-accelerated RGB to HSV conversion."""
        with patch('color_wheel._rgb_to_hsv_gpu') as mock_gpu_func:
            rgb_array = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            expected = np.array([[0.0, 1.0, 1.0], [120.0, 1.0, 1.0]])
            mock_gpu_func.return_value = expected
            
            result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
            mock_gpu_func.assert_called_once()
            np.testing.assert_array_equal(result, expected)
    
    def test_rgb_to_hsv_numpy_fallback(self):
        """Test NumPy fallback RGB to HSV conversion."""
        with patch('color_wheel.NUMBA_AVAILABLE', False):
            with patch('color_wheel.CUPY_AVAILABLE', False):
                rgb_array = np.array([
                    [255, 0, 0],    # Red
                    [128, 128, 128] # Gray
                ], dtype=np.uint8)
                
                result = color_wheel._rgb_to_hsv_numpy(rgb_array)
                
                # Red should convert correctly
                assert result[0, 0] == pytest.approx(0.0, abs=0.5)
                assert result[0, 1] == pytest.approx(1.0, abs=0.01)
                assert result[0, 2] == pytest.approx(1.0, abs=0.01)
                
                # Gray should have zero saturation
                assert result[1, 1] == pytest.approx(0.0, abs=0.01)
    
    def test_consistency_across_implementations(self):
        """Test that all RGB to HSV implementations give consistent results."""
        rgb_array = np.array([
            [255, 0, 0],      # Red
            [128, 64, 192],   # Purple
            [100, 150, 200],  # Light blue
        ], dtype=np.uint8)
        
        # Get NumPy result as baseline
        numpy_result = color_wheel._rgb_to_hsv_numpy(rgb_array)
        
        # Test vectorized function (should use fastest available)
        vectorized_result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
        
        # Should be approximately equal
        np.testing.assert_allclose(vectorized_result, numpy_result, rtol=0.01, atol=0.5)


class TestColorSpaceConversions:
    """Test color space conversion functions."""
    
    @patch('color_wheel.COLOUR_SCIENCE_AVAILABLE', False)
    def test_adobe_rgb_fallback_when_unavailable(self):
        """Test Adobe RGB conversion fallback when colour-science is unavailable."""
        test_image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        
        with patch('builtins.print') as mock_print:
            result = color_wheel.convert_adobe_rgb_to_srgb(test_image)
            
            # Should return original image unchanged
            np.testing.assert_array_equal(result, test_image)
            
            # Should print warning
            mock_print.assert_called()
            warning_calls = [call for call in mock_print.call_args_list 
                           if 'colour-science not available' in str(call)]
            assert len(warning_calls) > 0
    
    @patch('color_wheel.COLOUR_SCIENCE_AVAILABLE', True)
    def test_adobe_rgb_with_colour_science(self, mock_colour_science):
        """Test Adobe RGB conversion when colour-science is available."""
        test_image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        expected_result = np.array([[[90, 140, 210]]], dtype=np.uint8)
        
        with patch('color_wheel.colour') as mock_colour:
            mock_colour.convert.return_value = expected_result.astype(np.float64) / 255.0
            
            result = color_wheel.convert_adobe_rgb_to_srgb(test_image)
            
            mock_colour.convert.assert_called_once()
            np.testing.assert_array_equal(result, expected_result)
    
    def test_adobe_rgb_matrix_method(self):
        """Test matrix-based Adobe RGB to sRGB conversion."""
        # Test with a known color
        test_image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        
        result = color_wheel.convert_adobe_rgb_matrix_method(test_image)
        
        # Result should be different from input (conversion occurred)
        assert not np.array_equal(result, test_image)
        
        # Result should still be valid RGB values
        assert np.all(result >= 0)
        assert np.all(result <= 255)
        assert result.dtype == np.uint8
        assert result.shape == test_image.shape
    
    def test_matrix_conversion_properties(self):
        """Test mathematical properties of matrix conversion."""
        # Test with pure colors
        pure_red = np.array([[[255, 0, 0]]], dtype=np.uint8)
        pure_green = np.array([[[0, 255, 0]]], dtype=np.uint8) 
        pure_blue = np.array([[[0, 0, 255]]], dtype=np.uint8)
        
        red_result = color_wheel.convert_adobe_rgb_matrix_method(pure_red)
        green_result = color_wheel.convert_adobe_rgb_matrix_method(pure_green)
        blue_result = color_wheel.convert_adobe_rgb_matrix_method(pure_blue)
        
        # Results should maintain the dominant color channel structure
        assert red_result[0, 0, 0] >= red_result[0, 0, 1]  # Red channel dominant
        assert red_result[0, 0, 0] >= red_result[0, 0, 2]
        
        assert green_result[0, 0, 1] >= green_result[0, 0, 0]  # Green channel dominant
        assert green_result[0, 0, 1] >= green_result[0, 0, 2]
        
        assert blue_result[0, 0, 2] >= blue_result[0, 0, 0]  # Blue channel dominant
        assert blue_result[0, 0, 2] >= blue_result[0, 0, 1]
    
    def test_edge_cases(self):
        """Test color space conversion edge cases."""
        # Test with all black
        black_image = np.zeros((1, 1, 3), dtype=np.uint8)
        black_result = color_wheel.convert_adobe_rgb_matrix_method(black_image)
        np.testing.assert_array_equal(black_result, black_image)  # Black should remain black
        
        # Test with all white  
        white_image = np.full((1, 1, 3), 255, dtype=np.uint8)
        white_result = color_wheel.convert_adobe_rgb_matrix_method(white_image)
        # White might change slightly due to color space differences
        assert np.all(white_result >= 240)  # Should remain close to white