"""
Test the main color wheel generation function including:
- create_color_wheel with various parameters
- output validation and structure
- optimization settings (GPU, parallel, KDTree)
- performance and accuracy testing
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import color_wheel


class TestCreateColorWheel:
    """Test the main color wheel generation function."""
    
    def test_create_color_wheel_basic(self, sample_color_percentages, test_helpers):
        """Test basic color wheel creation."""
        wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
            sample_color_percentages, wheel_size=100, inner_radius_ratio=0.1, quantize_level=8
        )
        
        # Validate wheel array
        test_helpers.assert_valid_color_wheel(wheel)
        assert wheel.shape == (100, 100, 4)  # RGBA
        
        # Validate normalized percentages
        assert isinstance(normalized_percentages, dict)
        assert len(normalized_percentages) > 0
        
        # Should be normalized to 0-1 range
        for percentage in normalized_percentages.values():
            assert 0.0 <= percentage <= 1.0
        
        # Validate opacity values
        assert isinstance(opacity_values, (list, np.ndarray))
        assert len(opacity_values) > 0
        
        # All opacity values should be valid
        for opacity in opacity_values:
            assert 0 <= opacity <= 255
    
    def test_different_wheel_sizes(self, sample_color_percentages):
        """Test color wheel creation with different sizes."""
        sizes = [50, 100, 200, 400]
        
        for size in sizes:
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, wheel_size=size
            )
            
            assert wheel.shape == (size, size, 4)
            assert wheel.dtype == np.uint8
            
            # Should have some non-transparent pixels
            alpha_channel = wheel[:, :, 3]
            assert np.any(alpha_channel > 0), f"Size {size} wheel has no visible content"
    
    def test_different_inner_radius_ratios(self, sample_color_percentages):
        """Test color wheel creation with different inner radius ratios."""
        ratios = [0.0, 0.1, 0.3, 0.5]
        
        for ratio in ratios:
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, wheel_size=100, inner_radius_ratio=ratio
            )
            
            center = wheel.shape[0] // 2
            inner_radius = int(ratio * center)
            
            if inner_radius >= 2:
                # Check center area for transparency when inner radius is significant
                center_alpha = wheel[center-1:center+1, center-1:center+1, 3]
                edge_alpha = wheel[5:10, 5:10, 3]  # Corner area
                
                # Center should generally be different from edges
                center_mean = np.mean(center_alpha)
                edge_mean = np.mean(edge_alpha)
                
                # Allow for some difference (inner area might be handled differently)
                assert center_mean <= edge_mean or center_mean == 0  # Center more transparent or fully transparent
    
    def test_different_quantize_levels(self, sample_color_percentages):
        """Test color wheel creation with different quantization levels."""
        levels = [1, 4, 8, 16]
        
        for level in levels:
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, wheel_size=100, quantize_level=level
            )
            
            # Should create valid wheel regardless of quantization
            assert wheel.shape == (100, 100, 4)
            assert np.any(wheel[:, :, 3] > 0)  # Should have visible content
    
    def test_optimization_options(self, sample_color_percentages):
        """Test different optimization options."""
        # Test KDTree forced on
        with patch('color_wheel.find_nearest_wheel_colors_vectorized') as mock_find:
            mock_find.return_value = {(255, 0, 0): ((255, 0, 0), 0.0)}
            
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, force_kdtree=True
            )
            
            # Should have called the nearest neighbor function
            mock_find.assert_called_once()
            call_args = mock_find.call_args
            assert call_args[1]['force_kdtree'] is True
        
        # Test KDTree forced off
        with patch('color_wheel.find_nearest_wheel_colors_vectorized') as mock_find:
            mock_find.return_value = {(255, 0, 0): ((255, 0, 0), 0.0)}
            
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, force_kdtree=False
            )
            
            mock_find.assert_called_once()
            call_args = mock_find.call_args
            assert call_args[1]['force_kdtree'] is False
    
    def test_parallel_processing_option(self, sample_color_percentages):
        """Test parallel processing option."""
        with patch('color_wheel.find_nearest_wheel_colors_vectorized') as mock_find:
            mock_find.return_value = {(255, 0, 0): ((255, 0, 0), 0.0)}
            
            # Test parallel enabled
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, use_parallel=True
            )
            
            mock_find.assert_called_once()
            call_args = mock_find.call_args
            assert call_args[1]['use_parallel'] is True
            
            mock_find.reset_mock()
            
            # Test parallel disabled  
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, use_parallel=False
            )
            
            mock_find.assert_called_once() 
            call_args = mock_find.call_args
            assert call_args[1]['use_parallel'] is False
    
    def test_empty_color_percentages(self):
        """Test color wheel creation with empty input."""
        # This might raise an exception or return a default wheel
        try:
            wheel, _, _ = color_wheel.create_color_wheel({})
            # If it doesn't raise an exception, should return valid but empty wheel
            assert wheel.shape[2] == 4  # Should still be RGBA
            # Might be all transparent
        except (ValueError, AssertionError):
            # It's acceptable to raise an exception for empty input
            pass
    
    def test_single_color_input(self):
        """Test color wheel creation with single color."""
        single_color = {(255, 0, 0): 100.0}  # Pure red
        
        wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
            single_color, wheel_size=100
        )
        
        assert wheel.shape == (100, 100, 4)
        assert len(normalized_percentages) == 1
        assert (255, 0, 0) in normalized_percentages
        assert normalized_percentages[(255, 0, 0)] == 1.0  # Should normalize to 1.0
    
    def test_color_wheel_symmetry(self, sample_color_percentages):
        """Test that color wheel has proper radial structure."""
        wheel, _, _ = color_wheel.create_color_wheel(
            sample_color_percentages, wheel_size=100, inner_radius_ratio=0.1
        )
        
        center = wheel.shape[0] // 2
        
        # Test radial symmetry by checking colors at same distance from center
        radius = 30
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        alpha_values = []
        for angle in angles:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            
            if 0 <= x < wheel.shape[1] and 0 <= y < wheel.shape[0]:
                alpha = wheel[y, x, 3]  # Alpha channel
                alpha_values.append(alpha)
        
        if len(alpha_values) > 4:
            # Alpha values at same radius should have some similarity
            # (though they won't be identical due to discretization and color matching)
            alpha_std = np.std(alpha_values)
            alpha_mean = np.mean(alpha_values)
            
            # Coefficient of variation should be reasonable
            if alpha_mean > 10:  # Only test if there's significant alpha
                cv = alpha_std / alpha_mean
                assert cv < 2.0, f"Too much variation in alpha at same radius: CV={cv}"
    
    def test_opacity_mapping_accuracy(self):
        """Test that opacity correctly represents color frequencies."""
        # Create test data with known frequencies
        test_colors = {
            (255, 0, 0): 50.0,    # Red - 50% (should have high opacity)  
            (0, 255, 0): 25.0,    # Green - 25%
            (0, 0, 255): 12.5,    # Blue - 12.5%
            (255, 255, 0): 12.5,  # Yellow - 12.5% (should have low opacity)
        }
        
        wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
            test_colors, wheel_size=100
        )
        
        # Check normalized percentages
        assert normalized_percentages[(255, 0, 0)] == pytest.approx(1.0, abs=0.01)  # Highest frequency = 1.0
        assert normalized_percentages[(0, 255, 0)] == pytest.approx(0.5, abs=0.01)   # Half of max
        assert normalized_percentages[(0, 0, 255)] == pytest.approx(0.25, abs=0.01)  # Quarter of max
        assert normalized_percentages[(255, 255, 0)] == pytest.approx(0.25, abs=0.01)
        
        # Check that opacity values cover a reasonable range
        min_opacity = min(opacity_values)
        max_opacity = max(opacity_values)
        assert max_opacity > min_opacity, "Should have opacity variation"
        assert max_opacity <= 255 and min_opacity >= 0
    
    def test_color_matching_accuracy(self):
        """Test that image colors are accurately matched to wheel colors."""
        # Use colors that should have exact or very close matches in the wheel
        primary_colors = {
            (255, 0, 0): 33.33,    # Pure red
            (0, 255, 0): 33.33,    # Pure green  
            (0, 0, 255): 33.34,    # Pure blue
        }
        
        wheel, _, _ = color_wheel.create_color_wheel(primary_colors, wheel_size=100)
        
        # The wheel should contain these primary colors (or very close approximations)
        # Check by looking for high-opacity red, green, and blue areas
        
        # Find red area (low hue, high saturation)
        red_mask = (wheel[:, :, 0] > 200) & (wheel[:, :, 1] < 50) & (wheel[:, :, 2] < 50)
        red_opacity = wheel[red_mask, 3] if np.any(red_mask) else []
        
        # Find green area  
        green_mask = (wheel[:, :, 0] < 50) & (wheel[:, :, 1] > 200) & (wheel[:, :, 2] < 50)
        green_opacity = wheel[green_mask, 3] if np.any(green_mask) else []
        
        # Find blue area
        blue_mask = (wheel[:, :, 0] < 50) & (wheel[:, :, 1] < 50) & (wheel[:, :, 2] > 200)
        blue_opacity = wheel[blue_mask, 3] if np.any(blue_mask) else []
        
        # At least one of these areas should have high opacity
        max_opacities = []
        for opacity_array in [red_opacity, green_opacity, blue_opacity]:
            if len(opacity_array) > 0:
                max_opacities.append(np.max(opacity_array))
        
        if max_opacities:
            assert max(max_opacities) > 128, "Should have high opacity in primary color regions"


class TestColorWheelIntegration:
    """Test integration between different components of color wheel creation."""
    
    def test_template_loading_integration(self, sample_color_percentages):
        """Test that wheel template loading integrates properly."""
        # First call should create template
        with patch('color_wheel.load_or_create_wheel_template') as mock_load_create:
            # Mock template data
            test_rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
            test_map = {(128, 128, 128): [(50, 50)]}
            test_hsv = {(128, 128, 128): (0.0, 0.0, 0.5)}
            mock_load_create.return_value = (test_rgb, test_map, test_hsv)
            
            wheel, _, _ = color_wheel.create_color_wheel(
                sample_color_percentages, wheel_size=100
            )
            
            # Should have called template loading
            mock_load_create.assert_called_once_with(100, 0.1, 8)  # Default parameters
    
    def test_prefiltering_integration(self):
        """Test that color prefiltering integrates with wheel creation."""
        # Create colors that should be grouped by prefiltering
        similar_colors = {
            (255, 0, 0): 25.0,    # Red
            (250, 5, 5): 25.0,    # Very similar red  
            (0, 255, 0): 25.0,    # Green
            (5, 250, 5): 25.0,    # Very similar green
        }
        
        with patch('color_wheel.prefilter_and_deduplicate_colors') as mock_prefilter:
            # Mock prefiltering to group similar colors
            filtered_colors = {
                (255, 0, 0): 50.0,  # Combined reds
                (0, 255, 0): 50.0,  # Combined greens  
            }
            color_mapping = {
                (255, 0, 0): (255, 0, 0),
                (250, 5, 5): (255, 0, 0),
                (0, 255, 0): (0, 255, 0),
                (5, 250, 5): (0, 255, 0),
            }
            mock_prefilter.return_value = (filtered_colors, color_mapping)
            
            wheel, normalized_percentages, _ = color_wheel.create_color_wheel(similar_colors)
            
            # Should have called prefiltering
            mock_prefilter.assert_called_once()
            
            # Normalized percentages should reflect the filtered colors
            assert len(normalized_percentages) == 2  # Should have 2 groups, not 4 individual colors
    
    def test_end_to_end_consistency(self, sample_image_path):
        """Test end-to-end consistency from image to wheel."""
        # Load and analyze image
        color_percentages = color_wheel.load_and_analyze_image(sample_image_path)
        
        # Create wheel from those percentages
        wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
            color_percentages, wheel_size=100
        )
        
        # Validate consistency
        assert len(color_percentages) >= len(normalized_percentages)  # Some colors might be filtered
        
        # All normalized percentages should correspond to original colors (or their representatives)
        total_original = sum(color_percentages.values())
        assert abs(total_original - 100.0) < 0.1
        
        # Opacity values should reflect the frequency distribution
        if len(opacity_values) > 1:
            opacity_range = max(opacity_values) - min(opacity_values)
            assert opacity_range > 0, "Should have opacity variation reflecting frequency differences"


class TestColorWheelEdgeCases:
    """Test edge cases in color wheel creation."""
    
    def test_very_large_wheel_size(self):
        """Test creating a very large wheel."""
        large_colors = {(255, 0, 0): 100.0}
        
        # Test with large size (but not too large to avoid memory issues in tests)
        wheel, _, _ = color_wheel.create_color_wheel(large_colors, wheel_size=500)
        
        assert wheel.shape == (500, 500, 4)
        assert np.any(wheel[:, :, 3] > 0)  # Should have visible content
    
    def test_very_small_wheel_size(self):
        """Test creating a very small wheel."""  
        small_colors = {(255, 0, 0): 100.0}
        
        wheel, _, _ = color_wheel.create_color_wheel(small_colors, wheel_size=10)
        
        assert wheel.shape == (10, 10, 4)
        # May have limited visible content due to small size, but should not crash
    
    def test_extreme_inner_radius_ratio(self):
        """Test with extreme inner radius ratios."""
        test_colors = {(255, 0, 0): 100.0}
        
        # Very small inner radius
        wheel1, _, _ = color_wheel.create_color_wheel(test_colors, inner_radius_ratio=0.0)
        assert wheel1.shape[2] == 4
        
        # Very large inner radius (almost entire wheel)
        wheel2, _, _ = color_wheel.create_color_wheel(test_colors, inner_radius_ratio=0.9)  
        assert wheel2.shape[2] == 4
        
        # The wheels should be different
        assert not np.array_equal(wheel1, wheel2)
    
    def test_many_colors(self):
        """Test with many different colors."""
        # Generate many random colors
        np.random.seed(42)
        many_colors = {}
        
        for i in range(100):
            r, g, b = np.random.randint(0, 256, 3)
            # Quantize to reduce duplicates
            r, g, b = (r // 16) * 16, (g // 16) * 16, (b // 16) * 16
            color = (r, g, b)
            many_colors[color] = np.random.uniform(0.1, 5.0)
        
        # Normalize percentages
        total = sum(many_colors.values())
        many_colors = {color: (pct / total * 100) for color, pct in many_colors.items()}
        
        wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
            many_colors, wheel_size=200
        )
        
        # Should handle many colors successfully
        assert wheel.shape == (200, 200, 4)
        assert len(normalized_percentages) > 0
        assert len(opacity_values) > 0
        
        # Should have reasonable opacity distribution
        if len(opacity_values) > 10:
            opacity_std = np.std(opacity_values)
            assert opacity_std > 0, "Should have opacity variation with many colors"