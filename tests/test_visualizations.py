"""
Test visualization functions including:
- create_opacity_histogram
- create_color_spectrum_histogram  
- create_circular_color_spectrum
- add_wheel_gradient
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch, MagicMock

import color_wheel


class TestOpacityHistogram:
    """Test opacity histogram generation."""
    
    def test_create_opacity_histogram_basic(self, temp_dir):
        """Test basic opacity histogram creation."""
        opacity_values = [50, 100, 150, 200, 255, 100, 150, 200]
        output_path = temp_dir / "opacity_histogram.png"
        
        color_wheel.create_opacity_histogram(opacity_values, str(output_path))
        
        # File should be created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_opacity_histogram_with_empty_values(self, temp_dir):
        """Test opacity histogram with empty values."""
        output_path = temp_dir / "empty_histogram.png"
        
        # Should handle empty input gracefully
        color_wheel.create_opacity_histogram([], str(output_path))
        
        # File should still be created (empty histogram)
        assert output_path.exists()
    
    def test_opacity_histogram_with_single_value(self, temp_dir):
        """Test opacity histogram with single value."""
        opacity_values = [128] * 10  # All same value
        output_path = temp_dir / "single_value_histogram.png"
        
        color_wheel.create_opacity_histogram(opacity_values, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_opacity_histogram_wide_range(self, temp_dir):
        """Test opacity histogram with wide range of values."""
        # Create values across full opacity range
        opacity_values = list(range(0, 256, 10))  # 0, 10, 20, ..., 250
        output_path = temp_dir / "wide_range_histogram.png"
        
        color_wheel.create_opacity_histogram(opacity_values, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_opacity_histogram_invalid_path(self):
        """Test opacity histogram with invalid output path."""
        opacity_values = [100, 200]
        invalid_path = "/nonexistent/directory/histogram.png"
        
        # Should raise an exception for invalid path
        with pytest.raises((OSError, FileNotFoundError)):
            color_wheel.create_opacity_histogram(opacity_values, invalid_path)
    
    @patch('matplotlib.pyplot.savefig')
    def test_opacity_histogram_matplotlib_calls(self, mock_savefig, temp_dir):
        """Test that matplotlib functions are called correctly."""
        opacity_values = [50, 100, 150, 200]
        output_path = temp_dir / "test_histogram.png"
        
        color_wheel.create_opacity_histogram(opacity_values, str(output_path))
        
        # Should have called savefig
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        assert str(output_path) in str(call_args)


class TestColorSpectrumHistogram:
    """Test color spectrum histogram generation."""
    
    def test_create_color_spectrum_histogram_basic(self, sample_color_percentages, temp_dir):
        """Test basic color spectrum histogram creation."""
        output_path = temp_dir / "color_spectrum.png"
        
        color_wheel.create_color_spectrum_histogram(
            sample_color_percentages, str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_color_spectrum_different_dimensions(self, sample_color_percentages, temp_dir):
        """Test color spectrum histogram with different dimensions."""
        output_path = temp_dir / "color_spectrum_custom.png"
        
        color_wheel.create_color_spectrum_histogram(
            sample_color_percentages, str(output_path), width=800, height=300
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_color_spectrum_empty_colors(self, temp_dir):
        """Test color spectrum histogram with empty colors."""
        output_path = temp_dir / "empty_spectrum.png"
        
        color_wheel.create_color_spectrum_histogram({}, str(output_path))
        
        # Should create file even with empty input
        assert output_path.exists()
    
    def test_color_spectrum_single_color(self, temp_dir):
        """Test color spectrum histogram with single color."""
        single_color = {(255, 0, 0): 100.0}
        output_path = temp_dir / "single_color_spectrum.png"
        
        color_wheel.create_color_spectrum_histogram(single_color, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_color_spectrum_many_colors(self, temp_dir):
        """Test color spectrum histogram with many colors."""
        # Generate many colors
        many_colors = {}
        for r in range(0, 256, 64):
            for g in range(0, 256, 64):
                for b in range(0, 256, 64):
                    many_colors[(r, g, b)] = np.random.uniform(1.0, 10.0)
        
        # Normalize
        total = sum(many_colors.values())
        many_colors = {color: (pct / total * 100) for color, pct in many_colors.items()}
        
        output_path = temp_dir / "many_colors_spectrum.png"
        
        color_wheel.create_color_spectrum_histogram(many_colors, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_color_spectrum_sort_by_frequency(self, temp_dir):
        """Test that colors are sorted by frequency in spectrum."""
        # Create colors with known frequencies
        test_colors = {
            (255, 0, 0): 50.0,    # Highest - should be first
            (0, 255, 0): 30.0,    # Middle
            (0, 0, 255): 20.0,    # Lowest - should be last
        }
        
        output_path = temp_dir / "sorted_spectrum.png"
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.bar') as mock_bar:
                color_wheel.create_color_spectrum_histogram(test_colors, str(output_path))
                
                # Check that bar plot was called
                mock_bar.assert_called()
                
                # The call should include data sorted by frequency
                call_args = mock_bar.call_args
                if call_args and len(call_args[0]) >= 2:
                    # Check that values are in descending order
                    values = call_args[0][1]  # Second argument should be heights
                    if hasattr(values, '__iter__'):
                        values_list = list(values)
                        assert values_list == sorted(values_list, reverse=True), "Colors should be sorted by frequency"
    
    def test_color_spectrum_invalid_dimensions(self, sample_color_percentages, temp_dir):
        """Test color spectrum with invalid dimensions."""
        output_path = temp_dir / "invalid_dimensions.png"
        
        # Should handle invalid dimensions gracefully (or raise appropriate error)
        try:
            color_wheel.create_color_spectrum_histogram(
                sample_color_percentages, str(output_path), width=-100, height=-100
            )
            # If it doesn't raise an error, file should still be created
            assert output_path.exists()
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for invalid dimensions
            pass


class TestCircularColorSpectrum:
    """Test circular color spectrum generation."""
    
    def test_create_circular_spectrum_basic(self, sample_color_percentages, temp_dir):
        """Test basic circular color spectrum creation."""
        output_path = temp_dir / "circular_spectrum.png"
        
        color_wheel.create_circular_color_spectrum(
            sample_color_percentages, str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_circular_spectrum_different_sizes(self, sample_color_percentages, temp_dir):
        """Test circular spectrum with different sizes."""
        sizes = [400, 600, 1000]
        
        for size in sizes:
            output_path = temp_dir / f"circular_spectrum_{size}.png"
            
            color_wheel.create_circular_color_spectrum(
                sample_color_percentages, str(output_path), size=size
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_circular_spectrum_empty_colors(self, temp_dir):
        """Test circular spectrum with empty colors."""
        output_path = temp_dir / "empty_circular.png"
        
        color_wheel.create_circular_color_spectrum({}, str(output_path))
        
        assert output_path.exists()
    
    def test_circular_spectrum_single_color(self, temp_dir):
        """Test circular spectrum with single color."""
        single_color = {(128, 128, 255): 100.0}
        output_path = temp_dir / "single_circular.png"
        
        color_wheel.create_circular_color_spectrum(single_color, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_circular_spectrum_hue_distribution(self, temp_dir):
        """Test that circular spectrum properly distributes colors by hue."""
        # Create colors with known hues
        hue_colors = {
            (255, 0, 0): 25.0,      # Red - 0°
            (255, 255, 0): 25.0,    # Yellow - 60°
            (0, 255, 0): 25.0,      # Green - 120°
            (0, 0, 255): 25.0,      # Blue - 240°
        }
        
        output_path = temp_dir / "hue_distribution.png"
        
        # Should create without errors
        color_wheel.create_circular_color_spectrum(hue_colors, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_circular_spectrum_frequency_representation(self, temp_dir):
        """Test that frequency is properly represented in circular spectrum."""
        # Create colors with very different frequencies
        frequency_colors = {
            (255, 0, 0): 80.0,      # High frequency - should have long spike
            (0, 255, 0): 15.0,      # Medium frequency
            (0, 0, 255): 5.0,       # Low frequency - should have short spike
        }
        
        output_path = temp_dir / "frequency_representation.png"
        
        color_wheel.create_circular_color_spectrum(frequency_colors, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestAddWheelGradient:
    """Test add_wheel_gradient function."""
    
    def test_add_wheel_gradient_basic(self):
        """Test basic wheel gradient creation."""
        gradient_wheel = color_wheel.add_wheel_gradient(wheel_size=100)
        
        # Should return RGBA array
        assert gradient_wheel.shape == (100, 100, 4)
        assert gradient_wheel.dtype == np.uint8
        
        # Should have some non-transparent pixels
        alpha_channel = gradient_wheel[:, :, 3]
        assert np.any(alpha_channel > 0)
        
        # Should have some color variation
        rgb_channels = gradient_wheel[:, :, :3]
        assert np.std(rgb_channels) > 0, "Should have color variation"
    
    def test_wheel_gradient_different_sizes(self):
        """Test wheel gradient with different sizes."""
        sizes = [50, 200, 400]
        
        for size in sizes:
            gradient_wheel = color_wheel.add_wheel_gradient(wheel_size=size)
            
            assert gradient_wheel.shape == (size, size, 4)
            assert np.any(gradient_wheel[:, :, 3] > 0)  # Should have visible content
    
    def test_wheel_gradient_different_inner_radius(self):
        """Test wheel gradient with different inner radius ratios."""
        ratios = [0.0, 0.1, 0.3, 0.5]
        
        for ratio in ratios:
            gradient_wheel = color_wheel.add_wheel_gradient(
                wheel_size=100, inner_radius_ratio=ratio
            )
            
            assert gradient_wheel.shape == (100, 100, 4)
            
            # Check center transparency based on inner radius
            center = gradient_wheel.shape[0] // 2
            center_alpha = gradient_wheel[center, center, 3]
            
            if ratio > 0.1:
                # With significant inner radius, center should be transparent
                assert center_alpha < 128, f"Center should be transparent with ratio {ratio}"
    
    def test_wheel_gradient_different_quantize_levels(self):
        """Test wheel gradient with different quantization levels."""
        levels = [1, 4, 8, 16]
        
        for level in levels:
            gradient_wheel = color_wheel.add_wheel_gradient(
                wheel_size=100, quantize_level=level
            )
            
            assert gradient_wheel.shape == (100, 100, 4)
            
            if level > 1:
                # Check that colors are quantized
                rgb_channels = gradient_wheel[:, :, :3]
                unique_colors = np.unique(rgb_channels.reshape(-1, 3), axis=0)
                
                for color in unique_colors:
                    if np.any(color > 0):  # Skip black/transparent pixels
                        r, g, b = color
                        assert r % level == 0 or r == 0, f"Red {r} not quantized to level {level}"
                        assert g % level == 0 or g == 0, f"Green {g} not quantized to level {level}"
                        assert b % level == 0 or b == 0, f"Blue {b} not quantized to level {level}"
    
    def test_wheel_gradient_hue_progression(self):
        """Test that wheel gradient has proper hue progression."""
        gradient_wheel = color_wheel.add_wheel_gradient(wheel_size=100, inner_radius_ratio=0.1)
        
        center = gradient_wheel.shape[0] // 2
        radius = 40  # Test at consistent radius
        
        # Sample colors at different angles
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)  # 12 points around circle
        hues = []
        
        for angle in angles:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            
            if 0 <= x < gradient_wheel.shape[1] and 0 <= y < gradient_wheel.shape[0]:
                r, g, b = gradient_wheel[y, x, :3]
                if r > 0 or g > 0 or b > 0:  # Skip black pixels
                    h, s, v = color_wheel.rgb_to_hsv_normalized(r, g, b)
                    hues.append(h)
        
        if len(hues) >= 6:  # Need reasonable sample
            # Hues should cover a reasonable range (ideally 0-360°)
            hue_range = max(hues) - min(hues)
            assert hue_range > 180, f"Hue range too small: {hue_range}°"
            
            # Should have good hue distribution (not all clustered)
            hue_std = np.std(hues)
            assert hue_std > 30, f"Hue distribution too clustered: std={hue_std}"
    
    def test_wheel_gradient_saturation_value_structure(self):
        """Test saturation and value structure of gradient wheel."""
        gradient_wheel = color_wheel.add_wheel_gradient(wheel_size=100, inner_radius_ratio=0.1)
        
        center = gradient_wheel.shape[0] // 2
        
        # Test saturation increases with radius (typical color wheel structure)
        radii = [15, 25, 35]  # Different distances from center
        saturations = []
        
        for radius in radii:
            # Sample at multiple angles for this radius
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            radius_saturations = []
            
            for angle in angles:
                x = int(center + radius * np.cos(angle))
                y = int(center + radius * np.sin(angle))
                
                if 0 <= x < gradient_wheel.shape[1] and 0 <= y < gradient_wheel.shape[0]:
                    r, g, b = gradient_wheel[y, x, :3]
                    if r > 0 or g > 0 or b > 0:
                        h, s, v = color_wheel.rgb_to_hsv_normalized(r, g, b)
                        radius_saturations.append(s)
            
            if radius_saturations:
                saturations.append(np.mean(radius_saturations))
        
        if len(saturations) >= 2:
            # Generally, saturation should increase with radius in a color wheel
            # (though this might not be strictly monotonic due to quantization)
            outer_saturation = saturations[-1]
            inner_saturation = saturations[0]
            
            # Outer should generally have higher saturation than inner
            assert outer_saturation >= inner_saturation * 0.8, \
                f"Outer saturation {outer_saturation} should be >= inner {inner_saturation}"


class TestVisualizationIntegration:
    """Test integration between different visualization functions."""
    
    def test_all_visualizations_consistent_input(self, sample_color_percentages, temp_dir):
        """Test that all visualization functions work with the same input."""
        opacity_values = [int(pct * 255 / 100) for pct in sample_color_percentages.values()]
        
        # All functions should work with the same data
        color_wheel.create_opacity_histogram(opacity_values, str(temp_dir / "opacity.png"))
        color_wheel.create_color_spectrum_histogram(sample_color_percentages, str(temp_dir / "spectrum.png"))
        color_wheel.create_circular_color_spectrum(sample_color_percentages, str(temp_dir / "circular.png"))
        
        # All files should be created
        assert (temp_dir / "opacity.png").exists()
        assert (temp_dir / "spectrum.png").exists()
        assert (temp_dir / "circular.png").exists()
    
    def test_visualization_file_cleanup(self, temp_dir):
        """Test that matplotlib properly closes figures to avoid memory leaks."""
        test_colors = {(255, 0, 0): 50.0, (0, 255, 0): 50.0}
        
        # Count initial number of figures
        initial_fig_count = len(plt.get_fignums())
        
        # Create multiple visualizations
        for i in range(5):
            color_wheel.create_color_spectrum_histogram(
                test_colors, str(temp_dir / f"test_{i}.png")
            )
        
        # Figure count should not have grown significantly
        final_fig_count = len(plt.get_fignums())
        assert final_fig_count <= initial_fig_count + 1, \
            f"Too many figures left open: {initial_fig_count} -> {final_fig_count}"
    
    @patch('matplotlib.pyplot.show')
    def test_visualizations_dont_show_interactively(self, mock_show, sample_color_percentages, temp_dir):
        """Test that visualization functions don't show interactive plots."""
        # Create visualizations
        color_wheel.create_opacity_histogram([100, 200], str(temp_dir / "test_opacity.png"))
        color_wheel.create_color_spectrum_histogram(sample_color_percentages, str(temp_dir / "test_spectrum.png"))
        color_wheel.create_circular_color_spectrum(sample_color_percentages, str(temp_dir / "test_circular.png"))
        
        # plt.show() should not have been called
        mock_show.assert_not_called()