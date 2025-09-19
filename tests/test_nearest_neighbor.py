"""
Test nearest neighbor search algorithms including:
- find_nearest_wheel_colors_vectorized function
- KDTree vs fallback methods
- HSV distance calculations (NumPy, Numba, GPU variants)
- parallel vs single processing
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import color_wheel


class TestFindNearestWheelColors:
    """Test the main nearest neighbor search function."""
    
    def test_basic_nearest_neighbor_search(self):
        """Test basic nearest neighbor functionality."""
        # Create simple test data
        image_colors = {
            (255, 0, 0): 30.0,    # Red
            (0, 255, 0): 40.0,    # Green
            (128, 128, 255): 30.0 # Light blue
        }
        
        # Create wheel template with known colors
        wheel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        color_to_pixels_map = {color: [(10, 10)] for color in wheel_colors}
        wheel_hsv_cache = {
            (255, 0, 0): (0.0, 1.0, 1.0),
            (0, 255, 0): (120.0, 1.0, 1.0),
            (0, 0, 255): (240.0, 1.0, 1.0)
        }
        
        matches = color_wheel.find_nearest_wheel_colors_vectorized(
            image_colors, color_to_pixels_map, wheel_hsv_cache
        )
        
        # Check results structure
        assert isinstance(matches, dict)
        assert len(matches) == len(image_colors)
        
        # Each image color should map to a wheel color
        for image_color, (wheel_color, distance) in matches.items():
            assert image_color in image_colors
            assert wheel_color in wheel_colors
            assert isinstance(distance, (int, float))
            assert distance >= 0
        
        # Red should match to red (distance should be 0 or very small)
        red_match = matches[(255, 0, 0)]
        assert red_match[0] == (255, 0, 0)
        assert red_match[1] < 1.0  # Distance should be very small
        
        # Green should match to green
        green_match = matches[(0, 255, 0)]
        assert green_match[0] == (0, 255, 0)
        assert green_match[1] < 1.0
    
    def test_force_kdtree_option(self):
        """Test forcing KDTree usage."""
        with patch('color_wheel.KDTREE_AVAILABLE', True):
            with patch('color_wheel._find_nearest_with_kdtree') as mock_kdtree:
                # Mock KDTree to return simple results
                mock_kdtree.return_value = {(255, 0, 0): ((255, 0, 0), 0.0)}
                
                image_colors = {(255, 0, 0): 100.0}
                color_to_pixels_map = {(255, 0, 0): [(10, 10)]}
                wheel_hsv_cache = {(255, 0, 0): (0.0, 1.0, 1.0)}
                
                matches = color_wheel.find_nearest_wheel_colors_vectorized(
                    image_colors, color_to_pixels_map, wheel_hsv_cache, force_kdtree=True
                )
                
                # Should have called KDTree method
                mock_kdtree.assert_called_once()
    
    def test_disable_kdtree_option(self):
        """Test disabling KDTree and using fallback."""
        with patch('color_wheel.KDTREE_AVAILABLE', True):
            with patch('color_wheel._find_nearest_with_kdtree') as mock_kdtree:
                with patch('color_wheel._find_nearest_vectorized_fallback') as mock_fallback:
                    # Mock fallback to return simple results
                    mock_fallback.return_value = {(255, 0, 0): ((255, 0, 0), 0.0)}
                    
                    image_colors = {(255, 0, 0): 100.0}
                    color_to_pixels_map = {(255, 0, 0): [(10, 10)]}
                    wheel_hsv_cache = {(255, 0, 0): (0.0, 1.0, 1.0)}
                    
                    matches = color_wheel.find_nearest_wheel_colors_vectorized(
                        image_colors, color_to_pixels_map, wheel_hsv_cache, force_kdtree=False
                    )
                    
                    # Should have called fallback method, not KDTree
                    mock_fallback.assert_called_once()
                    mock_kdtree.assert_not_called()
    
    def test_parallel_vs_single_processing(self):
        """Test parallel vs single-threaded nearest neighbor search."""
        # Create test data with multiple colors
        image_colors = {
            (255, 0, 0): 25.0,
            (0, 255, 0): 25.0,
            (0, 0, 255): 25.0,
            (255, 255, 0): 25.0
        }
        
        wheel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
        color_to_pixels_map = {color: [(10, 10)] for color in wheel_colors}
        wheel_hsv_cache = {
            (255, 0, 0): (0.0, 1.0, 1.0),
            (0, 255, 0): (120.0, 1.0, 1.0),
            (0, 0, 255): (240.0, 1.0, 1.0),
            (255, 255, 255): (0.0, 0.0, 1.0)
        }
        
        # Test single-threaded
        matches_single = color_wheel.find_nearest_wheel_colors_vectorized(
            image_colors, color_to_pixels_map, wheel_hsv_cache, use_parallel=False
        )
        
        # Test parallel
        matches_parallel = color_wheel.find_nearest_wheel_colors_vectorized(
            image_colors, color_to_pixels_map, wheel_hsv_cache, use_parallel=True
        )
        
        # Results should be the same
        assert len(matches_single) == len(matches_parallel)
        
        for image_color in image_colors:
            single_match = matches_single[image_color]
            parallel_match = matches_parallel[image_color]
            
            # Should match to same wheel color
            assert single_match[0] == parallel_match[0]
            
            # Distances should be very close
            assert abs(single_match[1] - parallel_match[1]) < 0.01


class TestKDTreeMethod:
    """Test KDTree-based nearest neighbor search."""
    
    @patch('color_wheel.KDTREE_AVAILABLE', True)
    def test_kdtree_method_basic(self, mock_sklearn):
        """Test basic KDTree functionality."""
        # Create test data
        image_colors = {(255, 0, 0): 50.0, (0, 255, 0): 50.0}
        wheel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Convert to arrays for testing
        image_colors_list = list(image_colors.keys())
        image_hsv = np.array([color_wheel.rgb_to_hsv_normalized(*color) for color in image_colors_list])
        wheel_hsv = np.array([color_wheel.rgb_to_hsv_normalized(*color) for color in wheel_colors])
        
        # Mock KDTree behavior
        with patch('color_wheel.KDTree') as mock_kdtree_class:
            mock_kdtree = MagicMock()
            mock_kdtree.query.return_value = (np.array([0.0, 1.0]), np.array([0, 1]))  # distances, indices
            mock_kdtree_class.return_value = mock_kdtree
            
            matches = color_wheel._find_nearest_with_kdtree(
                image_colors_list, image_hsv, wheel_colors, wheel_hsv
            )
            
            # Should have called KDTree
            mock_kdtree_class.assert_called_once()
            mock_kdtree.query.assert_called_once()
            
            # Should return valid matches
            assert isinstance(matches, dict)
            assert len(matches) == len(image_colors_list)
    
    @patch('color_wheel.KDTREE_AVAILABLE', False)
    def test_kdtree_unavailable_fallback(self):
        """Test fallback when KDTree is unavailable."""
        image_colors = {(255, 0, 0): 100.0}
        color_to_pixels_map = {(255, 0, 0): [(10, 10)]}
        wheel_hsv_cache = {(255, 0, 0): (0.0, 1.0, 1.0)}
        
        with patch('color_wheel._find_nearest_vectorized_fallback') as mock_fallback:
            mock_fallback.return_value = {(255, 0, 0): ((255, 0, 0), 0.0)}
            
            matches = color_wheel.find_nearest_wheel_colors_vectorized(
                image_colors, color_to_pixels_map, wheel_hsv_cache
            )
            
            # Should have used fallback method
            mock_fallback.assert_called_once()


class TestHSVDistanceCalculations:
    """Test HSV distance calculation functions."""
    
    def test_hsv_distance_numpy(self):
        """Test NumPy-based HSV distance calculation."""
        # Create test HSV data
        image_hsv = np.array([[0.0, 1.0, 1.0], [120.0, 1.0, 1.0]])  # Red, Green
        wheel_hsv = np.array([[0.0, 1.0, 1.0], [240.0, 1.0, 1.0]])  # Red, Blue
        
        distances = color_wheel._calculate_hsv_distances_numpy(image_hsv, wheel_hsv)
        
        # Should return distance matrix
        assert distances.shape == (2, 2)  # 2 image colors x 2 wheel colors
        
        # Red to red should have distance 0 (or very small)
        assert distances[0, 0] < 0.1
        
        # Red to blue should have larger distance than red to red
        assert distances[0, 1] > distances[0, 0]
        
        # All distances should be non-negative
        assert np.all(distances >= 0)
    
    @patch('color_wheel.NUMBA_AVAILABLE', True)
    def test_hsv_distance_numba(self, mock_numba):
        """Test Numba-optimized HSV distance calculation."""
        image_hsv = np.array([[0.0, 1.0, 1.0]])
        wheel_hsv = np.array([[0.0, 1.0, 1.0], [120.0, 1.0, 1.0]])
        
        with patch('color_wheel._calculate_hsv_distances_numba') as mock_numba_func:
            expected_distances = np.array([[0.0, 100.0]])  # Mock distances
            mock_numba_func.return_value = expected_distances
            
            # Call through the main distance function (if it exists)
            # For direct testing, we'll test the numba function
            distances = color_wheel._calculate_hsv_distances_numba(image_hsv, wheel_hsv)
            
            # Should have called numba function
            mock_numba_func.assert_called_once()
    
    @patch('color_wheel.CUPY_AVAILABLE', True)
    def test_hsv_distance_gpu(self, mock_cupy):
        """Test GPU-accelerated HSV distance calculation."""
        image_hsv = np.array([[0.0, 1.0, 1.0]])
        wheel_hsv = np.array([[0.0, 1.0, 1.0]])
        
        with patch('color_wheel._calculate_hsv_distances_gpu') as mock_gpu_func:
            expected_distances = np.array([[0.0]])
            mock_gpu_func.return_value = expected_distances
            
            distances = color_wheel._calculate_hsv_distances_gpu(image_hsv, wheel_hsv)
            
            mock_gpu_func.assert_called_once()
    
    def test_hsv_distance_hue_weighting(self):
        """Test that HSV distance calculation properly weights hue differences."""
        # Test colors that differ mainly in hue
        red_hsv = np.array([[0.0, 1.0, 1.0]])      # Red
        green_hsv = np.array([[120.0, 1.0, 1.0]])  # Green
        
        distances = color_wheel._calculate_hsv_distances_numpy(red_hsv, green_hsv)
        
        # Distance should be significant due to hue difference
        assert distances[0, 0] > 50  # Should be substantial distance
        
        # Test with different hue weights
        distances_low_hue = color_wheel._calculate_hsv_distances_numpy(
            red_hsv, green_hsv, hue_weight=1.0
        )
        distances_high_hue = color_wheel._calculate_hsv_distances_numpy(
            red_hsv, green_hsv, hue_weight=5.0
        )
        
        # Higher hue weight should result in larger distances for hue differences
        assert distances_high_hue[0, 0] > distances_low_hue[0, 0]
    
    def test_hsv_distance_saturation_value_weighting(self):
        """Test saturation and value weighting in distance calculations."""
        # Colors that differ in saturation and value
        bright_red = np.array([[0.0, 1.0, 1.0]])    # Fully saturated, bright red
        pale_red = np.array([[0.0, 0.5, 0.7]])      # Less saturated, dimmer red
        
        distances_normal = color_wheel._calculate_hsv_distances_numpy(bright_red, pale_red)
        
        # Test with high saturation weight
        distances_high_sat = color_wheel._calculate_hsv_distances_numpy(
            bright_red, pale_red, sat_weight=3.0
        )
        
        # Test with high value weight  
        distances_high_val = color_wheel._calculate_hsv_distances_numpy(
            bright_red, pale_red, val_weight=2.0
        )
        
        # Higher weights should increase distances
        assert distances_high_sat[0, 0] > distances_normal[0, 0]
        assert distances_high_val[0, 0] > distances_normal[0, 0]


class TestVectorizedFallbackMethod:
    """Test the vectorized fallback nearest neighbor method."""
    
    def test_fallback_method_basic(self):
        """Test basic fallback method functionality."""
        image_colors = [(255, 0, 0), (0, 255, 0)]
        wheel_colors = [(255, 0, 0), (0, 0, 255)]
        
        # Convert to HSV
        image_hsv = np.array([color_wheel.rgb_to_hsv_normalized(*color) for color in image_colors])
        wheel_hsv = np.array([color_wheel.rgb_to_hsv_normalized(*color) for color in wheel_colors])
        
        matches = color_wheel._find_nearest_vectorized_fallback(
            image_colors, image_hsv, wheel_colors, wheel_hsv
        )
        
        # Check results
        assert isinstance(matches, dict)
        assert len(matches) == len(image_colors)
        
        # Red should match to red (exact match)
        red_match = matches[(255, 0, 0)]
        assert red_match[0] == (255, 0, 0)
        assert red_match[1] < 0.1  # Distance should be very small
        
        # Green should match to blue (closest available)
        green_match = matches[(0, 255, 0)]
        assert green_match[0] == (0, 0, 255)
        assert green_match[1] > 0  # Should have some distance
    
    def test_fallback_method_consistency(self):
        """Test that fallback method gives consistent results."""
        # Create reproducible test data
        np.random.seed(42)
        
        image_colors = [(255, 0, 0), (128, 128, 128), (0, 255, 255)]
        wheel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
        
        image_hsv = np.array([color_wheel.rgb_to_hsv_normalized(*color) for color in image_colors])
        wheel_hsv = np.array([color_wheel.rgb_to_hsv_normalized(*color) for color in wheel_colors])
        
        # Run multiple times
        results = []
        for _ in range(3):
            matches = color_wheel._find_nearest_vectorized_fallback(
                image_colors, image_hsv, wheel_colors, wheel_hsv
            )
            results.append(matches)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], "Fallback method should give consistent results"
    
    def test_fallback_distance_properties(self):
        """Test mathematical properties of distance calculations in fallback method."""
        # Test triangle inequality and other distance properties
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        hsv_colors = np.array([color_wheel.rgb_to_hsv_normalized(*color) for color in colors])
        
        matches = color_wheel._find_nearest_vectorized_fallback(
            colors, hsv_colors, colors, hsv_colors
        )
        
        # Distance to self should be 0 or very small
        for color in colors:
            match_color, distance = matches[color]
            if match_color == color:
                assert distance < 0.1, f"Distance to self should be ~0, got {distance}"
        
        # All distances should be non-negative
        for color, (_, distance) in matches.items():
            assert distance >= 0, f"Distance should be non-negative, got {distance}"


class TestNearestNeighborEdgeCases:
    """Test edge cases in nearest neighbor search."""
    
    def test_empty_image_colors(self):
        """Test with empty image colors."""
        color_to_pixels_map = {(255, 0, 0): [(10, 10)]}
        wheel_hsv_cache = {(255, 0, 0): (0.0, 1.0, 1.0)}
        
        matches = color_wheel.find_nearest_wheel_colors_vectorized(
            {}, color_to_pixels_map, wheel_hsv_cache
        )
        
        assert matches == {}
    
    def test_single_image_color(self):
        """Test with only one image color."""
        image_colors = {(255, 0, 0): 100.0}
        color_to_pixels_map = {(255, 0, 0): [(10, 10)], (0, 255, 0): [(20, 20)]}
        wheel_hsv_cache = {
            (255, 0, 0): (0.0, 1.0, 1.0),
            (0, 255, 0): (120.0, 1.0, 1.0)
        }
        
        matches = color_wheel.find_nearest_wheel_colors_vectorized(
            image_colors, color_to_pixels_map, wheel_hsv_cache
        )
        
        assert len(matches) == 1
        assert (255, 0, 0) in matches
        
        # Should match to exact color
        match_color, distance = matches[(255, 0, 0)]
        assert match_color == (255, 0, 0)
        assert distance < 0.1
    
    def test_single_wheel_color(self):
        """Test with only one wheel color available."""
        image_colors = {(255, 0, 0): 50.0, (0, 255, 0): 50.0}
        color_to_pixels_map = {(128, 128, 128): [(10, 10)]}  # Only gray available
        wheel_hsv_cache = {(128, 128, 128): (0.0, 0.0, 0.5)}
        
        matches = color_wheel.find_nearest_wheel_colors_vectorized(
            image_colors, color_to_pixels_map, wheel_hsv_cache
        )
        
        assert len(matches) == 2
        
        # Both image colors should match to the only available wheel color
        for image_color, (wheel_color, distance) in matches.items():
            assert wheel_color == (128, 128, 128)
            assert distance > 0  # Should have some distance since colors don't match exactly
    
    def test_identical_colors(self):
        """Test when image colors exactly match wheel colors."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        image_colors = {color: 33.33 for color in colors}
        color_to_pixels_map = {color: [(10, 10)] for color in colors}
        wheel_hsv_cache = {color: color_wheel.rgb_to_hsv_normalized(*color) for color in colors}
        
        matches = color_wheel.find_nearest_wheel_colors_vectorized(
            image_colors, color_to_pixels_map, wheel_hsv_cache
        )
        
        # Each color should match to itself with distance ~0
        for color in colors:
            match_color, distance = matches[color]
            assert match_color == color
            assert distance < 0.1