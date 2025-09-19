"""
Test wheel template system including:
- template creation and caching
- mmap template saving and loading  
- template path generation
- template loading with fallbacks
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import color_wheel


class TestWheelTemplateCreation:
    """Test wheel template creation functionality."""
    
    def test_create_wheel_template_basic(self):
        """Test basic wheel template creation."""
        wheel_rgb, color_to_pixels_map, wheel_hsv_cache = color_wheel.create_wheel_template(
            wheel_size=100, inner_radius_ratio=0.1, quantize_level=8
        )
        
        # Check wheel RGB array properties
        assert wheel_rgb.shape == (100, 100, 3)
        assert wheel_rgb.dtype == np.uint8
        assert np.all(wheel_rgb >= 0)
        assert np.all(wheel_rgb <= 255)
        
        # Check color to pixels mapping
        assert isinstance(color_to_pixels_map, dict)
        assert len(color_to_pixels_map) > 0
        
        # Each color should be a tuple of 3 integers
        for color, pixels in color_to_pixels_map.items():
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, (int, np.integer)) for c in color)
            assert isinstance(pixels, (list, np.ndarray))
        
        # Check HSV cache
        assert isinstance(wheel_hsv_cache, dict)
        assert len(wheel_hsv_cache) > 0
        
        # HSV cache should have same colors as color mapping
        assert set(wheel_hsv_cache.keys()) == set(color_to_pixels_map.keys())
    
    def test_different_wheel_sizes(self):
        """Test wheel template creation with different sizes."""
        sizes = [50, 100, 200]
        
        for size in sizes:
            wheel_rgb, color_map, hsv_cache = color_wheel.create_wheel_template(
                wheel_size=size, inner_radius_ratio=0.1, quantize_level=8
            )
            
            assert wheel_rgb.shape == (size, size, 3)
            
            # Larger wheels should generally have more colors
            if size > 50:
                assert len(color_map) > 10  # Should have reasonable number of colors
    
    def test_different_inner_radius_ratios(self):
        """Test wheel template creation with different inner radius ratios."""
        ratios = [0.0, 0.1, 0.3, 0.5]
        
        for ratio in ratios:
            wheel_rgb, color_map, hsv_cache = color_wheel.create_wheel_template(
                wheel_size=100, inner_radius_ratio=ratio, quantize_level=8
            )
            
            # Check that inner area has transparent/different treatment
            center = wheel_rgb.shape[0] // 2
            inner_radius = int(ratio * center)
            
            if inner_radius > 5:  # Only test if inner radius is significant
                # Colors near center should be different from edge colors
                center_colors = wheel_rgb[center-2:center+2, center-2:center+2]
                edge_colors = wheel_rgb[0:5, 0:5]  # Corner colors
                
                # They should be different (not identical)
                assert not np.array_equal(center_colors, edge_colors[:center_colors.shape[0], :center_colors.shape[1]])
    
    def test_different_quantize_levels(self):
        """Test wheel template creation with different quantization levels."""
        levels = [1, 4, 8, 16]
        color_counts = []
        
        for level in levels:
            wheel_rgb, color_map, hsv_cache = color_wheel.create_wheel_template(
                wheel_size=100, inner_radius_ratio=0.1, quantize_level=level
            )
            
            color_counts.append(len(color_map))
            
            # All colors should be quantized properly
            for color in color_map.keys():
                r, g, b = color
                if level > 1:
                    assert r % level == 0, f"Red {r} not quantized to level {level}"
                    assert g % level == 0, f"Green {g} not quantized to level {level}" 
                    assert b % level == 0, f"Blue {b} not quantized to level {level}"
        
        # Higher quantization should generally result in fewer unique colors
        # (though this isn't strictly guaranteed due to the wheel structure)
        assert len(set(color_counts)) > 1  # Should have different counts
    
    def test_wheel_template_mathematical_properties(self):
        """Test mathematical properties of the wheel template."""
        wheel_rgb, color_map, hsv_cache = color_wheel.create_wheel_template(
            wheel_size=100, inner_radius_ratio=0.1, quantize_level=8
        )
        
        size = wheel_rgb.shape[0]
        center = size // 2
        
        # Test circular symmetry properties
        # Colors at same radius should have similar saturation/value
        radius = 30  # Test radius
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        saturations = []
        for angle in angles:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                r, g, b = wheel_rgb[y, x]
                # Convert to HSV to check saturation
                h, s, v = color_wheel.rgb_to_hsv_normalized(r, g, b)
                saturations.append(s)
        
        if len(saturations) > 3:
            # Saturations at same radius should be similar
            sat_std = np.std(saturations)
            assert sat_std < 0.2, f"Saturation varies too much at same radius: {sat_std}"


class TestTemplatePathGeneration:
    """Test template file path generation."""
    
    def test_get_wheel_template_path(self):
        """Test wheel template path generation."""
        path = color_wheel.get_wheel_template_path(800, 0.1, 8)
        
        assert isinstance(path, (str, Path))
        path_str = str(path)
        
        # Should include wheel_templates directory
        assert "wheel_templates" in path_str
        
        # Should include parameters in filename
        assert "800" in path_str
        assert "0.1" in path_str or "01" in path_str  # May format decimals differently
        assert "8" in path_str
        assert path_str.endswith(".pkl")
    
    def test_get_mmap_template_paths(self):
        """Test mmap template paths generation."""
        rgb_path, map_path, hsv_path = color_wheel.get_mmap_template_paths(800, 0.1, 8)
        
        for path in [rgb_path, map_path, hsv_path]:
            assert isinstance(path, (str, Path))
            path_str = str(path)
            assert "wheel_templates" in path_str
            assert "800" in path_str
        
        # Each should have different suffix
        assert str(rgb_path).endswith("_rgb.dat")
        assert str(map_path).endswith("_map.pkl")  
        assert str(hsv_path).endswith("_hsv.pkl")
    
    def test_path_uniqueness(self):
        """Test that different parameters generate different paths."""
        path1 = color_wheel.get_wheel_template_path(800, 0.1, 8)
        path2 = color_wheel.get_wheel_template_path(400, 0.1, 8)  # Different size
        path3 = color_wheel.get_wheel_template_path(800, 0.2, 8)  # Different ratio
        path4 = color_wheel.get_wheel_template_path(800, 0.1, 4)  # Different quantize
        
        paths = [str(path1), str(path2), str(path3), str(path4)]
        assert len(set(paths)) == 4, "All paths should be unique"


class TestMmapTemplateSaving:
    """Test mmap template saving functionality."""
    
    def test_save_mmap_template(self, temp_dir):
        """Test saving mmap template files."""
        # Create test data
        wheel_rgb = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        color_to_pixels_map = {
            (255, 0, 0): [(10, 10), (11, 11)],
            (0, 255, 0): [(20, 20), (21, 21)],
        }
        wheel_hsv_cache = {
            (255, 0, 0): (0.0, 1.0, 1.0),
            (0, 255, 0): (120.0, 1.0, 1.0),
        }
        
        # Mock the path generation to use temp directory
        with patch('color_wheel.get_mmap_template_paths') as mock_paths:
            rgb_path = temp_dir / "test_rgb.dat"
            map_path = temp_dir / "test_map.pkl" 
            hsv_path = temp_dir / "test_hsv.pkl"
            mock_paths.return_value = (str(rgb_path), str(map_path), str(hsv_path))
            
            # Save the template
            color_wheel.save_mmap_template(
                wheel_rgb, color_to_pixels_map, wheel_hsv_cache, 50, 0.1, 8
            )
            
            # Check files were created
            assert rgb_path.exists()
            assert map_path.exists()
            assert hsv_path.exists()
            
            # Check RGB file size is correct
            expected_size = wheel_rgb.nbytes
            assert rgb_path.stat().st_size == expected_size
    
    def test_save_mmap_handles_directory_creation(self, temp_dir):
        """Test that saving creates directories if they don't exist."""
        non_existent_dir = temp_dir / "subdir" / "wheel_templates"
        
        with patch('color_wheel.get_mmap_template_paths') as mock_paths:
            rgb_path = non_existent_dir / "test_rgb.dat"
            map_path = non_existent_dir / "test_map.pkl"
            hsv_path = non_existent_dir / "test_hsv.pkl"
            mock_paths.return_value = (str(rgb_path), str(map_path), str(hsv_path))
            
            wheel_rgb = np.ones((10, 10, 3), dtype=np.uint8)
            color_map = {(255, 0, 0): [(5, 5)]}
            hsv_cache = {(255, 0, 0): (0.0, 1.0, 1.0)}
            
            # Should create directory and save files
            color_wheel.save_mmap_template(wheel_rgb, color_map, hsv_cache, 10, 0.1, 8)
            
            assert non_existent_dir.exists()
            assert rgb_path.exists()


class TestMmapTemplateLoading:
    """Test mmap template loading functionality."""
    
    def test_load_mmap_template_success(self, temp_dir):
        """Test successful mmap template loading."""
        # Create test files
        wheel_rgb = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        color_to_pixels_map = {
            (255, 0, 0): [(10, 10), (11, 11)],
            (0, 255, 0): [(20, 20)],
        }
        wheel_hsv_cache = {
            (255, 0, 0): (0.0, 1.0, 1.0),
            (0, 255, 0): (120.0, 1.0, 1.0),
        }
        
        # Save files manually
        rgb_path = temp_dir / "test_rgb.dat"
        map_path = temp_dir / "test_map.pkl"
        hsv_path = temp_dir / "test_hsv.pkl"
        
        wheel_rgb.tofile(str(rgb_path))
        
        import pickle
        with open(map_path, 'wb') as f:
            pickle.dump(color_to_pixels_map, f)
        with open(hsv_path, 'wb') as f:
            pickle.dump(wheel_hsv_cache, f)
        
        # Mock path generation
        with patch('color_wheel.get_mmap_template_paths') as mock_paths:
            mock_paths.return_value = (str(rgb_path), str(map_path), str(hsv_path))
            
            # Load template
            loaded_rgb, loaded_map, loaded_hsv = color_wheel.load_mmap_template(30, 0.1, 8)
            
            # Check loaded data matches
            np.testing.assert_array_equal(loaded_rgb, wheel_rgb)
            assert loaded_map == color_to_pixels_map
            assert loaded_hsv == wheel_hsv_cache
    
    def test_load_mmap_template_missing_files(self):
        """Test loading when mmap files don't exist."""
        with patch('color_wheel.get_mmap_template_paths') as mock_paths:
            mock_paths.return_value = ("/nonexistent/rgb.dat", "/nonexistent/map.pkl", "/nonexistent/hsv.pkl")
            
            result = color_wheel.load_mmap_template(100, 0.1, 8)
            assert result is None
    
    def test_load_mmap_template_corrupted_files(self, temp_dir):
        """Test loading when files exist but are corrupted."""
        rgb_path = temp_dir / "corrupt_rgb.dat"
        map_path = temp_dir / "corrupt_map.pkl"
        hsv_path = temp_dir / "corrupt_hsv.pkl"
        
        # Create corrupted files
        rgb_path.write_bytes(b"corrupted data")
        map_path.write_bytes(b"not pickle data")
        hsv_path.write_bytes(b"also not pickle")
        
        with patch('color_wheel.get_mmap_template_paths') as mock_paths:
            mock_paths.return_value = (str(rgb_path), str(map_path), str(hsv_path))
            
            result = color_wheel.load_mmap_template(100, 0.1, 8)
            assert result is None


class TestLoadOrCreateWheelTemplate:
    """Test the load or create wheel template function."""
    
    def test_load_existing_pickle_template(self, temp_dir):
        """Test loading existing pickle template."""
        # Create template data
        wheel_rgb = np.ones((50, 50, 3), dtype=np.uint8) * 128
        color_map = {(128, 128, 128): [(25, 25)]}
        hsv_cache = {(128, 128, 128): (0.0, 0.0, 0.5)}
        template_data = (wheel_rgb, color_map, hsv_cache)
        
        # Create pickle file
        template_path = temp_dir / "template.pkl"
        import pickle
        with open(template_path, 'wb') as f:
            pickle.dump(template_data, f)
        
        with patch('color_wheel.get_wheel_template_path') as mock_path:
            mock_path.return_value = str(template_path)
            
            loaded_rgb, loaded_map, loaded_hsv = color_wheel.load_or_create_wheel_template(50, 0.1, 8)
            
            np.testing.assert_array_equal(loaded_rgb, wheel_rgb)
            assert loaded_map == color_map
            assert loaded_hsv == hsv_cache
    
    def test_load_existing_mmap_template(self, temp_dir):
        """Test loading existing mmap template."""
        # Create mmap template files
        wheel_rgb = np.ones((40, 40, 3), dtype=np.uint8) * 100
        color_map = {(100, 100, 100): [(20, 20)]}
        hsv_cache = {(100, 100, 100): (0.0, 0.0, 0.39)}
        
        rgb_path = temp_dir / "mmap_rgb.dat"
        map_path = temp_dir / "mmap_map.pkl"
        hsv_path = temp_dir / "mmap_hsv.pkl"
        
        wheel_rgb.tofile(str(rgb_path))
        
        import pickle
        with open(map_path, 'wb') as f:
            pickle.dump(color_map, f)
        with open(hsv_path, 'wb') as f:
            pickle.dump(hsv_cache, f)
        
        with patch('color_wheel.get_wheel_template_path') as mock_pickle_path:
            with patch('color_wheel.get_mmap_template_paths') as mock_mmap_paths:
                mock_pickle_path.return_value = "/nonexistent.pkl"  # Pickle doesn't exist
                mock_mmap_paths.return_value = (str(rgb_path), str(map_path), str(hsv_path))
                
                loaded_rgb, loaded_map, loaded_hsv = color_wheel.load_or_create_wheel_template(40, 0.1, 8)
                
                np.testing.assert_array_equal(loaded_rgb, wheel_rgb)
                assert loaded_map == color_map
                assert loaded_hsv == hsv_cache
    
    def test_create_new_template_when_none_exist(self):
        """Test creating new template when no cached versions exist."""
        with patch('color_wheel.get_wheel_template_path') as mock_pickle_path:
            with patch('color_wheel.load_mmap_template') as mock_load_mmap:
                with patch('color_wheel.create_wheel_template') as mock_create:
                    with patch('color_wheel.save_mmap_template') as mock_save:
                        
                        # No existing files
                        mock_pickle_path.return_value = "/nonexistent.pkl"
                        mock_load_mmap.return_value = None
                        
                        # Mock creation
                        test_rgb = np.ones((60, 60, 3), dtype=np.uint8) * 75
                        test_map = {(75, 75, 75): [(30, 30)]}
                        test_hsv = {(75, 75, 75): (0.0, 0.0, 0.29)}
                        mock_create.return_value = (test_rgb, test_map, test_hsv)
                        
                        # Load or create
                        result_rgb, result_map, result_hsv = color_wheel.load_or_create_wheel_template(60, 0.1, 8)
                        
                        # Should have called create and save
                        mock_create.assert_called_once_with(60, 0.1, 8)
                        mock_save.assert_called_once_with(test_rgb, test_map, test_hsv, 60, 0.1, 8)
                        
                        # Should return created data
                        np.testing.assert_array_equal(result_rgb, test_rgb)
                        assert result_map == test_map
                        assert result_hsv == test_hsv
    
    def test_caching_behavior(self, temp_dir):
        """Test that templates are properly cached after creation."""
        with patch('color_wheel.create_wheel_template') as mock_create:
            # Mock template creation
            test_rgb = np.ones((70, 70, 3), dtype=np.uint8) * 50
            test_map = {(50, 50, 50): [(35, 35)]}
            test_hsv = {(50, 50, 50): (0.0, 0.0, 0.2)}
            mock_create.return_value = (test_rgb, test_map, test_hsv)
            
            # First call should create template
            with patch('color_wheel.get_wheel_template_path') as mock_path:
                template_path = temp_dir / "cache_test.pkl"
                mock_path.return_value = str(template_path)
                
                result1 = color_wheel.load_or_create_wheel_template(70, 0.1, 8)
                
                # Should have called create once
                assert mock_create.call_count == 1
                
                # Template file should now exist
                assert template_path.exists()
                
                # Second call should load from cache
                result2 = color_wheel.load_or_create_wheel_template(70, 0.1, 8)
                
                # Should not have called create again
                assert mock_create.call_count == 1
                
                # Results should be identical
                np.testing.assert_array_equal(result1[0], result2[0])
                assert result1[1] == result2[1]
                assert result1[2] == result2[2]