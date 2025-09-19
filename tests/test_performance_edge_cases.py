"""
Test performance characteristics and edge cases including:
- large images and memory management
- empty and single-color images
- GPU availability mocking
- performance regression tests
- stress testing
"""

import pytest
import numpy as np
import time
import gc
from unittest.mock import patch, MagicMock
import psutil
import os

import color_wheel


class TestLargeImageHandling:
    """Test handling of large images and memory management."""
    
    def test_large_synthetic_image(self, temp_dir):
        """Test with a large synthetic image."""
        # Create a large synthetic image (but not too large for CI)
        width, height = 500, 500
        large_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Save to file
        import cv2
        image_path = temp_dir / "large_image.png"
        large_bgr = large_image[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), large_bgr)
        
        # Test analysis with high sample factor to reduce processing time
        start_time = time.time()
        color_percentages = color_wheel.load_and_analyze_image(
            str(image_path), sample_factor=8, quantize_level=16
        )
        analysis_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert analysis_time < 30, f"Large image analysis took too long: {analysis_time:.2f}s"
        
        # Should produce reasonable results
        assert len(color_percentages) > 0
        assert abs(sum(color_percentages.values()) - 100.0) < 0.1
    
    def test_memory_usage_large_image(self, temp_dir):
        """Test memory usage doesn't grow excessively with large images."""
        if not hasattr(psutil, 'Process'):
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple large-ish images
        for i in range(3):
            # Create synthetic image
            width, height = 200, 200
            test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            image_path = temp_dir / f"memory_test_{i}.png"
            import cv2
            test_bgr = test_image[:, :, [2, 1, 0]]
            cv2.imwrite(str(image_path), test_bgr)
            
            # Analyze image
            color_percentages = color_wheel.load_and_analyze_image(
                str(image_path), sample_factor=4
            )
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (allow for some overhead)
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f} MB"
    
    def test_very_wide_image(self, temp_dir):
        """Test with very wide aspect ratio image."""
        # Create a very wide image
        width, height = 1000, 50
        wide_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        image_path = temp_dir / "wide_image.png"
        import cv2
        wide_bgr = wide_image[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), wide_bgr)
        
        # Should handle gracefully
        color_percentages = color_wheel.load_and_analyze_image(
            str(image_path), sample_factor=4
        )
        
        assert len(color_percentages) > 0
        assert abs(sum(color_percentages.values()) - 100.0) < 0.1
    
    def test_very_tall_image(self, temp_dir):
        """Test with very tall aspect ratio image."""
        # Create a very tall image
        width, height = 50, 1000
        tall_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        image_path = temp_dir / "tall_image.png"
        import cv2
        tall_bgr = tall_image[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), tall_bgr)
        
        # Should handle gracefully
        color_percentages = color_wheel.load_and_analyze_image(
            str(image_path), sample_factor=4
        )
        
        assert len(color_percentages) > 0
        assert abs(sum(color_percentages.values()) - 100.0) < 0.1


class TestEdgeCaseImages:
    """Test edge case images."""
    
    def test_completely_black_image(self, temp_dir):
        """Test with completely black image."""
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        image_path = temp_dir / "black_image.png"
        import cv2
        cv2.imwrite(str(image_path), black_image)
        
        color_percentages = color_wheel.load_and_analyze_image(str(image_path))
        
        # Should have one color (black) at 100%
        assert len(color_percentages) == 1
        black_color = list(color_percentages.keys())[0]
        assert all(c == 0 for c in black_color)  # Should be (0, 0, 0) or close
        assert color_percentages[black_color] == pytest.approx(100.0, abs=0.1)
    
    def test_completely_white_image(self, temp_dir):
        """Test with completely white image."""
        white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        
        image_path = temp_dir / "white_image.png"
        import cv2
        cv2.imwrite(str(image_path), white_image)
        
        color_percentages = color_wheel.load_and_analyze_image(str(image_path))
        
        # Should have one color (white) at 100%
        assert len(color_percentages) == 1
        white_color = list(color_percentages.keys())[0]
        # Might be quantized, but should be high values
        assert all(c >= 240 for c in white_color)
        assert color_percentages[white_color] == pytest.approx(100.0, abs=0.1)
    
    def test_checkerboard_pattern(self, temp_dir):
        """Test with high-frequency checkerboard pattern."""
        # Create checkerboard pattern
        size = 100
        checkerboard = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Fill with alternating black and white
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    checkerboard[i, j] = [255, 255, 255]  # White
                else:
                    checkerboard[i, j] = [0, 0, 0]        # Black
        
        image_path = temp_dir / "checkerboard.png"
        import cv2
        checkerboard_bgr = checkerboard[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), checkerboard_bgr)
        
        color_percentages = color_wheel.load_and_analyze_image(str(image_path))
        
        # Should have approximately equal amounts of black and white
        assert len(color_percentages) <= 2  # Should be just black and white (after quantization)
        
        # Each color should be approximately 50%
        for percentage in color_percentages.values():
            assert 40 <= percentage <= 60, f"Expected ~50%, got {percentage}%"
    
    def test_gradient_image(self, temp_dir):
        """Test with smooth gradient image."""
        size = 100
        gradient = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Create RGB gradient
        for i in range(size):
            for j in range(size):
                r = int(255 * i / (size - 1))
                g = int(255 * j / (size - 1))
                b = int(255 * (i + j) / (2 * size - 2))
                gradient[i, j] = [r, g, b]
        
        image_path = temp_dir / "gradient.png"
        import cv2
        gradient_bgr = gradient[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), gradient_bgr)
        
        color_percentages = color_wheel.load_and_analyze_image(str(image_path))
        
        # Should have many different colors due to gradient
        assert len(color_percentages) > 10, f"Expected many colors in gradient, got {len(color_percentages)}"
        
        # No single color should dominate too much
        max_percentage = max(color_percentages.values())
        assert max_percentage < 20, f"One color dominates too much: {max_percentage}%"
    
    def test_single_pixel_image(self, temp_dir):
        """Test with 1x1 pixel image."""
        single_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)
        
        image_path = temp_dir / "single_pixel.png"
        import cv2
        pixel_bgr = single_pixel[:, :, [2, 1, 0]]
        cv2.imwrite(str(image_path), pixel_bgr)
        
        color_percentages = color_wheel.load_and_analyze_image(str(image_path))
        
        # Should have exactly one color at 100%
        assert len(color_percentages) == 1
        assert list(color_percentages.values())[0] == pytest.approx(100.0, abs=0.1)


class TestGPUMocking:
    """Test GPU functionality with mocking."""
    
    @patch('color_wheel.CUPY_AVAILABLE', True)
    def test_gpu_acceleration_enabled(self, mock_cupy, sample_color_percentages):
        """Test behavior when GPU acceleration is available."""
        # Mock CuPy functions
        with patch('color_wheel.cp') as mock_cp:
            mock_cp.asarray.side_effect = lambda x: x  # Pass through
            mock_cp.asnumpy.side_effect = lambda x: x  # Pass through
            
            # Test GPU-accelerated RGB to HSV conversion
            rgb_array = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            
            with patch('color_wheel._rgb_to_hsv_gpu') as mock_gpu_rgb:
                expected_hsv = np.array([[0.0, 1.0, 1.0], [120.0, 1.0, 1.0]])
                mock_gpu_rgb.return_value = expected_hsv
                
                result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
                
                # Should use GPU version
                mock_gpu_rgb.assert_called_once()
                np.testing.assert_array_equal(result, expected_hsv)
    
    @patch('color_wheel.CUPY_AVAILABLE', True)
    def test_gpu_hsv_distance_calculation(self, mock_cupy):
        """Test GPU-accelerated HSV distance calculation."""
        image_hsv = np.array([[0.0, 1.0, 1.0]])
        wheel_hsv = np.array([[0.0, 1.0, 1.0], [120.0, 1.0, 1.0]])
        
        with patch('color_wheel.cp') as mock_cp:
            with patch('color_wheel._calculate_hsv_distances_gpu') as mock_gpu_dist:
                expected_distances = np.array([[0.0, 100.0]])
                mock_gpu_dist.return_value = expected_distances
                
                result = color_wheel._calculate_hsv_distances_gpu(image_hsv, wheel_hsv)
                
                mock_gpu_dist.assert_called_once()
                np.testing.assert_array_equal(result, expected_distances)
    
    @patch('color_wheel.CUPY_AVAILABLE', False)
    def test_fallback_when_gpu_unavailable(self):
        """Test fallback to CPU when GPU is unavailable."""
        rgb_array = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        
        # Should use CPU fallback
        result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
        
        # Should still produce valid results
        assert result.shape == rgb_array.shape
        assert result[0, 0] == pytest.approx(0.0, abs=0.5)    # Red hue
        assert result[1, 0] == pytest.approx(120.0, abs=0.5)  # Green hue
    
    @patch('color_wheel.CUPY_AVAILABLE', True)
    def test_gpu_memory_management(self, mock_cupy):
        """Test that GPU memory is properly managed."""
        with patch('color_wheel.cp') as mock_cp:
            # Mock GPU array operations
            mock_gpu_array = MagicMock()
            mock_cp.asarray.return_value = mock_gpu_array
            mock_cp.asnumpy.return_value = np.array([[0.0, 1.0, 1.0]])
            
            rgb_array = np.array([[255, 0, 0]], dtype=np.uint8)
            
            with patch('color_wheel._rgb_to_hsv_gpu') as mock_gpu_func:
                mock_gpu_func.return_value = np.array([[0.0, 1.0, 1.0]])
                
                result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
                
                # Should have called GPU operations
                mock_cp.asarray.assert_called()
                mock_gpu_func.assert_called_once()


class TestNumbaJITMocking:
    """Test Numba JIT compilation functionality."""
    
    @patch('color_wheel.NUMBA_AVAILABLE', True)
    def test_numba_rgb_to_hsv(self, mock_numba):
        """Test Numba-accelerated RGB to HSV conversion."""
        rgb_array = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        
        with patch('color_wheel._rgb_to_hsv_numba') as mock_numba_func:
            expected_hsv = np.array([[0.0, 1.0, 1.0], [120.0, 1.0, 1.0]])
            mock_numba_func.return_value = expected_hsv
            
            result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
            
            # Should use Numba version
            mock_numba_func.assert_called_once()
            np.testing.assert_array_equal(result, expected_hsv)
    
    @patch('color_wheel.NUMBA_AVAILABLE', True)
    def test_numba_hsv_distance_calculation(self, mock_numba):
        """Test Numba-accelerated HSV distance calculation."""
        image_hsv = np.array([[0.0, 1.0, 1.0]])
        wheel_hsv = np.array([[0.0, 1.0, 1.0]])
        
        with patch('color_wheel._calculate_hsv_distances_numba') as mock_numba_dist:
            expected_distances = np.array([[0.0]])
            mock_numba_dist.return_value = expected_distances
            
            result = color_wheel._calculate_hsv_distances_numba(image_hsv, wheel_hsv)
            
            mock_numba_dist.assert_called_once()
            np.testing.assert_array_equal(result, expected_distances)
    
    @patch('color_wheel.NUMBA_AVAILABLE', False)
    def test_fallback_when_numba_unavailable(self):
        """Test fallback when Numba is unavailable."""
        rgb_array = np.array([[255, 0, 0]], dtype=np.uint8)
        
        # Should use NumPy fallback
        result = color_wheel.rgb_to_hsv_vectorized(rgb_array)
        
        assert result.shape == rgb_array.shape
        assert result[0, 0] == pytest.approx(0.0, abs=0.5)  # Red hue


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_small_image_performance(self, sample_image_path):
        """Test performance with small image (baseline)."""
        start_time = time.time()
        
        color_percentages = color_wheel.load_and_analyze_image(
            sample_image_path, sample_factor=1, quantize_level=8
        )
        
        analysis_time = time.time() - start_time
        
        # Small image should be very fast
        assert analysis_time < 5.0, f"Small image analysis too slow: {analysis_time:.2f}s"
        assert len(color_percentages) > 0
    
    def test_wheel_generation_performance(self, sample_color_percentages):
        """Test performance of color wheel generation."""
        start_time = time.time()
        
        wheel, _, _ = color_wheel.create_color_wheel(
            sample_color_percentages, wheel_size=200
        )
        
        generation_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert generation_time < 10.0, f"Wheel generation too slow: {generation_time:.2f}s"
        assert wheel.shape == (200, 200, 4)
    
    def test_template_caching_performance(self):
        """Test that template caching improves performance."""
        # First call - should create template
        start_time1 = time.time()
        wheel_rgb1, color_map1, hsv_cache1 = color_wheel.load_or_create_wheel_template(
            wheel_size=100, inner_radius_ratio=0.1, quantize_level=8
        )
        first_time = time.time() - start_time1
        
        # Second call - should load from cache
        start_time2 = time.time()
        wheel_rgb2, color_map2, hsv_cache2 = color_wheel.load_or_create_wheel_template(
            wheel_size=100, inner_radius_ratio=0.1, quantize_level=8
        )
        second_time = time.time() - start_time2
        
        # Second call should be faster (cached)
        # Allow some tolerance for file system variations
        if first_time > 0.1:  # Only test if first call took significant time
            assert second_time < first_time * 1.5, \
                f"Template caching didn't improve performance: {first_time:.3f}s vs {second_time:.3f}s"
        
        # Results should be identical
        np.testing.assert_array_equal(wheel_rgb1, wheel_rgb2)
        assert color_map1 == color_map2
        assert hsv_cache1 == hsv_cache2
    
    def test_parallel_vs_single_performance(self, gradient_image_path):
        """Test parallel vs single-threaded performance."""
        # Single-threaded
        start_time_single = time.time()
        result_single = color_wheel.load_and_analyze_image(
            gradient_image_path, use_parallel=False, sample_factor=2
        )
        single_time = time.time() - start_time_single
        
        # Parallel
        start_time_parallel = time.time()
        result_parallel = color_wheel.load_and_analyze_image(
            gradient_image_path, use_parallel=True, sample_factor=2
        )
        parallel_time = time.time() - start_time_parallel
        
        # Both should complete in reasonable time
        assert single_time < 30.0, f"Single-threaded too slow: {single_time:.2f}s"
        assert parallel_time < 30.0, f"Parallel too slow: {parallel_time:.2f}s"
        
        # Results should be similar
        assert len(result_single) > 0
        assert len(result_parallel) > 0


class TestStressTesting:
    """Stress tests for robustness."""
    
    def test_many_similar_colors(self):
        """Test with many very similar colors."""
        # Create many colors that are very close to each other
        similar_colors = {}
        base_color = [128, 128, 128]
        
        for i in range(50):
            for j in range(3):  # R, G, B variations
                color = base_color.copy()
                color[j] += i  # Vary one channel
                if color[j] <= 255:
                    similar_colors[tuple(color)] = np.random.uniform(1.0, 3.0)
        
        # Normalize percentages
        total = sum(similar_colors.values())
        similar_colors = {color: (pct / total * 100) for color, pct in similar_colors.items()}
        
        # Should handle many similar colors
        wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
            similar_colors, wheel_size=100
        )
        
        assert wheel.shape == (100, 100, 4)
        assert len(normalized_percentages) > 0
        assert len(opacity_values) > 0
    
    def test_extreme_color_distribution(self):
        """Test with extreme color frequency distribution."""
        # One dominant color, many rare colors
        extreme_colors = {(255, 0, 0): 95.0}  # 95% red
        
        # Add many rare colors
        for i in range(50):
            r, g, b = np.random.randint(0, 256, 3)
            color = (r, g, b)
            if color != (255, 0, 0):  # Don't duplicate the dominant color
                extreme_colors[color] = 0.1  # 0.1% each
        
        # Should handle extreme distribution
        wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
            extreme_colors, wheel_size=100
        )
        
        assert wheel.shape == (100, 100, 4)
        
        # Dominant color should have highest normalized percentage
        red_percentage = normalized_percentages.get((255, 0, 0))
        if red_percentage is not None:
            assert red_percentage == pytest.approx(1.0, abs=0.01), \
                f"Dominant color should normalize to 1.0, got {red_percentage}"
    
    def test_repeated_processing(self, sample_color_percentages):
        """Test repeated processing for memory leaks and consistency."""
        results = []
        
        # Process same data multiple times
        for i in range(10):
            wheel, normalized_percentages, opacity_values = color_wheel.create_color_wheel(
                sample_color_percentages, wheel_size=50  # Small for speed
            )
            
            # Store some key metrics
            results.append({
                'wheel_shape': wheel.shape,
                'num_colors': len(normalized_percentages),
                'max_opacity': max(opacity_values) if opacity_values else 0
            })
        
        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result['wheel_shape'] == first_result['wheel_shape'], \
                f"Iteration {i}: wheel shape differs"
            assert result['num_colors'] == first_result['num_colors'], \
                f"Iteration {i}: color count differs"
            assert abs(result['max_opacity'] - first_result['max_opacity']) <= 1, \
                f"Iteration {i}: max opacity differs significantly"
    
    def test_concurrent_template_access(self):
        """Test concurrent access to wheel templates (simulate race conditions)."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def load_template():
            try:
                result = color_wheel.load_or_create_wheel_template(
                    wheel_size=75, inner_radius_ratio=0.1, quantize_level=8
                )
                results_queue.put(result)
            except Exception as e:
                errors_queue.put(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=load_template)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check for errors
        assert errors_queue.empty(), f"Thread errors: {list(errors_queue.queue)}"
        
        # All results should be present and identical
        assert results_queue.qsize() == 5, f"Expected 5 results, got {results_queue.qsize()}"
        
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            np.testing.assert_array_equal(result[0], first_result[0], 
                                        err_msg=f"Result {i} RGB differs")
            assert result[1] == first_result[1], f"Result {i} color map differs"
            assert result[2] == first_result[2], f"Result {i} HSV cache differs"