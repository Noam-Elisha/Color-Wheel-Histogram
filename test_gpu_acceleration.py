#!/usr/bin/env python3
"""
Test GPU acceleration performance compared to CPU-only processing.
This script will test the color wheel generation with and without GPU acceleration.
"""

import time
import numpy as np
import sys
import os

# Add current directory to path to import color_wheel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_performance():
    """Test performance with different acceleration methods."""
    
    # Test data - simulate image colors and wheel colors
    num_image_colors = 5000
    num_wheel_colors = 2000
    
    print(f"Performance Test: {num_image_colors:,} image colors × {num_wheel_colors:,} wheel colors")
    print(f"Total comparisons: {num_image_colors * num_wheel_colors:,}")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(42)  # For reproducible results
    image_hsv = np.random.rand(num_image_colors, 3).astype(np.float32)
    wheel_hsv = np.random.rand(num_wheel_colors, 3).astype(np.float32)
    
    # Import our functions
    try:
        from color_wheel import (_calculate_hsv_distances_numpy, 
                                _calculate_hsv_distances_gpu,
                                _calculate_hsv_distances_numba,
                                CUPY_AVAILABLE, NUMBA_AVAILABLE)
    except ImportError as e:
        print(f"Error importing color_wheel functions: {e}")
        return
    
    results = []
    
    # Test NumPy version (baseline)
    print("Testing NumPy (CPU) distance calculation...")
    start_time = time.time()
    numpy_distances = _calculate_hsv_distances_numpy(image_hsv, wheel_hsv)
    numpy_time = time.time() - start_time
    results.append(("NumPy (CPU)", numpy_time))
    print(f"NumPy time: {numpy_time:.3f}s")
    print()
    
    # Test Numba version if available
    if NUMBA_AVAILABLE:
        print("Testing Numba JIT (CPU) distance calculation...")
        start_time = time.time()
        numba_distances = _calculate_hsv_distances_numba(image_hsv, wheel_hsv)
        numba_time = time.time() - start_time
        results.append(("Numba JIT (CPU)", numba_time))
        print(f"Numba time: {numba_time:.3f}s")
        print(f"Numba speedup: {numpy_time / numba_time:.1f}x faster than NumPy")
        print()
    else:
        print("Numba not available - skipping Numba test")
        print()
    
    # Test GPU version if available
    if CUPY_AVAILABLE:
        print("Testing CuPy (GPU) distance calculation...")
        start_time = time.time()
        gpu_distances = _calculate_hsv_distances_gpu(image_hsv, wheel_hsv)
        gpu_time = time.time() - start_time
        results.append(("CuPy (GPU)", gpu_time))
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"GPU speedup: {numpy_time / gpu_time:.1f}x faster than NumPy")
        print()
        
        # Verify results are similar (allowing for floating point differences)
        if np.allclose(numpy_distances, gpu_distances, rtol=1e-5, atol=1e-6):
            print("✓ GPU results match NumPy results (within tolerance)")
        else:
            print("⚠ GPU results differ from NumPy results")
            max_diff = np.max(np.abs(numpy_distances - gpu_distances))
            print(f"Maximum difference: {max_diff:.2e}")
        print()
    else:
        print("CuPy not available - skipping GPU test")
        print("To enable GPU acceleration, install CuPy:")
        print("  pip install cupy-cuda11x   # For CUDA 11.x")
        print("  pip install cupy-cuda12x   # For CUDA 12.x")
        print()
    
    # Results summary
    print("Performance Summary:")
    print("=" * 50)
    results.sort(key=lambda x: x[1])  # Sort by time
    
    fastest_time = results[0][1]
    for method, exec_time in results:
        speedup = fastest_time / exec_time if exec_time > 0 else float('inf')
        print(f"{method:20s}: {exec_time:6.3f}s  ({speedup:4.1f}x vs fastest)")


def test_rgb_to_hsv_performance():
    """Test RGB to HSV conversion performance."""
    
    print("\nRGB to HSV Conversion Performance Test")
    print("=" * 50)
    
    # Generate test RGB data
    num_colors = 50000
    np.random.seed(42)
    rgb_data = np.random.rand(num_colors, 3).astype(np.float32)
    
    print(f"Converting {num_colors:,} RGB colors to HSV")
    
    try:
        from color_wheel import (_rgb_to_hsv_numpy, _rgb_to_hsv_gpu, 
                                _rgb_to_hsv_numba, CUPY_AVAILABLE, NUMBA_AVAILABLE)
    except ImportError as e:
        print(f"Error importing color_wheel functions: {e}")
        return
    
    # Test NumPy version
    print("Testing NumPy RGB→HSV conversion...")
    start_time = time.time()
    numpy_hsv = _rgb_to_hsv_numpy(rgb_data)
    numpy_time = time.time() - start_time
    print(f"NumPy time: {numpy_time:.3f}s")
    
    # Test Numba version if available
    if NUMBA_AVAILABLE:
        print("Testing Numba RGB→HSV conversion...")
        start_time = time.time()
        numba_hsv = _rgb_to_hsv_numba(rgb_data)
        numba_time = time.time() - start_time
        print(f"Numba time: {numba_time:.3f}s")
        print(f"Numba speedup: {numpy_time / numba_time:.1f}x faster than NumPy")
    
    # Test GPU version if available
    if CUPY_AVAILABLE:
        print("Testing GPU RGB→HSV conversion...")
        start_time = time.time()
        gpu_hsv = _rgb_to_hsv_gpu(rgb_data)
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"GPU speedup: {numpy_time / gpu_time:.1f}x faster than NumPy")
        
        # Verify results
        if np.allclose(numpy_hsv, gpu_hsv, rtol=1e-5, atol=1e-6):
            print("✓ GPU RGB→HSV results match NumPy results")
        else:
            print("⚠ GPU RGB→HSV results differ from NumPy results")


if __name__ == "__main__":
    print("GPU Acceleration Performance Test")
    print("=" * 80)
    
    # Check what's available
    try:
        from color_wheel import CUPY_AVAILABLE, NUMBA_AVAILABLE
        
        print("Available acceleration methods:")
        print(f"  CuPy (GPU):     {'✓ Available' if CUPY_AVAILABLE else '✗ Not available'}")
        print(f"  Numba (JIT):    {'✓ Available' if NUMBA_AVAILABLE else '✗ Not available'}")
        print()
        
    except ImportError:
        print("Could not import color_wheel module")
        sys.exit(1)
    
    # Run performance tests
    test_performance()
    test_rgb_to_hsv_performance()
    
    print("\nTest completed!")