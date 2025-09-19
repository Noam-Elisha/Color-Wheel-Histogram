#!/usr/bin/env python3
"""
Color Wheel Generator

This script creates a color wheel visualization where the opacity of each pixel
corresponds to the frequency of that color in the input image. It's like a 
visual color histogram mapped onto a color wheel.

Usage: python color_wheel.py input_image.jpg output_wheel.png
"""

import numpy as np
import cv2
import argparse
import colorsys
import math
import matplotlib.pyplot as plt
import pickle
import os
import time
import mmap
import multiprocessing as mp
from functools import partial
try:
    from sklearn.neighbors import KDTree
    KDTREE_AVAILABLE = True
except ImportError:
    KDTREE_AVAILABLE = False
    print("Warning: scikit-learn not available. Using fallback nearest neighbor search.")
    print("For better performance with large color sets, install scikit-learn: pip install scikit-learn")

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Using standard NumPy operations.")
    print("For better performance with numerical computations, install numba: pip install numba")

try:
    import colour
    COLOUR_SCIENCE_AVAILABLE = True
except ImportError:
    COLOUR_SCIENCE_AVAILABLE = False
    print("Warning: colour-science not available. Adobe RGB conversion not supported.")
    print("For Adobe RGB support, install colour-science: pip install colour-science")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print(f"GPU acceleration available: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    print("Warning: CuPy not available. GPU acceleration not supported.")
    print("For GPU acceleration, install CuPy: pip install cupy-cuda11x or cupy-cuda12x")
except Exception as e:
    CUPY_AVAILABLE = False
    cp = None
    print(f"Warning: CuPy available but GPU not accessible: {e}")
    print("Falling back to CPU processing.")


def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def prefilter_and_deduplicate_colors(color_percentages, similarity_threshold=8):
    """
    Pre-filter and deduplicate similar colors to reduce nearest-neighbor search complexity.
    Groups colors that are within similarity_threshold distance in RGB space.
    
    Args:
        color_percentages (dict): Original color frequency percentages {(r,g,b): percentage}
        similarity_threshold (int): RGB distance threshold for grouping colors (default: 8)
        
    Returns:
        tuple: (filtered_color_percentages, color_mapping)
            - filtered_color_percentages: dict with representative colors and combined frequencies
            - color_mapping: dict mapping original colors to their representative colors
    """
    if not color_percentages:
        return {}, {}
    
    original_count = len(color_percentages)
    
    # FAST APPROACH: Use spatial binning/hashing instead of O(N²) comparison
    # Create bins based on similarity threshold - colors in same bin are similar
    bin_size = similarity_threshold
    bins = {}  # bin_key -> list of colors in that bin
    
    for color in color_percentages.keys():
        r, g, b = color
        # Create bin coordinates by integer division
        bin_key = (r // bin_size, g // bin_size, b // bin_size)
        
        if bin_key not in bins:
            bins[bin_key] = []
        bins[bin_key].append(color)
    
    # For each bin, choose representative color and combine frequencies
    filtered_color_percentages = {}
    color_mapping = {}
    
    for bin_colors in bins.values():
        if not bin_colors:
            continue
            
        # Choose representative color (the one with highest frequency in this bin)
        representative_color = max(bin_colors, key=lambda c: color_percentages[c])
        
        # Combine frequencies
        combined_frequency = sum(color_percentages[color] for color in bin_colors)
        
        # Store the result
        filtered_color_percentages[representative_color] = combined_frequency
        
        # Create mapping for all colors in this bin
        for color in bin_colors:
            color_mapping[color] = representative_color
    
    filtered_count = len(filtered_color_percentages)
    reduction_percent = (1 - filtered_count / original_count) * 100
    
    print(f"Color pre-filtering: {original_count:,} → {filtered_count:,} colors ({reduction_percent:.1f}% reduction)")
    
    return filtered_color_percentages, color_mapping


def convert_adobe_rgb_to_srgb(image_rgb):
    """
    Convert Adobe RGB image to sRGB using colour-science library.
    
    Args:
        image_rgb (numpy.ndarray): RGB image in Adobe RGB color space, values 0-255
        
    Returns:
        numpy.ndarray: RGB image converted to sRGB color space, values 0-255
    """
    if not COLOUR_SCIENCE_AVAILABLE:
        print("Warning: colour-science not available. Cannot convert Adobe RGB.")
        print("Using image as-is (assuming sRGB). Install colour-science for proper conversion.")
        return image_rgb
    
    try:
        # Normalize to 0-1 range for colour-science
        normalized = image_rgb.astype(np.float64) / 255.0
        
        # Convert from Adobe RGB to XYZ, then XYZ to sRGB
        # Adobe RGB uses D65 illuminant and 2.2 gamma
        xyz = colour.RGB_to_XYZ(
            normalized,
            colourspace=colour.RGB_COLOURSPACES['Adobe RGB (1998)'],
            illuminant=colour.RGB_COLOURSPACES['Adobe RGB (1998)'].whitepoint
        )
        
        srgb = colour.XYZ_to_RGB(
            xyz,
            colourspace=colour.RGB_COLOURSPACES['sRGB'],
            illuminant=colour.RGB_COLOURSPACES['sRGB'].whitepoint
        )
        
        # Clamp to valid range and convert back to 0-255
        srgb = np.clip(srgb, 0, 1)
        return (srgb * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Warning: Adobe RGB conversion failed: {e}")
        print("Using image as-is (assuming sRGB)")
        return image_rgb


def convert_adobe_rgb_matrix_method(image_rgb):
    """
    Fallback Adobe RGB to sRGB conversion using transformation matrix.
    This is less accurate than the colour-science method but works without dependencies.
    
    Args:
        image_rgb (numpy.ndarray): RGB image in Adobe RGB color space, values 0-255
        
    Returns:
        numpy.ndarray: RGB image converted to sRGB color space, values 0-255
    """
    # Adobe RGB to XYZ matrix (D65 illuminant)
    adobe_to_xyz = np.array([
        [0.5767309, 0.1855540, 0.1881852],
        [0.2973769, 0.6273491, 0.0752741],
        [0.0270343, 0.0706872, 0.9911085]
    ])
    
    # XYZ to sRGB matrix (D65 illuminant)
    xyz_to_srgb = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])
    
    # Combine matrices: Adobe RGB -> XYZ -> sRGB
    conversion_matrix = np.dot(xyz_to_srgb, adobe_to_xyz)
    
    # Normalize input to 0-1
    normalized = image_rgb.astype(np.float64) / 255.0
    
    # Apply gamma correction (Adobe RGB uses 2.2 gamma)
    linearized = np.power(normalized, 2.2)
    
    # Apply color space conversion
    original_shape = linearized.shape
    linearized_flat = linearized.reshape(-1, 3)
    converted_flat = np.dot(linearized_flat, conversion_matrix.T)
    converted = converted_flat.reshape(original_shape)
    
    # Apply sRGB gamma correction
    srgb_linear = np.where(
        converted <= 0.0031308,
        12.92 * converted,
        1.055 * np.power(converted, 1.0/2.4) - 0.055
    )
    
    # Clamp to valid range and convert back to 0-255
    srgb_linear = np.clip(srgb_linear, 0, 1)
    return (srgb_linear * 255).astype(np.uint8)


def create_wheel_template(wheel_size=800, inner_radius_ratio=0.1, quantize_level=8):
    """
    Create a precomputed wheel template with full-resolution RGB values and quantized color lookup.
    This generates a high-quality wheel with smooth gradients while maintaining compatibility 
    with quantized input image analysis.
    
    Returns:
        tuple: (wheel_rgb, color_to_pixels_map, wheel_hsv_cache)
            - wheel_rgb: (H, W, 3) array with full-resolution RGB values
            - color_to_pixels_map: dict mapping quantized (r,g,b) -> list of (y,x) coordinates
            - wheel_hsv_cache: dict mapping quantized (r,g,b) -> (h,s,v) for cached HSV conversions
    """
    center = wheel_size // 2
    outer_radius = center - 10
    inner_radius = int(outer_radius * inner_radius_ratio)
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:wheel_size, 0:wheel_size]
    
    # Calculate distances and angles for all pixels at once
    dx = x_coords - center
    dy = y_coords - center
    distances = np.sqrt(dx*dx + dy*dy)
    
    # Create mask for valid pixels (within ring)
    valid_mask = (distances <= outer_radius) & (distances >= inner_radius)
    
    # Only process valid pixels to save computation
    valid_indices = np.where(valid_mask)
    valid_dx = dx[valid_indices]
    valid_dy = dy[valid_indices]
    valid_distances = distances[valid_indices]
    
    # Convert to polar coordinates (vectorized)
    # Fix angle calculation to match circular spectrum: 0° = right, counter-clockwise
    angles = np.arctan2(valid_dy, valid_dx)  # This gives -π to π
    hues = -angles * 180 / np.pi  # Convert to degrees and flip direction for counter-clockwise
    hues = (hues + 360) % 360     # Convert to 0-360 range
    
    # Calculate saturation with higher precision for smoother gradients
    saturations = (valid_distances - inner_radius) / (outer_radius - inner_radius)
    values = np.ones_like(saturations)
    
    # Vectorized HSV to RGB conversion with full precision
    hues_norm = hues / 360.0
    c = values * saturations
    x = c * (1 - np.abs((hues_norm * 6) % 2 - 1))
    m = values - c
    
    # Determine RGB values based on hue sector
    h_sector = (hues_norm * 6).astype(int) % 6
    
    r_vals = np.zeros_like(hues_norm)
    g_vals = np.zeros_like(hues_norm)
    b_vals = np.zeros_like(hues_norm)
    
    # Sector calculations with full precision
    mask0 = h_sector == 0
    r_vals[mask0] = c[mask0]
    g_vals[mask0] = x[mask0]
    
    mask1 = h_sector == 1
    r_vals[mask1] = x[mask1]
    g_vals[mask1] = c[mask1]
    
    mask2 = h_sector == 2
    g_vals[mask2] = c[mask2]
    b_vals[mask2] = x[mask2]
    
    mask3 = h_sector == 3
    g_vals[mask3] = x[mask3]
    b_vals[mask3] = c[mask3]
    
    mask4 = h_sector == 4
    r_vals[mask4] = x[mask4]
    b_vals[mask4] = c[mask4]
    
    mask5 = h_sector == 5
    r_vals[mask5] = c[mask5]
    b_vals[mask5] = x[mask5]
    
    # Add m to get final RGB values
    r_vals += m
    g_vals += m
    b_vals += m
    
    # Convert to full-resolution 0-255 range (no quantization for display)
    wheel_r_full = (r_vals * 255).astype(np.uint8)
    wheel_g_full = (g_vals * 255).astype(np.uint8)
    wheel_b_full = (b_vals * 255).astype(np.uint8)
    
    # Create the base RGB wheel with full resolution
    wheel_rgb = np.zeros((wheel_size, wheel_size, 3), dtype=np.uint8)
    wheel_rgb[valid_indices[0], valid_indices[1], 0] = wheel_r_full
    wheel_rgb[valid_indices[0], valid_indices[1], 1] = wheel_g_full
    wheel_rgb[valid_indices[0], valid_indices[1], 2] = wheel_b_full
    
    # Create quantized versions ONLY for the color mapping (to match input image analysis)
    if quantize_level > 1:
        wheel_r_quantized = (wheel_r_full // quantize_level) * quantize_level
        wheel_g_quantized = (wheel_g_full // quantize_level) * quantize_level
        wheel_b_quantized = (wheel_b_full // quantize_level) * quantize_level
    else:
        wheel_r_quantized = wheel_r_full
        wheel_g_quantized = wheel_g_full
        wheel_b_quantized = wheel_b_full
    
    # Create color-to-pixels mapping using quantized colors (for compatibility with input analysis)
    color_to_pixels_map = {}
    
    # Pre-allocate arrays for quantized colors to avoid repeated calculations
    num_pixels = len(valid_indices[0])
    if quantize_level > 1:
        wheel_r_quantized = (wheel_r_full // quantize_level) * quantize_level
        wheel_g_quantized = (wheel_g_full // quantize_level) * quantize_level
        wheel_b_quantized = (wheel_b_full // quantize_level) * quantize_level
    else:
        wheel_r_quantized = wheel_r_full
        wheel_g_quantized = wheel_g_full
        wheel_b_quantized = wheel_b_full
    
    # Group pixels by color using vectorized operations
    unique_colors_int = (wheel_r_quantized.astype(np.int64) * 65536 + 
                        wheel_g_quantized.astype(np.int64) * 256 + 
                        wheel_b_quantized.astype(np.int64))
    
    # Get unique colors and their first occurrence indices
    unique_color_ints, inverse_indices = np.unique(unique_colors_int, return_inverse=True)
    
    # Create mapping using pre-allocated arrays AND cache HSV conversions
    wheel_hsv_cache = {}
    
    # Get unique quantized colors for HSV conversion
    unique_colors_rgb = []
    unique_color_tuples = []
    
    for i, color_int in enumerate(unique_color_ints):
        # Convert back to RGB
        r = int((color_int // 65536) % 256)
        g = int((color_int // 256) % 256)  
        b = int(color_int % 256)
        color_tuple = (r, g, b)
        unique_color_tuples.append(color_tuple)
        unique_colors_rgb.append([r/255.0, g/255.0, b/255.0])  # Normalized for HSV conversion
        
        # Find all pixels with this color
        pixel_mask = inverse_indices == i
        pixel_indices = np.where(pixel_mask)[0]
        
        # Get y,x coordinates for these pixels
        y_coords = valid_indices[0][pixel_indices]
        x_coords = valid_indices[1][pixel_indices]
        color_to_pixels_map[color_tuple] = np.column_stack([y_coords, x_coords])
    
    # VECTORIZED: Convert all unique wheel colors to HSV at once and cache them
    if unique_colors_rgb:
        unique_colors_array = np.array(unique_colors_rgb, dtype=np.float32)
        unique_hsv = rgb_to_hsv_vectorized(unique_colors_array)  # Shape: (N, 3)
        
        # Cache HSV values for each unique color
        for i, color_tuple in enumerate(unique_color_tuples):
            wheel_hsv_cache[color_tuple] = unique_hsv[i]  # Store as numpy array [h, s, v]
    
    return wheel_rgb, color_to_pixels_map, wheel_hsv_cache


def get_wheel_template_path(wheel_size, inner_radius_ratio, quantize_level):
    """Get the path for the wheel template file in a dedicated templates folder."""
    # Create templates directory if it doesn't exist
    templates_dir = "wheel_templates"
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    filename = f"wheel_template_fullres_{wheel_size}_{inner_radius_ratio:.3f}_q{quantize_level}.pkl"
    return os.path.join(templates_dir, filename)


def get_mmap_template_paths(wheel_size, inner_radius_ratio, quantize_level):
    """Get the paths for the memory-mapped wheel template files."""
    templates_dir = "wheel_templates"
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    base_name = f"wheel_mmap_{wheel_size}_{inner_radius_ratio:.3f}_q{quantize_level}"
    return {
        'rgb': os.path.join(templates_dir, f"{base_name}_rgb.dat"),
        'pixels': os.path.join(templates_dir, f"{base_name}_pixels.pkl"),  # Complex dict, keep as pickle
        'hsv': os.path.join(templates_dir, f"{base_name}_hsv.pkl"),      # HSV cache as pickle
        'meta': os.path.join(templates_dir, f"{base_name}_meta.pkl")     # Metadata (shape, etc.)
    }


def save_mmap_template(wheel_rgb, color_to_pixels_map, wheel_hsv_cache, wheel_size, inner_radius_ratio, quantize_level):
    """Save wheel template using memory-mapped files for faster loading."""
    paths = get_mmap_template_paths(wheel_size, inner_radius_ratio, quantize_level)
    
    # Save RGB data as binary file for memory mapping
    wheel_rgb_flat = wheel_rgb.astype(np.uint8)  # Ensure uint8 for smaller files
    wheel_rgb_flat.tofile(paths['rgb'])
    
    # Save metadata
    metadata = {
        'shape': wheel_rgb.shape,
        'dtype': 'uint8'
    }
    with open(paths['meta'], 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save pixel mapping and HSV cache (these are complex dicts, keep as pickle for now)
    with open(paths['pixels'], 'wb') as f:
        pickle.dump(color_to_pixels_map, f)
    
    with open(paths['hsv'], 'wb') as f:
        pickle.dump(wheel_hsv_cache, f)
    
    rgb_size_kb = os.path.getsize(paths['rgb']) // 1024
    print(f"Memory-mapped template saved (RGB: {rgb_size_kb}KB, instant loading)")


def load_mmap_template(wheel_size, inner_radius_ratio, quantize_level):
    """Load wheel template using memory-mapped files for instant access."""
    paths = get_mmap_template_paths(wheel_size, inner_radius_ratio, quantize_level)
    
    # Check if all required files exist
    required_files = ['rgb', 'pixels', 'hsv', 'meta']
    if not all(os.path.exists(paths[key]) for key in required_files):
        return None
    
    try:
        # Load metadata first
        with open(paths['meta'], 'rb') as f:
            metadata = pickle.load(f)
        
        # Load RGB data using memory mapping (true zero-copy access)
        with open(paths['rgb'], 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                wheel_rgb = np.frombuffer(mm, dtype=np.uint8).copy()
                wheel_rgb = wheel_rgb.reshape(metadata['shape'])
        
        # Load pixel mapping and HSV cache (still using pickle, but these are smaller)
        with open(paths['pixels'], 'rb') as f:
            color_to_pixels_map = pickle.load(f)
        
        with open(paths['hsv'], 'rb') as f:
            wheel_hsv_cache = pickle.load(f)
        
        rgb_size_kb = os.path.getsize(paths['rgb']) // 1024
        print(f"Memory-mapped template loaded (RGB: {rgb_size_kb}KB, zero-copy)")
        return wheel_rgb, color_to_pixels_map, wheel_hsv_cache
    
    except Exception as e:
        print(f"Failed to load memory-mapped template: {e}")
        return None


def load_or_create_wheel_template(wheel_size=800, inner_radius_ratio=0.1, quantize_level=8):
    """
    Load existing wheel template or create a new one if it doesn't exist.
    Prefers memory-mapped format for fastest loading, falls back to pickle format.
    
    Returns:
        tuple: (wheel_rgb, color_to_pixels_map, wheel_hsv_cache)
    """
    # Try to load memory-mapped template first (fastest)
    mmap_result = load_mmap_template(wheel_size, inner_radius_ratio, quantize_level)
    if mmap_result is not None:
        return mmap_result
    
    # Fall back to pickle template
    template_path = get_wheel_template_path(wheel_size, inner_radius_ratio, quantize_level)
    
    if os.path.exists(template_path):
        print(f"Loading pickle template (will upgrade to memory-mapped): {template_path}")
        with open(template_path, 'rb') as f:
            template_data = pickle.load(f)
        
        # Handle both old format (2 items) and new format (3 items) for backward compatibility
        if len(template_data) == 2:
            wheel_rgb, color_to_pixels_map = template_data
            # Create HSV cache for old templates
            print("Upgrading old template with HSV cache...")
            wheel_colors = list(color_to_pixels_map.keys())
            wheel_colors_array = np.array(wheel_colors, dtype=np.float32) / 255.0
            wheel_hsv = rgb_to_hsv_vectorized(wheel_colors_array)
            wheel_hsv_cache = {}
            for i, color in enumerate(wheel_colors):
                wheel_hsv_cache[color] = wheel_hsv[i]
            
            # Update template_data for saving
            template_data = (wheel_rgb, color_to_pixels_map, wheel_hsv_cache)
        else:
            wheel_rgb, color_to_pixels_map, wheel_hsv_cache = template_data
        
        # Save memory-mapped version for faster future loading
        print("Saving memory-mapped version for faster future loading...")
        save_mmap_template(wheel_rgb, color_to_pixels_map, wheel_hsv_cache, 
                          wheel_size, inner_radius_ratio, quantize_level)
        
        return wheel_rgb, color_to_pixels_map, wheel_hsv_cache
    else:
        print(f"Creating new wheel template...")
        template_data = create_wheel_template(wheel_size, inner_radius_ratio, quantize_level)
        wheel_rgb, color_to_pixels_map, wheel_hsv_cache = template_data
        
        # Save both pickle format (for compatibility) and memory-mapped format (for speed)
        print("Saving template in both pickle and memory-mapped formats...")
        with open(template_path, 'wb') as f:
            pickle.dump(template_data, f)
        
        save_mmap_template(wheel_rgb, color_to_pixels_map, wheel_hsv_cache,
                          wheel_size, inner_radius_ratio, quantize_level)
        
        print(f"Templates saved - next run will use instant memory-mapped loading")
        return template_data


def load_and_analyze_image(image_path, sample_factor=4, quantize_level=8, use_parallel=None, color_space="sRGB"):
    """
    Load an image and analyze color frequencies as percentages.
    
    Args:
        image_path (str): Path to the input image
        sample_factor (int): Factor to downsample image for faster processing
        quantize_level (int): Color quantization level (1=no quantization, higher=more grouping)
        use_parallel (bool): Use parallel processing for large images (None=auto-detect)
        color_space (str): Color space to use ("sRGB", "Adobe RGB", "ProPhoto RGB")
        
    Returns:
        dict: Color frequency percentages {(r,g,b): percentage}
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Handle different color spaces
    if color_space.lower() == "adobe rgb" or color_space.lower() == "adobergb":
        print("Converting from Adobe RGB to sRGB for processing...")
        conversion_start = time.time()
        if COLOUR_SCIENCE_AVAILABLE:
            print("Using colour-science library for accurate conversion")
            image = convert_adobe_rgb_to_srgb(image)
        else:
            print("Using matrix-based conversion (colour-science not available)")
            image = convert_adobe_rgb_matrix_method(image)
        conversion_time = time.time() - conversion_start
        print(f"Color space conversion completed in {format_time(conversion_time)}")
    elif color_space.lower() == "prophoto rgb" or color_space.lower() == "prophotorgb":
        print("Warning: ProPhoto RGB conversion not yet implemented.")
        print("ProPhoto RGB has an even wider color gamut than Adobe RGB.")
        print("Processing as sRGB - colors may be inaccurate.")
    else:
        # Default sRGB - most common case
        pass
    
    # Downsample for faster processing
    if sample_factor > 1:
        height, width = image.shape[:2]
        new_height, new_width = height // sample_factor, width // sample_factor
        image = cv2.resize(image, (new_width, new_height))
    
    # Count color frequencies using vectorized operations
    pixels = image.reshape(-1, 3)
    total_pixels = len(pixels)
    
    print(f"Analyzing {total_pixels:,} pixels...")
    analysis_start = time.time()
    
    # Determine if we should use parallel processing
    if use_parallel is None:
        use_parallel = total_pixels > 1_000_000  # Use parallel for images > 1M pixels
    
    # Quantize colors to reduce noise (group similar colors) - vectorized
    quantize_start = time.time()
    if quantize_level > 1:
        quantized_pixels = (pixels // quantize_level) * quantize_level
    else:
        quantized_pixels = pixels  # No quantization
    quantize_time = time.time() - quantize_start
    
    color_counting_start = time.time()
    if use_parallel and total_pixels > 500_000:
        print("Using parallel processing for color analysis...")
        color_percentages = _analyze_colors_parallel(quantized_pixels, total_pixels)
    else:
        print("Using single-threaded color analysis...")
        color_percentages = _analyze_colors_single(quantized_pixels, total_pixels)
    color_counting_time = time.time() - color_counting_start
    
    analysis_time = time.time() - analysis_start
    print(f"Color analysis completed in {format_time(analysis_time)} (quantization: {format_time(quantize_time)}, counting: {format_time(color_counting_time)})")
    
    return color_percentages


def _analyze_colors_single(quantized_pixels, total_pixels):
    """Single-threaded color analysis (original implementation)."""
    # Use NumPy's unique function with return_counts for efficient counting
    # Convert RGB tuples to a single integer for efficient unique counting (using int64 to avoid overflow)
    rgb_as_int = quantized_pixels[:, 0].astype(np.int64) * 65536 + quantized_pixels[:, 1].astype(np.int64) * 256 + quantized_pixels[:, 2].astype(np.int64)
    unique_colors, counts = np.unique(rgb_as_int, return_counts=True)
    
    # Pre-allocate arrays for RGB conversion
    num_unique = len(unique_colors)
    r_values = ((unique_colors // 65536) % 256).astype(np.uint8)
    g_values = ((unique_colors // 256) % 256).astype(np.uint8)  
    b_values = (unique_colors % 256).astype(np.uint8)
    percentages = counts / total_pixels
    
    # Create percentage dictionary using vectorized operations
    color_percentages = {}
    for i in range(num_unique):
        color_percentages[(r_values[i], g_values[i], b_values[i])] = percentages[i]
    
    return color_percentages


def _analyze_colors_parallel(quantized_pixels, total_pixels):
    """Parallel color analysis using multiprocessing."""
    # Split pixels into chunks for parallel processing
    num_cores = min(mp.cpu_count(), 8)  # Limit to 8 cores to avoid memory issues
    chunk_size = len(quantized_pixels) // num_cores
    
    if chunk_size < 10000:  # Not worth parallelizing for small chunks
        return _analyze_colors_single(quantized_pixels, total_pixels)
    
    # Create chunks
    chunks = []
    for i in range(num_cores):
        start_idx = i * chunk_size
        if i == num_cores - 1:
            end_idx = len(quantized_pixels)  # Last chunk gets remainder
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append(quantized_pixels[start_idx:end_idx])
    
    # Process chunks in parallel
    with mp.Pool(num_cores) as pool:
        chunk_results = pool.map(_process_color_chunk, chunks)
    
    # Merge results from all chunks
    merged_color_counts = {}
    for chunk_result in chunk_results:
        for color_int, count in chunk_result.items():
            if color_int in merged_color_counts:
                merged_color_counts[color_int] += count
            else:
                merged_color_counts[color_int] = count
    
    # Convert back to RGB tuples and create percentage dictionary
    color_percentages = {}
    for color_int, count in merged_color_counts.items():
        # Convert back to RGB tuple
        r = int((color_int // 65536) % 256)
        g = int((color_int // 256) % 256)
        b = int(color_int % 256)
        color_percentages[(r, g, b)] = count / total_pixels
    
    return color_percentages


def _process_color_chunk(chunk):
    """Process a chunk of pixels for color counting. Used by parallel processing."""
    # Convert RGB tuples to integers for efficient counting
    rgb_as_int = chunk[:, 0].astype(np.int64) * 65536 + chunk[:, 1].astype(np.int64) * 256 + chunk[:, 2].astype(np.int64)
    unique_colors, counts = np.unique(rgb_as_int, return_counts=True)
    
    # Return as dictionary
    return dict(zip(unique_colors, counts))


def rgb_to_hsv_normalized(r, g, b):
    """
    Convert RGB values (0-255) to HSV values.
    
    Returns:
        tuple: (h, s, v) where h is in [0, 360), s and v are in [0, 1]
    """
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return h * 360, s, v


def rgb_to_hsv_vectorized(rgb_array):
    """
    Vectorized RGB to HSV conversion using NumPy operations.
    Automatically uses GPU acceleration with CuPy if available.
    
    Args:
        rgb_array: numpy array of shape (N, 3) with RGB values in range [0, 1]
        
    Returns:
        numpy array of shape (N, 3) with HSV values
    """
    if CUPY_AVAILABLE and len(rgb_array) > 1000:  # Use GPU for larger arrays
        return _rgb_to_hsv_gpu(rgb_array)
    elif NUMBA_AVAILABLE:
        return _rgb_to_hsv_numba(rgb_array)
    else:
        return _rgb_to_hsv_numpy(rgb_array)


def _rgb_to_hsv_gpu(rgb_array):
    """
    GPU-accelerated RGB to HSV conversion using CuPy.
    Significantly faster for large arrays.
    """
    if not CUPY_AVAILABLE:
        return _rgb_to_hsv_numpy(rgb_array)
    
    try:
        # Transfer to GPU
        rgb_gpu = cp.asarray(rgb_array)
        n = rgb_gpu.shape[0]
        hsv_gpu = cp.zeros((n, 3), dtype=cp.float32)
        
        # Vectorized operations on GPU
        maxc = cp.max(rgb_gpu, axis=1)
        minc = cp.min(rgb_gpu, axis=1)
        
        # Value is the maximum
        hsv_gpu[:, 2] = maxc
        
        # Saturation
        delta = maxc - minc
        hsv_gpu[:, 1] = cp.where(maxc != 0, delta / maxc, 0)
        
        # Hue calculation - vectorized on GPU
        h = cp.zeros(n)
        
        # Only calculate hue where there's color (delta > 0)
        mask = delta != 0
        
        if cp.any(mask):
            rgb_masked = rgb_gpu[mask]
            maxc_masked = maxc[mask]
            delta_masked = delta[mask]
            
            # Red is max
            red_max = (rgb_masked[:, 0] == maxc_masked)
            h[mask] = cp.where(red_max, 
                              (rgb_masked[:, 1] - rgb_masked[:, 2]) / delta_masked,
                              h[mask])
            
            # Green is max
            green_max = (rgb_masked[:, 1] == maxc_masked) & ~red_max
            h[mask] = cp.where(green_max,
                              2.0 + (rgb_masked[:, 2] - rgb_masked[:, 0]) / delta_masked,
                              h[mask])
            
            # Blue is max
            blue_max = (rgb_masked[:, 2] == maxc_masked) & ~red_max & ~green_max
            h[mask] = cp.where(blue_max,
                              4.0 + (rgb_masked[:, 0] - rgb_masked[:, 1]) / delta_masked,
                              h[mask])
            
            # Normalize hue to [0, 1]
            h[mask] = h[mask] / 6.0
            h[mask] = cp.where(h[mask] < 0, h[mask] + 1, h[mask])
        
        hsv_gpu[:, 0] = h
        
        # Transfer back to CPU
        result = cp.asnumpy(hsv_gpu)
        print(f"GPU RGB→HSV conversion completed for {n:,} colors")
        return result
        
    except Exception as e:
        print(f"GPU RGB→HSV conversion failed: {e}, falling back to CPU")
        return _rgb_to_hsv_numpy(rgb_array)


@jit(nopython=True, parallel=True, cache=True) if NUMBA_AVAILABLE else lambda f: f
def _rgb_to_hsv_numba(rgb_array):
    """
    Numba JIT-compiled RGB to HSV conversion for maximum performance.
    Uses parallel processing for large arrays.
    """
    n = rgb_array.shape[0]
    hsv = np.zeros((n, 3), dtype=np.float32)
    
    for i in prange(n):
        r, g, b = rgb_array[i, 0], rgb_array[i, 1], rgb_array[i, 2]
        
        maxc = max(r, g, b)
        minc = min(r, g, b)
        
        # Value
        hsv[i, 2] = maxc
        
        # Saturation
        if maxc != 0:
            hsv[i, 1] = (maxc - minc) / maxc
        else:
            hsv[i, 1] = 0
        
        # Hue
        if maxc == minc:
            hsv[i, 0] = 0  # Gray
        else:
            delta = maxc - minc
            if maxc == r:
                hsv[i, 0] = (g - b) / delta
                if hsv[i, 0] < 0:
                    hsv[i, 0] += 6
            elif maxc == g:
                hsv[i, 0] = 2.0 + (b - r) / delta
            else:  # maxc == b
                hsv[i, 0] = 4.0 + (r - g) / delta
            
            hsv[i, 0] = hsv[i, 0] / 6.0
    
    return hsv


def _rgb_to_hsv_numpy(rgb_array):
    """
    NumPy fallback RGB to HSV conversion (original implementation).
    """
    rgb = rgb_array
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    
    # Value is the maximum
    v = maxc
    
    # Saturation
    delta = maxc - minc
    s = np.where(maxc != 0, delta / maxc, 0)
    
    # Hue
    h = np.zeros(len(rgb))
    
    # Only calculate hue where there's color (delta > 0)
    mask = delta != 0
    
    if np.any(mask):
        rgb_masked = rgb[mask]
        maxc_masked = maxc[mask]
        delta_masked = delta[mask]
        
        # Red is max
        red_max = (rgb_masked[:, 0] == maxc_masked)
        h[mask] = np.where(red_max, 
                          (rgb_masked[:, 1] - rgb_masked[:, 2]) / delta_masked,
                          h[mask])
        
        # Green is max
        green_max = (rgb_masked[:, 1] == maxc_masked) & ~red_max
        h[mask] = np.where(green_max,
                          2.0 + (rgb_masked[:, 2] - rgb_masked[:, 0]) / delta_masked,
                          h[mask])
        
        # Blue is max
        blue_max = (rgb_masked[:, 2] == maxc_masked) & ~red_max & ~green_max
        h[mask] = np.where(blue_max,
                          4.0 + (rgb_masked[:, 0] - rgb_masked[:, 1]) / delta_masked,
                          h[mask])
        
        h[mask] = h[mask] / 6.0
        h[mask] = np.where(h[mask] < 0, h[mask] + 1.0, h[mask])
    
    return np.column_stack((h, s, v))


def find_nearest_wheel_colors_vectorized(image_colors, color_to_pixels_map, wheel_hsv_cache, force_kdtree=None, use_parallel=None):
    """
    Find the nearest wheel colors for multiple image colors using vectorized HSV-based matching.
    This properly maps colors based on hue (angle) and saturation (radius) like a real color wheel.
    Uses cached HSV values for wheel colors to avoid recomputation.
    
    Args:
        image_colors (list): List of RGB color tuples from the image [(r,g,b), ...]
        color_to_pixels_map (dict): Mapping of wheel colors to pixel coordinates
        wheel_hsv_cache (dict): Pre-computed HSV values for wheel colors {(r,g,b): [h,s,v]}
        force_kdtree (bool): Force KD-tree usage (True), disable it (False), or auto-detect (None)
        use_parallel (bool): Use parallel processing where applicable (None=auto-detect)
        
    Returns:
        dict: Mapping {image_color: nearest_wheel_color}
    """
    if not image_colors:
        return {}
    
    # Convert image colors to numpy array and normalize
    image_colors_array = np.array(image_colors, dtype=np.float32) / 255.0  # Shape: (N, 3)
    
    # Get wheel colors and their cached HSV values
    wheel_colors = list(color_to_pixels_map.keys())
    
    # Use cached HSV values for wheel colors (MUCH faster!)
    wheel_hsv = np.array([wheel_hsv_cache[color] for color in wheel_colors])  # Shape: (M, 3)
    
    # Vectorized RGB to HSV conversion for image colors only
    image_hsv = rgb_to_hsv_vectorized(image_colors_array)  # Shape: (N, 3)
    
    # Calculate dataset size metrics for decision making and logging
    num_image_colors = len(image_colors)
    num_wheel_colors = len(wheel_colors)
    total_comparisons = num_image_colors * num_wheel_colors
    
    # Determine whether to use KD-tree with intelligent thresholds
    if force_kdtree is True:
        use_kdtree = KDTREE_AVAILABLE
        decision_reasons = ["forced via --force-kdtree"]
        if not KDTREE_AVAILABLE:
            print("Warning: KD-tree requested but scikit-learn not available. Using fallback.")
            decision_reasons = ["forced but scikit-learn not available"]
    elif force_kdtree is False:
        use_kdtree = False
        decision_reasons = ["disabled via --no-kdtree"]
    else:
        # INTELLIGENT KD-TREE THRESHOLD LOGIC
        # KD-tree is beneficial when:
        # 1. We have enough data points (KD-tree has overhead)
        # 2. The brute force approach would be expensive
        # 3. We're not in a degenerate case (very few wheel colors)
        
        min_colors_for_kdtree = 20  # Lower threshold - KD-tree can help even with small datasets
        min_wheel_colors = 50       # Need reasonable number of wheel colors for KD-tree to be effective
        expensive_threshold = 10000  # Total comparisons threshold where KD-tree becomes beneficial
        
        # Memory consideration: brute force uses O(N*M) memory for distance matrix
        memory_limit_comparisons = 1_000_000  # ~8MB for float64 distances
        
        use_kdtree = (KDTREE_AVAILABLE and 
                     num_image_colors >= min_colors_for_kdtree and
                     num_wheel_colors >= min_wheel_colors and
                     (total_comparisons >= expensive_threshold or 
                      total_comparisons >= memory_limit_comparisons))
        
        # Debug info about the decision
        decision_reasons = []
        if not KDTREE_AVAILABLE:
            decision_reasons.append("scikit-learn not available")
        elif num_image_colors < min_colors_for_kdtree:
            decision_reasons.append(f"too few image colors ({num_image_colors} < {min_colors_for_kdtree})")
        elif num_wheel_colors < min_wheel_colors:
            decision_reasons.append(f"too few wheel colors ({num_wheel_colors} < {min_wheel_colors})")
        elif total_comparisons < expensive_threshold and total_comparisons < memory_limit_comparisons:
            decision_reasons.append(f"dataset too small ({total_comparisons:,} comparisons)")
        elif use_kdtree:
            if total_comparisons >= memory_limit_comparisons:
                decision_reasons.append("memory-limited dataset")
            else:
                decision_reasons.append("computationally expensive dataset")
    
    # Debug: Show what method will be used with reasoning
    print(f"Nearest neighbor search: {len(image_colors):,} image colors vs {len(wheel_colors):,} wheel colors ({total_comparisons:,} total comparisons)")
    
    if use_kdtree:
        reason = decision_reasons[0] if decision_reasons else "optimal choice"
        print(f"Using KD-tree for nearest neighbor search ({reason})")
        return _find_nearest_with_kdtree(image_colors, image_hsv, wheel_colors, wheel_hsv)
    else:
        reason = decision_reasons[0] if decision_reasons else "small dataset"
        method_parts = []
        if not KDTREE_AVAILABLE:
            method_parts.append("KD-tree not available")
        else:
            method_parts.append(f"vectorized fallback")
        
        if NUMBA_AVAILABLE and len(image_colors) * len(wheel_colors) > 10000:
            method_parts.append("with JIT compilation")
        
        method = ", ".join(method_parts)
        print(f"Using {method} for nearest neighbor search ({reason})")
        return _find_nearest_vectorized_fallback(image_colors, image_hsv, wheel_colors, wheel_hsv)


def _find_nearest_with_kdtree(image_colors, image_hsv, wheel_colors, wheel_hsv):
    """
    Use KD-tree for efficient nearest neighbor search in HSV space with custom distance metric.
    Handles hue wraparound properly by using a weighted Euclidean distance in a transformed space.
    
    Args:
        image_colors (list): Original RGB color tuples
        image_hsv (np.ndarray): HSV values for image colors (N, 3)
        wheel_colors (list): Wheel RGB color tuples  
        wheel_hsv (np.ndarray): HSV values for wheel colors (M, 3)
        
    Returns:
        dict: Mapping {image_color: nearest_wheel_color}
    """
    # Transform HSV to handle hue wraparound and apply weights
    # Convert hue to cartesian coordinates and apply weights
    hue_weight = 3.0
    sat_weight = 1.0  
    val_weight = 0.5
    
    # Transform wheel HSV: hue -> (cos, sin), then apply weights
    wheel_h_rad = wheel_hsv[:, 0] * 2 * np.pi  # Convert hue to radians
    wheel_transformed = np.column_stack([
        hue_weight * np.cos(wheel_h_rad),  # Hue X component (weighted)
        hue_weight * np.sin(wheel_h_rad),  # Hue Y component (weighted)  
        sat_weight * wheel_hsv[:, 1],      # Saturation (weighted)
        val_weight * wheel_hsv[:, 2]       # Value (weighted)
    ])
    
    # Transform image HSV the same way
    image_h_rad = image_hsv[:, 0] * 2 * np.pi
    image_transformed = np.column_stack([
        hue_weight * np.cos(image_h_rad),
        hue_weight * np.sin(image_h_rad),
        sat_weight * image_hsv[:, 1],
        val_weight * image_hsv[:, 2]
    ])
    
    # Build KD-tree with transformed wheel colors
    kdtree = KDTree(wheel_transformed, metric='euclidean')
    
    # Find nearest neighbors for all image colors at once
    distances, indices = kdtree.query(image_transformed, k=1)
    
    # Create mapping dictionary
    color_mapping = {}
    for i, image_color in enumerate(image_colors):
        nearest_wheel_color = wheel_colors[indices[i, 0]]  # indices shape is (N, 1)
        color_mapping[image_color] = nearest_wheel_color
    
    return color_mapping


@jit(nopython=True, parallel=True, cache=True) if NUMBA_AVAILABLE else lambda f: f
def _calculate_hsv_distances_numba(image_hsv, wheel_hsv, hue_weight=3.0, sat_weight=1.0, val_weight=0.5):
    """
    JIT-compiled HSV distance calculation with parallel processing.
    Much faster than NumPy broadcasting for large datasets.
    
    Args:
        image_hsv: (N, 3) array of image HSV values
        wheel_hsv: (M, 3) array of wheel HSV values
        
    Returns:
        (N, M) array of distances
    """
    n_image = image_hsv.shape[0]
    n_wheel = wheel_hsv.shape[0]
    distances = np.zeros((n_image, n_wheel), dtype=np.float32)
    
    for i in prange(n_image):
        img_h, img_s, img_v = image_hsv[i, 0], image_hsv[i, 1], image_hsv[i, 2]
        
        for j in range(n_wheel):
            wheel_h, wheel_s, wheel_v = wheel_hsv[j, 0], wheel_hsv[j, 1], wheel_hsv[j, 2]
            
            # Calculate hue difference with wraparound
            hue_diff = abs(img_h - wheel_h)
            hue_diff = min(hue_diff, 1.0 - hue_diff)
            
            # Calculate other differences
            sat_diff = img_s - wheel_s
            val_diff = img_v - wheel_v
            
            # Weighted distance
            distances[i, j] = (hue_weight * hue_diff * hue_diff + 
                             sat_weight * sat_diff * sat_diff + 
                             val_weight * val_diff * val_diff)
    
    return distances


def _calculate_hsv_distances_gpu(image_hsv, wheel_hsv, hue_weight=3.0, sat_weight=1.0, val_weight=0.5):
    """
    GPU-accelerated HSV distance calculation using CuPy.
    Much faster for large distance matrices.
    """
    if not CUPY_AVAILABLE:
        return _calculate_hsv_distances_numpy(image_hsv, wheel_hsv, hue_weight, sat_weight, val_weight)
    
    try:
        # Transfer to GPU
        image_hsv_gpu = cp.asarray(image_hsv, dtype=cp.float32)
        wheel_hsv_gpu = cp.asarray(wheel_hsv, dtype=cp.float32)
        
        num_image = image_hsv_gpu.shape[0]
        num_wheel = wheel_hsv_gpu.shape[0]
        
        print(f"GPU distance calculation: {num_image:,} × {num_wheel:,} = {num_image * num_wheel:,} comparisons")
        
        # Expand dimensions for broadcasting
        img_h = image_hsv_gpu[:, 0:1]  # Shape: (num_image, 1)
        img_s = image_hsv_gpu[:, 1:2]
        img_v = image_hsv_gpu[:, 2:3]
        
        wheel_h = wheel_hsv_gpu[:, 0].reshape(1, -1)  # Shape: (1, num_wheel)
        wheel_s = wheel_hsv_gpu[:, 1].reshape(1, -1)
        wheel_v = wheel_hsv_gpu[:, 2].reshape(1, -1)
        
        # Hue difference (circular, 0-1 range) - vectorized
        h_diff = cp.abs(img_h - wheel_h)
        h_diff = cp.minimum(h_diff, 1.0 - h_diff)  # Wrap around for circular hue
        
        # Saturation and value differences - vectorized
        s_diff = img_s - wheel_s
        v_diff = img_v - wheel_v
        
        # Weighted squared Euclidean distance - vectorized
        distances_gpu = (hue_weight * h_diff * h_diff + 
                        sat_weight * s_diff * s_diff + 
                        val_weight * v_diff * v_diff)
        
        # Transfer back to CPU
        result = cp.asnumpy(distances_gpu)
        print(f"GPU distance calculation completed")
        return result
        
    except Exception as e:
        print(f"GPU distance calculation failed: {e}, falling back to CPU")
        return _calculate_hsv_distances_numpy(image_hsv, wheel_hsv, hue_weight, sat_weight, val_weight)


def _calculate_hsv_distances_numpy(image_hsv, wheel_hsv, hue_weight=3.0, sat_weight=1.0, val_weight=0.5):
    """
    NumPy fallback HSV distance calculation with broadcasting.
    """
    # Expand dimensions for broadcasting
    img_h = image_hsv[:, 0:1]  # Shape: (num_image, 1)
    img_s = image_hsv[:, 1:2]
    img_v = image_hsv[:, 2:3]
    
    wheel_h = wheel_hsv[:, 0].reshape(1, -1)  # Shape: (1, num_wheel)
    wheel_s = wheel_hsv[:, 1].reshape(1, -1)
    wheel_v = wheel_hsv[:, 2].reshape(1, -1)
    
    # Hue difference (circular, 0-1 range)
    h_diff = np.abs(img_h - wheel_h)
    h_diff = np.minimum(h_diff, 1.0 - h_diff)
    
    # Saturation and value differences
    s_diff = img_s - wheel_s
    v_diff = img_v - wheel_v
    
    # Weighted squared Euclidean distance
    distances = (hue_weight * h_diff * h_diff + 
                sat_weight * s_diff * s_diff + 
                val_weight * v_diff * v_diff)
    
    return distances


def _find_nearest_vectorized_fallback(image_colors, image_hsv, wheel_colors, wheel_hsv):
    """
    Fallback vectorized nearest neighbor search with chunked processing.
    Used when KD-tree is not available or dataset is small.
    Uses chunking to avoid memory issues with very large datasets.
    
    Args:
        image_colors (list): Original RGB color tuples
        image_hsv (np.ndarray): HSV values for image colors (N, 3)
        wheel_colors (list): Wheel RGB color tuples
        wheel_hsv (np.ndarray): HSV values for wheel colors (M, 3)
        
    Returns:
        dict: Mapping {image_color: nearest_wheel_color}
    """
    num_image_colors = len(image_colors)
    num_wheel_colors = len(wheel_colors)
    total_comparisons = num_image_colors * num_wheel_colors
    
    # For very large datasets, use chunked processing to avoid memory issues
    # Memory usage for full distance matrix: N*M*8 bytes (float64)
    max_chunk_comparisons = 50_000_000  # ~400MB memory limit
    chunk_size = min(max_chunk_comparisons // num_wheel_colors, num_image_colors)
    chunk_size = max(100, chunk_size)  # Minimum chunk size
    
    print(f"Processing {total_comparisons:,} total comparisons in chunks of {chunk_size:,} image colors")
    
    color_mapping = {}
    
    # Process in chunks to manage memory usage
    for start_idx in range(0, num_image_colors, chunk_size):
        end_idx = min(start_idx + chunk_size, num_image_colors)
        chunk_image_hsv = image_hsv[start_idx:end_idx]
        chunk_size_actual = end_idx - start_idx
        
        # Choose the best distance calculation method based on available hardware and data size
        chunk_comparisons = chunk_size_actual * num_wheel_colors
        
        if CUPY_AVAILABLE and chunk_comparisons > 50000:  # Use GPU for large computations
            distances = _calculate_hsv_distances_gpu(chunk_image_hsv, wheel_hsv)
        elif NUMBA_AVAILABLE and chunk_comparisons > 10000:  # Use Numba JIT for medium computations
            distances = _calculate_hsv_distances_numba(chunk_image_hsv, wheel_hsv)
        else:  # Use NumPy for small computations
            distances = _calculate_hsv_distances_numpy(chunk_image_hsv, wheel_hsv)
        
        # Find nearest wheel color for each image color in this chunk
        nearest_indices = np.argmin(distances, axis=1)  # Shape: (chunk_size,)
        
        # Create mapping for this chunk
        for i, nearest_idx in enumerate(nearest_indices):
            image_color = image_colors[start_idx + i]
            nearest_wheel_color = wheel_colors[nearest_idx]
            color_mapping[image_color] = nearest_wheel_color
    
    return color_mapping


def create_color_wheel(color_percentages, wheel_size=800, inner_radius_ratio=0.1, quantize_level=8, force_kdtree=None, use_parallel=None):
    """
    Create a full color wheel where opacity represents color frequency.
    Areas with frequent colors are more opaque.
    Areas with rare/missing colors are more transparent.
    
    Uses a precomputed wheel template for maximum performance.
    
    Args:
        color_percentages (dict): Color frequency percentages {(r,g,b): percentage}
        wheel_size (int): Size of the output wheel image
        inner_radius_ratio (float): Ratio of inner radius to outer radius
        quantize_level (int): Color quantization level used in analysis
        force_kdtree (bool): Force KD-tree usage (True), disable it (False), or auto-detect (None)
        use_parallel (bool): Use parallel processing where applicable (None=auto-detect)
        
    Returns:
        tuple: (numpy.ndarray, dict, list) - RGBA image of the color wheel, normalized percentages, and opacity values
    """
    wheel_start = time.time()
    
    # Load or create the wheel template (cached on disk) WITH HSV cache
    template_start = time.time()
    wheel_rgb, color_to_pixels_map, wheel_hsv_cache = load_or_create_wheel_template(wheel_size, inner_radius_ratio, quantize_level)
    template_time = time.time() - template_start
    
    # Create output image (RGBA) - start with RGB template
    wheel = np.zeros((wheel_size, wheel_size, 4), dtype=np.uint8)
    wheel[:, :, :3] = wheel_rgb  # Copy RGB channels
    
    # Note: color_percentages already contains actual percentages (frequencies/total_pixels)
    # so they should sum to ~1.0. We'll use them directly for accumulation.
    image_colors = list(color_percentages.keys())
    
    # Use original percentages directly - they're already proper percentages
    normalized_percentages = color_percentages.copy()
    
    # Pre-allocate opacity values list with estimated size
    estimated_wheel_colors = min(len(color_to_pixels_map), len(image_colors) * 2)  # Rough estimate
    opacity_values = []  # Python lists grow dynamically, no need to pre-reserve
    
    # NEW APPROACH: Map each input image color to the nearest color in the wheel template
    # VECTORIZED: Do all color mappings at once for massive speed improvement
    # CACHED: Use pre-computed HSV values for wheel colors
    # SPATIAL INDEXING: Use KD-tree for large datasets
    # PARALLEL: Use JIT compilation and multiprocessing for performance
    wheel_color_frequencies = {}
    
    # Get all image colors and do vectorized nearest-neighbor lookup using cached HSV
    image_colors = list(normalized_percentages.keys())
    prefilter_time = 0  # Initialize timing variables
    nearest_neighbor_time = 0
    
    if image_colors:
        # PRE-FILTERING: Group similar colors to reduce search complexity
        # Only use pre-filtering if we have enough colors to make it worthwhile
        if len(image_colors) > 1000:  # Only pre-filter for large color sets
            prefilter_start = time.time()
            # More aggressive threshold for better reduction
            threshold = max(4, quantize_level * 2)  # Larger threshold for more reduction
            filtered_percentages, original_to_filtered_mapping = prefilter_and_deduplicate_colors(
                normalized_percentages, 
                similarity_threshold=threshold
            )
            filtered_colors = list(filtered_percentages.keys())
            prefilter_time = time.time() - prefilter_start
            
            # Only use filtered results if we got significant reduction (>20%)
            reduction_ratio = len(filtered_colors) / len(image_colors)
            if reduction_ratio < 0.8:  # If we reduced by more than 20%
                # Do nearest-neighbor search on filtered (reduced) set
                nearest_neighbor_start = time.time()
                filtered_color_mapping = find_nearest_wheel_colors_vectorized(filtered_colors, color_to_pixels_map, wheel_hsv_cache, force_kdtree, use_parallel)
                nearest_neighbor_time = time.time() - nearest_neighbor_start
                
                # Expand the filtered mapping back to original colors
                color_mapping = {}
                for original_color in image_colors:
                    filtered_color = original_to_filtered_mapping[original_color]
                    nearest_wheel_color = filtered_color_mapping[filtered_color]
                    color_mapping[original_color] = nearest_wheel_color
                
                print(f"Pre-filtering saved {len(image_colors) - len(filtered_colors):,} nearest-neighbor searches ({prefilter_time*1000:.1f}ms)")
            else:
                # Pre-filtering didn't help enough, use original approach
                print(f"Pre-filtering reduction too small ({reduction_ratio:.1%}), using direct search")
                prefilter_time = 0
                nearest_neighbor_start = time.time()
                color_mapping = find_nearest_wheel_colors_vectorized(image_colors, color_to_pixels_map, wheel_hsv_cache, force_kdtree, use_parallel)
                nearest_neighbor_time = time.time() - nearest_neighbor_start
        else:
            # Skip pre-filtering for small color sets
            print(f"Skipping pre-filtering for small color set ({len(image_colors):,} colors)")
            prefilter_time = 0
            nearest_neighbor_start = time.time()
            color_mapping = find_nearest_wheel_colors_vectorized(image_colors, color_to_pixels_map, wheel_hsv_cache, force_kdtree, use_parallel)
            nearest_neighbor_time = time.time() - nearest_neighbor_start
        
        # Accumulate frequencies for wheel colors using original percentages
        for image_color, percentage in normalized_percentages.items():
            nearest_wheel_color = color_mapping[image_color]
            
            if nearest_wheel_color in wheel_color_frequencies:
                wheel_color_frequencies[nearest_wheel_color] += percentage
            else:
                wheel_color_frequencies[nearest_wheel_color] = percentage
    
    # Normalize accumulated frequencies: most frequent wheel color = 1.0, least frequent = 0.0
    # This ensures proper opacity mapping regardless of how many input colors map to wheel colors
    max_accumulated_freq = max(wheel_color_frequencies.values()) if wheel_color_frequencies else 1.0
    min_accumulated_freq = min(wheel_color_frequencies.values()) if wheel_color_frequencies else 0.0
    freq_range = max_accumulated_freq - min_accumulated_freq
    
    if freq_range > 0:
        # Normalize to [0, 1] range: (value - min) / (max - min)
        for wheel_color in wheel_color_frequencies:
            wheel_color_frequencies[wheel_color] = (wheel_color_frequencies[wheel_color] - min_accumulated_freq) / freq_range
        print(f"Frequency normalization: {min_accumulated_freq:.6f} → 0.0, {max_accumulated_freq:.6f} → 1.0")
    else:
        # All frequencies are the same, set them to 1.0
        for wheel_color in wheel_color_frequencies:
            wheel_color_frequencies[wheel_color] = 1.0
        print(f"All wheel colors have equal frequency: {max_accumulated_freq:.6f}")
    
    # Debug: show frequency distribution
    if wheel_color_frequencies:
        freq_values = list(wheel_color_frequencies.values())
        print(f"Final frequency range: {min(freq_values):.6f} to {max(freq_values):.6f} (wheel colors: {len(wheel_color_frequencies)})")
    
    # Now apply frequencies to the wheel using the mapped colors
    opacity_mapping_start = time.time()
    for quantized_color, pixel_coords in color_to_pixels_map.items():
        # Get the accumulated frequency for this wheel color
        normalized_frequency = wheel_color_frequencies.get(quantized_color, 0)
        
        # Calculate opacity based on frequency
        if normalized_frequency > 0:
            # Map normalized frequency to opacity range (128-255)
            # Using linear mapping for even distribution
            
            curved_frequency = normalized_frequency ** 0.25 # curve mapping for even spread
            opacity = int(64 + (255 - 64) * curved_frequency)  # Map to 64-255 range
            opacity_values.append(opacity)  # Collect for histogram
        else:
            opacity = 32  # Low opacity for colors not in image
        


        # Set opacity for all pixels of this color at once (vectorized)
        if len(pixel_coords) > 0:
            y_coords = pixel_coords[:, 0]
            x_coords = pixel_coords[:, 1]
            wheel[y_coords, x_coords, 3] = opacity
    
    opacity_mapping_time = time.time() - opacity_mapping_start
    

    wheel_time = time.time() - wheel_start
    if image_colors:
        print(f"Wheel generation completed in {format_time(wheel_time)} (template: {format_time(template_time)}, pre-filtering: {format_time(prefilter_time)}, nearest-neighbor: {format_time(nearest_neighbor_time)}, opacity mapping: {format_time(opacity_mapping_time)})")
    else:
        print(f"Wheel generation completed in {format_time(wheel_time)} (template: {format_time(template_time)}, opacity mapping: {format_time(opacity_mapping_time)})")
            
    return wheel, normalized_percentages, opacity_values


def add_wheel_gradient(wheel_size=800, inner_radius_ratio=0.1, quantize_level=8):
    """
    Create a reference color wheel with full saturation gradient.
    Uses the same precomputed template for consistency and speed.
    
    Returns:
        numpy.ndarray: RGBA image of a standard color wheel
    """
    # Load or create the wheel template (cached on disk) - ignore HSV cache for reference wheel
    wheel_rgb, _, _ = load_or_create_wheel_template(wheel_size, inner_radius_ratio, quantize_level)
    
    # Create output image (RGBA) - start with RGB template
    wheel = np.zeros((wheel_size, wheel_size, 4), dtype=np.uint8)
    wheel[:, :, :3] = wheel_rgb  # Copy RGB channels
    
    # Set full opacity for all valid pixels (where RGB is not zero)
    valid_pixels = np.any(wheel_rgb > 0, axis=2)
    wheel[valid_pixels, 3] = 255  # Full opacity
    
    return wheel


def create_opacity_histogram(opacity_values, output_path):
    """
    Create and save a histogram showing the distribution of opacity values.
    
    Args:
        opacity_values (list): List of opacity values from the wheel generation
        output_path (str): Path to save the histogram image
    """
    if not opacity_values:
        print("No color data available for histogram")
        return
    
    # Create histogram using the actual opacity values from wheel generation
    plt.figure(figsize=(10, 6))
    plt.hist(opacity_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Opacity Value (0-255)')
    plt.ylabel('Frequency Count')
    plt.title('Distribution of Color Opacity Values\n(From Actual Wheel Generation)')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_opacity = np.mean(opacity_values)
    median_opacity = np.median(opacity_values)
    max_opacity = np.max(opacity_values)
    min_opacity = np.min(opacity_values)
    
    stats_text = f'Stats:\nMean: {mean_opacity:.1f}\nMedian: {median_opacity:.1f}\nMin: {min_opacity}\nMax: {max_opacity}\nColors: {len(opacity_values)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Opacity histogram saved to: {output_path}")


def create_color_spectrum_histogram(color_percentages, output_path, width=1200, height=400):
    """
    Create a complete color spectrum histogram showing all colors with their frequencies.
    Shows the full spectrum even for colors not in the image (with zero frequency).
    
    Args:
        color_percentages (dict): Color frequency percentages {(r,g,b): percentage}
        output_path (str): Path to save the histogram image
        width (int): Width of the output image
        height (int): Height of the output image
    """
    # Create a complete color spectrum (like HSV color wheel flattened)
    spectrum_width = 360  # One bar per degree of hue
    
    # Pre-allocate arrays for spectrum data
    spectrum_colors = np.zeros((spectrum_width, 3), dtype=np.float32)
    frequencies = np.zeros(spectrum_width, dtype=np.float32)
    
    # VECTORIZED: Generate full spectrum colors (HSV with full saturation and value)
    hue_degrees = np.arange(spectrum_width, dtype=np.float32)
    spectrum_hsv = np.column_stack([
        hue_degrees / 360.0,  # H: 0-1
        np.ones(spectrum_width, dtype=np.float32),  # S: 1.0 (full saturation)
        np.ones(spectrum_width, dtype=np.float32)   # V: 1.0 (full value)
    ])
    
    # Convert spectrum HSV to RGB using vectorized conversion
    def hsv_to_rgb_vectorized_simple(hsv_array):
        h, s, v = hsv_array[:, 0], hsv_array[:, 1], hsv_array[:, 2]
        
        c = v * s
        x = c * (1 - np.abs(((h * 6) % 2) - 1))
        m = v - c
        
        # Pre-allocate RGB arrays
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        # Determine RGB based on hue sector
        h_sector = (h * 6).astype(int) % 6
        
        mask0 = h_sector == 0; r[mask0] = c[mask0]; g[mask0] = x[mask0]
        mask1 = h_sector == 1; r[mask1] = x[mask1]; g[mask1] = c[mask1]
        mask2 = h_sector == 2; g[mask2] = c[mask2]; b[mask2] = x[mask2]
        mask3 = h_sector == 3; g[mask3] = x[mask3]; b[mask3] = c[mask3]
        mask4 = h_sector == 4; r[mask4] = x[mask4]; b[mask4] = c[mask4]
        mask5 = h_sector == 5; r[mask5] = c[mask5]; b[mask5] = x[mask5]
        
        return np.column_stack([r + m, g + m, b + m])
    
    spectrum_colors = hsv_to_rgb_vectorized_simple(spectrum_hsv)
    
    # VECTORIZED: Convert all image colors to HSV at once if there are any
    if color_percentages:
        image_colors = list(color_percentages.keys())
        image_colors_array = np.array(image_colors, dtype=np.float32) / 255.0
        image_hsv = rgb_to_hsv_vectorized(image_colors_array)
        image_hues = image_hsv[:, 0] * 360  # Convert to degrees
        image_sats = image_hsv[:, 1]
        frequencies_array = np.array(list(color_percentages.values()), dtype=np.float32)
        
        # Find frequencies for each spectrum hue using vectorized operations
        for i, spectrum_hue in enumerate(hue_degrees):
            # Calculate hue differences with wraparound
            hue_diffs = np.abs(image_hues - spectrum_hue)
            hue_diffs = np.minimum(hue_diffs, 360 - hue_diffs)
            
            # Find colors within threshold
            hue_mask = (hue_diffs <= 2) & (image_sats > 0.3)  # Within 2 degrees and not too gray
            
            if np.any(hue_mask):
                frequencies[i] = np.max(frequencies_array[hue_mask])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Create bars with spectrum colors
    x_positions = np.arange(len(spectrum_colors))
    bars = ax.bar(x_positions, frequencies, width=1.0, edgecolor='none')
    
    # Set bar colors to match the spectrum colors
    for i, (bar, rgb_color) in enumerate(zip(bars, spectrum_colors)):
        bar.set_facecolor(rgb_color)  # Already in 0-1 range
    
    # Customize the plot
    ax.set_xlim(-0.5, len(spectrum_colors) - 0.5)
    ax.set_xlabel('Hue (0° to 360°)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Complete Color Spectrum Histogram\n(Full Spectrum with Image Color Frequencies)', fontsize=14)
    
    # Add hue labels at key points
    hue_labels = [0, 60, 120, 180, 240, 300, 360]
    hue_positions = [h * spectrum_width / 360 for h in hue_labels]
    ax.set_xticks(hue_positions)
    ax.set_xticklabels([f'{h}°' for h in hue_labels])
    
    # Add grid for easier reading of frequencies
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add statistics
    non_zero_freqs = frequencies[frequencies > 0]
    total_colors_in_image = len(non_zero_freqs)
    max_freq = np.max(frequencies) if len(frequencies) > 0 else 0
    avg_freq = np.mean(non_zero_freqs) if len(non_zero_freqs) > 0 else 0
    
    stats_text = f'Colors in Image: {total_colors_in_image}/{spectrum_width}\nMax Frequency: {max_freq:.4f}\nAvg Frequency: {avg_freq:.4f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Color spectrum histogram saved to: {output_path}")


def create_circular_color_spectrum(color_percentages, output_path, size=800):
    """
    Create a circular color spectrum histogram where frequencies appear as spikes radiating outward.
    Like a color wheel with frequency spikes pointing outward from the center.
    
    Args:
        color_percentages (dict): Color frequency percentages {(r,g,b): percentage}
        output_path (str): Path to save the circular histogram image
        size (int): Size of the output image (square)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(size/100, size/100))
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Remove axes and ticks for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Generate spectrum data
    num_segments = 360  # One segment per degree
    inner_radius = 0.3  # Inner circle radius
    max_spike_length = 0.6  # Maximum spike length
    
    # VECTORIZED: Convert all image colors to HSV at once
    if color_percentages:
        image_colors = list(color_percentages.keys())
        image_colors_array = np.array(image_colors, dtype=np.float32) / 255.0
        image_hsv = rgb_to_hsv_vectorized(image_colors_array)  # Shape: (N, 3)
        image_hues = image_hsv[:, 0] * 360  # Convert to degrees
        image_sats = image_hsv[:, 1]
        frequencies_array = np.array(list(color_percentages.values()), dtype=np.float32)
    else:
        image_hues = np.array([])
        image_sats = np.array([])
        frequencies_array = np.array([])
    
    # VECTORIZED: Generate spectrum colors
    hue_degrees = np.arange(num_segments, dtype=np.float32)
    spectrum_hsv = np.column_stack([
        hue_degrees / 360.0,  # H: 0-1
        np.ones(num_segments, dtype=np.float32),  # S: 1.0 (full saturation)
        np.ones(num_segments, dtype=np.float32)   # V: 1.0 (full value)
    ])
    
    # Convert spectrum HSV to RGB using our vectorized function
    # Note: rgb_to_hsv_vectorized works in reverse too, but we need HSV->RGB
    # Let's use a vectorized HSV to RGB conversion
    def hsv_to_rgb_vectorized(hsv_array):
        h, s, v = hsv_array[:, 0], hsv_array[:, 1], hsv_array[:, 2]
        
        c = v * s
        x = c * (1 - np.abs(((h * 6) % 2) - 1))
        m = v - c
        
        # Initialize RGB arrays
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        # Determine RGB based on hue sector
        h_sector = (h * 6).astype(int) % 6
        
        mask0 = h_sector == 0
        r[mask0] = c[mask0]
        g[mask0] = x[mask0]
        
        mask1 = h_sector == 1
        r[mask1] = x[mask1]
        g[mask1] = c[mask1]
        
        mask2 = h_sector == 2
        g[mask2] = c[mask2]
        b[mask2] = x[mask2]
        
        mask3 = h_sector == 3
        g[mask3] = x[mask3]
        b[mask3] = c[mask3]
        
        mask4 = h_sector == 4
        r[mask4] = x[mask4]
        b[mask4] = c[mask4]
        
        mask5 = h_sector == 5
        r[mask5] = c[mask5]
        b[mask5] = x[mask5]
        
        return np.column_stack([r + m, g + m, b + m])
    
    spectrum_rgb = hsv_to_rgb_vectorized(spectrum_hsv)  # Shape: (360, 3)
    
    # VECTORIZED: Calculate frequencies for each spectrum hue
    frequencies = np.zeros(num_segments, dtype=np.float32)
    
    if len(image_hues) > 0:
        # For each spectrum hue, find matching image colors
        for i, spectrum_hue in enumerate(hue_degrees):
            # Calculate hue differences with wraparound
            hue_diffs = np.abs(image_hues - spectrum_hue)
            hue_diffs = np.minimum(hue_diffs, 360 - hue_diffs)
            
            # Find colors within threshold
            hue_mask = (hue_diffs <= 2) & (image_sats > 0.1)  # Within 2 degrees and not too desaturated
            
            if np.any(hue_mask):
                frequencies[i] = np.max(frequencies_array[hue_mask])
    
    # Normalize frequencies for spike lengths
    max_freq = np.max(frequencies) if np.any(frequencies > 0) else 1
    normalized_freqs = frequencies / max_freq
    
    # VECTORIZED: Calculate all angles and positions at once
    angles_rad = hue_degrees * np.pi / 180  # Convert to radians
    cos_angles = np.cos(angles_rad)
    sin_angles = np.sin(angles_rad)
    
    # Calculate all spike positions
    inner_x = inner_radius * cos_angles
    inner_y = inner_radius * sin_angles
    spike_lengths = normalized_freqs * max_spike_length
    outer_x = (inner_radius + spike_lengths) * cos_angles
    outer_y = (inner_radius + spike_lengths) * sin_angles
    
    # Draw spikes (only for non-zero frequencies to avoid clutter)
    spike_mask = spike_lengths > 0
    if np.any(spike_mask):
        for i in np.where(spike_mask)[0]:
            ax.plot([inner_x[i], outer_x[i]], [inner_y[i], outer_y[i]], 
                   color=spectrum_rgb[i], linewidth=2, alpha=0.8)
    
    # VECTORIZED: Draw inner circle segments
    segment_width = 2 * np.pi / num_segments
    for i in range(num_segments):
        angle_start = angles_rad[i] - segment_width/2
        angle_end = angles_rad[i] + segment_width/2
        
        # Create a small arc for the inner circle
        arc_angles = np.linspace(angle_start, angle_end, 5)
        arc_x = inner_radius * np.cos(arc_angles)
        arc_y = inner_radius * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color=spectrum_rgb[i], linewidth=3)
    
    # Add title and statistics
    non_zero_freqs = frequencies[frequencies > 0]
    total_colors = len(non_zero_freqs)
    avg_freq = np.mean(non_zero_freqs) if len(non_zero_freqs) > 0 else 0
    
    ax.set_title(f'Circular Color Spectrum\n{total_colors} colors detected, Max frequency: {max_freq:.4f}', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Circular color spectrum saved to: {output_path}")


def main():
    """Main function to run the color wheel generator."""
    parser = argparse.ArgumentParser(
        description="Generate a color wheel where opacity represents color frequency in an image"
    )
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_wheel", help="Path for the output color wheel image")
    parser.add_argument("--size", type=int, default=800, help="Size of the color wheel (default: 800)")
    parser.add_argument("--sample-factor", type=int, default=1, 
                       help="Factor to downsample input image for faster processing (default: 1)")
    parser.add_argument("--quantize", type=int, default=2, 
                       help="Color quantization level: 1=no quantization (most precise), higher=more grouping (default: 2)")
    parser.add_argument("--show-reference", action="store_true", 
                       help="Also save a reference color wheel for comparison")
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                       help="Output format: png (supports transparency) or jpg (black background, default: png)")
    parser.add_argument("--histogram", action="store_true",
                       help="Also generate a histogram showing the distribution of opacity values")
    parser.add_argument("--color-spectrum", action="store_true",
                       help="Also generate a color spectrum histogram showing colors and their frequencies")
    parser.add_argument("--circular-spectrum", action="store_true",
                       help="Also generate a circular color spectrum with frequency spikes radiating outward")
    parser.add_argument("--force-kdtree", action="store_true",
                       help="Force KD-tree usage for nearest neighbor search (requires scikit-learn)")
    parser.add_argument("--no-kdtree", action="store_true",
                       help="Disable KD-tree and use vectorized fallback method")
    parser.add_argument("--parallel", action="store_true",
                       help="Force parallel processing for color analysis and computations")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing and use single-threaded methods")
    parser.add_argument("--gpu", action="store_true",
                       help="Force GPU acceleration for computations (requires CuPy)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration and use CPU-only processing")
    parser.add_argument("--color-space", choices=["sRGB", "Adobe RGB", "ProPhoto RGB"], default="sRGB",
                       help="Color space to assume for input image (default: sRGB)")
    
    args = parser.parse_args()
    
    # Handle KD-tree options
    force_kdtree = None
    if args.force_kdtree and args.no_kdtree:
        print("Error: Cannot specify both --force-kdtree and --no-kdtree")
        return 1
    elif args.force_kdtree:
        force_kdtree = True
    elif args.no_kdtree:
        force_kdtree = False
    
    # Handle parallel processing options
    use_parallel = None
    if args.parallel and args.no_parallel:
        print("Error: Cannot specify both --parallel and --no-parallel")
        return 1
    elif args.parallel:
        use_parallel = True
    elif args.no_parallel:
        use_parallel = False
    
    # Handle GPU options
    global CUPY_AVAILABLE
    if args.gpu and args.no_gpu:
        print("Error: Cannot specify both --gpu and --no-gpu")
        return 1
    elif args.gpu:
        if not CUPY_AVAILABLE:
            print("Error: GPU acceleration requested but CuPy is not available")
            print("Install CuPy: pip install cupy-cuda11x or cupy-cuda12x")
            return 1
        print("GPU acceleration forced ON")
    elif args.no_gpu:
        CUPY_AVAILABLE = False  # Temporarily disable GPU for this run
        print("GPU acceleration disabled")
    
    # Print available optimizations
    optimizations = []
    if CUPY_AVAILABLE:
        optimizations.append("GPU acceleration (CuPy)")
    if NUMBA_AVAILABLE:
        optimizations.append("Numba JIT compilation")
    if KDTREE_AVAILABLE:
        optimizations.append("KD-tree spatial indexing")
    if mp.cpu_count() > 1:
        optimizations.append(f"Multiprocessing ({mp.cpu_count()} cores)")
    
    if optimizations:
        print(f"Available optimizations: {', '.join(optimizations)}")
    
    try:
        total_start = time.time()
        
        print(f"Loading and analyzing image: {args.input_image}")
        color_percentages = load_and_analyze_image(args.input_image, args.sample_factor, args.quantize, use_parallel, args.color_space)
        print(f"Found {len(color_percentages)} unique colors (quantization level: {args.quantize})")
        
        print("Generating color wheel...")
        wheel, normalized_percentages, opacity_values = create_color_wheel(color_percentages, args.size, quantize_level=args.quantize, force_kdtree=force_kdtree, use_parallel=use_parallel)
        
        # Auto-detect format from file extension if not explicitly specified
        output_path = args.output_wheel
        if args.format == "png" and (output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg')):
            print("Auto-detected JPG format from file extension")
            format_to_use = "jpg"
        elif args.format == "jpg" and output_path.lower().endswith('.png'):
            print("Auto-detected PNG format from file extension")
            format_to_use = "png"
        else:
            format_to_use = args.format
        
        # Handle output format
        if format_to_use == "jpg":
            # JPG doesn't support transparency, so blend with black background
            if not output_path.lower().endswith('.jpg') and not output_path.lower().endswith('.jpeg'):
                output_path = output_path.rsplit('.', 1)[0] + '.jpg'
                print(f"Changed output format to JPG: {output_path}")
            
            # Convert RGBA to RGB by blending with black background
            rgb_wheel = np.zeros((wheel.shape[0], wheel.shape[1], 3), dtype=np.uint8)
            alpha = wheel[:, :, 3] / 255.0  # Normalize alpha to 0-1
            
            for i in range(3):  # RGB channels
                rgb_wheel[:, :, i] = (wheel[:, :, i] * alpha + 0 * (1 - alpha)).astype(np.uint8)  # Black background (0)
            
            # Convert RGB to BGR for OpenCV
            wheel_bgr = rgb_wheel[:, :, [2, 1, 0]]  # RGB -> BGR
            success = cv2.imwrite(output_path, wheel_bgr)
            
        else:  # PNG format
            # Ensure output filename has .png extension for RGBA support
            if not output_path.lower().endswith('.png'):
                output_path = output_path.rsplit('.', 1)[0] + '.png'
                print(f"Changed output format to PNG for transparency support: {output_path}")
            
            # Ensure wheel array is proper uint8 format
            wheel = wheel.astype(np.uint8)
            
            # Save directly as RGBA PNG (no conversion needed)
            # OpenCV expects BGR or BGRA, so convert RGBA to BGRA
            wheel_bgra = wheel[:, :, [2, 1, 0, 3]]  # Swap R and B channels: RGBA -> BGRA
            success = cv2.imwrite(output_path, wheel_bgra)
        
        if not success:
            print(f"Error: Failed to save image to {output_path}")
            return 1
        
        print(f"Color wheel saved to: {output_path}")
        
        # Optionally save reference wheel
        if args.show_reference:
            reference_wheel = add_wheel_gradient(args.size, quantize_level=args.quantize)
            
            if format_to_use == "jpg":
                # Convert reference wheel to RGB with black background
                rgb_ref = np.zeros((reference_wheel.shape[0], reference_wheel.shape[1], 3), dtype=np.uint8)
                alpha = reference_wheel[:, :, 3] / 255.0
                
                for i in range(3):
                    rgb_ref[:, :, i] = (reference_wheel[:, :, i] * alpha + 0 * (1 - alpha)).astype(np.uint8)  # Black background
                
                reference_wheel_bgr = rgb_ref[:, :, [2, 1, 0]]
                reference_path = output_path.replace('.jpg', '_reference.jpg').replace('.jpeg', '_reference.jpg')
                cv2.imwrite(reference_path, reference_wheel_bgr)
            else:
                reference_wheel_bgra = reference_wheel[:, :, [2, 1, 0, 3]]
                reference_path = output_path.replace('.png', '_reference.png')
                cv2.imwrite(reference_path, reference_wheel_bgra)
                
            print(f"Reference wheel saved to: {reference_path}")
        
        # Generate histogram if requested
        if args.histogram:
            print("Generating opacity histogram...")
            histogram_path = output_path.rsplit('.', 1)[0] + '_histogram.png'
            create_opacity_histogram(opacity_values, histogram_path)
            
        # Generate color spectrum histogram if requested
        if args.color_spectrum:
            print("Generating color spectrum histogram...")
            spectrum_path = output_path.rsplit('.', 1)[0] + '_color_spectrum.png'
            create_color_spectrum_histogram(color_percentages, spectrum_path)
            
        # Generate circular color spectrum if requested
        if args.circular_spectrum:
            print("Generating circular color spectrum...")
            circular_path = output_path.rsplit('.', 1)[0] + '_circular_spectrum.png'
            create_circular_color_spectrum(color_percentages, circular_path)
            
        total_time = time.time() - total_start
        print(f"\nTotal processing completed in {format_time(total_time)}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
