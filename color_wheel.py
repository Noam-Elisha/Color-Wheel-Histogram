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


def create_wheel_template(wheel_size=800, inner_radius_ratio=0.1, quantize_level=8):
    """
    Create a precomputed wheel template with full-resolution RGB values and quantized color lookup.
    This generates a high-quality wheel with smooth gradients while maintaining compatibility 
    with quantized input image analysis.
    
    Returns:
        tuple: (wheel_rgb, color_to_pixels_map)
            - wheel_rgb: (H, W, 3) array with full-resolution RGB values
            - color_to_pixels_map: dict mapping quantized (r,g,b) -> list of (y,x) coordinates
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
    for i, (y, x) in enumerate(zip(valid_indices[0], valid_indices[1])):
        quantized_color = (wheel_r_quantized[i], wheel_g_quantized[i], wheel_b_quantized[i])
        if quantized_color not in color_to_pixels_map:
            color_to_pixels_map[quantized_color] = []
        color_to_pixels_map[quantized_color].append((y, x))
    
    # Convert lists to numpy arrays for faster indexing
    for color in color_to_pixels_map:
        color_to_pixels_map[color] = np.array(color_to_pixels_map[color])
    
    return wheel_rgb, color_to_pixels_map


def get_wheel_template_path(wheel_size, inner_radius_ratio, quantize_level):
    """Get the path for the wheel template file in a dedicated templates folder."""
    # Create templates directory if it doesn't exist
    templates_dir = "wheel_templates"
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    filename = f"wheel_template_fullres_{wheel_size}_{inner_radius_ratio:.3f}_q{quantize_level}.pkl"
    return os.path.join(templates_dir, filename)


def load_or_create_wheel_template(wheel_size=800, inner_radius_ratio=0.1, quantize_level=8):
    """
    Load existing wheel template or create a new one if it doesn't exist.
    
    Returns:
        tuple: (wheel_rgb, color_to_pixels_map)
    """
    template_path = get_wheel_template_path(wheel_size, inner_radius_ratio, quantize_level)
    
    if os.path.exists(template_path):
        print(f"Loading precomputed wheel template: {template_path}")
        with open(template_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Creating new wheel template: {template_path}")
        template_data = create_wheel_template(wheel_size, inner_radius_ratio, quantize_level)
        
        # Save the template for future use
        with open(template_path, 'wb') as f:
            pickle.dump(template_data, f)
        
        print(f"Wheel template saved to: {template_path}")
        return template_data


def load_and_analyze_image(image_path, sample_factor=4, quantize_level=8):
    """
    Load an image and analyze color frequencies as percentages.
    
    Args:
        image_path (str): Path to the input image
        sample_factor (int): Factor to downsample image for faster processing
        quantize_level (int): Color quantization level (1=no quantization, higher=more grouping)
        
    Returns:
        dict: Color frequency percentages {(r,g,b): percentage}
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Downsample for faster processing
    if sample_factor > 1:
        height, width = image.shape[:2]
        new_height, new_width = height // sample_factor, width // sample_factor
        image = cv2.resize(image, (new_width, new_height))
    
    # Count color frequencies using vectorized operations
    pixels = image.reshape(-1, 3)
    total_pixels = len(pixels)
    
    # Quantize colors to reduce noise (group similar colors) - vectorized
    if quantize_level > 1:
        quantized_pixels = (pixels // quantize_level) * quantize_level
    else:
        quantized_pixels = pixels  # No quantization
    
    # Use NumPy's unique function with return_counts for efficient counting
    # Convert RGB tuples to a single integer for efficient unique counting (using int64 to avoid overflow)
    rgb_as_int = quantized_pixels[:, 0].astype(np.int64) * 65536 + quantized_pixels[:, 1].astype(np.int64) * 256 + quantized_pixels[:, 2].astype(np.int64)
    unique_colors, counts = np.unique(rgb_as_int, return_counts=True)
    
    # Convert back to RGB tuples and create percentage dictionary
    color_percentages = {}
    for color_int, count in zip(unique_colors, counts):
        # Convert back to RGB tuple
        r = int((color_int // 65536) % 256)
        g = int((color_int // 256) % 256)
        b = int(color_int % 256)
        color_percentages[(r, g, b)] = count / total_pixels
    
    return color_percentages


def rgb_to_hsv_normalized(r, g, b):
    """
    Convert RGB values (0-255) to HSV values.
    
    Returns:
        tuple: (h, s, v) where h is in [0, 360), s and v are in [0, 1]
    """
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return h * 360, s, v


def find_nearest_wheel_colors_vectorized(image_colors, color_to_pixels_map):
    """
    Find the nearest wheel colors for multiple image colors using vectorized HSV-based matching.
    This properly maps colors based on hue (angle) and saturation (radius) like a real color wheel.
    
    Args:
        image_colors (list): List of RGB color tuples from the image [(r,g,b), ...]
        color_to_pixels_map (dict): Mapping of wheel colors to pixel coordinates
        
    Returns:
        dict: Mapping {image_color: nearest_wheel_color}
    """
    if not image_colors:
        return {}
    
    # Convert image colors to HSV using vectorized operations
    image_colors_array = np.array(image_colors, dtype=np.float32) / 255.0  # Shape: (N, 3)
    
    # Vectorized RGB to HSV conversion
    image_hsv = np.zeros_like(image_colors_array)  # Shape: (N, 3)
    for i in range(len(image_colors_array)):
        image_hsv[i] = colorsys.rgb_to_hsv(*image_colors_array[i])
    
    # Convert wheel colors to HSV
    wheel_colors = list(color_to_pixels_map.keys())
    wheel_colors_array = np.array(wheel_colors, dtype=np.float32) / 255.0  # Shape: (M, 3)
    
    wheel_hsv = np.zeros_like(wheel_colors_array)  # Shape: (M, 3)
    for i in range(len(wheel_colors_array)):
        wheel_hsv[i] = colorsys.rgb_to_hsv(*wheel_colors_array[i])
    
    # Vectorized HSV distance calculation with broadcasting
    # image_hsv shape: (N, 3), wheel_hsv shape: (M, 3)
    # After broadcasting: (N, 1, 3) - (1, M, 3) = (N, M, 3)
    
    img_h = image_hsv[:, None, 0]  # Shape: (N, 1)
    img_s = image_hsv[:, None, 1]  # Shape: (N, 1) 
    img_v = image_hsv[:, None, 2]  # Shape: (N, 1)
    
    wheel_h = wheel_hsv[None, :, 0]  # Shape: (1, M)
    wheel_s = wheel_hsv[None, :, 1]  # Shape: (1, M)
    wheel_v = wheel_hsv[None, :, 2]  # Shape: (1, M)
    
    # Calculate hue differences with proper wraparound
    hue_diff = np.abs(img_h - wheel_h)  # Shape: (N, M)
    hue_diff = np.minimum(hue_diff, 1 - hue_diff)  # Handle wraparound
    
    # Calculate saturation and value differences
    sat_diff = img_s - wheel_s  # Shape: (N, M)
    val_diff = img_v - wheel_v  # Shape: (N, M)
    
    # Weighted distance calculation (vectorized)
    hue_weight = 3.0    # Hue is most important for wheel position
    sat_weight = 1.0    # Saturation determines distance from center
    val_weight = 0.5    # Value is less important for wheel mapping
    
    distances = (hue_weight * hue_diff**2 + 
                sat_weight * sat_diff**2 + 
                val_weight * val_diff**2)  # Shape: (N, M)
    
    # Find nearest wheel color for each image color
    nearest_indices = np.argmin(distances, axis=1)  # Shape: (N,)
    
    # Create mapping dictionary
    color_mapping = {}
    for i, image_color in enumerate(image_colors):
        nearest_wheel_color = wheel_colors[nearest_indices[i]]
        color_mapping[image_color] = nearest_wheel_color
    
    return color_mapping


def create_color_wheel(color_percentages, wheel_size=800, inner_radius_ratio=0.1, quantize_level=8):
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
        
    Returns:
        tuple: (numpy.ndarray, dict, list) - RGBA image of the color wheel, normalized percentages, and opacity values
    """
    # Load or create the wheel template (cached on disk)
    wheel_rgb, color_to_pixels_map = load_or_create_wheel_template(wheel_size, inner_radius_ratio, quantize_level)
    
    # Create output image (RGBA) - start with RGB template
    wheel = np.zeros((wheel_size, wheel_size, 4), dtype=np.uint8)
    wheel[:, :, :3] = wheel_rgb  # Copy RGB channels
    
    # Find max percentage for normalization
    max_percentage = max(color_percentages.values()) if color_percentages else 1.0
    
    # Normalize all percentages so max becomes 1.0 (for full opacity)
    normalized_percentages = {}
    for color, percentage in color_percentages.items():
        normalized_percentages[color] = percentage / max_percentage
    
    # Collect opacity values for histogram
    opacity_values = []
    
    # NEW APPROACH: Map each input image color to the nearest color in the wheel template
    # VECTORIZED: Do all color mappings at once for massive speed improvement
    wheel_color_frequencies = {}
    
    # Get all image colors and do vectorized nearest-neighbor lookup
    image_colors = list(normalized_percentages.keys())
    if image_colors:
        color_mapping = find_nearest_wheel_colors_vectorized(image_colors, color_to_pixels_map)
        
        # Accumulate frequencies for wheel colors
        for image_color, percentage in normalized_percentages.items():
            nearest_wheel_color = color_mapping[image_color]
            
            if nearest_wheel_color in wheel_color_frequencies:
                wheel_color_frequencies[nearest_wheel_color] += percentage
            else:
                wheel_color_frequencies[nearest_wheel_color] = percentage
    
    # Normalize accumulated frequencies so max becomes 1.0 (to prevent overflow)
    max_accumulated_freq = max(wheel_color_frequencies.values()) if wheel_color_frequencies else 1.0
    for wheel_color in wheel_color_frequencies:
        wheel_color_frequencies[wheel_color] /= max_accumulated_freq
    
    # Now apply frequencies to the wheel using the mapped colors
    for quantized_color, pixel_coords in color_to_pixels_map.items():
        # Get the accumulated frequency for this wheel color
        normalized_frequency = wheel_color_frequencies.get(quantized_color, 0)
        
        # Calculate opacity based on frequency
        if normalized_frequency > 0:
            # Map normalized frequency to opacity range (128-255)
            # Using linear mapping for even distribution
            curved_frequency = normalized_frequency  # Linear mapping for even spread
            opacity = int(64 + (255 - 64) * curved_frequency)  # Map to 64-255 range
            opacity_values.append(opacity)  # Collect for histogram
        else:
            opacity = 32  # Low opacity for colors not in image
        
        # Set opacity for all pixels of this color at once (vectorized)
        if len(pixel_coords) > 0:
            y_coords = pixel_coords[:, 0]
            x_coords = pixel_coords[:, 1]
            wheel[y_coords, x_coords, 3] = opacity
            
    return wheel, normalized_percentages, opacity_values


def add_wheel_gradient(wheel_size=800, inner_radius_ratio=0.1, quantize_level=8):
    """
    Create a reference color wheel with full saturation gradient.
    Uses the same precomputed template for consistency and speed.
    
    Returns:
        numpy.ndarray: RGBA image of a standard color wheel
    """
    # Load or create the wheel template (cached on disk)
    wheel_rgb, _ = load_or_create_wheel_template(wheel_size, inner_radius_ratio, quantize_level)
    
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
    spectrum_colors = []
    frequencies = []
    
    # Generate full spectrum colors (HSV with full saturation and value)
    for hue in range(spectrum_width):
        # Convert HSV to RGB for this hue (full saturation and value)
        h = hue / 360.0
        s = 1.0  # Full saturation
        v = 1.0  # Full value/brightness
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        spectrum_rgb = (int(r * 255), int(g * 255), int(b * 255))
        spectrum_colors.append(spectrum_rgb)
        
        # Find if this color (or similar) exists in the image
        # We need to find the closest match since exact matches are unlikely
        closest_frequency = 0
        
        # Look for colors in the image that are close to this spectrum color
        for image_color, freq in color_percentages.items():
            # Convert image color to HSV to compare hue
            img_r, img_g, img_b = [x/255.0 for x in image_color]
            img_h, img_s, img_v = colorsys.rgb_to_hsv(img_r, img_g, img_b)
            img_hue = img_h * 360
            
            # Check if hues are close (within a few degrees)
            hue_diff = abs(img_hue - hue)
            # Handle wraparound (e.g., 359° vs 1°)
            hue_diff = min(hue_diff, 360 - hue_diff)
            
            # If hue is close and color has reasonable saturation, include its frequency
            if hue_diff <= 2 and img_s > 0.3:  # Within 2 degrees and not too gray
                closest_frequency = max(closest_frequency, freq)
        
        frequencies.append(closest_frequency)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Create bars with spectrum colors
    x_positions = range(len(spectrum_colors))
    bars = ax.bar(x_positions, frequencies, width=1.0, edgecolor='none')
    
    # Set bar colors to match the spectrum colors
    for i, (bar, rgb_color) in enumerate(zip(bars, spectrum_colors)):
        # Normalize RGB to 0-1 range for matplotlib
        normalized_color = [c/255.0 for c in rgb_color]
        bar.set_facecolor(normalized_color)
    
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
    non_zero_freqs = [f for f in frequencies if f > 0]
    total_colors_in_image = len(non_zero_freqs)
    max_freq = max(frequencies) if frequencies else 0
    avg_freq = np.mean(non_zero_freqs) if non_zero_freqs else 0
    
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
    
    # Calculate frequencies for each hue
    frequencies = []
    spectrum_colors = []
    
    for hue in range(num_segments):
        # Convert HSV to RGB for this hue
        h = hue / 360.0
        s = 1.0  # Full saturation
        v = 1.0  # Full value/brightness
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        spectrum_rgb = (r, g, b)  # Keep as 0-1 for matplotlib
        spectrum_colors.append(spectrum_rgb)
        
        # Find frequency for this hue
        closest_frequency = 0
        for image_color, freq in color_percentages.items():
            img_r, img_g, img_b = [x/255.0 for x in image_color]
            img_h, img_s, img_v = colorsys.rgb_to_hsv(img_r, img_g, img_b)
            img_hue = img_h * 360
            
            hue_diff = abs(img_hue - hue)
            hue_diff = min(hue_diff, 360 - hue_diff)
            
            # Include more desaturated colors (like sky blues) by lowering saturation threshold
            if hue_diff <= 2 and img_s > 0.1:  # Lowered from 0.3 to 0.1
                closest_frequency = max(closest_frequency, freq)
        
        frequencies.append(closest_frequency)
    
    # Normalize frequencies for spike lengths
    max_freq = max(frequencies) if frequencies and max(frequencies) > 0 else 1
    normalized_freqs = [f / max_freq for f in frequencies]
    
    # Draw the circular spectrum with spikes
    for i in range(num_segments):
        # Calculate angle (start from right at 0°, go counter-clockwise)
        angle = i * np.pi / 180  # Convert to radians, start from right (0°), go counter-clockwise
        
        # Calculate spike length
        spike_length = normalized_freqs[i] * max_spike_length
        
        # Calculate positions
        inner_x = inner_radius * np.cos(angle)
        inner_y = inner_radius * np.sin(angle)
        outer_x = (inner_radius + spike_length) * np.cos(angle)
        outer_y = (inner_radius + spike_length) * np.sin(angle)
        
        # Draw the spike (line from inner circle to outer point)
        if spike_length > 0:
            ax.plot([inner_x, outer_x], [inner_y, outer_y], 
                   color=spectrum_colors[i], linewidth=2, alpha=0.8)
        
        # Draw a small segment of the inner circle with the spectrum color
        segment_width = 2 * np.pi / num_segments
        angle_start = angle - segment_width/2
        angle_end = angle + segment_width/2
        
        # Create a small arc for the inner circle
        arc_angles = np.linspace(angle_start, angle_end, 5)
        arc_x = inner_radius * np.cos(arc_angles)
        arc_y = inner_radius * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color=spectrum_colors[i], linewidth=3)
    
    # Add title and statistics
    non_zero_freqs = [f for f in frequencies if f > 0]
    total_colors = len(non_zero_freqs)
    avg_freq = np.mean(non_zero_freqs) if non_zero_freqs else 0
    
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
    parser.add_argument("--sample-factor", type=int, default=2, 
                       help="Factor to downsample input image for faster processing (default: 4)")
    parser.add_argument("--quantize", type=int, default=8, 
                       help="Color quantization level: 1=no quantization (most precise), higher=more grouping (default: 8)")
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
    
    args = parser.parse_args()
    
    try:
        print(f"Loading and analyzing image: {args.input_image}")
        color_percentages = load_and_analyze_image(args.input_image, args.sample_factor, args.quantize)
        print(f"Found {len(color_percentages)} unique colors (quantization level: {args.quantize})")
        
        print("Generating color wheel...")
        wheel, normalized_percentages, opacity_values = create_color_wheel(color_percentages, args.size, quantize_level=args.quantize)
        
        # Handle output format
        if args.format == "jpg":
            # JPG doesn't support transparency, so blend with black background
            output_path = args.output_wheel
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
            output_path = args.output_wheel
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
            
            if args.format == "jpg":
                # Convert reference wheel to RGB with black background
                rgb_ref = np.zeros((reference_wheel.shape[0], reference_wheel.shape[1], 3), dtype=np.uint8)
                alpha = reference_wheel[:, :, 3] / 255.0
                
                for i in range(3):
                    rgb_ref[:, :, i] = (reference_wheel[:, :, i] * alpha + 0 * (1 - alpha)).astype(np.uint8)  # Black background
                
                reference_wheel_bgr = rgb_ref[:, :, [2, 1, 0]]
                reference_path = output_path.replace('.jpg', '_reference.jpg')
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
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
