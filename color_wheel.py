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


def create_wheel_template(wheel_size=800, inner_radius_ratio=0.1):
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
    angles = np.arctan2(valid_dy, valid_dx)
    hues = (angles + np.pi) * 180 / np.pi
    
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
    wheel_r_quantized = (wheel_r_full // 8) * 8
    wheel_g_quantized = (wheel_g_full // 8) * 8
    wheel_b_quantized = (wheel_b_full // 8) * 8
    
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


def get_wheel_template_path(wheel_size, inner_radius_ratio):
    """Get the path for the wheel template file in a dedicated templates folder."""
    # Create templates directory if it doesn't exist
    templates_dir = "wheel_templates"
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    filename = f"wheel_template_fullres_{wheel_size}_{inner_radius_ratio:.3f}.pkl"
    return os.path.join(templates_dir, filename)


def load_or_create_wheel_template(wheel_size=800, inner_radius_ratio=0.1):
    """
    Load existing wheel template or create a new one if it doesn't exist.
    
    Returns:
        tuple: (wheel_rgb, color_to_pixels_map)
    """
    template_path = get_wheel_template_path(wheel_size, inner_radius_ratio)
    
    if os.path.exists(template_path):
        print(f"Loading precomputed wheel template: {template_path}")
        with open(template_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Creating new wheel template: {template_path}")
        template_data = create_wheel_template(wheel_size, inner_radius_ratio)
        
        # Save the template for future use
        with open(template_path, 'wb') as f:
            pickle.dump(template_data, f)
        
        print(f"Wheel template saved to: {template_path}")
        return template_data


def load_and_analyze_image(image_path, sample_factor=4):
    """
    Load an image and analyze color frequencies as percentages.
    
    Args:
        image_path (str): Path to the input image
        sample_factor (int): Factor to downsample image for faster processing
        
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
    quantized_pixels = (pixels // 8) * 8
    
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


def create_color_wheel(color_percentages, wheel_size=800, inner_radius_ratio=0.1):
    """
    Create a full color wheel where opacity represents color frequency.
    Areas with frequent colors are more opaque.
    Areas with rare/missing colors are more transparent.
    
    Uses a precomputed wheel template for maximum performance.
    
    Args:
        color_percentages (dict): Color frequency percentages {(r,g,b): percentage}
        wheel_size (int): Size of the output wheel image
        inner_radius_ratio (float): Ratio of inner radius to outer radius
        
    Returns:
        tuple: (numpy.ndarray, dict) - RGBA image of the color wheel and normalized percentages
    """
    # Load or create the wheel template (cached on disk)
    wheel_rgb, color_to_pixels_map = load_or_create_wheel_template(wheel_size, inner_radius_ratio)
    
    # Create output image (RGBA) - start with RGB template
    wheel = np.zeros((wheel_size, wheel_size, 4), dtype=np.uint8)
    wheel[:, :, :3] = wheel_rgb  # Copy RGB channels
    
    # Find max percentage for normalization
    max_percentage = max(color_percentages.values()) if color_percentages else 1.0
    
    # Normalize all percentages so max becomes 1.0 (for full opacity)
    normalized_percentages = {}
    for color, percentage in color_percentages.items():
        normalized_percentages[color] = percentage / max_percentage
    
    # ULTRA-FAST APPROACH: Use the precomputed color-to-pixels mapping
    # Instead of iterating through pixels, iterate through colors
    for quantized_color, pixel_coords in color_to_pixels_map.items():
        # Get the normalized frequency for this color
        normalized_frequency = normalized_percentages.get(quantized_color, 0)
        
        # Calculate opacity based on frequency
        if normalized_frequency > 0:
            boosted_frequency = normalized_frequency ** 0.05  # More aggressive boost (lower power = more boost)
            opacity = int(255 * boosted_frequency)
            opacity = max(opacity, 128)  # Minimum 128/255 opacity for any visible color
        else:
            opacity = 32  # Low opacity for colors not in image
        
        # Set opacity for all pixels of this color at once (vectorized)
        if len(pixel_coords) > 0:
            y_coords = pixel_coords[:, 0]
            x_coords = pixel_coords[:, 1]
            wheel[y_coords, x_coords, 3] = opacity
            
    return wheel, normalized_percentages


def add_wheel_gradient(wheel_size=800, inner_radius_ratio=0.1):
    """
    Create a reference color wheel with full saturation gradient.
    Uses the same precomputed template for consistency and speed.
    
    Returns:
        numpy.ndarray: RGBA image of a standard color wheel
    """
    # Load or create the wheel template (cached on disk)
    wheel_rgb, _ = load_or_create_wheel_template(wheel_size, inner_radius_ratio)
    
    # Create output image (RGBA) - start with RGB template
    wheel = np.zeros((wheel_size, wheel_size, 4), dtype=np.uint8)
    wheel[:, :, :3] = wheel_rgb  # Copy RGB channels
    
    # Set full opacity for all valid pixels (where RGB is not zero)
    valid_pixels = np.any(wheel_rgb > 0, axis=2)
    wheel[valid_pixels, 3] = 255  # Full opacity
    
    return wheel


def create_opacity_histogram(normalized_percentages, output_path):
    """
    Create and save a histogram showing the distribution of normalized opacity values.
    
    Args:
        normalized_percentages (dict): Normalized color percentages {(r,g,b): percentage}
        output_path (str): Path to save the histogram image
    """
    if not normalized_percentages:
        print("No color data available for histogram")
        return
    
    # Get all opacity values (convert to 0-255 range for histogram)
    opacity_values = []
    for percentage in normalized_percentages.values():
        # Apply the same boosting logic as in the wheel generation
        if percentage > 0:
            boosted_frequency = percentage ** 0.3
            opacity = int(255 * boosted_frequency)
            opacity = max(opacity, 80)  # Same minimum as wheel
            opacity_values.append(opacity)
    
    if not opacity_values:
        print("No opacity values to plot")
        return
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(opacity_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Opacity Value (0-255)')
    plt.ylabel('Frequency Count')
    plt.title('Distribution of Color Opacity Values\n(Normalized and Boosted Frequencies)')
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


def main():
    """Main function to run the color wheel generator."""
    parser = argparse.ArgumentParser(
        description="Generate a color wheel where opacity represents color frequency in an image"
    )
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_wheel", help="Path for the output color wheel image")
    parser.add_argument("--size", type=int, default=800, help="Size of the color wheel (default: 800)")
    parser.add_argument("--sample-factor", type=int, default=4, 
                       help="Factor to downsample input image for faster processing (default: 4)")
    parser.add_argument("--show-reference", action="store_true", 
                       help="Also save a reference color wheel for comparison")
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                       help="Output format: png (supports transparency) or jpg (black background, default: png)")
    parser.add_argument("--histogram", action="store_true",
                       help="Also generate a histogram showing the distribution of opacity values")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading and analyzing image: {args.input_image}")
        color_percentages = load_and_analyze_image(args.input_image, args.sample_factor)
        print(f"Found {len(color_percentages)} unique colors")
        
        print("Generating color wheel...")
        wheel, normalized_percentages = create_color_wheel(color_percentages, args.size)
        
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
            reference_wheel = add_wheel_gradient(args.size)
            
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
            create_opacity_histogram(normalized_percentages, histogram_path)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
