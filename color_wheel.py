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
from collections import defaultdict
import math
import matplotlib.pyplot as plt


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
    
    # Count color frequencies
    color_counts = defaultdict(int)
    pixels = image.reshape(-1, 3)
    total_pixels = len(pixels)
    
    for pixel in pixels:
        # Quantize colors to reduce noise (group similar colors)
        quantized_pixel = tuple((pixel // 8) * 8)  # Reduce to 32 levels per channel
        color_counts[quantized_pixel] += 1
    
    # Convert counts to percentages
    color_percentages = {}
    for color, count in color_counts.items():
        color_percentages[color] = count / total_pixels
    
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
    
    Args:
        color_percentages (dict): Color frequency percentages {(r,g,b): percentage}
        wheel_size (int): Size of the output wheel image
        inner_radius_ratio (float): Ratio of inner radius to outer radius
        
    Returns:
        tuple: (numpy.ndarray, dict) - RGBA image of the color wheel and normalized percentages
    """
    # Create output image (RGBA)
    wheel = np.zeros((wheel_size, wheel_size, 4), dtype=np.uint8)
    center = wheel_size // 2
    outer_radius = center - 10  # Leave small border
    inner_radius = int(outer_radius * inner_radius_ratio)
    
    # Find max percentage for normalization
    max_percentage = max(color_percentages.values()) if color_percentages else 1.0
    
    # Normalize all percentages so max becomes 1.0 (for full opacity)
    normalized_percentages = {}
    for color, percentage in color_percentages.items():
        normalized_percentages[color] = percentage / max_percentage
    
    # Generate wheel pixels
    for y in range(wheel_size):
        for x in range(wheel_size):
            # Calculate distance from center
            dx = x - center
            dy = y - center
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Skip pixels outside the wheel or inside inner circle
            if distance > outer_radius or distance < inner_radius:
                continue
            
            # Convert to polar coordinates
            angle = math.atan2(dy, dx)  # -π to π
            hue = (angle + math.pi) * 180 / math.pi  # Convert to 0-360 degrees
            
            # Calculate saturation (0 at inner_radius, 1 at outer_radius)
            saturation = (distance - inner_radius) / (outer_radius - inner_radius)
            
            # Set maximum value/brightness for full color wheel
            value = 1.0
            
            # Convert HSV back to RGB for the base wheel color
            r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
            wheel_r, wheel_g, wheel_b = int(r*255), int(g*255), int(b*255)
            
            # Quantize the wheel color to match our analysis quantization
            quantized_color = ((wheel_r // 8) * 8, (wheel_g // 8) * 8, (wheel_b // 8) * 8)
            
            # Get the normalized frequency for this color (0-1 range)
            normalized_frequency = normalized_percentages.get(quantized_color, 0)
            
            # Convert normalized frequency to opacity with boosted values
            # Apply power curve to spread out the values and boost visibility
            if normalized_frequency > 0:
                # Boost all visible frequencies - even small ones become quite visible
                boosted_frequency = normalized_frequency ** 20  # More aggressive boost
                opacity = int(255 * boosted_frequency)
                # Ensure minimum opacity for any color that exists in the image
                opacity = max(opacity, 128)  # Minimum 128/255 opacity for any visible color
            else:
                opacity = 32  # Completely transparent for colors not in image
            
            # Set the pixel with frequency-based opacity
            wheel[y, x] = [wheel_r, wheel_g, wheel_b, opacity]
            
    return wheel, normalized_percentages


def add_wheel_gradient(wheel_size=800, inner_radius_ratio=0.1):
    """
    Create a reference color wheel with full saturation gradient.
    
    Returns:
        numpy.ndarray: RGBA image of a standard color wheel
    """
    wheel = np.zeros((wheel_size, wheel_size, 4), dtype=np.uint8)
    center = wheel_size // 2
    outer_radius = center - 10
    inner_radius = int(outer_radius * inner_radius_ratio)
    
    for y in range(wheel_size):
        for x in range(wheel_size):
            dx = x - center
            dy = y - center
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > outer_radius or distance < inner_radius:
                continue
            
            angle = math.atan2(dy, dx)
            hue = (angle + math.pi) * 180 / math.pi
            saturation = (distance - inner_radius) / (outer_radius - inner_radius)
            value = 1.0
            
            r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
            wheel_r, wheel_g, wheel_b = int(r*255), int(g*255), int(b*255)
            
            wheel[y, x] = [wheel_r, wheel_g, wheel_b, 255]
    
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
