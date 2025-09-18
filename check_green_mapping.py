#!/usr/bin/env python3
"""
Diagnostic script to check color wheel mapping for green colors
"""

import numpy as np
import cv2
import colorsys
import pickle
import os

def load_and_analyze_image(image_path, sample_factor=1, quantize_level=8):
    """Same as in color_wheel.py"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if sample_factor > 1:
        height, width = image.shape[:2]
        new_height, new_width = height // sample_factor, width // sample_factor
        image = cv2.resize(image, (new_width, new_height))
    
    pixels = image.reshape(-1, 3)
    total_pixels = len(pixels)
    
    if quantize_level > 1:
        quantized_pixels = (pixels // quantize_level) * quantize_level
    else:
        quantized_pixels = pixels
    
    rgb_as_int = quantized_pixels[:, 0].astype(np.int64) * 65536 + quantized_pixels[:, 1].astype(np.int64) * 256 + quantized_pixels[:, 2].astype(np.int64)
    unique_colors, counts = np.unique(rgb_as_int, return_counts=True)
    
    color_percentages = {}
    for color_int, count in zip(unique_colors, counts):
        r = int((color_int // 65536) % 256)
        g = int((color_int // 256) % 256)
        b = int(color_int % 256)
        color_percentages[(r, g, b)] = count / total_pixels
    
    return color_percentages

def get_wheel_template_path(wheel_size, inner_radius_ratio, quantize_level):
    templates_dir = "wheel_templates"
    filename = f"wheel_template_fullres_{wheel_size}_{inner_radius_ratio:.3f}_q{quantize_level}.pkl"
    return os.path.join(templates_dir, filename)

def check_green_mapping(image_path, quantize_level=8):
    """Check if green colors from image are mapped correctly in wheel template"""
    
    # Analyze image
    print("Analyzing image colors...")
    color_percentages = load_and_analyze_image(image_path, quantize_level=quantize_level)
    
    # Find green colors in image
    green_colors = []
    for (r, g, b), percentage in color_percentages.items():
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h_deg = h * 360
        
        # Check if this is green (60-180 degrees hue)
        if 60 <= h_deg <= 180 and s > 0.1 and v > 0.1:
            green_colors.append({
                'rgb': (r, g, b),
                'hsv': (h_deg, s, v),
                'percentage': percentage
            })
    
    print(f"Found {len(green_colors)} green colors in image")
    
    # Sort by percentage
    green_colors.sort(key=lambda x: x['percentage'], reverse=True)
    
    print("\nTop 10 green colors from image:")
    for i, color in enumerate(green_colors[:10]):
        r, g, b = color['rgb']
        h, s, v = color['hsv']
        print(f"{i+1}. RGB({r:3d},{g:3d},{b:3d}) HSV({h:5.1f}°,{s:4.2f},{v:4.2f}) {color['percentage']:.4f}")
    
    # Load wheel template
    template_path = get_wheel_template_path(800, 0.1, quantize_level)
    if not os.path.exists(template_path):
        print(f"\nWheel template not found at {template_path}")
        print("Run the main program first to generate the template.")
        return
    
    print(f"\nLoading wheel template from {template_path}")
    with open(template_path, 'rb') as f:
        wheel_rgb, color_to_pixels_map = pickle.load(f)
    
    print(f"Wheel template has {len(color_to_pixels_map)} unique colors")
    
    # Check how many green colors from image are in the wheel template
    found_in_wheel = 0
    missing_from_wheel = 0
    
    print("\nChecking if green colors from image exist in wheel template:")
    for i, color in enumerate(green_colors[:10]):
        rgb = color['rgb']
        if rgb in color_to_pixels_map:
            pixel_count = len(color_to_pixels_map[rgb])
            print(f"✓ {rgb} found in wheel ({pixel_count} pixels)")
            found_in_wheel += 1
        else:
            print(f"✗ {rgb} NOT found in wheel template")
            missing_from_wheel += 1
    
    print(f"\nSummary: {found_in_wheel} found, {missing_from_wheel} missing from wheel template")
    
    if missing_from_wheel > 0:
        print("\nThis suggests the wheel template doesn't contain all the green shades from your image.")
        print("This could happen if the quantization levels don't match or if the wheel")
        print("doesn't have sufficient resolution in the green region.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_green_mapping.py image.jpg")
        sys.exit(1)
    
    check_green_mapping(sys.argv[1], quantize_level=4)