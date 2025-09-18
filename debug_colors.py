#!/usr/bin/env python3
"""
Debug script to analyze colors in an image and check color wheel mapping
"""

import numpy as np
import cv2
import colorsys

def analyze_image_colors(image_path, quantize_level=1):
    """Analyze colors in an image and show top colors"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {image.shape}")
    
    # Count color frequencies
    pixels = image.reshape(-1, 3)
    total_pixels = len(pixels)
    
    # Quantize colors if needed
    if quantize_level > 1:
        quantized_pixels = (pixels // quantize_level) * quantize_level
    else:
        quantized_pixels = pixels
    
    # Count unique colors
    rgb_as_int = quantized_pixels[:, 0].astype(np.int64) * 65536 + quantized_pixels[:, 1].astype(np.int64) * 256 + quantized_pixels[:, 2].astype(np.int64)
    unique_colors, counts = np.unique(rgb_as_int, return_counts=True)
    
    # Convert back to RGB and sort by frequency
    color_data = []
    for color_int, count in zip(unique_colors, counts):
        r = int((color_int // 65536) % 256)
        g = int((color_int // 256) % 256)
        b = int(color_int % 256)
        percentage = count / total_pixels
        
        # Convert to HSV to analyze
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h_deg = h * 360
        
        color_data.append({
            'rgb': (r, g, b),
            'hsv': (h_deg, s, v),
            'count': count,
            'percentage': percentage
        })
    
    # Sort by frequency (most common first)
    color_data.sort(key=lambda x: x['percentage'], reverse=True)
    
    print(f"\nFound {len(color_data)} unique colors")
    print("\nTop 20 most frequent colors:")
    print("RGB           HSV            Count      %")
    print("-" * 55)
    
    green_colors = 0
    for i, color in enumerate(color_data[:20]):
        r, g, b = color['rgb']
        h, s, v = color['hsv']
        
        # Check if this is a green-ish color (hue around 120°, or 60-180°)
        is_green = 60 <= h <= 180 and s > 0.2 and v > 0.2
        if is_green:
            green_colors += 1
            green_marker = " <-- GREEN"
        else:
            green_marker = ""
            
        print(f"({r:3d},{g:3d},{b:3d})   ({h:5.1f}°,{s:4.2f},{v:4.2f})   {color['count']:6d}   {color['percentage']:6.3f}{green_marker}")
    
    print(f"\nFound {green_colors} green colors in top 20")
    
    # Count all green colors
    total_green = sum(1 for color in color_data if 60 <= color['hsv'][0] <= 180 and color['hsv'][1] > 0.2 and color['hsv'][2] > 0.2)
    total_green_percentage = sum(color['percentage'] for color in color_data if 60 <= color['hsv'][0] <= 180 and color['hsv'][1] > 0.2 and color['hsv'][2] > 0.2)
    
    print(f"Total green colors found: {total_green} ({total_green_percentage:.1%} of image)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_colors.py image.jpg")
        sys.exit(1)
    
    analyze_image_colors(sys.argv[1], quantize_level=1)