#!/usr/bin/env python3
"""
Create a simple test image with known colors for testing the color wheel generator.
"""

import numpy as np
import cv2

# Create a test image with distinct color regions
test_image = np.zeros((300, 300, 3), dtype=np.uint8)

# Add different colored regions
test_image[0:100, 0:100] = [255, 0, 0]      # Red
test_image[0:100, 100:200] = [0, 255, 0]    # Green  
test_image[0:100, 200:300] = [0, 0, 255]    # Blue
test_image[100:200, 0:100] = [255, 255, 0]  # Yellow
test_image[100:200, 100:200] = [255, 0, 255] # Magenta
test_image[100:200, 200:300] = [0, 255, 255] # Cyan
test_image[200:300, 0:150] = [255, 128, 0]   # Orange
test_image[200:300, 150:300] = [128, 0, 255] # Purple

# Save test image
cv2.imwrite('test_image.jpg', test_image)
print("Test image 'test_image.jpg' created successfully!")