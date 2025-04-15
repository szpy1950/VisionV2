import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

image_path = "images/hack2.png"

def display_image(title, image):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


print("Reading image from path:", image_path)
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if original_image is None:
    raise ValueError(f"Could not read image from {image_path}")
display_image("Original Image", original_image)

print("Converting to grayscale")
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
display_image("Grayscale Image", gray_image)

print("Threshold to separate pieces from background")
_, binary_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
display_image("Binary Image", binary_image)

# Store original binary for later reference
original_binary = binary_image.copy()

print("Doing morph type operations")
kernel = np.ones((12, 12), np.uint8)

# Close = filling the holes
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Open = removing the noise
morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
display_image("Morph Operations", morph_image)

# Applying
print("Filling holes in puzzle pieces")
contours_fill, _ = cv2.findContours(morph_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_fill:
    cv2.drawContours(morph_image, [cnt], 0, 255, -1)
display_image("Filled Holes", morph_image)

# NEW CODE: Subtract areas that are in morph_image but not in original_binary
print("Subtracting expanded areas to restore original boundaries")

# Get the external contours of both the original binary and morphed image
original_external, _ = cv2.findContours(original_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
morphed_external, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create final result image
result = np.zeros_like(binary_image)

# Process each original piece
for orig_cnt in original_external:
    # Create a mask for this original piece
    orig_mask = np.zeros_like(binary_image)
    cv2.drawContours(orig_mask, [orig_cnt], 0, 255, -1)

    # Find overlap with morphed image (this preserves filled holes)
    overlap = cv2.bitwise_and(morph_image, orig_mask)

    # Add to result
    result = cv2.bitwise_or(result, overlap)

display_image("Final Result with Original Boundaries", result)