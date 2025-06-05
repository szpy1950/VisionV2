# Edge-Based Puzzle Piece Detection - Jupyter Notebook Version
# This fixes the "everything disappears" problem with morphological operations

# Cell 1: Imports (same as before)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os


def display_image(title, image, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


# Cell 2: Load and prepare image
image_path = "images/edge_detect2.png"  # Your problematic image
print("Reading image from path:", image_path)
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if original_image is None:
    raise ValueError(f"Could not read image from {image_path}")

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
display_image("Original Image", original_image)
display_image("Grayscale Image", gray_image)

# Cell 3: Improved Edge Detection
print("=== EDGE-BASED DETECTION WITH PROPER MORPHOLOGY ===")

# Step 1: Noise reduction
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 2: Canny edge detection
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
display_image("Canny Edges", edges)

# Cell 4: CRITICAL - Thicken edges before morphology
print("Thickening edges before morphological operations...")

# Method A: Dilate edges to make them thicker
edge_kernel = np.ones((3, 3), np.uint8)  # Small kernel for edge thickening
thick_edges = cv2.dilate(edges, edge_kernel, iterations=2)
display_image("Thickened Edges", thick_edges)

# Cell 5: Close gaps in edges
print("Closing gaps in puzzle piece edges...")

# Use a slightly larger kernel to close gaps between edge segments
close_kernel = np.ones((5, 5), np.uint8)
closed_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, close_kernel, iterations=3)
display_image("Closed Edges", closed_edges)

# Cell 6: Fill enclosed regions
print("Filling enclosed puzzle piece regions...")

# Find contours and fill them
contours_temp, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create filled image
filled_image = np.zeros_like(gray_image)
cv2.drawContours(filled_image, contours_temp, -1, 255, thickness=cv2.FILLED)
display_image("Filled Regions", filled_image)

# Cell 7: Alternative Method - Flood Fill from Edges
print("Alternative: Using flood fill to create solid regions...")

# Create a copy for flood fill
flood_fill_image = closed_edges.copy()

# Add border to ensure proper flood fill
h, w = flood_fill_image.shape
flood_fill_image = cv2.copyMakeBorder(flood_fill_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

# Flood fill from (0,0) to fill background
cv2.floodFill(flood_fill_image, None, (0, 0), 255)

# Remove the border
flood_fill_image = flood_fill_image[1:h + 1, 1:w + 1]

# Invert to get puzzle pieces as white
flood_fill_result = cv2.bitwise_not(flood_fill_image)
display_image("Flood Fill Result", flood_fill_result)

# Cell 8: Choose best result and apply your original morphological operations
print("Applying final morphological cleanup...")

# Choose which method worked better (you can switch between filled_image and flood_fill_result)
binary_image = filled_image  # or flood_fill_result

# NOW apply your original morphological operations (they should work now!)
kernel = np.ones((12, 12), np.uint8)

# Close = filling the holes
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
display_image("After Morphological Close", morph_image)

# Open = removing the noise
morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
display_image("After Morphological Open", morph_image)

# Cell 9: Fill any remaining holes
print("Filling remaining holes...")
contours_fill, _ = cv2.findContours(morph_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_fill:
    cv2.drawContours(morph_image, [cnt], 0, 255, -1)
display_image("Final Filled Image", morph_image)

# Cell 10: Find and filter final contours
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"Found {len(contours)} potential puzzle pieces")

# Filter by size
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
if len(contours) > 1:
    reference_area = cv2.contourArea(contours[1])
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > reference_area / 3]

print(f"After filtering: {len(contours)} puzzle pieces")

# Draw final results
result_image = original_image.copy()
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
display_image("Final Detection Results", result_image)


# Cell 11: Parameter Tuning for Edge Detection
def test_edge_parameters(gray_img):
    """Test different Canny parameters"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Different parameter combinations
    params = [
        (30, 100), (50, 150), (70, 200),
        (40, 120), (60, 180), (80, 240)
    ]

    for i, (low, high) in enumerate(params):
        row, col = i // 3, i % 3

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Canny with different thresholds
        edges = cv2.Canny(blurred, low, high)

        axes[row, col].imshow(edges, cmap='gray')
        axes[row, col].set_title(f'Canny({low}, {high})')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


# Uncomment to test different edge detection parameters
# test_edge_parameters(gray_image)

# Cell 12: Advanced Edge Processing (if basic method doesn't work)
print("=== ADVANCED EDGE PROCESSING ===")
print("Uncomment this cell if you need more aggressive edge processing")

# # Method: Multi-scale edge detection
# def multi_scale_edges(img):
#     edges_combined = np.zeros_like(img)

#     # Different blur levels for multi-scale detection
#     for sigma in [1, 2, 3]:
#         blurred = cv2.GaussianBlur(img, (0, 0), sigma)
#         edges = cv2.Canny(blurred, 30, 100)
#         edges_combined = cv2.bitwise_or(edges_combined, edges)

#     return edges_combined

# multi_edges = multi_scale_edges(gray_image)
# display_image("Multi-scale Edges", multi_edges)

# # Process multi-scale edges
# edge_kernel = np.ones((3, 3), np.uint8)
# thick_multi_edges = cv2.dilate(multi_edges, edge_kernel, iterations=2)
# closed_multi_edges = cv2.morphologyEx(thick_multi_edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)

# # Fill regions
# contours_multi, _ = cv2.findContours(closed_multi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# filled_multi = np.zeros_like(gray_image)
# cv2.drawContours(filled_multi, contours_multi, -1, 255, thickness=cv2.FILLED)
# display_image("Multi-scale Result", filled_multi)

print("\n=== SUCCESS! ===")
print("Edge detection with proper morphological operations complete!")
print(f"Detected {len(contours)} puzzle pieces")
print("\nKey changes made:")
print("1. Thickened edges before morphological operations")
print("2. Used smaller kernels initially to preserve edge connectivity")
print("3. Filled regions properly before applying your original morphology")
print("4. Added flood fill as alternative method")