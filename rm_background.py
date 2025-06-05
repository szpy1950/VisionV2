# BALANCED JIGSAW PUZZLE DETECTION
# Keeps morphological operations but reduces them to avoid merging pieces

# ========================================
# CELL 1: Setup and Image Loading
# ========================================
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(title, image, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


# Load image
image_path = "images/test_images/image3.png"  # Change this to your image path
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

print("🔧 BALANCED JIGSAW PUZZLE DETECTION")
print("Moderate morphological operations - not too big, not too small")
print("=" * 60)

display_image("Original Image", original_image)
display_image("Grayscale Image", gray_image)

# ========================================
# CELL 2: Preprocessing - Gaussian Blur
# ========================================
print("🔧 STEP 1: Preprocessing - Gaussian blur for noise reduction")

# Apply Gaussian blur to reduce noise
blur_kernel_size = 5  # Must be odd number
blurred = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)

display_image("After Gaussian Blur", blurred)

print(f"✅ Applied {blur_kernel_size}x{blur_kernel_size} Gaussian blur")

# ========================================
# CELL 3: Canny Edge Detection
# ========================================
print("🎯 STEP 2: Canny edge detection")

# Canny edge detection parameters
low_threshold = 50  # Lower threshold for edge linking
high_threshold = 150  # Upper threshold for strong edges

# Apply Canny edge detection
canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)

display_image("Canny Edges", canny_edges)

# Count edge pixels
edge_pixels = np.count_nonzero(canny_edges)
total_pixels = canny_edges.shape[0] * canny_edges.shape[1]
edge_percentage = (edge_pixels / total_pixels) * 100

print(f"📊 Canny edge statistics:")
print(f"   Low threshold: {low_threshold}")
print(f"   High threshold: {high_threshold}")
print(f"   Edge pixels: {edge_pixels:,} ({edge_percentage:.2f}%)")

# ========================================
# CELL 4: Dilation - Thicken Edges
# ========================================
print("📈 STEP 3: Dilation - thicken edges")

# Create structuring element for dilation
dilation_kernel_size = 3
dilation_iterations = 2
dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)

# Apply dilation
dilated_edges = cv2.dilate(canny_edges, dilation_kernel, iterations=dilation_iterations)

display_image("After Dilation (Thickened Edges)", dilated_edges)

print(f"📊 Dilation parameters:")
print(f"   Kernel size: {dilation_kernel_size}x{dilation_kernel_size}")
print(f"   Iterations: {dilation_iterations}")

# ========================================
# CELL 5: Morphological Closing - Fill Gaps
# ========================================
print("🔗 STEP 4: Morphological closing - fill gaps between edges")

# Create structuring element for closing
closing_kernel_size = 3
closing_iterations = 3
closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)

# Apply morphological closing (dilation followed by erosion)
closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)

display_image("After Morphological Closing", closed_edges)

print(f"📊 Closing parameters:")
print(f"   Kernel size: {closing_kernel_size}x{closing_kernel_size}")
print(f"   Iterations: {closing_iterations}")

# ========================================
# CELL 6: Contour Detection and Filling
# ========================================
print("🎨 STEP 5: Contour detection and filling")

# Find contours from closed edges
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours")

# Create filled mask
filled_mask = np.zeros_like(gray_image)
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

display_image("Filled Contours", filled_mask)

print(f"📊 Contour filling:")
print(f"   Contours found: {len(contours)}")

# ========================================
# CELL 7: MODERATE Morphological Operations - Cleanup
# ========================================
print("🧹 STEP 6: MODERATE morphological operations for cleanup")

# REDUCED kernel size - much smaller than original 12x12
cleanup_kernel_size = 6  # Reduced from 12 to 6
cleanup_kernel = np.ones((cleanup_kernel_size, cleanup_kernel_size), np.uint8)

# Morphological closing - fill holes inside objects (REDUCED iterations)
morph_closed = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)  # Reduced from default
display_image("After MODERATE Morphological Close", morph_closed)

# Morphological opening - remove noise (REDUCED iterations)
morph_cleaned = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)  # Reduced from default
display_image("After MODERATE Morphological Open", morph_cleaned)

print(f"📊 MODERATE cleanup parameters:")
print(f"   Cleanup kernel size: {cleanup_kernel_size}x{cleanup_kernel_size} (was 12x12)")
print(f"   Iterations: 1 each (reduced from multiple)")

# Show what changed
difference = cv2.subtract(morph_closed, filled_mask)
display_image("What the Moderate Close Added", difference)

# ========================================
# CELL 8: Test Different Cleanup Kernel Sizes
# ========================================
print("🔬 STEP 6b: Testing different cleanup kernel sizes")

# Test different kernel sizes to find the sweet spot
test_kernels = [4, 6, 8, 10]

fig, axes = plt.subplots(1, len(test_kernels), figsize=(20, 5))

print("Testing cleanup kernel sizes:")
for i, kernel_size in enumerate(test_kernels):
    test_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    test_closed = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, test_kernel, iterations=1)
    test_cleaned = cv2.morphologyEx(test_closed, cv2.MORPH_OPEN, test_kernel, iterations=1)

    # Count objects
    test_contours, _ = cv2.findContours(test_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_test_contours = [cnt for cnt in test_contours if cv2.contourArea(cnt) > 200]

    axes[i].imshow(test_cleaned, cmap='gray')
    axes[i].set_title(f"Kernel {kernel_size}x{kernel_size}\n{len(valid_test_contours)} objects")
    axes[i].axis('off')

    print(f"   {kernel_size}x{kernel_size}: {len(valid_test_contours)} objects")

plt.tight_layout()
plt.show()

# Choose the best kernel size (you can adjust this based on results)
best_kernel_size = 6  # Adjust based on the test results above
best_kernel = np.ones((best_kernel_size, best_kernel_size), np.uint8)
morph_cleaned = cv2.morphologyEx(
    cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, best_kernel, iterations=1),
    cv2.MORPH_OPEN, best_kernel, iterations=1
)

print(f"✅ Using {best_kernel_size}x{best_kernel_size} cleanup kernel")

# ========================================
# CELL 9: Erosion - Reduce Size
# ========================================
print("📉 STEP 7: Erosion - reduce object size")

# Test different erosion levels
erosion_kernel_size = 3
erosion_iterations = [1, 2, 3]

print("Testing different erosion levels:")

fig, axes = plt.subplots(1, len(erosion_iterations), figsize=(15, 4))
erosion_results = {}

for i, iterations in enumerate(erosion_iterations):
    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded = cv2.erode(morph_cleaned, erosion_kernel, iterations=iterations)

    # Count objects after erosion
    contours_eroded, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours_eroded if cv2.contourArea(cnt) > 200]

    erosion_results[iterations] = (eroded, valid_contours)

    axes[i].imshow(eroded, cmap='gray')
    axes[i].set_title(f"Erosion {iterations}\n{len(valid_contours)} objects")
    axes[i].axis('off')

    print(f"   {iterations} iterations: {len(valid_contours)} objects")

plt.tight_layout()
plt.show()

# Choose erosion level
chosen_erosion = 2
eroded_mask, eroded_contours = erosion_results[chosen_erosion]

print(f"✅ Using {chosen_erosion} erosion iterations")

# ========================================
# CELL 10: Dilation - Restore Original Size
# ========================================
print("📈 STEP 8: Dilation - restore original size")

# Restore size by dilating back the same amount as erosion
restoration_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
restored_mask = cv2.dilate(eroded_mask, restoration_kernel, iterations=chosen_erosion)

display_image("After Size Restoration (Erosion + Dilation)", restored_mask)

print(f"📊 Size restoration:")
print(f"   Erosion kernel: {erosion_kernel_size}x{erosion_kernel_size}")
print(f"   Erosion iterations: {chosen_erosion}")
print(f"   Restoration iterations: {chosen_erosion}")

# ========================================
# CELL 11: Final Results
# ========================================
print("🎯 STEP 9: Final results and analysis")

# Get final contours
final_contours, _ = cv2.findContours(restored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Filter by area (remove tiny fragments)
min_area = 200
final_contours = [cnt for cnt in final_contours if cv2.contourArea(cnt) > min_area]

# Draw final result on original image
final_result = original_image.copy()
cv2.drawContours(final_result, final_contours, -1, (0, 255, 0), 2)

display_image(f"FINAL RESULT: {len(final_contours)} objects detected", final_result)

# Calculate statistics
areas = [cv2.contourArea(cnt) for cnt in final_contours]

print(f"📊 Final Results:")
print(f"   Objects detected: {len(final_contours)}")
print(f"   Expected: 24 puzzle pieces")
print(f"   Detection accuracy: {len(final_contours) / 24 * 100:.1f}%")

if areas:
    print(f"   Area statistics:")
    print(f"     - Largest: {max(areas):.0f} pixels")
    print(f"     - Smallest: {min(areas):.0f} pixels")
    print(f"     - Average: {np.mean(areas):.0f} pixels")

# ========================================
# CELL 12: Complete Pipeline Summary
# ========================================
print("📋 BALANCED PIPELINE SUMMARY")
print("=" * 60)

# Show complete pipeline
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

pipeline_steps = [
    (gray_image, "1. Grayscale"),
    (blurred, "2. Gaussian Blur"),
    (canny_edges, "3. Canny Edges"),
    (dilated_edges, "4. Dilation"),
    (closed_edges, "5. Morphological Close"),
    (filled_mask, "6. Contour Filling"),
    (morph_closed, f"7. MODERATE Close ({best_kernel_size}x{best_kernel_size})"),
    (morph_cleaned, f"8. MODERATE Open ({best_kernel_size}x{best_kernel_size})"),
    (eroded_mask, f"9. Erosion ({chosen_erosion})"),
    (restored_mask, "10. Size Restoration"),
    (restored_mask, "11. Final Mask"),
    (cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB), f"12. Result ({len(final_contours)})")
]

for i, (img, title) in enumerate(pipeline_steps):
    row, col = i // 4, i % 4
    if len(img.shape) == 3:
        axes[row, col].imshow(img)
    else:
        axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(title, fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

print(f"\n🔧 BALANCED Pipeline Parameters:")
print(f"   Gaussian blur: {blur_kernel_size}x{blur_kernel_size}")
print(f"   Canny thresholds: {low_threshold}, {high_threshold}")
print(f"   Dilation: {dilation_kernel_size}x{dilation_kernel_size}, {dilation_iterations} iterations")
print(f"   Closing: {closing_kernel_size}x{closing_kernel_size}, {closing_iterations} iterations")
print(f"   MODERATE cleanup: {best_kernel_size}x{best_kernel_size} (was 12x12)")
print(f"   Erosion/Dilation: {erosion_kernel_size}x{erosion_kernel_size}, {chosen_erosion} iterations")
print(f"   Minimum area: {min_area} pixels")

print(f"\n✅ BALANCED JIGSAW PUZZLE DETECTION COMPLETE!")
print(f"🧩 Detected {len(final_contours)} objects with moderate morphological operations")

# Store results
puzzle_contours = final_contours
puzzle_mask = restored_mask

print(f"\n💾 Results stored in 'puzzle_contours' and 'puzzle_mask'")
print(f"🔧 Balanced approach: keeps morphological ops but reduces them")