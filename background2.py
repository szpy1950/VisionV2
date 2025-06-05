# COMBINATION METHOD: Illumination Correction + Edge Detection
# Method 3 got you 45/49 pieces - let's try to find those missing 4!

# ========================================
# CELL 1: Setup (same as before)
# ========================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
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

# Load your image
image_path = "images/p.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

display_image("Original Image", original_image)

# ========================================
# CELL 2: Method 3 - Illumination Correction (Your Best Method)
# ========================================
print("🎯 STEP 1: Illumination Correction (Method 3)")
print("This found 45/49 pieces - let's reproduce this result first")

# Illumination correction (same as Method 3)
kernel_size = 50
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
corrected = cv2.add(cv2.subtract(gray_image, blackhat), tophat)
display_image("Illumination Corrected", corrected)

# Otsu threshold
_, binary_illumination = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
display_image("Illumination Binary", binary_illumination)

# Morphological operations
morph_kernel = np.ones((12, 12), np.uint8)
morph_illumination = cv2.morphologyEx(binary_illumination, cv2.MORPH_CLOSE, morph_kernel)
morph_illumination = cv2.morphologyEx(morph_illumination, cv2.MORPH_OPEN, morph_kernel)

# Fill holes
contours_fill, _ = cv2.findContours(morph_illumination, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_fill:
    cv2.drawContours(morph_illumination, [cnt], 0, 255, -1)

display_image("Method 3 Final Result", morph_illumination)

# ========================================
# CELL 3: Method 4 - Edge Detection (Complementary Method)
# ========================================
print("🎯 STEP 2: Edge Detection (Method 4)")
print("This will find pieces that illumination correction missed")

# Edge detection
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
display_image("Canny Edges", edges)

# Thicken edges
edge_kernel = np.ones((3, 3), np.uint8)
thick_edges = cv2.dilate(edges, edge_kernel, iterations=2)

# Close gaps
close_kernel = np.ones((5, 5), np.uint8)
closed_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, close_kernel, iterations=3)

# Fill regions
contours_temp, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
binary_edges = np.zeros_like(gray_image)
cv2.drawContours(binary_edges, contours_temp, -1, 255, thickness=cv2.FILLED)
display_image("Edge Detection Result", binary_edges)

# Apply same morphological operations as illumination method
morph_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, morph_kernel)
morph_edges = cv2.morphologyEx(morph_edges, cv2.MORPH_OPEN, morph_kernel)

# Fill holes
contours_fill, _ = cv2.findContours(morph_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_fill:
    cv2.drawContours(morph_edges, [cnt], 0, 255, -1)

display_image("Method 4 Final Result", morph_edges)

# ========================================
# CELL 4: COMBINATION APPROACH 1 - Union (OR Operation)
# ========================================
print("🚀 COMBINATION 1: Union of Both Methods")
print("Combines areas found by EITHER method")

# Combine using OR operation
combined_union = cv2.bitwise_or(morph_illumination, morph_edges)
display_image("Combined Union Result", combined_union)

# Get contours from union
contours_union, _ = cv2.findContours(combined_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_union = sorted(contours_union, key=lambda x: cv2.contourArea(x), reverse=True)

# Filter by area
if len(contours_union) > 1:
    reference_area = cv2.contourArea(contours_union[1])
    contours_union = [cnt for cnt in contours_union if cv2.contourArea(cnt) > reference_area / 3]

result_union = original_image.copy()
cv2.drawContours(result_union, contours_union, -1, (0, 255, 0), 2)
display_image(f"COMBINATION 1 RESULT: {len(contours_union)} pieces", result_union)

# ========================================
# CELL 5: COMBINATION APPROACH 2 - Weighted Combination
# ========================================
print("🚀 COMBINATION 2: Weighted Combination")
print("Gives more weight to illumination method (since it worked better)")

# Weighted combination: 70% illumination + 30% edges
weight_illumination = 0.7
weight_edges = 0.3

# Convert to float for weighted combination
illumination_float = morph_illumination.astype(np.float32) / 255.0
edges_float = morph_edges.astype(np.float32) / 255.0

# Weighted combination
combined_weighted = weight_illumination * illumination_float + weight_edges * edges_float
combined_weighted = (combined_weighted * 255).astype(np.uint8)

# Apply threshold to get binary result
_, combined_weighted_binary = cv2.threshold(combined_weighted, 127, 255, cv2.THRESH_BINARY)
display_image("Weighted Combination Result", combined_weighted_binary)

# Get contours from weighted combination
contours_weighted, _ = cv2.findContours(combined_weighted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_weighted = sorted(contours_weighted, key=lambda x: cv2.contourArea(x), reverse=True)

# Filter by area
if len(contours_weighted) > 1:
    reference_area = cv2.contourArea(contours_weighted[1])
    contours_weighted = [cnt for cnt in contours_weighted if cv2.contourArea(cnt) > reference_area / 3]

result_weighted = original_image.copy()
cv2.drawContours(result_weighted, contours_weighted, -1, (0, 255, 0), 2)
display_image(f"COMBINATION 2 RESULT: {len(contours_weighted)} pieces", result_weighted)

# ========================================
# CELL 6: COMBINATION APPROACH 3 - Smart Addition
# ========================================
print("🚀 COMBINATION 3: Smart Addition")
print("Add edges only where illumination method has gaps")

# Find areas where illumination method failed (gaps in pieces)
# Erode illumination result to find core areas
erode_kernel = np.ones((8, 8), np.uint8)
illumination_core = cv2.erode(morph_illumination, erode_kernel, iterations=2)

# Dilate back to find confident areas
illumination_confident = cv2.dilate(illumination_core, erode_kernel, iterations=2)

# Find uncertain areas (difference between original and confident)
illumination_uncertain = cv2.subtract(morph_illumination, illumination_confident)
display_image("Uncertain Areas from Illumination", illumination_uncertain)

# In uncertain areas, use edge detection
edges_in_uncertain = cv2.bitwise_and(morph_edges, illumination_uncertain)
display_image("Edges in Uncertain Areas", edges_in_uncertain)

# Combine confident illumination + edges in uncertain areas
combined_smart = cv2.bitwise_or(illumination_confident, edges_in_uncertain)
display_image("Smart Combination Result", combined_smart)

# Get contours from smart combination
contours_smart, _ = cv2.findContours(combined_smart, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_smart = sorted(contours_smart, key=lambda x: cv2.contourArea(x), reverse=True)

# Filter by area
if len(contours_smart) > 1:
    reference_area = cv2.contourArea(contours_smart[1])
    contours_smart = [cnt for cnt in contours_smart if cv2.contourArea(cnt) > reference_area / 3]

result_smart = original_image.copy()
cv2.drawContours(result_smart, contours_smart, -1, (0, 255, 0), 2)
display_image(f"COMBINATION 3 RESULT: {len(contours_smart)} pieces", result_smart)

# ========================================
# CELL 7: COMPARISON OF ALL COMBINATIONS
# ========================================
print("📊 COMPARISON OF COMBINATION METHODS:")

# Get individual method results for comparison
contours_illumination, _ = cv2.findContours(morph_illumination, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_illumination = sorted(contours_illumination, key=lambda x: cv2.contourArea(x), reverse=True)
if len(contours_illumination) > 1:
    reference_area = cv2.contourArea(contours_illumination[1])
    contours_illumination = [cnt for cnt in contours_illumination if cv2.contourArea(cnt) > reference_area / 3]

contours_edges_only, _ = cv2.findContours(morph_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_edges_only = sorted(contours_edges_only, key=lambda x: cv2.contourArea(x), reverse=True)
if len(contours_edges_only) > 1:
    reference_area = cv2.contourArea(contours_edges_only[1])
    contours_edges_only = [cnt for cnt in contours_edges_only if cv2.contourArea(cnt) > reference_area / 3]

print(f"Method 3 alone (Illumination): {len(contours_illumination)} pieces")
print(f"Method 4 alone (Edges): {len(contours_edges_only)} pieces")
print(f"Combination 1 (Union): {len(contours_union)} pieces")
print(f"Combination 2 (Weighted): {len(contours_weighted)} pieces")
print(f"Combination 3 (Smart): {len(contours_smart)} pieces")
print(f"🎯 Target: 49 pieces")

# Show all results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

results = [
    (result_union, f'Union: {len(contours_union)} pieces'),
    (result_weighted, f'Weighted: {len(contours_weighted)} pieces'),
    (result_smart, f'Smart: {len(contours_smart)} pieces'),
    (original_image.copy(), 'Original'),
    (original_image.copy(), ''),
    (original_image.copy(), '')
]

# Draw individual methods on last two slots for reference
temp_illum = original_image.copy()
cv2.drawContours(temp_illum, contours_illumination, -1, (0, 255, 0), 2)
results[3] = (temp_illum, f'Illumination: {len(contours_illumination)}')

temp_edges = original_image.copy()
cv2.drawContours(temp_edges, contours_edges_only, -1, (0, 255, 0), 2)
results[4] = (temp_edges, f'Edges: {len(contours_edges_only)}')

for i, (img, title) in enumerate(results):
    row, col = i // 3, i % 3
    if i < 5:  # Only show first 5
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(title)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# ========================================
# CELL 8: SELECT BEST COMBINATION
# ========================================
print("✅ SELECT YOUR BEST COMBINATION:")

# Choose the best combination (uncomment one):
# final_contours = contours_union     # Combination 1: Union
# final_contours = contours_weighted  # Combination 2: Weighted
# final_contours = contours_smart     # Combination 3: Smart

# Default to the one closest to 49
results_counts = [
    len(contours_union), len(contours_weighted), len(contours_smart)
]
best_idx = min(range(len(results_counts)), key=lambda i: abs(results_counts[i] - 49))

if best_idx == 0:
    final_contours = contours_union
    method_name = "Union"
elif best_idx == 1:
    final_contours = contours_weighted
    method_name = "Weighted"
else:
    final_contours = contours_smart
    method_name = "Smart"

print(f"🎯 Auto-selected: {method_name} method with {len(final_contours)} pieces")
print(f"📈 Improvement: {len(contours_illumination)} → {len(final_contours)} pieces")

# Final result
final_result = original_image.copy()
cv2.drawContours(final_result, final_contours, -1, (0, 255, 0), 3)
display_image(f"FINAL COMBINED RESULT: {len(final_contours)} pieces", final_result)

# ========================================
# CELL 9: FINE-TUNING PARAMETERS
# ========================================
print("🔧 FINE-TUNING FOR EVEN BETTER RESULTS:")
print()
print("To get from 45 to 49 pieces, try adjusting:")
print()
print("For Illumination Correction (Cell 2):")
print("   kernel_size: 40, 45, 55, 60 (current: 50)")
print()
print("For Edge Detection (Cell 3):")
print("   Canny thresholds: (40,120), (60,180) (current: 50,150)")
print("   close_kernel iterations: 2, 4, 5 (current: 3)")
print()
print("For Weighted Combination (Cell 5):")
print("   weight_illumination: 0.6, 0.8 (current: 0.7)")
print("   weight_edges: 0.2, 0.4 (current: 0.3)")

print(f"\n🚀 READY! Use 'final_contours' for corner detection")
print(f"🎉 Combined method found {len(final_contours)} pieces!")

# Ready for corner detection
contours = final_contours