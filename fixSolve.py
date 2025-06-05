#!/usr/bin/env python
# coding: utf-8

# In[31]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import math
import bisect


# ### Default image path

# In[32]:


image_paths = [
    "test_images/rgb0.png",    # Photo with red background
    "test_images/rgb1.png",  # Photo with green background
    "test_images/rgb2.png"    # Photo with blue background
]


# ### Display image help function

# In[33]:


def display_image(title, image):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


# ## Basic reading image and display

# In[34]:


## Multi-background image loading and processing

def remove_background_multi_color_notebook(image_paths, variance_threshold=40, debug=True,
                                           skip_morphology=False, light_morphology=True):
    """
    Background removal function optimized for Jupyter notebook integration.
    Returns the same format as your original code expects.

    Args:
        image_paths: List of 3 image paths
        variance_threshold: Threshold for background detection
        debug: Show intermediate steps
        skip_morphology: If True, skip all morphological operations
        light_morphology: If True, use lighter morphology operations
    """

    # Load the three images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
        images.append(img)

    print(f"Loaded {len(images)} images")

    # Ensure all images have the same dimensions
    height, width = images[0].shape[:2]
    for i, img in enumerate(images):
        if img.shape[:2] != (height, width):
            print(f"Resizing image {i} to match dimensions")
            images[i] = cv2.resize(img, (width, height))

    if debug:
        print("Showing input images:")
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.title(f'Image {i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Apply Gaussian blur to reduce noise from lighting variations
    print("Applying noise reduction...")
    blurred_images = []
    for img in images:
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        blurred_images.append(blurred)

    # Calculate color variance at each pixel
    print("Calculating color variance...")

    # Stack images to calculate variance efficiently
    image_stack = np.stack(blurred_images, axis=0)  # Shape: (3, height, width, 3)

    # Calculate variance across the 3 images for each color channel
    color_variance = np.var(image_stack, axis=0)  # Shape: (height, width, 3)

    # Calculate total variance (sum across color channels)
    total_variance = np.sum(color_variance, axis=2)  # Shape: (height, width)

    # Calculate adaptive threshold
    threshold_adaptive = np.percentile(total_variance, 25)  # 25th percentile
    final_threshold = min(variance_threshold, threshold_adaptive * 1.5)

    print(f"Adaptive threshold: {threshold_adaptive:.1f}")
    print(f"Final threshold: {final_threshold:.1f}")

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(total_variance, cmap='hot')
        plt.colorbar()
        plt.title('Color Variance Across Images')

        plt.subplot(1, 2, 2)
        plt.hist(total_variance.flatten(), bins=100, alpha=0.7)
        plt.axvline(final_threshold, color='red', linestyle='--', label=f'Threshold: {final_threshold:.1f}')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.title('Variance Distribution')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Create foreground mask: low variance = foreground, high variance = background
    foreground_mask = (total_variance < final_threshold).astype(np.uint8) * 255

    if debug:
        display_image("Initial Foreground Mask", foreground_mask)

    # Enhanced morphological operations - much lighter and optional
    if skip_morphology:
        print("Skipping all morphological operations")
        morph_image = foreground_mask.copy()
    else:
        print("Applying morphological operations...")

        if light_morphology:
            print("Using light morphology")
            # Much smaller kernels for lighter processing
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_medium = np.ones((5, 5), np.uint8)

            # Light operations
            morph_image = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_small)
            morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel_small)
        else:
            print("Using original heavy morphology")
            # Use similar kernel as your original code
            kernel = np.ones((12, 12), np.uint8)

            # Close = filling the holes
            morph_image = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
            # Open = removing the noise
            morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)

    if debug:
        display_image("Morph Operations", morph_image)

    # Fill holes (optional - can also be skipped)
    if not skip_morphology:
        print("Filling holes in puzzle pieces")
        contours_fill, _ = cv2.findContours(morph_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_fill:
            cv2.drawContours(morph_image, [cnt], 0, 255, -1)

        if debug:
            display_image("Filled Holes", morph_image)

    # Find contours (same as your original)
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} potential puzzle pieces")

    # Filter contours by area (same as your original)
    print("Filtering contours by size")
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if len(contours) > 1:
        reference_area = cv2.contourArea(contours[1])
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > reference_area / 3]
    print(f"After filtering: {len(contours)} puzzle pieces")

    return contours, images[0], morph_image

# Process the images with different morphology options
print("Reading images from paths:", image_paths)

# Option 1: Light morphology (recommended)
contours, original_image, binary_image = remove_background_multi_color_notebook(
    image_paths, variance_threshold=40, debug=True, skip_morphology=False, light_morphology=True
)

# Option 2: Skip all morphology (for testing)
# contours, original_image, binary_image = remove_background_multi_color_notebook(
#     image_paths, variance_threshold=40, debug=True, skip_morphology=True
# )

# Option 3: Heavy morphology (original style)
# contours, original_image, binary_image = remove_background_multi_color_notebook(
#     image_paths, variance_threshold=40, debug=True, skip_morphology=False, light_morphology=False
# )

display_image("Original Image", original_image)


# ## Grayscale conversion

# In[35]:


print("Converting to grayscale")
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
display_image("Grayscale Image", gray_image)


# ### Binary thresholding
# 
# Free parameters:
# - threshold value: `30`
# - maxval: `255`

# ## Morphing
# 
# Used to improve the shape and remove impurities
# 
# Free parameters:
# - kernel ``(12,12)`` // like the size of the morphing

# ## Contours finding

# ## Contours drawing
# 
# Free parameters:
# - thickness: `2`
# - rgb: `(0, 255, 0)`

# In[36]:


print("Drawing contours of the original image")
contour_image = original_image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
display_image("Detected Pieces", contour_image)


# In[37]:


output_folder_pieces = "images/fix/extracted_pieces"
os.makedirs(output_folder_pieces, exist_ok=True)

output_folder_contours = "images/fix/extracted_contours"
os.makedirs(output_folder_contours, exist_ok=True)

output_corner_folder = "images/fix/extracted_corners"
os.makedirs(output_corner_folder, exist_ok=True)

output_plots_folder = "images/fix/corner_plots"
os.makedirs(output_plots_folder, exist_ok=True)


# In[38]:


piece_images = []
for i, contour in enumerate(contours):
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    piece = np.zeros_like(original_image)
    piece[mask == 255] = original_image[mask == 255]

    x, y, w, h = cv2.boundingRect(contour)
    cropped_piece = piece[y:y + h, x:x + w]
    piece_images.append(cropped_piece)

    piece_path = os.path.join(output_folder_pieces, f"piece_{i + 1}.png")
    cv2.imwrite(piece_path, cropped_piece)

    # Save piece with contour
    contour_piece = cropped_piece.copy()
    mask_cropped = mask[y:y + h, x:x + w]
    cv2.drawContours(contour_piece, [contour - [x, y]], 0, (255, 0, 255), 1,
                     cv2.LINE_8)  # Pink, width 1, no anti-aliasing
    contour_path = os.path.join(output_folder_contours, f"contour_{i + 1}.png")
    cv2.imwrite(contour_path, contour_piece)


# ## Corners and Edges detection on single piece

# In[52]:


selected_image_index = 1


# In[53]:


piece_images = []

i = selected_image_index
contour = contours[i]

mask = np.zeros_like(gray_image)
cv2.drawContours(mask, [contour], 0, 255, -1)

display_image("Selected Pieces", mask)

piece = np.zeros_like(original_image)
piece[mask == 255] = original_image[mask == 255]
x, y, w, h = cv2.boundingRect(contour)
cropped_piece = piece[y:y + h, x:x + w]
piece_images.append(cropped_piece)

piece_path = os.path.join(output_folder_pieces, f"piece_{i + 1}.png")
cv2.imwrite(piece_path, cropped_piece)
contour_piece = cropped_piece.copy()

display_image(f"Cropped piece {selected_image_index+1}", contour_piece)


# ## Find the piece center
# 
# Free parameters:
# - thickness: `2`
# - rgb: `(0, 0, 255)`
# - circle pos: $(cx, cy)

# In[54]:


M = cv2.moments(contour)

if M["m00"] == 0:
    print("ERROR")
centroid_x = int(M["m10"] / M["m00"])
centroid_y = int(M["m01"] / M["m00"])
cv2.circle(contour_piece, (centroid_x - x, centroid_y - y), 2, (0, 0, 255), -1)
display_image(f"Centroid {i+1}", contour_piece)


# ## Rainbow contour ( for testing )

# In[55]:


def rainbow_color(t):
    t = t % 256  # wrap around after 255
    phase = t / 256 * 6
    x = int(255 * (1 - abs(phase % 2 - 1)))
    if phase < 1:     return (255, x, 0)       # Red → Yellow
    elif phase < 2:   return (x, 255, 0)       # Yellow → Green
    elif phase < 3:   return (0, 255, x)       # Green → Cyan
    elif phase < 4:   return (0, x, 255)       # Cyan → Blue
    elif phase < 5:   return (x, 0, 255)       # Blue → Magenta
    else:             return (255, 0, x)


# In[56]:


rainbow_piece = contour_piece.copy()

print(x, y)

contour_points = contour - np.array([x, y])
counter = 0
for point in contour:

    px, py = point[0]
    rainbow_piece[py - y, px -  x] = (rainbow_color(counter))
    counter += 1

display_image(f"Rainbow Contour {i+1}", rainbow_piece)


# ## Transform the contour into polar coordinates

# ***Note: inverted angles in the plot -> for clockwise plot***

# In[57]:


contour_points = contour - np.array([x, y])
distances = []
angles = []
for point in contour:
    px, py = point[0]
    dx = px - centroid_x
    dy = py - centroid_y
    distance = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    distances.append(distance)
    angles.append(angle)

angles_deg = np.array([(a * 180 / np.pi) % 360 for a in angles])

plt.polar(-np.array(angles), distances)
plt.title("Radial Distance Profile")
plt.show()


# # Data reduction

# In[58]:


# angles_deg = np.roll(angles_deg, -len(angles_deg) // 2)
# distances = np.roll(distances, -len(distances) // 2)

distances = gaussian_filter1d(distances, sigma=2)


# free parameters:
# - marker: `.`

# In[59]:


plt.style.use('default')
plt.scatter(angles_deg, distances, marker='.')
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance from centroid")
plt.title("Radial Distance vs. Angle")
plt.grid(True)

output_test = "images/fix/tests"
os.makedirs(output_test, exist_ok=True)

plt.savefig(os.path.join(output_test, f"distance_profile_{i + 1}.png"))

plt.show()


# ## Peaks detection
# 
# Free parameters:
# - ``min_distance_between_peaks``
# - ``prominence``

# In[60]:


delta_s = len(angles_deg) // 4

angles_deg_s = np.roll(angles_deg, -delta_s)
distances_s = np.roll(distances, -delta_s)

# find the maxima
min_distance_between_peaks = len(distances) // 20
# print(min_distance_between_peaks, len(distances)) # for data analysis
all_peak_indices, all_peak_properties = find_peaks(distances,                                 distance=min_distance_between_peaks,prominence=2)

all_peak_indices_s, all_peak_properties_s = find_peaks(distances_s,                                 distance=min_distance_between_peaks,prominence=2)

all_peak_indices_u = [(x+delta_s)%len(angles_deg) for x in all_peak_indices_s]
all_peak_indices_f = [y for y in all_peak_indices_u if y not in all_peak_indices]

for val in all_peak_indices_f:
    pos = np.searchsorted(all_peak_indices, val)
    all_peak_indices = np.insert(all_peak_indices, pos, val)

# print(all_peak_properties)
peak_angles = [angles_deg[peak_idx] for peak_idx in all_peak_indices]
peak_distances = [distances[peak_idx] for peak_idx in all_peak_indices]

# find the minima
inverted_distances = [-d for d in distances]
all_min_indices, all_min_properties = find_peaks(inverted_distances,
                                 distance=min_distance_between_peaks,prominence=0.1)

min_angles = [angles_deg[min_idx] for min_idx in all_min_indices]
min_distances = [distances[min_idx] for min_idx in all_min_indices]

min_distances_avg = np.mean(min_distances)


# In[61]:


print(all_peak_indices)
print(all_peak_indices_s)
print(all_peak_indices_u)
print(all_peak_indices_f)


# ### Peaks table display
# 
# note: before etra peak removal
# 

# In[62]:


df = pd.DataFrame({
    'Index': all_peak_indices,         # Peak indices
    'Angle (degrees)': peak_angles,    # Peak angles
    'Distance': peak_distances         # Peak distances
})

df


# In[63]:


plt.style.use('default')
plt.plot(angles_deg, distances, marker='.')
plt.scatter(peak_angles, peak_distances, color='red', marker='o', label='Peaks')
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance from centroid")
plt.title("Radial Distance vs. Angle with Peaks")
plt.grid(True)
plt.show()


# ## Minima table display

# In[64]:


df2 = pd.DataFrame({
    'Index': all_min_indices,         # Peak indices
    'Angle (degrees)': min_angles,    # Peak angles
    'Distance': min_distances         # Peak distances
})

df2


# In[65]:


plt.style.use('default')
plt.scatter(angles_deg, distances, marker='.')
plt.scatter(min_angles, min_distances, color='orange', marker='o', label='Peaks')
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance from centroid")
plt.title("Radial Distance vs. Angle with Peaks")
plt.grid(True)
plt.show()


# ## Remove the extra peaks
# 
# Free parameters:
# - ``max_angle_diff``

# In[66]:


max_angle_diff = 25
delCounter = 0

remaining_peak_indices = all_peak_indices.tolist() if isinstance(all_peak_indices, np.ndarray) else all_peak_indices.copy()
remaining_peak_angles = peak_angles.tolist() if isinstance(peak_angles, np.ndarray) else peak_angles.copy()
remaining_peak_distances = peak_distances.tolist() if isinstance(peak_distances, np.ndarray) else peak_distances.copy()

while True:
    removals_made = False

    for k in range(len(remaining_peak_angles)):
        if len(remaining_peak_angles) <= 1:
            break

        if remaining_peak_distances[k] <= min_distances_avg:
            print("removing a bottom peak")
            remaining_peak_indices.pop(k)
            remaining_peak_angles.pop(k)
            remaining_peak_distances.pop(k)
            break


        next_k = (k + 1) % len(remaining_peak_angles)  # Wrap-around logic

        if abs(remaining_peak_angles[next_k] - remaining_peak_angles[k]) < max_angle_diff:
            delCounter += 1
            removals_made = True

            if remaining_peak_distances[k] < remaining_peak_distances[next_k]:
                remaining_peak_indices.pop(k)
                remaining_peak_angles.pop(k)
                remaining_peak_distances.pop(k)
            else:
                remaining_peak_indices.pop(next_k)
                remaining_peak_angles.pop(next_k)
                remaining_peak_distances.pop(next_k)

            break


    if not removals_made or len(remaining_peak_angles) <= 1:
        break

all_peak_indices = np.array(remaining_peak_indices) if isinstance(all_peak_indices, np.ndarray) else remaining_peak_indices
peak_angles = np.array(remaining_peak_angles) if isinstance(peak_angles, np.ndarray) else remaining_peak_angles
peak_distances = np.array(remaining_peak_distances) if isinstance(peak_distances, np.ndarray) else remaining_peak_distances

# Print out the result
print(f"We deleted {delCounter} extra peak angles")


# ## Drawing the maxima

# In[67]:


plt.style.use('default')
plt.scatter(angles_deg, distances, marker='.')
plt.scatter(peak_angles, peak_distances, color='red', marker='o', label='Peaks')
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance from centroid")
plt.title("Radial Distance vs. Angle with Peaks")
plt.grid(True)
plt.show()


# ## Drawing the minima

# In[68]:


print(all_min_indices)
print(min_distances_avg)


# In[69]:


# new_min_indice = []
#
# for min_indice in all_min_indices:
#     if distances[min_indice] < min_distances_avg:
#         new_min_indice.append(min_indice)
#
# all_min_indices = np.array(new_min_indice)


new_min_indice = all_min_indices.copy()


# In[70]:


min_angles = [angles_deg[min_idx] for min_idx in new_min_indice]
min_distances = [distances[min_idx] for min_idx in new_min_indice]


# In[71]:


plt.style.use('default')
plt.scatter(angles_deg, distances, marker='.')
plt.scatter(peak_angles, peak_distances, color='red', marker='o', label='Peaks')
plt.scatter(min_angles, min_distances, color='orange', marker='o', label='Minima')
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance from centroid")
plt.title("Radial Distance vs. Angle with Peaks")
plt.grid(True)
plt.show()

# [52 146 255 385 464 565 655]


# ## Find the 4 corners
# 
# They are defined as the top4 most pointy minima

# In[72]:


def robust_pointiness(angles_deg, distances, peak_indices, window_small=5, window_large=15):
    pointiness_scores = []

    for peak_idx in peak_indices:
        small_left = max(0, peak_idx - window_small)
        small_right = min(len(distances) - 1, peak_idx + window_small)
        small_window = distances[small_left:small_right+1]

        large_left = max(0, peak_idx - window_large)
        large_right = min(len(distances) - 1, peak_idx + window_large)
        large_window = distances[large_left:large_right+1]

        peak_height = distances[peak_idx]
        small_avg = np.mean(small_window)
        large_avg = np.mean(large_window)

        pointiness = (peak_height - large_avg) * (small_avg - large_avg)
        pointiness_scores.append(pointiness)

    return pointiness_scores

pointiness_scores = robust_pointiness(angles_deg, distances, all_peak_indices)

top_4_indices = sorted(range(len(pointiness_scores)),
                      key=lambda i: pointiness_scores[i],
                      reverse=True)[:4]

top_4_indices = sorted(top_4_indices)
top_4_peak_indices = [all_peak_indices[i] for i in top_4_indices]

temp = [int(w) for w in top_4_peak_indices]
remaining_indices = [z for z in all_peak_indices if z not in temp]

top_4_angles = [angles_deg[i] for i in top_4_peak_indices]
top_4_distances = [distances[i] for i in top_4_peak_indices]

print(all_peak_indices)
print(top_4_indices)
print(top_4_peak_indices)
print(temp)
print(remaining_indices)
print(top_4_angles)
print(top_4_distances)

plt.figure(figsize=(10, 6))
plt.scatter(angles_deg, distances, marker='.', alpha=0.5, label='All Points')
plt.scatter(peak_angles, peak_distances, color='red', marker='o', label='All Peaks')
plt.scatter(top_4_angles, top_4_distances, color='green', marker='*', s=200, label='Top 4 Pointy Peaks')
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance from centroid")
plt.title("Radial Distance vs. Angle with Top 4 Pointiest Peaks")
plt.legend()
plt.grid(True)
plt.show()

# Print pointiness scores for all peaks
for i, (angle, distance, score) in enumerate(zip(peak_angles, peak_distances, pointiness_scores)):
    print(f"Peak {i+1}: Angle = {angle:.2f}°, Distance = {distance:.2f}, Pointiness = {score:.2f}")
print("\nTop 4 pointiest peaks:")
for i, idx in enumerate(top_4_indices):
    print(f"Peak {idx+1}: Angle = {peak_angles[idx]:.2f}°, Distance = {peak_distances[idx]:.2f}, Pointiness = {pointiness_scores[idx]:.2f}")


# ## Draw the 4 corners

# In[73]:


contour_piece_with_peaks = contour_piece.copy()

cv2.circle(contour_piece_with_peaks, (centroid_x - x, centroid_y - y), 2, (0, 0, 255), -1)

peak_coords = []
for angle, distance in zip(top_4_angles, top_4_distances):
    angle_rad = np.radians(angle)

    dx = int(distance * np.cos(angle_rad))
    dy = int(distance * np.sin(angle_rad))

    peak_x = (centroid_x - x) + dx
    peak_y = (centroid_y - y) + dy

    cv2.circle(contour_piece_with_peaks, (peak_x, peak_y), 3, (0, 255, 0), -1)
    peak_coords.append((peak_x, peak_y))

display_image(f"Centroid and Top 4 Peaks {selected_image_index+1}", contour_piece_with_peaks)

print("Peak coordinates (x, y):")
for idx, (px, py) in enumerate(peak_coords):
    print(f"Peak {idx+1}: ({px}, {py})")


# In[74]:


class puzzlePiece:
    def __init__(self, piece_id, contour, peaks_ids, center, angles, distances):
        self.piece_id = piece_id
        self.contour = contour
        self.corners = peaks_ids
        self.center = center
        self.contour_polar = (angles, distances)
        self.edges = []

    def __repr__(self):
        ret = ""
        ret += f"{self.piece_id}\n"
        ret += f"Contour: {len(self.contour)}\n"
        ret += f"Corners: {self.corners}\n"

        return ret

puzzle_pieces_holder = {}

this_piece = puzzlePiece(
    1,
    contour,
    top_4_peak_indices,
    (centroid_x - x, centroid_y - y),
    angles_deg,
    distances
)

print(this_piece)


# ## Edge type detection

# In[75]:


def exists_peak_between(a,b,peak_indices):
    if b < a:
        for p in peak_indices:
            if p > a or p < b:
                print(f"Found maxima {p}")
                return True

    else:
        for p in peak_indices:
            if p > a and p < b:
                print(f"Found maxima {p}")
                return True

        print("no peak between")
        return False


# In[76]:


def exists_minima_between(a,b,min_indices):
    if b < a:
        for p in min_indices:
            if p > a or p < b:
                # hack to set distance smaller than custom threshold
                if distances[p] < ((max(distances[a], distances[b])*0.8 ) /1.41):
                    print(f"Found minima {p} with distance {distances[p]}")
                    return True
                else:
                    return False

    else:
        for p in min_indices:
            if p > a and p < b:
                if distances[p] < ((max(distances[a], distances[b])*0.8 ) /1.41):
                    print(f"Found minima {p} with distance {distances[p]}")
                    return True
                else:
                    return False

        print("no min between")
        return False


# In[77]:


edge_types = {0: "FLAT", 1: "IN", 2: "OUT"}
print(all_min_indices)

def get_edge_type(a, b):
    print("-----")
    print(a, b)
    print(angles_deg[a], angles_deg[b])

    print(remaining_indices)
    print([angles_deg[r] for r in remaining_indices])
    if exists_peak_between(a, b, remaining_indices):
        return 2
    else:
        if exists_minima_between(a, b, all_min_indices):
            return 1
        else:
            return 0


# In[78]:


class Edge:
    def __init__(self, edge_type, left_corner, right_corner):
        self.edge_type = edge_type
        self.left_corner = left_corner
        self.right_corner = right_corner


# In[79]:


edges = []

for c, corner in enumerate(top_4_peak_indices):
    corner1 = top_4_peak_indices[c]
    corner2 = top_4_peak_indices[(c+1) % len(top_4_peak_indices)]

    edges.append((c, corner1, corner2, get_edge_type(corner1, corner2)))

this_piece.edges = edges

print(edges)


# In[80]:


for edge_id, start_idx, end_idx, edge_type in edges:
    print(angles_deg[start_idx], angles_deg[end_idx])


# ## Draw edges in a different color ( testing )

# In[81]:


color_edge_piece = contour_piece.copy()

contour_points = contour.reshape(-1, 2)
contour_points_shifted = contour_points - np.array([x, y])  # apply shift

edge_colors = [
    (0, 255, 255),   # Cyan
    (255, 165, 0),   # Orange
    (0, 255, 0),     # Green
    (255, 105, 180)  # Pink
]

for edge_id, start_idx, end_idx, edge_type in edges:
    color = edge_colors[edge_id]
    if start_idx < end_idx:
        indices = range(start_idx, end_idx)
    else:
        indices = list(range(start_idx, len(contour_points))) + list(range(0, end_idx))

    for idx in indices:
        px, py = contour_points_shifted[idx]
        color_edge_piece[py, px] = color

display_image(f"Colored Edges", color_edge_piece)


# ## Draw edges types with Corners

# In[82]:


# angles_deg = np.roll(angles_deg, +len(angles_deg) // 4)
# distances = np.roll(distances, +len(distances) // 4)


# In[83]:


color_edge_corner_piece = contour_piece.copy()

contour_points = contour.reshape(-1, 2)
contour_points_shifted = contour_points - np.array([x, y])  # apply shift

edge_colors = {
    0: (0, 0, 255),   # Blue for edge_type 0
    1: (255, 255, 0), # Yellow for edge_type 1
    2: (0, 255, 0)    # Green for edge_type 2
}

corner_color = (255, 0, 255)  # Purple color for corners
for edge_id, start_idx, end_idx, edge_type in edges:
    start_px, start_py = contour_points[start_idx] - np.array([x, y])
    end_px, end_py = contour_points[end_idx] - np.array([x, y])

    color = edge_colors[edge_type]
    if start_idx < end_idx:
        indices = range(start_idx, end_idx)
    else:
        indices = list(range(start_idx, len(contour_points))) + list(range(0, end_idx))

    for idx in indices:
        px, py = contour_points_shifted[idx]
        color_edge_corner_piece[py, px] = color

    cv2.circle(color_edge_corner_piece, (start_px, start_py), 3, corner_color, -1)  # Start corner
    cv2.circle(color_edge_corner_piece, (end_px, end_py), 3, corner_color, -1)    # End corner

display_image(f"Edges with Corners types {selected_image_index+1}", color_edge_corner_piece)


# In[ ]:





# In[ ]:




