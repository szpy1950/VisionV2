#!/usr/bin/env python
# coding: utf-8

# In[594]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import math


# In[595]:


DEBUG = True


# In[596]:


base_path = "images/exam"
image_path = "images/test_images/image.jpeg"


# In[597]:


def display_image(title, image):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if (DEBUG):
        plt.show()


# In[598]:


def remove_background_canny_edge_detection(image_path, debug=True):
    # Load single image instead of multiple images
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    if debug:
        print("🔧 CANNY EDGE JIGSAW PUZZLE DETECTION")
        print("=" * 60)
        display_image("Original Image", original_image)
        display_image("Grayscale Image", gray_image)

    # STEP 1: Preprocessing - Gaussian blur for noise reduction
    if debug:
        print("🔧 STEP 1: Preprocessing - Gaussian blur for noise reduction")

    blur_kernel_size = 5  # Must be odd number
    blurred = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)

    if debug:
        display_image("After Gaussian Blur", blurred)
        print(f"✅ Applied {blur_kernel_size}x{blur_kernel_size} Gaussian blur")

    # STEP 2: Canny edge detection
    if debug:
        print("🎯 STEP 2: Canny edge detection")

    low_threshold = 10  # Lower threshold for edge linking
    high_threshold = 150  # Upper threshold for strong edges

    canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)
    #
    if debug:
        display_image("Canny Edges", canny_edges)
        edge_pixels = np.count_nonzero(canny_edges)
        total_pixels = canny_edges.shape[0] * canny_edges.shape[1]
        edge_percentage = (edge_pixels / total_pixels) * 100
        print(f"📊 Canny edge statistics:")
        print(f"   Low threshold: {low_threshold}")
        print(f"   High threshold: {high_threshold}")
        print(f"   Edge pixels: {edge_pixels:,} ({edge_percentage:.2f}%)")

    # STEP 3: Dilation - thicken edges
    if debug:
        print("📈 STEP 3: Dilation - thicken edges")

    dilation_kernel_size = 4
    dilation_iterations = 2
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)

    dilated_edges = cv2.dilate(canny_edges, dilation_kernel, iterations=dilation_iterations)

    if debug:
        display_image("After Dilation (Thickened Edges)", dilated_edges)
        print(f"📊 Dilation parameters:")
        print(f"   Kernel size: {dilation_kernel_size}x{dilation_kernel_size}")
        print(f"   Iterations: {dilation_iterations}")

    # STEP 4: Morphological closing - fill gaps between edges
    if debug:
        print("🔗 STEP 4: Morphological closing - fill gaps between edges")

    closing_kernel_size = 3
    closing_iterations = 3
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)

    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)

    if debug:
        display_image("After Morphological Closing", closed_edges)
        print(f"📊 Closing parameters:")
        print(f"   Kernel size: {closing_kernel_size}x{closing_kernel_size}")
        print(f"   Iterations: {closing_iterations}")

    # STEP 5: Contour detection and filling
    if debug:
        print("🎨 STEP 5: Contour detection and filling")

    contours_temp, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"Found {len(contours_temp)} contours")

    filled_mask = np.zeros_like(gray_image)
    cv2.drawContours(filled_mask, contours_temp, -1, 255, thickness=cv2.FILLED)

    if debug:
        display_image("Filled Contours", filled_mask)

    # STEP 6: MODERATE morphological operations for cleanup
    if debug:
        print("🧹 STEP 6: MODERATE morphological operations for cleanup")

    cleanup_kernel_size = 6  # Reduced from 12 to 6
    cleanup_kernel = np.ones((cleanup_kernel_size, cleanup_kernel_size), np.uint8)

    morph_closed = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)
    morph_cleaned = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)

    if debug:
        display_image("After MODERATE Morphological Close", morph_closed)
        display_image("After MODERATE Morphological Open", morph_cleaned)
        print(f"📊 MODERATE cleanup parameters:")
        print(f"   Cleanup kernel size: {cleanup_kernel_size}x{cleanup_kernel_size}")
        print(f"   Iterations: 1 each")

    # STEP 7: Erosion - reduce object size
    if debug:
        print("📉 STEP 7: Erosion - reduce object size")

    erosion_kernel_size = 3
    chosen_erosion = 2
    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(morph_cleaned, erosion_kernel, iterations=chosen_erosion)

    if debug:
        display_image("After Erosion", eroded_mask)

    # STEP 8: Dilation - restore original size
    if debug:
        print("📈 STEP 8: Dilation - restore original size")

    restoration_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    restored_mask = cv2.dilate(eroded_mask, restoration_kernel, iterations=chosen_erosion)

    if debug:
        display_image("After Size Restoration (Erosion + Dilation)", restored_mask)

    # STEP 8.5: Final shrinking - reduce dilation slightly
    if debug:
        print("🔧 STEP 8.5: Final shrinking - reduce dilation slightly")

    shrink_kernel = np.ones((10, 10), np.uint8)  # Small kernel for slight shrinking
    final_mask = cv2.erode(restored_mask, shrink_kernel, iterations=1)

    if debug:
        display_image("After Final Shrinking", final_mask)

    # STEP 9: Final contour extraction
    # if debug:
    #     print("🎯 STEP 9: Final results and analysis")

    final_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter by area (remove tiny fragments) - same as original code
    min_area = 200
    final_contours = [cnt for cnt in final_contours if cv2.contourArea(cnt) > min_area]

    # Sort contours by area (same as original code)
    final_contours = sorted(final_contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Additional filtering similar to original code
    if len(final_contours) > 1:
        reference_area = cv2.contourArea(final_contours[1]) if len(final_contours) > 1 else cv2.contourArea(final_contours[0])
        final_contours = [cnt for cnt in final_contours if cv2.contourArea(cnt) > reference_area / 3]

    if debug:
        print(f"📊 Final Results:")
        print(f"   Objects detected: {len(final_contours)}")
        print(f"After filtering: {len(final_contours)} puzzle pieces")

    return final_contours, original_image, final_mask


# In[599]:


print("Reading image from path:", image_path)

contours, original_image, binary_image = remove_background_canny_edge_detection(
    image_path, debug=True
)

# Convert to grayscale for compatibility with rest of code
print("Converting to grayscale")
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
display_image("Grayscale Image", gray_image)


# In[600]:


print("Drawing contours of the original image")
contour_image = original_image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
display_image("Detected Pieces", contour_image)


# In[601]:


output_folder_pieces = f"{base_path}/extracted_pieces"
os.makedirs(output_folder_pieces, exist_ok=True)

output_folder_contours = f"{base_path}/extracted_contours"
os.makedirs(output_folder_contours, exist_ok=True)

output_corner_folder = f"{base_path}/extracted_corners"
os.makedirs(output_corner_folder, exist_ok=True)

output_transparent_folder = f"{base_path}/transparent_pieces"
os.makedirs(output_transparent_folder, exist_ok=True)

output_plots_folder = f"{base_path}/corner_plots"
os.makedirs(output_plots_folder, exist_ok=True)

output_canvas = f"{base_path}/canvas"
os.makedirs(output_canvas, exist_ok=True)

output_oriented_folder = f"{base_path}/oriented_pieces"
os.makedirs(output_oriented_folder, exist_ok=True)

output_edges_and_corner_folder = f"{base_path}/edges_and_corners"
os.makedirs(output_edges_and_corner_folder, exist_ok=True)

output_matching = f"{base_path}/matching"
os.makedirs(output_matching, exist_ok=True)

output_tests = f"{base_path}/tests"
os.makedirs(output_tests, exist_ok=True)


# In[602]:


class Edge:
    def __init__(self, edge_id, edge_type, left_corner, right_corner):
        self.edge_type = edge_type
        self.edge_id = edge_id
        self.left_corner = left_corner
        self.right_corner = right_corner
        self.contour_indices = []
        self.length = None
        self.orientation = None
        self.color_edge_line = None


# In[603]:


class puzzlePiece:
    def __init__(self, piece_id):
        self.piece_id = piece_id
        self.absolute_contour = None
        self.relative_contour = None
        self.corners = None
        self.absolute_center = None
        self.relative_center = None
        self.contour_polar = None
        self.rotation_matrix = None
        self.edges = None
        self.edges_cat = None
        self.edges_ord = None
        self.piece_type = None
        self.mask = None
        self.bounding_rect = None
        self.absolute_coords = None
        self.relative_coords = None
        self.color_stats = None
        self.cropped_image = None
        self.relative_mask = None
        self.transparent_image = None

        self.edge_direction_map = None
        self.final_edge_direction_map = None

        self.matching_edges = {}
        self.sorted_score = None

        self.flat_edge_id = None

        self.rotation_angle = None
        self.rotation_matrix = None
        self.expanded_image = None
        self.expansion_offset = None  # (x_offset, y_offset)
        self.expansion_ratio = None
        self.rotated_image = None
        self.rotated_center = None
        self.rotated_corners = []  # List of rotated corner coordinates
        self.rotated_edges = []  # List of rotated edge data
        self.rotated_contour = None  # Full rotated contour
        self.flat_edge_id = None  # Which edge ID is the flat one being rotated


    def __repr__(self):
        ret = ""
        ret += f"{self.piece_id}\n"
        ret += f"Contour: {len(self.contour)}\n"
        ret += f"Corners: {self.corners}\n"

        return ret


# In[604]:


class Puzzle:
    _counter = 0
    def __init__(self):
        self.puzzle_id = Puzzle._counter
        Puzzle._counter += 1
        self.all_pieces = {}
        self.corners_pieces = []
        self.borders_pieces = []
        self.middle_pieces = []


# In[605]:


my_puzzle = Puzzle()


# In[606]:


print(len(contours))


# In[607]:


for contours_indices in range(len(contours)):
    # In[108]:
    selected_image_index = contours_indices


    this_piece = puzzlePiece(selected_image_index)
    this_piece.contour = contours[contours_indices]

    # In[109]:
    piece_images = []

    i = selected_image_index
    contour = contours[i]

    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    this_piece.mask = mask

    piece = np.zeros_like(original_image)
    piece[mask == 255] = original_image[mask == 255]
    x, y, w, h = cv2.boundingRect(contour)
    cropped_piece = piece[y:y + h, x:x + w]
    piece_images.append(cropped_piece)

    this_piece.bounding_rect = (x, y, w, h)
    this_piece.absolute_contour = contour
    this_piece.relative_contour = contour - np.array([[x, y]])
    this_piece.cropped_image = cropped_piece

    relative_mask = mask[y:y+h, x:x+w]
    this_piece.relative_mask = relative_mask

    transparent_piece = np.zeros((h, w, 4), dtype=np.uint8)
    transparent_piece[:, :, 0:3] = cropped_piece

    transparent_piece[:, :, 3] = relative_mask
    this_piece.transparent_image = transparent_piece

    transparent_path = os.path.join(output_transparent_folder, f"piece_{selected_image_index}_transparent.png")
    cv2.imwrite(transparent_path, transparent_piece)


    piece_path = os.path.join(output_folder_pieces, f"piece_{selected_image_index + 1}.png")
    cv2.imwrite(piece_path, cropped_piece)
    contour_piece = cropped_piece.copy()

    display_image(f"Cropped piece {selected_image_index+1}", contour_piece)


    # In[110]:
    M = cv2.moments(contour)

    if M["m00"] == 0:
        print("ERROR")
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    cv2.circle(contour_piece, (centroid_x - x, centroid_y - y), 2, (0, 0, 255), -1)
    # display_image(f"Centroid {i+1}", contour_piece)

    this_piece.relative_center = (centroid_x - x, centroid_y - y)
    this_piece.absolute_center = (centroid_x, centroid_y)

    # In[113]:
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
    distances = gaussian_filter1d(distances, sigma=2)

    this_piece.contour_polar = (angles_deg, distances)

    delta_s = len(angles_deg) // 4

    angles_deg_s = np.roll(angles_deg, -delta_s)
    distances_s = np.roll(distances, -delta_s)

    # find the maxima
    min_distance_between_peaks = len(distances) // 20
    all_peak_indices, all_peak_properties = find_peaks(distances,                                 distance=min_distance_between_peaks,prominence=2)

    all_peak_indices_s, all_peak_properties_s = find_peaks(distances_s,                                 distance=min_distance_between_peaks,prominence=2)
    all_peak_indices_u = [(x+delta_s)%len(angles_deg) for x in all_peak_indices_s]
    all_peak_indices_f = [y for y in all_peak_indices_u if y not in all_peak_indices]

    for val in all_peak_indices_f:
        pos = np.searchsorted(all_peak_indices, val)
        all_peak_indices = np.insert(all_peak_indices, pos, val)

    inverted_distances = [-d for d in distances]
    all_min_indices, all_min_properties = find_peaks(inverted_distances,
                                                     distance=min_distance_between_peaks, prominence=0.1)

    # Calculate min_distances_avg
    min_distances = [distances[min_idx] for min_idx in all_min_indices]
    min_distances_avg = np.mean(min_distances)

    # Filtering the peaks - following EXACT same logic as original
    max_angle_diff = 25
    delCounter = 0

    # Convert to list for easier manipulation if needed
    remaining_peak_indices = all_peak_indices.tolist() if isinstance(all_peak_indices,
                                                                     np.ndarray) else all_peak_indices.copy()

    while True:
        removals_made = False

        for k in range(len(remaining_peak_indices)):
            if len(remaining_peak_indices) <= 1:
                break

            if distances[remaining_peak_indices[k]] <= min_distances_avg*1.1:
                remaining_peak_indices.pop(k)
                break

            next_k = (k + 1) % len(remaining_peak_indices)  # Wrap-around logic
            angle_k = angles_deg[remaining_peak_indices[k]]
            angle_next_k = angles_deg[remaining_peak_indices[next_k]]

            if abs(angle_next_k - angle_k) < max_angle_diff:
                delCounter += 1
                removals_made = True

                dist_k = distances[remaining_peak_indices[k]]
                dist_next_k = distances[remaining_peak_indices[next_k]]

                if dist_k < dist_next_k:
                    remaining_peak_indices.pop(k)
                else:
                    remaining_peak_indices.pop(next_k)

                break

        if not removals_made or len(remaining_peak_indices) <= 1:
            break

    all_peak_indices = np.array(remaining_peak_indices) if isinstance(all_peak_indices,
                                                                      np.ndarray) else remaining_peak_indices


    avg_peak_height = np.mean([distances[idx] for idx in all_peak_indices])
    threshold = avg_peak_height / 2

    print(f"THRESHOLD: {threshold}")


    new_min_indice = all_min_indices.copy()
    # new_min_indice = [idx for idx in new_min_indice if distances[idx] <= threshold]
    new_min_indice = [int(idx) for idx in new_min_indice if distances[idx] <= threshold]



    print(f" BEFORE: {all_min_indices}")
    print(f" AFTER: {new_min_indice}")




    all_min_indices = new_min_indice

    min_angles = [angles_deg[min_idx] for min_idx in new_min_indice]
    min_distances = [distances[min_idx] for min_idx in new_min_indice]

    print(f" AFTER MIN ANGLES: {min_angles}")



    # if selected_image_index == 23:
    #     raise ValueError(f"Avg all_peak_indices height: {avg_peak_height:.4f}\n"
    #                      f"All min indices: {all_min_indices:.4f}\n")








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

    # #### JUST ADDED DEBUG
    plt.figure(figsize=(10, 6))
    plt.scatter(angles_deg, distances, marker='.', alpha=0.5, label='All Points')
    # plt.scatter(peak_angles, peak_distances, color='red', marker='o', label='All Peaks')
    plt.scatter(top_4_angles, top_4_distances, color='green', marker='*', s=200, label='Top 4 Pointy Peaks')
    plt.scatter(min_angles, min_distances, color='orange', marker='o', label='Minima')
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Distance from centroid")
    plt.title("Radial Distance vs. Angle with Top 4 Pointiest Peaks")
    plt.legend()
    plt.grid(True)
    # plt.show()

    plt.savefig(f"{output_plots_folder}/plot{selected_image_index}.png")

    peak_angles = angles_deg[all_peak_indices]
    peak_distances = distances[all_peak_indices]

    this_piece.corners_idx = top_4_peak_indices

    # ## Edge type detection

    def exists_peak_between(a,b,peak_indices):
        if b < a:
            for p in peak_indices:
                if p > a or p < b:
                    # print(f"Found maxima {p}")
                    return True

        else:
            for p in peak_indices:
                if p > a and p < b:
                    # print(f"Found maxima {p}")
                    return True

            # print("no peak between")
            return False

    resize_factor = 0.8
    width = 15

    def exists_minima_between(a,b,min_indices):

        if b < a:
            for p in min_indices:
                print("--------------")
                print(f"a: {a}, b: {b} p: {p}, min_indices: {min_indices}")
                if p < b or p > a:
                    if distances[p] < ((max(distances[a], distances[b])*resize_factor ) /1.41):
                        print("returning TRUE external")
                        return True
                    else:
                        print("returning FALSE external, fake minimal")
                        continue
                else:
                    print("returning FALSE external")
                    continue
        else:
            for p in min_indices:
                print("--------------")
                print(f"a: {a}, b: {b}, p: {p}, min_indices: {min_indices}")
                if p > a and p < b:

                    if distances[p] < ((max(distances[a], distances[b])*resize_factor ) /1.41):
                        print("returning TRUE internal")
                        return True
                    else:
                        print("returning FALSE internal, fake minimal")
                        continue
                else:
                    print("returning FALSE internal")
                    continue


        # print("no min between")
        return False

    edge_types = {0: "FLAT", 1: "IN", 2: "OUT"}
    # print(all_min_indices)

    def get_edge_type(a, b):
        # print("-----")
        # print(a, b)
        # print(angles_deg[a], angles_deg[b])
        #
        # print(remaining_indices)
        # print([angles_deg[r] for r in remaining_indices])
        if exists_peak_between(a, b, remaining_indices):
            return 2
        else:
            if exists_minima_between(a, b, all_min_indices):
                return 1
            else:
                return 0


    edges = []

    edges_cat = {0: 0, 1: 0, 2: 0}
    edges_ord = []

    for c, corner in enumerate(top_4_peak_indices):
        corner1 = top_4_peak_indices[c]
        corner2 = top_4_peak_indices[(c+1) % len(top_4_peak_indices)]

        type = get_edge_type(corner1, corner2)
        edges_cat[type] += 1
        edges_ord.append(type)

        edges.append((chr(97 + c), corner1, corner2, get_edge_type(corner1, corner2)))

    this_piece.edges_desc = ("EDGE_ID", "CORNER1", "CORNER2", "EDGE_TYPE")

    # edge_types = {0: "FLAT", 1: "IN", 2: "OUT"}
    # piece_types = {0 : "CORNER", 1 : "BORDER", 2 : "MIDDLE"}

    # print("----edges cat---")
    # print(edges_cat)

    if edges_cat[0] == 2:
        piece_type = 0
        print(f"Adding {selected_image_index} to corners")
        my_puzzle.corners_pieces.append(selected_image_index)
    elif edges_cat[0] == 1:
        piece_type = 1
        my_puzzle.borders_pieces.append(selected_image_index)
    else:
        piece_type = 2
        my_puzzle.middle_pieces.append(selected_image_index)

    this_piece.edges = edges
    this_piece.edges_cat = edges_cat
    this_piece.edges_ord = edges_ord
    this_piece.piece_type = piece_type

    ## Draw edges types with Corners
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

    edges_corners_path = os.path.join(output_tests, f"edges_corners_x_{selected_image_index}.png")
    cv2.imwrite(edges_corners_path, color_edge_corner_piece)

    my_puzzle.all_pieces[selected_image_index] = this_piece


# In[608]:


print("----------- STATS ---------- ")

print(f"Number of pieces: {len(my_puzzle.all_pieces)}")

print("Corners: ", my_puzzle.corners_pieces)
print("Borders: ", my_puzzle.borders_pieces)
print("Middles: ", my_puzzle.middle_pieces)


# ## Algorithm to find the puzzle size

# In[609]:


def find_puzzle_size(total, corners, borders, middles):
    for h in range(1, total + 1):
        if total % h != 0:
            continue
        w = total // h
        if corners == 4 and \
           borders == 2 * (w + h - 4) and \
           middles == (w - 2) * (h - 2):
            return (w, h)
    return None


# In[610]:


print(len(my_puzzle.all_pieces))      # should be 49
print(len(my_puzzle.corners_pieces))  # should be 4
print(len(my_puzzle.borders_pieces))  # should be 20
print(len(my_puzzle.middle_pieces))   # should be 25


# In[611]:


print(my_puzzle.corners_pieces)
print(my_puzzle.borders_pieces)


# In[612]:


# puzzle_c, puzzle_r = find_puzzle_size(len(my_puzzle.all_pieces), len(my_puzzle.corners_pieces), len(my_puzzle.borders_pieces), len(my_puzzle.middle_pieces))
puzzle_r = 4
puzzle_c = 6
print(puzzle_r)
print(puzzle_c)


# In[613]:


def display_image_cv2(title, image):
    if len(image.shape) == 3 and image.shape[2] == 4:
        white_background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

        alpha = image[:, :, 3] / 255.0

        alpha_3channel = np.stack([alpha, alpha, alpha], axis=2)
        bgr = image[:, :, 0:3]

        result = (alpha_3channel * bgr + (1 - alpha_3channel) * white_background).astype(np.uint8)

        cv2.imshow(title, result)
    else:
        cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[614]:


# target_index = 5
# test_piece = my_puzzle.all_pieces[target_index]


# ## Piece reorientation
# 
# pieces with a straight edge -> y axis orientation

# In[615]:


# Create a new white canvas to place all oriented pieces
# Define this before the loop
canvas_width = 1800
canvas_height = 1800
piece_spacing = 250
margin = 100

max_pieces_per_row = puzzle_r
canvas = np.ones((canvas_height, canvas_width, puzzle_c), dtype=np.uint8) * 255  # White background with alpha

# Keep track of the maximum piece height in each row
row_heights = [0] * (len(my_puzzle.borders_pieces + my_puzzle.corners_pieces) // max_pieces_per_row + 1)
piece_positions = []

current_x = margin
current_y = margin
max_height_in_row = 0
piece_count = 0


# In[616]:


class Canvas:
    def __init__(self, width, height, max_pieces_per_row=6,
                 column_spacing=100, row_spacing=50, margin=100):
        self.width = width
        self.height = height
        self.max_pieces_per_row = max_pieces_per_row
        self.column_spacing = column_spacing  # Horizontal spacing between pieces
        self.row_spacing = row_spacing  # Vertical spacing between rows
        self.margin = margin

        # Initialize canvas with transparent white background
        self.canvas = np.ones((height, width, 4), dtype=np.uint8) * 255
        self.canvas[:,:,3] = 255  # Full alpha

        # Placement tracking
        self.current_x = margin
        self.current_y = margin
        self.max_height_in_row = 0
        self.piece_count = 0
        # Avoid division by zero when column_spacing is 0
        row_heights_size = max_pieces_per_row * 2 if column_spacing == 0 else ((width // (column_spacing + 1)) + 1)
        self.row_heights = [0] * row_heights_size
        self.piece_positions = []  # Store (piece_id, x, y) tuples

        # For puzzle reconstruction
        self.grid = {}  # Dictionary mapping (x,y) grid positions to piece IDs

    def rotate_piece(self, piece, rotations=1):
        # Make sure rotations is between 0 and 3
        rotations = rotations % 4
        if rotations == 0:
            return piece  # No rotation needed

        print(f"Rotating piece {piece.piece_id} by {rotations * 90} degrees")

        # Step 1: Rotate the image
        h, w = piece.rotated_image.shape[:2]
        rotated_image = piece.rotated_image.copy()

        for _ in range(rotations):
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)

        # Step 2: Rotate the corners
        rotated_corners = piece.rotated_corners.copy()
        center = (w/2, h/2)

        for _ in range(rotations):
            # For each 90-degree rotation, (x,y) -> (y, w-x)
            new_corners = []
            for x, y in rotated_corners:
                # Rotate around center
                new_x = center[0] + (y - center[1])
                new_y = center[1] - (x - center[0])
                new_corners.append((int(new_x), int(new_y)))
            rotated_corners = new_corners

        # Step 3: Update the piece's properties
        piece.rotated_image = rotated_image
        piece.rotated_corners = rotated_corners

        # Step 4: Rotate the rotated center
        if piece.rotated_center:
            cx, cy = piece.rotated_center
            rotated_center = (cx, cy)
            for _ in range(rotations):
                rx = center[0] + (rotated_center[1] - center[1])
                ry = center[1] - (rotated_center[0] - center[0])
                rotated_center = (int(rx), int(ry))
            piece.rotated_center = rotated_center

        # Step 5: Update the edges to reflect the rotation
        # Rotate the edges_ord list
        if piece.edges_ord:
            piece.edges_ord = piece.edges_ord[-rotations:] + piece.edges_ord[:-rotations]

        # For edge connections, we need to also rotate the edge indices for proper matching
        if hasattr(piece, 'edges') and piece.edges:
            # We need to create a new list with updated edge indices
            rotated_edges = []
            for i in range(len(piece.edges)):
                # Calculate the new edge index after rotation
                new_idx = (i + rotations) % len(piece.edges)
                edge_data = list(piece.edges[i])  # Convert tuple to list for modification

                # Update the edge ID (first element is typically an ID)
                edge_data[0] = chr(ord('a') + new_idx)

                # The corner indices need to be remapped too, but this requires the full contour
                # For now, we'll just update the other elements
                rotated_edges.append(tuple(edge_data))

            # Sort by the new edge IDs if needed
            # piece.edges = sorted(rotated_edges, key=lambda x: x[0])

        return piece

    def display(self):
        # Create a figure with appropriate size
        fig_width = min(12, self.width / 100)
        fig_height = min(10, self.height / 100)
        plt.figure(figsize=(fig_width, fig_height))

        # Handle image with alpha channel (transparency)
        if self.canvas.shape[2] == 4:
            # Create RGB image with white background
            rgb_image = np.ones((self.canvas.shape[0], self.canvas.shape[1], 3), dtype=np.uint8) * 255

            # Extract alpha channel
            alpha = self.canvas[:, :, 3] / 255.0

            # Create 3-channel alpha
            alpha_3channel = np.stack([alpha, alpha, alpha], axis=2)

            # Extract BGR channels and convert to RGB
            bgr = self.canvas[:, :, :3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Blend using alpha
            result = (alpha_3channel * rgb + (1 - alpha_3channel) * rgb_image).astype(np.uint8)

            plt.imshow(result)
        else:
            # Regular RGB image
            plt.imshow(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))

        plt.axis('off')
        plt.title("Puzzle Reconstruction")
        plt.tight_layout()
        plt.show()

    def place_piece(self, piece, grid_x=None, grid_y=None, rotations=0):
        # Apply any requested rotation
        if rotations > 0:
            self.rotate_piece(piece, rotations)

        # If grid positions are specified, use them for placement
        if grid_x is not None and grid_y is not None:
            # Calculate position based on grid coordinates
            x = self.margin + grid_x * self.column_spacing
            y = self.margin + grid_y * self.row_spacing
            self.grid[(grid_x, grid_y)] = piece.piece_id

            # For reconstruction, we need to adjust according to piece features
            # This depends on your specific needs for how pieces should align

            # Store placement info
            self.piece_positions.append((piece.piece_id, x, y))

            # Set current position for drawing
            self.current_x = x
            self.current_y = y
        else:
            # Calculate position on the grid (for regular display)
            row = self.piece_count // self.max_pieces_per_row
            col = self.piece_count % self.max_pieces_per_row

            # Start a new row if needed
            if col == 0 and self.piece_count > 0:
                self.current_y += self.max_height_in_row + self.row_spacing
                self.current_x = self.margin
                self.max_height_in_row = 0

        # Get piece dimensions
        piece_height, piece_width = piece.rotated_image.shape[:2]
        self.max_height_in_row = max(self.max_height_in_row, piece_height)

        # Get reference corner (first rotated corner), if available
        ref_corner = (0, 0)
        if hasattr(piece, 'rotated_corners') and piece.rotated_corners and hasattr(piece, 'flat_edge_id'):
            ref_corner = piece.rotated_corners[piece.flat_edge_id]

        # Calculate adjusted placement coordinates
        adjusted_x = self.current_x - ref_corner[0]
        adjusted_y = self.current_y - ref_corner[1]

        # Make sure we don't exceed canvas dimensions
        y_start = max(0, adjusted_y)
        x_start = max(0, adjusted_x)
        canvas_y_end = min(self.height, adjusted_y + piece_height)
        canvas_x_end = min(self.width, adjusted_x + piece_width)

        # Calculate corresponding regions in rotated image
        img_y_start = max(0, -adjusted_y)
        img_x_start = max(0, -adjusted_x)
        img_y_end = img_y_start + (canvas_y_end - y_start)
        img_x_end = img_x_start + (canvas_x_end - x_start)

        if (canvas_y_end > y_start) and (canvas_x_end > x_start):
            # Alpha blending
            for c in range(4):
                mask = piece.rotated_image[img_y_start:img_y_end, img_x_start:img_x_end, 3:4] / 255.0
                if c < 3:  # RGB channels
                    self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, c] = (
                        self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, c] * (1 - mask[:,:,0]) +
                        piece.rotated_image[img_y_start:img_y_end, img_x_start:img_x_end, c] * mask[:,:,0]
                    )
                else:  # Alpha channel
                    self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, 3] = np.maximum(
                        self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, 3],
                        piece.rotated_image[img_y_start:img_y_end, img_x_start:img_x_end, 3]
                    )

        # Add piece ID text - only if requested
        if piece.rotated_center and not hasattr(piece, 'skip_id_label'):
            center_x = x_start + (piece.rotated_center[0] - img_x_start)
            center_y = y_start + (piece.rotated_center[1] - img_y_start)
            # cv2.putText(self.canvas, str(piece.piece_id), (center_x, center_y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255, 255), 2)

        # Debug marker
        cv2.circle(self.canvas, (self.current_x, self.current_y), 3, (0, 0, 255, 255), -1)

        # Move to next position for regular placement
        if grid_x is None or grid_y is None:
            self.current_x += self.column_spacing + 100
            self.piece_count += 1

        return self.piece_count - 1  # Return the position index

    def place_corner(self, piece, corner_position="top-left"):
        # Determine grid position based on requested corner
        if corner_position == "top-left":
            grid_x, grid_y = 0, 0
            target_flat_edges = [0, 3]  # North and West
        elif corner_position == "top-right":
            grid_x, grid_y = self.max_pieces_per_row - 1, 0
            target_flat_edges = [0, 1]  # North and East
        elif corner_position == "bottom-left":
            grid_x, grid_y = 0, self.max_pieces_per_row - 1
            target_flat_edges = [2, 3]  # South and West
        elif corner_position == "bottom-right":
            grid_x, grid_y = self.max_pieces_per_row - 1, self.max_pieces_per_row - 1
            target_flat_edges = [1, 2]  # East and South
        else:
            raise ValueError(f"Invalid corner position: {corner_position}")

        # Find which edges of the piece are flat (type 0)
        flat_edge_indices = [i for i, edge_data in enumerate(piece.edges) if edge_data[3] == 0]
        print(f"Flat edges for piece {piece.piece_id}: {flat_edge_indices}")

        # Determine how many rotations are needed
        rotations_needed = None
        for r in range(4):
            # Calculate where the flat edges would be after r rotations
            rotated_flat_edges = [(idx + r) % 4 for idx in flat_edge_indices]
            rotated_flat_edges.sort()

            if rotated_flat_edges == sorted(target_flat_edges):
                rotations_needed = r
                break

        # If we couldn't find a valid rotation, try best approximation
        if rotations_needed is None:
            print("Warning: Couldn't find perfect corner orientation, using best approximation")
            # Try to match at least one flat edge
            for r in range(4):
                rotated_flat_edges = [(idx + r) % 4 for idx in flat_edge_indices]
                if any(edge in target_flat_edges for edge in rotated_flat_edges):
                    rotations_needed = r
                    break
            # If still none found, just use 0
            if rotations_needed is None:
                rotations_needed = 0

        print(f"Placing {corner_position} corner (piece {piece.piece_id}) with {rotations_needed * 90}° rotation")

        # Place the piece with the calculated rotation
        self.place_piece(piece, grid_x, grid_y, rotations=rotations_needed)

        # Add to the grid
        self.grid[(grid_x, grid_y)] = piece.piece_id

        # Print verification
        flat_edge_indices_after = [(i + rotations_needed) % 4 for i in flat_edge_indices]
        print(f"Flat edges after rotation: {flat_edge_indices_after}")

        return grid_x, grid_y

    def save(self, path):
        """Save the canvas to file"""
        cv2.imwrite(path, self.canvas)

    def get_piece_position(self, piece_id):
        """Get the position of a piece on the canvas"""
        for p_id, x, y in self.piece_positions:
            if p_id == piece_id:
                return (x, y)
        return None


# In[617]:


# First process and reorient all pieces
for index in (my_puzzle.borders_pieces + my_puzzle.corners_pieces + my_puzzle.middle_pieces):

    print("------- ROTATE TO ORTHOGONAL ORIENTATION ----------")
    print(f"Testing index: {index}")

    test_piece = my_puzzle.all_pieces[index]

    edge_id = 0

    print(test_piece.edges_ord)

    # Check if this is a middle piece (no flat edges)
    if index in my_puzzle.middle_pieces:
        print(f"-> Middle piece detected, using edge 0")
        edge_id = 0
    else:
        # Original logic for border/corner pieces
        for k, id in enumerate(test_piece.edges_ord):
            print(f"Testing id: {id}")
            if id == 0:
                print(f"-> Found flat edge: {k}")
                edge_id = k
                break

    # Store the flat edge ID
    test_piece.flat_edge_id = edge_id

    corner1 = test_piece.edges[edge_id][1]
    corner2 = test_piece.edges[edge_id][2]

    point_1 = test_piece.relative_contour[corner1][0]
    point_2 = test_piece.relative_contour[corner2][0]

    vec = point_2 - point_1
    orth = np.array([0, 1])

    cos_theta = np.dot(vec, orth) / (np.linalg.norm(vec) * np.linalg.norm(orth))
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    det = vec[0] * orth[1] - vec[1] * orth[0]
    sign = -1 if det > 0 else 1

    # Apply the sign to the angle
    angle_deg = sign * angle_deg

    # Store rotation angle
    test_piece.rotation_angle = angle_deg

    image_with_center = test_piece.transparent_image.copy()
    cv2.circle(image_with_center, test_piece.relative_center, 2, (0, 0, 255, 255), -1)

    (h, w) = image_with_center.shape[:2]
    center = tuple(test_piece.relative_center)

    # Expand canvas
    old_h, old_w = image_with_center.shape[:2]
    expansion_ratio = 1.8
    new_w = int(old_w * expansion_ratio)
    new_h = int(old_h * expansion_ratio)

    # Calculate offsets to center the image in the expanded canvas
    x_offset = (new_w - old_w) // 2
    y_offset = (new_h - old_h) // 2

    # Store expansion data
    test_piece.expansion_ratio = expansion_ratio
    test_piece.expansion_offset = (x_offset, y_offset)

    # Create expanded canvas with transparent background
    expanded_image = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    expanded_image[y_offset:y_offset+old_h, x_offset:x_offset+old_w] = image_with_center

    # Store expanded image
    test_piece.expanded_image = expanded_image

    # Update center for rotation
    new_center_x = test_piece.relative_center[0] + x_offset
    new_center_y = test_piece.relative_center[1] + y_offset
    new_center = (new_center_x, new_center_y)

    # Create rotation matrix for the expanded image
    M = cv2.getRotationMatrix2D(new_center, angle_deg, 1.0)

    # Store rotation matrix
    test_piece.rotation_matrix = M

    # Apply the rotation to the expanded image
    rotated_image = cv2.warpAffine(expanded_image, M, (new_w, new_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)

    # Store rotated image
    test_piece.rotated_image = rotated_image

    edge_direction_map = {}

    # Calculate the rotated points (need to adjust original points with offset)
    p1_offset = (point_1[0] + x_offset, point_1[1] + y_offset)
    p2_offset = (point_2[0] + x_offset, point_2[1] + y_offset)

    p1_rot = M @ np.array([p1_offset[0], p1_offset[1], 1])
    p2_rot = M @ np.array([p2_offset[0], p2_offset[1], 1])

    p1_rot = tuple(np.round(p1_rot).astype(int))
    p2_rot = tuple(np.round(p2_rot).astype(int))

    # Store rotated center
    rotated_center = M @ np.array([new_center[0], new_center[1], 1])
    test_piece.rotated_center = tuple(np.round(rotated_center[:2]).astype(int))

    # Initialize rotated corners list
    test_piece.rotated_corners = []
    test_piece.rotated_edges = []

    edge_colors = {
        0: (0, 0, 255, 255),   # Blue for edge_type 0 (flat edge)
        1: (255, 255, 0, 255), # Yellow for edge_type 1
        2: (0, 255, 0, 255),   # Green for edge_type 2
        3: (255, 0, 0, 255)    # Red for edge_type 3
    }

    corner_color = (255, 0, 255, 255)  # Purple color for corners

    colored_edges_image = rotated_image.copy()

    # Create a list to store the entire rotated contour
    rotated_contour = []

    # Rotate the entire contour
    for point in test_piece.relative_contour:
        pt = point[0]
        pt_offset = (pt[0] + x_offset, pt[1] + y_offset)
        pt_rot = M @ np.array([pt_offset[0], pt_offset[1], 1])
        pt_rot = tuple(np.round(pt_rot[:2]).astype(int))
        rotated_contour.append(pt_rot)

    # Store rotated contour
    test_piece.rotated_contour = np.array(rotated_contour)

    print("Edges structure:", test_piece.edges)

    for edge_idx, edge_data in enumerate(test_piece.edges):
        # [edge_type, corner1_idx, corner2_idx]
        edge_id = edge_data[0]
        corner1_idx = edge_data[1]
        corner2_idx = edge_data[2]
        edge_type = edge_data[3]

        # Get the corners in the original coordinate system
        c1 = test_piece.relative_contour[corner1_idx][0]
        c2 = test_piece.relative_contour[corner2_idx][0]

        # Apply the same offset as we did for the image expansion
        c1_offset = (c1[0] + x_offset, c1[1] + y_offset)
        c2_offset = (c2[0] + x_offset, c2[1] + y_offset)

        # Apply the rotation matrix to get the coordinates in the rotated image
        c1_rot = M @ np.array([c1_offset[0], c1_offset[1], 1])
        c2_rot = M @ np.array([c2_offset[0], c2_offset[1], 1])

        c1_rot = tuple(np.round(c1_rot[:2]).astype(int))
        c2_rot = tuple(np.round(c2_rot[:2]).astype(int))

        # Store rotated corners
        test_piece.rotated_corners.append(c1_rot)

        # Store rotated edge
        rotated_edge = {
            'edge_id': edge_id,
            'start_corner': c1_rot,
            'end_corner': c2_rot,
            'edge_type': edge_type
        }
        test_piece.rotated_edges.append(rotated_edge)

        # Draw a colored circle at each corner
        cv2.circle(colored_edges_image, c1_rot, 5, corner_color, -1)
        cv2.circle(colored_edges_image, c2_rot, 5, corner_color, -1)

        # Draw the edge with a colored line
        color = edge_colors.get(edge_type, (128, 128, 128, 255))  # Default gray if edge type not found

        if (DEBUG):
            cv2.line(colored_edges_image, c1_rot, c2_rot, color, 2)

        # Add edge number label
        # Add edge number label - moved 50 pixels inward toward center
        mid_point = ((c1_rot[0] + c2_rot[0]) // 2, (c1_rot[1] + c2_rot[1]) // 2)

        # Calculate vector from mid_point toward piece center
        center_x, center_y = test_piece.rotated_center
        direction_x = center_x - mid_point[0]
        direction_y = center_y - mid_point[1]

        # Normalize the direction vector
        direction_length = np.sqrt(direction_x**2 + direction_y**2)
        if direction_length > 0:
            unit_x = direction_x / direction_length
            unit_y = direction_y / direction_length
        else:
            unit_x, unit_y = 0, 0

        # Move 50 pixels inward toward center
        inward_distance = 25
        inward_point = (int(mid_point[0] + unit_x * inward_distance),
                        int(mid_point[1] + unit_y * inward_distance))

        cv2.putText(colored_edges_image, str(edge_idx), inward_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0, 255), 2)


    for edge_idx in range(len(test_piece.edges)):
        if edge_idx == test_piece.flat_edge_id:
            final_direction = 3  # WEST
        else:
            # Calculate how many steps we are from the flat edge (going backwards)
            steps_from_flat = (test_piece.flat_edge_id - edge_idx) % 4
            # Each step backwards adds 1 to direction (counter-clockwise from WEST)
            final_direction = (3 + steps_from_flat) % 4

        edge_direction_map[edge_idx] = final_direction

    test_piece.edge_direction_map = edge_direction_map


    # Use the colored edges image instead of rotated_image for placement
    test_piece.rotated_image = colored_edges_image

    # Draw circles on rotated points
    cv2.circle(test_piece.rotated_image, p1_rot, 4, (0, 255, 0, 255), -1)
    cv2.circle(test_piece.rotated_image, p2_rot, 4, (0, 255, 0, 255), -1)

    font_scale = max(0.8, test_piece.rotated_image.shape[0] / 500)
    cv2.putText(test_piece.rotated_image, str(test_piece.piece_id),
                test_piece.rotated_center, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 255, 255), 2)



    # Save the individual piece for verification
    piece_path = os.path.join(output_oriented_folder, f"oriented_piece_{index}.png")
    cv2.imwrite(piece_path, test_piece.rotated_image)

print("All pieces oriented and processed!")


# In[618]:


# Create a canvas object with custom spacing
canvas_obj = Canvas(1800, 1800, column_spacing=100, row_spacing=-150, margin=100)

# Now place all the pieces on the canvas
piece_indices = my_puzzle.borders_pieces + my_puzzle.corners_pieces + my_puzzle.middle_pieces
print(f"Placing {len(piece_indices)} pieces on canvas...")

for index in piece_indices:
    piece = my_puzzle.all_pieces[index]
    print(f"Placing piece {index} on canvas (type: {piece.piece_type})")
    canvas_obj.place_piece(piece)

# Save the final canvas
canvas_path = os.path.join(output_canvas, "all_pieces_arranged.png")
canvas_obj.save(canvas_path)
print(f"Canvas saved to {canvas_path}")


# In[647]:


# Edge matching function to compute a matching score between two puzzle piece edges
def compute_edge_matching_score(piece1, edge1_idx, piece2, edge2_idx, debug=False):
    # Get edge types
    edge1_type = piece1.edges[edge1_idx][3]
    edge2_type = piece2.edges[edge2_idx][3]

    # Check if either edge is flat (type 0)
    if edge1_type == 0 or edge2_type == 0:
        if debug:
            print(f"Edge {edge1_idx} of piece {piece1.piece_id} or edge {edge2_idx} of piece {piece2.piece_id} is flat. Score = 0")
        return 0

    # Check if both edges are the same type (both IN or both OUT)
    if edge1_type == edge2_type:
        if debug:
            print(f"Both edges are type {edge1_type}. Score = 0")
        return 0

    # Check if pieces are border or corner pieces (have at least one flat edge)
    piece1_flat_edges = [i for i, edge in enumerate(piece1.edges) if edge[3] == 0]
    piece2_flat_edges = [i for i, edge in enumerate(piece2.edges) if edge[3] == 0]

    piece1_is_border = len(piece1_flat_edges) == 1
    piece1_is_corner = len(piece1_flat_edges) == 2
    piece1_is_middle = len(piece1_flat_edges) == 0
    piece2_is_border = len(piece2_flat_edges) == 1
    piece2_is_corner = len(piece2_flat_edges) == 2
    piece2_is_middle = len(piece2_flat_edges) == 0

    # NEW: Add middle piece constraints
    # Middle pieces cannot match with corner pieces
    if (piece1_is_middle and piece2_is_corner) or (piece2_is_middle and piece1_is_corner):
        if debug:
            print(f"Middle piece cannot match with corner piece. Score = 0")
        return 0

    # If middle piece is matching with border piece, must be the edge opposite to flat edge
    if piece1_is_middle and piece2_is_border:
        flat_edge2_idx = piece2_flat_edges[0]
        opposite_edge2_idx = (flat_edge2_idx + 2) % 4
        if edge2_idx != opposite_edge2_idx:
            if debug:
                print(f"Middle piece {piece1.piece_id} can only match border piece {piece2.piece_id} on edge opposite to flat edge. Score = 0")
            return 0

    if piece2_is_middle and piece1_is_border:
        flat_edge1_idx = piece1_flat_edges[0]
        opposite_edge1_idx = (flat_edge1_idx + 2) % 4
        if edge1_idx != opposite_edge1_idx:
            if debug:
                print(f"Middle piece {piece2.piece_id} can only match border piece {piece1.piece_id} on edge opposite to flat edge. Score = 0")
            return 0

    # Only apply the "opposite to flat edge" restriction for border pieces, not corners
    if (piece1_is_border and piece2_is_border):
        # For border pieces: check if current edge is opposite to the flat edge
        flat_edge1_idx = piece1_flat_edges[0] if piece1_flat_edges else None
        flat_edge2_idx = piece2_flat_edges[0] if piece2_flat_edges else None

        # Check if current edge is opposite to a flat edge (2 positions away in clockwise order)
        if (flat_edge1_idx is not None and (edge1_idx == (flat_edge1_idx + 2) % 4)) or \
                (flat_edge2_idx is not None and (edge2_idx == (flat_edge2_idx + 2) % 4)):
            if debug:
                print(f"Edge {edge1_idx} of piece {piece1.piece_id} or edge {edge2_idx} of piece {piece2.piece_id} is opposite to a flat edge on a border piece. Score = 0")
            return 0

    # For corner pieces, we need different rules
    if piece1_is_corner or piece2_is_corner:
        # If either piece is a corner, we need to make sure we're not trying to match
        # an edge that's adjacent to two flat edges with an edge that needs non-flat neighbors

        # For piece1, if it's a corner, check if the current edge is between two flat edges
        if piece1_is_corner:
            edge1_prev_idx = (edge1_idx - 1) % 4
            edge1_next_idx = (edge1_idx + 1) % 4

            # If both adjacent edges are flat, this edge can only match with specific edges
            if edge1_prev_idx in piece1_flat_edges and edge1_next_idx in piece1_flat_edges:
                # This edge is between two flat edges - it can only match with a similar edge
                # or an edge that doesn't require flat neighbors
                if piece2_is_corner:
                    edge2_prev_idx = (edge2_idx - 1) % 4
                    edge2_next_idx = (edge2_idx + 1) % 4

                    # If the other edge isn't between two flat edges, they can't match
                    if not (edge2_prev_idx in piece2_flat_edges and edge2_next_idx in piece2_flat_edges):
                        if debug:
                            print(f"Corner piece {piece1.piece_id} edge {edge1_idx} is between two flat edges, but corner piece {piece2.piece_id} edge {edge2_idx} is not. Score = 0")
                        return 0

        # Same check for piece2
        if piece2_is_corner:
            edge2_prev_idx = (edge2_idx - 1) % 4
            edge2_next_idx = (edge2_idx + 1) % 4

            if edge2_prev_idx in piece2_flat_edges and edge2_next_idx in piece2_flat_edges:
                if piece1_is_corner:
                    edge1_prev_idx = (edge1_idx - 1) % 4
                    edge1_next_idx = (edge1_idx + 1) % 4

                    if not (edge1_prev_idx in piece1_flat_edges and edge1_next_idx in piece1_flat_edges):
                        if debug:
                            print(f"Corner piece {piece2.piece_id} edge {edge2_idx} is between two flat edges, but corner piece {piece1.piece_id} edge {edge1_idx} is not. Score = 0")
                        return 0

    # Check if adjacent edges match for border pieces
    # For piece1: adjacent edges are (edge1_idx-1)%4 and (edge1_idx+1)%4
    # For piece2: adjacent edges are (edge2_idx-1)%4 and (edge2_idx+1)%4
    edge1_prev_idx = (edge1_idx - 1) % 4
    edge1_next_idx = (edge1_idx + 1) % 4
    edge2_prev_idx = (edge2_idx - 1) % 4
    edge2_next_idx = (edge2_idx + 1) % 4

    # Get the types of adjacent edges
    edge1_prev_type = piece1.edges[edge1_prev_idx][3]
    edge1_next_type = piece1.edges[edge1_next_idx][3]
    edge2_prev_type = piece2.edges[edge2_prev_idx][3]
    edge2_next_type = piece2.edges[edge2_next_idx][3]

    # Check if the adjacent edges match for border pieces
    # If edge1_prev is flat, then edge2_next should be flat
    # If edge1_next is flat, then edge2_prev should be flat
    if edge1_prev_type == 0 and edge2_next_type != 0:
        if debug:
            print(f"Edge {edge1_prev_idx} of piece {piece1.piece_id} is flat but edge {edge2_next_idx} of piece {piece2.piece_id} is not. Score = 0")
        return 0

    if edge1_next_type == 0 and edge2_prev_type != 0:
        if debug:
            print(f"Edge {edge1_next_idx} of piece {piece1.piece_id} is flat but edge {edge2_prev_idx} of piece {piece2.piece_id} is not. Score = 0")
        return 0

    if edge1_type != 0 and edge1_prev_type != 0 and edge1_next_type != 0:
        if edge2_prev_type == 0 or edge2_next_type == 0:
            if debug:
                print(f"Edge {edge1_idx} and its neighbors of piece {piece1.piece_id} are not flat, "
                      f"but edge {edge2_idx} has flat neighbor(s) in piece {piece2.piece_id}. Score = 0")
            return 0

    if edge2_type != 0 and edge2_prev_type != 0 and edge2_next_type != 0:
        if edge1_prev_type == 0 or edge1_next_type == 0:
            if debug:
                print(f"Edge {edge2_idx} and its neighbors of piece {piece2.piece_id} are not flat, "
                      f"but edge {edge1_idx} has flat neighbor(s) in piece {piece1.piece_id}. Score = 0")
            return 0

    # Ensure edge1 is OUT (2) and edge2 is IN (1) for consistency
    if edge1_type == 1 and edge2_type == 2:  # if edge1 is IN and edge2 is OUT, swap them
        piece1, piece2 = piece2, piece1
        edge1_idx, edge2_idx = edge2_idx, edge1_idx
        edge1_type, edge2_type = edge2_type, edge1_type

    # Get the corner indices for the edges
    start_corner1 = piece1.edges[edge1_idx][1]
    end_corner1 = piece1.edges[edge1_idx][2]
    start_corner2 = piece2.edges[edge2_idx][1]
    end_corner2 = piece2.edges[edge2_idx][2]

    # Get rotated corner coordinates
    if hasattr(piece1, 'rotated_corners') and piece1.rotated_corners:
        start_rot1 = piece1.rotated_corners[edge1_idx]
        end_rot1 = piece1.rotated_corners[(edge1_idx + 1) % len(piece1.rotated_corners)]
    else:
        # Use relative corners for middle pieces
        start_rot1 = piece1.relative_contour[piece1.edges[edge1_idx][1]][0]
        end_rot1 = piece1.relative_contour[piece1.edges[edge1_idx][2]][0]

    if hasattr(piece2, 'rotated_corners') and piece2.rotated_corners:
        start_rot2 = piece2.rotated_corners[edge2_idx]
        end_rot2 = piece2.rotated_corners[(edge2_idx + 1) % len(piece2.rotated_corners)]
    else:
        # Use relative corners for middle pieces
        start_rot2 = piece2.relative_contour[piece2.edges[edge2_idx][1]][0]
        end_rot2 = piece2.relative_contour[piece2.edges[edge2_idx][2]][0]

    # Get rotated contour points between the corners for each edge
    # def get_contour_between_corners(piece, start_idx, end_idx):
    #     contour = piece.absolute_contour  # Use the full contour
    #     if start_idx < end_idx:
    #         return contour[start_idx:end_idx+1]
    #     else:
    #         # Wrap around the contour
    #         return np.vstack([contour[start_idx:], contour[:end_idx+1]])

    def get_contour_between_corners(piece, start_idx, end_idx):
        # Check if piece has rotated contour - if not, throw error
        if not hasattr(piece, 'rotated_contour') or piece.rotated_contour is None:
            raise ValueError(f"ERROR: Piece {piece.piece_id} does not have a rotated_contour!")

        contour = piece.rotated_contour

        # ADD THESE MISSING LINES:
        if start_idx < end_idx:
            return contour[start_idx:end_idx+1]
        else:
            # Wrap around the contour
            return np.vstack([contour[start_idx:], contour[:end_idx+1]])

    # Get contour points for each edge
    contour1 = get_contour_between_corners(piece1, start_corner1, end_corner1)
    contour2 = get_contour_between_corners(piece2, start_corner2, end_corner2)

    # Sample N points along each edge for comparison NNNNN
    N = 30  # Number of sampling points

    # Function to sample points along an edge
    def sample_points_along_edge(contour, N):
        total_points = len(contour)
        # Sample N+2 points to exclude endpoints
        indices = np.linspace(0, total_points-1, N+2, dtype=int)
        sampled_points = contour[indices]

        # Exclude first and last points (endpoints)
        sampled_points = sampled_points[1:-1]

        # Convert to tuples to match rotated contour format
        result = []
        for point in sampled_points:
            if isinstance(point, tuple):
                result.append(point)  # Already a tuple
            elif hasattr(point, 'shape'):
                result.append((int(point[0]), int(point[1])))  # Convert numpy array to tuple
            else:
                result.append((int(point[0][0]), int(point[0][1])))  # Convert nested array to tuple

        return result


    # print("Debug: contour1 =", contour1)
    # print("Debug: contour1 type =", contour1.__class__)
    # print("Debug: contour2 =", contour2)
    # print("Debug: contour2 type =", contour2.__class__)

    # Sample points
    # In the main function, after sampling points:
    sample_points1 = sample_points_along_edge(contour1, N)
    sample_points2 = sample_points_along_edge(contour2, N)

    def calculate_tangent_directions(sample_points):
        tangent_directions = []
        for i, point in enumerate(sample_points):
            # Get previous and next points
            if i == 0:
                prev_pt = point  # Use current point if no previous
            else:
                prev_pt = sample_points[i - 1]

            if i == len(sample_points) - 1:
                next_pt = point  # Use current point if no next
            else:
                next_pt = sample_points[i + 1]

            # Get vector from previous to next
            tangent = (next_pt[0] - prev_pt[0], next_pt[1] - prev_pt[1])

            # Get perpendicular vector (rotate 90 degrees)
            perpendicular = (tangent[1], -tangent[0])

            # Normalize it
            length = np.sqrt(perpendicular[0]**2 + perpendicular[1]**2)


            # raise ValueError(f"TANGENT CALCULATION ERROR:\n"
            #                  f"  sample points: {sample_points}\n"
            #                  f"  Point index: {i}\n"
            #                  f"  Current point: {point}\n"
            #                  f"  Previous point: {prev_pt}\n"
            #                  f"  Next point: {next_pt}\n"
            #                  f"  Tangent vector: {tangent}\n"
            #                  f"  Perpendicular vector: {perpendicular}\n"
            #                  f"  Length: {length}")

            if length > 0:
                perpendicular = (perpendicular[0]/length, perpendicular[1]/length)
            else:
                raise ValueError(f"TANGENT CALCULATION ERROR:\n"
                                 f"  sample points: {sample_points}\n"
                                 f"  Point index: {i}\n"
                                 f"  Current point: {point}\n"
                                 f"  Previous point: {prev_pt}\n"
                                 f"  Next point: {next_pt}\n"
                                 f"  Tangent vector: {tangent}\n"
                                 f"  Perpendicular vector: {perpendicular}\n"
                                 f"  Length: {length}")

            tangent_directions.append(perpendicular)

        return tangent_directions

    tangent_directions1 = calculate_tangent_directions(sample_points1)
    sample_points2 = sample_points2[::-1]
    tangent_directions2 = calculate_tangent_directions(sample_points2)  # ADD THIS LINE

    inward_offset = 8

    inward_points1 = []
    for i, point in enumerate(sample_points1):
        dx, dy = tangent_directions1[i]
        inside_x = int(point[0] + dx * inward_offset)
        inside_y = int(point[1] + dy * inward_offset)
        inward_points1.append((inside_x, inside_y))

    inward_points2 = []
    for i, point in enumerate(sample_points2):
        dx, dy = tangent_directions2[i]
        inside_x = int(point[0] - dx * inward_offset)
        inside_y = int(point[1] - dy * inward_offset)
        inward_points2.append((inside_x, inside_y))


    original_edge_points1 = sample_points1.copy()
    original_edge_points2 = sample_points2.copy()

    # raise ValueError(f"DEBUG TANGENT CALCULATION RESULTS:\n"
    #                  f"sample_points1: {sample_points1}\n"
    #                  f"sample_points2: {sample_points2}\n"
    #                  f"tangent_directions1: {tangent_directions1}\n"
    #                  f"tangent_directions2: {tangent_directions2}\n"
    #                  f"Length of sample_points1: {len(sample_points1)}\n"
    #                  f"Length of sample_points2: {len(sample_points2)}\n"
    #                  f"Length of tangent_directions1: {len(tangent_directions1)}\n"
    #                  f"Length of tangent_directions2: {len(tangent_directions2)}")


    def get_color_at_point(piece, point):
        cx, cy = piece.rotated_center
        contour = piece.rotated_contour
        idx = None

        # Calculate point slightly inside the piece
        inside_x, inside_y = point
        color = piece.rotated_image[inside_y, inside_x, :3]  # Get BGR channels

        return color, point


    inward_offset = 8

    colors_and_points1 = [get_color_at_point(piece1, pt) for i, pt in enumerate(inward_points1)]
    colors_and_points2 = [get_color_at_point(piece2, pt) for i, pt in enumerate(inward_points2)]

    # print("---------DEBUGGING---------")

    colors1 = np.array([cp[0] for cp in colors_and_points1])
    sample_inward_points1 = inward_points1.copy()

    colors2 = np.array([cp[0] for cp in colors_and_points2])
    sample_inward_points2 = inward_points2.copy()

    # Filter out black points (background)
    valid_indices = []
    for i in range(N):
        valid_indices.append(i)

    # print("---------- #### DEBUG COMPUTING COLORS DIFF ------------")
    def bgr_to_hs(bgr_color):
        bgr_color = np.array(bgr_color, dtype=np.uint8)
        if bgr_color.shape == (3,):
            bgr_color = bgr_color.reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
        return hsv[:2].astype(np.float32)

    def get_edge_points(piece, edge_idx):
        """Extract the actual edge points for shape comparison"""
        if hasattr(piece, 'rotated_corners') and piece.rotated_corners:
            # If using rotated corners, we need to get the edge segment
            start_corner = piece.rotated_corners[edge_idx]
            end_corner = piece.rotated_corners[(edge_idx + 1) % len(piece.rotated_corners)]
            # This is a simplified approach - you might need to extract actual contour points
            # between these corners from the piece's contour
            return np.array([start_corner, end_corner])
        else:
            # Extract contour points for this edge
            edge_info = piece.edges[edge_idx]
            start_idx = edge_info[1]  # Start point index
            end_idx = edge_info[2]    # End point index

            # Get contour points between start and end indices
            if start_idx <= end_idx:
                edge_points = piece.relative_contour[start_idx:end_idx+1]
            else:
                # Handle wraparound case
                edge_points = np.concatenate([
                    piece.relative_contour[start_idx:],
                    piece.relative_contour[:end_idx+1]
                ])

            return np.array([point[0] for point in edge_points])

    def normalize_edge_shape(edge_points):
        """Normalize edge shape for comparison"""
        if len(edge_points) < 2:
            return edge_points

        # Center the edge points
        centroid = np.mean(edge_points, axis=0)
        centered = edge_points - centroid

        # Scale to unit length
        edge_length = np.sqrt(np.sum((edge_points[-1] - edge_points[0])**2))
        if edge_length > 0:
            normalized = centered / edge_length
        else:
            normalized = centered

        return normalized

    def resample_edge(edge_points, num_points=50):
        if len(edge_points) < 2:
            return edge_points

        # Calculate cumulative distances along the edge
        distances = [0]
        for i in range(1, len(edge_points)):
            dist = np.sqrt(np.sum((edge_points[i] - edge_points[i-1])**2))
            distances.append(distances[-1] + dist)

        total_length = distances[-1]
        if total_length == 0:
            return edge_points

        # Create evenly spaced points along the edge
        target_distances = np.linspace(0, total_length, num_points)
        resampled_points = []

        for target_dist in target_distances:
            # Find the segment containing this distance
            for i in range(len(distances) - 1):
                if distances[i] <= target_dist <= distances[i + 1]:
                    # Interpolate between points i and i+1
                    if distances[i + 1] == distances[i]:
                        ratio = 0
                    else:
                        ratio = (target_dist - distances[i]) / (distances[i + 1] - distances[i])

                    interpolated_point = (1 - ratio) * edge_points[i] + ratio * edge_points[i + 1]
                    resampled_points.append(interpolated_point)
                    break

        return np.array(resampled_points)

    def calculate_shape_score(piece1, edge1_idx, piece2, edge2_idx):
        """Calculate orientation-independent shape similarity score between two edges"""
        try:
            # Get edge vertices (corners)
            vertex1_1 = piece1.rotated_corners[edge1_idx]
            vertex1_2 = piece1.rotated_corners[(edge1_idx + 1) % len(piece1.rotated_corners)]
            vertex2_1 = piece2.rotated_corners[edge2_idx]
            vertex2_2 = piece2.rotated_corners[(edge2_idx + 1) % len(piece2.rotated_corners)]

            # Get contour points between vertices
            contour1 = get_contour_between_corners(piece1, piece1.edges[edge1_idx][1], piece1.edges[edge1_idx][2])
            contour2 = get_contour_between_corners(piece2, piece2.edges[edge2_idx][1], piece2.edges[edge2_idx][2])

            # Normalize both edges to 0-1 coordinate system
            def normalize_edge_to_01(contour, v1, v2):
                contour_array = np.array([pt if isinstance(pt, tuple) else pt[0] for pt in contour])
                v1, v2 = np.array(v1), np.array(v2)

                # Translate so v1 is at origin
                translated = contour_array - v1

                # Get edge vector and length
                edge_vector = v2 - v1
                edge_length = np.linalg.norm(edge_vector)

                if edge_length == 0:
                    return translated

                # Scale so v2 is at distance 1
                normalized = translated / edge_length

                return normalized

            norm_contour1 = normalize_edge_to_01(contour1, vertex1_1, vertex1_2)
            norm_contour2 = normalize_edge_to_01(contour2, vertex2_1, vertex2_2)

            # Sample N points along normalized edges
            N = 15
            samples1 = sample_points_along_edge(norm_contour1, N)
            samples2 = sample_points_along_edge(norm_contour2, N)

            # Try both orientations (normal and flipped)
            distances_normal = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(samples1, samples2)]
            distances_flipped = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(samples1, samples2[::-1])]

            # Take the better (smaller average distance) orientation
            avg_dist_normal = np.mean(distances_normal)
            avg_dist_flipped = np.mean(distances_flipped)
            avg_distance = min(avg_dist_normal, avg_dist_flipped)

            # Convert to similarity score
            shape_score = np.exp(-avg_distance * 8)

            return max(0, min(1, shape_score))

        except Exception as e:
            print(f"Shape comparison error: {e}")
            return 0.0


    # Improved color comparison focusing on actual color similarity
    # color_diffs = []
    # sq_diffs = []
    #
    # for i in valid_indices:
    #     try:
    #         # Get BGR colors
    #         bgr1 = np.array(colors1[i], dtype=np.float32)
    #         bgr2 = np.array(colors2[i], dtype=np.float32)
    #
    #         # Convert to HSV for better color analysis
    #         hs1 = bgr_to_hs(colors1[i])
    #         hs2 = bgr_to_hs(colors2[i])
    #
    #         # Simple RGB distance for base comparison
    #         rgb_dist = np.sqrt(np.sum((bgr1 - bgr2)**2))
    #
    #         # Saturation-aware comparison
    #         sat1, sat2 = hs1[1], hs2[1]
    #
    #         # Both colors are very unsaturated (grey-ish)
    #         if sat1 < 20 and sat2 < 20:
    #             # For grey colors, focus mainly on brightness similarity
    #             brightness1 = np.mean(bgr1)
    #             brightness2 = np.mean(bgr2)
    #             brightness_diff = abs(brightness1 - brightness2)
    #             color_similarity = np.exp(-brightness_diff / 50.0)
    #
    #         # Both colors are saturated - compare hue and saturation
    #         elif sat1 > 30 and sat2 > 30:
    #             # Hue comparison with wraparound
    #             hue_diff = min(abs(hs1[0] - hs2[0]), 180 - abs(hs1[0] - hs2[0]))
    #             sat_diff = abs(sat1 - sat2)
    #
    #             # Weight hue difference more heavily for saturated colors
    #             hue_score = np.exp(-hue_diff / 15.0)  # More sensitive to hue differences
    #             sat_score = np.exp(-sat_diff / 30.0)
    #
    #             # Combine hue and saturation scores
    #             color_similarity = (hue_score * 0.7) + (sat_score * 0.3)
    #
    #         # One saturated, one unsaturated - this should be penalized
    #         else:
    #             # Heavy penalty for mixing saturated and unsaturated colors
    #             sat_diff = abs(sat1 - sat2)
    #             color_similarity = np.exp(-(sat_diff + 50) / 30.0)  # Extra penalty
    #
    #         # Also consider RGB distance as a backup
    #         rgb_similarity = np.exp(-rgb_dist / 80.0)
    #
    #         # Combine both similarities - favor the HSV approach but use RGB as backup
    #         final_similarity = (color_similarity * 0.8) + (rgb_similarity * 0.2)
    #
    #         color_diffs.append(final_similarity)
    #
    #         # Store squared difference for backward compatibility
    #         sq_diff = rgb_dist**2
    #         sq_diffs.append(sq_diff)
    #
    #     except Exception as e:
    #         print(f"Color comparison error: {e}")
    #         color_diffs.append(0.0)
    #         sq_diffs.append(10000.0)
    #
    # # raise ValueError(f"DEBUG COLOR SCORES:\n"
    # #                  f"  Piece1 ID: {piece1.piece_id}, Edge: {edge1_idx}\n"
    # #                  f"  Piece2 ID: {piece2.piece_id}, Edge: {edge2_idx}\n"
    # #                  f"  Number of points: {len(color_diffs)}\n"
    # #                  f"  Individual color scores: {color_diffs}\n"
    # #                  f"  Overall color score: {np.mean(color_diffs):.4f}")
    #
    # # Avg similarity score (same as original)
    # good_matches = [score for score in color_diffs if score > 0.7]  # High similarity scores
    # if len(good_matches) >= 3:  # If we have at least 3 really good matches
    #     # Give extra weight to good matches
    #     boosted_score = np.mean(color_diffs) + (len(good_matches) / len(color_diffs)) * 0.3
    #     color_score = min(1.0, boosted_score)
    # else:
    #     color_score = np.mean(color_diffs)
    #
    # color_score = max(0, min(1, color_score))

    color_diffs = []
    distinctive_matches = []

    for i in valid_indices:
        try:
            bgr1 = np.array(colors1[i], dtype=np.float32)
            bgr2 = np.array(colors2[i], dtype=np.float32)

            hs1 = bgr_to_hs(colors1[i])
            hs2 = bgr_to_hs(colors2[i])

            rgb_dist = np.sqrt(np.sum((bgr1 - bgr2)**2))
            sat1, sat2 = hs1[1], hs2[1]

            # More forgiving base scoring
            if sat1 < 20 and sat2 < 20:
                brightness1 = np.mean(bgr1)
                brightness2 = np.mean(bgr2)
                brightness_diff = abs(brightness1 - brightness2)
                color_similarity = np.exp(-brightness_diff / 80.0)  # More forgiving
            elif sat1 > 30 and sat2 > 30:
                hue_diff = min(abs(hs1[0] - hs2[0]), 180 - abs(hs1[0] - hs2[0]))
                sat_diff = abs(sat1 - sat2)
                hue_score = np.exp(-hue_diff / 25.0)  # More forgiving
                sat_score = np.exp(-sat_diff / 50.0)   # More forgiving
                color_similarity = (hue_score * 0.7) + (sat_score * 0.3)

                # HUGE BONUS for distinctive color matches
                if hue_diff < 15 and sat_diff < 30 and min(sat1, sat2) > 50:
                    color_similarity = min(1.0, color_similarity + 0.4)  # Massive boost
                    distinctive_matches.append(color_similarity)
            else:
                sat_diff = abs(sat1 - sat2)
                color_similarity = np.exp(-sat_diff / 60.0)  # Much more forgiving

            rgb_similarity = np.exp(-rgb_dist / 120.0)  # More forgiving
            final_similarity = max(color_similarity, rgb_similarity)  # Take the BEST

            if (sat1 > 60 or sat2 > 60) and hue_diff > 30:  # One distinctive, very different hues
                final_similarity *= 0.5  # Small negative penalty

            color_diffs.append(final_similarity)

        except Exception as e:
            print(f"Color comparison error: {e}")
            color_diffs.append(0.3)

    # Extra bonus if multiple distinctive colors match
    base_score = np.mean(color_diffs)
    if len(distinctive_matches) >= 2:
        distinctive_bonus = len(distinctive_matches) / len(color_diffs) * 0.3
        color_score = min(1.0, base_score + distinctive_bonus)
    else:
        color_score = base_score

    color_score = max(0, min(1, color_score))

    # Edge length comparison (your existing code)
    def get_edge_length(piece, edge_idx):
        if hasattr(piece, 'rotated_corners') and piece.rotated_corners:
            start_corner = piece.rotated_corners[edge_idx]
            end_corner = piece.rotated_corners[(edge_idx + 1) % len(piece.rotated_corners)]
        else:
            start_corner = piece.relative_contour[piece.edges[edge_idx][1]][0]
            end_corner = piece.relative_contour[piece.edges[edge_idx][2]][0]

        distance = np.sqrt((end_corner[0] - start_corner[0])**2 + (end_corner[1] - start_corner[1])**2)
        return distance

    edge1_length = get_edge_length(piece1, edge1_idx)
    edge2_length = get_edge_length(piece2, edge2_idx)

    # Calculate length difference percentage
    length_diff_percent = abs(edge1_length - edge2_length) / max(edge1_length, edge2_length)

    # Convert to score (0-1, where 1 is best match)
    if length_diff_percent <= 0.05:  # Less than 5% difference
        length_score = 1.0
    elif length_diff_percent >= 0.30:  # More than 30% difference
        length_score = 0.0
    else:
        # Linear interpolation between 5% and 30%
        length_score = 1.0 - ((length_diff_percent - 0.05) / (0.30 - 0.05))

    # NEW: Shape comparison score
    shape_score = calculate_shape_score(piece1, edge1_idx, piece2, edge2_idx)

    # Combine all scores with weights (adjust as needed)
    final_score = (color_score * 0.8) + (length_score * 0.2) + (shape_score * 0)
    score = max(0, min(1, final_score))

    k1 =11
    v1 = 0

    k2 = 18
    v2 = 1
    #
    # if ((piece1.piece_id == k1 and edge1_idx == v1 and piece2.piece_id == k2 and edge2_idx == v2) or
    #         (piece1.piece_id == k2 and edge1_idx == v2 and piece2.piece_id == k1 and edge2_idx == v1)):
    #     raise ValueError(f"DEBUG SPECIFIC MATCH:\n"
    #                      f"  Piece1: {piece1.piece_id}, Edge: {edge1_idx}\n"
    #                      f"  Piece2: {piece2.piece_id}, Edge: {edge2_idx}\n"
    #                      f"  Color score: {color_score:.4f}\n"
    #                      f"  Length score: {length_score:.4f}\n"
    #                      f"  Shape score: {shape_score:.4f}\n"
    #                      f"  Final score: {final_score:.4f}")




    # if ((piece1.piece_id == 11 and edge1_idx == 0 and piece2.piece_id == 18 and edge2_idx == 1) or
    #         (piece1.piece_id == 18 and edge1_idx == 1 and piece2.piece_id == 11 and edge2_idx == 0)):
    #     # Extract the actual colors used in comparison
    #     colors1_list = [colors1[i] for i in valid_indices]
    #     colors2_list = [colors2[i] for i in valid_indices]
    #
    #     raise ValueError(f"DEBUG COLOR ARRAYS:\n"
    #                      f"  Piece1: {piece1.piece_id}, Edge: {edge1_idx}\n"
    #                      f"  Piece2: {piece2.piece_id}, Edge: {edge2_idx}\n"
    #                      f"  Colors1 (BGR): {colors1_list}\n"
    #                      f"  Colors2 (BGR): {colors2_list}\n"
    #                      f"  Color differences: {color_diffs}\n"
    #                      f"  Valid indices: {valid_indices}\n"
    #                      f"  Color score: {color_score:.4f}")

    # Also keep avg_diff and max_possible_diff
    # avg_diff = np.mean(sq_diffs)
    max_possible_diff = (255**2) * 3

    #### DEBUG PRINT PIECE MATCHING TO CV2 IMAGE ####

    if debug:
        # Create output directory
        output_matching = f"{base_path}/matching"
        os.makedirs(output_matching, exist_ok=True)

        # Create a visualization of the edge matching
        visualization_height = 600
        visualization_width = 1000
        # Create visualization with transparent background (4 channels)
        visualization = np.ones((visualization_height, visualization_width, 4), dtype=np.uint8) * 255
        visualization[:, :, 3] = 255  # Full alpha

        # Define regions for the two pieces
        region_width = 400
        margin = 50
        divider_x = visualization_width // 2

        c_score = score

        # Draw title
        title = f"P{piece1.piece_id}(E{edge1_idx}) vs P{piece2.piece_id}(E{edge2_idx}) - CS {c_score:.3f} SC: {shape_score} DS: {length_score:.3f}"
        cv2.putText(visualization, title, (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 255), 2)

        # Draw vertical divider
        cv2.line(visualization, (divider_x, 0), (divider_x, visualization_height),
                 (100, 100, 100, 255), 2)

        # Extract regions around the edges being tested
        def extract_edge_region(piece, edge_idx):
            # Get edge corners
            start_corner = piece.rotated_corners[edge_idx]
            end_corner = piece.rotated_corners[(edge_idx + 1) % len(piece.rotated_corners)]

            # Calculate region center and size
            center_x = (start_corner[0] + end_corner[0]) // 2
            center_y = (start_corner[1] + end_corner[1]) // 2

            # Calculate distance between corners
            dist = np.sqrt((end_corner[0] - start_corner[0])**2 + (end_corner[1] - start_corner[1])**2)
            size = int(dist * 2.5)  # Make region larger than just the edge

            # Extract region
            x1 = max(0, center_x - size//2)
            y1 = max(0, center_y - size//2)
            x2 = min(piece.rotated_image.shape[1], center_x + size//2)
            y2 = min(piece.rotated_image.shape[0], center_y + size//2)

            # Make a copy of the region
            region = piece.rotated_image[y1:y2, x1:x2].copy()

            # Resize to fit in visualization
            if region.shape[0] > 0 and region.shape[1] > 0:
                scale = min(region_width / region.shape[1],
                            (visualization_height - 200) / region.shape[0])
                new_width = int(region.shape[1] * scale)
                new_height = int(region.shape[0] * scale)

                resized = cv2.resize(region, (new_width, new_height))
                return resized, (x1, y1, x2, y2), scale

            return np.ones((10, 10, 4), dtype=np.uint8) * 255, (0, 0, 0, 0), 1.0

        # Extract regions
        region1, bounds1, scale1 = extract_edge_region(piece1, edge1_idx)
        region2, bounds2, scale2 = extract_edge_region(piece2, edge2_idx)

        # Place regions in visualization
        y_offset1 = (visualization_height - region1.shape[0]) // 3
        y_offset2 = (visualization_height - region2.shape[0]) // 3

        x_offset1 = divider_x - region1.shape[1] - margin
        x_offset2 = divider_x + margin

        # Place regions with proper alpha handling
        if region1.shape[2] == 4:  # Has alpha channel
            for c in range(4):  # For all channels including alpha
                alpha = region1[:, :, 3:4] / 255.0
                if c < 3:  # RGB channels
                    visualization[y_offset1:y_offset1+region1.shape[0],
                    x_offset1:x_offset1+region1.shape[1], c] = (
                            visualization[y_offset1:y_offset1+region1.shape[0],
                            x_offset1:x_offset1+region1.shape[1], c] * (1 - alpha[:, :, 0]) +
                            region1[:, :, c] * alpha[:, :, 0]
                    )
                else:  # Alpha channel
                    visualization[y_offset1:y_offset1+region1.shape[0],
                    x_offset1:x_offset1+region1.shape[1], 3] = np.maximum(
                        visualization[y_offset1:y_offset1+region1.shape[0],
                        x_offset1:x_offset1+region1.shape[1], 3],
                        region1[:, :, 3]
                    )

        if region2.shape[2] == 4:  # Has alpha channel
            for c in range(4):  # For all channels including alpha
                alpha = region2[:, :, 3:4] / 255.0
                if c < 3:  # RGB channels
                    visualization[y_offset2:y_offset2+region2.shape[0],
                    x_offset2:x_offset2+region2.shape[1], c] = (
                            visualization[y_offset2:y_offset2+region2.shape[0],
                            x_offset2:x_offset2+region2.shape[1], c] * (1 - alpha[:, :, 0]) +
                            region2[:, :, c] * alpha[:, :, 0]
                    )
                else:  # Alpha channel
                    visualization[y_offset2:y_offset2+region2.shape[0],
                    x_offset2:x_offset2+region2.shape[1], 3] = np.maximum(
                        visualization[y_offset2:y_offset2+region2.shape[0],
                        x_offset2:x_offset2+region2.shape[1], 3],
                        region2[:, :, 3]
                    )

        # Function to generate rainbow colors
        def rainbow_color(i, num_points, with_alpha=False):
            if num_points <= 1:
                t = 0  # If only one point, use first color
            else:
                t = (i / (num_points - 1)) * 255  # Scale to 0-255 range

            phase = t / 255 * 6
            x = int(255 * (1 - abs(phase % 2 - 1)))

            if phase < 1:     color = (255, x, 0)       # Red → Yellow
            elif phase < 2:   color = (x, 255, 0)       # Yellow → Green
            elif phase < 3:   color = (0, 255, x)       # Green → Cyan
            elif phase < 4:   color = (0, x, 255)       # Cyan → Blue
            elif phase < 5:   color = (x, 0, 255)       # Blue → Magenta
            else:             color = (255, 0, x)       # Magenta → Red

            if with_alpha:
                return (color[0], color[1], color[2], 255)
            return color

        # Draw sample points on both pieces with rainbow colors
        for i, idx in enumerate(valid_indices):
            # Get the original sample points from the full contour
            pt1 = sample_inward_points1[idx]
            pt2 = sample_inward_points2[idx]

            # First, we need to translate points to the correct coordinate system
            # Original points are in absolute coordinates
            # First convert to coordinates relative to the original cropped piece
            pt1_rel = (pt1[0] - piece1.bounding_rect[0], pt1[1] - piece1.bounding_rect[1])
            pt2_rel = (pt2[0] - piece2.bounding_rect[0], pt2[1] - piece2.bounding_rect[1])

            # # Now apply the rotation transformation if it exists
            # if hasattr(piece1, 'rotation_matrix') and piece1.rotation_matrix is not None:
            #     # Apply offset for expanded canvas
            #     pt1_expanded = (pt1_rel[0] + piece1.expansion_offset[0],
            #                     pt1_rel[1] + piece1.expansion_offset[1], 1)
            #     # Apply rotation matrix
            #     pt1_rotated = piece1.rotation_matrix @ pt1_expanded
            #     pt1_rotated = pt1_rotated[:2].astype(int)
            # else:
            #     pt1_rotated = pt1_rel
            #
            # if hasattr(piece2, 'rotation_matrix') and piece2.rotation_matrix is not None:
            #     # Apply offset for expanded canvas
            #     pt2_expanded = (pt2_rel[0] + piece2.expansion_offset[0],
            #                     pt2_rel[1] + piece2.expansion_offset[1], 1)
            #     # Apply rotation matrix
            #     pt2_rotated = piece2.rotation_matrix @ pt2_expanded
            #     pt2_rotated = pt2_rotated[:2].astype(int)
            # else:
            #     pt2_rotated = pt2_rel

            pt1_rotated = pt1  # Points are already in rotated space
            pt2_rotated = pt2  # Points are already in rotated space

            # Now transform to visualization coordinates
            # Calculate position within extracted region (bounds1/bounds2 are x1,y1,x2,y2)
            rel_x1 = pt1_rotated[0] - bounds1[0]
            rel_y1 = pt1_rotated[1] - bounds1[1]
            rel_x2 = pt2_rotated[0] - bounds2[0]
            rel_y2 = pt2_rotated[1] - bounds2[1]

            # Apply scaling from resize
            rel_x1 = int(rel_x1 * scale1)
            rel_y1 = int(rel_y1 * scale1)
            rel_x2 = int(rel_x2 * scale2)
            rel_y2 = int(rel_y2 * scale2)

            # Final position in visualization
            vis_x1 = x_offset1 + rel_x1
            vis_y1 = y_offset1 + rel_y1
            vis_x2 = x_offset2 + rel_x2
            vis_y2 = y_offset2 + rel_y2

            # Get rainbow color for this point
            total_sample_points = len(valid_indices)
            color = rainbow_color(i, total_sample_points, with_alpha=True)



            if 0 <= vis_x1 < visualization_width and 0 <= vis_y1 < visualization_height:
                cv2.circle(visualization, (vis_x1, vis_y1), 2, color, -1)
                cv2.putText(visualization, str(i+1), (vis_x1-5, vis_y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1)

            # # Draw circle at the sample point position
            # if 0 <= vis_x1 < visualization_width and 0 <= vis_y1 < visualization_height:
            #     cv2.circle(visualization, (vis_x1, vis_y1), 5, color, -1)
            #     cv2.putText(visualization, str(i+1), (vis_x1-5, vis_y1-5),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1)

            if 0 <= vis_x2 < visualization_width and 0 <= vis_y2 < visualization_height:
                cv2.circle(visualization, (vis_x2, vis_y2), 2, color, -1)
                cv2.putText(visualization, str(i+1), (vis_x2-5, vis_y2-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1)


        # Draw original edge points in black
        for i, idx in enumerate(valid_indices):
            # Get original edge points (before inward translation)
            edge_pt1 = original_edge_points1[idx]
            edge_pt2 = original_edge_points2[idx]

            # Transform to visualization coordinates (same logic as for colored circles)
            # For piece 1
            pt1_rotated = edge_pt1  # Already in rotated space
            rel_x1 = pt1_rotated[0] - bounds1[0]
            rel_y1 = pt1_rotated[1] - bounds1[1]
            rel_x1 = int(rel_x1 * scale1)
            rel_y1 = int(rel_y1 * scale1)
            vis_x1 = x_offset1 + rel_x1
            vis_y1 = y_offset1 + rel_y1

            # For piece 2
            pt2_rotated = edge_pt2  # Already in rotated space
            rel_x2 = pt2_rotated[0] - bounds2[0]
            rel_y2 = pt2_rotated[1] - bounds2[1]
            rel_x2 = int(rel_x2 * scale2)
            rel_y2 = int(rel_y2 * scale2)
            vis_x2 = x_offset2 + rel_x2
            vis_y2 = y_offset2 + rel_y2

            # Draw black circles for original edge points
            black_color = (0, 0, 0, 255)  # Black
            if 0 <= vis_x1 < visualization_width and 0 <= vis_y1 < visualization_height:
                cv2.circle(visualization, (vis_x1, vis_y1), 4, black_color, -1)  # Slightly larger

            if 0 <= vis_x2 < visualization_width and 0 <= vis_y2 < visualization_height:
                cv2.circle(visualization, (vis_x2, vis_y2), 4, black_color, -1)  # Slightly larger

        # Draw sample colors for comparison
        sample_size = 15  # Slightly smaller squares to fit more
        sample_spacing = 3  # Reduced spacing for tighter layout
        score_circle_radius = 5  # Size of score indicator circles
        sample_y1 = visualization_height - 140  # Position for first piece colors
        sample_y2 = visualization_height - 100  # Position for second piece colors
        score_y = visualization_height - 70     # Position for score circles

        # Calculate how many samples we can fit
        max_display_width = visualization_width - 100  # Leave margins
        max_columns = min(len(valid_indices), max_display_width // (sample_size + sample_spacing))

        # Calculate total width for centering
        total_width = max_columns * (sample_size + sample_spacing)
        start_x = (visualization_width - total_width) // 2

        # Draw color samples and scores in a vertically aligned grid
        for i in range(max_columns):
            idx = valid_indices[i]

            # Get rainbow color for this point
            total_sample_points = len(valid_indices)
            rainbow_color_i = rainbow_color(i, total_sample_points, with_alpha=True)

            # Calculate x position for this column
            sample_x = start_x + i * (sample_size + sample_spacing)

            # Top row: piece1 color sample with rainbow border
            color1 = (int(colors1[idx][0]), int(colors1[idx][1]), int(colors1[idx][2]), 255)
            cv2.rectangle(visualization,
                          (sample_x, sample_y1),
                          (sample_x + sample_size, sample_y1 + sample_size),
                          color1, -1)

            # Add rainbow border to match sample point
            cv2.rectangle(visualization,
                          (sample_x-2, sample_y1-2),
                          (sample_x + sample_size+2, sample_y1 + sample_size+2),
                          rainbow_color_i, 2)

            # Bottom row: piece2 color sample with rainbow border
            color2 = (int(colors2[idx][0]), int(colors2[idx][1]), int(colors2[idx][2]), 255)
            cv2.rectangle(visualization,
                          (sample_x, sample_y2),
                          (sample_x + sample_size, sample_y2 + sample_size),
                          color2, -1)

            # Add rainbow border to match sample point
            cv2.rectangle(visualization,
                          (sample_x-2, sample_y2-2),
                          (sample_x + sample_size+2, sample_y2 + sample_size+2),
                          rainbow_color_i, 2)

            # Draw vertical connection line between the two colors
            match_quality = 1.0 - (color_diffs[i] / max_possible_diff)
            line_color = (int(255 * (1 - match_quality)), int(255 * match_quality), 0, 255)
            cv2.line(visualization,
                     (sample_x + sample_size//2, sample_y1 + sample_size),
                     (sample_x + sample_size//2, sample_y2),
                     line_color, 2)

            # Use the individual similarity score for this specific pixel pair
            individual_similarity = color_diffs[i]  # This is already 0-1 where 1 = perfect match

            # Create color based on individual similarity (red=bad, green=good)
            individual_score_color = (
                0,                                    # B: No blue
                int(255 * individual_similarity),     # G: More green for better matches
                int(255 * (1 - individual_similarity)), # R: More red for worse matches
                255                                   # Alpha
            )

            cv2.circle(visualization,
                       (sample_x + sample_size//2, score_y),
                       score_circle_radius,
                       individual_score_color,
                       -1)  # -1 = filled circle

        # Add labels for the color rows
        cv2.putText(visualization, "L:",
                    (start_x - 60, sample_y1 + sample_size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 1)

        cv2.putText(visualization, "R:",
                    (start_x - 60, sample_y2 + sample_size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 1)

        # Add legend for score circles
        legend_x = visualization_width - 150
        legend_y = score_y

        # Modify the filename format to match_p1_e1_p7_e2.png

        if piece1.piece_id < piece2.piece_id:
            match_filename = f"match_p{piece1.piece_id:02d}_e{edge1_idx}_p{piece2.piece_id:02d}_e{edge2_idx}.png"
        else:
            match_filename = f"match_p{piece2.piece_id:02d}_e{edge2_idx}_p{piece1.piece_id:02d}_e{edge1_idx}.png"

        match_filepath = os.path.join(output_matching, match_filename)
        cv2.imwrite(match_filepath, visualization)

        # Also display the visualization
        # display_image_cv2(f"Edge Match (Score: {score:.3f})", visualization)

        # ### TXT FILE ###
        #
        # # Create detailed edge analysis file
        # analysis_filename = f"edge_analysis_p{piece1.piece_id}_e{edge1_idx}_p{piece2.piece_id}_e{edge2_idx}.txt"
        # analysis_filepath = os.path.join(output_matching, analysis_filename)
        #
        # with open(analysis_filepath, 'w') as f:
        #     f.write(f"EDGE MATCHING ANALYSIS\n")
        #     f.write(f"=====================\n\n")
        #     f.write(f"Piece 1 ID: {piece1.piece_id}, Edge: {edge1_idx}\n")
        #     f.write(f"Piece 2 ID: {piece2.piece_id}, Edge: {edge2_idx}\n")
        #     f.write(f"Final Matching Score: {score:.4f}\n")
        #     f.write(f"Color Score: {color_score:.4f}\n")
        #     f.write(f"Length Score: {length_score:.4f}\n")
        #     f.write(f"Shape Score: {shape_score:.4f}\n\n")
        #
        #     # Edge vertices for both pieces
        #     f.write("PIECE 1 EDGE VERTICES:\n")
        #     start_corner1 = piece1.rotated_corners[edge1_idx] if hasattr(piece1, 'rotated_corners') and piece1.rotated_corners else piece1.relative_contour[piece1.edges[edge1_idx][1]][0]
        #     end_corner1 = piece1.rotated_corners[(edge1_idx + 1) % len(piece1.rotated_corners)] if hasattr(piece1, 'rotated_corners') and piece1.rotated_corners else piece1.relative_contour[piece1.edges[edge1_idx][2]][0]
        #     f.write(f"  Start vertex: ({start_corner1[0]:.2f}, {start_corner1[1]:.2f})\n")
        #     f.write(f"  End vertex: ({end_corner1[0]:.2f}, {end_corner1[1]:.2f})\n")
        #     f.write(f"  Edge length: {edge1_length:.2f}\n\n")
        #
        #     f.write("PIECE 2 EDGE VERTICES:\n")
        #     start_corner2 = piece2.rotated_corners[edge2_idx] if hasattr(piece2, 'rotated_corners') and piece2.rotated_corners else piece2.relative_contour[piece2.edges[edge2_idx][1]][0]
        #     end_corner2 = piece2.rotated_corners[(edge2_idx + 1) % len(piece2.rotated_corners)] if hasattr(piece2, 'rotated_corners') and piece2.rotated_corners else piece2.relative_contour[piece2.edges[edge2_idx][2]][0]
        #     f.write(f"  Start vertex: ({start_corner2[0]:.2f}, {start_corner2[1]:.2f})\n")
        #     f.write(f"  End vertex: ({end_corner2[0]:.2f}, {end_corner2[1]:.2f})\n")
        #     f.write(f"  Edge length: {edge2_length:.2f}\n\n")
        #
        #     f.write("SAMPLING DETAILS:\n")
        #     f.write(f"Number of sample points: {N}\n")
        #     f.write(f"Inward offset: {inward_offset} pixels\n")
        #     f.write(f"Valid color points: {len(valid_indices)}\n\n")
        #
        #     # Detailed point analysis
        #     f.write("POINT-BY-POINT ANALYSIS:\n")
        #     f.write("=" * 80 + "\n")
        #
        #     for i, idx in enumerate(valid_indices):
        #         f.write(f"\nPoint {i+1} (Sample index {idx}):\n")
        #         f.write("-" * 40 + "\n")
        #
        #         # Original edge points
        #         edge_pt1 = sample_points1[idx]  # Now it's directly (x, y)
        #         edge_pt2 = sample_points2[idx]  # Now it's directly (x, y)
        #         f.write(f"PIECE 1:\n")
        #         f.write(f"  Edge point: ({edge_pt1[0]:.2f}, {edge_pt1[1]:.2f})\n")
        #
        #         # Get the inward direction vector (recreate the calculation from get_color_at_point)
        #         # This is approximate since we'd need to call the function to get exact values
        #         inward_pt1 = sample_inward_points1[idx]
        #         direction1 = (inward_pt1[0] - edge_pt1[0], inward_pt1[1] - edge_pt1[1])
        #         direction_length1 = np.sqrt(direction1[0]**2 + direction1[1]**2)
        #         if direction_length1 > 0:
        #             unit_direction1 = (direction1[0]/direction_length1, direction1[1]/direction_length1)
        #         else:
        #             unit_direction1 = (0, 0)
        #
        #         f.write(f"  Inward direction vector: ({unit_direction1[0]:.4f}, {unit_direction1[1]:.4f})\n")
        #         f.write(f"  Inward sample point: ({inward_pt1[0]:.2f}, {inward_pt1[1]:.2f})\n")
        #         f.write(f"  Sampled color (BGR): ({colors1[idx][0]}, {colors1[idx][1]}, {colors1[idx][2]})\n")
        #
        #         f.write(f"PIECE 2:\n")
        #         f.write(f"  Edge point: ({edge_pt2[0]:.2f}, {edge_pt2[1]:.2f})\n")
        #
        #         inward_pt2 = sample_inward_points2[idx]
        #         direction2 = (inward_pt2[0] - edge_pt2[0], inward_pt2[1] - edge_pt2[1])
        #         direction_length2 = np.sqrt(direction2[0]**2 + direction2[1]**2)
        #         if direction_length2 > 0:
        #             unit_direction2 = (direction2[0]/direction_length2, direction2[1]/direction_length2)
        #         else:
        #             unit_direction2 = (0, 0)
        #
        #         f.write(f"  Inward direction vector: ({unit_direction2[0]:.4f}, {unit_direction2[1]:.4f})\n")
        #         f.write(f"  Inward sample point: ({inward_pt2[0]:.2f}, {inward_pt2[1]:.2f})\n")
        #         f.write(f"  Sampled color (BGR): ({colors2[idx][0]}, {colors2[idx][1]}, {colors2[idx][2]})\n")
        #
        #         f.write(f"COLOR COMPARISON:\n")
        #         f.write(f"  Individual similarity score: {color_diffs[i]:.4f}\n")
        #
        #         # Calculate RGB distance for reference
        #         rgb_dist = np.sqrt(np.sum((colors1[idx].astype(float) - colors2[idx].astype(float))**2))
        #         f.write(f"  RGB distance: {rgb_dist:.2f}\n")
        #
        # print(f"Detailed analysis saved to: {analysis_filepath}")

    return score


# In[648]:


all_matches = []

pieces_to_match = [my_puzzle.all_pieces[i] for i in
                   my_puzzle.borders_pieces + my_puzzle.corners_pieces + my_puzzle.middle_pieces]

tested_pairs = set()

for i, piece1 in enumerate(pieces_to_match):
    for edge1_idx in range(len(piece1.edges)):
        for j, piece2 in enumerate(pieces_to_match):
            if piece1.piece_id == piece2.piece_id:
                continue

            for edge2_idx in range(len(piece2.edges)):
                edge_pair = tuple(sorted([
                    (piece1.piece_id, edge1_idx),
                    (piece2.piece_id, edge2_idx)
                ]))

                if edge_pair in tested_pairs:
                    continue

                tested_pairs.add(edge_pair)

                score = compute_edge_matching_score(piece1, edge1_idx, piece2, edge2_idx, debug=True)

                match_info = {
                    'piece1_id': piece1.piece_id,
                    'edge1_idx': edge1_idx,
                    'piece2_id': piece2.piece_id,
                    'edge2_idx': edge2_idx,
                    'score': score
                }
                all_matches.append(match_info)

                # Update matching_edges for both pieces
                # Update matching_edges for both pieces (only if score > 0)
                if score > 0:
                    # For piece1: store from perspective of edge1_idx
                    if edge1_idx not in piece1.matching_edges:
                        piece1.matching_edges[edge1_idx] = {}
                    if piece2.piece_id not in piece1.matching_edges[edge1_idx]:
                        piece1.matching_edges[edge1_idx][piece2.piece_id] = {}
                    piece1.matching_edges[edge1_idx][piece2.piece_id][edge2_idx] = score

                    # For piece2: store from perspective of edge2_idx
                    if edge2_idx not in piece2.matching_edges:
                        piece2.matching_edges[edge2_idx] = {}
                    if piece1.piece_id not in piece2.matching_edges[edge2_idx]:
                        piece2.matching_edges[edge2_idx][piece1.piece_id] = {}
                    piece2.matching_edges[edge2_idx][piece1.piece_id][edge1_idx] = score

my_puzzle.all_edge_matches = all_matches

# Sort matches for each piece
for piece in pieces_to_match:
    piece.sorted_matches = {}

    for edge_id in piece.matching_edges:
        sorted_matches = []
        for other_piece_id, other_edges in piece.matching_edges[edge_id].items():
            for other_edge_id, score in other_edges.items():
                sorted_matches.append((other_piece_id, other_edge_id, score))

        # Sort by score (highest first)
        sorted_matches.sort(key=lambda x: x[2], reverse=True)
        piece.sorted_matches[edge_id] = sorted_matches

my_puzzle.all_edge_matches = all_matches


# In[649]:


# Sort all matches by score (highest first)
all_matches.sort(key=lambda x: x['score'], reverse=True)

# Print overall results
print(f"\nFound {len(all_matches)} total potential matches with score > 0.5")

# Create a dictionary mapping (piece_id, edge_idx) to its best matches for quick lookup
edge_match_lookup = {}
for match in all_matches:
    key = (match['piece1_id'], match['edge1_idx'])
    if key not in edge_match_lookup:
        edge_match_lookup[key] = []
    edge_match_lookup[key].append(match)

# Sort the matches for each edge by score
for key in edge_match_lookup:
    edge_match_lookup[key].sort(key=lambda x: x['score'], reverse=True)

# Print top 3 matches for each edge
print("\nTop matches for each edge:")
for key in sorted(edge_match_lookup.keys()):
    piece_id, edge_idx = key
    matches = edge_match_lookup[key][:3]  # Get top 3 matches for this edge

    print(f"Piece {piece_id} edge {edge_idx} - Top matches:")
    for i, match in enumerate(matches):
        print(f"  Match {i+1}: with Piece {match['piece2_id']} edge {match['edge2_idx']} - Score: {match['score']:.3f}")

# Store matches in puzzle object for later use
my_puzzle.all_edge_matches = all_matches
my_puzzle.edge_match_lookup = edge_match_lookup

print("Match calculation complete and stored in my_puzzle.all_edge_matches and my_puzzle.edge_match_lookup")


# In[650]:


class GridCanvas:
    def __init__(self, width, height, rows, cols, column_spacing=100, row_spacing=50, margin=100):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.column_spacing = column_spacing
        self.row_spacing = row_spacing
        self.margin = margin

        # Initialize 2D grid with -1 values (no pieces placed yet)
        # Each cell will store a tuple of (piece_id, rotation)
        self.rep_array = [[(-1, 0) for _ in range(cols)] for _ in range(rows)]

        # Create canvas for visualization - only used when saving
        self.canvas = np.ones((height, width, 4), dtype=np.uint8) * 255
        self.canvas[:,:,3] = 255  # Full alpha

    def place_piece(self, piece_id, row, col, rotation=0):
        # Validate position
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            print(f"Position ({row},{col}) is outside the grid bounds.")
            return False

        # Validate rotation
        rotation = rotation % 4  # Ensure rotation is 0-3

        # Store the piece ID and rotation in the grid
        self.rep_array[row][col] = (piece_id, rotation)

        return True

    def print_grid(self):
        print("Grid Arrangement (Piece IDs):")
        for row in range(self.rows):
            row_str = ""
            for col in range(self.cols):
                cell = self.rep_array[row][col]
                piece_id, rotation = cell
                row_str += f"{piece_id} "
            print(row_str.strip())

    def render_grid(self):
        # Clear the canvas first
        self.canvas = np.ones((self.height, self.width, 4), dtype=np.uint8) * 255
        self.canvas[:,:,3] = 255  # Full alpha

        for row in range(self.rows):
            for col in range(self.cols):
                piece_id, rotation = self.rep_array[row][col]

                # Calculate position - this is where the top-left of the image should go
                x = self.margin + col * self.column_spacing
                y = self.margin + row * self.row_spacing

                # Skip empty cells
                if piece_id == -1:
                    cv2.circle(self.canvas, (x, y), 5, (255, 0, 255, 255), -1)
                    continue

                # Get the piece from my_puzzle
                if piece_id in my_puzzle.all_pieces:
                    piece = my_puzzle.all_pieces[piece_id]

                    # Use the rotated_image
                    if hasattr(piece, 'rotated_image') and piece.rotated_image is not None:
                        rotated_image = piece.rotated_image.copy()

                        # Apply additional rotations if needed
                        for _ in range(rotation):
                            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)

                        # Get piece dimensions
                        piece_height, piece_width = rotated_image.shape[:2]

                        # Handle case where the image might extend beyond canvas boundaries
                        # but ensure that the portion we can place starts exactly at (x,y)
                        y_start = max(0, y)
                        x_start = max(0, x)
                        canvas_y_end = min(self.height, y + piece_height)
                        canvas_x_end = min(self.width, x + piece_width)

                        # Calculate the portion of the image to use
                        # This is adjusted to ensure top-left alignment
                        img_y_start = y_start - y  # Will be 0 if y ≥ 0
                        img_x_start = x_start - x  # Will be 0 if x ≥ 0
                        img_y_end = img_y_start + (canvas_y_end - y_start)
                        img_x_end = img_x_start + (canvas_x_end - x_start)

                        # Only proceed if we have valid dimensions
                        if (canvas_y_end > y_start) and (canvas_x_end > x_start):
                            # Alpha blending for the rotated image
                            for c in range(4):
                                mask = rotated_image[img_y_start:img_y_end, img_x_start:img_x_end, 3:4] / 255.0
                                if c < 3:  # RGB channels
                                    self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, c] = (
                                        self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, c] * (1 - mask[:,:,0]) +
                                        rotated_image[img_y_start:img_y_end, img_x_start:img_x_end, c] * mask[:,:,0]
                                    )
                                else:  # Alpha channel
                                    self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, 3] = np.maximum(
                                        self.canvas[y_start:canvas_y_end, x_start:canvas_x_end, 3],
                                        rotated_image[img_y_start:img_y_end, img_x_start:img_x_end, 3]
                                    )

                            # Draw orange rectangle around the piece's bounding box
                            # cv2.rectangle(self.canvas,
                            #              (x_start, y_start),
                            #              (canvas_x_end, canvas_y_end),
                            #              (0, 165, 255, 255), 2)  # BGR format: Orange
                    else:
                        print(f"Warning: Piece {piece_id} does not have a rotated_image")

                # Draw a purple dot at the exact grid coordinate to verify placement
                # cv2.circle(self.canvas, (x, y), 5, (255, 0, 255, 255), -1)  # BGR format: Purple, -1 means filled circle

    def save(self, filename="grid_arrangement.png"):
        # Render the grid before saving
        self.render_grid()

        # Ensure the output directory exists
        output_dir = f"{base_path}/grid_arrangement"
        os.makedirs(output_dir, exist_ok=True)

        # Full path
        filepath = os.path.join(output_dir, filename)

        # Save the canvas image
        cv2.imwrite(filepath, self.canvas)
        print(f"Grid arrangement saved to {filepath}")


# In[623]:


# # Create a grid canvas
# grid_canvas = GridCanvas(
#     width=1800,
#     height=1200,
#     rows=puzzle_r,
#     cols=puzzle_c,
#     column_spacing=250,
#     row_spacing=250,
#     margin=100
# )
#
# # Place pieces with direct IDs - only updates the rep_array, not the visualization
#
# # Manual placement for testing
# grid_canvas.place_piece(17, 0, 0, rotation=1)
# grid_canvas.place_piece(14, 0, 1, rotation=1)
# grid_canvas.place_piece(4, 0, 2, rotation=1)
# grid_canvas.place_piece(1, 0, 3, rotation=1)
# grid_canvas.place_piece(13, 0, 4, rotation=1)
# grid_canvas.place_piece(6, 0, 5, rotation=2)
#
# grid_canvas.place_piece(12, 1, 0, rotation=0)
# grid_canvas.place_piece(18, 2, 0, rotation=0)
#
# grid_canvas.place_piece(5, 1, 5, rotation=2)
# grid_canvas.place_piece(7, 2, 5, rotation=2)
#
# grid_canvas.place_piece(19, 3, 0, rotation=3)
# grid_canvas.place_piece(16, 3, 1, rotation=3)
# grid_canvas.place_piece(21, 3, 2, rotation=3)
# grid_canvas.place_piece(9, 3, 3, rotation=3)
# grid_canvas.place_piece(22, 3, 4, rotation=3)
# grid_canvas.place_piece(2, 3, 5, rotation=3)
#
# # Print the grid to console
# grid_canvas.print_grid()
#
# # When ready to save, the render_grid() function is called by save() to create the visualization
# grid_canvas.save("my_puzzle_arrangement.png")


# In[624]:


def find_adjacent_coordinates_to_explore(rep_array, placed_pieces):
    """
    Find all valid adjacent coordinates to explore next.
    Returns a list of (row, col) tuples for empty cells adjacent to placed pieces.
    """
    rows = len(rep_array)
    cols = len(rep_array[0])

    # Directions: NORTH, EAST, SOUTH, WEST
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    adjacent_coords = set()  # Use set to avoid duplicates

    # Check all placed pieces for adjacent empty cells
    for row in range(rows):
        for col in range(cols):
            piece_id, rotation = rep_array[row][col]

            # Skip empty cells
            if piece_id == -1:
                continue

            # Check all four directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                # Ensure new coordinates are within bounds
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    # Ensure the cell is empty
                    if rep_array[new_row][new_col][0] == -1:
                        adjacent_coords.add((new_row, new_col))

    return list(adjacent_coords)


# In[625]:


def get_adjacent_pieces_for_coordinate(adjacent_info, target_row, target_col):
    """
    Helper function to get all pieces adjacent to a specific coordinate.
    Returns list of [piece_id, direction] for the given coordinate.
    """
    adjacent_pieces = []
    for piece_id, (row, col), direction in adjacent_info:
        if row == target_row and col == target_col:
            adjacent_pieces.append([piece_id, direction])

    return adjacent_pieces


# In[626]:


def print_adjacent_analysis(adjacent_info):
    """
    Print a nice analysis of the adjacent coordinates.
    """
    direction_names = ["NORTH", "EAST", "SOUTH", "WEST"]

    print("Adjacent coordinates analysis:")
    print("=" * 50)

    # Group by coordinates
    coord_groups = {}
    for piece_id, coords, direction in adjacent_info:
        if coords not in coord_groups:
            coord_groups[coords] = []
        coord_groups[coords].append([piece_id, direction])

    for coords, adjacent_pieces in sorted(coord_groups.items()):
        row, col = coords
        print(f"Coordinate ({row}, {col}):")
        for piece_id, direction in adjacent_pieces:
            # The direction is FROM the piece TO the coordinate
            # So if direction is EAST, the piece is to the WEST of the coordinate
            opposite_direction = (direction + 2) % 4
            print(f"  - Piece {piece_id} is to the {direction_names[opposite_direction]} (direction {direction})")
        print()


# In[627]:


# Example usage with your current setup:
def test_adjacent_finder(grid_canvas):
    """
    Test the enhanced adjacent finder with your current grid setup.
    """
    placed_pieces = []

    # Find pieces that are currently placed (not -1)
    for row in range(len(grid_canvas.rep_array)):
        for col in range(len(grid_canvas.rep_array[0])):
            piece_id, rotation = grid_canvas.rep_array[row][col]
            if piece_id != -1:
                placed_pieces.append(piece_id)

    print(f"Currently placed pieces: {placed_pieces}")

    # Find adjacent coordinates
    adjacent_info = find_adjacent_coordinates_to_explore(grid_canvas.rep_array, placed_pieces)

    print(f"\nAdjacent coordinates (raw): {adjacent_info}")

    # Print nice analysis
    print_adjacent_analysis(adjacent_info)

    return adjacent_info


# In[628]:


def display_piece_matches(piece):
    """
    Display piece matching edges in a readable format
    """
    print(f"Piece {piece.piece_id} matching edges:")
    print("=" * 40)

    if not piece.sorted_matches:
        print("No matches found")
        return

    for edge_id in sorted(piece.sorted_matches.keys()):
        print(f"\nEdge {edge_id}:")

        matches = piece.sorted_matches[edge_id]
        if not matches:
            print("  No matches")
            continue

        for other_piece_id, other_edge_id, score in matches:
            print(f"  -> matching with piece {other_piece_id}, edge {other_edge_id}, score: {score:.4f}")


# In[629]:


def get_spiral_coordinate(counter, rows, cols):
    """
    Generate the coordinate at position 'counter' in a spiral traversal.
    Returns (row, col) tuple for the given counter position.
    """
    if counter < 0 or counter >= rows * cols:
        return None

    # Generate all coordinates in spiral order
    coordinates = []
    top, bottom, left, right = 0, rows - 1, 0, cols - 1

    while top <= bottom and left <= right:
        # Go right along the top row
        for col in range(left, right + 1):
            coordinates.append((top, col))
        top += 1

        # Go down along the right column
        for row in range(top, bottom + 1):
            coordinates.append((row, right))
        right -= 1

        # Go left along the bottom row
        if top <= bottom:
            for col in range(right, left - 1, -1):
                coordinates.append((bottom, col))
            bottom -= 1

        # Go up along the left column
        if left <= right:
            for row in range(bottom, top - 1, -1):
                coordinates.append((row, left))
            left += 1

    return coordinates[counter] if counter < len(coordinates) else None


# In[630]:


def get_nearby_occupied_cells(grid_canvas, target_row, target_col):
    """
    Get nearby occupied cells (non-diagonal, within bounds) and their directions
    relative to the target cell.

    Returns: list of tuples [(piece_id, direction_name, neighbor_row, neighbor_col), ...]
    where direction_name is the direction FROM the target cell TO the neighbor cell
    """
    rows = grid_canvas.rows
    cols = grid_canvas.cols

    # Directions: NORTH, EAST, SOUTH, WEST
    # Each direction is (row_delta, col_delta) FROM target TO neighbor
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction_names = ["NORTH", "EAST", "SOUTH", "WEST"]

    nearby_occupied = []

    for i, (dr, dc) in enumerate(directions):
        neighbor_row = target_row + dr
        neighbor_col = target_col + dc

        # Check if neighbor is within bounds
        if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
            piece_id, rotation = grid_canvas.rep_array[neighbor_row][neighbor_col]

            # Check if neighbor is occupied (not -1)
            if piece_id != -1:
                # The direction name is where the neighbor is relative to target
                # So if we go NORTH from target to find neighbor,
                # the neighbor is NORTH of the target
                neighbor_direction = direction_names[i]  # Same direction
                nearby_occupied.append((piece_id, neighbor_direction, neighbor_row, neighbor_col))

    return nearby_occupied


# In[631]:


def get_direction_name_from_index(index):
    """Convert direction index to name"""
    direction_names = ["NORTH", "EAST", "SOUTH", "WEST"]
    return direction_names[index]

def get_direction_index_from_name(name):
    """Convert direction name to index"""
    direction_map = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}
    return direction_map[name]

def calculate_rotation_needed(piece, target_edge_id, target_direction_name):
    """
    Calculate how many rotations (0-3) are needed to orient a specific edge
    to face a target direction.

    Args:
        piece: The puzzle piece object
        target_edge_id: Which edge should face the target direction
        target_direction_name: Direction the edge should face ("NORTH", "EAST", "SOUTH", "WEST")

    Returns:
        Number of clockwise rotations needed (0-3)
    """
    target_direction_index = get_direction_index_from_name(target_direction_name)

    # Get the current direction of the target edge
    if hasattr(piece, 'edge_direction_map') and piece.edge_direction_map:
        current_direction_index = piece.edge_direction_map[target_edge_id]
    else:
        print(f"Warning: Piece {piece.piece_id} has no edge_direction_map")
        return 0

    # Calculate rotations needed
    # Each clockwise rotation adds 1 to the direction index (mod 4)
    rotations_needed = (target_direction_index - current_direction_index) % 4

    return rotations_needed


# In[632]:


def is_corner_position(row, col, rows, cols):
    """Check if a position is a corner of the grid"""
    return (row == 0 and col == 0) or \
        (row == 0 and col == cols - 1) or \
        (row == rows - 1 and col == 0) or \
        (row == rows - 1 and col == cols - 1)

def is_border_position(row, col, rows, cols):
    """Check if a position is on the border of the grid (but not a corner)"""
    if is_corner_position(row, col, rows, cols):
        return False
    return row == 0 or row == rows - 1 or col == 0 or col == cols - 1

def get_piece_type(piece):
    """Get piece type based on number of flat edges"""
    if hasattr(piece, 'edges'):
        flat_edges = [i for i, edge in enumerate(piece.edges) if edge[3] == 0]
        num_flat_edges = len(flat_edges)
        if num_flat_edges == 2:
            return "CORNER"
        elif num_flat_edges == 1:
            return "BORDER"
        else:
            return "MIDDLE"
    return "UNKNOWN"


# In[633]:


def get_priority_neighbor(grid_canvas, row, col):
    """Get neighbor in priority order: EAST, WEST, NORTH, SOUTH"""
    directions = [
        (0, 1, "EAST"),   # East first
        (0, -1, "WEST"),  # Then West
        (-1, 0, "NORTH"), # Then North
        (1, 0, "SOUTH")   # Finally South
    ]

    for dr, dc, direction_name in directions:
        neighbor_row = row + dr
        neighbor_col = col + dc

        if (0 <= neighbor_row < grid_canvas.rows and
                0 <= neighbor_col < grid_canvas.cols):
            piece_id, rotation = grid_canvas.rep_array[neighbor_row][neighbor_col]
            if piece_id != -1:
                return piece_id, direction_name, neighbor_row, neighbor_col

    return None

def calculate_edge_vector(piece, edge_id):
    """Calculate vector from lower corner ID to higher corner ID for an edge"""
    edge_data = piece.edges[edge_id]
    corner1_idx = edge_data[1]
    corner2_idx = edge_data[2]

    # Get the actual corner coordinates
    if hasattr(piece, 'rotated_corners') and piece.rotated_corners:
        corner1 = piece.rotated_corners[edge_id]
        corner2 = piece.rotated_corners[(edge_id + 1) % len(piece.rotated_corners)]
    else:
        corner1 = piece.relative_contour[corner1_idx][0]
        corner2 = piece.relative_contour[corner2_idx][0]

    # Vector from corner1 to corner2
    vector = np.array([corner2[0] - corner1[0], corner2[1] - corner1[1]])
    return vector

def orient_middle_piece(candidate_piece, reference_piece, reference_edge_id, candidate_edge_id):
    """Orient a middle piece to match a reference piece's edge"""

    # Get the reference vector
    ref_vector = calculate_edge_vector(reference_piece, reference_edge_id)

    # Get the candidate vector (inverted for matching)
    candidate_vector = -calculate_edge_vector(candidate_piece, candidate_edge_id)

    # Calculate angle between vectors
    ref_angle = np.arctan2(ref_vector[1], ref_vector[0])
    candidate_angle = np.arctan2(candidate_vector[1], candidate_vector[0])

    angle_diff = ref_angle - candidate_angle
    angle_deg = np.degrees(angle_diff)

    # Store rotation angle
    candidate_piece.rotation_angle = angle_deg

    # Apply the same expansion and rotation process as for border pieces
    image_with_center = candidate_piece.transparent_image.copy()
    # cv2.circle(image_with_center, candidate_piece.relative_center, 2, (0, 0, 255, 255), -1)

    (h, w) = image_with_center.shape[:2]

    # Expand canvas
    old_h, old_w = image_with_center.shape[:2]
    expansion_ratio = 1.8
    new_w = int(old_w * expansion_ratio)
    new_h = int(old_h * expansion_ratio)

    x_offset = (new_w - old_w) // 2
    y_offset = (new_h - old_h) // 2

    candidate_piece.expansion_ratio = expansion_ratio
    candidate_piece.expansion_offset = (x_offset, y_offset)

    expanded_image = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    expanded_image[y_offset:y_offset+old_h, x_offset:x_offset+old_w] = image_with_center
    candidate_piece.expanded_image = expanded_image

    new_center_x = candidate_piece.relative_center[0] + x_offset
    new_center_y = candidate_piece.relative_center[1] + y_offset
    new_center = (new_center_x, new_center_y)

    M = cv2.getRotationMatrix2D(new_center, angle_deg, 1.0)
    candidate_piece.rotation_matrix = M

    rotated_image = cv2.warpAffine(expanded_image, M, (new_w, new_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
    candidate_piece.rotated_image = rotated_image

    # Calculate rotated center and corners
    rotated_center = M @ np.array([new_center[0], new_center[1], 1])
    candidate_piece.rotated_center = tuple(np.round(rotated_center[:2]).astype(int))

    # Rotate all corners and contour
    candidate_piece.rotated_corners = []
    candidate_piece.rotated_edges = []

    rotated_contour = []
    for point in candidate_piece.relative_contour:
        pt = point[0]
        pt_offset = (pt[0] + x_offset, pt[1] + y_offset)
        pt_rot = M @ np.array([pt_offset[0], pt_offset[1], 1])
        pt_rot = tuple(np.round(pt_rot[:2]).astype(int))
        rotated_contour.append(pt_rot)

    candidate_piece.rotated_contour = np.array(rotated_contour)

    # Rotate corners and edges
    for edge_idx, edge_data in enumerate(candidate_piece.edges):
        edge_id = edge_data[0]
        corner1_idx = edge_data[1]
        corner2_idx = edge_data[2]
        edge_type = edge_data[3]

        c1 = candidate_piece.relative_contour[corner1_idx][0]
        c2 = candidate_piece.relative_contour[corner2_idx][0]

        c1_offset = (c1[0] + x_offset, c1[1] + y_offset)
        c2_offset = (c2[0] + x_offset, c2[1] + y_offset)

        c1_rot = M @ np.array([c1_offset[0], c1_offset[1], 1])
        c2_rot = M @ np.array([c2_offset[0], c2_offset[1], 1])

        c1_rot = tuple(np.round(c1_rot[:2]).astype(int))
        c2_rot = tuple(np.round(c2_rot[:2]).astype(int))

        candidate_piece.rotated_corners.append(c1_rot)

        rotated_edge = {
            'edge_id': edge_id,
            'start_corner': c1_rot,
            'end_corner': c2_rot,
            'edge_type': edge_type
        }
        candidate_piece.rotated_edges.append(rotated_edge)

    # Calculate final edge direction map based on the new orientation
    # We need to determine how the edges are oriented after rotation
    candidate_piece.final_edge_direction_map = {}

    # For middle pieces, we calculate directions relative to the rotated state
    for edge_idx in range(len(candidate_piece.edges)):
        # Calculate the direction of each edge after rotation
        start_corner = candidate_piece.rotated_corners[edge_idx]
        end_corner = candidate_piece.rotated_corners[(edge_idx + 1) % len(candidate_piece.rotated_corners)]

        edge_vector = np.array([end_corner[0] - start_corner[0], end_corner[1] - start_corner[1]])
        edge_angle = np.arctan2(edge_vector[1], edge_vector[0])

        # Convert angle to direction (0=NORTH, 1=EAST, 2=SOUTH, 3=WEST)
        # Normalize angle to [0, 2π]
        if edge_angle < 0:
            edge_angle += 2 * np.pi

        # Map angle to direction
        direction_idx = int((edge_angle + np.pi/4) / (np.pi/2)) % 4
        direction_idx = (direction_idx + 1) % 4  # Adjust for coordinate system

        candidate_piece.final_edge_direction_map[edge_idx] = direction_idx


# In[635]:


def generate_vertices_spiral(grid_canvas):
    # Vertices grid dimensions
    vertex_rows = grid_canvas.rows + 1
    vertex_cols = grid_canvas.cols + 1

    # Direction mapping: NORTH=0, EAST=1, SOUTH=2, WEST=3

    # Corner direction mappings for each cell position relative to vertex
    corner_mappings = {
        'NORTH-WEST': (0, 3),   # NORTH + WEST
        'NORTH-EAST': (0, 1),   # NORTH + EAST
        'SOUTH-WEST': (2, 3),   # SOUTH + WEST
        'SOUTH-EAST': (2, 1)    # SOUTH + EAST
    }

    print(f"Cell grid: {grid_canvas.rows} x {grid_canvas.cols}")
    print(f"Vertex grid: {vertex_rows} x {vertex_cols}")

    # Generate complete spiral to identify corners
    all_spiral_coords = []
    top, bottom, left, right = 0, vertex_rows - 1, 0, vertex_cols - 1

    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            all_spiral_coords.append((top, col))
        top += 1

        for row in range(top, bottom + 1):
            all_spiral_coords.append((row, right))
        right -= 1

        if top <= bottom:
            for col in range(right, left - 1, -1):
                all_spiral_coords.append((bottom, col))
            bottom -= 1

        if left <= right:
            for row in range(bottom, top - 1, -1):
                all_spiral_coords.append((row, left))
            left += 1

    # Identify corners
    corners_to_skip = set()
    for i in range(len(all_spiral_coords)):
        if i == 0 or i == len(all_spiral_coords) - 1:
            corners_to_skip.add(all_spiral_coords[i])
        else:
            prev_coord = all_spiral_coords[i-1]
            curr_coord = all_spiral_coords[i]
            next_coord = all_spiral_coords[i+1]

            dir1 = (curr_coord[0] - prev_coord[0], curr_coord[1] - prev_coord[1])
            dir2 = (next_coord[0] - curr_coord[0], next_coord[1] - curr_coord[1])

            if dir1 != dir2:
                corners_to_skip.add(curr_coord)

    # Build compact data structure
    vertices_data = []
    total_puzzle_pieces = grid_canvas.rows * grid_canvas.cols
    counter = 1

    for coord in all_spiral_coords:
        vertex_row, vertex_col = coord

        if coord in corners_to_skip:
            print(f"skipped ({vertex_row}, {vertex_col})")
            continue

        if counter >= total_puzzle_pieces:
            print(f"Reached puzzle length limit at count {counter}, stopping")
            break

        # Find adjacent cells for this vertex
        adjacent_cells = []

        # Check all 4 possible adjacent cells
        cell_checks = [
            (vertex_row - 1, vertex_col - 1, 'SOUTH-EAST'),  # Top-left cell
            (vertex_row - 1, vertex_col, 'SOUTH-WEST'),      # Top-right cell
            (vertex_row, vertex_col - 1, 'NORTH-EAST'),      # Bottom-left cell
            (vertex_row, vertex_col, 'NORTH-WEST')           # Bottom-right cell
        ]

        for cell_row, cell_col, corner_position in cell_checks:
            if 0 <= cell_row < grid_canvas.rows and 0 <= cell_col < grid_canvas.cols:
                corner_dirs = corner_mappings[corner_position]
                adjacent_cells.append({
                    'cell': (cell_row, cell_col),
                    'corner_dirs': corner_dirs
                })

        vertex_data = {
            'vertex': (vertex_row, vertex_col),
            'adjacent_cells': adjacent_cells
        }

        vertices_data.append(vertex_data)

        print(f"{counter:2d}: vertex ({vertex_row}, {vertex_col}) -> {len(adjacent_cells)} cells")
        for cell_info in adjacent_cells:
            cell_pos = cell_info['cell']
            dirs = cell_info['corner_dirs']
            print(f"    Cell {cell_pos} corner dirs: {dirs}")

        counter += 1

    print(f"\nGenerated {len(vertices_data)} vertex data entries")
    return vertices_data

# Test and store the data
def create_vertices_lookup(grid_canvas):
    """Create and store the vertices data for later lookup"""
    vertices_data = generate_vertices_spiral(grid_canvas)
    return vertices_data

# Example usage:
# vertices_lookup = create_vertices_lookup(grid_canvas)


# In[636]:


vertices_lookup = create_vertices_lookup(reconstruction_canvas)


# In[637]:


def get_vertex_data_by_step(vertices_lookup, step_number):
    # Convert to 0-based index
    index = step_number - 1

    if 0 <= index < len(vertices_lookup):
        return vertices_lookup[index]
    else:
        return None

def print_vertex_data_for_step(vertices_lookup, step_number):
    """
    Print vertex data for a specific step in the same format as before.
    """
    vertex_data = get_vertex_data_by_step(vertices_lookup, step_number)

    if vertex_data is None:
        print(f"No vertex data for step {step_number}")
        return

    vertex_pos = vertex_data['vertex']
    adjacent_cells = vertex_data['adjacent_cells']

    print(f"Step {step_number}: vertex {vertex_pos} -> {len(adjacent_cells)} cells")
    for cell_info in adjacent_cells:
        cell_pos = cell_info['cell']
        dirs = cell_info['corner_dirs']
        print(f"    Cell {cell_pos} corner dirs: {dirs}")


# In[638]:


def get_corner_coordinate_by_direction(piece, direction_tuple):
    if not hasattr(piece, 'final_edge_direction_map') or not hasattr(piece, 'rotated_corners'):
        print(f"Piece {piece.piece_id} missing orientation data")
        return None

    # Find edges that match the two directions
    matching_edges = []
    for edge_id, edge_direction in piece.final_edge_direction_map.items():
        if edge_direction in direction_tuple:
            matching_edges.append(edge_id)

    if len(matching_edges) != 2:
        print(f"Could not find exactly 2 matching edges for directions {direction_tuple}")
        return None

    # The corner is where these two edges meet
    # Since edges are consecutive, we need to find the shared corner
    edge1, edge2 = matching_edges

    # Check if edges are consecutive
    if (edge1 + 1) % 4 == edge2:
        # edge2 follows edge1, so the shared corner is the end of edge1
        corner_idx = (edge1 + 1) % 4
    elif (edge2 + 1) % 4 == edge1:
        # edge1 follows edge2, so the shared corner is the end of edge2
        corner_idx = (edge2 + 1) % 4
    else:
        print(f"Edges {edge1} and {edge2} are not consecutive")
        return None

    return piece.rotated_corners[corner_idx]

def translate_piece_to_align_corners(new_piece, new_corner_dirs, prev_piece, prev_corner_dirs):
    # Get corner coordinates
    new_corner = get_corner_coordinate_by_direction(new_piece, new_corner_dirs)
    prev_corner = get_corner_coordinate_by_direction(prev_piece, prev_corner_dirs)

    if new_corner is None or prev_corner is None:
        print("Could not get corner coordinates")
        return (0, 0)

    # Calculate translation: where prev_corner is minus where new_corner is
    dx = prev_corner[0] - new_corner[0]
    dy = prev_corner[1] - new_corner[1]

    print(f"New piece corner at {new_corner}, prev piece corner at {prev_corner}")
    print(f"Translation needed: ({dx}, {dy})")

    return (dx, dy)


# In[644]:


import random
def reconstruct_puzzle(grid_canvas, my_puzzle):
    available_pieces = my_puzzle.corners_pieces + my_puzzle.borders_pieces
    placed_pieces = []
    isMiddleCell = False
    total_cells = grid_canvas.rows * grid_canvas.cols

    for counter in range(total_cells):
        row, col = get_spiral_coordinate(counter, grid_canvas.rows, grid_canvas.cols)
        piece_id, rotation = grid_canvas.rep_array[row][col]

        print(f"{counter}: ({row}, {col})")

        if not isMiddleCell and counter == len(my_puzzle.corners_pieces + my_puzzle.borders_pieces):
            print("*** SWITCHING TO MIDDLE PIECES ***")
            available_pieces.extend(my_puzzle.middle_pieces)
            isMiddleCell = True
            print(f"Added {len(my_puzzle.middle_pieces)} middle pieces to available_pieces")
            print(f"Total available pieces now: {len(available_pieces)}")

        if counter == 0:  # Special case for (0,0)
            """_____ Place the top left corner _____"""
            first_corner_id = 15
            # corner_piece = my_puzzle.all_pieces[first_corner_id]
            corner_piece = my_puzzle.all_pieces[first_corner_id]
            flat_edges = [i for i, edge in enumerate(corner_piece.edges) if edge[3] == 0]

            if len(flat_edges) != 2:
                print(f"Error: Corner piece {first_corner_id} doesn't have exactly 2 flat edges!")
                return available_pieces

            correct_rotation = None
            for rotation in range(4):
                directions = ["EAST", "NORTH", "WEST", "SOUTH"]
                rotated_directions = []
                for flat_edge in flat_edges:
                    final_direction_idx = (flat_edge - corner_piece.flat_edge_id + 2 - rotation) % 4
                    rotated_directions.append(directions[final_direction_idx])

                if set(rotated_directions) == {"NORTH", "WEST"}:
                    correct_rotation = rotation
                    break

            if correct_rotation is None:
                print(f"Warning: Could not find correct rotation for piece {first_corner_id}, using rotation=1")
                correct_rotation = 1

            # Calculate final edge directions after grid placement rotation
            final_edge_direction_map = {}
            for edge_id, original_direction in corner_piece.edge_direction_map.items():
                final_direction = (original_direction + correct_rotation) % 4
                final_edge_direction_map[edge_id] = final_direction

            corner_piece.final_edge_direction_map = final_edge_direction_map

            grid_canvas.place_piece(first_corner_id, 0, 0, rotation=correct_rotation)
            available_pieces.remove(first_corner_id)
            placed_pieces.append(first_corner_id)

            # Update piece_id and rotation after placement
            piece_id, rotation = grid_canvas.rep_array[row][col]
            print(f"{counter}: ({row}, {col}) - Piece: {piece_id}, Rotation: {rotation} [PLACED CORNER]")
            grid_canvas.print_grid()
            print_vertex_data_for_step(vertices_lookup, counter)
            grid_canvas.save(f"reconstruction_step_{counter:02d}.png")

        else:
            """_____ For all other cells _____"""
            # Only try to place if the cell is empty
            if piece_id == -1:
                nearby_occupied = get_nearby_occupied_cells(grid_canvas, row, col)

                if nearby_occupied:
                    print(f"Cell ({row}, {col}) has nearby occupied cells:")

                    # We'll try to find a piece that can connect to ALL nearby occupied cells
                    # For now, let's handle the case with one neighbor (which is most common in spiral)
                    if len(nearby_occupied) == 1:
                        neighbor_piece_id, direction, neighbor_row, neighbor_col = nearby_occupied[0]
                        print(f"  - Piece {neighbor_piece_id} at ({neighbor_row}, {neighbor_col}) is {direction} of current cell")

                        # Get the piece data
                        neighbor_piece = my_puzzle.all_pieces[neighbor_piece_id]

                        # Find the opposite direction (the edge that touches the current cell)
                        direction_map = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}
                        opposite_direction = direction_map[direction]

                        # Convert direction name to index (0=NORTH, 1=EAST, 2=SOUTH, 3=WEST)
                        direction_to_index = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}
                        opposite_direction_index = direction_to_index[opposite_direction]

                        # Find the edge_id that corresponds to this direction
                        touching_edge_id = None
                        if hasattr(neighbor_piece, 'final_edge_direction_map') and neighbor_piece.final_edge_direction_map:
                            # Use final_edge_direction_map if available (for placed pieces)
                            for edge_id, edge_direction in neighbor_piece.final_edge_direction_map.items():
                                if edge_direction == opposite_direction_index:
                                    touching_edge_id = edge_id
                                    break
                        elif hasattr(neighbor_piece, 'edge_direction_map') and neighbor_piece.edge_direction_map:
                            # Fallback to original edge_direction_map
                            for edge_id, edge_direction in neighbor_piece.edge_direction_map.items():
                                if edge_direction == opposite_direction_index:
                                    touching_edge_id = edge_id
                                    break

                        if touching_edge_id is not None:
                            print(f"    -> Piece {neighbor_piece_id} has edge {touching_edge_id} facing {opposite_direction} (touching current cell)")

                            # Get sorted matches for this edge
                            if hasattr(neighbor_piece, 'sorted_matches') and touching_edge_id in neighbor_piece.sorted_matches:
                                matches = neighbor_piece.sorted_matches[touching_edge_id]
                                if matches:
                                    print(f"    -> Edge {touching_edge_id} matches:")

                                    # Try matches in order until we find a suitable one
                                    placed_successfully = False

                                    for match_idx, (match_piece_id, match_edge_id, match_score) in enumerate(matches):
                                        # Check if the piece is still available
                                        if match_piece_id not in available_pieces:
                                            print(f"       - Match {match_idx + 1}: Piece {match_piece_id} is no longer available")
                                            continue

                                        candidate_piece = my_puzzle.all_pieces[match_piece_id]
                                        candidate_piece_type = get_piece_type(candidate_piece)

                                        # Check position constraints
                                        position_valid = True
                                        if is_corner_position(row, col, grid_canvas.rows, grid_canvas.cols):
                                            # Corner positions must have corner pieces (exactly 2 flat edges)
                                            if candidate_piece_type != "CORNER":
                                                print(f"       - Match {match_idx + 1}: Piece {match_piece_id} ({candidate_piece_type}) cannot be placed at corner position (needs CORNER piece)")
                                                position_valid = False
                                        elif is_border_position(row, col, grid_canvas.rows, grid_canvas.cols):
                                            # Border positions must have border pieces (exactly 1 flat edge)
                                            if candidate_piece_type != "BORDER":
                                                print(f"       - Match {match_idx + 1}: Piece {match_piece_id} ({candidate_piece_type}) cannot be placed at border position (needs BORDER piece)")
                                                position_valid = False

                                        if not position_valid:
                                            continue

                                        print(f"       - Match {match_idx + 1}: Piece {match_piece_id}, edge {match_edge_id}, score: {match_score:.4f} ({candidate_piece_type})")

                                        # Calculate placement details
                                        target_direction_for_candidate = direction  # The edge should face toward the neighbor

                                        rotations_needed = calculate_rotation_needed(
                                            candidate_piece,
                                            match_edge_id,
                                            target_direction_for_candidate
                                        )

                                        print(f"    -> PLACEMENT: Place piece {match_piece_id} at ({row}, {col})")
                                        print(f"       - Orient edge {match_edge_id} to face {target_direction_for_candidate}")
                                        print(f"       - Rotations needed: {rotations_needed}")

                                        # Actually place the piece
                                        success = grid_canvas.place_piece(match_piece_id, row, col, rotation=rotations_needed)

                                        if success:
                                            # Calculate final edge directions after placement
                                            final_edge_direction_map = {}
                                            for edge_id, original_direction in candidate_piece.edge_direction_map.items():
                                                final_direction = (original_direction + rotations_needed) % 4
                                                final_edge_direction_map[edge_id] = final_direction

                                            candidate_piece.final_edge_direction_map = final_edge_direction_map

                                            # Remove from available pieces and add to placed pieces
                                            available_pieces.remove(match_piece_id)
                                            placed_pieces.append(match_piece_id)

                                            print(f"       - Successfully placed piece {match_piece_id}!")
                                            print(f"       - Remaining pieces: {len(available_pieces)}")

                                            # GET VERTEX DATA AND APPLY TRANSLATION
                                            vertex_data = get_vertex_data_by_step(vertices_lookup, counter)
                                            if vertex_data and len(vertex_data['adjacent_cells']) == 2:
                                                # Get the two cells and their corner directions
                                                cell1_info = vertex_data['adjacent_cells'][0]
                                                cell2_info = vertex_data['adjacent_cells'][1]

                                                # Determine which is the new piece and which is previous
                                                new_cell = (row, col)
                                                if cell1_info['cell'] == new_cell:
                                                    new_corner_dirs = cell1_info['corner_dirs']
                                                    prev_cell = cell2_info['cell']
                                                    prev_corner_dirs = cell2_info['corner_dirs']
                                                else:
                                                    new_corner_dirs = cell2_info['corner_dirs']
                                                    prev_cell = cell1_info['cell']
                                                    prev_corner_dirs = cell1_info['corner_dirs']

                                                # Get the previous piece
                                                prev_piece_id, _ = grid_canvas.rep_array[prev_cell[0]][prev_cell[1]]
                                                if prev_piece_id != -1:
                                                    prev_piece = my_puzzle.all_pieces[prev_piece_id]

                                                    print(f"       - Aligning corners: new piece {match_piece_id} dirs {new_corner_dirs} with prev piece {prev_piece_id} dirs {prev_corner_dirs}")

                                                    # Calculate translation
                                                    dx, dy = translate_piece_to_align_corners(
                                                        candidate_piece, new_corner_dirs,
                                                        prev_piece, prev_corner_dirs
                                                    )

                                                    # Apply translation to the new piece (you'll need to implement this)
                                                    # apply_translation_to_piece(candidate_piece, dx, dy)

                                            grid_canvas.print_grid()
                                            print_vertex_data_for_step(vertices_lookup, counter)
                                            grid_canvas.save(f"reconstruction_step_{counter:02d}.png")

                                            placed_successfully = True
                                            break
                                        else:
                                            print(f"       - Failed to place piece {match_piece_id}")

                                    if not placed_successfully:
                                        print(f"    -> No suitable piece found for position ({row}, {col})")
                                else:
                                    print(f"    -> Edge {touching_edge_id} has no matches")
                            else:
                                print(f"    -> No sorted matches found for edge {touching_edge_id}")
                        else:
                            print(f"    -> Could not find edge facing {opposite_direction} for piece {neighbor_piece_id}")
                    else:
                        print(f"  - Multiple neighbors detected ({len(nearby_occupied)} neighbors)")

                        # Collect all neighbor constraints
                        neighbor_constraints = []
                        for neighbor_piece_id, direction, neighbor_row, neighbor_col in nearby_occupied:
                            print(f"    - Piece {neighbor_piece_id} at ({neighbor_row}, {neighbor_col}) is {direction} of current cell")

                            # Get the piece data
                            neighbor_piece = my_puzzle.all_pieces[neighbor_piece_id]

                            # Find the opposite direction (the edge that touches the current cell)
                            direction_map = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}
                            opposite_direction = direction_map[direction]
                            opposite_direction_index = get_direction_index_from_name(opposite_direction)

                            # Find the edge_id that corresponds to this direction
                            touching_edge_id = None
                            if hasattr(neighbor_piece, 'final_edge_direction_map') and neighbor_piece.final_edge_direction_map:
                                for edge_id, edge_direction in neighbor_piece.final_edge_direction_map.items():
                                    if edge_direction == opposite_direction_index:
                                        touching_edge_id = edge_id
                                        break
                            elif hasattr(neighbor_piece, 'edge_direction_map') and neighbor_piece.edge_direction_map:
                                for edge_id, edge_direction in neighbor_piece.edge_direction_map.items():
                                    if edge_direction == opposite_direction_index:
                                        touching_edge_id = edge_id
                                        break

                            if touching_edge_id is not None:
                                print(f"      -> Piece {neighbor_piece_id} has edge {touching_edge_id} facing {opposite_direction}")

                                # Store constraint: which direction the candidate piece's edge should face
                                target_direction_for_candidate = direction
                                neighbor_constraints.append({
                                    'neighbor_piece_id': neighbor_piece_id,
                                    'neighbor_edge_id': touching_edge_id,
                                    'target_direction': target_direction_for_candidate,
                                    'neighbor_piece': neighbor_piece
                                })

                        # Now find candidate pieces that can satisfy ALL constraints
                        candidate_scores = {}  # piece_id -> total_score
                        candidate_details = {}  # piece_id -> list of (edge_id, rotation, individual_scores)


                        for candidate_piece_id in available_pieces:
                            candidate_piece = my_puzzle.all_pieces[candidate_piece_id]
                            candidate_piece_type = get_piece_type(candidate_piece)

                            # Check position constraints first
                            position_valid = True
                            if is_corner_position(row, col, grid_canvas.rows, grid_canvas.cols):
                                if candidate_piece_type != "CORNER":
                                    position_valid = False
                            elif is_border_position(row, col, grid_canvas.rows, grid_canvas.cols):
                                if candidate_piece_type != "BORDER":
                                    position_valid = False

                            if not position_valid:
                                continue

                            # Try all possible rotations (0-3) for this candidate piece
                            for rotation in range(4):
                                total_score = 0
                                individual_scores = []
                                valid_for_all_neighbors = True

                                # Check if this rotation satisfies all neighbor constraints
                                for constraint in neighbor_constraints:
                                    neighbor_piece = constraint['neighbor_piece']
                                    neighbor_edge_id = constraint['neighbor_edge_id']
                                    target_direction = constraint['target_direction']
                                    target_direction_index = get_direction_index_from_name(target_direction)

                                    # Find which edge of the candidate would face this direction after rotation
                                    candidate_edge_facing_neighbor = None
                                    for edge_id, original_direction in candidate_piece.edge_direction_map.items():
                                        final_direction = (original_direction + rotation) % 4
                                        if final_direction == target_direction_index:
                                            candidate_edge_facing_neighbor = edge_id
                                            break

                                    if candidate_edge_facing_neighbor is None:
                                        valid_for_all_neighbors = False
                                        break

                                    # Get the matching score between the edges
                                    score = 0
                                    if hasattr(neighbor_piece, 'sorted_matches') and neighbor_edge_id in neighbor_piece.sorted_matches:
                                        for other_piece_id, other_edge_id, match_score in neighbor_piece.sorted_matches[neighbor_edge_id]:
                                            if other_piece_id == candidate_piece_id and other_edge_id == candidate_edge_facing_neighbor:
                                                score = match_score
                                                break

                                    individual_scores.append(score)
                                    total_score += score

                                if valid_for_all_neighbors:
                                    if candidate_piece_id not in candidate_scores or total_score > candidate_scores[candidate_piece_id]:
                                        candidate_scores[candidate_piece_id] = total_score
                                        candidate_details[candidate_piece_id] = (rotation, individual_scores, total_score)

                        # Find the best candidate
                        if candidate_scores:
                            best_candidate_id = max(candidate_scores, key=candidate_scores.get)
                            best_rotation, best_individual_scores, best_total_score = candidate_details[best_candidate_id]

                            print(f"    -> Best candidate: Piece {best_candidate_id}")
                            print(f"       - Total score: {best_total_score:.4f}")
                            print(f"       - Individual scores: {best_individual_scores}")
                            print(f"       - Rotation needed: {best_rotation}")

                            # Place the piece
                            success = grid_canvas.place_piece(best_candidate_id, row, col, rotation=best_rotation)

                            if success:
                                candidate_piece = my_puzzle.all_pieces[best_candidate_id]

                                # Calculate final edge directions after placement
                                final_edge_direction_map = {}
                                for edge_id, original_direction in candidate_piece.edge_direction_map.items():
                                    final_direction = (original_direction + best_rotation) % 4
                                    final_edge_direction_map[edge_id] = final_direction

                                candidate_piece.final_edge_direction_map = final_edge_direction_map

                                # Remove from available pieces and add to placed pieces
                                available_pieces.remove(best_candidate_id)
                                placed_pieces.append(best_candidate_id)

                                print(f"       - Successfully placed piece {best_candidate_id}!")
                                print(f"       - Remaining pieces: {len(available_pieces)}")

                                # Save progress
                                grid_canvas.print_grid()
                                grid_canvas.save(f"reconstruction_step_{counter:02d}.png")
                            else:
                                print(f"       - Failed to place piece {best_candidate_id}")
                        else:
                            print(f"    -> No suitable piece found for multiple neighbor constraints")
                            # TODO: Could implement random selection here if needed

                else:
                    print(f"Cell ({row}, {col}) has no nearby occupied cells")
            else:
                print(f"Cell ({row}, {col}) already occupied by piece {piece_id}")



    print(f"\nRemaining pieces to place: {available_pieces}")
    print(f"Placed pieces: {placed_pieces}")
    return available_pieces


# In[645]:


# Create a grid canvas
reconstruction_canvas = GridCanvas(
    width=300 * puzzle_c,
    height=300 * puzzle_r,
    rows=puzzle_r,
    cols=puzzle_c,
    column_spacing=250,
    row_spacing=250,
    margin=100
)


# In[646]:


reconstruct_puzzle(reconstruction_canvas, my_puzzle)

reconstruction_canvas.print_grid()
reconstruction_canvas.save("new_reconstruction.png")


# In[641]:


# In[ ]:

print("Middle pieces matching scores:")
print("=" * 40)

for piece_id in my_puzzle.middle_pieces:
    piece = my_puzzle.all_pieces[piece_id]

    print(f"Piece {piece_id}:")

    if hasattr(piece, 'sorted_matches') and piece.sorted_matches:
        for edge_id in sorted(piece.sorted_matches.keys()):
            matches = piece.sorted_matches[edge_id]
            if matches:
                print(f"  Edge {edge_id}:")
                for other_piece_id, other_edge_id, score in matches[:3]:  # Show top 3 matches
                    print(f"    -> Piece {other_piece_id}, Edge {other_edge_id}: {score:.4f}")
            else:
                print(f"  Edge {edge_id}: No matches")
    else:
        print("  No sorted matches available")

    print()


# In[ ]:





# In[ ]:




