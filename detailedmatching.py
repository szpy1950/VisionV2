#!/usr/bin/env python
# coding: utf-8

# In[510]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import math


# In[511]:


image_path = "images/hack2.png"


# In[512]:


def display_image(title, image):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


# In[513]:


# ## Basic reading image and display

original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if original_image is None:
    raise ValueError(f"Could not read image from {image_path}")


# In[514]:


# ## Grayscale conversion

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# display_image("Grayscale Image", gray_image)


# In[515]:


# print("Threshold to separate pieces from background")
_, binary_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
# display_image("Binary Image", binary_image)


# In[516]:


kernel = np.ones((12, 12), np.uint8)

# Close = filling the holes
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
# Open = removing the noise
morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
# display_image("Morph Operations", morph_image)


# In[517]:


# print("Filling holes in puzzle pieces")
contours_fill, _ = cv2.findContours(morph_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_fill:
    cv2.drawContours(morph_image, [cnt], 0, 255, -1)
# display_image("Filled Holes", morph_image)


# In[518]:


# ## Contours finding
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print(f"Found {len(contours)} potential puzzle pieces")


# In[519]:


# print("Filtering contours by size")
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
if len(contours) > 1:
    reference_area = cv2.contourArea(contours[1])
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > reference_area / 3]
# print(f"After filtering: {len(contours)} puzzle pieces")


# In[520]:


# print("Drawing contours of the original image")
contour_image = original_image.copy()
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)


# In[521]:


output_folder_pieces = "images/extracted_pieces"
os.makedirs(output_folder_pieces, exist_ok=True)

output_folder_contours = "images/extracted_contours"
os.makedirs(output_folder_contours, exist_ok=True)

output_corner_folder = "images/extracted_corners"
os.makedirs(output_corner_folder, exist_ok=True)

output_transparent_folder = "images/transparent_pieces"
os.makedirs(output_transparent_folder, exist_ok=True)

output_plots_folder = "images/corner_plots"
os.makedirs(output_plots_folder, exist_ok=True)

output_canvas = "images/canvas"
os.makedirs(output_canvas, exist_ok=True)

output_oriented_folder = "images/oriented_pieces"
os.makedirs(output_oriented_folder, exist_ok=True)

output_edges_and_corner_folder = "images/edges_and_corners"
os.makedirs(output_edges_and_corner_folder, exist_ok=True)

output_tests = "images/tests"
os.makedirs(output_tests, exist_ok=True)


# In[522]:


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


# In[523]:


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


# In[524]:


class Puzzle:
    _counter = 0
    def __init__(self):
        self.puzzle_id = Puzzle._counter
        Puzzle._counter += 1
        self.all_pieces = {}
        self.corners_pieces = []
        self.borders_pieces = []
        self.middle_pieces = []


# In[525]:


my_puzzle = Puzzle()


# In[526]:


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

    # transparent_path = os.path.join(output_transparent_folder, f"piece_{selected_image_index}_transparent.png")
    # cv2.imwrite(transparent_path, transparent_piece)


    # piece_path = os.path.join(output_folder_pieces, f"piece_{selected_image_index + 1}.png")
    # cv2.imwrite(piece_path, cropped_piece)
    contour_piece = cropped_piece.copy()

    # display_image(f"Cropped piece {selected_image_index+1}", contour_piece)


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

            if distances[remaining_peak_indices[k]] <= min_distances_avg:
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

    new_min_indice = all_min_indices.copy()

    min_angles = [angles_deg[min_idx] for min_idx in new_min_indice]
    min_distances = [distances[min_idx] for min_idx in new_min_indice]

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

    def exists_minima_between(a,b,min_indices):
        if b < a:
            for p in min_indices:
                if p > a or p < b:
                    # hack to set distance smaller than custom threshold
                    if distances[p] < ((max(distances[a], distances[b])*resize_factor ) /1.41):
                        # print(f"Found minima {p}")
                        return True
                    else:
                        return False

        else:
            for p in min_indices:
                if p > a and p < b:
                    if distances[p] < ((max(distances[a], distances[b])*resize_factor ) /1.41):
                        # print(f"Found minima {p}")
                        return True
                    else:
                        return False

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


# In[527]:


print("----------- STATS ---------- ")

print(f"Number of pieces: {len(my_puzzle.all_pieces)}")

print("Corners: ", my_puzzle.corners_pieces)
print("Borders: ", my_puzzle.borders_pieces)
print("Middles: ", my_puzzle.middle_pieces)


# ## Algorithm to find the puzzle size

# In[528]:


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


# In[529]:


puzzle_r, puzzle_c = find_puzzle_size(len(my_puzzle.all_pieces), len(my_puzzle.corners_pieces), len(my_puzzle.borders_pieces), len(my_puzzle.middle_pieces))
print(puzzle_r)
print(puzzle_c)


# In[530]:


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


# In[531]:


# target_index = 5
# test_piece = my_puzzle.all_pieces[target_index]


# ## Piece reorientation
# 
# pieces with a straight edge -> y axis orientation

# In[532]:


# Create a new white canvas to place all oriented pieces
# Define this before the loop
canvas_width = 1500
canvas_height = 1200
piece_spacing = 50
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


# In[533]:


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
        self.canvas = np.ones((height, width, puzzle_c), dtype=np.uint8) * 255
        self.canvas[:,:,3] = 255  # Full alpha

        # Placement tracking
        self.current_x = margin
        self.current_y = margin
        self.max_height_in_row = 0
        self.piece_count = 0
        self.row_heights = [0] * ((width // (piece_spacing + 0)) + 1)
        self.piece_positions = []  # Store (piece_id, x, y) tuples

    def place_piece(self, piece):
        # Calculate position on the grid
        row = self.piece_count // self.max_pieces_per_row
        col = self.piece_count % self.max_pieces_per_row

        # Start a new row if needed
        if col == 0 and self.piece_count > 0:
            self.current_y += self.max_height_in_row + self.row_spacing  # Use row_spacing
            self.current_x = self.margin
            self.max_height_in_row = 0

        # Get piece dimensions
        piece_height, piece_width = piece.rotated_image.shape[:2]
        self.max_height_in_row = max(self.max_height_in_row, piece_height)

        # Get reference corner (first rotated corner)
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

        cv2.putText(piece.rotated_image, str(piece.piece_id), piece.rotated_center,
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255, 255), 2)

        # Store piece position for future reference
        self.piece_positions.append((piece.piece_id, self.current_x, self.current_y))

        # Debug marker
        cv2.circle(self.canvas, (self.current_x, self.current_y), 3, (0, 0, 255, 255), -1)

        # Move to next position
        self.current_x += self.column_spacing + 100
        self.piece_count += 1

        return self.piece_count - 1  # Return the position index

    def save(self, path):
        """Save the canvas to file"""
        cv2.imwrite(path, self.canvas)

    def get_piece_position(self, piece_id):
        """Get the position of a piece on the canvas"""
        for p_id, x, y in self.piece_positions:
            if p_id == piece_id:
                return (x, y)
        return None


# In[534]:


# First process and reorient all pieces
for index in (my_puzzle.borders_pieces + my_puzzle.corners_pieces):
    print("------- ROTATE TO ORTHOGONAL ORIENTATION ----------")
    print(f"Testing index: {index}")

    test_piece = my_puzzle.all_pieces[index]

    edge_id = 0

    print(test_piece.edges_ord)

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
        cv2.line(colored_edges_image, c1_rot, c2_rot, color, 2)

        # Add edge number label
        mid_point = ((c1_rot[0] + c2_rot[0]) // 2, (c1_rot[1] + c2_rot[1]) // 2)
        cv2.putText(colored_edges_image, str(edge_idx), mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0, 255), 2)

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


# In[535]:


# Create a canvas object with custom spacing
canvas_obj = Canvas(1500, 1200, column_spacing=100, row_spacing=-150, margin=100)

# Now place all the pieces on the canvas
piece_indices = my_puzzle.borders_pieces + my_puzzle.corners_pieces
print(f"Placing {len(piece_indices)} pieces on canvas...")

for index in piece_indices:
    piece = my_puzzle.all_pieces[index]
    print(f"Placing piece {index} on canvas (type: {piece.piece_type})")
    canvas_obj.place_piece(piece)

# Save the final canvas
canvas_path = os.path.join(output_canvas, "all_pieces_arranged.png")
canvas_obj.save(canvas_path)
print(f"Canvas saved to {canvas_path}")


# In[539]:


# Edge matching function to compute a matching score between two puzzle piece edges
def compute_edge_matching_score(piece1, edge1_idx, piece2, edge2_idx, debug=False):
    """
    Compute a matching score between two edges of different puzzle pieces.

    Args:
        piece1: First puzzle piece
        edge1_idx: Index of the edge in piece1
        piece2: Second puzzle piece
        edge2_idx: Index of the edge in piece2
        debug: Whether to save debug images

    Returns:
        score: A matching score between 0 and 1, where 1 is perfect match
    """
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
    start_rot1 = piece1.rotated_corners[edge1_idx]
    end_rot1 = piece1.rotated_corners[(edge1_idx + 1) % len(piece1.rotated_corners)]
    start_rot2 = piece2.rotated_corners[edge2_idx]
    end_rot2 = piece2.rotated_corners[(edge2_idx + 1) % len(piece2.rotated_corners)]

    # Get rotated contour points between the corners for each edge
    def get_contour_between_corners(piece, start_idx, end_idx):
        contour = piece.absolute_contour  # Use the full contour
        if start_idx < end_idx:
            return contour[start_idx:end_idx+1]
        else:
            # Wrap around the contour
            return np.vstack([contour[start_idx:], contour[:end_idx+1]])

    # Get contour points for each edge
    contour1 = get_contour_between_corners(piece1, start_corner1, end_corner1)
    contour2 = get_contour_between_corners(piece2, start_corner2, end_corner2)

    # Sample N points along each edge for comparison
    N = 20  # Number of sampling points

    # Function to sample points along an edge
    def sample_points_along_edge(contour, N):
        total_points = len(contour)
        indices = np.linspace(0, total_points-1, N, dtype=int)
        return contour[indices]

    # Sample points
    sample_points1 = sample_points_along_edge(contour1, N)
    sample_points2 = sample_points_along_edge(contour2, N)

    # Reverse the second contour for comparison (since they should fit together)
    sample_points2 = sample_points2[::-1]

    # Function to get color at a point and slightly inside the piece
    def get_color_at_point(piece, point, inward_offset=3):
        x, y = point[0]

        # Calculate piece center
        cx, cy = piece.absolute_center

        # Calculate unit vector from point to center
        dx, dy = cx - x, cy - y
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx/length, dy/length
        else:
            return np.array([0, 0, 0])  # Default color if at center

        # Calculate point slightly inside the piece
        inside_x = int(x + dx * inward_offset)
        inside_y = int(y + dy * inward_offset)

        # Make sure coordinates are within image bounds
        h, w = piece.mask.shape[:2]
        inside_x = max(0, min(inside_x, w-1))
        inside_y = max(0, min(inside_y, h-1))

        # Get color from original image at this point
        if piece.mask[inside_y, inside_x] > 0:  # Check if point is inside the mask
            color = original_image[inside_y, inside_x]
            # Check if the color is not black (background)
            if np.sum(color) > 30:  # simple threshold to avoid black pixels
                return color

        # If outside mask or color is black, try a different offset
        for offset in [5, 7, 10]:
            inside_x = int(x + dx * offset)
            inside_y = int(y + dy * offset)
            inside_x = max(0, min(inside_x, w-1))
            inside_y = max(0, min(inside_y, h-1))
            if piece.mask[inside_y, inside_x] > 0:
                color = original_image[inside_y, inside_x]
                if np.sum(color) > 30:
                    return color

        return np.array([0, 0, 0])  # Return black if no valid color found

    # Get colors for each sample point
    colors1 = np.array([get_color_at_point(piece1, pt) for pt in sample_points1])
    colors2 = np.array([get_color_at_point(piece2, pt) for pt in sample_points2])

    # Filter out black points (background)
    valid_indices = []
    for i in range(N):
        if np.sum(colors1[i]) > 30 and np.sum(colors2[i]) > 30:  # Not black
            valid_indices.append(i)

    if len(valid_indices) < 3:
        # Not enough valid color points to compare
        if debug:
            print(f"Not enough valid color points. Score = 0")
        return 0

    # Compute color difference for valid points
    color_diffs = []
    for i in valid_indices:
        # Euclidean distance in RGB space
        diff = np.sqrt(np.sum((colors1[i].astype(float) - colors2[i].astype(float))**2))
        color_diffs.append(diff)

    # Normalize color differences to get a score [0-1]
    avg_diff = np.mean(color_diffs)
    max_possible_diff = 255 * np.sqrt(3)  # Maximum possible RGB distance
    score = 1.0 - (avg_diff / max_possible_diff)

    if debug:
        print(f"Edge {edge1_idx} of piece {piece1.piece_id} (OUT) and edge {edge2_idx} of piece {piece2.piece_id} (IN)")
        print(f"Average color difference: {avg_diff:.2f}, Score: {score:.2f}")

        # Create debug visualization
        debug_img1 = piece1.rotated_image.copy()
        debug_img2 = piece2.rotated_image.copy()

        # Mark the sample points
        for pt in sample_points1.reshape(-1, 2):
            cv2.circle(debug_img1, (pt[0], pt[1]), 3, (0, 255, 255, 255), -1)

        for pt in sample_points2.reshape(-1, 2):
            cv2.circle(debug_img2, (pt[0], pt[1]), 3, (0, 255, 255, 255), -1)

        # Save debug images
        debug_path1 = os.path.join(output_tests, f"match_debug_p{piece1.piece_id}_e{edge1_idx}.png")
        debug_path2 = os.path.join(output_tests, f"match_debug_p{piece2.piece_id}_e{edge2_idx}.png")
        cv2.imwrite(debug_path1, debug_img1)
        cv2.imwrite(debug_path2, debug_img2)

    return score


# In[540]:


# Test the matching function only for the first piece against all others
matches = []

# Get all pieces we want to match
pieces_to_match = [my_puzzle.all_pieces[i] for i in my_puzzle.borders_pieces + my_puzzle.corners_pieces]

# Get the first piece
first_piece = pieces_to_match[0]
print(f"Testing piece {first_piece.piece_id} against all others")

# Compare with all other pieces
for other_piece in pieces_to_match[1:]:
    for edge1_idx in range(len(first_piece.edges)):
        for edge2_idx in range(len(other_piece.edges)):
            # Compute score with debug info for the first few comparisons
            debug = len(matches) < 5  # Debug only the first few matches
            score = compute_edge_matching_score(first_piece, edge1_idx, other_piece, edge2_idx, debug=debug)

            # Only display non-zero scores
            if score > 0:
                print(f"Piece {first_piece.piece_id} edge {edge1_idx}, piece {other_piece.piece_id} edge {edge2_idx}, score: {score:.3f}")

            # Store if score is above threshold
            if score > 0.5:  # Adjust threshold as needed
                matches.append({
                    'piece1_id': first_piece.piece_id,
                    'edge1_idx': edge1_idx,
                    'piece2_id': other_piece.piece_id,
                    'edge2_idx': edge2_idx,
                    'score': score
                })

# Sort matches by score (highest first)
matches.sort(key=lambda x: x['score'], reverse=True)

# Print top matches
print(f"\nFound {len(matches)} potential matches with score > 0.5 for piece {first_piece.piece_id}")
for i, match in enumerate(matches):
    print(f"Match {i+1}: Piece {match['piece1_id']} edge {match['edge1_idx']} with "
          f"Piece {match['piece2_id']} edge {match['edge2_idx']} - Score: {match['score']:.3f}")

# Store matches in puzzle object for later use
my_puzzle.edge_matches = matches

