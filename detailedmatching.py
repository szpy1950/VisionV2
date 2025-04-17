#!/usr/bin/env python
# coding: utf-8

# In[102]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import math


# In[103]:


image_path = "images/hack2.png"


# In[104]:


def display_image(title, image):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


# In[105]:


# ## Basic reading image and display

original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if original_image is None:
    raise ValueError(f"Could not read image from {image_path}")


# In[106]:


# ## Grayscale conversion

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# display_image("Grayscale Image", gray_image)


# In[107]:


# print("Threshold to separate pieces from background")
_, binary_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
# display_image("Binary Image", binary_image)


# In[108]:


kernel = np.ones((12, 12), np.uint8)

# Close = filling the holes
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
# Open = removing the noise
morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
# display_image("Morph Operations", morph_image)


# In[109]:


# print("Filling holes in puzzle pieces")
contours_fill, _ = cv2.findContours(morph_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_fill:
    cv2.drawContours(morph_image, [cnt], 0, 255, -1)
# display_image("Filled Holes", morph_image)


# In[110]:


# ## Contours finding
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print(f"Found {len(contours)} potential puzzle pieces")


# In[111]:


# print("Filtering contours by size")
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
if len(contours) > 1:
    reference_area = cv2.contourArea(contours[1])
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > reference_area / 3]
# print(f"After filtering: {len(contours)} puzzle pieces")


# In[112]:


# print("Drawing contours of the original image")
contour_image = original_image.copy()
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)


# In[113]:


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

output_edges_and_corner_folder = "images/edges_and_corners"
os.makedirs(output_edges_and_corner_folder, exist_ok=True)

output_tests = "images/tests"
os.makedirs(output_tests, exist_ok=True)


# In[114]:


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


# In[115]:


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


    def __repr__(self):
        ret = ""
        ret += f"{self.piece_id}\n"
        ret += f"Contour: {len(self.contour)}\n"
        ret += f"Corners: {self.corners}\n"

        return ret


# In[116]:


class Puzzle:
    _counter = 0
    def __init__(self):
        self.puzzle_id = Puzzle._counter
        Puzzle._counter += 1
        self.all_pieces = {}
        self.corners_pieces = []
        self.borders_pieces = []
        self.middle_pieces = []


# In[117]:


my_puzzle = Puzzle()


# In[118]:


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


# In[119]:


print("----------- STATS ---------- ")

print(f"Number of pieces: {len(my_puzzle.all_pieces)}")

print("Corners: ", my_puzzle.corners_pieces)
print("Borders: ", my_puzzle.borders_pieces)
print("Middles: ", my_puzzle.middle_pieces)


# In[120]:


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


# In[121]:


find_puzzle_size(len(my_puzzle.all_pieces), len(my_puzzle.corners_pieces), len(my_puzzle.borders_pieces), len(my_puzzle.middle_pieces))


# In[122]:


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


# In[123]:


target_index = 5
test_piece = my_puzzle.all_pieces[target_index]


# In[128]:


print(test_piece.edges_ord)
print(test_piece.piece_type)
data = test_piece.edges
columns = test_piece.edges_desc

df = pd.DataFrame(data, columns=columns)
print(df)


# In[173]:


corner1 = data[2][1]
corner2 = data[2][2]

point_1 = test_piece.relative_contour[corner1][0]
point_2 = test_piece.relative_contour[corner2][0]


# get angle with y-axis and find theta angle

vec = point_2 - point_1
orth = np.array([0, 1])

cos_theta = np.dot(vec, orth) / (np.linalg.norm(vec) * np.linalg.norm(orth))
angle_rad = np.arccos(cos_theta)
angle_deg = np.degrees(angle_rad)

print(angle_rad)
print(angle_deg)

if angle_deg > 90:
    angle_deg = 180 - angle_deg

print(angle_deg)


# In[178]:


# draw center
image_with_center = test_piece.transparent_image.copy()
cv2.circle(image_with_center, test_piece.relative_center, 2, (0, 0, 255, 255), -1)

print(test_piece.relative_center)

# cv2.circle(image_with_center, point_1, 2, (255, 255, 0, 255), -1)
# cv2.circle(image_with_center, point_2, 2, (255, 255, 0, 255), -1)

(h, w) = image_with_center.shape[:2]
center = tuple(test_piece.relative_center)

# rotate

M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
image_with_center = cv2.warpAffine(image_with_center, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)



display_image_cv2(f"Transparent piece {target_index}", image_with_center)


# In[181]:


p1_rot = M @ np.array([point_1[0], point_1[1], 1])
p2_rot = M @ np.array([point_2[0], point_2[1], 1])


p1_rot = tuple(np.round(p1_rot).astype(int))
p2_rot = tuple(np.round(p2_rot).astype(int))


print("Rotated point_1:", p1_rot)
print("Rotated point_2:", p2_rot)


# In[182]:


rotated_image = image_with_center.copy()

cv2.circle(rotated_image, p1_rot, 4, (0, 255, 0, 255), -1)
cv2.circle(rotated_image, p2_rot, 4, (0, 255, 0, 255), -1)

display_image_cv2(f"Transparent piece {target_index}", rotated_image)


# In[ ]:




