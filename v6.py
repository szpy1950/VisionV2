import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

def display_image(title, image):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def extract_puzzle_pieces(image_path):
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

    print("Finding contours of pieces")
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} potential puzzle pieces")

    print("Filtering contours by size")
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if len(contours) > 1:
        reference_area = cv2.contourArea(contours[1])
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > reference_area / 3]
    print(f"After filtering: {len(contours)} puzzle pieces")

    print("Drawing contours of the original image")
    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    display_image("Detected Pieces", contour_image)

    output_folder_contours = "images/extracted_contours"
    os.makedirs(output_folder_contours, exist_ok=True)

    print("Extracting individual parts into piece images")

    piece_images = []
    output_folder_pieces = "images/extracted_pieces"
    os.makedirs(output_folder_pieces, exist_ok=True)

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

        # Display up to 3 test pieces -> presentation
        if i < 3:
            display_image(f"Piece {i + 1}", cropped_piece)

    output_corner_folder = "images/extracted_corners"
    os.makedirs(output_corner_folder, exist_ok=True)

    output_plots_folder = "images/corner_plots"
    os.makedirs(output_plots_folder, exist_ok=True)

    test_number = 24

    """
        EDGE AND CORNER DETECTION
    """

    ### separated part because I want to put it in a different file later, i dont want to redo the whole piece
    ### extraction every time

    print("Analyzing edges of pieces using contour-based corner detection")
    piece_images = []

    for i, contour in enumerate(contours):
        # Process only the first 3 pieces for testing
        if i >= 10:
            break


        # 1 ) get the pieces contour

        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        piece = np.zeros_like(original_image)
        piece[mask == 255] = original_image[mask == 255]
        x, y, w, h = cv2.boundingRect(contour)
        cropped_piece = piece[y:y + h, x:x + w]
        piece_images.append(cropped_piece)

        piece_path = os.path.join(output_folder_pieces, f"piece_{i + 1}.png")
        cv2.imwrite(piece_path, cropped_piece)
        contour_piece = cropped_piece.copy()



        # 2)  get each contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        cv2.circle(contour_piece, (centroid_x - x, centroid_y - y), 5, (0, 0, 255), -1)


        # 3) get the angle for each point on the contour
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

        # Sort points by angle
        sorted_indices = np.argsort(angles_deg)
        sorted_distances = np.array(distances)[sorted_indices]
        sorted_angles_deg = angles_deg[sorted_indices]

        # Find all peaks in the distance profile with lower prominence
        min_distance_between_peaks = len(sorted_distances) // 20  # More sensitive distance
        all_peak_indices, all_peak_properties = find_peaks(sorted_distances,
                                                           distance=min_distance_between_peaks,
                                                           prominence=2)  # Lower prominence to catch more peaks

        # Define this variable to fix the error
        all_peaks_to_plot = all_peak_indices

        # Check each peak based on the values at ±5 degrees
        angle_window = 5  # Check points at ±5 degrees
        min_height_diff = 5  # Minimum height difference to consider a peak as sharp
        min_distance_threshold = 75  # Minimum distance from centroid to consider a peak

        # Store all indices for plotting
        sharp_peak_indices = []
        left_indices = []
        right_indices = []
        peak_point_distances = []  # Store the distance between the left and right points

        for peak_idx in all_peak_indices:
            peak_angle = sorted_angles_deg[peak_idx]
            peak_value = sorted_distances[peak_idx]

            # Skip peaks below the minimum distance threshold
            if peak_value < min_distance_threshold:
                left_indices.append(None)
                right_indices.append(None)
                peak_point_distances.append(float('inf'))
                continue

            # Find points approximately ±5 degrees from the peak
            left_angle = (peak_angle - angle_window) % 360
            right_angle = (peak_angle + angle_window) % 360

            # Find the closest points to these angles
            left_idx = np.argmin(np.abs(sorted_angles_deg - left_angle))
            right_idx = np.argmin(np.abs(sorted_angles_deg - right_angle))

            # Store these indices for plotting
            left_indices.append(left_idx)
            right_indices.append(right_idx)

            left_value = sorted_distances[left_idx]
            right_value = sorted_distances[right_idx]

            # Calculate the "closeness" of the left and right points (smaller is closer)
            # Use sum of absolute differences from peak value
            left_distance_diff = abs(peak_value - left_value)
            right_distance_diff = abs(peak_value - right_value)

            # Sum of differences (smaller means closer points)
            point_distance = left_distance_diff + right_distance_diff
            peak_point_distances.append(point_distance)

            # Check if peak is at least min_height_diff higher than both sides
            if (peak_value - left_value >= min_height_diff) and (peak_value - right_value >= min_height_diff):
                sharp_peak_indices.append(peak_idx)

        # Among the sharp peaks above the distance threshold, take the 4 with the closest pairs of points
        if len(sharp_peak_indices) >= 4:
            # Get the point_distance values for the sharp peaks
            sharp_peak_distances = []
            for idx in sharp_peak_indices:
                idx_in_all = np.where(all_peak_indices == idx)[0][0]
                sharp_peak_distances.append(peak_point_distances[idx_in_all])

            # Sort by point distance (closest pairs first) and take top 4
            top_indices = np.argsort(sharp_peak_distances)[:4]
            corner_peaks_to_plot = [sharp_peak_indices[i] for i in top_indices]
        else:
            # If we don't have 4 sharp peaks, use what we have
            corner_peaks_to_plot = sharp_peak_indices
            print(f"Warning: Only found {len(sharp_peak_indices)} qualifying corners for piece {i + 1}")

        # Get the original contour indices for the peaks
        corner_indices = [sorted_indices[idx] for idx in corner_peaks_to_plot]

        # Get corner points in the original image coordinates
        corner_points = [contour[idx][0] for idx in corner_indices]

        # Draw corners on the visualization image (green)
        for corner in corner_points:
            cv2.circle(contour_piece, (corner[0] - x, corner[1] - y), 7, (0, 255, 0), -1)

        # Draw the contour (pink)
        cv2.drawContours(contour_piece, [contour_points], 0, (255, 0, 255), 1, cv2.LINE_8)

        # Save piece with contour and corners to the output_corner_folder
        contour_path = os.path.join(output_corner_folder, f"contour_{i + 1}.png")
        cv2.imwrite(contour_path, contour_piece)

        # No longer saving to output_folder_contours
        # contour_path = os.path.join(output_folder_contours, f"contour_{i + 1}.png")
        # cv2.imwrite(contour_path, contour_piece)

        # No longer saving to output_plots_folder because we're already saving the contour images to output_corner_folder
        # corner_path = os.path.join(output_plots_folder, f"corner_piece_{i + 1}.png")
        # cv2.imwrite(corner_path, contour_piece)

        # Create a plot of the distance profile with detected peaks
        plt.figure(figsize=(10, 5))
        plt.plot(sorted_angles_deg, sorted_distances)

        # Plot all peaks in orange
        plt.plot(sorted_angles_deg[all_peaks_to_plot], sorted_distances[all_peaks_to_plot], 'o', color='orange',
                 label='All Peaks')

        # Plot sharp corner peaks in green
        plt.plot(sorted_angles_deg[corner_peaks_to_plot], sorted_distances[corner_peaks_to_plot], 'o', color='green',
                 label='Selected Corners')

        # Add a horizontal line at the minimum distance threshold
        plt.axhline(y=min_distance_threshold, color='blue', linestyle='--',
                    label=f'Min Distance ({min_distance_threshold})')

        # Plot the points ±5 degrees away from each peak in red
        for j, peak_idx in enumerate(all_peaks_to_plot):
            if j < len(left_indices) and left_indices[j] is not None and right_indices[j] is not None:
                # Plot the left and right points in red
                plt.plot(sorted_angles_deg[left_indices[j]], sorted_distances[left_indices[j]], 'ro', markersize=4)
                plt.plot(sorted_angles_deg[right_indices[j]], sorted_distances[right_indices[j]], 'ro', markersize=4)

                # For selected corners, use a thicker line
                if peak_idx in corner_peaks_to_plot:
                    linewidth = 2
                    linestyle = '-'
                else:
                    linewidth = 1
                    linestyle = ':'

                # Draw lines connecting the peak to its left and right points
                plt.plot([sorted_angles_deg[peak_idx], sorted_angles_deg[left_indices[j]]],
                         [sorted_distances[peak_idx], sorted_distances[left_indices[j]]],
                         'r-', linewidth=linewidth, linestyle=linestyle)
                plt.plot([sorted_angles_deg[peak_idx], sorted_angles_deg[right_indices[j]]],
                         [sorted_distances[peak_idx], sorted_distances[right_indices[j]]],
                         'r-', linewidth=linewidth, linestyle=linestyle)

        plt.title(f'Distance Profile for Piece {i + 1} (±5° Window, Min Dist: {min_distance_threshold})')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Distance from Centroid')
        plt.legend()
        plt.grid(True)

        # Save the plot to output_plots_folder instead of output_corner_folder
        plt.savefig(os.path.join(output_plots_folder, f"distance_profile_{i + 1}.png"))
        plt.close()

        # Display up to 3 test pieces
        display_image(f"Piece {i + 1}", contour_piece)




        ### drawing lines


    return original_image, contours, piece_images

def main():
    # we are using the -> "modifed black image"
    image_path = "images/hack2.png"

    try:
        original_image, pieces, piece_images = extract_puzzle_pieces(image_path)
        print(f"✅ We have found {len(pieces)} puzzle pieces")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()