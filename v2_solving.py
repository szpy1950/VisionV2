import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage


def detect_puzzle_pieces(image_path, display_steps=True, min_contour_area=500, k_clusters=4):
    """
    Detect puzzle pieces with improved handling of background-colored regions.

    Args:
        image_path: Path to the puzzle image
        display_steps: Whether to display intermediate steps
        min_contour_area: Minimum area for a contour to be considered a puzzle piece
        k_clusters: Number of clusters for color segmentation

    Returns:
        Tuple of (original image, mask of pieces, final image with red background,
                 list of individual piece images)
    """
    """
    Detect puzzle pieces with improved handling of background-colored regions.

    Args:
        image_path: Path to the puzzle image
        display_steps: Whether to display intermediate steps
        min_contour_area: Minimum area for a contour to be considered a puzzle piece
        k_clusters: Number of clusters for color segmentation

    Returns:
        Tuple of (original image, mask of pieces, final image with red background,
                 list of individual piece images)
    """
    # Load image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to RGB for display and LAB for processing
    rgb_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

    # Step 1: Apply bilateral filter to smooth similar colors while preserving edges
    bilateral = cv2.bilateralFilter(lab_image, 9, 75, 75)

    # Step 2: K-means clustering for color segmentation
    reshaped = bilateral.reshape((-1, 3))
    reshaped = np.float32(reshaped)

    # Use more clusters for complex backgrounds
    K = k_clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # For complex backgrounds, we might need to try different approaches to find pieces
    labels_flat = labels.flatten()

    # Try to identify the background by analyzing color differences
    # This is more robust than just taking the most common color

    # Reshape the centers to match the image structure for visualization
    segmented_image = centers[labels.flatten()].reshape(lab_image.shape)

    # Try to determine which cluster is background
    # For phone photos, edges of the image are more likely to be background
    height, width = lab_image.shape[:2]
    border_points = []

    # Sample points from the border
    border_width = max(5, int(min(height, width) * 0.05))  # 5% of the smaller dimension

    # Make sure labels is properly shaped
    # Get the real shape of the labels array after reshaping
    labels_height, labels_width = labels.reshape(lab_image.shape[:2]).shape

    # Top and bottom borders - use step size based on image width
    step_size = max(1, width // 20)  # Ensure at least 20 samples across width
    for x in range(0, labels_width, step_size):
        for y in range(0, min(border_width, labels_height)):
            border_points.append((y, x))
        for y in range(max(0, labels_height - border_width), labels_height):
            border_points.append((y, x))

    # Left and right borders
    step_size = max(1, height // 20)  # Ensure at least 20 samples across height
    for y in range(border_width, labels_height - border_width, step_size):
        for x in range(0, min(border_width, labels_width)):
            border_points.append((y, x))
        for x in range(max(0, labels_width - border_width), labels_width):
            border_points.append((y, x))

    # Get cluster labels at border points
    labels_reshaped = labels.reshape(lab_image.shape[:2])
    border_labels = [labels_reshaped[y, x] for y, x in border_points
                     if 0 <= y < labels_height and 0 <= x < labels_width]

    if border_labels:
        # Most common cluster at the border is likely the background
        background_label = np.bincount(border_labels).argmax()
    else:
        # Fallback to most common overall
        background_label = np.bincount(labels_flat).argmax()

    # Create initial background mask
    mask = np.zeros(labels_flat.shape, dtype=np.uint8)
    mask[labels_flat != background_label] = 255
    mask = mask.reshape(rgb_image.shape[:2])

    # Step 3: Apply multiple edge detection methods and combine them
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Canny with different parameters
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 30, 100)

    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)

    # Also try Sobel for gradient-based edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = np.uint8(np.clip(sobel / sobel.max() * 255, 0, 255))

    # Threshold Sobel
    _, sobel_thresh = cv2.threshold(sobel, 40, 255, cv2.THRESH_BINARY)

    # Combine all edge detection methods
    all_edges = cv2.bitwise_or(edges, sobel_thresh)

    # Step 4: Find and fill external contours from edge detection
    contours, _ = cv2.findContours(all_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise)
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    # Create a contour mask
    contour_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(contour_mask, significant_contours, -1, 255, thickness=cv2.FILLED)

    # Step 5: Combine the color segmentation mask with the contour mask
    combined_mask = cv2.bitwise_or(mask, contour_mask)

    # Step 6: Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # For complex backgrounds, apply additional processing
    # Use larger kernel for more aggressive closing
    large_kernel = np.ones((11, 11), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, large_kernel)

    # Then use original kernel for opening to remove noise
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Step 7: Find contours on the cleaned mask for final piece detection
    final_contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter again to ensure we only have significant pieces
    final_contours = [c for c in final_contours if cv2.contourArea(c) > min_contour_area]

    # Create final mask and draw filled contours
    final_mask = np.zeros_like(gray, dtype=np.uint8)
    for contour in final_contours:
        cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Create a red background
    red_background = np.full_like(original, (0, 0, 255), dtype=np.uint8)

    # Apply final mask
    foreground = cv2.bitwise_and(original, original, mask=final_mask)
    background = cv2.bitwise_and(red_background, red_background, mask=cv2.bitwise_not(final_mask))
    final_image = cv2.add(foreground, background)
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    # Extract individual pieces
    piece_images = []
    for i, contour in enumerate(final_contours):
        # Create mask for this contour
        piece_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(piece_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(rgb_image.shape[1] - x, w + 2 * padding)
        h = min(rgb_image.shape[0] - y, h + 2 * padding)

        # Extract piece
        piece = cv2.bitwise_and(rgb_image, rgb_image, mask=piece_mask)
        piece = piece[y:y + h, x:x + w]
        piece_images.append(piece)

    # Display intermediate steps if requested
    if display_steps:
        plt.figure(figsize=(15, 10))

        plt.subplot(231)
        plt.imshow(rgb_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(232)
        plt.imshow(mask, cmap='gray')
        plt.title('Initial Color Mask')
        plt.axis('off')

        plt.subplot(233)
        plt.imshow(all_edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')

        plt.subplot(234)
        plt.imshow(contour_mask, cmap='gray')
        plt.title('Contour Mask')
        plt.axis('off')

        plt.subplot(235)
        plt.imshow(final_mask, cmap='gray')
        plt.title('Final Mask')
        plt.axis('off')

        plt.subplot(236)
        plt.imshow(final_image_rgb)
        plt.title('Result with Red Background')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Display individual pieces
        if piece_images:
            num_pieces = min(6, len(piece_images))
            plt.figure(figsize=(15, 3))
            for i in range(num_pieces):
                plt.subplot(1, num_pieces, i + 1)
                plt.imshow(piece_images[i])
                plt.title(f'Piece {i + 1}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()

    return rgb_image, final_mask, final_image_rgb, piece_images

    # Create a red background
    red_background = np.full_like(original, (0, 0, 255), dtype=np.uint8)

    # Apply final mask
    foreground = cv2.bitwise_and(original, original, mask=final_mask)
    background = cv2.bitwise_and(red_background, red_background, mask=cv2.bitwise_not(final_mask))
    final_image = cv2.add(foreground, background)
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    # Extract individual pieces
    piece_images = []
    for i, contour in enumerate(final_contours):
        # Create mask for this contour
        piece_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(piece_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(rgb_image.shape[1] - x, w + 2 * padding)
        h = min(rgb_image.shape[0] - y, h + 2 * padding)

        # Extract piece
        piece = cv2.bitwise_and(rgb_image, rgb_image, mask=piece_mask)
        piece = piece[y:y + h, x:x + w]
        piece_images.append(piece)

    # Display intermediate steps if requested
    if display_steps:
        plt.figure(figsize=(15, 10))

        plt.subplot(231)
        plt.imshow(rgb_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(232)
        plt.imshow(mask, cmap='gray')
        plt.title('Initial Color Mask')
        plt.axis('off')

        plt.subplot(233)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')

        plt.subplot(234)
        plt.imshow(contour_mask, cmap='gray')
        plt.title('Contour Mask')
        plt.axis('off')

        plt.subplot(235)
        plt.imshow(final_mask, cmap='gray')
        plt.title('Final Mask')
        plt.axis('off')

        plt.subplot(236)
        plt.imshow(final_image_rgb)
        plt.title('Result with Red Background')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Display individual pieces
        if piece_images:
            num_pieces = min(6, len(piece_images))
            plt.figure(figsize=(15, 3))
            for i in range(num_pieces):
                plt.subplot(1, num_pieces, i + 1)
                plt.imshow(piece_images[i])
                plt.title(f'Piece {i + 1}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()

    return rgb_image, final_mask, final_image_rgb, piece_images


# Additional function to apply GrabCut algorithm for even better results
def apply_grabcut(image_path, init_mask):
    """
    Apply GrabCut algorithm to refine puzzle piece segmentation.

    Args:
        image_path: Path to the puzzle image
        init_mask: Initial mask from the basic method

    Returns:
        Refined mask or original mask if GrabCut fails
    """
    # Load image
    image = cv2.imread(image_path)

    # Check if there are enough foreground and background pixels
    fg_pixels = np.count_nonzero(init_mask)
    bg_pixels = init_mask.size - fg_pixels

    # Ensure we have enough foreground and background pixels (at least 100 of each)
    if fg_pixels < 100 or bg_pixels < 100:
        print("Warning: Not enough foreground or background pixels for GrabCut. Using original mask.")
        return init_mask

    try:
        # Initialize mask for GrabCut
        # 0 = background, 1 = foreground, 2 = probable background, 3 = probable foreground
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Add a border of background pixels to ensure we have enough background
        dilated_mask = cv2.dilate(init_mask, np.ones((10, 10), np.uint8))
        eroded_mask = cv2.erode(init_mask, np.ones((10, 10), np.uint8))

        # Set initial mask values with more conservative approach
        mask[dilated_mask == 0] = 0  # Definite background
        mask[eroded_mask == 255] = 1  # Definite foreground
        mask[(dilated_mask == 255) & (eroded_mask == 0)] = 3  # Probable foreground

        # Create temporary arrays for GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Apply GrabCut
        rect = (0, 0, image.shape[1], image.shape[0])  # ROI that contains the object
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        # Create final mask
        final_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

        return final_mask

    except cv2.error as e:
        print(f"GrabCut error: {e}")
        print("Falling back to original mask")
        return init_mask


# Example usage:
if __name__ == "__main__":
    # Replace with your image path
    image_path = "images/puzzlepieces.jpg"

    # Run the basic detection with enhanced parameters for complex backgrounds
    original, mask, result, pieces = detect_puzzle_pieces(image_path, display_steps=True)

    try:
        # Try GrabCut refinement, but have a fallback
        print("Attempting GrabCut refinement...")
        refined_mask = apply_grabcut(image_path, mask)
        print("GrabCut completed successfully.")

        # Create refined result with red background
        image = cv2.imread(image_path)
        red_background = np.full_like(image, (0, 0, 255), dtype=np.uint8)
        foreground = cv2.bitwise_and(image, image, mask=refined_mask)
        background = cv2.bitwise_and(red_background, red_background, mask=cv2.bitwise_not(refined_mask))
        refined_result = cv2.add(foreground, background)
        refined_result_rgb = cv2.cvtColor(refined_result, cv2.COLOR_BGR2RGB)

        # Display comparison
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(result)
        plt.title('Basic Method')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(refined_result_rgb)
        plt.title('With GrabCut Refinement')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during refinement: {e}")
        print("Displaying basic results only.")

        # Display basic results
        plt.figure(figsize=(15, 5))

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(result)
        plt.title('Detected Pieces')
        plt.axis('off')

        plt.tight_layout()
        plt.show()