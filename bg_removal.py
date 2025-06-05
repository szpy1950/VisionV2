#!/usr/bin/env python3
"""
Background removal using multiple colored backgrounds.
This method can be integrated into your existing puzzle code.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def remove_background_multi_color(image_paths, variance_threshold=30, debug=False):
    """
    Remove background using multiple images with different colored backgrounds.

    Args:
        image_paths: List of 3 image paths with different background colors
        variance_threshold: Threshold for color variance to detect background vs foreground
        debug: If True, shows intermediate steps

    Returns:
        tuple: (foreground_mask, composite_image, num_pieces)
    """

    # Load the three images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        images.append(img)

    print(f"Loaded {len(images)} images")

    # Ensure all images have the same dimensions
    height, width = images[0].shape[:2]
    for i, img in enumerate(images):
        if img.shape[:2] != (height, width):
            print(f"Resizing image {i} to match dimensions")
            images[i] = cv2.resize(img, (width, height))

    # Calculate color variance at each pixel
    print("Calculating color variance...")

    # Stack images to calculate variance efficiently
    image_stack = np.stack(images, axis=0)  # Shape: (3, height, width, 3)

    # Calculate variance across the 3 images for each color channel
    color_variance = np.var(image_stack, axis=0)  # Shape: (height, width, 3)

    # Calculate total variance (sum across color channels)
    total_variance = np.sum(color_variance, axis=2)  # Shape: (height, width)

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        plt.title('Image 1')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
        plt.title('Image 3')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(total_variance, cmap='hot')
        plt.colorbar()
        plt.title('Color Variance Across Images')
        plt.show()

    # Create foreground mask: low variance = foreground, high variance = background
    foreground_mask = (total_variance < variance_threshold).astype(np.uint8) * 255

    if debug:
        plt.figure(figsize=(8, 6))
        plt.imshow(foreground_mask, cmap='gray')
        plt.title(f'Initial Foreground Mask (threshold={variance_threshold})')
        plt.show()

    # Apply morphological operations to clean up the mask
    print("Cleaning up mask...")

    # Remove small noise
    kernel_small = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel_small)

    # Fill small holes
    kernel_medium = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_medium)

    if debug:
        plt.figure(figsize=(8, 6))
        plt.imshow(foreground_mask, cmap='gray')
        plt.title('Cleaned Foreground Mask')
        plt.show()

    # Create composite image (use first image as base)
    composite_image = images[0].copy()

    # Apply mask to create final result
    result_image = np.zeros_like(composite_image)
    result_image[foreground_mask > 0] = composite_image[foreground_mask > 0]

    if debug:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Foreground Objects Only')
        plt.show()

    return foreground_mask, result_image, composite_image


def detect_pieces_from_mask(foreground_mask, min_area=100, debug=False):
    """
    Detect individual pieces from the foreground mask.

    Args:
        foreground_mask: Binary mask of foreground objects
        min_area: Minimum area for a valid piece
        debug: If True, shows detection steps

    Returns:
        tuple: (contours, num_pieces, piece_info)
    """

    print("Detecting individual pieces...")

    # Find contours
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    valid_contours = []
    piece_info = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid_contours.append(contour)

            # Calculate bounding rectangle and center
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            piece_info.append({
                'id': len(valid_contours) - 1,
                'area': area,
                'bounding_rect': (x, y, w, h),
                'center': (center_x, center_y),
                'contour': contour
            })

    num_pieces = len(valid_contours)
    print(f"Detected {num_pieces} pieces")

    if debug:
        # Create visualization
        vis_image = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

        # Draw contours with different colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]

        for i, contour in enumerate(valid_contours):
            color = colors[i % len(colors)]
            cv2.drawContours(vis_image, [contour], -1, color, 2)

            # Draw piece number
            info = piece_info[i]
            cv2.putText(vis_image, str(i), info['center'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Pieces: {num_pieces}')
        plt.axis('off')
        plt.show()

        # Print piece information
        print("\nPiece Information:")
        print("=" * 50)
        for info in piece_info:
            print(f"Piece {info['id']}: Area={info['area']:.0f}, "
                  f"Center=({info['center'][0]}, {info['center'][1]}), "
                  f"BBox={info['bounding_rect']}")

    return valid_contours, num_pieces, piece_info


def test_background_removal():
    """Test the background removal on generated test images"""

    # Test image paths
    image_paths = [
        'test_images/rgb0.png',
        'test_images/rgb1.png',
        'test_images/rgb2.png'
    ]

    # Check if test images exist
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Test image not found: {path}")
            print("Please run the generator first to create test images")
            return

    print("Testing background removal...")

    # Remove background
    foreground_mask, result_image, composite_image = remove_background_multi_color(
        image_paths, variance_threshold=50, debug=True
    )

    # Detect pieces
    contours, num_pieces, piece_info = detect_pieces_from_mask(
        foreground_mask, min_area=200, debug=True
    )

    print(f"\nFinal Results:")
    print(f"Number of pieces detected: {num_pieces}")

    # Save results
    os.makedirs('results', exist_ok=True)
    cv2.imwrite('results/foreground_mask.png', foreground_mask)
    cv2.imwrite('results/result_image.png', result_image)

    print("Results saved to 'results/' directory")

    return foreground_mask, contours, num_pieces, piece_info


# Integration function for your puzzle code
def integrate_with_puzzle_code(image_paths, variance_threshold=30):
    """
    Function to integrate with your existing puzzle processing code.

    Args:
        image_paths: List of 3 image paths with different backgrounds
        variance_threshold: Threshold for background detection

    Returns:
        tuple: (contours, original_image) - ready for your existing pipeline
    """

    # Remove background
    foreground_mask, result_image, original_image = remove_background_multi_color(
        image_paths, variance_threshold=variance_threshold, debug=False
    )

    # Detect pieces
    contours, num_pieces, piece_info = detect_pieces_from_mask(
        foreground_mask, min_area=100, debug=False
    )

    print(f"Background removal complete. Found {num_pieces} pieces.")

    # Return contours and original image in the format your code expects
    return contours, original_image


if __name__ == "__main__":
    test_background_removal()