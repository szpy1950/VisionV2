#!/usr/bin/env python3
"""
Enhanced background removal that handles color variations and lighting conditions.
This version is more robust for real-world scenarios.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def remove_background_multi_color_enhanced(image_paths, variance_threshold=40,
                                           adaptive_threshold=True, debug=False):
    """
    Enhanced background removal using multiple images with varied colored backgrounds.

    Args:
        image_paths: List of 3 image paths with different background colors
        variance_threshold: Base threshold for color variance detection
        adaptive_threshold: If True, automatically adjusts threshold based on image statistics
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

    # Calculate adaptive threshold if requested
    if adaptive_threshold:
        # Calculate statistics of the variance
        variance_mean = np.mean(total_variance)
        variance_std = np.std(total_variance)

        # Use percentile-based threshold (more robust than mean/std)
        threshold_adaptive = np.percentile(total_variance, 25)  # 25th percentile

        # Combine with base threshold
        final_threshold = min(variance_threshold, threshold_adaptive * 1.5)

        print(f"Variance stats: mean={variance_mean:.1f}, std={variance_std:.1f}")
        print(f"Adaptive threshold: {threshold_adaptive:.1f}")
        print(f"Final threshold: {final_threshold:.1f}")
    else:
        final_threshold = variance_threshold

    if debug:
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.title(f'Image {i + 1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

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
        plt.figure(figsize=(8, 6))
        plt.imshow(foreground_mask, cmap='gray')
        plt.title(f'Initial Foreground Mask (threshold={final_threshold:.1f})')
        plt.show()

    # Enhanced morphological operations
    print("Cleaning up mask with enhanced morphology...")

    # Remove small noise (opening)
    kernel_small = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel_small)

    # Fill holes (closing)
    kernel_medium = np.ones((7, 7), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_medium)

    # Remove remaining small objects
    kernel_large = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel_large)

    if debug:
        plt.figure(figsize=(8, 6))
        plt.imshow(foreground_mask, cmap='gray')
        plt.title('Enhanced Cleaned Foreground Mask')
        plt.show()

    # Create composite image (use first image as base)
    composite_image = images[0].copy()

    # Apply mask to create final result
    result_image = np.zeros_like(composite_image)
    result_image[foreground_mask > 0] = composite_image[foreground_mask > 0]

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Foreground Objects Only')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return foreground_mask, result_image, composite_image


def detect_pieces_enhanced(foreground_mask, min_area=200, debug=False):
    """
    Enhanced piece detection with better filtering.
    """

    print("Detecting individual pieces with enhanced filtering...")

    # Find contours
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Enhanced filtering
    valid_contours = []
    piece_info = []

    # Calculate area statistics for adaptive filtering
    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        adaptive_min_area = max(min_area, mean_area - 2 * std_area)
        print(f"Area stats: mean={mean_area:.0f}, std={std_area:.0f}, adaptive_min={adaptive_min_area:.0f}")
    else:
        adaptive_min_area = min_area

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Area filter
        if area < adaptive_min_area:
            continue

        # Aspect ratio filter (remove very thin objects)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 10:  # Remove very elongated objects
            continue

        # Solidity filter (remove objects with too many holes)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.3:  # Remove objects that are too "holey"
                continue

        valid_contours.append(contour)

        # Calculate additional properties
        center_x = x + w // 2
        center_y = y + h // 2

        piece_info.append({
            'id': len(valid_contours) - 1,
            'area': area,
            'bounding_rect': (x, y, w, h),
            'center': (center_x, center_y),
            'contour': contour,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity if hull_area > 0 else 0
        })

    num_pieces = len(valid_contours)
    print(f"Detected {num_pieces} valid pieces after enhanced filtering")

    if debug:
        # Create visualization
        vis_image = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

        # Draw contours with different colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0),
                  (128, 255, 128), (255, 128, 255), (128, 255, 255), (192, 192, 192)]

        for i, contour in enumerate(valid_contours):
            color = colors[i % len(colors)]
            cv2.drawContours(vis_image, [contour], -1, color, 2)

            # Draw piece number
            info = piece_info[i]
            cv2.putText(vis_image, str(i), info['center'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw bounding rectangle
            x, y, w, h = info['bounding_rect']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Enhanced Detection: {num_pieces} Valid Pieces')
        plt.axis('off')
        plt.show()

        # Print piece information
        print("\nEnhanced Piece Information:")
        print("=" * 70)
        for info in piece_info:
            print(f"Piece {info['id']:2d}: Area={info['area']:6.0f}, "
                  f"Center=({info['center'][0]:3d}, {info['center'][1]:3d}), "
                  f"Aspect={info['aspect_ratio']:.2f}, Solidity={info['solidity']:.2f}")

    return valid_contours, num_pieces, piece_info


def test_enhanced_background_removal():
    """Test the enhanced background removal on generated test images"""

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
            print("Please run the enhanced generator first to create test images")
            return

    print("Testing enhanced background removal...")

    # Remove background with enhanced method
    foreground_mask, result_image, composite_image = remove_background_multi_color_enhanced(
        image_paths, variance_threshold=40, adaptive_threshold=True, debug=True
    )

    # Detect pieces with enhanced method
    contours, num_pieces, piece_info = detect_pieces_enhanced(
        foreground_mask, min_area=300, debug=True
    )

    print(f"\nEnhanced Results:")
    print(f"Number of pieces detected: {num_pieces}")
    print(f"Expected pieces: ~12")
    print(f"Detection accuracy: {min(100, num_pieces / 12 * 100):.1f}%")

    # Save results
    os.makedirs('results', exist_ok=True)
    cv2.imwrite('results/enhanced_foreground_mask.png', foreground_mask)
    cv2.imwrite('results/enhanced_result_image.png', result_image)

    print("Enhanced results saved to 'results/' directory")

    return foreground_mask, contours, num_pieces, piece_info


if __name__ == "__main__":
    test_enhanced_background_removal()