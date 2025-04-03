import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

    output_folder_pieces = "images/extracted_pieces"
    os.makedirs(output_folder_pieces, exist_ok=True)

    output_folder_contours = "images/extracted_contours"
    os.makedirs(output_folder_contours, exist_ok=True)


    print("Extracting individual parts into piece images")

    piece_images = []
    output_folder_pieces = "images/extracted_pieces"
    os.makedirs(output_folder_pieces, exist_ok=True)

    output_folder_contours = "images/extracted_contours"
    os.makedirs(output_folder_contours, exist_ok=True)

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

    test_number = 24


    ### separated part because I want to put it in a different file later, i dont want to redo the whole piece
    ### extraction every time
    print("Analyzing edges of pieces using Harris Corner Detection")
    for idx, contour in enumerate(contours[:test_number]):
        # Extract individual piece
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        piece = np.zeros_like(original_image)
        piece[mask == 255] = original_image[mask == 255]
        cropped_piece = piece[y:y + h, x:x + w]
        mask_cropped = mask[y:y + h, x:x + w]

        # Harris detection only on the piece mask
        gray_piece = cv2.cvtColor(cropped_piece, cv2.COLOR_BGR2GRAY)
        gray_piece = np.float32(gray_piece)

        # Create a masked version of the grayscale piece
        # This ensures Harris only processes the actual piece
        masked_gray_piece = gray_piece.copy()
        masked_gray_piece[mask_cropped == 0] = 0

        # Apply Harris corner detection
        harris_corners = cv2.cornerHarris(masked_gray_piece, blockSize=5, ksize=3, k=0.06)
        harris_corners = cv2.dilate(harris_corners, None)

        # Focus on the mask edges by zeroing out non-edge areas
        # Create edge mask from the piece mask
        edge_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.Canny(mask_cropped, 100, 200)

        # Only keep corner responses near edges
        edge_dilated = cv2.dilate(edges, edge_kernel, iterations=2)
        harris_corners[edge_dilated == 0] = 0

        # Find corner points above threshold
        threshold = 0.01 * harris_corners.max()
        corner_points = np.argwhere(harris_corners > threshold)

        # Convert to original image coordinates
        detected_corners = [(cx, cy) for cy, cx in corner_points]

        # Draw pink contour
        cv2.drawContours(cropped_piece, [contour - [x, y]], -1, (255, 0, 255), 1, lineType=cv2.LINE_8)

        # Draw ALL detected corners
        for corner in detected_corners:
            cv2.circle(cropped_piece, corner, 1, (255, 0, 0), -1)

        # Save the processed image
        corner_path = os.path.join(output_corner_folder, f"corner_piece_{idx + 1}.png")
        cv2.imwrite(corner_path, cropped_piece)

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