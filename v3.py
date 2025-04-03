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

    print("Extracting individual parts into piece images")
    piece_images = []
    test_number = 20
    for i, contour in enumerate(contours):
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        piece = np.zeros_like(original_image)
        piece[mask == 255] = original_image[mask == 255]

        x, y, w, h = cv2.boundingRect(contour)

        cropped_piece = piece[y:y + h, x:x + w]
        piece_images.append(cropped_piece)

        # storing the pieces
        output_folder = "images/extracted_pieces"
        os.makedirs(output_folder, exist_ok=True)
        piece_path = os.path.join(output_folder, f"piece_{i + 1}.png")
        cv2.imwrite(piece_path, cropped_piece)

        # Here : we chose how many pieces we want to display
        if i < 3:  # Limit to 3 pieces for display temporarily -> thursday presentation
            display_image(f"Piece {i + 1}", cropped_piece)

    output_corner_folder = "images/extracted_corners"
    os.makedirs(output_corner_folder, exist_ok=True)

    ## TODO: better edge and corners detection
    print("Analyzing edges of pieces")
    for i, contour in enumerate(contours[:test_number]):
        # Noise reduction
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Create a mask for the individual piece
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        piece = np.zeros_like(original_image)
        piece[mask == 255] = original_image[mask == 255]

        x, y, w, h = cv2.boundingRect(contour)
        cropped_piece = piece[y:y + h, x:x + w]

        # Finding the hull
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        if defects is not None:
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                far = tuple(contour[f][0])

                # Draw defect points (potential tabs or blanks) on the cropped piece
                cv2.circle(cropped_piece, (far[0] - x, far[1] - y), 5, [0, 0, 255], -1)

        # Save the processed image
        corner_path = os.path.join(output_corner_folder, f"corner_piece_{i + 1}.png")
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