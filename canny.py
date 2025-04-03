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

    print("Applying Canny edge detection")
    edges = cv2.Canny(gray_image, 100, 200)
    display_image("Canny Edges", edges)

    print("Finding contours of pieces")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    output_folder = "images/canny_extraction"
    os.makedirs(output_folder, exist_ok=True)

    for i, contour in enumerate(contours):
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        piece = np.zeros_like(original_image)
        piece[mask == 255] = original_image[mask == 255]

        x, y, w, h = cv2.boundingRect(contour)
        cropped_piece = piece[y:y + h, x:x + w]
        piece_images.append(cropped_piece)

        piece_path = os.path.join(output_folder, f"piece_{i + 1}.png")
        cv2.imwrite(piece_path, cropped_piece)

        if i < 3:
            display_image(f"Piece {i + 1}", cropped_piece)

    print("Analyzing edges of pieces")
    for i, contour in enumerate(contours[:3]):
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        edge_image = np.zeros_like(original_image)
        cv2.drawContours(edge_image, [approx], 0, (0, 255, 0), 2)
        display_image(f"Analyzing edges of piece {i + 1}", edge_image)

    return original_image, contours, piece_images

def main():
    image_path = "images/hack2.png"
    try:
        original_image, pieces, piece_images = extract_puzzle_pieces(image_path)
        print(f"We have found {len(pieces)} puzzle pieces")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    main()
