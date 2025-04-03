from PIL import Image
import scipy.ndimage
import numpy as np
import cv2
import skimage.util
import matplotlib.pyplot as plt

# im = Image.open('images/pieces_black.jpg')
im = Image.open('images/puzzlepieces.jpg')
color_im = np.array(im)
gray_im = np.asarray(im.convert('L'), dtype="uint8") / 255.0

binarized_image = np.zeros(gray_im.shape)
binarized_image[gray_im > 0.06] = 1.0

image = scipy.ndimage.binary_opening(binarized_image, structure=np.ones((5, 5)), iterations=1)
image = scipy.ndimage.binary_closing(image, structure=np.ones((5, 5)), iterations=1)

cv_image = skimage.util.img_as_ubyte(image)
contours, hierarchy = cv2.findContours(cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

im2 = cv_image.copy()
im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
cv2.drawContours(im2, contours, -1, (255, 255, 0), 2)

plt.imshow(im2, cmap=plt.cm.gray)
plt.title("Pieces")
plt.axis('off')
plt.show()
piece_array = []
contour_images = []
for i in range(len(contours)):
    c = np.asarray(contours[i])

    # Apply erosion to move the contour 5 pixels inward
    kernel = np.ones((5, 5), np.uint8)  # Define kernel size for erosion
    mask = np.zeros_like(gray_im, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)  # Erode the mask by 5 pixels

    # Find the new contour after erosion
    eroded_contour, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eroded_c = eroded_contour[0]  # Take the first (and only) contour

    # Get the bounding box for the eroded contour
    x, y, w, h = cv2.boundingRect(eroded_c)
    padding = 2
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, color_im.shape[1] - x)
    h = min(h + 2 * padding, color_im.shape[0] - y)

    # Create a cropped image using the eroded contour mask
    cropped_piece = cv2.bitwise_and(color_im, color_im, mask=eroded_mask)
    cropped_piece = cropped_piece[y:y + h, x:x + w]
    piece_array.append(cropped_piece)

    # Create contour-only image with original colors (using eroded contour)
    contour_mask = np.zeros((h, w), dtype="uint8")
    cv2.drawContours(contour_mask, [eroded_c - [x, y]], -1, 255, thickness=1)
    contour_colored = np.zeros_like(cropped_piece)
    contour_colored = cv2.bitwise_and(cropped_piece, cropped_piece, mask=contour_mask)
    contour_images.append(contour_colored)

plt.imshow(im2, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

num_pieces = min(3, len(piece_array))
plt.figure(figsize=(10, 5))
for i in range(num_pieces):
    plt.subplot(1, 3, i + 1)
    plt.imshow(piece_array[i])
    plt.title(f"Piece {i + 1}")
    plt.axis("off")
plt.show()

# Display contour-only images
plt.figure(figsize=(10, 5))
for i in range(num_pieces):
    plt.subplot(1, 3, i + 1)
    plt.imshow(contour_images[i])
    plt.title(f"Contour {i + 1}")
    plt.axis("off")
plt.show()

for i, c in enumerate(contours):
    print(f"Contour {i + 1}: {len(c)} points")

plt.figure(figsize=(10, 5))

for i, c in enumerate(contours):
    # Create mask for the exact contour (not the filled region)
    mask = np.zeros_like(gray_im, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, thickness=1)

    # Perform morphological operations to clean up the contour (remove small noise)
    kernel = np.ones((3, 3), np.uint8)  # Small 3x3 kernel for noise removal
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Remove small holes
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)  # Remove small isolated noise

    # Extract only the cleaned contour pixels
    contour_pixels = color_im[mask_cleaned == 255]

    if contour_pixels.size == 0:
        continue  # Skip if no valid contour pixels

    # Convert to a horizontal strip and duplicate it 10 times to get a height of 10
    contour_line = np.tile(contour_pixels.reshape(1, -1, 3), (10, 1, 1))  # Shape (10, N, 3)

    plt.subplot(len(contours), 1, i + 1)
    plt.imshow(contour_line)
    plt.axis("off")

plt.show()
