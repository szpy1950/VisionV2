import cv2
import sys
import numpy as np

# Load the image in color
image = cv2.imread('images/edge_detect3.png')

# Check if the image is loaded properly
if image is None:
    print("Error: Image not found.")
    sys.exit()

# Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

channels = cv2.split(image)  # Split into B, G, R channels
edges = [cv2.Canny(ch, 100, 200) for ch in channels]  # Apply Canny on each channel
canny_edges = cv2.bitwise_or(edges[0], edges[1])
canny_edges = cv2.bitwise_or(canny_edges, edges[2])  # Merge results

#
# # Sobel Edge Detection
# sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# sobel_edges = cv2.magnitude(sobel_x, sobel_y)
# sobel_edges = np.uint8(sobel_edges)
# sobel_image = image.copy()
# sobel_image[sobel_edges > 50] = [0, 0, 255]  # Red edges
#
# # Laplacian Edge Detection
# laplacian_edges = cv2.Laplacian(gray, cv2.CV_64F)
# laplacian_edges = np.uint8(np.abs(laplacian_edges))
# laplacian_image = image.copy()
# laplacian_image[laplacian_edges > 50] = [0, 0, 255]  # Red edges
#
# # Canny Edge Detection
# canny_edges = cv2.Canny(gray, 100, 200)
# canny_image = image.copy()
# canny_image[canny_edges > 0] = [0, 0, 255]  # Red edges

# Contour Detection
contour_image = image.copy()
contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)  # Draw contours in red

# Display the original image
# cv2.imshow("Original Image", image)


# Display edge detection results (Uncomment one at a time to test)
# cv2.imshow("Sobel Edges", sobel_image)
# cv2.imshow("Laplacian Edges", laplacian_image)
# cv2.imshow("Canny Edges", canny_image)
cv2.imshow("Contour Detection", contour_image)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
