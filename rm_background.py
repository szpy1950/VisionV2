import cv2
import numpy as np

# Load image
image = cv2.imread("images/pieces_black.jpg")

# Convert to LAB color space for better color segmentation
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
reshaped = lab.reshape((-1, 3))

# Apply K-means clustering
K = 3  # Adjust as needed
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(np.float32(reshaped), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Find the background cluster (largest color region)
labels = labels.flatten()
background_label = np.bincount(labels).argmax()

# Create initial background mask
mask = np.where(labels == background_label, 0, 255).astype(np.uint8)
mask = mask.reshape(image.shape[:2])

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Find contours from edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill detected object contours to refine the mask
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Smooth edges with Gaussian blur
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Create a red background
red_background = np.full_like(image, (0, 0, 255), dtype=np.uint8)

# Apply mask: keep foreground, change background to red
foreground = cv2.bitwise_and(image, image, mask=mask)
background = cv2.bitwise_and(red_background, red_background, mask=cv2.bitwise_not(mask))
final_image = cv2.add(foreground, background)

# Show results
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Final Image with Red Background", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
