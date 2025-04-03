import cv2
import numpy as np
import sys

# Load the image in color
image = cv2.imread('images/edge_detect3.png')

# Check if the image is loaded properly
if image is None:
    print("Error: Image not found.")
    sys.exit()

# Step 1: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Calculate the gradient (use Sobel or Scharr to get the gradient)
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction
grad_magnitude = cv2.magnitude(grad_x, grad_y)

# Step 3: Smooth the gradient magnitude to detect regions with smooth transitions (background)
smoothed_grad = cv2.GaussianBlur(grad_magnitude, (5, 5), 0)

# Step 4: Create a mask based on the smoothed gradient (lower gradient values indicate background)
background_mask = smoothed_grad < np.percentile(smoothed_grad, 10)  # Threshold to select the smoothest gradients

# Step 5: Replace background regions with white
output_image = image.copy()
output_image[background_mask] = [255, 255, 255]  # Replace with white

# Display the results
# cv2.imshow("Original Image", image)
cv2.imshow("Processed Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
