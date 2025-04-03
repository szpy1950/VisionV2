import cv2
import sys

# Load the image in grayscale
image = cv2.imread('images/edge_detect3.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if image is None:
    print("Error: Image not found.")
    sys.exit()

# Apply threshold to create a binary image
_, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("there are {} contours".format(len(contours)))

# Convert grayscale image to BGR for drawing
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw contours on the image
cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)

# Display the image with drawn contours
cv2.imshow("Contours", image_color)

# Print numpy array of the largest contour
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    print(largest_contour)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
