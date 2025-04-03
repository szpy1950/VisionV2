import numpy as np
import cv2
import os

# Create the 'images' folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Create a 400x400 black image
image = np.zeros((400, 400), dtype=np.uint8)

# Draw a white square (sharp edge)
image[100:300, 100:300] = 255

# Save the image in the 'images' folder with the name 'edge_detect.png'
cv2.imwrite('images/edge_detect.png', image)

# Display the image
cv2.imshow("Sharp Edge Image 400x400", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
