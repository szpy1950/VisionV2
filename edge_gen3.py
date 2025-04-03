import numpy as np
import cv2
import os
import random


def generate_non_overlapping_shapes(image_size, num_shapes):
    # Create a gradient background
    bg = np.linspace(50, 200, image_size, dtype=np.uint8)  # Gradient from dark to light gray
    image = np.tile(bg, (image_size, 1))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image

    shapes = []

    def precise_overlap(new_shape, existing_shapes, min_distance=10):
        x1_min, y1_min, x1_max, y1_max = new_shape
        x1_min -= min_distance
        y1_min -= min_distance
        x1_max += min_distance
        y1_max += min_distance

        for x2_min, y2_min, x2_max, y2_max in existing_shapes:
            if x1_max > x2_min and x1_min < x2_max and y1_max > y2_min and y1_min < y2_max:
                return True
        return False

    for _ in range(num_shapes):
        attempts = 0
        while attempts < 100:  # Limit attempts to prevent infinite loops
            shape_type = random.choice(['square', 'triangle'])
            size = random.randint(20, 40)
            x, y = random.randint(0, image_size - size), random.randint(0, image_size - size)

            new_shape = (x, y, x + size, y + size)
            if not precise_overlap(new_shape, shapes, min_distance=10):
                shapes.append(new_shape)

                # Random color including different shades of gray
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if random.random() < 0.3:  # 30% chance of a grayscale color
                    gray_value = random.randint(30, 220)
                    color = (gray_value, gray_value, gray_value)

                if shape_type == 'square':
                    cv2.rectangle(image, (x, y), (x + size, y + size), color, thickness=-1)
                else:
                    pts = np.array([[x, y + size], [x + size // 2, y], [x + size, y + size]], np.int32)
                    cv2.fillPoly(image, [pts], color)
                break

            attempts += 1  # Increase attempt counter

    return image


# Create 'images' folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Generate image with gradient background and colored shapes
num_shapes = 50
image = generate_non_overlapping_shapes(400, num_shapes)

# Save the image
cv2.imwrite('images/edge_detect3.png', image)

# Display the image
cv2.imshow("Gradient Background with Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()