#!/usr/bin/env python3
"""
Generator for test images with different colored backgrounds and random shapes.
Creates 3 images with pure Red, Green, and Blue backgrounds with identical shapes.
"""

import cv2
import numpy as np
import random
import os


def generate_test_images():
    """Generate 3 test images with different background colors and identical shapes"""

    # Image dimensions
    width, height = 800, 600

    # Base background colors with random variations (BGR format for OpenCV)
    base_colors = {
        'red': (0, 0, 255),  # Base Red
        'green': (0, 255, 0),  # Base Green
        'blue': (255, 0, 0)  # Base Blue
    }

    # Function to add random variation to a color
    def add_color_variation(color, variation=50):
        """Add random variation to each color channel independently within acceptable range"""
        varied_color = []
        for channel in color:
            # Add random variation between -variation and +variation to EACH channel
            new_value = channel + random.randint(-variation, variation)
            # Clamp to valid range [0, 255]
            new_value = max(0, min(255, new_value))
            varied_color.append(new_value)
        return tuple(varied_color)

    # Generate varied colors for each image
    colors = {}
    for color_name, base_color in base_colors.items():
        colors[color_name] = add_color_variation(base_color, variation=10)

    # Shape colors (different from backgrounds)
    shape_colors = [
        (128, 128, 128),  # Gray
        (255, 255, 255),  # White
        (0, 0, 0),  # Black
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Dark Cyan
        (128, 128, 0),  # Olive
        (192, 192, 192),  # Light Gray
    ]

    # Generate random shapes (consistent across all images)
    shapes = []
    random.seed(42)  # Fixed seed for reproducible results

    # Generate 12 non-overlapping rectangles
    occupied_areas = []

    for i in range(12):
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            # Random rectangle dimensions
            rect_width = random.randint(40, 80)
            rect_height = random.randint(30, 70)

            # Random position
            x = random.randint(20, width - rect_width - 20)
            y = random.randint(20, height - rect_height - 20)

            # Check for overlap with existing shapes
            new_rect = (x, y, x + rect_width, y + rect_height)
            overlap = False

            for existing_rect in occupied_areas:
                if rectangles_overlap(new_rect, existing_rect):
                    overlap = True
                    break

            if not overlap:
                # Add some padding around the rectangle
                padded_rect = (x - 10, y - 10, x + rect_width + 10, y + rect_height + 10)
                occupied_areas.append(padded_rect)

                # Choose random shape type and color
                shape_type = random.choice(['rectangle', 'circle', 'triangle'])
                color = shape_colors[i % len(shape_colors)]

                shapes.append({
                    'type': shape_type,
                    'x': x,
                    'y': y,
                    'width': rect_width,
                    'height': rect_height,
                    'color': color
                })
                break

            attempts += 1

    print(f"Generated {len(shapes)} shapes")

    # Create output directory
    os.makedirs('test_images', exist_ok=True)

    # Generate varied colors for each image (each image gets slightly different background)
    all_image_colors = []
    for i, (color_name, base_color) in enumerate(base_colors.items()):
        # Each image gets its own variation of the base color
        varied_color = add_color_variation(base_color, variation=10)
        all_image_colors.append((color_name, varied_color))
        print(f"Image {i} ({color_name}): Base {base_color} -> Varied {varied_color}")

    # Also add some variation to each pixel within the background
    def create_varied_background(width, height, base_color, pixel_variation=8):
        """Create background with slight pixel-level variations on all channels"""
        # Start with solid color
        background = np.full((height, width, 3), base_color, dtype=np.uint8)

        # Add random noise to each pixel on ALL channels
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    # Each channel gets independent random variation
                    noise = random.randint(-pixel_variation, pixel_variation)
                    new_value = int(background[y, x, c]) + noise
                    background[y, x, c] = max(0, min(255, new_value))

        return background

    # Generate images for each background color
    for i, (color_name, bg_color) in enumerate(all_image_colors):
        # Create image with varied background (simulates lighting variations)
        image = create_varied_background(width, height, bg_color, pixel_variation=8)

        # Draw all shapes
        for shape in shapes:
            if shape['type'] == 'rectangle':
                cv2.rectangle(image,
                              (shape['x'], shape['y']),
                              (shape['x'] + shape['width'], shape['y'] + shape['height']),
                              shape['color'], -1)

            elif shape['type'] == 'circle':
                center_x = shape['x'] + shape['width'] // 2
                center_y = shape['y'] + shape['height'] // 2
                radius = min(shape['width'], shape['height']) // 2
                cv2.circle(image, (center_x, center_y), radius, shape['color'], -1)

            elif shape['type'] == 'triangle':
                # Create triangle points
                pts = np.array([
                    [shape['x'] + shape['width'] // 2, shape['y']],  # Top point
                    [shape['x'], shape['y'] + shape['height']],  # Bottom left
                    [shape['x'] + shape['width'], shape['y'] + shape['height']]  # Bottom right
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(image, [pts], shape['color'])

        # Save image
        filename = f'test_images/rgb{i}.png'
        cv2.imwrite(filename, image)
        print(
            f"Created {filename} with varied {color_name} background (base: {base_colors[color_name]}, actual: {bg_color})")

    return shapes


def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap"""
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    return not (x1_max <= x2_min or x2_max <= x1_min or
                y1_max <= y2_min or y2_max <= y1_min)


if __name__ == "__main__":
    shapes = generate_test_images()
    print(f"Generated test images with {len(shapes)} shapes")
    print("Files created: test_images/rgb0.png, test_images/rgb1.png, test_images/rgb2.png")