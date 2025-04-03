import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the image
image = cv2.imread('images/edge_detect3.png')

# Convert image from BGR (OpenCV) to RGB (Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Flatten the image to a 2D array of pixels (each pixel is a row of RGB values)
pixels = image_rgb.reshape((-1, 3))

# Use KMeans to cluster the colors (we can choose the number of clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)

# Get the most frequent colors (centroids of the clusters)
colors = kmeans.cluster_centers_

# Count the frequency of each color (labels from KMeans)
labels = kmeans.labels_
unique, counts = np.unique(labels, return_counts=True)
color_frequencies = dict(zip(unique, counts))

# Show the most frequent colors as bars in a histogram
plt.figure(figsize=(8, 6))
plt.bar(range(len(color_frequencies)), list(color_frequencies.values()), color=np.array(colors / 255), width=0.8)
plt.xticks(range(len(color_frequencies)), [f'Color {i}' for i in color_frequencies.keys()])
plt.xlabel('Color Clusters')
plt.ylabel('Frequency')
plt.title('Most Frequent Colors in the Image')
plt.show()

# Also, plot the histograms of individual RGB channels
plt.figure(figsize=(10, 6))

# Plot Red Channel Histogram
plt.subplot(131)
plt.hist(image_rgb[:, :, 0].ravel(), bins=256, color='red', alpha=0.7)
plt.title('Red Channel Histogram')

# Plot Green Channel Histogram
plt.subplot(132)
plt.hist(image_rgb[:, :, 1].ravel(), bins=256, color='green', alpha=0.7)
plt.title('Green Channel Histogram')

# Plot Blue Channel Histogram
plt.subplot(133)
plt.hist(image_rgb[:, :, 2].ravel(), bins=256, color='blue', alpha=0.7)
plt.title('Blue Channel Histogram')

plt.tight_layout()
plt.show()
