import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in color
image = cv2.imread('images/000000271177.jpg')

# Convert the image to RGB (OpenCV loads images in BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply GaussianBlur on each color channel
blurred_image = cv2.GaussianBlur(image_rgb, (11, 11), 0)

# Compute the Laplacian on each color channel separately
laplacian = np.zeros_like(blurred_image)
for i in range(3):  # Loop over the color channels
    laplacian[:, :, i] = cv2.Laplacian(blurred_image[:, :, i], cv2.CV_64F)

# Display the results
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1), plt.title('Original Image'), plt.imshow(image_rgb)
plt.subplot(1, 3, 2), plt.title('Gaussian Blurred'), plt.imshow(blurred_image)
plt.subplot(1, 3, 3), plt.title('Laplacian of Gaussian'), plt.imshow(laplacian)
plt.show()
