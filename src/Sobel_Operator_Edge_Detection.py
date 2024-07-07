import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded
    if image is None:
        print("Error: Image not found.")
        return

    # Apply Sobel operator in the x direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel operator in the y direction
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradients
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)

    # Normalize the result to 0-255 range for display
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Plot the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X')

    plt.subplot(1, 3, 3)
    plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y')

    plt.figure()
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Combined')

    plt.show()

# Example usage
sobel_edge_detection('images/000000391895.jpg')
