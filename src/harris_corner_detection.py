import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32
    gray = np.float32(gray)
    
    # Apply Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    # Result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    harris_image = image.copy()
    harris_image[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    # Display results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detection')
    plt.axis('off')
    
    plt.show()

# Example usage
harris_corner_detection('images/000000078371.jpg')
