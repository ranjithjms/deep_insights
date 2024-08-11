import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_histogram_equalization(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Histogram Equalization
    equalized = cv2.equalizeHist(image)
    return equalized

def apply_adaptive_histogram_equalization(image, clip_limit=2.0):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    equalized = clahe.apply(image)
    return equalized

def apply_lcclahe(image, clip_limit=2.0):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform LCCLAHE (Conceptual approach, not exact implementation)
    # This is a simplified version for illustration purposes
    # In a real implementation, you would define a curve-based enhancement strategy
    # Here, we apply CLAHE and then modify the enhancement using a curve

    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    equalized = clahe.apply(image)

    # Apply Lorenz Curve or custom enhancement curve here if defined

    return equalized

# Load an image
image_path = 'images/White_Marble_Stone.jpg'
image = cv2.imread(image_path)

# Apply each method
he_output = apply_histogram_equalization(image)
ahe_output = apply_adaptive_histogram_equalization(image)
clahe_output = apply_adaptive_histogram_equalization(image, clip_limit=2.0)
lcclahe_output = apply_lcclahe(image, clip_limit=2.0)

# Display results
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(he_output, cmap='gray')
plt.title('Histogram Equalization (HE)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(ahe_output, cmap='gray')
plt.title('Adaptive Histogram Equalization (AHE)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(clahe_output, cmap='gray')
plt.title('Contrast Limited AHE (CLAHE)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(lcclahe_output, cmap='gray')
plt.title('Lorenz Curved CLAHE (LCCLAHE)')
plt.axis('off')

plt.tight_layout()
plt.show()
