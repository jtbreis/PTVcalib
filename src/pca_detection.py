import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def detect_circles(image):
    # Normalize image from 0 to 1
    # image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Step 1: artificially move grid and get data for PCA
    X = create_shifted_images(image)

    # Step 2: perform PCA
    pca = PCA(n_components=10)
    pca.fit(X)

    # Reshape the first principal component to image shape and plot
    first_pc_img = pca.components_[1].reshape(image.shape[0], image.shape[1])
    plt.figure(figsize=(8, 6))
    plt.imshow(first_pc_img, cmap='inferno')
    plt.title('First Principal Component (PCA)')
    plt.colorbar()
    plt.show()

    detect_circles_locations(first_pc_img)
    return True

def create_shifted_images(image, nimages=50, radius=9):
    # Create a dataset of 50 images by shifting img_data in a circle around the center
    # returns array X for PCA
    height, width = image.shape
    shifted_images = np.zeros((nimages, height, width), dtype=image.dtype)

    for i in range(nimages):
        angle = 2 * np.pi * i / nimages
        shift_x = int(radius * np.cos(angle))
        shift_y = int(radius * np.sin(angle))
        shifted_images[i] = np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)

    return shifted_images.reshape(nimages, -1)

def detect_circles_locations(image):
    # Detect circles
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,           # Inverse accumulator ratio
        minDist=20,       # Minimum distance between circle centers
        param1=100,       # Canny high threshold
        param2=10,        # Accumulator threshold (lower -> more detections)
        minRadius=10,     # Smallest circle radius
        maxRadius=15     # Largest circle radius
    )

    # Draw detected circles
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            # cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            cv2.circle(output, (x, y), 10, (0, 0, 255), 3)

    plt.imshow(output)
    plt.axis("off")
    plt.show()