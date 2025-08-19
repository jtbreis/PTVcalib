import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from pydmd import DMD


## TODO: add criterias based on the radius of a target point in the image

def detect_circles(image, radiusDot=5):
    # Normalize image from 0 to 1
    # image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Step 1: artificially move grid and get data for PCA
    X = create_shifted_images(image, radius=radiusDot*1.5)

    first_mode = perform_PCA(X, image)
    # first_mode = perform_DMD(X, image)

    points = detect_circles_locations(first_mode, radiusDot)
    return points

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

def detect_circles_locations(image, radiusMin):
    # Detect circles
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,           # Inverse accumulator ratio
        minDist=20,       # Minimum distance between circle centers
        param1=100,       # Canny high threshold
        param2=10,        # Accumulator threshold (lower -> more detections)
        minRadius=int(radiusMin),     # Smallest circle radius
        maxRadius=int(radiusMin*1.5)     # Largest circle radius
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

    if circles is not None:
        centers = [(int(x), int(y)) for x, y, r in circles[0, :]]
        return centers
    else:
        return []
    
def perform_PCA(X, image):
    # Step 2: perform PCA
    pca = PCA(n_components=10)
    pca.fit(X)

    # Reshape the first principal component to image shape and plot
    leading_mode = pca.components_[0].reshape(image.shape[0], image.shape[1])
    plt.figure(figsize=(8, 6))
    plt.imshow(leading_mode, cmap='inferno')
    plt.title('First Principal Component (PCA)')
    plt.colorbar()
    plt.show()
    return leading_mode

def perform_DMD(X, image, n_modes=10):
    """
    Perform Dynamic Mode Decomposition (DMD) on the shifted images dataset X using pydmd.
    Returns the leading DMD mode reshaped to the image shape.
    """
    # Arrange X as snapshots: columns are flattened images
    snapshots = X.T  # shape: (pixels, nimages)

    dmd = DMD(svd_rank=n_modes)
    dmd.fit(snapshots)

    # Take the leading DMD mode and reshape
    leading_mode = np.abs(dmd.modes[:, 0]).reshape(image.shape)
    plt.figure(figsize=(8, 6))
    plt.imshow(leading_mode, cmap='viridis')
    plt.title('Leading DMD Mode (PyDMD)')
    plt.colorbar()
    plt.show()
    return leading_mode
