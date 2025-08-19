import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_target_points(image, diameterDot=10):
    # Step 1: create a shifted image with a known radius
    radiusDot = int(diameterDot/2)
    img = create_shifted_image(image, radius=radiusDot)

    points = detect_circles_locations(img, radiusDot)
    return points

def create_shifted_image(image, nimages=50, radius=5):
    # Create a dataset of 50 images by shifting img_data in a circle around the center
    # returns array X for PCA
    height, width = image.shape
    shifted_image = np.zeros((height, width), dtype=image.dtype)

    for i in range(nimages):
        angle = 2 * np.pi * i / nimages
        shift_x = int(radius * np.cos(angle))
        shift_y = int(radius * np.sin(angle))
        shifted_image += np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)

    return shifted_image

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