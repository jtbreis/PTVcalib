import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_matched_points_from_path(image_path, matches_layer, title=None, save_path=None):
    """Load image from path and display it with matched grid points (for a single layer)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    matches = np.asarray(matches_layer)
    if matches.size > 0 and matches.ndim == 1:
        matches = matches.reshape(1, -1)
    display_matched_points(img, matches, output_path=save_path, title=title)


def display_matched_points(img, matches, output_path=None, title=None):
    # OpenCV cvtColor requires uint8 or float32; preloaded images from fft_filter are float64
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(display)
    for match in matches:
        # img_pt: (x, y) in image, real_world_pt: (X, Y) in real world
        [X, Y, Z, x, y] = match
        plt.scatter(x, y, s=120, edgecolors='yellow',
                    facecolors='none', linewidths=2)
        plt.text(x + 5, y - 15, f'{X:.2f}', color='white', fontsize=10,
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))
        plt.text(x + 5, y + 15, f'{Y:.2f}', color='white', fontsize=10,
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))
    plt.title(title if title else "Matched Points with Real World Coordinates")
    plt.axis("off")
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()
