import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def display_matched_points_from_path(image_path, matches_layer, title=None, save_path=None):
    """Load image from path and display it with matched grid points (for a single layer)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    matches = np.asarray(matches_layer)
    if matches.size > 0 and matches.ndim == 1:
        matches = matches.reshape(1, -1)
    display_matched_points(img, matches, output_path=save_path, title=title)


def _draw_annotated_points(ax, display, matches, values=None, cmap='viridis', label=''):
    """Draw image, optional value overlay (with colorbar), and annotated grid points on ax."""
    ax.imshow(display)
    if values is not None:
        x_img = matches[:, 3].astype(float)
        y_img = matches[:, 4].astype(float)
        valid = np.isfinite(values)
        if np.sum(valid) >= 3:
            x_v, y_v = x_img[valid], y_img[valid]
            v = values[valid]
            try:
                tri = Triangulation(x_v, y_v)
                tcf = ax.tricontourf(tri, v, levels=20, cmap=cmap, alpha=0.4)
                plt.colorbar(tcf, ax=ax, shrink=0.5, pad=0.02, label=label)
            except Exception:
                pass
    for match in matches:
        [X, Y, Z, x, y] = match
        ax.scatter(x, y, s=120, edgecolors='yellow', facecolors='none', linewidths=2)
        ax.text(x + 5, y - 15, f'{X:.2f}', color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=1))
        ax.text(x + 5, y + 15, f'{Y:.2f}', color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=1))
    ax.axis("off")


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
    matches = np.asarray(matches)
    if matches.ndim == 1:
        matches = matches.reshape(1, -1)
    X_world = matches[:, 0].astype(float)   # real-world X (mm) at each grid point
    Y_world = matches[:, 1].astype(float)   # real-world Y (mm) at each grid point
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(40, 20))
    _draw_annotated_points(ax_left, display, matches, X_world, cmap='coolwarm', label='X (mm)')
    _draw_annotated_points(ax_right, display, matches, Y_world, cmap='coolwarm', label='Y (mm)')
    ax_left.set_title('X (mm)' if title is None else f'{title} — X')
    ax_right.set_title('Y (mm)' if title is None else f'{title} — Y')
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()
