import cv2
import numpy as np
import matplotlib.pyplot as plt


def _ensure_uint8_bgr(image):
    """Convert image to uint8 BGR for OpenCV drawing (rejects CV_64F)."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def visualize_voroni(image, facets, centers, points):
    image = _ensure_uint8_bgr(image)
    for idx, facet in enumerate(facets):
        pts = np.array(facet, np.int32)
        cv2.fillConvexPoly(image, pts, (np.random.randint(
            256), np.random.randint(256), np.random.randint(256)))
        cv2.polylines(image, [pts], True, (0, 0, 0), 1)
        # Annotate number of vertices for each facet
        num_vertices = len(facet)
        plt.text(centers[idx, 0], centers[idx, 1], str(num_vertices), color='black', fontsize=10,
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw points
    for p in points:
        center = (int(p[0]), int(p[1]))
        cv2.circle(image, center, 4, (0, 0, 255), -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def visualize_center(image, facets, center, center_point):
    # Draw all facets faintly
    img_highlight = _ensure_uint8_bgr(image)
    for facet in facets:
        pts = np.array(facet, np.int32)
        overlay = img_highlight.copy()
        cv2.fillConvexPoly(overlay, pts, (200, 200, 200))
        alpha = 0.3  # transparency factor
        cv2.addWeighted(overlay, alpha, img_highlight,
                        1 - alpha, 0, img_highlight)

    # Highlight the center facet in red
    pts_center = np.array(center, np.int32)
    cv2.fillConvexPoly(img_highlight, pts_center, (255, 0, 0))
    cv2.polylines(img_highlight, [pts_center], True, (0, 0, 0), 2)

    # Draw the center point as a green dot
    center_coords = (int(center_point[0]), int(center_point[1]))
    cv2.circle(img_highlight, center_coords, 6, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(img_highlight, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def visualize_grid_points(image, grid_points):
    plt.scatter(grid_points[:, 0], grid_points[:, 1])
    img = _ensure_uint8_bgr(image)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def visualize_detected_points(image, image_points):
    points = np.vstack(image_points)
    print(points.shape)
    plt.scatter(points[:, 0], points[:, 1])
    img = _ensure_uint8_bgr(image)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def visualize_connections(image, centers, edges):
    """
    Draw lines between centers for the given edges only.
    edges: (N, 2) array of (i, j) facet indices; line is drawn between centers[i] and centers[j].
    """
    img = _ensure_uint8_bgr(image)
    for i, j in edges:
        pt1 = (int(centers[i, 0]), int(centers[i, 1]))
        pt2 = (int(centers[j, 0]), int(centers[j, 1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)
    for idx in range(len(centers)):
        c = (int(centers[idx, 0]), int(centers[idx, 1]))
        cv2.circle(img, c, 3, (0, 0, 255), -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
