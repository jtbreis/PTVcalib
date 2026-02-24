import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


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


def _close_pairs(points, factor=0.4):
    """Return list of (i, j) pairs whose distance is < factor * median(nearest-neighbor distance)."""
    n = len(points)
    if n < 2:
        return []
    d = squareform(pdist(points))
    np.fill_diagonal(d, np.inf)
    nn_dist = np.min(d, axis=1)
    median_nn = np.median(nn_dist)
    if median_nn <= 0:
        return []
    thresh = factor * median_nn
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < thresh:
                pairs.append((i, j))
    return pairs


def _far_points(points, factor=1.6):
    """Return set of point indices whose nearest-neighbor distance is > factor * median(nn distance)."""
    n = len(points)
    if n < 2:
        return set()
    d = squareform(pdist(points))
    np.fill_diagonal(d, np.inf)
    nn_dist = np.min(d, axis=1)
    median_nn = np.median(nn_dist)
    if median_nn <= 0:
        return set()
    thresh = factor * median_nn
    return set(i for i in range(n) if nn_dist[i] > thresh)


def _edge_points(points, width, height, margin_fraction=0.015):
    """Return set of point indices that lie really close to the image border (e.g. bottom-right corner)."""
    margin = max(8, margin_fraction * min(width, height))
    n = len(points)
    out = set()
    for i in range(n):
        x, y = points[i, 0], points[i, 1]
        if x < margin or x > width - margin or y < margin or y > height - margin:
            out.add(i)
    return out


def visualize_detected_points(image, image_points, show_indices=True, camera_index=None, layer_index=None, show_close_pairs=True):
    """
    Plot detected points on the image. image_points: Nx2 array or list of (x,y).
    If show_indices is True, each point is labeled with its index (0, 1, 2, ...).
    If show_close_pairs is True, highlights: red = very close pairs; yellow = other potential weird (close);
    blue = too far from neighbors (isolated); orange = near image edge. Index backgrounds match.
    Figure size and DPI are set so the image and index text are readable.
    If camera_index and layer_index are provided, they are shown in the plot title.
    """
    points = np.asarray(image_points, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2))
    else:
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        if points.shape[1] != 2:
            points = points.T
    img = _ensure_uint8_bgr(image)
    h, w = img.shape[:2]
    # Larger figure so the image and index numbers are readable (longer side ~14 inches, 150 DPI)
    fig_inches = 14
    scale = max(h, w) / fig_inches
    figsize = (w / scale, h / scale)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.scatter(points[:, 0], points[:, 1], c='red', s=30, zorder=2, picker=5)
    # Red = very close. Yellow = other potential weird (close). Blue = too far. Orange = near image edge.
    close_point_indices = set()   # red
    yellow_point_indices = set()  # yellow only (not in red)
    far_point_indices = set()     # too far from neighbors
    edge_point_indices = _edge_points(points, w, h)  # near border (e.g. bottom-right)
    if show_close_pairs and points.size >= 4:
        close_red = _close_pairs(points, factor=0.4)
        close_yellow = _close_pairs(points, factor=0.78)  # more sensitive (potential weird candidates)
        far_point_indices = _far_points(points, factor=1.6)  # isolated / too far
        for i, j in close_red:
            close_point_indices.add(i)
            close_point_indices.add(j)
        for i, j in close_yellow:
            if i not in close_point_indices:
                yellow_point_indices.add(i)
            if j not in close_point_indices:
                yellow_point_indices.add(j)
        for i, j in close_red:
            ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'r-', lw=2.5, zorder=2)
        for i, j in close_yellow:
            if (i, j) not in set(close_red) and (j, i) not in set(close_red):
                ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='gold', lw=2, zorder=2)
        if close_point_indices:
            cx = points[list(close_point_indices), 0]
            cy = points[list(close_point_indices), 1]
            ax.scatter(cx, cy, s=120, facecolors='none', edgecolors='red', linewidths=2.5, zorder=2)
        if yellow_point_indices:
            yx = points[list(yellow_point_indices), 0]
            yy = points[list(yellow_point_indices), 1]
            ax.scatter(yx, yy, s=100, facecolors='none', edgecolors='gold', linewidths=2, zorder=2)
        if far_point_indices:
            fx = points[list(far_point_indices), 0]
            fy = points[list(far_point_indices), 1]
            ax.scatter(fx, fy, s=100, facecolors='none', edgecolors='dodgerblue', linewidths=2, zorder=2)
    if edge_point_indices:
        ex = points[list(edge_point_indices), 0]
        ey = points[list(edge_point_indices), 1]
        ax.scatter(ex, ey, s=100, facecolors='none', edgecolors='darkorange', linewidths=2, zorder=2)
    if show_indices:
        for i, (x, y) in enumerate(points):
            if i in close_point_indices:
                bbox_facecolor = 'red'
                text_color = 'white'
            elif i in yellow_point_indices:
                bbox_facecolor = 'gold'
                text_color = 'black'
            elif i in far_point_indices:
                bbox_facecolor = 'dodgerblue'
                text_color = 'white'
            elif i in edge_point_indices:
                bbox_facecolor = 'darkorange'
                text_color = 'white'
            else:
                bbox_facecolor = 'black'
                text_color = 'white'
            ax.text(x, y, str(i), color=text_color, fontsize=4, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor=bbox_facecolor, alpha=1, edgecolor='none'), zorder=3)
    if camera_index is not None and layer_index is not None:
        ax.set_title(f"Camera {camera_index}, Layer {layer_index}")
    ax.axis("off")
    plt.tight_layout()
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
