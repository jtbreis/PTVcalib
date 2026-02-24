import logging
import cv2
import freud
import numpy as np

from .grid_manipulation import merge_close_vertices
from .find_target_center import find_center
from scipy.spatial.distance import pdist, squareform

from ..visualization.debug_plots import (
    visualize_center,
    visualize_connections,
    visualize_grid_edges,
    visualize_voroni,
    visualize_detected_points,
)
from ..visualization.plotting import display_matched_points

logger = logging.getLogger(__name__)

# Minimum edge length as fraction of typical grid-neighbor distance (pixels); edges shorter than this are dropped
MIN_EDGE_DISTANCE_FRACTION = 0.5

# Max angle (degrees) off cardinal for a step to count as orthogonal (no diagonals)
ORTHOGONAL_ANGLE_TOL_DEG = 25.0

# Direction weights: one-way step from current node (image: +x=right, +y=down).
# 0 reserved for "no connection"; directions are 1..4.
DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 1, 2, 3, 4


def _is_orthogonal_direction(v, tol_deg=ORTHOGONAL_ANGLE_TOL_DEG):
    """True if 2D vector v is approximately horizontal or vertical (no diagonal)."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        return False
    v = v / n
    angle = np.arctan2(v[1], v[0])
    return abs(np.cos(angle)) >= np.cos(np.radians(tol_deg)) or abs(np.sin(angle)) >= np.cos(np.radians(tol_deg))


def _direction_weight(v):
    """
    Return direction weight 1=right, 2=down, 3=left, 4=up for orthogonal 2D vector v (image coords).
    Call only when _is_orthogonal_direction(v) is True. 0 is reserved for no connection.
    """
    angle = np.arctan2(v[1], v[0])
    if angle < 0:
        angle += 2 * np.pi
    if angle <= np.pi / 4 or angle > 7 * np.pi / 4:
        return DIR_RIGHT   # 1
    if angle <= 3 * np.pi / 4:
        return DIR_DOWN   # 2
    if angle <= 5 * np.pi / 4:
        return DIR_LEFT   # 3
    return DIR_UP         # 4


def _grid_adjacency(grid_points, grid_spacing, tol=0.6):
    """
    Adjacency of grid points in world XY: only direct (orthogonal) neighbors.
    i,j are neighbors iff 0 < dist <= (1+tol)*spacing and the direction i->j is horizontal or vertical (no diagonals).
    """
    grid_xy = grid_points[:, :2]
    d = squareform(pdist(grid_xy))
    thresh = (1.0 + tol) * grid_spacing
    adj = np.zeros_like(d, dtype=np.int8)
    n = d.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= 1e-9 or d[i, j] > thresh:
                continue
            v = grid_xy[j] - grid_xy[i]
            if _is_orthogonal_direction(v):
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def perform_matching(image_path: str, output_path: str, image_points, grid_points, grid_spacing, diameterDot, center_method='Simple', plot='None', image=None):
    """
    image_path: path to image (used if image is None)
    image: optional pre-loaded grayscale image to avoid re-reading from disk
    image_points: numpy array of detected points
    grid_points: numpy array with grid points
    matches numpy array (Nx5) [X, Y, Z, x, y]
    """
    if image is not None:
        image = np.asarray(image)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    raw_image = image.copy() if plot != 'None' else image
    # Preloaded images from fft_filter are float64; OpenCV display/drawing need uint8, 3-channel for cvtColor(BGR2RGB)
    if plot != 'None':
        if raw_image.dtype != np.uint8:
            raw_image = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if raw_image.ndim == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)

    # Step1: Voronoi tessellation via freud (box is centered at origin)
    box = freud.box.Box(Lx=w+10, Ly=h+10, Lz=0)
    # Convert to (N, 3) and center coordinates for freud
    points_centered = np.hstack([
        image_points - np.array([w / 2.0, h / 2.0]),
        np.zeros((len(image_points), 1)),
    ])
    voro = freud.locality.Voronoi()
    voro.compute((box, points_centered))

    # Polytopes: list of vertex arrays per cell; convert back to image coords and take xy
    offset = np.array([w / 2.0, h / 2.0])
    facets = []
    for poly in voro.polytopes:
        verts = np.asarray(poly)
        if verts.size == 0:
            facets.append(np.empty((0, 2)))
            continue
        xy = verts[:, :2] + offset
        facets.append(xy)

    # Centers: one per cell, same order as input points (use original image coordinates)
    centers = np.asarray(image_points, dtype=float)

    facets = [merge_close_vertices(np.array(f), diameterDot) for f in facets]

    center_facet, center_point = find_center(
        center_method=center_method, facets=facets, centers=centers)

    # Build adjacency matrix from Voronoi neighbor list (no periodicity)
    # voro.nlist is periodic; keep only edges where centers are close in the image
    n_cells = len(facets)
    adjacency_matrix = np.zeros((n_cells, n_cells), dtype=np.int8)
    for i, j in voro.nlist:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    # Pairwise distances between cell centers (symmetric, diagonal zero)
    center_distances = squareform(pdist(centers))

    # Drop periodic wraparound neighbors: keep edge (i,j) only if distance is "local"
    edge_dists = adjacency_matrix * center_distances
    nonzero = edge_dists[np.triu(adjacency_matrix, 1).astype(bool)]
    if len(nonzero) > 0:
        typical_spacing = np.median(nonzero)
        threshold = 2.0 * typical_spacing
        adjacency_matrix[center_distances > threshold] = 0
        # Drop edges that are too close: true grid neighbors have ~typical_spacing in pixels
        min_pixel_distance = typical_spacing * MIN_EDGE_DISTANCE_FRACTION
        adjacency_matrix[center_distances < min_pixel_distance] = 0

    # For Debug plot: edges where distance <= mean (computed only when needed)
    if plot == 'Debug':
        edge_dists = adjacency_matrix * center_distances
        n_edges = max(1, int(adjacency_matrix.sum()) // 2)
        mean_distance = edge_dists.sum() / (2 * n_edges)
        keep = (adjacency_matrix == 1) & (center_distances <= mean_distance)
        edges_to_plot = np.argwhere(np.triu(keep, 1))

    # Build one-way edges with direction weight (right, down, left, up) for image and grid
    grid_adj = _grid_adjacency(grid_points, grid_spacing)
    grid_xy = grid_points[:, :2]

    # Image: for each facet i, one outgoing edge per direction (i -> j with direction d)
    # Keep only orthogonal; if multiple neighbors in same direction, keep closest to cardinal
    image_out = [{} for _ in range(n_cells)]
    for i in range(n_cells):
        for j in np.flatnonzero(adjacency_matrix[i]):
            v = centers[j] - centers[i]
            if not _is_orthogonal_direction(v):
                continue
            d = _direction_weight(v)
            if d not in image_out[i]:
                image_out[i][d] = j
            else:
                # Keep j whose angle is closer to cardinal for this direction (d is 1..4)
                card_angles = {DIR_RIGHT: 0.0, DIR_DOWN: np.pi / 2, DIR_LEFT: np.pi, DIR_UP: -np.pi / 2}
                ang_j = np.arctan2(v[1], v[0])
                ang_curr = np.arctan2(
                    centers[image_out[i][d]][1] - centers[i][1],
                    centers[image_out[i][d]][0] - centers[i][0],
                )
                c = card_angles[d]
                if abs(np.arctan2(np.sin(ang_j - c), np.cos(ang_j - c))) < abs(
                    np.arctan2(np.sin(ang_curr - c), np.cos(ang_curr - c))
                ):
                    image_out[i][d] = j

    # Grid: for each grid point g, one outgoing edge per direction (g -> k with direction d)
    # Use image-like convention: v = (dx, -dy) for direction weight
    grid_out = [{} for _ in range(len(grid_points))]
    for g in range(len(grid_points)):
        for k in np.flatnonzero(grid_adj[g]):
            v_world = grid_xy[k] - grid_xy[g]
            v_img = np.array([v_world[0], -v_world[1]])
            if not _is_orthogonal_direction(v_img):
                continue
            d = _direction_weight(v_img)
            if d not in grid_out[g]:
                grid_out[g][d] = k
            else:
                card_angles = {DIR_RIGHT: 0.0, DIR_DOWN: np.pi / 2, DIR_LEFT: np.pi, DIR_UP: -np.pi / 2}
                ang_k = np.arctan2(v_img[1], v_img[0])
                v_curr = np.array([
                    grid_xy[grid_out[g][d]][0] - grid_xy[g][0],
                    -(grid_xy[grid_out[g][d]][1] - grid_xy[g][1]),
                ])
                ang_curr = np.arctan2(v_curr[1], v_curr[0])
                c = card_angles[d]
                if abs(np.arctan2(np.sin(ang_k - c), np.cos(ang_k - c))) < abs(
                    np.arctan2(np.sin(ang_curr - c), np.cos(ang_curr - c))
                ):
                    grid_out[g][d] = k

    # Plot grid edge connections once (orthogonal one-way edges used for stepping)
    if plot == "Debug":
        grid_edges = set(frozenset([g, k]) for g in range(len(grid_points)) for k in grid_out[g].values())
        visualize_grid_edges(grid_xy, [tuple(e) for e in grid_edges], title="Grid points and orthogonal edge connections")

    n_grid = len(grid_points)
    facet_to_grid = {}
    grid_to_facet = {k: None for k in range(n_grid)}

    # Seed: detected center <-> grid center
    center_facet_idx = int(
        np.argmin(np.linalg.norm(centers - center_point, axis=1)))
    grid_centroid = np.median(grid_points[:, :2], axis=0)
    center_grid_idx = int(np.argmin(np.linalg.norm(grid_points[:, :2] - grid_centroid, axis=1)))

    facet_to_grid[center_facet_idx] = center_grid_idx
    grid_to_facet[center_grid_idx] = center_facet_idx
    queue = [center_facet_idx]

    # BFS: step by direction; match only when image step and grid step have the same direction
    while queue:
        i = queue.pop(0)
        g = facet_to_grid[i]
        for d in (DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP):
            j = image_out[i].get(d)
            k = grid_out[g].get(d)
            if j is None or k is None:
                continue
            if j in facet_to_grid or grid_to_facet[k] is not None:
                continue
            facet_to_grid[j] = k
            grid_to_facet[k] = j
            queue.append(j)

    matches = [np.hstack([grid_points[g], centers[i]])
               for i, g in facet_to_grid.items()]

    if plot == 'Debug':
        visualize_detected_points(raw_image, centers, show_indices=True)
        visualize_center(raw_image, facets, center_facet, center_point)
        visualize_connections(raw_image, centers, edges_to_plot)
        visualize_voroni(image, facets, centers, image_points)
        display_matched_points(raw_image, matches)

    if plot == 'Normal':
        display_matched_points(raw_image, matches, output_path=output_path)

    logger.info("Matched %d calibration points (BFS from center, direction-based stepping).", len(matches))
    return matches
