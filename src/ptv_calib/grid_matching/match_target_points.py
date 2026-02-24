import logging
import cv2
import freud
import numpy as np

from .grid_manipulation import scale_grid, merge_close_vertices
from .find_target_center import find_center
from scipy.spatial.distance import pdist, squareform

from ..visualization.debug_plots import (
    visualize_center,
    visualize_connections,
    visualize_grid_points,
    visualize_voroni,
    visualize_detected_points,
)
from ..visualization.plotting import display_matched_points

logger = logging.getLogger(__name__)

# Minimum edge length as fraction of typical grid-neighbor distance (pixels); edges shorter than this are dropped
MIN_EDGE_DISTANCE_FRACTION = 0.5


def _grid_adjacency(grid_points, grid_spacing, tol=0.6):
    """Adjacency of grid points in world XY: i,j are neighbors if 0 < dist <= (1+tol)*spacing*sqrt(2)."""
    d = squareform(pdist(grid_points[:, :2]))
    thresh = (1.0 + tol) * grid_spacing * np.sqrt(2)
    return ((d > 1e-9) & (d <= thresh)).astype(np.int8)


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

    grid_points_in_image = scale_grid(
        image=image, grid_points=grid_points, facets=facets, centers=centers, center_facet=center_facet, center_point=center_point, grid_spacing=grid_spacing, plot=plot)

    # Grid adjacency (world coords) for stepping along the grid
    grid_adj = _grid_adjacency(grid_points, grid_spacing)

    # Match using kept connections only: seed at center, then step facet→facet and grid→grid by center location
    n_grid = len(grid_points)
    facet_to_grid = {}
    grid_to_facet = {k: None for k in range(n_grid)}

    # Seed: center facet = facet whose center is closest to center_point; center grid = grid point (in image) closest to center_point
    center_facet_idx = int(
        np.argmin(np.linalg.norm(centers - center_point, axis=1)))
    dists_img = np.linalg.norm(grid_points_in_image - center_point, axis=1)
    center_grid_idx = int(np.argmin(dists_img))

    facet_to_grid[center_facet_idx] = center_grid_idx
    grid_to_facet[center_grid_idx] = center_facet_idx
    queue = [center_facet_idx]

    # BFS along kept-edge adjacency; at each step pick the grid neighbor whose projected position is closest to the facet center
    while queue:
        i = queue.pop(0)
        g = facet_to_grid[i]
        for j in np.flatnonzero(adjacency_matrix[i]):
            if j in facet_to_grid:
                continue
            center_j = centers[j]
            best_k, best_d = None, np.inf
            for k in np.flatnonzero(grid_adj[g]):
                if grid_to_facet[k] is not None:
                    continue
                d = np.linalg.norm(grid_points_in_image[k] - center_j)
                if d < best_d:
                    best_d, best_k = d, k
            if best_k is not None:
                facet_to_grid[j] = best_k
                grid_to_facet[best_k] = j
                queue.append(j)

    matches = [np.hstack([grid_points[g], centers[i]])
               for i, g in facet_to_grid.items()]

    if plot == 'Debug':
        visualize_detected_points(raw_image, image_points)
        visualize_center(raw_image, facets, center_facet, center_point)
        visualize_grid_points(raw_image, grid_points_in_image)
        visualize_connections(raw_image, centers, edges_to_plot)
        visualize_voroni(image, facets, centers, image_points)
        display_matched_points(raw_image, matches)

    if plot == 'Normal':
        display_matched_points(raw_image, matches, output_path=output_path)

    logger.info(
        "Matched %d calibration points using kept-edge stepping.", len(matches))
    return matches
