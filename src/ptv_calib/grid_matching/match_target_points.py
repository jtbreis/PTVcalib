import cv2
import freud
import numpy as np
import pandas as pd

from .grid_manipulation import scale_grid, merge_close_vertices
from .grid_checks import is_almost_square, point_in_polygon
from .find_target_center import find_center
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from ..visualization.debug_plots import (
    visualize_center,
    visualize_connections,
    visualize_grid_points,
    visualize_voroni,
    visualize_detected_points,
)
from ..visualization.plotting import display_matched_points


def perform_matching(image_path: str, output_path: str, image_points, grid_points, grid_spacing, diameterDot, center_method='Simple', plot='None'):
    """
    image: input image make sure it is grayscale
    image_points: numpy array of detected points
    grid_points: numpy array with grid points
    matches numpy array (Nx5) [X, Y, Z, x, y]
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    raw_image = image.copy()

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
        threshold = 2.0 * np.median(nonzero)
        periodic = center_distances > threshold
        adjacency_matrix[periodic] = 0

    # Connections to plot: only where distance >= mean (over edges)
    edge_distances = adjacency_matrix * center_distances
    n_edges = max(1, int(adjacency_matrix.sum()) // 2)
    mean_distance = edge_distances.sum() / (2 * n_edges)
    keep = (adjacency_matrix == 1) & (center_distances <= mean_distance)
    # (i, j) with i < j, each edge once
    edges_to_plot = np.argwhere(np.triu(keep, 1))

    grid_points_in_image = scale_grid(
        image=image, grid_points=grid_points, facets=facets, centers=centers, center_facet=center_facet, center_point=center_point, grid_spacing=grid_spacing, plot=plot)

    matches = []
    matched_facets = set()

    # Map each facet index to the list of grid point indices it contains
    facet_to_grid_indices = {i: [] for i in range(len(facets))}
    for idx, cp in enumerate(grid_points_in_image):
        for i, facet in enumerate(facets):
            if point_in_polygon(cp, facet):
                facet_to_grid_indices[i].append(idx)

    # Only allow facets that contain exactly one grid point
    for i, grid_indices in facet_to_grid_indices.items():
        if len(grid_indices) == 1:
            idx = grid_indices[0]
            matches.append(np.hstack([grid_points[idx], centers[i]]))
            matched_facets.add(i)

    if plot == 'Debug':
        visualize_detected_points(raw_image, image_points)
        visualize_center(raw_image, facets, center_facet, center_point)
        visualize_grid_points(raw_image, grid_points_in_image)
        visualize_connections(raw_image, centers, edges_to_plot)
        visualize_voroni(image, facets, centers, image_points)
        display_matched_points(raw_image, matches)
        # visualize_matched_facets(matches) - TODO: might want to fix the visualization for this

    if plot == 'Normal':
        display_matched_points(raw_image, matches, output_path=output_path)

    print(f"Matched {len(matches)} calibration points to facets.")
    return matches
