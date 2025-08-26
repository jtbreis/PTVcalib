import cv2
import numpy as np
import pandas as pd

from .grid_manipulation import scale_grid, merge_close_vertices
from .grid_checks import is_almost_square, point_in_polygon
from .find_target_center import find_center

from ..visualization.debug_plots import visualize_center, visualize_grid_points, visualize_voroni
from ..visualization.plotting import display_matched_points


def perform_matching(image_path: str, image_points, grid_points, grid_spacing, diameterDot, center_method='Simple', plot='None'):
    """
    image: input image make sure it is grayscale
    image_points: numpy array of detected points
    grid_points: numpy array with grid points
    matches numpy array (Nx5) [X, Y, Z, x, y]
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    raw_image = image.copy()

    # Step1: Subdiv for Voronoi
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in image_points:
        subdiv.insert(p)

    # Get Voronoi facets
    facets, centers = subdiv.getVoronoiFacetList([])

    centers = np.array(centers)

    # TODO: filter for correct and wrong facets
    # square_facets = tuple(f for f in facets if is_almost_square(f))
    facets = [merge_close_vertices(np.array(f), diameterDot) for f in facets]

    center_facet, center_point = find_center(
        center_method=center_method, facets=facets, centers=centers)

    grid_points_in_image = scale_grid(
        image, grid_points, center_facet, center_point, grid_spacing, plot)

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
        visualize_center(raw_image, facets, center_facet, center_point)
        visualize_grid_points(raw_image, grid_points_in_image)
        visualize_voroni(image, facets, centers, image_points)
        display_matched_points(raw_image, matches)
        # visualize_matched_facets(matches) - TODO: might want to fix the visualization for this

    if plot == 'Normal':
        display_matched_points(raw_image, matches)

    print(f"Matched {len(matches)} calibration points to facets.")
    return matches
