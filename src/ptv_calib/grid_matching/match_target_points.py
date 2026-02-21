import cv2
import freud
import numpy as np
import pandas as pd

from .grid_manipulation import scale_grid, merge_close_vertices
from .grid_checks import is_almost_square, point_in_polygon
from .find_target_center import find_center
from scipy.spatial import ConvexHull

from ..visualization.debug_plots import visualize_center, visualize_grid_points, visualize_voroni, visualize_detected_points
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

    # TODO: something like this could be used to capture more calibration points in the future
    # # Compute the convex hull
    # hull = ConvexHull(image_points)
    # img_pts = np.vstack(image_points)
    # # Extract the outer layer points (vertices of the hull)
    # outer_points_indices = np.array(hull.vertices)
    # outer_points = img_pts[outer_points_indices]

    # # Move outer points 20 pixels further out from the center of the hull
    # hull_center = np.mean(outer_points, axis=0)
    # direction_vectors = outer_points - hull_center
    # norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    # norms[norms == 0] = 1  # Prevent division by zero
    # unit_vectors = direction_vectors / norms
    # moved_outer_points = outer_points + unit_vectors * 5
    # # Ensure moved_outer_points stay within image bounds (1 pixel away from edge)
    # moved_outer_points[:, 0] = np.clip(moved_outer_points[:, 0], 1, w - 2)
    # moved_outer_points[:, 1] = np.clip(moved_outer_points[:, 1], 1, h - 2)

    # # Optionally, you can append these moved points to image_points if needed
    # # image_points = np.vstack([image_points, moved_outer_points])

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

    # TODO: filter for correct and wrong facets
    # square_facets = tuple(f for f in facets if is_almost_square(f))
    facets = [merge_close_vertices(np.array(f), diameterDot) for f in facets]

    center_facet, center_point = find_center(
        center_method=center_method, facets=facets, centers=centers)

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
        visualize_voroni(image, facets, centers, image_points)
        display_matched_points(raw_image, matches)
        # visualize_matched_facets(matches) - TODO: might want to fix the visualization for this

    if plot == 'Normal':
        display_matched_points(raw_image, matches, output_path=output_path)

    print(f"Matched {len(matches)} calibration points to facets.")
    return matches
