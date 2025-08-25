import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.tools.visualization import *

# TODO make this a class so that I don't have to pass as many parameters
def match_calibration_grid(image, image_points, grid_points, grid_spacing, diameterDot, center_find='Simple', plot=False):
    """
    image: input image make sure it is grayscale
    image_points: numpy array of detected points
    grid_points: numpy array with grid points
    matches numpy array (Nx5) [X, Y, Z, x, y]
    """
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

    # Step2: Find center of all facets
    if center_find == 'Simple':
        center_facet, center_point = find_center_facet(facets, centers)
    elif center_find == 'TSI-backlight':
        center_facet, center_point = detect_center(facets, centers)
    else:
        raise ValueError(f"Unknown center_find method: {center_find}")

    grid_points_in_image = scale_grid(image, grid_points, center_facet, center_point, grid_spacing, plot)

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

    # for idx, cp in enumerate(grid_points_in_image):
    #     for i, facet in enumerate(facets):
    #         if i in matched_facets:
    #             matched_facets.add(i)
    #             continue  # Skip facets already matched
    #         if point_in_polygon(cp, facet):
    #             matches.append((grid_points[idx], centers[i]))
    #             matched_facets.add(i)
    #             break  # Each calibration point matched to at most one facet

    if plot is True:
        visualize_center(raw_image, facets, center_facet, center_point)
        visualize_grid_points(raw_image, grid_points_in_image)
        visualize_voroni(image, facets, centers, image_points)
        # visualize_matched_facets(matches) - TODO: might want to fix the visualization for this
        display_matched_points(raw_image, matches)

    print(f"Matched {len(matches)} calibration points to facets.")
    return matches

def merge_close_vertices(facet, threshold):
    merged = []
    for v in facet:
        if not merged:
            merged.append(v)
        else:
            dists = [np.linalg.norm(np.array(v) - np.array(m)) for m in merged]
            if all(dist > threshold for dist in dists):
                merged.append(v)
    return np.array(merged)

def point_in_polygon(point, polygon):
    # point: (x, y), polygon: Nx2 array
    return cv2.pointPolygonTest(np.array(polygon, np.int32), tuple(point), False) >= 0

def scale_grid(image, grid_points, center_facet, center_point, grid_spacing, plot):
    # TODO: account for rotation and different spacing inside the grid
    grid_points_in_image = np.copy(grid_points)[:, :2]
    grid_points_in_image[:, 1] *= -1 # flip bc of image coordinate systems

    facetW, facetH = get_facet_width_height(center_facet)
    # Scale x-Axis
    grid_points_in_image[:, 0] *= facetW / grid_spacing
    grid_points_in_image[:, 0] += center_point[0]
    # Scale y-Axis
    grid_points_in_image[:, 1] *= facetH / grid_spacing
    grid_points_in_image[:, 1] += center_point[1]

    return grid_points_in_image

def get_facet_width_height(facet):
    xs = facet[:, 0]
    ys = facet[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    return width, height

def is_almost_square(facet, tolerance=0.1):
    # facet: Nx2 array of points
    # tolerance: allowed relative difference between width and height
    xs = facet[:, 0]
    ys = facet[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    if width == 0 or height == 0:
        return False
    ratio = min(width, height) / max(width, height)
    return ratio >= (1 - tolerance)

def find_center_facet(facets, centers):
    # Find the facet whose center is closest to the mean of detected centers
    if len(centers) == 0:
        return None
    centers = np.array(centers)
    mean_center = np.mean(centers, axis=0)
    dists = np.linalg.norm(centers - mean_center, axis=1)
    idx = np.argmin(dists)
    return facets[idx], centers[idx]

def detect_center(facets, centers):
    # TODO: this function is super specific to this calibration target, find a more general method.
    for i, facet in enumerate(facets):
        if len(facet) != 4:
            continue
        # Find neighbors: facets whose centers are closest to this facet's center
        center = centers[i]
        dists = np.linalg.norm(np.array(centers) - center, axis=1)
        # Exclude self
        neighbor_indices = np.argsort(dists)[1:5]  # 4 closest neighbors
        neighbor_vertex_counts = [len(facets[j]) for j in neighbor_indices]
        if neighbor_vertex_counts.count(6) == 2 and neighbor_vertex_counts.count(5) == 2:
            return facet, center