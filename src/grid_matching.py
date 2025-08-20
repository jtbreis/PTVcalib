import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tools.visualization import *

def match_calibration_grid(image, image_points, grid_points, grid_spacing):
    """
    image: input image
    image_points: numpy array of detected points
    grid_points: numpy array with grid points
    """
    h, w, _ = image.shape
    raw_image = image.copy()

    # Step1: Subdiv for Voronoi
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in image_points:
        subdiv.insert(p)

    # Get Voronoi facets
    facets, centers = subdiv.getVoronoiFacetList([])

    # TODO: filter for correct and wrong facets
    # square_facets = tuple(f for f in facets if is_almost_square(f))

    # Step2: Find center of all facets
    center_facet, center_point = find_center_facet(facets, centers)
    visualize_center(image, facets, center_facet, center_point)

    visualize_voroni(image, facets, image_points)
    grid_points_in_image = scale_grid(image, grid_points, center_facet, center_point, grid_spacing)

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
            matches.append((grid_points[idx], centers[i]))
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

    print(matched_facets)
    visualize_voroni(raw_image, facets, image_points)
    print(f"Matched {len(matches)} calibration points to facets.")
    visualize_matched_facets(matches)
    return matches

def point_in_polygon(point, polygon):
    # point: (x, y), polygon: Nx2 array
    return cv2.pointPolygonTest(np.array(polygon, np.int32), tuple(point), False) >= 0

def scale_grid(image, grid_points, center_facet, center_point, grid_spacing):
    grid_points_in_image = np.copy(grid_points)[:, :2]
    grid_points_in_image[:, 1] *= -1 # flip bc of image coordinate systems

    facetW, facetH = get_facet_width_height(center_facet)
    # Scale x-Axis
    grid_points_in_image[:, 0] *= facetW / grid_spacing
    grid_points_in_image[:, 0] += center_point[0]
    # Scale y-Axis
    grid_points_in_image[:, 1] *= facetH / grid_spacing
    grid_points_in_image[:, 1] += center_point[1]

    visualize_grid_points(image, grid_points_in_image)

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