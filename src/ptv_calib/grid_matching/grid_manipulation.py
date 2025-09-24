import numpy as np


def scale_grid(image, grid_points, facets, centers, center_facet, center_point, grid_spacing, plot):
    # TODO: account for rotation and different spacing inside the grid
    grid_points_in_image = np.copy(grid_points)[:, :2]
    grid_points_in_image[:, 1] *= -1  # flip bc of image coordinate systems

    facetW, facetH = get_facet_width_height(center_facet, center_point)

    # Scale grid points
    grid_points_in_image[:, 0] *= facetW / grid_spacing
    grid_points_in_image[:, 1] *= facetH / grid_spacing

    alpha = get_rotation_angle(
        facets=facets, centers=centers, center_facet=center_facet, threshold=facetW)

    # Build rotation matrix for angle alpha
    rotation_matrix = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha),  np.cos(alpha)]
    ])

    # Rotate grid points
    grid_points_in_image = np.dot(grid_points_in_image, rotation_matrix.T)

    # Translate to center_point
    grid_points_in_image[:, 0] += center_point[0]
    grid_points_in_image[:, 1] += center_point[1]

    return grid_points_in_image


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


def get_rotation_angle(facets, centers, center_facet, threshold):
    # Find the indices of the center facet and its center point
    center_idx = None
    for i, facet in enumerate(facets):
        if np.array_equal(facet, center_facet):
            center_idx = i
            break

    if center_idx is not None:
        center_center = centers[center_idx]
        distances = [(i, np.linalg.norm(centers[i] - center_center))
                     for i in range(len(centers)) if i != center_idx]
        # Sort by distance but preserve original indices
        distances_sorted = sorted(distances, key=lambda x: x[1])
        # Find the 4 closest centers (excluding the center itself)
        closest_indices = [i for i, _ in distances_sorted[:4]]
        closest_centers = [centers[i] for i in closest_indices]

        # Calculate vectors from center to each of the closest centers
        vectors = [np.array(center) - np.array(center_center)
                   for center in closest_centers]
        # Calculate angles of these vectors with respect to the x-axis
        angles = [np.arctan2(vec[1], vec[0]) for vec in vectors]
    else:
        RuntimeError("Center doesn't seem to be in facet set!")

    alpha = angles[np.argmin(np.abs(angles))]
    beta = angles[np.argmax(np.abs(angles))]
    beta = beta - np.pi * np.sign(beta)
    return (alpha+beta)/2


def get_facet_width_height(facet, center):
    xs = facet[:, 0]
    ys = facet[:, 1]
    cx = xs - center[0]
    cy = ys - center[1]

    AB = np.array([xs[0] - xs[1], ys[0] - ys[1]])
    BC = np.array([xs[1] - xs[2], ys[1] - ys[2]])
    CD = np.array([xs[2] - xs[3], ys[2] - ys[3]])
    DA = np.array([xs[3] - xs[0], ys[3] - ys[0]])
    width = (np.linalg.norm(BC) + np.linalg.norm(DA)) / 2
    height = (np.linalg.norm(AB) + np.linalg.norm(CD)) / 2

    return width, height
