import numpy as np


def scale_grid(image, grid_points, center_facet, center_point, grid_spacing, plot):
    # TODO: account for rotation and different spacing inside the grid
    grid_points_in_image = np.copy(grid_points)[:, :2]
    grid_points_in_image[:, 1] *= -1  # flip bc of image coordinate systems

    facetW, facetH, alpha = get_facet_width_height(center_facet, center_point)

    # Scale grid points
    grid_points_in_image[:, 0] *= facetW / grid_spacing
    grid_points_in_image[:, 1] *= facetH / grid_spacing

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


def get_facet_width_height(facet, center):
    xs = facet[:, 0]
    ys = facet[:, 1]
    cx = xs - center[0]
    cy = ys - center[1]
    angles = np.atan2(cx, cy) - np.pi/4
    alpha = np.min(np.abs(angles))

    AB = np.array([xs[0] - xs[1], ys[0] - ys[1]])
    BC = np.array([xs[1] - xs[2], ys[1] - ys[2]])
    CD = np.array([xs[2] - xs[3], ys[2] - ys[3]])
    DA = np.array([xs[3] - xs[0], ys[3] - ys[0]])
    width = (np.linalg.norm(BC) + np.linalg.norm(DA)) / 2
    height = (np.linalg.norm(AB) + np.linalg.norm(CD)) / 2

    return width, height, alpha
