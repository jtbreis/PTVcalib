import numpy as np


def scale_grid(image, grid_points, center_facet, center_point, grid_spacing, plot):
    # TODO: account for rotation and different spacing inside the grid
    grid_points_in_image = np.copy(grid_points)[:, :2]
    grid_points_in_image[:, 1] *= -1  # flip bc of image coordinate systems

    facetW, facetH = get_facet_width_height(center_facet)
    # Scale x-Axis
    grid_points_in_image[:, 0] *= facetW / grid_spacing
    grid_points_in_image[:, 0] += center_point[0]
    # Scale y-Axis
    grid_points_in_image[:, 1] *= facetH / grid_spacing
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


def get_facet_width_height(facet):
    xs = facet[:, 0]
    ys = facet[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    return width, height
