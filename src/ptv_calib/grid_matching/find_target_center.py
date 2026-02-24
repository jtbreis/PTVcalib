import numpy as np


def find_center(center_method: str, facets, centers):
    # Step2: Find center of all facets
    if center_method == 'Mean':
        center_facet, center_point = mean_center(facets, centers)
    elif center_method == 'TSI-backlight':
        result = tsi_backlight_center(facets, centers)
        if result is None:
            center_facet, center_point = mean_center(facets, centers)
        else:
            center_facet, center_point = result
    else:
        raise ValueError(f"Unknown center_method method: {center_method}")

    if center_facet is None:
        raise RuntimeError("Couldn't find a center for the target!")

    return center_facet, center_point

# TODO: find a more general method to find the center of a calibration target


def mean_center(facets, centers):
    if len(centers) == 0:
        return None
    centers = np.array(centers)
    mean_center = np.mean(centers, axis=0)
    dists = np.linalg.norm(centers - mean_center, axis=1)
    idx = np.argmin(dists)
    return facets[idx], centers[idx]


def tsi_backlight_center(facets, centers):
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
    return None
