import numpy as np


def group_matches_by_planes(XYZ, xy):
    Z_values = [plane[2] for plane in XYZ]
    unique_Z = np.unique(Z_values)
    n_planes = len(unique_Z)
    # Group XYZ and xy by unique Z values
    XYZ_grouped = []
    xy_grouped = []
    for Z in unique_Z:
        indices = [i for i, plane in enumerate(
            XYZ) if np.isclose(plane[2], Z)]
        XYZ_grouped.append(np.vstack([XYZ[i] for i in indices]))
        xy_grouped.append(np.vstack([xy[i] for i in indices]))

    return n_planes, XYZ_grouped, xy_grouped
