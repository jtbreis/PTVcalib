import numpy as np

def create_target_points(targetname, npointsX, npointsY, spacing, zOffset):
    x_coords = (np.arange(npointsX) - (npointsX - 1) / 2) * spacing
    y_coords = (np.arange(npointsY) - (npointsY - 1) / 2) * spacing
    xx, yy = np.meshgrid(x_coords, y_coords)
    zz = np.full_like(xx, zOffset, dtype=float)
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    filename = f'{targetname}_nX{npointsX}_nY{npointsY}'
    np.savetxt(f"calibration_targets/{filename}.csv", points, fmt='%.3f',delimiter=",", header="x,y,z", comments="")