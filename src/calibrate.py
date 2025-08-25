import numpy as np
import os
import cv2
import pandas as pd

from src.fft_grid_filtering import filter
from src.target_point_detection import detect_target_points
from src.grid_matching import match_calibration_grid

from src.calibration_methods.apply_calibration_method import perform_soloff

def process_moving_target(folder_path, calibration_target, grid_spacing, zmin, zmax, zstep, diameterDot, centerFindMethod='Simple', contrast='equalizeHist', plot=False):
    calibration_points = pd.read_csv(f'calibration_targets/{calibration_target}.csv').to_numpy()
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    image_files.sort()
    images = [cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE) for fname in image_files]

    zpositions = np.arange(start=zmin, stop=zmax+zstep, step=zstep)

    all_plane_matches = []

    for idx, z in enumerate(zpositions):
        print(f'Processing image {image_files[idx]}')
        filtered_image = filter(img=images[idx], diameterDot=diameterDot, contrast=contrast)
        image_points = detect_target_points(filtered_image, diameterDot=diameterDot)
        print(f'Detected: {len(image_points)} points in the image')
        calibration_points[:,2] = z
        matches = match_calibration_grid(images[idx], image_points=image_points, grid_points=calibration_points, grid_spacing=grid_spacing, diameterDot=diameterDot, center_find=centerFindMethod, plot=plot)
        all_plane_matches.append(matches)

    return np.vstack(all_plane_matches)

def perform_calibration(matches, method='Soloff'):
    XYZ = matches[:, :3]
    xy = matches[:,3:]
    print(XYZ.shape)
    print(xy.shape)

    [sx, sy] = perform_soloff(xy, XYZ)

    return sx, sy

    


