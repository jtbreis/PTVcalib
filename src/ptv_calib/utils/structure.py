from enum import Enum
import os


class Folders(Enum):
    # Can be renamed in the future
    GENERAL = "/Calibration"
    MATCHES = "/Calibration/Matches"
    ANNOTATIONS = "/Calibration/Matches/Annotations"
    TESTS = "/Calibration/Tests"


class Filenames(Enum):
    CALIBRATION = "/calib.h5"
    MATCHES = "/matches.h5"


def create_folder_structure(output_path: str):
    for folder in Folders:
        if os.path.isdir(output_path + folder.value) == False:
            os.mkdir(output_path + folder.value)
