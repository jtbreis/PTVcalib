from enum import Enum
import os


class Folders(Enum):
    # Can be renamed in the future
    MATCHES = "/Calibration/Matches"
    ANNOTATIONS = "/Calibration/Matches/Annotations"
    TESTS = "/Calibration/Tests"


def create_folder_structure(output_path: str):
    for folder in Folders:
        if os.path.isdir(output_path + folder.value) == False:
            os.mkdir(output_path + folder.value)
