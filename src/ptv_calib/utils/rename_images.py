import os


def rename_images(folder_path):
    # Get list of files and filter for image files (common extensions)
    image_extensions = ('.jpg', '.jpeg', '.png',
                        '.bmp', '.tiff', '.gif', '.tif')
    files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(image_extensions)]
    files.sort()  # Sort to keep order

    for idx, filename in enumerate(files):
        name, ext = os.path.splitext(filename)
        # Remove spaces and ensure format 'Camera1_XXXX'
        if name.startswith("Camera "):
            cam_part, rest = name.split(" ", 1)
            cam_number, suffix = rest.split("_", 1)
            new_name = f"Camera{cam_number}_{idx:04d}{ext}"
        else:
            new_name = f"{name}_{idx:04d}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")
