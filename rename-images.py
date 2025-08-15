import os
import sys

def renumber_images(folder):
    # Get list of files, filter for images (common extensions)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp')
    print(os.listdir(folder))
    files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    files.sort()  # Sort for consistent ordering

    for idx, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        prefix = filename.split('_')[0] + '_'
        new_name = prefix + f"{idx}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f"Renamed {filename} -> {new_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 rename-images.py <folder>")
        exit(1)
    renumber_images(sys.argv[1])