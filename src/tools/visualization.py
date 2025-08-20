import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def read_and_visualize_tiff(folder_path, file_name=None):
    """
    Read TIFF files from a folder and create comprehensive visualizations
    
    Parameters:
    folder_path (str): Path to the folder containing TIFF files
    file_name (str, optional): Specific TIFF file name. If None, processes first TIFF found
    """
    
    # Convert to Path object for easier handling
    folder = Path(folder_path)
    
    # Find TIFF files in the folder
    tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    tiff_files = []
    for ext in tiff_extensions:
        tiff_files.extend(folder.glob(f'*{ext}'))
    
    if not tiff_files:
        print(f"No TIFF files found in {folder_path}")
        return
    
    # Select file to process
    if file_name:
        target_file = folder / file_name
        if target_file not in tiff_files:
            print(f"File {file_name} not found in {folder_path}")
            return
    else:
        target_file = tiff_files[0]
        print(f"Processing: {target_file.name}")
    
    try:
        # Read the TIFF file
        img = Image.open(target_file)
        img_array = np.array(img)
        
        # Get image information
        print(f"Image shape: {img_array.shape}")
        print(f"Data type: {img_array.dtype}")
        print(f"Min pixel value: {np.min(img_array)}")
        print(f"Max pixel value: {np.max(img_array)}")
        print(f"Mean pixel value: {np.mean(img_array):.2f}")
        
        # Create comprehensive visualization
        create_comprehensive_plot(img_array, target_file.name)
        
        return img_array
        
    except Exception as e:
        print(f"Error reading {target_file}: {str(e)}")
        return None

def create_comprehensive_plot(img_array, filename):
    """
    Create a comprehensive visualization of the TIFF image
    """
    # Determine if image is grayscale or color
    is_color = len(img_array.shape) == 3 and img_array.shape[2] > 1
    
    if is_color:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'TIFF Analysis: {filename}', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Individual color channels
        channel_names = ['Red', 'Green', 'Blue'] if img_array.shape[2] >= 3 else [f'Channel {i}' for i in range(img_array.shape[2])]
        colors = ['Reds', 'Greens', 'Blues'] if img_array.shape[2] >= 3 else ['viridis'] * img_array.shape[2]
        
        for i in range(min(3, img_array.shape[2])):
            if i < 2:  # Use positions [0,1] and [0,2] for channels
                axes[0, i+1].imshow(img_array[:,:,i], cmap=colors[i])
                axes[0, i+1].set_title(f'{channel_names[i]} Channel')
                axes[0, i+1].axis('off')
        
        # Pixel value histogram for all channels
        axes[1, 0].hist(img_array.flatten(), bins=50, alpha=0.7, color='gray', edgecolor='black')
        axes[1, 0].set_title('Overall Pixel Value Distribution')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Individual channel histograms
        for i in range(min(2, img_array.shape[2])):
            channel_data = img_array[:,:,i].flatten()
            axes[1, i+1].hist(channel_data, bins=50, alpha=0.7, 
                            color=['red', 'green'][i], edgecolor='black')
            axes[1, i+1].set_title(f'{channel_names[i]} Channel Distribution')
            axes[1, i+1].set_xlabel('Pixel Value')
            axes[1, i+1].set_ylabel('Frequency')
            axes[1, i+1].grid(True, alpha=0.3)
        
    else:
        # Grayscale image visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'TIFF Analysis: {filename}', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(img_array, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image with colormap
        axes[0, 1].imshow(img_array, cmap='viridis')
        axes[0, 1].set_title('Viridis Colormap')
        axes[0, 1].axis('off')
        
        # Image with hot colormap
        axes[0, 2].imshow(img_array, cmap='hot')
        axes[0, 2].set_title('Hot Colormap')
        axes[0, 2].axis('off')
        
        # Pixel value histogram
        axes[1, 0].hist(img_array.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Pixel Value Distribution')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2D histogram (heatmap of pixel positions vs values)
        h, w = img_array.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        axes[1, 1].scatter(x_coords.flatten()[::100], img_array.flatten()[::100], 
                          alpha=0.5, s=1, c=img_array.flatten()[::100], cmap='viridis')
        axes[1, 1].set_title('Pixel Values vs X Position (sampled)')
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Pixel Value')
        
        # Box plot of pixel values by image regions
        # Divide image into 4 quadrants
        h_mid, w_mid = h//2, w//2
        q1 = img_array[:h_mid, :w_mid].flatten()
        q2 = img_array[:h_mid, w_mid:].flatten()
        q3 = img_array[h_mid:, :w_mid].flatten()
        q4 = img_array[h_mid:, w_mid:].flatten()
        
        axes[1, 2].boxplot([q1, q2, q3, q4], labels=['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1, 2].set_title('Pixel Value Distribution by Quadrant')
        axes[1, 2].set_ylabel('Pixel Value')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def display_image_with_histogram(image):
    plt.figure(figsize=(12, 6))

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='inferno')
    plt.title('Image')
    plt.axis('off')

    # Display the histogram
    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def visualize_voroni(image, facets, points):
    for facet in facets:
        pts = np.array(facet, np.int32)
        cv2.fillConvexPoly(image, pts, (np.random.randint(256), np.random.randint(256), np.random.randint(256)))
        cv2.polylines(image, [pts], True, (0, 0, 0), 1)

    # Draw points
    for p in points:
        center = (int(p[0]), int(p[1]))
        cv2.circle(image, center, 4, (0, 0, 255), -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def visualize_center(image, facets, center, center_point):
    # Draw all facets faintly
    img_highlight = image.copy()
    for facet in facets:
        pts = np.array(facet, np.int32)
        cv2.fillConvexPoly(img_highlight, pts, (200, 200, 200))

    # Highlight the center facet in red
    pts_center = np.array(center, np.int32)
    cv2.fillConvexPoly(img_highlight, pts_center, (255, 0, 0))
    cv2.polylines(img_highlight, [pts_center], True, (0, 0, 0), 2)

    # Draw the center point as a green dot
    center_coords = (int(center_point[0]), int(center_point[1]))
    cv2.circle(img_highlight, center_coords, 6, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(img_highlight, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def visualize_grid_points(image, grid_points):
    # plt.scatter(center_facet_centroid[0], center_facet_centroid[1], color='red', s=120, marker='x', label='Center Facet Centroid')
    plt.scatter(grid_points[:, 0], grid_points[:, 1])
    plt.legend()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def visualize_matched_facets(matches):
    plt.figure(figsize=(8, 8))
    for grid_pt, facet_center in matches:
        plt.plot([grid_pt[0]*30, facet_center[0]], [grid_pt[1]*30, -facet_center[1]], 'g--', linewidth=1)
        plt.scatter(grid_pt[0]*30, grid_pt[1]*30, color='blue', s=60, label='Grid Point' if 'Grid Point' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(facet_center[0], -facet_center[1], color='red', s=60, label='Facet Center' if 'Facet Center' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.legend()
    plt.title("Matched Grid Points and Facet Centers")
    plt.axis("equal")
    plt.show()