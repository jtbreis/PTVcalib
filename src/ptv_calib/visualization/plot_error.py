import numpy as np
import matplotlib.pyplot as plt


def plot_2d_error(camera_2d_error):
    num_cameras = len(camera_2d_error)
    num_layers = len(camera_2d_error[0])
    fig, axes = plt.subplots(num_layers, num_cameras, figsize=(
        4*num_cameras, 3*num_layers), squeeze=True)

    for layer_idx in range(num_layers):
        for cam_idx in range(num_cameras):
            ax = axes[layer_idx][cam_idx]
            error = camera_2d_error[cam_idx][layer_idx]['error']
            xy = camera_2d_error[cam_idx][layer_idx]['xy']
            u = error[:, 0]
            v = error[:, 1]
            std_dev = np.linalg.norm(error, axis=1)
            contour = ax.tricontourf(
                xy[:, 0], xy[:, 1], std_dev, levels=30, cmap='coolwarm', alpha=0.5)
            ax.quiver(xy[:, 0], xy[:, 1], u, v,
                      color='black', width=0.005)
            fig.colorbar(contour, ax=ax, orientation='vertical',
                         label='Deviation')
            ax.set_title(f'Layer {layer_idx+1}, Camera {cam_idx+1}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_2d_mean_error(camera_2d_error):
    num_cameras = len(camera_2d_error)
    num_layers = len(camera_2d_error[0])
    fig, axes = plt.subplots(1, num_cameras, figsize=(
        4*num_cameras, 4), squeeze=True)

    for cam_idx in range(num_cameras):
        ax = axes[cam_idx] if num_cameras > 1 else axes
        mean_errors = []
        for layer_idx in range(num_layers):
            mean = camera_2d_error[cam_idx][layer_idx]['mean']
            mean_errors.append(mean)
        ax.plot(range(1, num_layers+1), mean_errors, marker='o')
        ax.set_title(f'Camera {cam_idx+1}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean 2D Error')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
