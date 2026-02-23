import cv2
import matplotlib.pyplot as plt


def display_matched_points(img, matches, output_path=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for match in matches:
        # img_pt: (x, y) in image, real_world_pt: (X, Y) in real world
        [X, Y, Z, x, y] = match
        plt.scatter(x, y, s=120, edgecolors='yellow',
                    facecolors='none', linewidths=2)
        plt.text(x + 5, y - 15, f'{X:.2f}', color='white', fontsize=10,
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))
        plt.text(x + 5, y + 15, f'{Y:.2f}', color='white', fontsize=10,
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))
    plt.title("Matched Points with Real World Coordinates")
    plt.axis("off")
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()
