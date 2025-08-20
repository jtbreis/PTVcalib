import numpy as np
import matplotlib.pyplot as plt

from tools.visualization import *

def filter(img):
    # Compute 2D Fourier Transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # Shift zero frequency to center
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    plot_fft_spectrum(img, magnitude_spectrum)
    mask = create_mask(img, radius=30)
    fshift_filtered = fshift * mask

    # Inverse FFT to get filtered image
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    plot_enhanced_comparison(img, img_filtered)
    return img_filtered


def create_mask(img, radius):
    # Create a mask to filter frequencies
    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2  # center

    # Example: high-pass filter (removes smooth background, keeps sharp patterns)
    mask = np.ones((rows, cols), np.uint8)
    r = int(rows*0.03)  # radius of low frequencies to block
    print(r)
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    return mask