import numpy as np
import matplotlib.pyplot as plt

from src.tools.visualization import *

def filter(img, diameterDot, contrast='equalizeHist', plot=False):
    img = cv2.fastNlMeansDenoising(img, None, int(diameterDot/2))

    if contrast == 'equalizeHist':
        img = cv2.equalizeHist(img)

    # Compute 2D Fourier Transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # Shift zero frequency to center
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    mask = create_mask(img)
    fshift_filtered = fshift * mask

    # Inverse FFT to get filtered image
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    if plot is True:
        plot_fft_spectrum(img, magnitude_spectrum)
        plot_enhanced_comparison(img, img_filtered)
        
    return img_filtered


def create_mask(img, radius=None):
    # Create a mask to filter frequencies
    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2  # center

    # Example: high-pass filter (removes smooth background, keeps sharp patterns)
    mask = np.ones((rows, cols), np.uint8)
    if radius is None:
        r = int(rows*0.03)  # radius of low frequencies to block
    else:
        r = radius
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    return mask