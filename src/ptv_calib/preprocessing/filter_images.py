import cv2
import numpy as np

from ..visualization.image_processing import plot_fft_spectrum, plot_enhanced_comparison


def fft_filter(img, diameterDot, contrast='equalizeHist', plotting='None', denoise_method='nlmeans'):
    """Apply FFT-based filtering. denoise_method: 'nlmeans' (slower, default) or 'bilateral' (faster)."""
    if denoise_method == 'nlmeans':
        img = cv2.fastNlMeansDenoising(img, None, int(diameterDot/2))
    elif denoise_method == 'bilateral':
        d = max(1, int(diameterDot / 2))
        img = cv2.bilateralFilter(img, d=d, sigmaColor=50, sigmaSpace=50)
    elif denoise_method != 'none':
        raise ValueError(
            f"denoise_method must be 'nlmeans', 'bilateral', or 'none', got {denoise_method!r}")

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

    if plotting == 'Debug':
        plot_fft_spectrum(img, magnitude_spectrum)
        plot_enhanced_comparison(img, img_filtered)

    return img_filtered


def create_mask(img, radius=None):
    # Create a mask to filter frequencies
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2  # center

    # Example: high-pass filter (removes smooth background, keeps sharp patterns)
    mask = np.ones((rows, cols), np.uint8)
    if radius is None:
        r = int(rows*0.03)  # radius of low frequencies to block
    else:
        r = radius
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    return mask
