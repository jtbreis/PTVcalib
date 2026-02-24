import cv2
import numpy as np

from ..visualization.image_processing import plot_fft_spectrum, plot_enhanced_comparison


def fft_filter(img, diameterDot, contrast='equalizeHist', plotting='None', denoise_method='nlmeans',
               highpass_sigma_fraction=0.03, highpass_strength=0.4):
    """
    Apply FFT-based filtering that preserves sharp dot centers and avoids dark-blob artifacts.

    Uses a smooth Gaussian high-pass (no hard mask) and adds the high-pass component to the
    original image instead of replacing it, so DC and low frequencies (bright centers) are
    preserved and only background non-uniformity is reduced.

    denoise_method: 'nlmeans' (slower, default), 'bilateral' (faster), or 'none'.
    highpass_sigma_fraction: cutoff scale for Gaussian high-pass, as fraction of image size (default 0.03).
    highpass_strength: weight of high-pass added to image (default 0.4); higher = more background flattening.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    if denoise_method == 'nlmeans':
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_u8 = cv2.fastNlMeansDenoising(img_u8, None, int(diameterDot / 2))
        img = img_u8.astype(np.float64)
    elif denoise_method == 'bilateral':
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d = max(1, int(diameterDot / 2))
        img_u8 = cv2.bilateralFilter(img_u8, d=d, sigmaColor=50, sigmaSpace=50)
        img = img_u8.astype(np.float64)
    elif denoise_method != 'none':
        raise ValueError(
            f"denoise_method must be 'nlmeans', 'bilateral', or 'none', got {denoise_method!r}")

    if contrast == 'equalizeHist':
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_u8 = cv2.equalizeHist(img_u8)
        img = img_u8.astype(np.float64)

    # Original (preserves sharp centers); we'll add a scaled high-pass to this
    base = np.copy(img)

    # Smooth Gaussian high-pass mask to avoid ringing (no hard edges in freq domain)
    rows, cols = img.shape
    mask_hp = _gaussian_highpass_mask(rows, cols, highpass_sigma_fraction)

    # FFT: get high-pass component only (zero-mean detail)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    fshift_hp = fshift * mask_hp
    f_ishift = np.fft.ifftshift(fshift_hp)
    highpass = np.real(np.fft.ifft2(f_ishift))
    # Zero-mean so we don't shift global intensity
    highpass = highpass - np.mean(highpass)
    # Scale so typical values are on the order of image range
    scale = np.std(base) / (np.std(highpass) + 1e-8)
    highpass = highpass * scale

    # Add high-pass to base: preserves bright centers, flattens slow background
    img_filtered = base + highpass_strength * highpass
    img_filtered = np.clip(img_filtered, 0, 255)

    if plotting == 'Debug':
        plot_fft_spectrum(
            cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            magnitude_spectrum,
        )
        plot_enhanced_comparison(
            cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            img_filtered.astype(np.float64),
        )

    return img_filtered


def _gaussian_highpass_mask(rows, cols, sigma_fraction=0.03):
    """Smooth high-pass mask: 1 - Gaussian low-pass. Reduces ringing vs binary mask."""
    crow, ccol = rows // 2, cols // 2
    sigma = sigma_fraction * max(rows, cols)
    y = np.arange(rows, dtype=np.float64) - crow
    x = np.arange(cols, dtype=np.float64) - ccol
    xx, yy = np.meshgrid(x, y)
    r2 = xx * xx + yy * yy
    lowpass = np.exp(-r2 / (2 * sigma * sigma))
    highpass = 1.0 - lowpass
    return highpass.astype(np.complex128)


def create_mask(img, radius=None):
    """Legacy binary high-pass mask (can cause ringing). Prefer _gaussian_highpass_mask."""
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    if radius is None:
        r = int(rows * 0.03)
    else:
        r = radius
    mask[crow - r : crow + r, ccol - r : ccol + r] = 0
    return mask
