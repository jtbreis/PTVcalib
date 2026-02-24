import cv2
import numpy as np

from ..visualization.image_processing import plot_fft_spectrum, plot_enhanced_comparison


def fft_filter(img, diameterDot, contrast='equalizeHist', plotting='None', denoise_method='nlmeans',
               lowcut_sigma_fraction=0.04, highcut_sigma_fraction=None):
    """
    Apply FFT-based band-pass filtering to keep grid points and discard reflections.

    Removes low spatial frequencies (reflections, glare, smooth background, illumination
    gradients) and optionally very high frequencies (noise). Mid frequencies that contain
    the grid dots are preserved. The removed low-frequency content is replaced by a
    constant (mean level) so the result has even background and clear dots.

    denoise_method: 'nlmeans' (slower, default), 'bilateral' (faster), or 'none'.
    lowcut_sigma_fraction: cutoff for removing low frequencies, as fraction of image size (default 0.04).
      Larger = more aggressive removal of reflections/smooth areas.
    highcut_sigma_fraction: if set, removes frequencies above this (fraction of image size); None = keep all high freq.
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

    base = np.copy(img)
    rows, cols = img.shape
    mean_level = np.mean(base)

    # Band-pass mask: remove low freq (reflections) and optionally very high freq (noise)
    mask = _bandpass_mask(rows, cols, lowcut_sigma_fraction, highcut_sigma_fraction)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    fshift_bp = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_bp)
    img_bp = np.real(np.fft.ifft2(f_ishift))
    # Band-pass output is roughly zero-mean; restore mean level so grid dots stay bright
    img_filtered = img_bp + mean_level
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


def _bandpass_mask(rows, cols, lowcut_sigma_fraction=0.04, highcut_sigma_fraction=None):
    """
    Smooth band-pass in frequency domain: attenuate very low (reflections) and optionally very high (noise).
    Returns a complex mask (real-valued in practice) for multiplying with fftshifted FFT.
    """
    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows, dtype=np.float64) - crow
    x = np.arange(cols, dtype=np.float64) - ccol
    xx, yy = np.meshgrid(x, y)
    r2 = xx * xx + yy * yy
    r = np.sqrt(r2)

    # Low cutoff: remove frequencies below this (Gaussian high-pass part)
    sigma_low = lowcut_sigma_fraction * max(rows, cols)
    lowpass = np.exp(-r2 / (2 * sigma_low * sigma_low))
    mask = 1.0 - lowpass

    # High cutoff: optionally remove frequencies above this (Gaussian roll-off)
    if highcut_sigma_fraction is not None:
        sigma_high = highcut_sigma_fraction * max(rows, cols)
        highpass_rolloff = np.exp(-r2 / (2 * sigma_high * sigma_high))
        mask = mask * highpass_rolloff

    return mask.astype(np.complex128)


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
