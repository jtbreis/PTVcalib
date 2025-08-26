import matplotlib.pyplot as plt


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


def plot_fft_spectrum(img, magnitude_spectrum):
    # Plot original + FFT
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("FFT Magnitude Spectrum")

    plt.show()


def plot_enhanced_comparison(img, filtered_img):
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title("Original")

    plt.subplot(122), plt.imshow(filtered_img, cmap='gray')
    plt.title("Pattern Enhanced")
    plt.show()
