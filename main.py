import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def gaussian_kernel(size, sigma=1.0):
    """Create a 2D Gaussian kernel."""
    x = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return np.outer(g, g)

def nl_means(image, patch_size=7, search_size=21, h=None, sigma=20):
    # h = 10 * sigma
    if h is None:
        h = 10 * sigma

    pad = patch_size // 2
    padded = np.pad(image, pad, mode='reflect').astype(np.float32)
    output = np.zeros_like(image, dtype=np.float32)

    H, W = image.shape
    search_radius = search_size // 2

    # Gaussian kernel for weighted patch comparison
    gaussian_weight = gaussian_kernel(patch_size, sigma=1.0)
    gaussian_weight /= gaussian_weight.sum()  # Normalize

    for i in range(H):
        for j in range(W):
            i1, j1 = i + pad, j + pad
            ref_patch = padded[i1-pad:i1+pad+1, j1-pad:j1+pad+1]

            x_min = max(pad, i1 - search_radius)
            x_max = min(H + pad - 1, i1 + search_radius)
            y_min = max(pad, j1 - search_radius)
            y_max = min(W + pad - 1, j1 + search_radius)

            weights_sum = 0.0
            pixel_sum = 0.0

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    comp_patch = padded[x-pad:x+pad+1, y-pad:y+pad+1]
                    if comp_patch.shape != ref_patch.shape:
                        continue

                    # Weighted Euclidean distance with Gaussian kernel
                    # This implements: ||v(N_i) - v(N_j)||²_{2,a}
                    patch_diff_sq = (ref_patch - comp_patch) ** 2
                    dist = np.sum(gaussian_weight * patch_diff_sq)

                    w = np.exp(-dist / (h * h))

                    weights_sum += w
                    pixel_sum += w * padded[x, y]

            if weights_sum == 0.0:
                output[i, j] = image[i, j]
            else:
                output[i, j] = pixel_sum / weights_sum

    return np.clip(output, 0, 255).astype(np.uint8)


def visualize_comparison(noisy, denoised, name):
    """Display noisy and denoised images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title(f'{name} - Noisy')
    axes[0].axis('off')

    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title(f'{name} - NL-Means Denoised')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{name}_comparison.png", dpi=100, bbox_inches='tight')
    plt.show()


def run_pipeline(image_path, name):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sigma = 20
    noisy = add_gaussian_noise(img, sigma=sigma)
    # settings: patch_size=7, search_size=21, h=10*sigma
    denoised = nl_means(noisy, patch_size=7, search_size=21, sigma=sigma)

    gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.5)

    # Save results
    cv2.imwrite(f"{name}_original.png", img)
    cv2.imwrite(f"{name}_noisy.png", noisy)
    cv2.imwrite(f"{name}_nlmeans.png", denoised)
    cv2.imwrite(f"{name}_gaussian.png", gaussian)

    # Display side-by-side comparison
    visualize_comparison(noisy, denoised, name)

    print(f"{name} done")


if __name__ == "__main__":
    # RUN ALL THREE IMAGES
    run_pipeline("lena.tif", "lena")
    run_pipeline("baboon.png", "baboon")
    run_pipeline("real.png", "real")