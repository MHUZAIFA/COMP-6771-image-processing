"""
COMP 478/6771 Image Processing - Course Project

Non-Local Means Image Denoising
Based on the paper by Buades, Coll & Morel (CVPR 2005)

Mohammed Huzaifa (40242080)
Masoumeh Farokhpour (40309733)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class NLMeansDenoiser:
    """
    NL-means denoiser class.
    Uses 7x7 patches and 21x21 search window by default (from the paper).
    """
    def __init__(self, patch_size=7, search_window=21):
        self.patch_size = patch_size
        self.search_window = search_window
        self.pad_patch = patch_size // 2
        self.pad_search = search_window // 2

    def denoise(self, image, h, verbose=False):
        # make sure image is in 0-255 range
        if image.max() <= 1.0:
            image = image * 255.0

        image = image.astype(np.float64)
        rows, cols = image.shape

        # pad the image so we don't go out of bounds
        total_pad = self.pad_patch + self.pad_search
        padded = np.pad(image, total_pad, mode='reflect')

        # gaussian weights for comparing patches - center pixels matter more
        t = np.arange(self.patch_size) - self.pad_patch
        xx, yy = np.meshgrid(t, t)
        a = self.patch_size / 4
        gauss_weights = np.exp(-(xx**2 + yy**2) / (2 * a**2))
        gauss_weights = gauss_weights / gauss_weights.sum()

        h_sq = h * h

        # these will store our running totals
        weighted_sum = np.zeros((rows, cols))
        weight_sum = np.zeros((rows, cols))
        max_weight = np.zeros((rows, cols))

        total_offsets = (2 * self.pad_search + 1) ** 2
        count = 0

        # go through each offset in the search window
        for di in range(-self.pad_search, self.pad_search + 1):
            for dj in range(-self.pad_search, self.pad_search + 1):
                count += 1
                if verbose and count % 100 == 0:
                    print(f"  Progress: {count}/{total_offsets}")

                # skip the center pixel (we handle it separately)
                if di == 0 and dj == 0:
                    continue

                # calculate patch distance for all pixels at once
                dist_sq = np.zeros((rows, cols))

                for pi in range(-self.pad_patch, self.pad_patch + 1):
                    for pj in range(-self.pad_patch, self.pad_patch + 1):
                        ref = padded[total_pad + pi:total_pad + pi + rows,
                                    total_pad + pj:total_pad + pj + cols]
                        nbr = padded[total_pad + di + pi:total_pad + di + pi + rows,
                                    total_pad + dj + pj:total_pad + dj + pj + cols]

                        gw = gauss_weights[pi + self.pad_patch, pj + self.pad_patch]
                        dist_sq += gw * (ref - nbr) ** 2

                # weight formula from the paper: w = exp(-d^2 / h^2)
                weights = np.exp(-dist_sq / h_sq)

                # keep track of max weight for later
                max_weight = np.maximum(max_weight, weights)

                # add weighted contribution
                neighbors = padded[total_pad + di:total_pad + di + rows,
                                  total_pad + dj:total_pad + dj + cols]
                weighted_sum += weights * neighbors
                weight_sum += weights

        # add self with max weight
        weighted_sum += max_weight * image
        weight_sum += max_weight

        # normalize
        denoised = np.divide(weighted_sum, weight_sum, out=np.copy(image), where=weight_sum > 1e-10)
        return np.clip(denoised, 0, 255)


# --- Comparison methods from the paper ---

def gaussian_filter_denoise(image, sigma):
    """Simple gaussian blur"""
    if image.max() <= 1.0:
        image = image * 255.0
    return gaussian_filter(image.astype(np.float64), sigma=sigma)


def anisotropic_filter(image, iterations=20, kappa=25, gamma=0.1):
    """
    Perona-Malik anisotropic diffusion.
    Smooths flat areas but tries to keep edges.
    """
    if image.max() <= 1.0:
        image = image * 255.0

    img = image.astype(np.float64).copy()

    for _ in range(iterations):
        # gradients in 4 directions
        dN = np.roll(img, -1, axis=0) - img
        dS = np.roll(img, 1, axis=0) - img
        dE = np.roll(img, -1, axis=1) - img
        dW = np.roll(img, 1, axis=1) - img

        # diffusion coefficients - small near edges
        cN = np.exp(-(dN/kappa)**2)
        cS = np.exp(-(dS/kappa)**2)
        cE = np.exp(-(dE/kappa)**2)
        cW = np.exp(-(dW/kappa)**2)

        img += gamma * (cN*dN + cS*dS + cE*dE + cW*dW)

    return np.clip(img, 0, 255)


def tv_denoise(image, weight=0.1, iterations=100):
    """
    Total variation denoising (ROF model).
    Good at removing noise but can make images look "cartoony".
    """
    if image.max() <= 1.0:
        image = image * 255.0

    u = image.astype(np.float64).copy()
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    tau = 0.125

    for _ in range(iterations):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        px_new = px + tau * ux
        py_new = py + tau * uy

        # project onto unit ball
        norm_p = np.sqrt(px_new**2 + py_new**2)
        norm_p = np.maximum(norm_p, 1.0)
        px = px_new / norm_p
        py = py_new / norm_p

        div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = image + weight * div_p

    return np.clip(u, 0, 255)


def neighborhood_filter(image, h, window_size=11):
    """
    Yaroslavsky neighborhood filter.
    Like NL-means but only compares single pixels, not patches.
    """
    if image.max() <= 1.0:
        image = image * 255.0

    image = image.astype(np.float64)
    rows, cols = image.shape
    pad = window_size // 2
    padded = np.pad(image, pad, mode='reflect')

    h_sq = h * h
    denoised = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            center_val = image[i, j]
            window = padded[i:i+window_size, j:j+window_size]
            diff_sq = (window - center_val) ** 2
            weights = np.exp(-diff_sq / h_sq)
            denoised[i, j] = np.sum(weights * window) / np.sum(weights)

    return np.clip(denoised, 0, 255)


# --- Helper functions ---

def add_noise(image, sigma):
    """Add gaussian noise to image"""
    if image.max() <= 1.0:
        image = image * 255.0
    noise = np.random.randn(*image.shape) * sigma
    return np.clip(image + noise, 0, 255)


def compute_mse(original, denoised):
    if original.max() <= 1.0:
        original = original * 255.0
    if denoised.max() <= 1.0:
        denoised = denoised * 255.0
    return np.mean((original - denoised) ** 2)


def compute_psnr(original, denoised):
    mse = compute_mse(original, denoised)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(255**2 / mse)


def compute_ssim(original, denoised):
    if original.max() > 1.0:
        original = original / 255.0
    if denoised.max() > 1.0:
        denoised = denoised / 255.0
    return ssim(original, denoised, data_range=1.0)


def compute_weight_map(image, center, patch_size=7, search_window=21, h=10):
    """Get the weight map for a specific pixel - used for visualization"""
    if image.max() <= 1.0:
        image = image * 255.0
    image = image.astype(np.float64)

    pad_patch = patch_size // 2
    pad_search = search_window // 2
    total_pad = pad_patch + pad_search
    padded = np.pad(image, total_pad, mode='reflect')

    t = np.arange(patch_size) - pad_patch
    xx, yy = np.meshgrid(t, t)
    gauss = np.exp(-(xx**2 + yy**2) / (2 * (patch_size/4)**2))
    gauss = gauss / gauss.sum()

    ci, cj = center[0] + total_pad, center[1] + total_pad
    ref_patch = padded[ci - pad_patch:ci + pad_patch + 1,
                       cj - pad_patch:cj + pad_patch + 1]

    weight_map = np.zeros((search_window, search_window))
    h_sq = h * h

    for di in range(-pad_search, pad_search + 1):
        for dj in range(-pad_search, pad_search + 1):
            ni, nj = ci + di, cj + dj
            comp_patch = padded[ni - pad_patch:ni + pad_patch + 1,
                                nj - pad_patch:nj + pad_patch + 1]
            dist_sq = np.sum(gauss * (ref_patch - comp_patch) ** 2)
            weight_map[di + pad_search, dj + pad_search] = np.exp(-dist_sq / h_sq)

    return weight_map


# --- Main experiment function ---

def run_all_experiments(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("NL-MEANS DENOISING PROJECT")
    print("=" * 60)

    # load image
    print(f"\nLoading: {image_path}")
    image = io.imread(image_path)

    # convert to grayscale if needed
    if len(image.shape) == 3:
        image = color.rgb2gray(image) * 255
    image = image.astype(np.float64)

    # resize if too big
    if image.shape[0] > 512:
        image = resize(image, (512, 512), preserve_range=True)

    print(f"Size: {image.shape}")

    nlm = NLMeansDenoiser(patch_size=7, search_window=21)

    # ========== EXPERIMENT 1: Compare all methods ==========
    print("\n" + "=" * 60)
    print("Experiment 1: Method Comparison")
    print("=" * 60)

    noise_sigma = 20
    noisy = add_noise(image, noise_sigma)
    h_nlm = 15  # found this works well for sigma=20

    print(f"Noise level: sigma = {noise_sigma}")
    print("Running filters...")

    print("  Gaussian...")
    gf_result = gaussian_filter_denoise(noisy, sigma=1.5)

    print("  Anisotropic...")
    af_result = anisotropic_filter(noisy, iterations=25, kappa=noise_sigma)

    print("  Total Variation...")
    tv_result = tv_denoise(noisy, weight=0.12, iterations=150)

    print("  Neighborhood...")
    nf_result = neighborhood_filter(noisy, h=50, window_size=11)

    print("  NL-means...")
    nlm_result = nlm.denoise(noisy, h=h_nlm)

    # print results table
    results = {
        'Gaussian': gf_result,
        'Anisotropic': af_result,
        'Total Variation': tv_result,
        'Neighborhood': nf_result,
        'NL-means': nlm_result
    }

    print("\nResults:")
    print("-" * 50)
    print(f"{'Method':<20} {'MSE':>8} {'PSNR':>10} {'SSIM':>8}")
    print("-" * 50)

    for name, result in results.items():
        mse = compute_mse(image, result)
        psnr_val = compute_psnr(image, result)
        ssim_val = compute_ssim(image, result)
        print(f"{name:<20} {mse:>8.1f} {psnr_val:>10.2f} {ssim_val:>8.4f}")
    print("-" * 50)

    # save figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Noisy (sigma={noise_sigma})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gf_result, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f'Gaussian ({compute_psnr(image, gf_result):.1f} dB)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(af_result, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Anisotropic ({compute_psnr(image, af_result):.1f} dB)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(tv_result, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f'Total Variation ({compute_psnr(image, tv_result):.1f} dB)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(nlm_result, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title(f'NL-means ({compute_psnr(image, nlm_result):.1f} dB)', fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f'Denoising Comparison (sigma={noise_sigma})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_method_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig1_method_comparison.png")

    # ========== EXPERIMENT 2: Method noise ==========
    print("\n" + "=" * 60)
    print("Experiment 2: Method Noise")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    noise_range = 50

    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image - gf_result, cmap='gray', vmin=-noise_range, vmax=noise_range)
    axes[0, 1].set_title('Gaussian noise')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image - af_result, cmap='gray', vmin=-noise_range, vmax=noise_range)
    axes[0, 2].set_title('Anisotropic noise')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(image - tv_result, cmap='gray', vmin=-noise_range, vmax=noise_range)
    axes[1, 0].set_title('TV noise')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(image - nf_result, cmap='gray', vmin=-noise_range, vmax=noise_range)
    axes[1, 1].set_title('Neighborhood noise')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(image - nlm_result, cmap='gray', vmin=-noise_range, vmax=noise_range)
    axes[1, 2].set_title('NL-means noise', fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle('Method Noise (original - denoised)\nLess structure = better', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_method_noise.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig2_method_noise.png")

    # ========== EXPERIMENT 3: Parameter study ==========
    print("\n" + "=" * 60)
    print("Experiment 3: Parameter Study")
    print("=" * 60)

    # use smaller image for speed
    small_img = resize(image, (128, 128), preserve_range=True)
    small_noisy = add_noise(small_img, noise_sigma)

    # test different h values
    print("\nTesting h parameter...")
    h_values = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40]
    h_psnr = []
    h_ssim = []

    for h in h_values:
        result = nlm.denoise(small_noisy, h=h)
        h_psnr.append(compute_psnr(small_img, result))
        h_ssim.append(compute_ssim(small_img, result))
        print(f"  h={h}: PSNR={h_psnr[-1]:.2f} dB")

    best_h = h_values[np.argmax(h_psnr)]
    print(f"Best h = {best_h}")

    # test patch sizes
    print("\nTesting patch size...")
    patch_sizes = [3, 5, 7, 9, 11]
    ps_psnr = []

    for ps in patch_sizes:
        temp_nlm = NLMeansDenoiser(patch_size=ps, search_window=21)
        result = temp_nlm.denoise(small_noisy, h=best_h)
        ps_psnr.append(compute_psnr(small_img, result))
        print(f"  {ps}x{ps}: PSNR={ps_psnr[-1]:.2f} dB")

    # test window sizes
    print("\nTesting window size...")
    window_sizes = [11, 15, 21, 25, 31]
    sw_psnr = []

    for sw in window_sizes:
        temp_nlm = NLMeansDenoiser(patch_size=7, search_window=sw)
        result = temp_nlm.denoise(small_noisy, h=best_h)
        sw_psnr.append(compute_psnr(small_img, result))
        print(f"  {sw}x{sw}: PSNR={sw_psnr[-1]:.2f} dB")

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(h_values, h_psnr, 'b-o')
    axes[0].axvline(x=best_h, color='r', linestyle='--', label=f'Best h={best_h}')
    axes[0].set_xlabel('h parameter')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Effect of h')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(patch_sizes, ps_psnr, 'g-s')
    axes[1].axvline(x=7, color='r', linestyle='--', label='Default 7x7')
    axes[1].set_xlabel('Patch size')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Effect of patch size')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(window_sizes, sw_psnr, 'r-^')
    axes[2].axvline(x=21, color='r', linestyle='--', label='Default 21x21')
    axes[2].set_xlabel('Window size')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('Effect of search window')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle('Parameter Study', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_parameter_study.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig3_parameter_study.png")

    # ========== EXPERIMENT 4: Different noise levels ==========
    print("\n" + "=" * 60)
    print("Experiment 4: Noise Level Study")
    print("=" * 60)

    noise_levels = [10, 15, 20, 25, 30, 35, 40, 50]

    psnr_nlm = []
    psnr_gf = []
    psnr_af = []
    psnr_tv = []

    print("Testing different noise levels...")
    for sigma in noise_levels:
        test_noisy = add_noise(small_img, sigma)
        h_opt = 0.75 * sigma

        nlm_res = nlm.denoise(test_noisy, h=h_opt)
        gf_res = gaussian_filter_denoise(test_noisy, sigma=max(1.0, sigma/15))
        af_res = anisotropic_filter(test_noisy, iterations=25, kappa=sigma)
        tv_res = tv_denoise(test_noisy, weight=0.08 + sigma*0.002, iterations=150)

        psnr_nlm.append(compute_psnr(small_img, nlm_res))
        psnr_gf.append(compute_psnr(small_img, gf_res))
        psnr_af.append(compute_psnr(small_img, af_res))
        psnr_tv.append(compute_psnr(small_img, tv_res))

        print(f"  sigma={sigma}: NL-means={psnr_nlm[-1]:.1f}, Gaussian={psnr_gf[-1]:.1f}")

    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, psnr_nlm, 'b-o', label='NL-means')
    plt.plot(noise_levels, psnr_gf, 'g-s', label='Gaussian')
    plt.plot(noise_levels, psnr_af, 'r-^', label='Anisotropic')
    plt.plot(noise_levels, psnr_tv, 'm-d', label='Total Variation')
    plt.xlabel('Noise sigma')
    plt.ylabel('PSNR (dB)')
    plt.title('Performance vs Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_noise_level_study.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig4_noise_level_study.png")

    # ========== EXPERIMENT 5: Weight visualization ==========
    print("\n" + "=" * 60)
    print("Experiment 5: Weight Distribution")
    print("=" * 60)

    rows, cols = small_img.shape
    points = [
        (rows//4, cols//4, 'Flat'),
        (rows//2, cols//2, 'Edge'),
        (3*rows//4, 3*cols//4, 'Texture')
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(small_img, cmap='gray', vmin=0, vmax=255)
    for (py, px, label) in points:
        axes[0, 0].plot(px, py, 'ro', markersize=8)
        axes[0, 0].annotate(label, (px+2, py-2), color='red', fontsize=8)
    axes[0, 0].set_title('Sample points')
    axes[0, 0].axis('off')

    for idx, (py, px, label) in enumerate(points):
        wmap = compute_weight_map(small_img, (py, px), patch_size=7, search_window=21, h=15)
        axes[0, idx+1].imshow(wmap, cmap='hot', vmin=0, vmax=1)
        axes[0, idx+1].set_title(f'{label} weights')
        axes[0, idx+1].axis('off')

        center = wmap.shape[0] // 2
        axes[1, idx+1].plot(wmap[center, :], 'b-')
        axes[1, idx+1].set_title(f'{label} profile')
        axes[1, idx+1].set_xlabel('Position')
        axes[1, idx+1].set_ylabel('Weight')
        axes[1, idx+1].grid(True, alpha=0.3)

    axes[1, 0].axis('off')

    plt.suptitle('NL-means Weight Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_weight_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig5_weight_distribution.png")

    # ========== EXPERIMENT 6: High noise ==========
    print("\n" + "=" * 60)
    print("Experiment 6: High Noise Test")
    print("=" * 60)

    high_sigma = 35
    high_noisy = add_noise(image, high_sigma)
    h_high = 0.75 * high_sigma

    print(f"Testing with sigma={high_sigma}...")

    high_gf = gaussian_filter_denoise(high_noisy, sigma=2.5)
    high_af = anisotropic_filter(high_noisy, iterations=30, kappa=high_sigma)
    high_tv = tv_denoise(high_noisy, weight=0.15, iterations=200)
    high_nlm = nlm.denoise(high_noisy, h=h_high)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(high_noisy, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'Noisy (sigma={high_sigma})')
    axes[0].axis('off')

    axes[1].imshow(high_gf, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Gaussian ({compute_psnr(image, high_gf):.1f} dB)')
    axes[1].axis('off')

    axes[2].imshow(high_af, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Anisotropic ({compute_psnr(image, high_af):.1f} dB)')
    axes[2].axis('off')

    axes[3].imshow(high_tv, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(f'TV ({compute_psnr(image, high_tv):.1f} dB)')
    axes[3].axis('off')

    axes[4].imshow(high_nlm, cmap='gray', vmin=0, vmax=255)
    axes[4].set_title(f'NL-means ({compute_psnr(image, high_nlm):.1f} dB)', fontweight='bold')
    axes[4].axis('off')

    plt.suptitle(f'High Noise Comparison (sigma={high_sigma})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_high_noise_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig6_high_noise_comparison.png")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n" + "-" * 60)
    print(f"{'Sigma':<8} {'Method':<20} {'PSNR':>10} {'SSIM':>8}")
    print("-" * 60)

    for sigma in [20, 35]:
        test_noisy = add_noise(image, sigma)
        h_opt = 0.75 * sigma

        methods = {
            'Gaussian': gaussian_filter_denoise(test_noisy, sigma=max(1.5, sigma/12)),
            'Anisotropic': anisotropic_filter(test_noisy, iterations=25, kappa=sigma),
            'Total Variation': tv_denoise(test_noisy, weight=0.08 + sigma*0.002, iterations=150),
            'NL-means': nlm.denoise(test_noisy, h=h_opt)
        }

        for name, result in methods.items():
            psnr_val = compute_psnr(image, result)
            ssim_val = compute_ssim(image, result)
            print(f"{sigma:<8} {name:<20} {psnr_val:>10.2f} {ssim_val:>8.4f}")
        print()

    print("-" * 60)
    print(f"\nAll figures saved to: {output_dir}/")
    print("Done!")

    return output_dir


if __name__ == "__main__":
    image_path = "lena.png"
    run_all_experiments(image_path, './results')