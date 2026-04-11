import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def nl_means(image, patch_size=3, search_size=11, h=10):
    pad = patch_size // 2
    padded = np.pad(image, pad, mode='reflect').astype(np.float32)
    output = np.zeros_like(image, dtype=np.float32)

    H, W = image.shape
    search_radius = search_size // 2

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

                    dist = np.sum((ref_patch - comp_patch) ** 2)
                    w = np.exp(-dist / (h * h))

                    weights_sum += w
                    pixel_sum += w * padded[x, y]

            if weights_sum == 0.0:
                output[i, j] = image[i, j]
            else:
                output[i, j] = pixel_sum / weights_sum

    return np.clip(output, 0, 255).astype(np.uint8)


def run_pipeline(image_path, name):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    noisy = add_gaussian_noise(img, sigma=20)
    denoised = nl_means(noisy, patch_size=3, search_size=11, h=10)

    gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.5)

    # Save results
    cv2.imwrite(f"{name}_original.png", img)
    cv2.imwrite(f"{name}_noisy.png", noisy)
    cv2.imwrite(f"{name}_nlmeans.png", denoised)
    cv2.imwrite(f"{name}_gaussian.png", gaussian)

    print(f"{name} done")


if __name__ == "__main__":
    # RUN ALL THREE IMAGES
    run_pipeline("lena.tif", "lena")
    run_pipeline("baboon.png", "baboon")
    run_pipeline("real.png", "real")