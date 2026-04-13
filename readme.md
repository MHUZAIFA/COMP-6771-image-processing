# Non-Local Means Image Denoising

## COMP 478/6771 Image Processing - Course Project

### Project Overview

This project implements the Non-Local Means (NL-means) image denoising algorithm from:

**"A non-local algorithm for image denoising"**
Buades, Coll & Morel (CVPR 2005)

The NL-means algorithm exploits the self-similarity property of natural images, computing
a weighted average of all pixels where weights depend on patch similarity.

### Files Included

```
├── nlmeans.py
├── lena.png
├── requirements.txt
├── README.txt
└── results/
    ├── fig1_method_comparison.png
    ├── fig2_method_noise.png
    ├── fig3_parameter_study.png
    ├── fig4_noise_level_study.png
    ├── fig5_weight_distribution.png
    └── fig6_high_noise_comparison.png
```

### Installation

1. Ensure Python 3.8+ is installed

2. Install required packages:

   ```
   pip install -r requirements.txt
   ```

### Usage

Run the main script:

```
python nlmeans.py
```

To use a different image, modify the path in the main block:

```python
if __name__ == "__main__":
    image_path = "path/to/your/image.png"  # Change this path
    output_dir = run_all_experiments(lena_path, './results')
```

### Algorithm Parameters

The implementation uses the following default parameters:

| Parameter      | Default Value | Description                          |
|----------------|---------------|--------------------------------------|
| patch_size     | 7             | Size of similarity patches (7x7)     |
| search_window  | 21            | Size of search window (21x21)        |
| h              | ~0.75*sigma   | Filtering parameter (noise-adaptive) |

### Experiments Performed

1. **Method Comparison** - Compares NL-means with:
   - Gaussian Filter (GF)
   - Anisotropic Diffusion Filter (AF) - Perona-Malik
   - Total Variation Filter (TVF) - ROF model
   - Neighborhood Filter (YNF) - Yaroslavsky

2. **Method Noise Analysis** - Visualizes what each filter removes

3. **Parameterization** - Tests sensitivity to h, patch size, and window size

4. **Noise Level Study** - Performance across sigma = 10 to 50

5. **Weight Distribution** - Visualizes NL-means weights for different regions

6. **High Noise Comparison** - Visual quality at sigma = 35

### Expected Results

For Lena image with Gaussian noise (sigma = 20):

| Method              | PSNR (dB) | SSIM   |
|---------------------|-----------|--------|
| Gaussian Filter     | 29.34     | 0.8160 |
| Anisotropic Filter  | 27.20     | 0.6334 |
| Total Variation     | 22.22     | 0.3672 |
| Neighborhood Filter | 28.72     | 0.7189 |
| **NL-means**        | **31.37** | **0.8074** |

### Notes

- Processing time scales with image size and search window
- A 512x512 image takes approximately 2-3 minutes with default parameters
- For faster testing, resize images to 256x256 or reduce search window

### Author

***Mohammed Huzaifa - 40242080***
Masters of Applied Computer Science, Concordia University

***Masoumeh Farokhpour - 40309733***
Masters of Applied Computer Science, Concordia University

April 2026
