# CustomCV (rust_cv_lib)

A high-performance Computer Vision library written in Rust with Python bindings using PyO3 and rust-numpy. This library accelerates typical CV operations by executing them natively in Rust, allowing seamless integration with Python's NumPy ecosystem.

## Features

This library includes various modules for image processing tasks:

### 1. Point / Pixel Transforms (`transforms`)
- **Negative:** `apply_negative(image)`
- **Log Transform:** `apply_log(image, c)`
- **Gamma Correction:** `apply_gamma(image, gamma, c)`
- **Thresholding:** `apply_threshold(image, thresh)`
- **Color Space Conversions:** `rgb_to_gray(image)`, `rgb_to_cmy(image)`
- **Frequency Filtering:** `apply_frequency_filter(image, filter_mask)`

### 2. Histogram Operations (`histogram`)
- **Histogram Equalization:** `hist_equalize_rgb(image)`, `hist_equalize_gray(image)`
- **Histogram Specification:** `hist_spec_rgb(source, reference)`, `hist_spec_gray(source, reference)`
- **Otsu's Thresholding:** `apply_otsu_threshold(image)`

### 3. Edge Detection & Spatial Filters (`edge_detection`)
- **Median Filter:** `median_filter(image, kernel_size)`
- **Laplacian Filter:** `laplacian_filter(image, kernel)`
- **Canny Edge Detection:** `apply_canny(image, low_threshold, high_threshold)`

### 4. Morphological Operations (`morphological`)
- **Erosion:** `apply_erosion(image, kernel)`
- **Dilation:** `apply_dilation(image, kernel)`
- **Opening:** `opening(image, kernel)`
- **Closing:** `apply_closing(image, kernel)`

### 5. Arithmetic & Bitwise Operations (`arithematic`)
- **Arithmetic:** `add_images(img1, img2)`, `sub_images(img1, img2)`, `add_weighted(img1, alpha, img2, beta, gamma)`
- **Bitwise:** `bitwise_and(img1, img2)`, `bitwise_or(img1, img2)`, `bitwise_xor(img1, img2)`, `bitwise_not(img)`

## Installation and Build

You need Rust installed on your machine along with a Python environment.

1. Install `maturin` in your Python environment:
   ```bash
   pip install maturin
   ```

2. Build and install the module directly into your current Python environment:
   ```bash
   maturin develop --release
   ```
   *Note: Using `--release` ensures the Rust code is optimized for better performance.*

## Usage Example

After building with `maturin`, you can import and use the library directly in Python via NumPy arrays.

```python
import numpy as np
import cv2
import rust_cv_lib

# Load an image using OpenCV (or any other library yielding a NumPy array)
image = cv2.imread("image.png")

# Convert to Grayscale
gray_image = rust_cv_lib.rgb_to_gray(image)

# Apply Canny Edge Detection
edges = rust_cv_lib.apply_canny(gray_image, 50, 150)

# Apply Morphological Dilation
kernel = np.ones((3, 3), dtype=np.uint8)
dilated = rust_cv_lib.apply_dilation(edges, kernel)

# Display the result
cv2.imshow("Edges", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## License
MIT License
