# CustomCV (`rust_cv_lib`)

A high-performance Computer Vision library written in **Rust** with seamless **Python** bindings.
Operations execute natively in Rust through [PyO3](https://pyo3.rs) + [rust-numpy](https://github.com/PyO3/rust-numpy), so every function accepts and returns regular NumPy arrays — no data-copy overhead.

---

## ✨ What's Inside

| Category | Highlights |
|---|---|
| **Point / Pixel Transforms** | Negative, log, gamma, threshold, RGB ↔ Gray / CMY, frequency-domain filtering, **adaptive thresholding** |
| **Color Space Conversions** | HSV, HLS, YCrCb, XYZ, CIE Lab, CIE Luv, YUV, BGR ↔ RGB, Gray → RGB, **universal `cvtColor` dispatcher** |
| **Histogram Operations** | Equalization (RGB & gray), specification, Otsu's thresholding, 1D histogram calculation (`calcHist`), histogram comparison (`compareHist` via Correlation/Chi-Square/Intersection/Bhattacharyya), template matching (`matchTemplate` via SQDIFF/CCORR/CCOEFF and normalized modes), back projection (`calcBackProject`), Earth Mover's Distance (`EMD` / `emd_1d`) |
| **Spatial Filters & Smoothing** | 3×3 median, Laplacian edge sharpening, box blur, Gaussian blur, median blur, Bilateral filter (all supporting custom **border padding modes**: reflect, replicate, wrap, constant) |
| **Gradient & Edge Operators** | Sobel, Scharr, Laplacian (variable kernel size), filter2D (custom convolution with border modes) |
| **Edge & Feature Detection** | Canny, Harris corners, Shi-Tomasi corners, Hough lines & circles |
| **Feature Matching** | **KD-Tree FLANN matching**, brute-force matching, KNN matching, **drawKeypoints** (standard/rich), **drawMatches** |
| **Geometry & Alignment** | Resize, translate, rotate, perspective warp, **homography estimation (DLT + RANSAC solved via Jacobi rotations)** |
| **Contour & Shape Analysis** | Outer & inner contours (Suzuki85), draw contours (with thickness & fill), area, perimeter (arc length), bounding box, rotated box, enclosing circle, ellipse fitting |
| **Drawing & Annotation** | Draw line segments, rectangles (outline/filled), circles (outline/filled), ellipses & elliptic arcs (outline/filled), polylines (outline), fillPoly (filled polygons) |
| **Image Segmentation** | Marker-based watershed, GrabCut foreground extraction (GMM + ICM spatial smoothing), connected components (4/8 connectivity with stats/centroids), distance transform (Chamfer 3x3), region-growing flood fill |
| **Morphological Operations** | Erosion, dilation, opening, closing, gradient, top-hat, black-hat (**all supporting 2D grayscale & 3D color images**) |
| **Arithmetic & Bitwise Ops** | Add, subtract, weighted blend, AND / OR / XOR / NOT |
| **Video & Background Subtraction** | Webcam capture, frame extraction, image→video, MOG2 background subtraction |

> For the complete function reference and repo structure, see [`collection/README.md`](collection/README.md).

---

## 📋 Prerequisites

| Tool | Version |
|---|---|
| **Rust** | Stable toolchain (install via [rustup.rs](https://rustup.rs)) |
| **Python** | ≥ 3.8 |
| **pip** | Latest recommended |

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shardz4/CustomCV.git
   cd CustomCV/collection
   ```

2. **Install the build tool** ([maturin](https://github.com/PyO3/maturin))
   ```bash
   pip install maturin
   ```

3. **Build & install into your current Python environment**
   ```bash
   maturin develop --release
   ```
   > `--release` compiles with full Rust optimisations — highly recommended for any real workload.

4. **Verify the install**
   ```python
   import rust_cv_lib
   print(dir(rust_cv_lib))   # should list all available functions
   ```

---

## ⚡ Quick Start

```python
import numpy as np
import rust_cv_lib

# Load an image natively using our imread function
image = rust_cv_lib.imread("photo.png")

# Convert to grayscale
gray = rust_cv_lib.rgb_to_gray(image)

# Canny edge detection
edges = rust_cv_lib.apply_canny(gray, 50.0, 150.0)

# Dilate the edges
kernel = np.ones((3, 3), dtype=np.uint8)
dilated = rust_cv_lib.apply_dilation(edges.astype(np.uint8), kernel)

# Save result natively using our imwrite function
rust_cv_lib.imwrite("edges.png", dilated)
```

---

## 🛠 Development

```bash
# Build in debug mode (faster compile, slower runtime)
maturin develop

# Run Rust tests
cargo test

# Lint
cargo clippy
```

The project also ships a GitHub Actions CI workflow (`.github/workflows/CI.yml`) that builds wheels for Linux, macOS, and Windows on every push.

---

## 📄 License

MIT License
