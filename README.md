# CustomCV (`rust_cv_lib`)

A high-performance Computer Vision library written in **Rust** with seamless **Python** bindings.
Operations execute natively in Rust through [PyO3](https://pyo3.rs) + [rust-numpy](https://github.com/PyO3/rust-numpy), so every function accepts and returns regular NumPy arrays — no data-copy overhead.

---

## ✨ What's Inside

| Category | Highlights |
|---|---|
| **Point / Pixel Transforms** | Negative, log, gamma, threshold, RGB ↔ Gray / CMY, frequency-domain filtering |
| **Histogram Operations** | Equalization (RGB & gray), specification, Otsu's thresholding |
| **Spatial Filters** | 3×3 median filter, Laplacian edge sharpening |
| **Edge & Feature Detection** | Canny, Harris corners, Shi-Tomasi corners, Hough lines & circles |
| **Morphological Operations** | Erosion, dilation, opening, closing, gradient, top-hat, black-hat |
| **Arithmetic & Bitwise Ops** | Add, subtract, weighted blend, AND / OR / XOR / NOT |
| **Geometric Transforms** | Resize (nearest-neighbor), translate, rotate, perspective warp |

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
import cv2
import rust_cv_lib

# Load an image (any loader that gives you a NumPy array works)
image = cv2.imread("photo.png")

# Convert to grayscale
gray = rust_cv_lib.rgb_to_gray(image)

# Canny edge detection
edges = rust_cv_lib.apply_canny(gray, 50.0, 150.0)

# Dilate the edges
kernel = np.ones((3, 3), dtype=np.uint8)
dilated = rust_cv_lib.apply_dilation(edges.astype(np.uint8), kernel)

# Show result
cv2.imshow("Edges", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
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
