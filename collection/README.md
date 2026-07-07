# CustomCV — Function Reference & Repo Structure

This document covers the source layout and every public function exposed by `rust_cv_lib`.

---

## 📁 Repository Structure

```
collection/
├── .cargo/
│   └── config.toml          # PYO3 ABI3 forward-compat flag
├── .github/
│   └── workflows/
│       └── CI.yml            # GitHub Actions CI — builds wheels for Linux / macOS / Windows
├── src/
│   ├── lib.rs                # PyO3 module entry-point — registers all functions
│   ├── helpers.rs            # Internal utilities (median/laplacian kernels, Otsu, structure tensor)
│   ├── transforms.rs         # Point / pixel transforms & frequency filtering
│   ├── histogram.rs          # Histogram equalization, specification, Otsu threshold
│   ├── filters.rs            # Spatial filters (median, laplacian) — delegates to helpers
│   ├── smoothing.rs          # Image smoothing filters (box blur, Gaussian blur)
│   ├── edge_detection.rs     # Canny, Harris, Shi-Tomasi, Hough lines & circles
│   ├── morphological.rs      # Erosion, dilation, opening, closing, gradient, top/black hat
│   ├── arithematic.rs        # Arithmetic & bitwise image operations
│   ├── geometric.rs          # Resize, translate, rotate, perspective warp
│   ├── color_convert.rs      # Color space conversions (HSV, HLS, YCrCb, XYZ, Lab, Luv, YUV)
│   ├── gradient.rs           # Gradient & edge operators (Sobel, Scharr, Laplacian)
│   └── contours.rs           # Contour & shape analysis (Suzuki85, boundingRect, minAreaRect, minEnclosingCircle, fitEllipse)
├── Cargo.toml                # Rust crate config (cdylib for PyO3)
├── Cargo.lock
├── pyproject.toml            # Maturin / PEP 517 build config
└── .gitignore
```

### Key Dependencies (`Cargo.toml`)

| Crate | Version | Purpose |
|---|---|---|
| `pyo3` | 0.21.2 | Rust ↔ Python bindings |
| `numpy` | 0.21.0 | NumPy array interop (`PyReadonlyArrayDyn`, `IntoPyArray`) |
| `ndarray` | 0.15.6 | N-dimensional array operations in Rust |
| `num-complex` | 0.4.5 | Complex-number support for frequency-domain filters |

---

## 📖 Function Reference

All functions are called from Python as `rust_cv_lib.<function_name>(...)`.
Images are passed in as NumPy arrays (`np.ndarray`) and results are returned the same way.

---

### 1. Point / Pixel Transforms — `transforms.rs`

| Function | Signature | Description |
|---|---|---|
| `apply_negative` | `(image: ndarray[u8]) → ndarray[u8]` | Inverts pixel values: `255 − pixel`. |
| `apply_log` | `(image: ndarray[u8]) → ndarray[u8]` | Log transform: `c · ln(1 + pixel)`, auto-scaled to [0, 255]. |
| `apply_gamma` | `(image: ndarray[u8], gamma: float) → ndarray[u8]` | Power-law (gamma) correction. |
| `rgb_to_gray` | `(image: ndarray[u8]) → ndarray[u8]` | Weighted luminance conversion (0.299R + 0.587G + 0.114B). |
| `apply_threshold` | `(image: ndarray[u8], threshold_value: int) → ndarray[u8]` | Binary threshold — pixels > thresh → 255, else 0. |
| `apply_threshold_binary_inv` | `(image: ndarray[u8], threshold_value: int) → ndarray[u8]` | Inverse binary — pixels > thresh → 0, else 255. |
| `apply_threshold_trunc` | `(image: ndarray[u8], threshold_value: int) → ndarray[u8]` | Truncate — pixels > thresh → thresh, else unchanged. |
| `apply_threshold_tozero` | `(image: ndarray[u8], threshold_value: int) → ndarray[u8]` | To-zero — pixels > thresh → unchanged, else 0. |
| `apply_threshold_tozero_inv` | `(image: ndarray[u8], threshold_value: int) → ndarray[u8]` | To-zero inverse — pixels > thresh → 0, else unchanged. |
| `apply_threshold_triangle` | `(image: ndarray[u8]) → (int, ndarray[u8])` | Triangle auto-threshold + binary. Returns `(threshold, image)`. |
| `apply_otsu_with_mode` | `(image: ndarray[u8], mode: str) → (int, ndarray[u8])` | Otsu auto-threshold + any mode (`binary`, `binary_inv`, `trunc`, `tozero`, `tozero_inv`). |
| `rgb_to_cmy` | `(image: ndarray[u8]) → ndarray[f32]` | Converts RGB [0, 255] → CMY [0.0, 1.0]. Input must be (H, W, 3). |
| `apply_frequency_filter` | `(f_shifted: ndarray[complex128], d0: float, filter_type: str) → ndarray[complex128]` | Applies a frequency-domain mask. Supported types: `"ILPF"`, `"IHPF"`, `"GLPF"`, `"GHPF"`. |

---

### 1b. Color Space Conversions — `color_convert.rs`

All colour functions accept and return NumPy arrays. RGB inputs must have shape **(H, W, 3)**.

| Function | Signature | Description |
|---|---|---|
| `rgb_to_hsv` | `(image: ndarray[u8]) → ndarray[u8]` | RGB → HSV. H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255] (OpenCV convention). |
| `rgb_to_hls` | `(image: ndarray[u8]) → ndarray[u8]` | RGB → HLS. H ∈ [0, 180], L ∈ [0, 255], S ∈ [0, 255]. |
| `rgb_to_ycrcb` | `(image: ndarray[u8]) → ndarray[u8]` | RGB → YCrCb (BT.601). All channels ∈ [0, 255]. Cr/Cb offset by 128. |
| `rgb_to_xyz` | `(image: ndarray[u8]) → ndarray[f32]` | RGB → CIE XYZ (sRGB, D65). Applies inverse-gamma linearization. |
| `rgb_to_lab` | `(image: ndarray[u8]) → ndarray[f32]` | RGB → CIE L\*a\*b\* (D65). L ∈ [0, 100], a/b ≈ [−128, 127]. |
| `rgb_to_luv` | `(image: ndarray[u8]) → ndarray[f32]` | RGB → CIE L\*u\*v\* (D65). L ∈ [0, 100]; u/v can be negative. |
| `bgr_to_rgb` | `(image: ndarray[u8]) → ndarray[u8]` | Swaps channels 0 ↔ 2 (BGR to RGB or vice-versa). |
| `gray_to_rgb` | `(image: ndarray[u8]) → ndarray[u8]` | 2D grayscale (H, W) → 3-channel (H, W, 3) by replication. |
| `rgb_to_yuv` | `(image: ndarray[u8]) → ndarray[u8]` | RGB → YUV (BT.601). All channels ∈ [0, 255]. U/V offset by 128. |

---

### 2. Histogram Operations — `histogram.rs`

| Function | Signature | Description |
|---|---|---|
| `hist_equalize_rgb` | `(image: ndarray[u8]) → ndarray[u8]` | Per-channel histogram equalization on an RGB image (H, W, 3). |
| `hist_equalize_gray` | `(image: ndarray[u8]) → ndarray[u8]` | Converts RGB → grayscale, then equalises. Returns 2D result. |
| `hist_spec_rgb` | `(image: ndarray[u8], target_hist: ndarray[f32]) → ndarray[u8]` | Histogram specification (matching) per channel. `target_hist` must be 1D with 256 bins. |
| `hist_spec_gray` | `(image: ndarray[u8], target_hist: ndarray[f32]) → ndarray[u8]` | Grayscale histogram specification. `target_hist` must be 1D with 256 bins. |
| `apply_otsu_threshold` | `(image: ndarray[u8]) → (int, ndarray[u8])` | Otsu's method — returns `(optimal_threshold, binary_image)`. Input must be 2D grayscale. |

---

### 3. Spatial Filters — `filters.rs`

| Function | Signature | Description |
|---|---|---|
| `median_filter` | `(image: ndarray[u8]) → ndarray[u8]` | 3×3 median filter. Supports 2D (grayscale) and 3D (colour) images. |
| `laplacian_filter` | `(image: ndarray[u8]) → ndarray[u8]` | 3×3 Laplacian edge-sharpening filter. Supports 2D and 3D images. |

---

### 3b. Gradient & Edge Operators — `gradient.rs` & `smoothing.rs`

These functions return signed gradients or allow custom 2D filtering.

| Function | Signature | Description |
|---|---|---|
| `apply_sobel` | `(image: ndarray[u8], dx: int, dy: int, ksize: int) → ndarray[f32]` | Sobel gradient operator. `dx`/`dy` ∈ {0, 1, 2}, `ksize` ∈ {1, 3, 5, 7}. Returns signed float values. |
| `apply_scharr` | `(image: ndarray[u8], dx: int, dy: int) → ndarray[f32]` | Scharr gradient operator (3x3). `dx`/`dy` must be exactly one 1 and one 0. Returns signed float values. |
| `apply_laplacian` | `(image: ndarray[u8], ksize: int) → ndarray[f32]` | Laplacian operator with configurable `ksize` ∈ {1, 3, 5, 7}. Returns signed float values. |
| `apply_filter2d` | `(image: ndarray[u8], kernel: ndarray[f64]) → ndarray[u8]` | General 2D convolution with any custom 2D float kernel. |

---

### 4. Edge & Feature Detection — `edge_detection.rs`

| Function | Signature | Description |
|---|---|---|
| `apply_canny` | `(image: ndarray[f64], low_thresh: float, high_thresh: float) → ndarray[f64]` | Full Canny pipeline: Gaussian blur → Sobel gradients → non-max suppression → hysteresis. Supports 2D & 3D. |
| `harris_corner` | `(image: ndarray[u8], window_size: int, k: int) → ndarray[f32]` | Harris corner response map. Input must be 2D grayscale. |
| `shi_tomasi_corners` | `(image: ndarray[u8], window_size: int) → ndarray[f32]` | Shi-Tomasi (min-eigenvalue) corner response map. Input must be 2D grayscale. |
| `hough_lines` | `(image: ndarray[u8], threshold: int, theta_res: int) → list[(float, float)]` | Hough line transform on a binary/edge image. Returns list of `(rho, theta)` pairs. |
| `hough_circles` | `(image: ndarray[u8], radius: int, threshold: int) → list[(int, int, int)]` | Hough circle detection for a fixed radius. Returns list of `(cx, cy, radius)` tuples. |

---

### 5. Morphological Operations — `morphological.rs`

All morphological functions require **2D grayscale** input and a **2D structuring element (kernel)**.

| Function | Signature | Description |
|---|---|---|
| `apply_erosion` | `(image, kernel) → ndarray[u8]` | Erosion — shrinks bright regions. |
| `apply_dilation` | `(image, kernel) → ndarray[u8]` | Dilation — expands bright regions. |
| `opening` | `(image, kernel) → ndarray[u8]` | Opening = erosion → dilation. Removes small bright spots. |
| `apply_closing` | `(image, kernel) → ndarray[u8]` | Closing = dilation → erosion. Fills small dark holes. |
| `morphological_gradient` | `(image, kernel) → ndarray[u8]` | Dilation − erosion. Highlights object boundaries. |
| `top_hat` | `(image, kernel) → ndarray[u8]` | Original − opening. Extracts bright details smaller than the kernel. |
| `black_hat` | `(image, kernel) → ndarray[u8]` | Closing − original. Extracts dark details smaller than the kernel. |

---

### 6. Arithmetic & Bitwise Operations — `arithematic.rs`

All binary operations require both images to have **identical shapes**.

| Function | Signature | Description |
|---|---|---|
| `add_images` | `(img1, img2) → ndarray[u8]` | Saturating addition (clamped at 255). |
| `sub_images` | `(img1, img2) → ndarray[u8]` | Saturating subtraction (clamped at 0). |
| `add_weighted` | `(img1, alpha, img2, beta, gamma) → ndarray[u8]` | Weighted blend: `α·img1 + β·img2 + γ`, clamped to [0, 255]. |
| `bitwise_and` | `(img1, img2) → ndarray[u8]` | Per-element bitwise AND. |
| `bitwise_or` | `(img1, img2) → ndarray[u8]` | Per-element bitwise OR. |
| `bitwise_xor` | `(img1, img2) → ndarray[u8]` | Per-element bitwise XOR. |
| `bitwise_not` | `(img) → ndarray[u8]` | Per-element bitwise NOT (invert all bits). |

---

### 7. Geometric Transforms — `geometric.rs`

All geometric functions support **2D (grayscale)** and **3D (colour)** images.

| Function | Signature | Description |
|---|---|---|
| `apply_resize` | `(image: ndarray[u8], new_w: int, new_h: int) → ndarray[u8]` | Nearest-neighbor resize to `(new_h, new_w)`. |
| `apply_translate` | `(image: ndarray[u8], tx: int, ty: int) → ndarray[u8]` | Shifts the image by `(tx, ty)` pixels. Out-of-bounds pixels become 0. |
| `apply_rotate` | `(image: ndarray[u8], angle: float, center: Optional[(int,int)]) → ndarray[u8]` | Rotates the image by `angle` radians around `center` (default: image centre). |
| `apply_warp` | `(image: ndarray[u8], inv_matrix: ndarray[f64], out_w: int, out_h: int) → ndarray[u8]` | Perspective warp using a 3×3 inverse transformation matrix. |

---

### 8. Smoothing Filters — `smoothing.rs`

All smoothing functions support **2D (grayscale)** and **3D (colour)** images.

| Function | Signature | Description |
|---|---|---|
| `apply_blur` | `(image: ndarray[u8], ksize_w: int, ksize_h: int) → ndarray[u8]` | Applies a normalized box filter to blur the image. |
| `apply_gaussian_blur` | `(image: ndarray[u8], ksize: int, sigma: float) → ndarray[u8]` | Applies a Gaussian blur with the specified kernel size and standard deviation. |
| `apply_median_blur` | `(image: ndarray[u8], ksize: int) → ndarray[u8]` | Applies a median filter to blur the image using a sliding window. |
| `apply_bilateral_filter` | `(image: ndarray[u8], diameter: int, sigma_color: float, sigma_space: float) → ndarray[u8]` | Applies a bilateral filter to the image, reducing noise while preserving edges. |

---

### 9. Video Operations — `vid.rs`

Provides utilities for camera input, frame extraction, and compiling image sequences into videos.

| Function | Signature | Description |
|---|---|---|
| `video_capture` | `(device_index: int = 0, save_path: Optional[str] = None) → None` | Opens webcam feed in a window. Press 'q' or 'ESC' to exit. Optionally saves captured frames to a file path. |
| `extract_images_from_video` | `(video_path: str, output_dir: str, frame_interval: int = 1) → None` | Decodes a video and saves individual frames to the output directory. |
| `extract_video_from_images` | `(image_paths: list[str], output_video_path: str, fps: float = 20.0) → None` | Takes a list of image file paths, sorts them alphabetically, and compiles them into a video. |
| `background_subtract_mog2` | `(video_path: str, history: int = 500, var_threshold: float = 16.0, detect_shadows: bool = True, kernel_size: int = 5, output_path: Optional[str] = None) → None` | Applies MOG2 background subtraction. Outputs a foreground mask per frame (0=bg, 127=shadow, 255=fg), cleaned with morphological opening. Displays live or saves to `output_path`. |

---

### 10. Contour & Shape Analysis — `contours.rs`

All contour functions accept and return NumPy arrays representing points of shape **(N, 2)** or **(N, 1, 2)**.

| Function | Signature | Description |
|---|---|---|
| `find_contours` | `(image: ndarray[u8]) → list[ndarray[i32]]` | Suzuki-Abe border following to find all contours in a binary image. Returns list of `(N, 2)` coordinate arrays. |
| `draw_contours` | `(image: ndarray[u8], contours: list[ndarray[i32]], contour_idx: int, color: Union[int, list[int]], thickness: int = 1) → ndarray[u8]` | Draws outlines or fills contours on a 2D grayscale or 3D color image. Returns the annotated image. |
| `contour_area` | `(contour: ndarray[i32], oriented: bool = False) → float` | Area enclosed by a contour (Shoelace formula). If `oriented` is True, returns signed area. |
| `arc_length` | `(curve: ndarray[i32], closed: bool = True) → float` | Perimeter or arc length of a contour or curve. |
| `bounding_rect` | `(contour: ndarray[i32]) → (int, int, int, int)` | Computes the axis-aligned bounding box: `(x, y, width, height)`. |
| `min_area_rect` | `(contour: ndarray[i32]) → ((float, float), (float, float), float)` | Minimum-area rotated bounding box. Returns `((cx, cy), (w, h), angle_in_degrees)`. |
| `min_enclosing_circle` | `(contour: ndarray[i32]) → ((float, float), float)` | Smallest enclosing circle. Returns `((cx, cy), radius)`. |
| `fit_ellipse` | `(contour: ndarray[i32]) → ((float, float), (float, float), float)` | Fits an ellipse to the point set using algebraic least squares. Returns `((cx, cy), (axis_width, axis_height), angle_in_degrees)`. |

---

### Internal Helpers — `helpers.rs`

These are **not** exposed to Python. They are used internally by other modules.

| Function | Used By | Purpose |
|---|---|---|
| `apply_median_3x3` | `filters.rs` | Applies a 3×3 median filter to a single 2D channel. |
| `apply_laplacian_3x3` | `filters.rs` | Applies a 3×3 Laplacian kernel to a single 2D channel. |
| `calculate_otsu_threshold` | `histogram.rs` | Computes the optimal Otsu threshold for a grayscale channel. |
| `compute_structure_tensor` | `edge_detection.rs` | Computes Sxx, Syy, Sxy structure tensor components (Sobel-based) for corner detection. |
| `convolve_2d_channel` | `smoothing.rs` | Convolves a single 2D channel with an arbitrary 2D float kernel. |
