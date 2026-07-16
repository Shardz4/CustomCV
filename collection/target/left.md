# OpenCV Gap Analysis — CustomCV (`rust_cv_lib`)

> **Generated**: 2026-07-06  
> **Source Comparison**: [OpenCV 4.x](https://github.com/opencv/opencv) vs [CustomCV (Shardz4/CustomCV)](https://github.com/Shardz4/CustomCV)  
> **Location**: `collection/target/docs/` (gitignored — not tracked)

---

## Summary

CustomCV currently implements **~45 functions** across 10 source files, covering basic image processing, edge detection, morphological ops, smoothing, arithmetic/bitwise ops, geometric transforms, histogram ops, and video I/O. OpenCV's `imgproc` module alone exposes **300+ functions**, and there are 15+ additional modules (video, features2d, calib3d, photo, objdetect, etc.).

This document catalogs every significant OpenCV capability that is **missing** from CustomCV.

---

## ✅ Already Implemented (What You Have)

| Category | Module | Functions |
|---|---|---|
| Point Transforms | `transforms.rs` | `apply_negative`, `apply_log`, `apply_gamma`, `rgb_to_gray`, `apply_threshold`, `rgb_to_cmy`, `apply_frequency_filter`, `apply_threshold_binary_inv`, `apply_threshold_trunc`, `apply_threshold_tozero`, `apply_threshold_tozero_inv`, `apply_threshold_triangle`, `apply_otsu_with_mode` |
| Histogram Ops | `histogram.rs` | `hist_equalize_rgb`, `hist_equalize_gray`, `hist_spec_rgb`, `hist_spec_gray`, `apply_otsu_threshold`, `calc_hist`, `compare_hist`, `match_template`, `calc_back_project`, `emd_1d` |
| Spatial Filters | `filters.rs` | `median_filter` (3×3), `laplacian_filter` (3×3), `pyr_down`, `pyr_up` |
| Smoothing | `smoothing.rs` | `apply_blur`, `apply_gaussian_blur`, `apply_median_blur`, `apply_bilateral_filter`, `apply_filter2d` |
| Edge & Feature Detection | `edge_detection.rs` | `apply_canny`, `harris_corner`, `shi_tomasi_corners`, `hough_lines`, `hough_circles` |
| Morphological Ops | `morphological.rs` | `apply_erosion`, `apply_dilation`, `opening`, `apply_closing`, `morphological_gradient`, `top_hat`, `black_hat` |
| Arithmetic & Bitwise | `arithematic.rs` | `add_images`, `sub_images`, `add_weighted`, `bitwise_and`, `bitwise_or`, `bitwise_not`, `bitwise_xor` |
| Geometric Transforms | `geometric.rs` | `apply_resize` (nearest-neighbor), `apply_translate`, `apply_rotate`, `apply_warp` |
| Video I/O | `vid.rs` | `video_capture`, `extract_images_from_video`, `extract_video_from_images`, `background_subtract_mog2` |
| Color Conversions | `color_convert.rs` | `rgb_to_hsv`, `rgb_to_hls`, `rgb_to_ycrcb`, `rgb_to_xyz`, `rgb_to_lab`, `rgb_to_luv`, `bgr_to_rgb`, `gray_to_rgb`, `rgb_to_yuv` |
| Gradient & Edge Ops | `gradient.rs` | `apply_sobel`, `apply_scharr`, `apply_laplacian` |
| Contour & Shape Analysis | `contours.rs` | `find_contours`, `draw_contours`, `contour_area`, `arc_length`, `bounding_rect`, `min_area_rect`, `min_enclosing_circle`, `fit_ellipse`, `convex_hull`, `convexity_defects`, `approx_poly_dp`, `moments`, `hu_moments`, `match_shapes`, `is_contour_convex`, `point_polygon_test` |
| Image Segmentation | `segmentation.rs` | `connected_components`, `connected_components_with_stats`, `distance_transform`, `flood_fill`, `watershed`, `grab_cut` |

---

## ❌ Missing — Organized by OpenCV Module

### 1. Color Space Conversions (`cv2.cvtColor`)

**Priority: 🔴 HIGH** — OpenCV's most-used function

| OpenCV Function | What It Does | Status |
|---|---|---|
| `cvtColor` (general) | Universal color-space converter (supports 150+ conversion codes) | ❌ Missing (individual conversion functions implemented instead) |
| RGB → HSV | Hue-Saturation-Value (critical for color-based segmentation) | ✅ Completed |
| RGB → HSL/HLS | Hue-Lightness-Saturation | ✅ Completed |
| RGB → YCrCb | Luma + chroma (used in JPEG, face detection) | ✅ Completed |
| RGB → Lab / Luv | Perceptually uniform color spaces | ✅ Completed |
| RGB → XYZ | CIE 1931 color space | ✅ Completed |
| BGR → RGB | Channel reordering (OpenCV default is BGR) | ✅ Completed |
| Gray → RGB | Single-channel to 3-channel expansion | ✅ Completed |
| RGB → YUV | Used in video encoding | ✅ Completed |

> **Note**: RGB↔HSV, RGB↔HLS, RGB↔YCrCb, RGB↔Lab/Luv, RGB↔XYZ, BGR↔RGB, Gray↔RGB, and RGB↔YUV conversions are now fully implemented.

---

### 2. Thresholding (Advanced)

**Priority: 🔴 HIGH**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `adaptiveThreshold` | Per-region threshold (Gaussian or mean adaptive) | ❌ Missing |
| `threshold` with `THRESH_BINARY_INV` | Inverse binary threshold | ✅ Completed |
| `threshold` with `THRESH_TRUNC` | Truncate values above threshold | ✅ Completed |
| `threshold` with `THRESH_TOZERO` | Set to zero below threshold | ✅ Completed |
| `threshold` with `THRESH_TOZERO_INV` | Set to zero above threshold | ✅ Completed |
| `threshold` with `THRESH_OTSU` flag | Combined with other modes | ✅ Completed |
| `threshold` with `THRESH_TRIANGLE` | Triangle algorithm for bimodal histograms | ✅ Completed |

> **Note**: Standard, inverse, trunc, to-zero, triangle, and Otsu-combined threshold modes are now fully implemented.

---

### 3. Gradient & Edge Operators

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `Sobel` | First/second derivative with Sobel kernel | ✅ Completed |
| `Scharr` | More accurate gradient than Sobel (3×3 only) | ✅ Completed |
| `Laplacian` (configurable ksize) | Second derivative with variable kernel sizes | ✅ Completed |
| `filter2D` | General 2D convolution with any kernel | ✅ Completed |

---

### 4. Contour & Shape Analysis

**Priority: 🔴 HIGH** — Core for object detection, measurement, and segmentation

| OpenCV Function | What It Does | Status |
|---|---|---|
| `findContours` | Extract contour chains from binary images | ✅ Completed |
| `drawContours` | Render contours onto an image | ✅ Completed |
| `contourArea` | Area enclosed by a contour | ✅ Completed |
| `arcLength` | Perimeter of a contour | ✅ Completed |
| `boundingRect` | Axis-aligned bounding rectangle | ✅ Completed |
| `minAreaRect` | Minimum-area rotated bounding rectangle | ✅ Completed |
| `minEnclosingCircle` | Smallest enclosing circle | ✅ Completed |
| `fitEllipse` | Fit an ellipse to a point set | ✅ Completed |
| `convexHull` | Convex hull of a point set | ✅ Completed |
| `convexityDefects` | Defects in the convex hull | ✅ Completed |
| `approxPolyDP` | Polygon approximation (Douglas-Peucker) | ✅ Completed |
| `moments` | Image moments (centroid, orientation, etc.) | ✅ Completed |
| `HuMoments` | 7 Hu invariant moments (rotation/scale invariant) | ✅ Completed |
| `matchShapes` | Compare contour shapes using Hu moments | ✅ Completed |
| `isContourConvex` | Test if contour is convex | ✅ Completed |
| `pointPolygonTest` | Test if a point is inside/on/outside a contour | ✅ Completed |

---

### 5. Image Segmentation

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `watershed` | Marker-based watershed segmentation | ✅ Completed |
| `grabCut` | Interactive foreground extraction (GrabCut algorithm) | ✅ Completed |
| `connectedComponents` | Label connected regions in a binary image | ✅ Completed |
| `connectedComponentsWithStats` | Same + area, bounding boxes, centroids | ✅ Completed |
| `distanceTransform` | Distance to nearest zero pixel | ✅ Completed |
| `floodFill` | Region growing from a seed point | ✅ Completed |

---

### 6. Template Matching & Histogram Comparison

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `matchTemplate` | Slide a template across an image and score matches | ✅ Completed |
| `calcHist` | Compute histogram bins from an image | ✅ Completed |
| `compareHist` | Compare two histograms (correlation, chi-squared, Bhattacharyya, intersection) | ✅ Completed |
| `calcBackProject` | Back-project a histogram model onto an image | ✅ Completed |
| `EMD` / `EMDL1` | Earth mover's distance between histograms | ✅ Completed |

---

### 7. Drawing & Annotation

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `line` | Draw a line segment | ✅ Completed |
| `rectangle` | Draw a rectangle | ✅ Completed |
| `circle` | Draw a circle | ✅ Completed |
| `ellipse` | Draw an ellipse or arc | ✅ Completed |
| `polylines` | Draw polylines | ✅ Completed |
| `fillPoly` | Fill a polygon | ✅ Completed |
| `putText` | Render text on an image | ✅ Completed |
| `arrowedLine` | Draw an arrowed line | ✅ Completed |

---

### 8. Geometric Transforms (Advanced)

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `resize` (bilinear) | Bilinear interpolation resize | ✅ Completed |
| `resize` (bicubic) | Bicubic interpolation resize | ✅ Completed |
| `resize` (Lanczos) | Lanczos resampling | ✅ Completed |
| `warpAffine` | 2×3 affine transform | ✅ Completed |
| `getRotationMatrix2D` | Compute 2×3 rotation matrix | ✅ Completed |
| `getAffineTransform` | Compute affine matrix from 3 point pairs | ✅ Completed |
| `getPerspectiveTransform` | Compute 3×3 perspective matrix from 4 point pairs | ✅ Completed |
| `remap` | General pixel remapping with interpolation | ✅ Completed |
| `flip` | Flip image horizontally/vertically/both | ✅ Completed |
| `transpose` | Matrix transpose | ✅ Completed |
| `invertAffineTransform` | Invert an affine matrix | ✅ Completed |
| `logPolar` / `linearPolar` | Log-polar / linear-polar transforms | ✅ Completed |

---

### 9. Core Array Operations (Missing from `arithematic.rs`)

**Priority: 🟢 LOW** (many can be done with NumPy)

| OpenCV Function | What It Does | Status |
|---|---|---|
| `multiply` | Per-element multiplication | ✅ Completed |
| `divide` | Per-element division | ✅ Completed |
| `absdiff` | Absolute difference of two images | ✅ Completed |
| `min` / `max` | Per-element min/max of two arrays | ✅ Completed |
| `normalize` | Normalize array values to a range | ✅ Completed |
| `convertScaleAbs` | Scale, compute absolute value, and convert to 8-bit | ✅ Completed |
| `split` | Split multi-channel image into separate channels | ✅ Completed |
| `merge` | Merge separate channels into a multi-channel image | ✅ Completed |
| `mixChannels` | Shuffle channels between arrays | ✅ Completed |
| `inRange` | Mask pixels within a value range (color filtering) | ✅ Completed |
| `LUT` | Apply a lookup table to an image | ✅ Completed |
| `countNonZero` | Count non-zero pixels | ✅ Completed |
| `meanStdDev` | Compute mean and standard deviation | ✅ Completed |
| `minMaxLoc` | Find min/max values and their locations | ✅ Completed |
| `pow` | Per-element power | ✅ Completed |
| `sqrt` | Per-element square root | ✅ Completed |
| `exp` | Per-element exponential | ✅ Completed |
| `log` | Per-element natural log | ✅ Completed |
| `phase` | Compute phase (angle) from 2D vectors | ✅ Completed |
| `magnitude` | Compute magnitude from 2D vectors | ✅ Completed |
| `cartToPolar` / `polarToCart` | Coordinate system conversions | ✅ Completed |

---

### 10. Feature Detection & Description (`features2d` module)

**Priority: 🔴 HIGH** — Fundamental for image matching, stitching, and recognition

| OpenCV Function | What It Does | Status |
|---|---|---|
| `SIFT_create` | Scale-Invariant Feature Transform detector + descriptor | ❌ Missing |
| `ORB_create` | Oriented FAST and Rotated BRIEF | ❌ Missing |
| `BRISK_create` | Binary Robust Invariant Scalable Keypoints | ❌ Missing |
| `AKAZE_create` | Accelerated KAZE features | ❌ Missing |
| `FastFeatureDetector_create` | FAST corner detection | ❌ Missing |
| `MSER_create` | Maximally Stable Extremal Regions (blob detection) | ❌ Missing |
| `SimpleBlobDetector_create` | Detect blobs by color/size/shape | ❌ Missing |
| `GFTTDetector_create` | Good Features to Track (Shi-Tomasi wrapper) | ⚠️ Partial (raw response map only, no NMS/selection) |
| `BFMatcher_create` | Brute-force descriptor matcher | ❌ Missing |
| `FlannBasedMatcher_create` | Approximate nearest-neighbor matcher | ❌ Missing |
| `drawKeypoints` | Visualize keypoints on image | ❌ Missing |
| `drawMatches` | Visualize descriptor matches | ❌ Missing |
| `findHomography` | RANSAC-based homography estimation | ❌ Missing |

---

### 11. Video Analysis (`video` module)

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `calcOpticalFlowPyrLK` | Lucas-Kanade sparse optical flow | ✅ Completed |
| `calcOpticalFlowFarneback` | Farneback dense optical flow | ✅ Completed |
| `createBackgroundSubtractorKNN` | KNN-based background subtraction | ✅ Completed |
| `createBackgroundSubtractorMOG2` | MOG2 background subtraction | ✅ Completed |
| `meanShift` | Mean-shift object tracking | ✅ Completed |
| `CamShift` | Continuously Adaptive Mean Shift tracking | ✅ Completed |
| `KalmanFilter` | Kalman filter for motion prediction | ✅ Completed |
| `DISOpticalFlow` | Dense Inverse Search optical flow | ✅ Completed |
| `SparsePyrLKOpticalFlow` | Sparse Pyramidal Lucas-Kanade | ✅ Completed |
| `buildOpticalFlowPyramid` | Pre-build image pyramid for optical flow | ✅ Completed |

---

### 12. Object Detection (`objdetect` module)

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `CascadeClassifier` | Haar/LBP cascade object detector (face detection, etc.) | ✅ Completed |
| `HOGDescriptor` | Histogram of Oriented Gradients (pedestrian detection) | ✅ Completed |
| `QRCodeDetector` | QR code detection and decoding | ✅ Completed |
| `groupRectangles` | Group overlapping detection rectangles | ✅ Completed |

---

### 13. Computational Photography (`photo` module)

**Priority: 🟡 MEDIUM**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `inpaint` | Image inpainting (Telea / Navier-Stokes) | ✅ Completed |
| `fastNlMeansDenoising` | Non-local means denoising (grayscale) | ✅ Completed |
| `fastNlMeansDenoisingColored` | Non-local means denoising (color) | ✅ Completed |
| `seamlessClone` | Seamless image compositing | ✅ Completed |
| `colorChange` | Local color adjustment | ✅ Completed |
| `illuminationChange` | Lighting adjustment | ✅ Completed |
| `textureFlattening` | Flatten textures while preserving edges | ✅ Completed |
| `decolor` | Contrast-preserving color-to-gray conversion | ✅ Completed |
| `createTonemap` | HDR tone mapping | ✅ Completed |
| `createMergeMertens` | Exposure fusion (multi-exposure blending) | ✅ Completed |
| `createCalibrateDebevec` | Camera response calibration for HDR | ✅ Completed |
| `createMergeDebevec` | HDR image merging | ✅ Completed |
| `edgePreservingFilter` | Edge-preserving smoothing | ✅ Completed |
| `detailEnhance` | Detail enhancement filter | ✅ Completed |
| `pencilSketch` | Non-photorealistic pencil sketch rendering | ✅ Completed |
| `stylization` | Stylization filter (painting effect) | ✅ Completed |

---

### 14. Camera Calibration & 3D (`calib3d` module)

**Priority: 🟢 LOW** (specialized)

| OpenCV Function | What It Does | Status |
|---|---|---|
| `calibrateCamera` | Intrinsic camera parameter estimation | ✅ Completed |
| `findChessboardCorners` | Detect chessboard pattern for calibration | ✅ Completed |
| `undistort` | Remove lens distortion | ✅ Completed |
| `solvePnP` | Pose estimation from 3D-2D correspondences | ✅ Completed |
| `stereoCalibrate` | Stereo camera calibration | ✅ Completed |
| `stereoRectify` | Stereo image rectification | ✅ Completed |
| `reprojectImageTo3D` | Disparity map to 3D point cloud | ✅ Completed |
| `findEssentialMat` | Essential matrix estimation | ✅ Completed |
| `findFundamentalMat` | Fundamental matrix estimation | ✅ Completed |
| `decomposeHomographyMat` | Decompose homography into rotation/translation | ✅ Completed |
| `triangulatePoints` | 3D point triangulation from stereo | ✅ Completed |

---

### 15. DNN Module (`dnn`)

**Priority: 🟢 LOW**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `readNet` / `readNetFromONNX` / `readNetFromTensorflow` / etc. | Load pre-trained neural networks | ✅ Completed |
| `blobFromImage` | Pre-process image for neural network input | ✅ Completed |
| `Net.forward` | Run inference on loaded network | ✅ Completed |
| `NMSBoxes` | Non-maximum suppression for detection boxes | ✅ Completed |

---

### 16. Image I/O (`imgcodecs` module)

**Priority: 🟢 LOW**

| OpenCV Function | What It Does | Status |
|---|---|---|
| `imread` | Read image from disk | ✅ Completed |
| `imwrite` | Write image to disk | ✅ Completed |
| `imdecode` / `imencode` | Decode/encode image from/to memory buffer | ✅ Completed |

---

### 17. Miscellaneous (imgproc) — Still Missing

| OpenCV Function | What It Does | Status |
|---|---|---|
| `getStructuringElement` | Create morphological kernels (rect, cross, ellipse) | ❌ Missing |
| `morphologyEx` | Unified morphological operations entry point | ❌ Missing (you implement each operation separately) |
| `Canny` (L2 gradient) | Canny with L2 gradient norm option | ❌ Missing |
| `cornerSubPix` | Sub-pixel corner refinement | ❌ Missing |
| `goodFeaturesToTrack` | Complete Shi-Tomasi with NMS and max corners | ⚠️ Partial |
| `HoughLinesP` | Probabilistic Hough line transform | ❌ Missing |
| `createLineSegmentDetector` | LSD line segment detector | ❌ Missing |
| `cornerEigenValsAndVecs` | Eigenvalues and eigenvectors at each pixel | ❌ Missing |
| `preCornerDetect` | Pre-corner detection function | ❌ Missing |
| `integral` | Integral image (summed area table) | ❌ Missing |
| `sqrBoxFilter` | Squared box filter (for variance computation) | ❌ Missing |
| `sepFilter2D` | Separable 2D filter (faster than filter2D) | ❌ Missing |
| `getGaborKernel` | Generate Gabor filter kernels | ❌ Missing |
| `GaussianBlur` with `BORDER_*` options | Configurable border handling modes | ❌ Missing |

---

## 📊 Gap Statistics

| Category | OpenCV Functions (approx.) | CustomCV Has | Missing |
|---|---|---|---|
| Color Conversions | 30+ | 11 | ~19 |
| Thresholding | 8 | 8 | 0 |
| Gradient/Edge Operators | 6 | 7 | 0 |
| Contour & Shape Analysis | 20+ | 16 | ~4 |
| Segmentation | 6 | 6 | 0 |
| Template Matching & Histograms | 5 | 5 | 0 |
| Drawing & Annotation | 8+ | 8 | 0 |
| Geometric Transforms (Advanced) | 15+ | 16 | 0 |
| Core Array Ops | 25+ | 28 | 0 |
| Feature Detection (features2d) | 15+ | 3 | ~12 |
| Video Analysis | 10+ | 10 | 0 |
| Object Detection | 4+ | 4 | 0 |
| Computational Photography | 16+ | 16 | 0 |
| Camera Calibration & 3D | 12+ | 11 | 0 |
| DNN | 5+ | 5 | 0 |
| Image I/O | 4+ | 4 | 0 |
| Misc imgproc | 15+ | 0 | ~15 |
| **TOTAL** | **~200+** | **158** | **~50** |

---

## 🎯 Recommended Implementation Priorities

### Phase 1 — Essential Gaps (Highest Impact)
1. **Adaptive threshold** — `adaptiveThreshold`
2. **Drawing primitives** — `line`, `rectangle`, `circle`, `putText`
3. **`flip`** and **`transpose`** — Trivial to implement, frequently used

### Phase 2 — Feature Detection & Matching
4. **ORB or SIFT** — Feature detector + descriptor
5. **BFMatcher** — Brute-force descriptor matching
6. **`findHomography`** — RANSAC homography

### Phase 3 — Advanced Processing
7. **Optical flow** — `calcOpticalFlowPyrLK` or Farneback
8. **Inpainting / Denoising** — `inpaint`, `fastNlMeansDenoising`

### Phase 4 — Nice to Have
9. **Advanced resize** — Bilinear, bicubic interpolation
10. **Affine transforms** — `warpAffine`, `getRotationMatrix2D`
11. **HOG / Cascade classifiers** — Object detection
12. **HDR / Tone mapping** — Computational photography

---

## 🔧 Partial Implementations Needing Enhancement

| Function | Current State | What's Missing |
|---|---|---|
| `apply_resize` | Fully featured | None (supports nearest, bilinear, bicubic, and Lanczos4) |
| `shi_tomasi_corners` | Returns raw response map | Add NMS, max corners, quality threshold (like `goodFeaturesToTrack`) |
| `hough_circles` | Fixed single radius | Multi-radius detection, gradient-based voting |
| Morphological ops | 2D grayscale only | Support 3D (color) images |
| Border handling | Implicit clamping | Explicit BORDER_REFLECT, BORDER_WRAP, BORDER_CONSTANT modes |
