use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use numpy::ndarray::{Array2, ArrayView2, s};
use crate::helpers;

// ==========================================
// getStructuringElement
// ==========================================

/// get_structuring_element() - Create morphological structuring elements.
/// @py: Python interpreter token.
/// @shape: Element shape: 0 = MORPH_RECT, 1 = MORPH_CROSS, 2 = MORPH_ELLIPSE.
/// @ksize_w: Width of the structuring element.
/// @ksize_h: Height of the structuring element.
/// @anchor_x: Anchor x-position within the element.
/// @anchor_y: Anchor y-position within the element.
///
/// Creates structuring elements of specified shape and size for morphological operations.
///
/// Return: Structuring element array.
#[pyfunction]
#[pyo3(signature = (shape, ksize_w, ksize_h, anchor_x = -1, anchor_y = -1))]
pub fn get_structuring_element<'py>(
    py: Python<'py>,
    shape: i32,
    ksize_w: usize,
    ksize_h: usize,
    anchor_x: i32,
    anchor_y: i32,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let ax = if anchor_x < 0 { (ksize_w / 2) as i32 } else { anchor_x };
    let ay = if anchor_y < 0 { (ksize_h / 2) as i32 } else { anchor_y };

    let mut kernel = Array2::<u8>::zeros((ksize_h, ksize_w));

    match shape {
        0 => {
            // MORPH_RECT: all ones
            kernel.fill(1);
        }
        1 => {
            // MORPH_CROSS: cross through the anchor
            for x in 0..ksize_w {
                kernel[[ay as usize, x]] = 1;
            }
            for y in 0..ksize_h {
                kernel[[y, ax as usize]] = 1;
            }
        }
        2 => {
            // MORPH_ELLIPSE: inscribed ellipse
            let cx = (ksize_w as f64 - 1.0) / 2.0;
            let cy = (ksize_h as f64 - 1.0) / 2.0;
            let rx = cx.max(0.5);
            let ry = cy.max(0.5);
            for y in 0..ksize_h {
                for x in 0..ksize_w {
                    let dx = (x as f64 - cx) / rx;
                    let dy = (y as f64 - cy) / ry;
                    if dx * dx + dy * dy <= 1.0 {
                        kernel[[y, x]] = 1;
                    }
                }
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "shape must be 0 (RECT), 1 (CROSS), or 2 (ELLIPSE)",
            ));
        }
    }

    Ok(kernel.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// morphologyEx
// ==========================================

/// morphology_ex() - Unified morphological operations entry point.
/// @py: Python interpreter token.
/// @image: 2D grayscale input image (u8).
/// @op: Morphological operation: 0=ERODE, 1=DILATE, 2=OPEN, 3=CLOSE, 4=GRADIENT, 5=TOPHAT, 6=BLACKHAT.
/// @kernel: Morphological structuring element array.
/// @iterations: Number of times erosion and dilation are applied.
///
/// Performs advanced morphological transformations on the input image.
///
/// Return: Transformed image array.
#[pyfunction]
#[pyo3(signature = (image, op, kernel, iterations = 1))]
pub fn morphology_ex<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    op: i32,
    kernel: PyReadonlyArrayDyn<'py, u8>,
    iterations: usize,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;

    use crate::morphological::{erode_2d, dilate_2d};

    let mut result = img_2d.to_owned();

    match op {
        0 => {
            // MORPH_ERODE
            for _ in 0..iterations {
                result = erode_2d(result.view(), k_2d.view());
            }
        }
        1 => {
            // MORPH_DILATE
            for _ in 0..iterations {
                result = dilate_2d(result.view(), k_2d.view());
            }
        }
        2 => {
            // MORPH_OPEN = erode then dilate
            for _ in 0..iterations {
                result = erode_2d(result.view(), k_2d.view());
                result = dilate_2d(result.view(), k_2d.view());
            }
        }
        3 => {
            // MORPH_CLOSE = dilate then erode
            for _ in 0..iterations {
                result = dilate_2d(result.view(), k_2d.view());
                result = erode_2d(result.view(), k_2d.view());
            }
        }
        4 => {
            // MORPH_GRADIENT = dilate - erode
            let dilated = dilate_2d(img_2d.view(), k_2d.view());
            let eroded = erode_2d(img_2d.view(), k_2d.view());
            result = &dilated - &eroded;
        }
        5 => {
            // MORPH_TOPHAT = src - open(src)
            let mut opened = img_2d.to_owned();
            for _ in 0..iterations {
                opened = erode_2d(opened.view(), k_2d.view());
                opened = dilate_2d(opened.view(), k_2d.view());
            }
            result = &img_2d - &opened;
        }
        6 => {
            // MORPH_BLACKHAT = close(src) - src
            let mut closed = img_2d.to_owned();
            for _ in 0..iterations {
                closed = dilate_2d(closed.view(), k_2d.view());
                closed = erode_2d(closed.view(), k_2d.view());
            }
            result = &closed - &img_2d;
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "op must be 0-6: ERODE, DILATE, OPEN, CLOSE, GRADIENT, TOPHAT, BLACKHAT",
            ));
        }
    }

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// Canny with L2 gradient norm
// ==========================================

/// canny_l2() - Canny edge detection with L2 gradient norm option.
/// @py: Python interpreter token.
/// @image: 2D grayscale input image (u8).
/// @low_thresh: Lower threshold for hysteresis thresholding.
/// @high_thresh: Upper threshold for hysteresis thresholding.
/// @aperture_size: Size of Sobel aperture.
/// @l2_gradient: Flag indicating whether L2 norm should be used.
///
/// Finds edges in the input image and marks them in the output map using the Canny algorithm.
///
/// Return: Binary edge map image array.
#[pyfunction]
#[pyo3(signature = (image, low_thresh, high_thresh, aperture_size = 3, l2_gradient = false))]
pub fn canny_l2<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    low_thresh: f64,
    high_thresh: f64,
    aperture_size: usize,
    l2_gradient: bool,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);

    // Gaussian smoothing (5x5)
    let sigma = 1.4;
    let gk = 2usize;
    let mut blurred = numpy::ndarray::Array2::<f64>::zeros((h, w));
    for y in gk..h.saturating_sub(gk) {
        for x in gk..w.saturating_sub(gk) {
            let mut sum = 0.0;
            let mut wsum = 0.0;
            for dy in -(gk as isize)..=(gk as isize) {
                for dx in -(gk as isize)..=(gk as isize) {
                    let ny = (y as isize + dy) as usize;
                    let nx = (x as isize + dx) as usize;
                    let g = (-(dx * dx + dy * dy) as f64 / (2.0 * sigma * sigma)).exp();
                    sum += img_2d[[ny, nx]] as f64 * g;
                    wsum += g;
                }
            }
            blurred[[y, x]] = sum / wsum;
        }
    }

    // Sobel gradients
    let kx: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let ky: [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let mut mag = numpy::ndarray::Array2::<f64>::zeros((h, w));
    let mut angle = numpy::ndarray::Array2::<f64>::zeros((h, w));

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    let val = blurred[[(y + dy).saturating_sub(1), (x + dx).saturating_sub(1)]];
                    gx += val * kx[dy][dx];
                    gy += val * ky[dy][dx];
                }
            }
            mag[[y, x]] = if l2_gradient {
                (gx * gx + gy * gy).sqrt()
            } else {
                gx.abs() + gy.abs()
            };
            let mut theta = gy.atan2(gx).to_degrees();
            if theta < 0.0 { theta += 180.0; }
            angle[[y, x]] = theta;
        }
    }

    // Non-maximum suppression
    let mut suppressed = numpy::ndarray::Array2::<f64>::zeros((h, w));
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let a = angle[[y, x]];
            let (q, r) = if (0.0 <= a && a < 22.5) || (157.5 <= a && a <= 180.0) {
                (mag[[y, x + 1]], mag[[y, x.saturating_sub(1)]])
            } else if 22.5 <= a && a < 67.5 {
                (mag[[y + 1, x.saturating_sub(1)]], mag[[y.saturating_sub(1), x + 1]])
            } else if 67.5 <= a && a < 112.5 {
                (mag[[y + 1, x]], mag[[y.saturating_sub(1), x]])
            } else {
                (mag[[y.saturating_sub(1), x.saturating_sub(1)]], mag[[y + 1, x + 1]])
            };
            if mag[[y, x]] >= q && mag[[y, x]] >= r {
                suppressed[[y, x]] = mag[[y, x]];
            }
        }
    }

    // Hysteresis thresholding
    let mut result = numpy::ndarray::Array2::<u8>::zeros((h, w));
    let mut strong = Vec::new();

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if suppressed[[y, x]] >= high_thresh {
                result[[y, x]] = 255;
                strong.push((y, x));
            } else if suppressed[[y, x]] >= low_thresh {
                result[[y, x]] = 128; // weak edge
            }
        }
    }

    while let Some((y, x)) = strong.pop() {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let ny = (y as i32 + dy) as usize;
                let nx = (x as i32 + dx) as usize;
                if ny > 0 && ny < h - 1 && nx > 0 && nx < w - 1 && result[[ny, nx]] == 128 {
                    result[[ny, nx]] = 255;
                    strong.push((ny, nx));
                }
            }
        }
    }

    // Clear weak edges
    result.mapv_inplace(|v| if v == 128 { 0 } else { v });

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// cornerSubPix
// ==========================================

/// corner_sub_pix() - Sub-pixel corner refinement.
/// @py: Python interpreter token.
/// @image: 2D grayscale input image (u8).
/// @corners: Initial coordinates of the input corners.
/// @win_size: Half-width of the search window size.
/// @max_iter: Maximum number of iterations.
/// @epsilon: Minimum change value for stopping iteration.
///
/// Refines the corner locations to sub-pixel accuracy using the gradient autocorrelation method.
///
/// Return: Vector of refined corner coordinates.
#[pyfunction]
#[pyo3(signature = (image, corners, win_size = 5, max_iter = 30, epsilon = 0.001))]
pub fn corner_sub_pix<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    corners: Vec<(f64, f64)>,
    win_size: usize,
    max_iter: usize,
    epsilon: f64,
) -> PyResult<Vec<(f64, f64)>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    let half = win_size as i32 / 2;

    // Compute gradients (Sobel)
    let mut gx = numpy::ndarray::Array2::<f64>::zeros((h, w));
    let mut gy = numpy::ndarray::Array2::<f64>::zeros((h, w));
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            gx[[y, x]] = (img_2d[[y, x + 1]] as f64 - img_2d[[y, x.saturating_sub(1)]] as f64) / 2.0;
            gy[[y, x]] = (img_2d[[y + 1, x]] as f64 - img_2d[[y.saturating_sub(1), x]] as f64) / 2.0;
        }
    }

    let mut refined = corners.clone();

    for i in 0..refined.len() {
        let (mut cx, mut cy) = refined[i];

        for _iter in 0..max_iter {
            // Accumulate gradient-based autocorrelation
            let mut a11 = 0.0;
            let mut a12 = 0.0;
            let mut a22 = 0.0;
            let mut b1 = 0.0;
            let mut b2 = 0.0;

            for dy in -half..=half {
                for dx in -half..=half {
                    let px = (cx as i32 + dx).clamp(1, w as i32 - 2) as usize;
                    let py_coord = (cy as i32 + dy).clamp(1, h as i32 - 2) as usize;

                    let ix = gx[[py_coord, px]];
                    let iy = gy[[py_coord, px]];

                    a11 += ix * ix;
                    a12 += ix * iy;
                    a22 += iy * iy;

                    b1 += ix * ix * px as f64 + ix * iy * py_coord as f64;
                    b2 += ix * iy * px as f64 + iy * iy * py_coord as f64;
                }
            }

            let det = a11 * a22 - a12 * a12;
            if det.abs() < 1e-10 {
                break;
            }

            let new_cx = (a22 * b1 - a12 * b2) / det;
            let new_cy = (a11 * b2 - a12 * b1) / det;

            let shift = ((new_cx - cx).powi(2) + (new_cy - cy).powi(2)).sqrt();
            cx = new_cx;
            cy = new_cy;

            if shift < epsilon {
                break;
            }
        }

        refined[i] = (cx, cy);
    }

    Ok(refined)
}

// ==========================================
// HoughLinesP (Probabilistic Hough)
// ==========================================

/// hough_lines_p() - Probabilistic Hough Line Transform.
/// @image: 2D binary/edge input image (u8).
/// @rho: Distance resolution of the accumulator in pixels.
/// @theta_deg: Angle resolution of the accumulator in degrees.
/// @threshold: Accumulator threshold parameter.
/// @min_line_length: Minimum line length.
/// @max_line_gap: Maximum allowed gap between line segments to link them.
///
/// Finds line segments in a binary image using the probabilistic Hough transform.
///
/// Return: Vector of line segment tuples (x1, y1, x2, y2).
#[pyfunction]
#[pyo3(signature = (image, rho = 1.0, theta_deg = 1.0, threshold = 50, min_line_length = 30.0, max_line_gap = 10.0))]
pub fn hough_lines_p(
    image: PyReadonlyArrayDyn<'_, u8>,
    rho: f64,
    theta_deg: f64,
    threshold: u32,
    min_line_length: f64,
    max_line_gap: f64,
) -> PyResult<Vec<(i32, i32, i32, i32)>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Requires 2D binary/edge image"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);

    let max_rho = ((h as f64).powi(2) + (w as f64).powi(2)).sqrt().ceil() as usize;
    let rho_offset = max_rho;
    let theta_step = theta_deg.to_radians();
    let num_thetas = (std::f64::consts::PI / theta_step).ceil() as usize;

    // Precompute sin/cos
    let mut cos_t = Vec::with_capacity(num_thetas);
    let mut sin_t = Vec::with_capacity(num_thetas);
    for i in 0..num_thetas {
        let t = i as f64 * theta_step;
        cos_t.push(t.cos());
        sin_t.push(t.sin());
    }

    // Accumulator
    let mut accumulator = vec![vec![0u32; num_thetas]; max_rho * 2 + 1];

    // Collect edge pixels
    let mut edge_pixels: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if img_2d[[y, x]] > 0 {
                edge_pixels.push((y, x));
            }
        }
    }

    // Vote
    for &(y, x) in &edge_pixels {
        for i in 0..num_thetas {
            let r = (x as f64 * cos_t[i] + y as f64 * sin_t[i]) / rho;
            let r_idx = (r.round() as isize + rho_offset as isize) as usize;
            if r_idx < accumulator.len() {
                accumulator[r_idx][i] += 1;
            }
        }
    }

    // Extract line segments from peaks
    let mut lines = Vec::new();
    let mut used = vec![vec![false; w]; h];

    for r_idx in 0..(max_rho * 2 + 1) {
        for t_idx in 0..num_thetas {
            if accumulator[r_idx][t_idx] < threshold {
                continue;
            }

            let actual_rho = (r_idx as f64 - rho_offset as f64) * rho;
            let ct = cos_t[t_idx];
            let st = sin_t[t_idx];

            // Walk along the line and find segments
            let mut points: Vec<(i32, i32)> = Vec::new();

            if st.abs() > ct.abs() {
                // More horizontal: iterate over x
                for x in 0..w as i32 {
                    let y = ((actual_rho - x as f64 * ct) / st).round() as i32;
                    if y >= 0 && y < h as i32 && img_2d[[y as usize, x as usize]] > 0 && !used[y as usize][x as usize] {
                        points.push((x, y));
                    }
                }
            } else {
                // More vertical: iterate over y
                for y in 0..h as i32 {
                    let x = ((actual_rho - y as f64 * st) / ct).round() as i32;
                    if x >= 0 && x < w as i32 && img_2d[[y as usize, x as usize]] > 0 && !used[y as usize][x as usize] {
                        points.push((x, y));
                    }
                }
            }

            // Group consecutive points into segments
            if points.len() < 2 {
                continue;
            }

            let mut seg_start = 0;
            for j in 1..points.len() {
                let dx = (points[j].0 - points[j - 1].0) as f64;
                let dy = (points[j].1 - points[j - 1].1) as f64;
                let gap = (dx * dx + dy * dy).sqrt();

                if gap > max_line_gap {
                    // Check if segment is long enough
                    let sx = (points[seg_start].0 - points[j - 1].0) as f64;
                    let sy = (points[seg_start].1 - points[j - 1].1) as f64;
                    let length = (sx * sx + sy * sy).sqrt();
                    if length >= min_line_length {
                        lines.push((points[seg_start].0, points[seg_start].1, points[j - 1].0, points[j - 1].1));
                        for k in seg_start..j {
                            let (px, py) = points[k];
                            if py >= 0 && (py as usize) < h && px >= 0 && (px as usize) < w {
                                used[py as usize][px as usize] = true;
                            }
                        }
                    }
                    seg_start = j;
                }
            }

            // Final segment
            let sx = (points[seg_start].0 - points[points.len() - 1].0) as f64;
            let sy = (points[seg_start].1 - points[points.len() - 1].1) as f64;
            let length = (sx * sx + sy * sy).sqrt();
            if length >= min_line_length {
                let last = points.len() - 1;
                lines.push((points[seg_start].0, points[seg_start].1, points[last].0, points[last].1));
                for k in seg_start..=last {
                    let (px, py) = points[k];
                    if py >= 0 && (py as usize) < h && px >= 0 && (px as usize) < w {
                        used[py as usize][px as usize] = true;
                    }
                }
            }
        }
    }

    Ok(lines)
}

// ==========================================
// createLineSegmentDetector (LSD)
// ==========================================

/// line_segment_detector() - Line Segment Detector using gradient-based region growing.
/// @image: 2D grayscale input image (u8).
/// @scale: Scale of the image that will be used in detection.
/// @sigma_scale: Sigma value for Gaussian filter.
/// @ang_th: Gradient angle tolerance threshold in degrees.
/// @density_th: Log-density threshold.
/// @n_bins: Number of bins in gradient histogram.
///
/// Detects straight line segments in the grayscale image.
///
/// Return: Vector of line segments as (x1, y1, x2, y2, width) tuples.
#[pyfunction]
#[pyo3(signature = (image, scale = 0.8, sigma_scale = 0.6, ang_th = 22.5, density_th = 0.7, n_bins = 1024))]
pub fn line_segment_detector(
    image: PyReadonlyArrayDyn<'_, u8>,
    scale: f64,
    sigma_scale: f64,
    ang_th: f64,
    density_th: f64,
    n_bins: usize,
) -> PyResult<Vec<(f64, f64, f64, f64, f64)>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);

    // Compute gradients
    let mut gx = numpy::ndarray::Array2::<f64>::zeros((h, w));
    let mut gy = numpy::ndarray::Array2::<f64>::zeros((h, w));
    let mut mag = numpy::ndarray::Array2::<f64>::zeros((h, w));
    let mut angle = numpy::ndarray::Array2::<f64>::zeros((h, w));

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let dx = img_2d[[y, x + 1]] as f64 - img_2d[[y, x.saturating_sub(1)]] as f64;
            let dy = img_2d[[y + 1, x]] as f64 - img_2d[[y.saturating_sub(1), x]] as f64;
            gx[[y, x]] = dx;
            gy[[y, x]] = dy;
            mag[[y, x]] = (dx * dx + dy * dy).sqrt();
            angle[[y, x]] = dy.atan2(dx);
        }
    }

    // Sort pixels by gradient magnitude (descending)
    let mut pixels: Vec<(usize, usize, f64)> = Vec::new();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if mag[[y, x]] > 5.0 {
                pixels.push((y, x, mag[[y, x]]));
            }
        }
    }
    pixels.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut used = vec![vec![false; w]; h];
    let mut segments = Vec::new();
    let ang_th_rad = ang_th.to_radians();

    for &(sy, sx, _) in &pixels {
        if used[sy][sx] {
            continue;
        }

        // Region growing
        let seed_angle = angle[[sy, sx]];
        let mut region: Vec<(usize, usize)> = Vec::new();
        let mut stack = vec![(sy, sx)];

        while let Some((cy, cx)) = stack.pop() {
            if cy >= h || cx >= w || used[cy][cx] {
                continue;
            }
            let diff = (angle[[cy, cx]] - seed_angle).abs();
            let diff = if diff > std::f64::consts::PI { 2.0 * std::f64::consts::PI - diff } else { diff };
            if diff > ang_th_rad || mag[[cy, cx]] < 2.0 {
                continue;
            }

            used[cy][cx] = true;
            region.push((cy, cx));

            if cy > 0 { stack.push((cy - 1, cx)); }
            if cy + 1 < h { stack.push((cy + 1, cx)); }
            if cx > 0 { stack.push((cy, cx - 1)); }
            if cx + 1 < w { stack.push((cy, cx + 1)); }
        }

        if region.len() < 5 {
            continue;
        }

        // Fit line to region using PCA
        let n = region.len() as f64;
        let mean_x: f64 = region.iter().map(|&(_, x)| x as f64).sum::<f64>() / n;
        let mean_y: f64 = region.iter().map(|&(y, _)| y as f64).sum::<f64>() / n;

        let mut cov_xx = 0.0;
        let mut cov_xy = 0.0;
        let mut cov_yy = 0.0;
        for &(ry, rx) in &region {
            let dx = rx as f64 - mean_x;
            let dy = ry as f64 - mean_y;
            cov_xx += dx * dx;
            cov_xy += dx * dy;
            cov_yy += dy * dy;
        }

        // Principal direction
        let theta = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy);
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Project points onto principal axis
        let mut min_proj = f64::MAX;
        let mut max_proj = f64::MIN;
        let mut total_perp = 0.0;
        for &(ry, rx) in &region {
            let dx = rx as f64 - mean_x;
            let dy = ry as f64 - mean_y;
            let proj = dx * cos_t + dy * sin_t;
            let perp = (-dx * sin_t + dy * cos_t).abs();
            if proj < min_proj { min_proj = proj; }
            if proj > max_proj { max_proj = proj; }
            total_perp += perp;
        }

        let length = max_proj - min_proj;
        if length < 5.0 {
            continue;
        }

        let width = 2.0 * total_perp / n;
        let x1 = mean_x + min_proj * cos_t;
        let y1 = mean_y + min_proj * sin_t;
        let x2 = mean_x + max_proj * cos_t;
        let y2 = mean_y + max_proj * sin_t;

        segments.push((x1, y1, x2, y2, width));
    }

    Ok(segments)
}

// ==========================================
// cornerEigenValsAndVecs
// ==========================================

/// Compute eigenvalues and eigenvectors at each pixel from the structure tensor.
/// Returns (lambda1, lambda2, x1, y1, x2, y2) per pixel as a 6-channel float image.
#[pyfunction]
#[pyo3(signature = (image, block_size = 3, aperture_size = 3))]
pub fn corner_eigen_vals_and_vecs<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    block_size: usize,
    aperture_size: usize,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);

    let (sxx, syy, sxy) = helpers::compute_structure_tensor(&img_2d, block_size);

    let mut result = numpy::ndarray::Array3::<f32>::zeros((h, w, 6));

    for y in 0..h {
        for x in 0..w {
            let a = sxx[[y, x]];
            let b = sxy[[y, x]];
            let c = syy[[y, x]];

            let trace = a + c;
            let det = a * c - b * b;
            let disc = ((trace * trace) / 4.0 - det).max(0.0).sqrt();

            let lambda1 = trace / 2.0 + disc;
            let lambda2 = trace / 2.0 - disc;

            // Eigenvectors
            let (vx1, vy1, vx2, vy2) = if b.abs() > 1e-10 {
                let v1_x = lambda1 - c;
                let v1_y = b;
                let len1 = (v1_x * v1_x + v1_y * v1_y).sqrt().max(1e-10);
                let v2_x = lambda2 - c;
                let v2_y = b;
                let len2 = (v2_x * v2_x + v2_y * v2_y).sqrt().max(1e-10);
                (v1_x / len1, v1_y / len1, v2_x / len2, v2_y / len2)
            } else {
                (1.0, 0.0, 0.0, 1.0)
            };

            result[[y, x, 0]] = lambda1;
            result[[y, x, 1]] = lambda2;
            result[[y, x, 2]] = vx1;
            result[[y, x, 3]] = vy1;
            result[[y, x, 4]] = vx2;
            result[[y, x, 5]] = vy2;
        }
    }

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// preCornerDetect
// ==========================================

/// Pre-corner detection function.
/// Computes Dx^2 * Dyy + Dy^2 * Dxx - 2 * Dx * Dy * Dxy
#[pyfunction]
pub fn pre_corner_detect<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);

    let mut result = numpy::ndarray::Array2::<f32>::zeros((h, w));

    for y in 2..h.saturating_sub(2) {
        for x in 2..w.saturating_sub(2) {
            let dx = (img_2d[[y, x + 1]] as f32 - img_2d[[y, x - 1]] as f32) / 2.0;
            let dy = (img_2d[[y + 1, x]] as f32 - img_2d[[y - 1, x]] as f32) / 2.0;
            let dxx = img_2d[[y, x + 1]] as f32 - 2.0 * img_2d[[y, x]] as f32 + img_2d[[y, x - 1]] as f32;
            let dyy = img_2d[[y + 1, x]] as f32 - 2.0 * img_2d[[y, x]] as f32 + img_2d[[y - 1, x]] as f32;
            let dxy = (img_2d[[y + 1, x + 1]] as f32 - img_2d[[y + 1, x - 1]] as f32
                - img_2d[[y - 1, x + 1]] as f32 + img_2d[[y - 1, x - 1]] as f32) / 4.0;

            result[[y, x]] = dx * dx * dyy + dy * dy * dxx - 2.0 * dx * dy * dxy;
        }
    }

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// integral (summed area table)
// ==========================================

/// Compute integral image (summed area table).
/// Returns accumulated sum at each pixel position.
#[pyfunction]
pub fn integral<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);

    // Output is (h+1) x (w+1) with first row/col being zeros (matches OpenCV)
    let mut result = numpy::ndarray::Array2::<f64>::zeros((h + 1, w + 1));

    for y in 0..h {
        for x in 0..w {
            result[[y + 1, x + 1]] = img_2d[[y, x]] as f64
                + result[[y, x + 1]]
                + result[[y + 1, x]]
                - result[[y, x]];
        }
    }

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// sqrBoxFilter
// ==========================================

/// Squared box filter (unnormalized or normalized sum of squared pixel values).
/// Useful for computing local variance.
#[pyfunction]
#[pyo3(signature = (image, ksize = 3, normalize = true))]
pub fn sqr_box_filter<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    ksize: usize,
    normalize: bool,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    let pad = ksize / 2;
    let area = (ksize * ksize) as f64;

    let mut result = numpy::ndarray::Array2::<f64>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for ky in 0..ksize {
                for kx in 0..ksize {
                    let iy = (y as isize + ky as isize - pad as isize).clamp(0, h as isize - 1) as usize;
                    let ix = (x as isize + kx as isize - pad as isize).clamp(0, w as isize - 1) as usize;
                    let v = img_2d[[iy, ix]] as f64;
                    sum += v * v;
                }
            }
            result[[y, x]] = if normalize { sum / area } else { sum };
        }
    }

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// sepFilter2D
// ==========================================

/// Separable 2D filter: applies row kernel then column kernel.
/// This is faster than a full 2D convolution when the kernel is separable.
#[pyfunction]
pub fn sep_filter_2d<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    kernel_x: PyReadonlyArrayDyn<'py, f64>,
    kernel_y: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);

    let kx_arr = kernel_x.as_array();
    let ky_arr = kernel_y.as_array();
    let kx_1d: Vec<f64> = kx_arr.iter().cloned().collect();
    let ky_1d: Vec<f64> = ky_arr.iter().cloned().collect();

    let kx_len = kx_1d.len();
    let ky_len = ky_1d.len();
    let kx_pad = kx_len / 2;
    let ky_pad = ky_len / 2;

    // First pass: horizontal convolution
    let mut temp = numpy::ndarray::Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in 0..kx_len {
                let ix = (x as isize + k as isize - kx_pad as isize).clamp(0, w as isize - 1) as usize;
                sum += img_2d[[y, ix]] as f64 * kx_1d[k];
            }
            temp[[y, x]] = sum;
        }
    }

    // Second pass: vertical convolution
    let mut result = numpy::ndarray::Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in 0..ky_len {
                let iy = (y as isize + k as isize - ky_pad as isize).clamp(0, h as isize - 1) as usize;
                sum += temp[[iy, x]] * ky_1d[k];
            }
            result[[y, x]] = sum.clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// getGaborKernel
// ==========================================

/// Generate a Gabor filter kernel.
#[pyfunction]
#[pyo3(signature = (ksize, sigma, theta, lambd, gamma, psi = 0.0))]
pub fn get_gabor_kernel<'py>(
    py: Python<'py>,
    ksize: usize,
    sigma: f64,
    theta: f64,
    lambd: f64,
    gamma: f64,
    psi: f64,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let mut kernel = numpy::ndarray::Array2::<f64>::zeros((ksize, ksize));
    let center = (ksize as f64 - 1.0) / 2.0;
    let sigma_sq = sigma * sigma;
    let gamma_sq = gamma * gamma;

    for y in 0..ksize {
        for x in 0..ksize {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            let x_theta = dx * theta.cos() + dy * theta.sin();
            let y_theta = -dx * theta.sin() + dy * theta.cos();

            let exp_part = -(x_theta * x_theta + gamma_sq * y_theta * y_theta) / (2.0 * sigma_sq);
            let cos_part = (2.0 * std::f64::consts::PI * x_theta / lambd + psi).cos();

            kernel[[y, x]] = exp_part.exp() * cos_part;
        }
    }

    Ok(kernel.into_dyn().into_pyarray_bound(py).unbind())
}

// ==========================================
// GaussianBlur with configurable border modes
// ==========================================

/// Gaussian blur with configurable border handling.
/// border_type: 0=REFLECT_101 (default), 1=REFLECT, 2=REPLICATE (clamp), 3=WRAP, 4=CONSTANT
#[pyfunction]
#[pyo3(signature = (image, ksize, sigma, border_type = 0, border_value = 0))]
pub fn gaussian_blur_border<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    ksize: usize,
    sigma: f64,
    border_type: i32,
    border_value: u8,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    let pad = ksize / 2;

    // Build Gaussian kernel
    let center = pad as f64;
    let s2 = 2.0 * sigma * sigma;
    let mut gk = numpy::ndarray::Array2::<f64>::zeros((ksize, ksize));
    let mut gsum = 0.0;
    for y in 0..ksize {
        for x in 0..ksize {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            let val = (-(dx * dx + dy * dy) / s2).exp();
            gk[[y, x]] = val;
            gsum += val;
        }
    }
    gk.mapv_inplace(|v| v / gsum);

    // Border pixel access
    let get_pixel = |y: isize, x: isize| -> u8 {
        let h_i = h as isize;
        let w_i = w as isize;
        let (iy, ix) = match border_type {
            0 => {
                // REFLECT_101 (gfedcb|abcdefgh|gfedcba)
                let iy = if y < 0 { -y } else if y >= h_i { 2 * h_i - y - 2 } else { y };
                let ix = if x < 0 { -x } else if x >= w_i { 2 * w_i - x - 2 } else { x };
                (iy.clamp(0, h_i - 1), ix.clamp(0, w_i - 1))
            }
            1 => {
                // REFLECT (fedcba|abcdefgh|hgfedcb)
                let iy = if y < 0 { -y - 1 } else if y >= h_i { 2 * h_i - y - 1 } else { y };
                let ix = if x < 0 { -x - 1 } else if x >= w_i { 2 * w_i - x - 1 } else { x };
                (iy.clamp(0, h_i - 1), ix.clamp(0, w_i - 1))
            }
            2 => {
                // REPLICATE (clamp)
                (y.clamp(0, h_i - 1), x.clamp(0, w_i - 1))
            }
            3 => {
                // WRAP
                let iy = ((y % h_i) + h_i) % h_i;
                let ix = ((x % w_i) + w_i) % w_i;
                (iy, ix)
            }
            4 => {
                // CONSTANT
                if y < 0 || y >= h_i || x < 0 || x >= w_i {
                    return border_value;
                }
                (y, x)
            }
            _ => (y.clamp(0, h_i - 1), x.clamp(0, w_i - 1)),
        };
        img_2d[[iy as usize, ix as usize]]
    };

    let mut result = numpy::ndarray::Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for ky in 0..ksize {
                for kx in 0..ksize {
                    let iy = y as isize + ky as isize - pad as isize;
                    let ix = x as isize + kx as isize - pad as isize;
                    sum += get_pixel(iy, ix) as f64 * gk[[ky, kx]];
                }
            }
            result[[y, x]] = sum.round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result.into_dyn().into_pyarray_bound(py).unbind())
}
