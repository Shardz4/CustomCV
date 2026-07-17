use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use numpy::ndarray::{Array2, ArrayView2, s};
use crate::helpers;

// ==========================================
// getStructuringElement
// ==========================================

/// Create morphological structuring elements.
/// shape: 0 = MORPH_RECT, 1 = MORPH_CROSS, 2 = MORPH_ELLIPSE
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

/// Unified morphological operations entry point.
/// op: 0=ERODE, 1=DILATE, 2=OPEN, 3=CLOSE, 4=GRADIENT, 5=TOPHAT, 6=BLACKHAT
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

/// Canny edge detection with L2 gradient norm option.
/// When l2_gradient is true, uses L2 norm (sqrt(gx^2+gy^2)) instead of L1 norm (|gx|+|gy|).
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
