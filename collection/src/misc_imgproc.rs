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
