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
