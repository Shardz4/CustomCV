use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3},
    IntoPyArray, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn,
};
use num_complex::Complex64;

// ==========================================
// POINT TRANSFORMATIONS
// ==========================================

#[pyfunction]
pub fn apply_negative<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let x_array = x.as_array();
    let result = x_array.mapv(|pixel| 255 - pixel);
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn apply_log<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let x_array = x.as_array();
    let c = 255.0 / (256.0 as f64).ln();
    let result = x_array.mapv(|pixel| {
        let val = c * (pixel as f64 + 1.0).ln();
        val.min(255.0) as u8
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn apply_gamma<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, gamma: f64) -> PyResult<&'py PyArrayDyn<u8>> {
    let x_array = x.as_array();
    let result = x_array.mapv(|pixel| {
        let r = pixel as f64 / 255.0;
        let s = r.powf(gamma);
        (s * 255.0).min(255.0) as u8
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn rgb_to_gray<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let r = arr.slice(s![.., .., 0]).mapv(|v| v as f32);
    let g = arr.slice(s![.., .., 1]).mapv(|v| v as f32);
    let b = arr.slice(s![.., .., 2]).mapv(|v| v as f32);
    let gray = (0.299 * &r + 0.587 * &g + 0.114 * &b).mapv(|v| v as u8);
    Ok(gray.into_pyarray(py).to_dyn())
}

#[pyfunction]
pub fn apply_threshold<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, threshold_value: u8) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let result = arr.mapv(|pixel| {
        if pixel > threshold_value { 255 } else { 0 }
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn rgb_to_cmy<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<f32>> {
    let arr = x.as_array();
    
    // Ensure the input is a 3D RGB image
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must be an RGB image with shape (H, W, 3)"));
    }

    // Convert to CMY floats [0.0, 1.0]
    let result = arr.mapv(|pixel| 1.0 - (pixel as f32 / 255.0));

    Ok(result.into_pyarray(py).to_dyn())
}

// ==========================================
// FREQUENCY DOMAIN FILTER
// ==========================================

#[pyfunction]
pub fn apply_frequency_filter<'py>(py: Python<'py>, f_shifted: PyReadonlyArray3<'py, Complex64>, d0: f64, filter_type: &str) -> &'py PyArray3<Complex64> {
    let f_arr = f_shifted.as_array();
    let (rows, cols, channels) = (f_arr.shape()[0], f_arr.shape()[1], f_arr.shape()[2]);
    let mut output = Array3::<Complex64>::zeros((rows, cols, channels));
    let center_u = (rows as f64) / 2.0;
    let center_v = (cols as f64) / 2.0;

    for u in 0..rows {
        for v in 0..cols {
            let du = (u as f64) - center_u;
            let dv = (v as f64) - center_v;
            let d = (du * du + dv * dv).sqrt();

            let mask_val = match filter_type {
                "ILPF" => if d <= d0 { 1.0 } else { 0.0 },
                "IHPF" => if d <= d0 { 0.0 } else { 1.0 },
                "GLPF" => (-(d * d) / (2.0 * d0 * d0)).exp(),
                "GHPF" => 1.0 - (-(d * d) / (2.0 * d0 * d0)).exp(),
                _ => 1.0, 
            };

            let complex_mask = Complex64::new(mask_val, 0.0);
            for c in 0..channels {
                output[[u, v, c]] = f_arr[[u, v, c]] * complex_mask;
            }
        }
    }
    output.into_pyarray(py)
}
