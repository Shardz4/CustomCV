use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3},
    IntoPyArray, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn,
};
use num_complex::Complex64;
use crate::helpers::{calculate_otsu_threshold, calculate_triangle_threshold};

// ==========================================
// POINT TRANSFORMATIONS
// ==========================================

/// apply_negative() - Compute image negative.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
///
/// Inverts the pixel values of the input image (255 - pixel).
///
/// Return: Inverted image array.
#[pyfunction]
pub fn apply_negative<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let x_array = x.as_array();
    let result = x_array.mapv(|pixel| 255 - pixel);
    Ok(result.into_pyarray(py))
}

/// apply_log() - Apply logarithmic transformation to an image.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
///
/// Compresses the dynamic range of an image using a log transform.
///
/// Return: Transformed image array.
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

/// apply_gamma() - Apply power-law (gamma) transformation to an image.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @gamma: Gamma exponent value.
///
/// Adjusts the brightness/contrast of an image using power-law curves.
///
/// Return: Transformed image array.
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

/// rgb_to_gray() - Convert an RGB image to grayscale.
/// @py: Python interpreter token.
/// @x: Input RGB image array (u8) of shape (H, W, 3).
///
/// Computes grayscale values using the NTSC/BT.601 luminance formula.
///
/// Return: Grayscale 2D image array.
#[pyfunction]
pub fn rgb_to_gray<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let r = arr.slice(s![.., .., 0]).mapv(|v| v as f32);
    let g = arr.slice(s![.., .., 1]).mapv(|v| v as f32);
    let b = arr.slice(s![.., .., 2]).mapv(|v| v as f32);
    let gray = (0.299 * &r + 0.587 * &g + 0.114 * &b).mapv(|v| v as u8);
    Ok(gray.into_pyarray(py).to_dyn())
}

/// apply_threshold() - Apply binary thresholding.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @threshold_value: Threshold value.
///
/// Sets pixels above the threshold value to 255 and all others to 0.
///
/// Return: Binary thresholded image array.
#[pyfunction]
pub fn apply_threshold<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, threshold_value: u8) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let result = arr.mapv(|pixel| {
        if pixel > threshold_value { 255 } else { 0 }
    });
    Ok(result.into_pyarray(py))
}

/// apply_threshold_binary_inv() - Apply inverse binary thresholding.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @threshold_value: Threshold value.
///
/// Sets pixels above the threshold value to 0 and all others to 255.
///
/// Return: Inverse binary thresholded image array.
#[pyfunction]
pub fn apply_threshold_binary_inv<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, threshold_value: u8) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let result = arr.mapv(|pixel| {
        if pixel > threshold_value { 0 } else { 255 }
    });
    Ok(result.into_pyarray(py))
}

/// apply_threshold_trunc() - Apply truncate thresholding.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @threshold_value: Threshold value.
///
/// Sets pixels above the threshold value to the threshold value, leaving others unchanged.
///
/// Return: Thresholded image array.
#[pyfunction]
pub fn apply_threshold_trunc<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, threshold_value: u8) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let result = arr.mapv(|pixel| {
        if pixel > threshold_value { threshold_value } else { pixel }
    });
    Ok(result.into_pyarray(py))
}

/// apply_threshold_tozero() - Apply threshold to zero.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @threshold_value: Threshold value.
///
/// Sets pixels below or equal to the threshold value to 0, leaving others unchanged.
///
/// Return: Thresholded image array.
#[pyfunction]
pub fn apply_threshold_tozero<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, threshold_value: u8) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let result = arr.mapv(|pixel| {
        if pixel > threshold_value { pixel } else { 0 }
    });
    Ok(result.into_pyarray(py))
}

/// apply_threshold_tozero_inv() - Apply inverse threshold to zero.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @threshold_value: Threshold value.
///
/// Sets pixels above the threshold value to 0, leaving others unchanged.
///
/// Return: Thresholded image array.
#[pyfunction]
pub fn apply_threshold_tozero_inv<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, threshold_value: u8) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let result = arr.mapv(|pixel| {
        if pixel > threshold_value { 0 } else { pixel }
    });
    Ok(result.into_pyarray(py))
}

/// Triangle automatic threshold.
/// Computes the optimal threshold using the Triangle algorithm, then applies binary thresholding.
/// Returns (threshold_value, thresholded_image).
#[pyfunction]
pub fn apply_threshold_triangle<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<(u8, &'py PyArrayDyn<u8>)> {
    let arr = x.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Grayscale (2D) image required"))?;
    let thresh = calculate_triangle_threshold(channel.view());
    let result = channel.mapv(|pixel| if pixel > thresh { 255 } else { 0 });
    Ok((thresh, result.into_pyarray(py).to_dyn()))
}

/// Otsu threshold combined with any threshold mode.
/// Auto-computes the optimal threshold using Otsu's method, then applies the given mode.
/// Supported modes: "binary", "binary_inv", "trunc", "tozero", "tozero_inv".
/// Returns (threshold_value, thresholded_image).
#[pyfunction]
pub fn apply_otsu_with_mode<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>, mode: &str) -> PyResult<(u8, &'py PyArrayDyn<u8>)> {
    let arr = x.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Grayscale (2D) image required"))?;
    let thresh = calculate_otsu_threshold(channel.view());
    let result = match mode {
        "binary"     => channel.mapv(|p| if p > thresh { 255 } else { 0 }),
        "binary_inv" => channel.mapv(|p| if p > thresh { 0 } else { 255 }),
        "trunc"      => channel.mapv(|p| if p > thresh { thresh } else { p }),
        "tozero"     => channel.mapv(|p| if p > thresh { p } else { 0 }),
        "tozero_inv" => channel.mapv(|p| if p > thresh { 0 } else { p }),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            "Unknown mode. Use: binary, binary_inv, trunc, tozero, tozero_inv"
        )),
    };
    Ok((thresh, result.into_pyarray(py).to_dyn()))
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
