use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array2, Array3},
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

/// apply_threshold_triangle() - Triangle automatic threshold.
/// @py: Python interpreter token.
/// @x: Input grayscale image array (u8).
///
/// Computes the optimal threshold using the Triangle algorithm, then applies binary thresholding.
///
/// Return: A tuple containing (computed threshold value, binary thresholded image array).
#[pyfunction]
pub fn apply_threshold_triangle<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<(u8, &'py PyArrayDyn<u8>)> {
    let arr = x.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Grayscale (2D) image required"))?;
    let thresh = calculate_triangle_threshold(channel.view());
    let result = channel.mapv(|pixel| if pixel > thresh { 255 } else { 0 });
    Ok((thresh, result.into_pyarray(py).to_dyn()))
}

/// apply_otsu_with_mode() - Otsu threshold combined with any threshold mode.
/// @py: Python interpreter token.
/// @x: Input grayscale image array (u8).
/// @mode: Thresholding mode ("binary", "binary_inv", "trunc", "tozero", "tozero_inv").
///
/// Auto-computes the optimal threshold using Otsu's method, then applies the given mode.
///
/// Return: A tuple containing (computed threshold value, thresholded image array).
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

/// rgb_to_cmy() - Convert an RGB image to CMY color space.
/// @py: Python interpreter token.
/// @x: Input RGB image array (u8) of shape (H, W, 3).
///
/// Converts the RGB pixel values to Cyan, Magenta, and Yellow (CMY) floating-point format [0.0, 1.0].
///
/// Return: CMY image array of shape (H, W, 3) (f32).
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

/// apply_frequency_filter() - Apply a frequency domain filter.
/// @py: Python interpreter token.
/// @f_shifted: Shifted 2D DFT spectrum (3D array of Complex64).
/// @d0: Cut-off frequency.
/// @filter_type: Type of filter ("ILPF", "IHPF", "GLPF", "GHPF").
///
/// Filters the shifted discrete Fourier transform spectrum in the frequency domain.
///
/// Return: Filtered frequency spectrum (Complex64).
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

/// adaptive_threshold() - Per-region adaptive thresholding.
/// @py: Python interpreter token.
/// @image: 2D single-channel grayscale input image (u8).
/// @max_value: Non-zero value assigned to pixels that pass the threshold test.
/// @adaptive_method: Adaptive thresholding algorithm to use.
///     - 0 (ADAPTIVE_THRESH_MEAN_C): threshold is the mean of the
///       block_size × block_size neighbourhood minus constant c.
///     - 1 (ADAPTIVE_THRESH_GAUSSIAN_C): threshold is a Gaussian-weighted
///       sum of the neighbourhood minus constant c.
/// @threshold_type: Type of thresholding to apply.
///     - 0 (THRESH_BINARY):     dst = if src > T { max_value } else { 0 }
///     - 1 (THRESH_BINARY_INV): dst = if src > T { 0 } else { max_value }
/// @block_size: Size of the pixel neighbourhood used to compute the
///     threshold value.  Must be an odd number >= 3.
/// @c: Constant subtracted from the computed mean / weighted mean before
///     the threshold comparison.  Can be positive or negative.
///
/// The function computes a per-pixel threshold by averaging the surrounding
/// block_size × block_size neighbourhood (mean or Gaussian-weighted), then
/// subtracting c.  Each pixel is compared against its own local threshold to
/// produce the binary output.
///
/// Context: Adaptive thresholding is superior to global thresholding when
/// illumination varies across the image, e.g. scanned documents or natural
/// scenes.
///
/// Return: Thresholded 2D image array (u8).
#[pyfunction]
pub fn adaptive_threshold<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    max_value: f64,
    adaptive_method: i32,
    threshold_type: i32,
    block_size: usize,
    c: f64,
) -> PyResult<&'py PyArrayDyn<u8>> {
    if block_size < 3 || block_size % 2 == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "block_size must be an odd number >= 3",
        ));
    }
    let arr = image.as_array();
    let arr2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(
            "adaptiveThreshold requires a 2D single-channel image",
        ))?;
    let (h, w) = (arr2d.shape()[0], arr2d.shape()[1]);
    let half = (block_size / 2) as isize;
    let max_val = max_value.round().clamp(0.0, 255.0) as u8;

    // Pre-compute Gaussian kernel weights if needed.
    let gauss_weights: Option<Vec<Vec<f64>>> = if adaptive_method == 1 {
        let sigma = 0.3 * ((block_size as f64 / 2.0) - 1.0) + 0.8;
        let mut kernel = vec![vec![0.0f64; block_size]; block_size];
        let mut sum = 0.0;
        for ky in 0..block_size {
            for kx in 0..block_size {
                let dy = ky as f64 - half as f64;
                let dx = kx as f64 - half as f64;
                let val = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                kernel[ky][kx] = val;
                sum += val;
            }
        }
        // Normalise
        for row in kernel.iter_mut() {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        Some(kernel)
    } else {
        None
    };

    let block_area = (block_size * block_size) as f64;
    let mut out = Array2::<u8>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let local_threshold = if adaptive_method == 0 {
                // ADAPTIVE_THRESH_MEAN_C: arithmetic mean of neighbourhood
                let mut sum = 0.0f64;
                for ky in -half..=half {
                    for kx in -half..=half {
                        let iy = (y as isize + ky).clamp(0, h as isize - 1) as usize;
                        let ix = (x as isize + kx).clamp(0, w as isize - 1) as usize;
                        sum += arr2d[[iy, ix]] as f64;
                    }
                }
                sum / block_area - c
            } else {
                // ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian-weighted mean
                let kernel = gauss_weights.as_ref().unwrap();
                let mut sum = 0.0f64;
                for (ky_i, ky) in (-half..=half).enumerate() {
                    for (kx_i, kx) in (-half..=half).enumerate() {
                        let iy = (y as isize + ky).clamp(0, h as isize - 1) as usize;
                        let ix = (x as isize + kx).clamp(0, w as isize - 1) as usize;
                        sum += arr2d[[iy, ix]] as f64 * kernel[ky_i][kx_i];
                    }
                }
                sum - c
            };

            let pixel = arr2d[[y, x]] as f64;
            out[[y, x]] = if threshold_type == 0 {
                // THRESH_BINARY
                if pixel > local_threshold { max_val } else { 0 }
            } else {
                // THRESH_BINARY_INV
                if pixel > local_threshold { 0 } else { max_val }
            };
        }
    }

    Ok(out.into_pyarray(py).to_dyn())
}
