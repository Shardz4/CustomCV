use pyo3::prelude::*;
use numpy::{
    ndarray::Array3,
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};

// ==========================================
// INTERNAL HELPERS
// ==========================================

/// Linearize an sRGB component from [0,1] gamma-encoded to [0,1] linear.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// CIE Lab f(t) nonlinear mapping.
fn lab_f(t: f64) -> f64 {
    const EPSILON: f64 = 0.008856; // (6/29)^3
    if t > EPSILON {
        t.cbrt()
    } else {
        7.787037 * t + 16.0 / 116.0
    }
}

// D65 standard illuminant white-point tristimulus values
const XN: f64 = 0.95047;
const YN: f64 = 1.00000;
const ZN: f64 = 1.08883;

// ==========================================
// COLOR SPACE CONVERSION FUNCTIONS
// ==========================================

/// rgb_to_hsv() - Convert RGB to HSV color space.
/// @py: Python interpreter token.
/// @x: Input 3D RGB image array (u8).
///
/// Converts a 3D RGB image array to the HSV color space.
/// The output has H ∈ [0, 180], S ∈ [0, 255], and V ∈ [0, 255].
/// Hue is halved (0–360° mapped to 0–180) to fit within a single u8 byte.
///
/// Return: A 3D PyArrayDyn containing HSV channels.
#[pyfunction]
pub fn rgb_to_hsv<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D RGB image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<u8>::zeros((rows, cols, 3));

    for row in 0..rows {
        for col in 0..cols {
            let r = arr[[row, col, 0]] as f64 / 255.0;
            let g = arr[[row, col, 1]] as f64 / 255.0;
            let b = arr[[row, col, 2]] as f64 / 255.0;
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            let delta = cmax - cmin;

            let hue = if delta < 1e-10 {
                0.0
            } else if (cmax - r).abs() < 1e-10 {
                60.0 * (((g - b) / delta) % 6.0)
            } else if (cmax - g).abs() < 1e-10 {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            let hue = if hue < 0.0 { hue + 360.0 } else { hue };
            let sat = if cmax < 1e-10 { 0.0 } else { delta / cmax };

            out[[row, col, 0]] = (hue / 2.0).round().clamp(0.0, 180.0) as u8;
            out[[row, col, 1]] = (sat * 255.0).round().clamp(0.0, 255.0) as u8;
            out[[row, col, 2]] = (cmax * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// rgb_to_hls() - Convert RGB to HLS color space.
/// @py: Python interpreter token.
/// @x: Input 3D RGB image array (u8).
///
/// Converts a 3D RGB image array to the HLS (Hue-Lightness-Saturation) color space.
/// The output has H ∈ [0, 180], L ∈ [0, 255], and S ∈ [0, 255].
/// Hue is halved to fit in a single byte.
///
/// Return: A 3D PyArrayDyn containing HLS channels.
#[pyfunction]
pub fn rgb_to_hls<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D RGB image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<u8>::zeros((rows, cols, 3));

    for row in 0..rows {
        for col in 0..cols {
            let r = arr[[row, col, 0]] as f64 / 255.0;
            let g = arr[[row, col, 1]] as f64 / 255.0;
            let b = arr[[row, col, 2]] as f64 / 255.0;
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            let delta = cmax - cmin;
            let light = (cmax + cmin) / 2.0;

            let hue = if delta < 1e-10 {
                0.0
            } else if (cmax - r).abs() < 1e-10 {
                60.0 * (((g - b) / delta) % 6.0)
            } else if (cmax - g).abs() < 1e-10 {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            let hue = if hue < 0.0 { hue + 360.0 } else { hue };

            let sat = if delta < 1e-10 {
                0.0
            } else {
                let denom = 1.0 - (2.0 * light - 1.0).abs();
                if denom < 1e-10 { 0.0 } else { (delta / denom).min(1.0) }
            };

            out[[row, col, 0]] = (hue / 2.0).round().clamp(0.0, 180.0) as u8;
            out[[row, col, 1]] = (light * 255.0).round().clamp(0.0, 255.0) as u8;
            out[[row, col, 2]] = (sat * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// rgb_to_ycrcb() - Convert RGB to YCrCb color space.
/// @py: Python interpreter token.
/// @x: Input 3D RGB image array (u8).
///
/// Converts a 3D RGB image array to the YCrCb color space using BT.601 coefficients.
/// The chroma components Cr and Cb are offset by 128 to stay within [0, 255] u8 range.
///
/// Return: A 3D PyArrayDyn containing YCrCb channels.
#[pyfunction]
pub fn rgb_to_ycrcb<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D RGB image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<u8>::zeros((rows, cols, 3));

    for row in 0..rows {
        for col in 0..cols {
            let r = arr[[row, col, 0]] as f64;
            let g = arr[[row, col, 1]] as f64;
            let b = arr[[row, col, 2]] as f64;
            let y  =  0.299 * r + 0.587 * g + 0.114 * b;
            let cr =  0.500 * r - 0.419 * g - 0.081 * b + 128.0;
            let cb = -0.169 * r - 0.331 * g + 0.500 * b + 128.0;
            out[[row, col, 0]] = y.round().clamp(0.0, 255.0) as u8;
            out[[row, col, 1]] = cr.round().clamp(0.0, 255.0) as u8;
            out[[row, col, 2]] = cb.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// rgb_to_xyz() - Convert RGB to CIE XYZ color space.
/// @py: Python interpreter token.
/// @x: Input 3D RGB image array (u8).
///
/// Converts a 3D RGB image array to the CIE XYZ color space with D65 illuminant.
/// Applies sRGB inverse-gamma (linearization) prior to the matrix multiplication.
///
/// Return: A 3D f32 PyArrayDyn containing XYZ channels.
#[pyfunction]
pub fn rgb_to_xyz<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<f32>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D RGB image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<f32>::zeros((rows, cols, 3));

    for row in 0..rows {
        for col in 0..cols {
            let rl = srgb_to_linear(arr[[row, col, 0]] as f64 / 255.0);
            let gl = srgb_to_linear(arr[[row, col, 1]] as f64 / 255.0);
            let bl = srgb_to_linear(arr[[row, col, 2]] as f64 / 255.0);
            out[[row, col, 0]] = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) as f32;
            out[[row, col, 1]] = (0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl) as f32;
            out[[row, col, 2]] = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) as f32;
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// rgb_to_lab() - Convert RGB to CIE L*a*b* color space.
/// @py: Python interpreter token.
/// @x: Input 3D RGB image array (u8).
///
/// Converts a 3D RGB image array to the CIE L*a*b* color space under D65 illuminant.
/// L ∈ [0, 100], a ∈ [-128, 127], b ∈ [-128, 127].
///
/// Return: A 3D f32 PyArrayDyn containing Lab channels.
#[pyfunction]
pub fn rgb_to_lab<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<f32>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D RGB image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<f32>::zeros((rows, cols, 3));

    for row in 0..rows {
        for col in 0..cols {
            // Linearize sRGB
            let rl = srgb_to_linear(arr[[row, col, 0]] as f64 / 255.0);
            let gl = srgb_to_linear(arr[[row, col, 1]] as f64 / 255.0);
            let bl = srgb_to_linear(arr[[row, col, 2]] as f64 / 255.0);
            // sRGB → XYZ (D65)
            let xv = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
            let yv = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
            let zv = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;
            // XYZ → Lab
            let fx = lab_f(xv / XN);
            let fy = lab_f(yv / YN);
            let fz = lab_f(zv / ZN);
            out[[row, col, 0]] = (116.0 * fy - 16.0) as f32;   // L
            out[[row, col, 1]] = (500.0 * (fx - fy)) as f32;    // a
            out[[row, col, 2]] = (200.0 * (fy - fz)) as f32;    // b
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// rgb_to_luv() - Convert RGB to CIE L*u*v* color space.
/// @py: Python interpreter token.
/// @x: Input 3D RGB image array (u8).
///
/// Converts a 3D RGB image array to the CIE L*u*v* color space under D65 illuminant.
/// L ∈ [0, 100]; u and v can be negative.
///
/// Return: A 3D f32 PyArrayDyn containing Luv channels.
#[pyfunction]
pub fn rgb_to_luv<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<f32>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D RGB image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<f32>::zeros((rows, cols, 3));

    // D65 reference chromaticity
    let un = 4.0 * XN / (XN + 15.0 * YN + 3.0 * ZN);
    let vn = 9.0 * YN / (XN + 15.0 * YN + 3.0 * ZN);

    for row in 0..rows {
        for col in 0..cols {
            let rl = srgb_to_linear(arr[[row, col, 0]] as f64 / 255.0);
            let gl = srgb_to_linear(arr[[row, col, 1]] as f64 / 255.0);
            let bl = srgb_to_linear(arr[[row, col, 2]] as f64 / 255.0);
            let xv = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
            let yv = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
            let zv = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;

            // L (same formula as Lab)
            let yr = yv / YN;
            let l_val = if yr > 0.008856 {
                116.0 * yr.cbrt() - 16.0
            } else {
                903.3 * yr
            };

            // u', v' chromaticity
            let denom = xv + 15.0 * yv + 3.0 * zv;
            let (u_val, v_val) = if denom.abs() < 1e-10 {
                (0.0, 0.0)
            } else {
                let u_prime = 4.0 * xv / denom;
                let v_prime = 9.0 * yv / denom;
                (13.0 * l_val * (u_prime - un), 13.0 * l_val * (v_prime - vn))
            };

            out[[row, col, 0]] = l_val as f32;
            out[[row, col, 1]] = u_val as f32;
            out[[row, col, 2]] = v_val as f32;
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// bgr_to_rgb() - Convert BGR to RGB color space.
/// @py: Python interpreter token.
/// @x: Input 3D BGR image array (u8).
///
/// Swaps channels 0 and 2 of a 3D BGR image array to convert it to RGB.
///
/// Return: A 3D u8 PyArrayDyn containing RGB channels.
#[pyfunction]
pub fn bgr_to_rgb<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<u8>::zeros((rows, cols, 3));
    for row in 0..rows {
        for col in 0..cols {
            out[[row, col, 0]] = arr[[row, col, 2]];
            out[[row, col, 1]] = arr[[row, col, 1]];
            out[[row, col, 2]] = arr[[row, col, 0]];
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// gray_to_rgb() - Convert Grayscale to RGB.
/// @py: Python interpreter token.
/// @x: Input 2D grayscale image array (u8).
///
/// Replicates the single channel of a 2D grayscale image three times to form an RGB image.
///
/// Return: A 3D u8 PyArrayDyn of shape (H, W, 3) representing the RGB image.
#[pyfunction]
pub fn gray_to_rgb<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 2D grayscale image with shape (H, W)"))?;
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<u8>::zeros((rows, cols, 3));
    for row in 0..rows {
        for col in 0..cols {
            let v = arr[[row, col]];
            out[[row, col, 0]] = v;
            out[[row, col, 1]] = v;
            out[[row, col, 2]] = v;
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// rgb_to_yuv() - Convert RGB to YUV color space.
/// @py: Python interpreter token.
/// @x: Input 3D RGB image array (u8).
///
/// Converts a 3D RGB image array to the YUV color space using BT.601 coefficients.
/// U and V are offset by 128 to stay within [0, 255] u8 range.
///
/// Return: A 3D u8 PyArrayDyn containing YUV channels.
#[pyfunction]
pub fn rgb_to_yuv<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array()
        .into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 3D RGB image"))?;
    if arr.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must have 3 channels (H, W, 3)"));
    }
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut out = Array3::<u8>::zeros((rows, cols, 3));

    for row in 0..rows {
        for col in 0..cols {
            let r = arr[[row, col, 0]] as f64;
            let g = arr[[row, col, 1]] as f64;
            let b = arr[[row, col, 2]] as f64;
            let y_val =  0.299 * r + 0.587 * g + 0.114 * b;
            let u_val = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0;
            let v_val =  0.615 * r - 0.51499 * g - 0.10001 * b + 128.0;
            out[[row, col, 0]] = y_val.round().clamp(0.0, 255.0) as u8;
            out[[row, col, 1]] = u_val.round().clamp(0.0, 255.0) as u8;
            out[[row, col, 2]] = v_val.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out.into_pyarray(py).to_dyn())
}

/// cvt_color() - Universal color-space converter dispatcher.
/// @py: Python interpreter token.
/// @image: Input image array (u8).
/// @code: Integer conversion code matching OpenCV color conversion constants.
///
/// Maps a standard OpenCV color conversion code to the corresponding internal
/// conversion function.  The supported codes and their mappings are:
///
///   - COLOR_BGR2GRAY  (6)  / COLOR_RGB2GRAY (7)  → rgb_to_gray
///   - COLOR_BGR2HSV   (40) / COLOR_RGB2HSV  (41) → rgb_to_hsv
///   - COLOR_BGR2HLS   (52) / COLOR_RGB2HLS  (53) → rgb_to_hls
///   - COLOR_BGR2YCrCb (36) / COLOR_RGB2YCrCb(37) → rgb_to_ycrcb
///   - COLOR_BGR2XYZ   (32) / COLOR_RGB2XYZ  (33) → rgb_to_xyz
///   - COLOR_BGR2Lab   (44) / COLOR_RGB2Lab  (45) → rgb_to_lab
///   - COLOR_BGR2Luv   (50) / COLOR_RGB2Luv  (51) → rgb_to_luv
///   - COLOR_BGR2RGB   (4)  / COLOR_RGB2BGR  (4)  → bgr_to_rgb
///   - COLOR_GRAY2BGR  (8)  / COLOR_GRAY2RGB (8)  → gray_to_rgb
///   - COLOR_BGR2YUV   (82) / COLOR_RGB2YUV  (83) → rgb_to_yuv
///   - COLOR_RGB2CMY   (128)                      → rgb_to_cmy (custom)
///
/// For BGR‐prefixed codes the input is first channel-swapped to RGB before
/// the conversion is applied.
///
/// Return: Converted image as a PyObject (array type depends on conversion).
#[pyfunction]
#[pyo3(name = "cvtColor")]
pub fn cvt_color<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    code: i32,
) -> PyResult<PyObject> {
    // BGR-prefixed codes require a channel swap first.
    // We handle pairs: (bgr_code, rgb_code) → same converter
    match code {
        // --- To Grayscale ---
        6 | 7 => {
            let input = if code == 6 { swap_rb(py, &image)? } else { image };
            let result = crate::transforms::rgb_to_gray(py, input)?;
            Ok(result.into_py(py))
        }
        // --- To HSV ---
        40 | 41 => {
            let input = if code == 40 { swap_rb(py, &image)? } else { image };
            let result = rgb_to_hsv(py, input)?;
            Ok(result.into_py(py))
        }
        // --- To HLS ---
        52 | 53 => {
            let input = if code == 52 { swap_rb(py, &image)? } else { image };
            let result = rgb_to_hls(py, input)?;
            Ok(result.into_py(py))
        }
        // --- To YCrCb ---
        36 | 37 => {
            let input = if code == 36 { swap_rb(py, &image)? } else { image };
            let result = rgb_to_ycrcb(py, input)?;
            Ok(result.into_py(py))
        }
        // --- To XYZ ---
        32 | 33 => {
            let input = if code == 32 { swap_rb(py, &image)? } else { image };
            let result = rgb_to_xyz(py, input)?;
            Ok(result.into_py(py))
        }
        // --- To Lab ---
        44 | 45 => {
            let input = if code == 44 { swap_rb(py, &image)? } else { image };
            let result = rgb_to_lab(py, input)?;
            Ok(result.into_py(py))
        }
        // --- To Luv ---
        50 | 51 => {
            let input = if code == 50 { swap_rb(py, &image)? } else { image };
            let result = rgb_to_luv(py, input)?;
            Ok(result.into_py(py))
        }
        // --- BGR ↔ RGB swap ---
        4 => {
            let result = bgr_to_rgb(py, image)?;
            Ok(result.into_py(py))
        }
        // --- Gray to BGR / RGB ---
        8 => {
            let result = gray_to_rgb(py, image)?;
            Ok(result.into_py(py))
        }
        // --- To YUV ---
        82 | 83 => {
            let input = if code == 82 { swap_rb(py, &image)? } else { image };
            let result = rgb_to_yuv(py, input)?;
            Ok(result.into_py(py))
        }
        // --- Custom: RGB to CMY (128) ---
        128 => {
            let result = crate::transforms::rgb_to_cmy(py, image)?;
            Ok(result.into_py(py))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported color conversion code: {}. See OpenCV COLOR_* constants.",
            code
        ))),
    }
}

/// swap_rb() - Swap the R and B channels of a 3-channel image.
/// @py: Python interpreter token.
/// @image: Reference to input 3D image array (u8).
///
/// Internal helper used by cvt_color to convert BGR input to RGB
/// before passing to the RGB-based converter functions.
///
/// Return: A readonly 3D PyReadonlyArrayDyn representing the RGB image.
fn swap_rb<'py>(
    py: Python<'py>,
    image: &PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<PyReadonlyArrayDyn<'py, u8>> {
    let arr = image.as_array();
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "BGR↔RGB swap requires a 3-channel image (H, W, 3)",
        ));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let mut out = Array3::<u8>::zeros((rows, cols, 3));
    for r in 0..rows {
        for c in 0..cols {
            out[[r, c, 0]] = arr[[r, c, 2]];
            out[[r, c, 1]] = arr[[r, c, 1]];
            out[[r, c, 2]] = arr[[r, c, 0]];
        }
    }
    let py_arr = out.into_pyarray(py).to_dyn();
    Ok(py_arr.readonly())
}
