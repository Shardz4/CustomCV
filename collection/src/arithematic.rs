use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3, Zip},
    IntoPyArray, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn,
};

/// add_images() - Saturating addition of two images.
/// @py: Python interpreter token.
/// @img1: First input image array.
/// @img2: Second input image array.
///
/// Performs per-element saturating addition of two input image arrays of shape (h, w, c) or (h, w).
/// The two input images must have the exact same shape. If the shapes differ,
/// a PyValueError is returned.
///
/// Return: A new PyArrayDyn containing the summed image pixels.
#[pyfunction]
pub fn add_images<'py>(py: Python<'py>, img1:PyReadonlyArrayDyn<'py,u8>,img2:PyReadonlyArrayDyn<'py,u8>)->PyResult<Py<PyArrayDyn<u8>>>{

    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the same shape"))
    }
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|r, a, b| { *r = a.saturating_add(*b); });
    
    Ok(result.into_pyarray_bound(py).into())   
}

/// sub_images() - Saturating subtraction of two images.
/// @py: Python interpreter token.
/// @img1: First input image array (minuend).
/// @img2: Second input image array (subtrahend).
///
/// Performs per-element saturating subtraction of two input image arrays.
/// The operation is computed as: result = saturating_sub(img1, img2).
/// The input images must have the exact same shape. If the shapes differ,
/// a PyValueError is returned.
///
/// Return: A new PyArrayDyn containing the subtracted image pixels.
#[pyfunction]
pub fn sub_images<'py>(py: Python<'py>, img1:PyReadonlyArrayDyn<'py,u8>,img2:PyReadonlyArrayDyn<'py,u8>)->PyResult<Py<PyArrayDyn<u8>>>{

    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the same shape"))
    }
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|r, a, b| { *r = a.saturating_sub(*b); });
    
    Ok(result.into_pyarray_bound(py).into())   
}

/// add_weighted() - Compute weighted sum of two images.
/// @py: Python interpreter token.
/// @img1: First input image array.
/// @alpha: Weight of the first image elements.
/// @img2: Second input image array.
/// @beta: Weight of the second image elements.
/// @gamma: Scalar added to each sum.
///
/// Computes the weighted sum of two arrays as follows:
/// result = clamp(img1 * alpha + img2 * beta + gamma, 0, 255)
/// The input images must have the exact same shape. If the shapes differ,
/// a PyValueError is returned.
///
/// Return: A new PyArrayDyn containing the weighted sum of elements.
#[pyfunction]
pub fn add_weighted<'py>(py: Python<'py>, img1: PyReadonlyArrayDyn<'py, u8>, alpha: f64, img2: PyReadonlyArrayDyn<'py, u8>, beta: f64, gamma: f64) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the exact same shape"));
    }
    
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|res, &a, &b| {
        let val = (a as f64 * alpha) + (b as f64 * beta) + gamma;
        *res = val.clamp(0.0, 255.0) as u8;
    });
    Ok(result.into_pyarray_bound(py).into())
}

/// bitwise_and() - Compute bitwise conjunction of two images.
/// @py: Python interpreter token.
/// @img1: First input image array.
/// @img2: Second input image array.
///
/// Computes the per-element bitwise logical AND of two input images.
/// The input images must have the exact same shape. If the shapes differ,
/// a PyValueError is returned.
///
/// Return: A new PyArrayDyn containing the bitwise ANDed elements.
#[pyfunction]
pub fn bitwise_and<'py>(py: Python<'py>, img1: PyReadonlyArrayDyn<'py,u8>,img2: PyReadonlyArrayDyn<'py,u8>)->PyResult<Py<PyArrayDyn<u8>>>{

    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the exact same shape"));
    }
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|r, a, b| { *r = *a & *b });
    Ok(result.into_pyarray_bound(py).into())   
}

/// bitwise_or() - Compute bitwise disjunction of two images.
/// @py: Python interpreter token.
/// @img1: First input image array.
/// @img2: Second input image array.
///
/// Computes the per-element bitwise logical OR of two input images.
/// The input images must have the exact same shape. If the shapes differ,
/// a PyValueError is returned.
///
/// Return: A new PyArrayDyn containing the bitwise ORed elements.
#[pyfunction]
pub fn bitwise_or<'py>(py: Python<'py>, img1: PyReadonlyArrayDyn<'py,u8>,img2: PyReadonlyArrayDyn<'py,u8>)->PyResult<Py<PyArrayDyn<u8>>>{

    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the exact same shape"));
    }
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|r, a, b| { *r = *a | *b });
    Ok(result.into_pyarray_bound(py).into())   
}

/// bitwise_not() - Compute bitwise inversion of an image.
/// @py: Python interpreter token.
/// @img: Input image array.
///
/// Computes the per-element bitwise logical NOT (inversion) of an image.
///
/// Return: A new PyArrayDyn containing the bitwise inverted elements.
#[pyfunction]
pub fn bitwise_not<'py>(py:Python<'py>,img:PyReadonlyArrayDyn<'py,u8>)->PyResult<Py<PyArrayDyn<u8>>>{
    let arr = img.as_array();
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr.shape());
    Zip::from(&mut result).and(&arr).for_each(|r, a| { *r = !*a });
    Ok(result.into_pyarray_bound(py).into())   
}

/// bitwise_xor() - Compute bitwise exclusive-OR of two images.
/// @py: Python interpreter token.
/// @img1: First input image array.
/// @img2: Second input image array.
///
/// Computes the per-element bitwise logical XOR of two input images.
/// The input images must have the exact same shape. If the shapes differ,
/// a PyValueError is returned.
///
/// Return: A new PyArrayDyn containing the bitwise XORed elements.
#[pyfunction]
pub fn bitwise_xor<'py>(py:Python<'py>,img1:PyReadonlyArrayDyn<'py,u8>,img2:PyReadonlyArrayDyn<'py,u8>)->PyResult<Py<PyArrayDyn<u8>>>{

    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the exact same shape"));
    }
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|r, a, b| { *r = *a ^ *b });
    Ok(result.into_pyarray_bound(py).into())   
}

/// extract_f64_array() - Extract array as f64 ndarray.
/// @arr_py: PyAny object representing the source array.
///
/// Extracts the dynamic dimensional array as an f64 ndarray.
/// It converts and supports u8, i32, f32, or f64 source data types.
/// If the data type is not supported, a PyTypeError is returned.
///
/// Return: A 64-bit float dynamic-dimensional ndarray on success, or PyResult error.
fn extract_f64_array<'py>(arr_py: &'py pyo3::PyAny) -> PyResult<numpy::ndarray::ArrayD<f64>> {
    if let Ok(arr) = arr_py.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(arr.as_array().to_owned());
    }
    if let Ok(arr) = arr_py.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(arr.as_array().mapv(|v| v as f64));
    }
    if let Ok(arr) = arr_py.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(arr.as_array().mapv(|v| v as f64));
    }
    if let Ok(arr) = arr_py.extract::<PyReadonlyArrayDyn<i32>>() {
        return Ok(arr.as_array().mapv(|v| v as f64));
    }
    Err(pyo3::exceptions::PyTypeError::new_err("Unsupported array type. Expected u8, i32, f32, or f64."))
}

/// apply_multiply() - Per-element multiplication of two arrays with scale.
/// @py: Python interpreter token.
/// @src1: First input array (PyAny).
/// @src2: Second input array (PyAny).
/// @scale: Scaling factor.
///
/// Performs per-element multiplication of two arrays and scales the result.
/// The arrays are extracted as f64. If the shapes differ, a PyValueError is returned.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
#[pyo3(signature = (src1, src2, scale = 1.0))]
pub fn apply_multiply<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
    scale: f64,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr1 = extract_f64_array(src1)?;
    let arr2 = extract_f64_array(src2)?;
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Arrays must have the same shape"));
    }
    let mut out = numpy::ndarray::ArrayD::<f64>::zeros(arr1.shape());
    Zip::from(&mut out).and(&arr1).and(&arr2).for_each(|res, &a, &b| {
        *res = a * b * scale;
    });
    Ok(out.into_pyarray(py).to_dyn())
}

/// apply_divide() - Per-element division of two arrays with scale.
/// @py: Python interpreter token.
/// @src1: First input array (numerator, PyAny).
/// @src2: Second input array (denominator, PyAny).
/// @scale: Scaling factor.
///
/// Performs per-element division of two arrays and scales the result.
/// If any element in the divisor is close to zero (absolute value < 1e-9),
/// the result for that element is set to 0.0. If shapes differ, a PyValueError
/// is returned.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
#[pyo3(signature = (src1, src2, scale = 1.0))]
pub fn apply_divide<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
    scale: f64,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr1 = extract_f64_array(src1)?;
    let arr2 = extract_f64_array(src2)?;
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Arrays must have the same shape"));
    }
    let mut out = numpy::ndarray::ArrayD::<f64>::zeros(arr1.shape());
    Zip::from(&mut out).and(&arr1).and(&arr2).for_each(|res, &a, &b| {
        if b.abs() < 1e-9 {
            *res = 0.0;
        } else {
            *res = a / b * scale;
        }
    });
    Ok(out.into_pyarray(py).to_dyn())
}

/// absdiff() - Compute absolute difference between two images.
/// @py: Python interpreter token.
/// @src1: First input image array.
/// @src2: Second input image array.
///
/// Computes the absolute difference between two images of the same shape.
/// For each pixel, result = |src1 - src2|. If shapes differ, a PyValueError is returned.
///
/// Return: A new PyArrayDyn containing the absolute difference.
#[pyfunction]
pub fn absdiff<'py>(
    py: Python<'py>,
    src1: PyReadonlyArrayDyn<'py, u8>,
    src2: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr1 = src1.as_array();
    let arr2 = src2.as_array();
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the exact same shape"));
    }
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|res, &a, &b| {
        *res = if a > b { a - b } else { b - a };
    });
    Ok(result.into_pyarray_bound(py).into())
}

/// apply_min() - Compute per-element minimum of two arrays.
/// @py: Python interpreter token.
/// @src1: First input array (PyAny).
/// @src2: Second input array (PyAny).
///
/// Computes the per-element minimum of two dynamic dimensional arrays.
/// The input arrays are extracted as f64. If their shapes differ, a PyValueError
/// is returned.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
pub fn apply_min<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr1 = extract_f64_array(src1)?;
    let arr2 = extract_f64_array(src2)?;
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Arrays must have the same shape"));
    }
    let mut out = numpy::ndarray::ArrayD::<f64>::zeros(arr1.shape());
    Zip::from(&mut out).and(&arr1).and(&arr2).for_each(|res, &a, &b| {
        *res = a.min(b);
    });
    Ok(out.into_pyarray(py).to_dyn())
}

/// apply_max() - Compute per-element maximum of two arrays.
/// @py: Python interpreter token.
/// @src1: First input array (PyAny).
/// @src2: Second input array (PyAny).
///
/// Computes the per-element maximum of two dynamic dimensional arrays.
/// The input arrays are extracted as f64. If their shapes differ, a PyValueError
/// is returned.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
pub fn apply_max<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr1 = extract_f64_array(src1)?;
    let arr2 = extract_f64_array(src2)?;
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Arrays must have the same shape"));
    }
    let mut out = numpy::ndarray::ArrayD::<f64>::zeros(arr1.shape());
    Zip::from(&mut out).and(&arr1).and(&arr2).for_each(|res, &a, &b| {
        *res = a.max(b);
    });
    Ok(out.into_pyarray(py).to_dyn())
}

/// apply_pow() - Compute per-element power.
/// @py: Python interpreter token.
/// @src: Source input array (PyAny).
/// @power: Exponent value.
///
/// Computes the per-element exponentiation (src[i]^power) of the input array.
/// The input array is extracted as f64.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
pub fn apply_pow<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    power: f64,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let out = arr.mapv(|val| val.powf(power));
    Ok(out.into_pyarray(py).to_dyn())
}

/// apply_sqrt() - Compute per-element square root.
/// @py: Python interpreter token.
/// @src: Source input array (PyAny).
///
/// Computes the per-element square root (sqrt(src[i])) of the input array.
/// The input array is extracted as f64.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
pub fn apply_sqrt<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let out = arr.mapv(|val| val.sqrt());
    Ok(out.into_pyarray(py).to_dyn())
}

/// apply_exp() - Compute per-element exponential.
/// @py: Python interpreter token.
/// @src: Source input array (PyAny).
///
/// Computes the per-element exponential (e^src[i]) of the input array.
/// The input array is extracted as f64.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
pub fn apply_exp<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let out = arr.mapv(|val| val.exp());
    Ok(out.into_pyarray(py).to_dyn())
}

/// apply_log_op() - Compute per-element natural logarithm.
/// @py: Python interpreter token.
/// @src: Source input array (PyAny).
///
/// Computes the per-element natural logarithm (ln(src[i])) of the input array.
/// The input array is extracted as f64.
///
/// Return: A dynamic dimensional 64-bit float PyArray on success.
#[pyfunction]
pub fn apply_log_op<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let out = arr.mapv(|val| val.ln());
    Ok(out.into_pyarray(py).to_dyn())
}

/// apply_normalize() - Normalize the norm or value range of an array.
/// @py: Python interpreter token.
/// @src: Source input array (PyAny).
/// @alpha: Lower bound value (min/max range) or norm target.
/// @beta: Upper bound value (min/max range).
/// @norm_type: Normalization type (32 for MINMAX, 4 for L2, otherwise L1).
///
/// Normalizes the input array according to the specified norm_type.
/// If norm_type is 32, values are scaled to be within [alpha, beta] range.
/// If norm_type is 4, L2 norm is normalized. Otherwise, L1 norm is normalized.
///
/// Return: A dynamic dimensional normalized 64-bit float PyArray on success.
#[pyfunction]
#[pyo3(signature = (src, alpha = 0.0, beta = 255.0, norm_type = 32))]
pub fn apply_normalize<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    alpha: f64,
    beta: f64,
    norm_type: i32,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let mut out = numpy::ndarray::ArrayD::<f64>::zeros(arr.shape());
    
    if norm_type == 32 {
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        arr.for_each(|&v| {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        });
        
        let range = max_val - min_val;
        let target_min = alpha.min(beta);
        let target_max = alpha.max(beta);
        let target_range = target_max - target_min;
        
        Zip::from(&mut out).and(&arr).for_each(|res, &val| {
            if range.abs() > 1e-9 {
                *res = target_min + ((val - min_val) / range) * target_range;
            } else {
                *res = target_min;
            }
        });
    } else {
        let mut norm = 0.0;
        if norm_type == 4 {
            let mut sum_sq = 0.0;
            arr.for_each(|&v| sum_sq += v * v);
            norm = sum_sq.sqrt();
        } else {
            arr.for_each(|&v| norm += v.abs());
        }
        
        Zip::from(&mut out).and(&arr).for_each(|res, &val| {
            if norm.abs() > 1e-9 {
                *res = val * alpha / norm;
            } else {
                *res = 0.0;
            }
        });
    }
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// convert_scale_abs() - Scale, compute absolute value, and convert to 8-bit.
/// @py: Python interpreter token.
/// @src: Source input array (PyAny).
/// @alpha: Optional scale factor.
/// @beta: Optional delta added to scaled values.
///
/// Scales, computes absolute values, and clamps the result to [0, 255] range as u8.
/// The operation is computed as: result[i] = clamp(|src[i] * alpha + beta|, 0, 255).
///
/// Return: A dynamic dimensional 8-bit unsigned integer PyArray on success.
#[pyfunction]
#[pyo3(signature = (src, alpha = 1.0, beta = 0.0))]
pub fn convert_scale_abs<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    alpha: f64,
    beta: f64,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = extract_f64_array(src)?;
    let mut out = numpy::ndarray::ArrayD::<u8>::zeros(arr.shape());
    
    Zip::from(&mut out).and(&arr).for_each(|res, &val| {
        let computed = (val * alpha + beta).abs();
        *res = computed.round().clamp(0.0, 255.0) as u8;
    });
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// in_range() - Check if array elements lie between lowerb and upperb.
/// @py: Python interpreter token.
/// @src: Source input image array (u8).
/// @lowerb: Lower boundary vector.
/// @upperb: Upper boundary vector.
///
/// Checks if the elements of the source array lie between the values of the
/// lower and upper boundary vectors. If the image is multi-channel, boundary checks
/// are performed on all channels, and all must satisfy the condition.
///
/// Return: A binary 2D PyArrayDyn where matching elements are 255, and others are 0.
#[pyfunction]
pub fn in_range<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    lowerb: Vec<f64>,
    upperb: Vec<f64>,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = src.as_array();
    let ndim = arr.ndim();
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        
        if lowerb.len() < c || upperb.len() < c {
            return Err(pyo3::exceptions::PyValueError::new_err("Bounds length must match number of channels"));
        }
        
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let mut in_bounds = true;
                for ch in 0..c {
                    let val = img_3d[[y, x, ch]] as f64;
                    if val < lowerb[ch] || val > upperb[ch] {
                        in_bounds = false;
                        break;
                    }
                }
                out[[y, x]] = if in_bounds { 255 } else { 0 };
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
        
        if lowerb.is_empty() || upperb.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Bounds cannot be empty"));
        }
        
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let val = img_2d[[y, x]] as f64;
                let in_bounds = val >= lowerb[0] && val <= upperb[0];
                out[[y, x]] = if in_bounds { 255 } else { 0 };
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// apply_lut() - Apply a lookup table to an image.
/// @py: Python interpreter token.
/// @src: Source input image array (u8).
/// @lut: Lookup table containing at least 256 elements.
///
/// Performs lookup table transformation for each element of the input image:
/// result[i] = lut[src[i]].
///
/// Return: A new PyArrayDyn containing lookup table mapped values.
#[pyfunction]
pub fn apply_lut<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    lut: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = src.as_array();
    let lut_arr = lut.as_array();
    
    if lut_arr.len() < 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("Lookup table (lut) must contain at least 256 elements"));
    }
    
    let mut out = numpy::ndarray::ArrayD::<u8>::zeros(arr.shape());
    Zip::from(&mut out).and(&arr).for_each(|res, &val| {
        *res = lut_arr[[val as usize]];
    });
    
    Ok(out.into_pyarray_bound(py).into())
}

/// split_channels() - Split a multi-channel image into a list of single-channel images.
/// @py: Python interpreter token.
/// @image: Multi-channel or single-channel input image (u8).
///
/// Splits a multi-channel image into a vector of single-channel 2D images.
/// If the input image is already 2D (single-channel), it returns a vector with the original image.
///
/// Return: A vector of 2D PyArrayDyn channels.
#[pyfunction]
pub fn split_channels<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Vec<&'py PyArrayDyn<u8>>> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        
        let mut out_channels = Vec::new();
        for ch in 0..c {
            let mut chan_arr = numpy::ndarray::Array2::<u8>::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    chan_arr[[y, x]] = img_3d[[y, x, ch]];
                }
            }
            out_channels.push(chan_arr.into_pyarray(py).to_dyn());
        }
        Ok(out_channels)
    } else if ndim == 2 {
        Ok(vec![image.to_object(py).extract::<&PyArrayDyn<u8>>(py)?])
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// Merge several single-channel images into a single multi-channel image.
#[pyfunction]
pub fn merge_channels<'py>(
    py: Python<'py>,
    channels: Vec<PyReadonlyArrayDyn<'py, u8>>,
) -> PyResult<&'py PyArrayDyn<u8>> {
    if channels.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Channels list cannot be empty"));
    }
    
    let shape_ref = channels[0].as_array().shape().to_vec();
    if shape_ref.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Each channel must be a 2D array"));
    }
    let h = shape_ref[0];
    let w = shape_ref[1];
    
    for (i, chan) in channels.iter().enumerate() {
        let sh = chan.as_array().shape().to_vec();
        if sh != [h, w] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Channel {} has shape {:?}, but expected {:?}",
                i, sh, [h, w]
            )));
        }
    }
    
    let c = channels.len();
    let mut out = numpy::ndarray::Array3::<u8>::zeros((h, w, c));
    
    for ch in 0..c {
        let chan_2d = channels[ch].as_array().into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast channel to 2D"))?;
        for y in 0..h {
            for x in 0..w {
                out[[y, x, ch]] = chan_2d[[y, x]];
            }
        }
    }
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// Copy specified channels from input arrays to specified channels of output arrays.
#[pyfunction]
pub fn mix_channels<'py>(
    py: Python<'py>,
    src: Vec<PyReadonlyArrayDyn<'py, u8>>,
    dst: Vec<PyReadonlyArrayDyn<'py, u8>>,
    from_to: Vec<usize>,
) -> PyResult<Vec<&'py PyArrayDyn<u8>>> {
    if from_to.len() % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("from_to list must contain an even number of elements (pairs of src_chan, dst_chan)"));
    }
    
    let mut src_flat_chans = Vec::new();
    for img in src.iter() {
        let arr = img.as_array();
        let ndim = arr.ndim();
        if ndim == 2 {
            src_flat_chans.push(arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap().to_owned());
        } else if ndim == 3 {
            let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>().unwrap();
            let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
            for ch in 0..c {
                let mut chan = numpy::ndarray::Array2::<u8>::zeros((h, w));
                for y in 0..h {
                    for x in 0..w {
                        chan[[y, x]] = img_3d[[y, x, ch]];
                    }
                }
                src_flat_chans.push(chan);
            }
        }
    }
    
    let mut dst_flat_chans = Vec::new();
    let mut dst_shapes = Vec::new();
    for img in dst.iter() {
        let arr = img.as_array();
        let ndim = arr.ndim();
        if ndim == 2 {
            let shape = [arr.shape()[0], arr.shape()[1], 1];
            dst_shapes.push(shape);
            dst_flat_chans.push(numpy::ndarray::Array2::<u8>::zeros((shape[0], shape[1])));
        } else if ndim == 3 {
            let shape = [arr.shape()[0], arr.shape()[1], arr.shape()[2]];
            dst_shapes.push(shape);
            for _ in 0..shape[2] {
                dst_flat_chans.push(numpy::ndarray::Array2::<u8>::zeros((shape[0], shape[1])));
            }
        }
    }
    
    for i in (0..from_to.len()).step_by(2) {
        let s_idx = from_to[i];
        let d_idx = from_to[i + 1];
        
        if s_idx >= src_flat_chans.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("source channel index {} is out of bounds", s_idx)));
        }
        if d_idx >= dst_flat_chans.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("destination channel index {} is out of bounds", d_idx)));
        }
        
        let src_chan = &src_flat_chans[s_idx];
        let dst_chan = &mut dst_flat_chans[d_idx];
        
        if src_chan.shape() != dst_chan.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err("Source and destination channel sizes must match"));
        }
        
        dst_chan.assign(src_chan);
    }
    
    let mut out_images = Vec::new();
    let mut flat_idx = 0;
    for shape in dst_shapes.iter() {
        let h = shape[0];
        let w = shape[1];
        let c = shape[2];
        
        if c == 1 {
            let out_arr = dst_flat_chans[flat_idx].clone();
            out_images.push(out_arr.into_pyarray(py).to_dyn());
            flat_idx += 1;
        } else {
            let mut out_arr = numpy::ndarray::Array3::<u8>::zeros((h, w, c));
            for ch in 0..c {
                let chan = &dst_flat_chans[flat_idx];
                for y in 0..h {
                    for x in 0..w {
                        out_arr[[y, x, ch]] = chan[[y, x]];
                    }
                }
                flat_idx += 1;
            }
            out_images.push(out_arr.into_pyarray(py).to_dyn());
        }
    }
    
    Ok(out_images)
}

/// Count non-zero array elements.
#[pyfunction]
pub fn count_non_zero<'py>(
    _py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<usize> {
    let arr = extract_f64_array(src)?;
    let mut count = 0;
    arr.for_each(|&val| {
        if val != 0.0 {
            count += 1;
        }
    });
    Ok(count)
}

/// Compute mean and standard deviation of array elements.
#[pyfunction]
pub fn mean_std_dev<'py>(
    _py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let arr = src.as_array();
    let ndim = arr.ndim();
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let total_pixels = (h * w) as f64;
        
        let mut means = vec![0.0; c];
        let mut stddevs = vec![0.0; c];
        
        for ch in 0..c {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for y in 0..h {
                for x in 0..w {
                    let val = img_3d[[y, x, ch]] as f64;
                    let val_f = val;
                    sum += val_f;
                    sum_sq += val_f * val_f;
                }
            }
            let mean = sum / total_pixels;
            let variance = (sum_sq / total_pixels) - (mean * mean);
            means[ch] = mean;
            stddevs[ch] = variance.max(0.0).sqrt();
        }
        Ok((means, stddevs))
    } else if ndim == 2 {
        let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let total_pixels = (h * w) as f64;
        
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for y in 0..h {
            for x in 0..w {
                let val = img_2d[[y, x]] as f64;
                sum += val;
                sum_sq += val * val;
            }
        }
        let mean = sum / total_pixels;
        let variance = (sum_sq / total_pixels) - (mean * mean);
        Ok((vec![mean], vec![variance.max(0.0).sqrt()]))
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// Find minimum and maximum values and their locations in a 2D single-channel image.
#[pyfunction]
pub fn min_max_loc<'py>(
    _py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<(f64, f64, (usize, usize), (usize, usize))> {
    let arr = src.as_array();
    let ndim = arr.ndim();
    if ndim != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("minMaxLoc is only supported on 2D single-channel images"));
    }
    
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    let mut min_loc = (0, 0);
    let mut max_loc = (0, 0);
    
    for y in 0..h {
        for x in 0..w {
            let val = img_2d[[y, x]] as f64;
            if val < min_val {
                min_val = val;
                min_loc = (x, y);
            }
            if val > max_val {
                max_val = val;
                max_loc = (x, y);
            }
        }
    }
    Ok((min_val, max_val, min_loc, max_loc))
}

/// Calculate the rotation angle of 2D vectors.
#[pyfunction]
#[pyo3(signature = (x, y, angle_in_degrees = false))]
pub fn calculate_phase<'py>(
    py: Python<'py>,
    x: &pyo3::PyAny,
    y: &pyo3::PyAny,
    angle_in_degrees: bool,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr_x = extract_f64_array(x)?;
    let arr_y = extract_f64_array(y)?;
    if arr_x.shape() != arr_y.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("X and Y arrays must have the same shape"));
    }
    
    let mut out = numpy::ndarray::ArrayD::<f64>::zeros(arr_x.shape());
    Zip::from(&mut out).and(&arr_x).and(&arr_y).for_each(|res, &vx, &vy| {
        let mut angle = vy.atan2(vx);
        if angle < 0.0 {
            angle += 2.0 * std::f64::consts::PI;
        }
        if angle_in_degrees {
            *res = angle.to_degrees();
        } else {
            *res = angle;
        }
    });
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// Calculate the magnitude of 2D vectors.
#[pyfunction]
pub fn calculate_magnitude<'py>(
    py: Python<'py>,
    x: &pyo3::PyAny,
    y: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr_x = extract_f64_array(x)?;
    let arr_y = extract_f64_array(y)?;
    if arr_x.shape() != arr_y.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("X and Y arrays must have the same shape"));
    }
    
    let mut out = numpy::ndarray::ArrayD::<f64>::zeros(arr_x.shape());
    Zip::from(&mut out).and(&arr_x).and(&arr_y).for_each(|res, &vx, &vy| {
        *res = (vx * vx + vy * vy).sqrt();
    });
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// Convert Cartesian coordinates to polar.
#[pyfunction]
#[pyo3(signature = (x, y, angle_in_degrees = false))]
pub fn cart_to_polar<'py>(
    py: Python<'py>,
    x: &pyo3::PyAny,
    y: &pyo3::PyAny,
    angle_in_degrees: bool,
) -> PyResult<(&'py PyArrayDyn<f64>, &'py PyArrayDyn<f64>)> {
    let arr_x = extract_f64_array(x)?;
    let arr_y = extract_f64_array(y)?;
    if arr_x.shape() != arr_y.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("X and Y arrays must have the same shape"));
    }
    
    let mut mag = numpy::ndarray::ArrayD::<f64>::zeros(arr_x.shape());
    let mut ang = numpy::ndarray::ArrayD::<f64>::zeros(arr_x.shape());
    
    Zip::from(&mut mag).and(&mut ang).and(&arr_x).and(&arr_y).for_each(|m, a, &vx, &vy| {
        *m = (vx * vx + vy * vy).sqrt();
        let mut angle = vy.atan2(vx);
        if angle < 0.0 {
            angle += 2.0 * std::f64::consts::PI;
        }
        if angle_in_degrees {
            *a = angle.to_degrees();
        } else {
            *a = angle;
        }
    });
    
    Ok((mag.into_pyarray(py).to_dyn(), ang.into_pyarray(py).to_dyn()))
}

/// Convert polar coordinates to Cartesian.
#[pyfunction]
#[pyo3(signature = (magnitude, angle, angle_in_degrees = false))]
pub fn polar_to_cart<'py>(
    py: Python<'py>,
    magnitude: &pyo3::PyAny,
    angle: &pyo3::PyAny,
    angle_in_degrees: bool,
) -> PyResult<(&'py PyArrayDyn<f64>, &'py PyArrayDyn<f64>)> {
    let arr_m = extract_f64_array(magnitude)?;
    let arr_a = extract_f64_array(angle)?;
    if arr_m.shape() != arr_a.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Magnitude and Angle arrays must have the same shape"));
    }
    
    let mut x = numpy::ndarray::ArrayD::<f64>::zeros(arr_m.shape());
    let mut y = numpy::ndarray::ArrayD::<f64>::zeros(arr_m.shape());
    
    Zip::from(&mut x).and(&mut y).and(&arr_m).and(&arr_a).for_each(|vx, vy, &m, &a| {
        let angle_rad = if angle_in_degrees { a.to_radians() } else { a };
        *vx = m * angle_rad.cos();
        *vy = m * angle_rad.sin();
    });
    
    Ok((x.into_pyarray(py).to_dyn(), y.into_pyarray(py).to_dyn()))
}

/// Alias for apply_multiply matching OpenCV name.
#[pyfunction(name = "multiply")]
#[pyo3(signature = (src1, src2, scale = 1.0))]
pub fn multiply<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
    scale: f64,
) -> PyResult<&'py PyArrayDyn<f64>> {
    apply_multiply(py, src1, src2, scale)
}

/// Alias for apply_divide matching OpenCV name.
#[pyfunction(name = "divide")]
#[pyo3(signature = (src1, src2, scale = 1.0))]
pub fn divide<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
    scale: f64,
) -> PyResult<&'py PyArrayDyn<f64>> {
    apply_divide(py, src1, src2, scale)
}

/// Alias for apply_min matching OpenCV name.
#[pyfunction(name = "min")]
pub fn min<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    apply_min(py, src1, src2)
}

/// Alias for apply_max matching OpenCV name.
#[pyfunction(name = "max")]
pub fn max<'py>(
    py: Python<'py>,
    src1: &pyo3::PyAny,
    src2: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    apply_max(py, src1, src2)
}

/// Alias for apply_normalize matching OpenCV name.
#[pyfunction(name = "normalize")]
#[pyo3(signature = (src, alpha = 0.0, beta = 255.0, norm_type = 32))]
pub fn normalize<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    alpha: f64,
    beta: f64,
    norm_type: i32,
) -> PyResult<&'py PyArrayDyn<f64>> {
    apply_normalize(py, src, alpha, beta, norm_type)
}

/// Alias for split_channels matching OpenCV name.
#[pyfunction(name = "split")]
pub fn split<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Vec<&'py PyArrayDyn<u8>>> {
    split_channels(py, image)
}

/// Alias for merge_channels matching OpenCV name.
#[pyfunction(name = "merge")]
pub fn merge<'py>(
    py: Python<'py>,
    channels: Vec<PyReadonlyArrayDyn<'py, u8>>,
) -> PyResult<&'py PyArrayDyn<u8>> {
    merge_channels(py, channels)
}

/// Alias for mix_channels matching OpenCV name.
#[pyfunction(name = "mixChannels")]
pub fn mix_channels_cv<'py>(
    py: Python<'py>,
    src: Vec<PyReadonlyArrayDyn<'py, u8>>,
    dst: Vec<PyReadonlyArrayDyn<'py, u8>>,
    from_to: Vec<usize>,
) -> PyResult<Vec<&'py PyArrayDyn<u8>>> {
    mix_channels(py, src, dst, from_to)
}

/// Alias for in_range matching OpenCV name.
#[pyfunction(name = "inRange")]
pub fn in_range_cv<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    lowerb: Vec<f64>,
    upperb: Vec<f64>,
) -> PyResult<&'py PyArrayDyn<u8>> {
    in_range(py, src, lowerb, upperb)
}

/// Alias for apply_lut matching OpenCV name.
#[pyfunction(name = "LUT")]
pub fn lut_cv<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    lut: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    apply_lut(py, src, lut)
}

/// Alias for count_non_zero matching OpenCV name.
#[pyfunction(name = "countNonZero")]
pub fn count_non_zero_cv<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<usize> {
    count_non_zero(py, src)
}

/// Alias for mean_std_dev matching OpenCV name.
#[pyfunction(name = "meanStdDev")]
pub fn mean_std_dev_cv<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    mean_std_dev(py, src)
}

/// Alias for min_max_loc matching OpenCV name.
#[pyfunction(name = "minMaxLoc")]
pub fn min_max_loc_cv<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<(f64, f64, (usize, usize), (usize, usize))> {
    min_max_loc(py, src)
}