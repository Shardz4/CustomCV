use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3, Zip},
    IntoPyArray, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn,
};

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

#[pyfunction]
pub fn bitwise_not<'py>(py:Python<'py>,img:PyReadonlyArrayDyn<'py,u8>)->PyResult<Py<PyArrayDyn<u8>>>{
    let arr = img.as_array();
    let mut result = numpy::ndarray::ArrayD::<u8>::zeros(arr.shape());
    Zip::from(&mut result).and(&arr).for_each(|r, a| { *r = !*a });
    Ok(result.into_pyarray_bound(py).into())   
}

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

/// Per-element multiplication of two arrays with scale.
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

/// Per-element division of two arrays with scale.
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

/// Compute absolute difference between two images.
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

/// Compute per-element minimum of two arrays.
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

/// Compute per-element maximum of two arrays.
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

/// Compute per-element power.
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

/// Compute per-element square root.
#[pyfunction]
pub fn apply_sqrt<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let out = arr.mapv(|val| val.sqrt());
    Ok(out.into_pyarray(py).to_dyn())
}

/// Compute per-element exponential.
#[pyfunction]
pub fn apply_exp<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let out = arr.mapv(|val| val.exp());
    Ok(out.into_pyarray(py).to_dyn())
}

/// Compute per-element natural log.
#[pyfunction]
pub fn apply_log_op<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = extract_f64_array(src)?;
    let out = arr.mapv(|val| val.ln());
    Ok(out.into_pyarray(py).to_dyn())
}

/// Normalize the norm or value range of an array.
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

/// Scale, compute absolute value, and convert to 8-bit.
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

/// Check if array elements lie between lowerb and upperb.
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

/// Apply a lookup table to an image.
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