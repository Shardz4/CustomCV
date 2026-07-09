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