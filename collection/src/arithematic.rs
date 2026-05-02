use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3},
    IntoPyArray, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn,
};

#[pyfunction]
fn add_images<'py>(py: Python<'py>, img1:PyReadonlyArrayDyn<u8>,img2:PyReadonlyArrayDyn<u8>)->PyResult<PyArrayDyn<u8>>{

    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the same shape"))
    }
    let mut result = numpy::ndarray::ArrayDyn::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|(r,a,b)| {*r.a.saturating_add(b);});
    
    Ok(result.into_pyarray_bound(py).into())   
}

#[pyfunction]
fn sub_images<'py>(py: Python<'py>, img1:PyReadonlyArrayDyn<u8>,img2:PyReadonlyArrayDyn<u8>)->PyResult<PyArrayDyn<u8>>{

    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Images must have the same shape"))
    }
    let mut result = numpy::ndarray::ArrayDyn::<u8>::zeros(arr1.shape());
    Zip::from(&mut result).and(&arr1).and(&arr2).for_each(|(r,a,b)| {*r.a.saturating_sub(b);});
    
    Ok(result.into_pyarray_bound(py).into())   
}