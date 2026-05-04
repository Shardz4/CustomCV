use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};

use crate::helpers::{apply_median_3x3, apply_laplacian_3x3};

// ==========================================
// SPATIAL FILTERS (Median / Laplacian)
// ==========================================

#[pyfunction]
pub fn median_filter<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let ndim = arr.ndim();
    if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let filtered = apply_median_3x3(channel.view());
        return Ok(filtered.into_pyarray(py).to_dyn());
    } else if ndim == 3 {
        let (h, w, c) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
        let mut out_arr = Array3::<u8>::zeros((h, w, c));
        for ch in 0..c {
            let channel = arr.slice(s![.., .., ch]);
            out_arr.slice_mut(s![.., .., ch]).assign(&apply_median_3x3(channel));
        }
        return Ok(out_arr.into_pyarray(py).to_dyn());
    }
    panic!("Unsupported image dimensions!");
}

#[pyfunction]
pub fn laplacian_filter<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let ndim = arr.ndim();
    if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let filtered = apply_laplacian_3x3(channel.view());
        return Ok(filtered.into_pyarray(py).to_dyn());
    } else if ndim == 3 {
        let (h, w, c) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
        let mut out_arr = Array3::<u8>::zeros((h, w, c));
        for ch in 0..c {
            let channel = arr.slice(s![.., .., ch]);
            out_arr.slice_mut(s![.., .., ch]).assign(&apply_laplacian_3x3(channel));
        }
        return Ok(out_arr.into_pyarray(py).to_dyn());
    }
    panic!("Unsupported image dimensions!");
}
