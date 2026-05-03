use pyo3::prelude::*;
use numpy::{
    ndarray::{Array2, ArrayView2},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};

fn erode_2d(image: ArrayView2<u8>, kernel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (image.shape()[0], image.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut min_val = 255u8;
            for ky in 0..kh {
                for kx in 0..kw {
                    if kernel[[ky, kx]] > 0 {
                        let iy = y as isize + ky as isize - pad_h as isize;
                        let ix = x as isize + kx as isize - pad_w as isize;
                        let pixel = if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            image[[iy as usize, ix as usize]]
                        } else { 255 };
                        if pixel < min_val { min_val = pixel; }
                    }
                }
            }
            out[[y, x]] = min_val;
        }
    }
    out
}

fn dilate_2d(image: ArrayView2<u8>, kernel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (image.shape()[0], image.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut max_val = 0u8;
            for ky in 0..kh {
                for kx in 0..kw {
                    if kernel[[ky, kx]] > 0 {
                        let iy = y as isize + ky as isize - pad_h as isize;
                        let ix = x as isize + kx as isize - pad_w as isize;
                        let pixel = if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            image[[iy as usize, ix as usize]]
                        } else { 0 };
                        if pixel > max_val { max_val = pixel; }
                    }
                }
            }
            out[[y, x]] = max_val;
        }
    }
    out
}

#[pyfunction]
pub fn apply_erosion<'py>(
    py: Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>
) -> PyResult<&'py PyArrayDyn<u8>> {
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Erosion requires a 2D image"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
    let result = erode_2d(img_2d.view(), k_2d.view());
    Ok(result.into_pyarray(py).to_dyn())
}

#[pyfunction]
pub fn apply_dilation<'py>(
    py: Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>
) -> PyResult<&'py PyArrayDyn<u8>> {
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Dilation requires a 2D image"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
    let result = dilate_2d(img_2d.view(), k_2d.view());
    Ok(result.into_pyarray(py).to_dyn())
}

#[pyfunction]
pub fn opening<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Opening requires 2D image"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
    
    let eroded = erode_2d(img_2d.view(), k_2d.view());
    let opened = dilate_2d(eroded.view(), k_2d.view());
    
    Ok(opened.into_pyarray(py).to_dyn())
}

#[pyfunction]
pub fn apply_closing<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Closing requires 2D image"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
    
    let dilated = dilate_2d(img_2d.view(), k_2d.view());
    let closed = erode_2d(dilated.view(), k_2d.view());
    
    Ok(closed.into_pyarray(py).to_dyn())
}
