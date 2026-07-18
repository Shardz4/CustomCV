use pyo3::prelude::*;
use numpy::{
    ndarray::{Array2, ArrayView2},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};

/// erode_2d() - Erode a 2D grayscale image with a structuring element.
/// @image: 2D input image slice (u8).
/// @kernel: 2D structuring element (kernel) slice.
///
/// Computes the local minimum over the neighborhood specified by the kernel.
///
/// Return: Eroded 2D image array.
pub fn erode_2d(image: ArrayView2<u8>, kernel: ArrayView2<u8>) -> Array2<u8> {
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

/// dilate_2d() - Dilate a 2D grayscale image with a structuring element.
/// @image: 2D input image slice (u8).
/// @kernel: 2D structuring element (kernel) slice.
///
/// Computes the local maximum over the neighborhood specified by the kernel.
///
/// Return: Dilated 2D image array.
pub fn dilate_2d(image: ArrayView2<u8>, kernel: ArrayView2<u8>) -> Array2<u8> {
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

/// apply_erosion() - Applies erosion to an image using a structuring element.
/// @py: Python interpreter token.
/// @image: 2D input image array (u8).
/// @kernel: Structuring element array.
///
/// Erodes the input image using the specified structuring element.
///
/// Return: Eroded image array.
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

/// apply_dilation() - Applies dilation to an image using a structuring element.
/// @py: Python interpreter token.
/// @image: 2D input image array (u8).
/// @kernel: Structuring element array.
///
/// Dilates the input image using the specified structuring element.
///
/// Return: Dilated image array.
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

#[pyfunction]
pub fn morphological_gradient<'py>(py:Python<'py>, image:PyReadonlyArrayDyn<'py, u8>, kernel:PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
    .map_err(|_| pyo3::exceptions::PyValueError::new_err("Morphological Gradient requires 2D image"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
    .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
    
    let dilated = dilate_2d(img_2d.view(), k_2d.view());
    let eroded = erode_2d(img_2d.view(), k_2d.view());
    
    let gradient = &dilated - &eroded;
    Ok(gradient.into_pyarray(py).to_dyn())
}


#[pyfunction]
pub fn top_hat<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Top Hat requires 2D image"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
    
    let eroded = erode_2d(img_2d.view(), k_2d.view());
    let opened = dilate_2d(eroded.view(), k_2d.view());
    let top_hat = &img_2d - &opened;
    Ok(top_hat.into_pyarray(py).to_dyn())
}

#[pyfunction]
pub fn black_hat<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Black Hat requires 2D image"))?;
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
    
    let dilated = dilate_2d(img_2d.view(), k_2d.view());
    let closed = erode_2d(dilated.view(), k_2d.view());
    let black_hat = &closed - &img_2d;
    Ok(black_hat.into_pyarray(py).to_dyn())
}


