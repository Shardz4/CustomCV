use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array2, Array3, ArrayView2},
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

// Helper to process 2D or 3D images channel-by-channel.
fn run_morph_op<'py, F>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    kernel: PyReadonlyArrayDyn<'py, u8>,
    op: F,
) -> PyResult<&'py PyArrayDyn<u8>>
where
    F: Fn(ArrayView2<u8>, ArrayView2<u8>) -> Array2<u8>,
{
    let img_arr = image.as_array();
    let k_arr = kernel.as_array();
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2D"))?;
        
    let ndim = img_arr.ndim();
    if ndim == 2 {
        let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let res = op(img_2d, k_2d);
        Ok(res.into_pyarray(py).to_dyn())
    } else if ndim == 3 {
        let shape = img_arr.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out = Array3::<u8>::zeros((h, w, c));
        for ch in 0..c {
            let channel = img_arr.slice(s![.., .., ch]);
            let channel_2d = channel.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
            let res = op(channel_2d, k_2d);
            out.slice_mut(s![.., .., ch]).assign(&res);
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// apply_erosion() - Applies erosion to an image using a structuring element.
/// @py: Python interpreter token.
/// @image: Input image array (u8), 2D or 3D.
/// @kernel: Structuring element array.
///
/// Erodes the input image using the specified structuring element. If the image
/// is 3D, the operation is applied channel-by-channel.
///
/// Return: Eroded image array.
#[pyfunction]
pub fn apply_erosion<'py>(
    py: Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>
) -> PyResult<&'py PyArrayDyn<u8>> {
    run_morph_op(py, image, kernel, |img, k| erode_2d(img, k))
}

/// apply_dilation() - Applies dilation to an image using a structuring element.
/// @py: Python interpreter token.
/// @image: Input image array (u8), 2D or 3D.
/// @kernel: Structuring element array.
///
/// Dilates the input image using the specified structuring element. If the image
/// is 3D, the operation is applied channel-by-channel.
///
/// Return: Dilated image array.
#[pyfunction]
pub fn apply_dilation<'py>(
    py: Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>
) -> PyResult<&'py PyArrayDyn<u8>> {
    run_morph_op(py, image, kernel, |img, k| dilate_2d(img, k))
}

/// opening() - Applies morphological opening (erosion followed by dilation) to an image.
/// @py: Python interpreter token.
/// @image: Input image array (u8), 2D or 3D.
/// @kernel: Structuring element array.
///
/// Useful for removing small objects or noise. If the image is 3D, the operation
/// is applied channel-by-channel.
///
/// Return: Opened image array.
#[pyfunction]
pub fn opening<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    run_morph_op(py, image, kernel, |img, k| {
        let eroded = erode_2d(img, k);
        dilate_2d(eroded.view(), k)
    })
}

/// apply_closing() - Applies morphological closing (dilation followed by erosion) to an image.
/// @py: Python interpreter token.
/// @image: Input image array (u8), 2D or 3D.
/// @kernel: Structuring element array.
///
/// Useful for closing small holes inside the objects, or connecting components.
/// If the image is 3D, the operation is applied channel-by-channel.
///
/// Return: Closed image array.
#[pyfunction]
pub fn apply_closing<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    run_morph_op(py, image, kernel, |img, k| {
        let dilated = dilate_2d(img, k);
        erode_2d(dilated.view(), k)
    })
}

/// morphological_gradient() - Computes the morphological gradient of an image.
/// @py: Python interpreter token.
/// @image: Input image array (u8), 2D or 3D.
/// @kernel: Structuring element array.
///
/// Computes the difference between the dilation and the erosion of an image.
/// If the image is 3D, the operation is applied channel-by-channel.
///
/// Return: Gradient image array.
#[pyfunction]
pub fn morphological_gradient<'py>(py:Python<'py>, image:PyReadonlyArrayDyn<'py, u8>, kernel:PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    run_morph_op(py, image, kernel, |img, k| {
        let dilated = dilate_2d(img, k);
        let eroded = erode_2d(img, k);
        dilated - eroded
    })
}

/// top_hat() - Computes the top-hat transform of an image.
/// @py: Python interpreter token.
/// @image: Input image array (u8), 2D or 3D.
/// @kernel: Structuring element array.
///
/// Computes the difference between the input image and its opening.
/// If the image is 3D, the operation is applied channel-by-channel.
///
/// Return: Top-hat image array.
#[pyfunction]
pub fn top_hat<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    run_morph_op(py, image, kernel, |img, k| {
        let eroded = erode_2d(img, k);
        let opened = dilate_2d(eroded.view(), k);
        img.to_owned() - opened
    })
}

/// black_hat() - Computes the black-hat transform of an image.
/// @py: Python interpreter token.
/// @image: Input image array (u8), 2D or 3D.
/// @kernel: Structuring element array.
///
/// Computes the difference between the closing of the image and the input image.
/// If the image is 3D, the operation is applied channel-by-channel.
///
/// Return: Black-hat image array.
#[pyfunction]
pub fn black_hat<'py>(py:Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, u8>)->PyResult<&'py PyArrayDyn<u8>>{
    run_morph_op(py, image, kernel, |img, k| {
        let dilated = dilate_2d(img, k);
        let closed = erode_2d(dilated.view(), k);
        closed - img.to_owned()
    })
}


