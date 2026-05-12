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


// -------------------------------------------------//
// -----IMAGE PYRAMIDS-----//
// -------------------------------------------------//


// ==========================================
// IMAGE PYRAMIDS
// ==========================================

#[pyfunction]
fn pyr_down<'py>(py: Python<'py>, img: PyReadonlyArrayDyn<'py, u8>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();

    // Standard 5x5 Gaussian kernel for image pyramids
    let k1d = [1.0, 4.0, 6.0, 4.0, 1.0];
    let mut kernel = numpy::ndarray::Array2::<f64>::zeros((5, 5));
    for y in 0..5 {
        for x in 0..5 {
            kernel[[y, x]] = (k1d[y] * k1d[x]) / 256.0;
        }
    }

    let process_channel = |channel: numpy::ndarray::ArrayView2<u8>| -> numpy::ndarray::Array2<u8> {
        let (h, w) = (channel.shape()[0], channel.shape()[1]);
        let out_h = (h + 1) / 2;
        let out_w = (w + 1) / 2;
        let mut out = numpy::ndarray::Array2::<u8>::zeros((out_h, out_w));

        for y in 0..out_h {
            for x in 0..out_w {
                let mut sum = 0.0;
                let src_y = y * 2; // Map to original coordinates
                let src_x = x * 2;

                for ky in 0..5 {
                    for kx in 0..5 {
                        let iy = (src_y as isize + ky as isize - 2).clamp(0, h as isize - 1) as usize;
                        let ix = (src_x as isize + kx as isize - 2).clamp(0, w as isize - 1) as usize;
                        sum += channel[[iy, ix]] as f64 * kernel[[ky, kx]];
                    }
                }
                out[[y, x]] = sum.clamp(0.0, 255.0) as u8;
            }
        }
        out
    };

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let out_h = (h + 1) / 2;
        let out_w = (w + 1) / 2;
        let mut out = numpy::ndarray::Array3::<u8>::zeros((out_h, out_w, c));
        for ch in 0..c {
            out.slice_mut(s![.., .., ch]).assign(&process_channel(arr.slice(s![.., .., ch])));
        }
        Ok(out.into_pyarray_bound(py).to_dyn().into())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        Ok(process_channel(channel.view()).into_pyarray_bound(py).to_dyn().into())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

#[pyfunction]
fn pyr_up<'py>(py: Python<'py>, img: PyReadonlyArrayDyn<'py, u8>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();

    // Standard 5x5 Gaussian kernel for pyramids, MULTIPLIED BY 4. 
    // Since we insert zeros when upsampling, we must multiply the kernel by 4 to maintain brightness.
    let k1d = [1.0, 4.0, 6.0, 4.0, 1.0];
    let mut kernel = numpy::ndarray::Array2::<f64>::zeros((5, 5));
    for y in 0..5 {
        for x in 0..5 {
            kernel[[y, x]] = (k1d[y] * k1d[x]) / 64.0; // 256.0 / 4.0 = 64.0
        }
    }

    let process_channel = |channel: numpy::ndarray::ArrayView2<u8>| -> numpy::ndarray::Array2<u8> {
        let (h, w) = (channel.shape()[0], channel.shape()[1]);
        let out_h = h * 2;
        let out_w = w * 2;
        let mut out = numpy::ndarray::Array2::<u8>::zeros((out_h, out_w));

        for y in 0..out_h {
            for x in 0..out_w {
                let mut sum = 0.0;
                for ky in 0..5 {
                    for kx in 0..5 {
                        let src_y_isize = y as isize + ky as isize - 2;
                        let src_x_isize = x as isize + kx as isize - 2;

                        // In an upscaled grid, only the even pixels actually exist in the source image
                        if src_y_isize % 2 == 0 && src_x_isize % 2 == 0 {
                            let orig_y = src_y_isize / 2;
                            let orig_x = src_x_isize / 2;

                            let clamp_y = orig_y.clamp(0, h as isize - 1) as usize;
                            let clamp_x = orig_x.clamp(0, w as isize - 1) as usize;
                            sum += channel[[clamp_y, clamp_x]] as f64 * kernel[[ky, kx]];
                        }
                    }
                }
                out[[y, x]] = sum.clamp(0.0, 255.0) as u8;
            }
        }
        out
    };

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let out_h = h * 2;
        let out_w = w * 2;
        let mut out = numpy::ndarray::Array3::<u8>::zeros((out_h, out_w, c));
        for ch in 0..c {
            out.slice_mut(s![.., .., ch]).assign(&process_channel(arr.slice(s![.., .., ch])));
        }
        Ok(out.into_pyarray_bound(py).to_dyn().into())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        Ok(process_channel(channel.view()).into_pyarray_bound(py).to_dyn().into())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}