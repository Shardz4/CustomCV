use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use numpy::ndarray::s;
use crate::helpers;

#[pyfunction]
fn apply_filter2d<'py>(py: Python<'py>, img: PyReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = img.as_array();
    let k_arr = kernel.as_array();
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2d"))?;
    
    let ndim = arr.ndim();
    let shape = arr.shape();

    if ndim ==3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((h, w, c));

        for ch in 0..c{
            let filtered = helpers::convolve_2d_channel(arr.slice(s![.., .., ch]), k_2d.view());
            out.slice_mut(s![.., .., ch]).assign(&filtered);
        }
        Ok(out.into_dyn().into_pyarray_bound(py).unbind())
    } else if ndim ==2{
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let filtered = helpers::convolve_2d_channel(channel.view(), k_2d.view());
        return Ok(filtered.into_dyn().into_pyarray_bound(py).unbind());
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be a 2d or 3d"))
    }
}

#[pyfunction]
pub fn apply_blur<'py>(py: Python<'py>, img: PyReadonlyArrayDyn<'py, u8>, ksize_w: usize, ksize_h: usize) -> PyResult<Py<PyArrayDyn<u8>>> {
    let area = (ksize_w * ksize_h) as f64;
    let kernel = numpy::ndarray::Array2::<f64>::from_elem((ksize_h, ksize_w), 1.0 / area);
    apply_filter2d(py, img, kernel.into_dyn().into_pyarray_bound(py).readonly())
}

#[pyfunction]
pub fn apply_gaussian_blur<'py>(py: Python<'py>, img: PyReadonlyArrayDyn<'py, u8>, ksize: usize, sigma: f64) -> PyResult<Py<PyArrayDyn<u8>>> {
    let mut kernel = numpy::ndarray::Array2::<f64>::zeros((ksize, ksize));
    let center  = (ksize / 2) as f64;
    let mut sum = 0.0;
    let s2 = 2.0 * sigma * sigma;
    for y in 0..ksize {
        for x in 0..ksize {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            let val = (-(dx * dx + dy * dy) / s2).exp();
            kernel[[y , x]] = val;
            sum += val;
        }
    }
    kernel.mapv_inplace(|v| v / sum);
    apply_filter2d(py, img, kernel.into_dyn().into_pyarray_bound(py).readonly())
}

[#pyfunction]
pub fn apply_median_blur<'py>(py:Python<'py>, img: PyReadonlyArrayDyn<'py, u8>, ksize: usize) -> PyResult<Py<PyArrayDyn<u8>>>{
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    let pad = ksize/2;

    let process_channel = |channel: numpy::ndarray::ArrayView2<u8>|-> numpy::ndarray::Array2<u8> {
        let(h, w) = (channel.shape()[0], channel.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        let mut window = Vec::with_capacity(ksize * ksize);

        for y in 0..h{
            for x in 0..w{
                window.clear();
                for ky in 0..ksize{
                    for kx in 0..ksize{
                        let iy = (u as isize + ky as isize - pad as isize).clamp(0, h as isize - 1) as usize;
                        let ix = (x as isize + kx as isize - pad as isize).clamp(0, w as isize - 1) as usize;
                        window.push(channel[[iy,ix]])
                    }
                }
                window.sort_unstable();
                out[[y,x]] = window[window.len()/2];

            }
            out

        };
        if ndim ==3 {
            let (h,w,c) = (shape[0], shape[1], shape[2]);
            let mut out_arr = numpy::ndarray::Array3::<u8>::zeros((h, w, c));
            for ch in 0..c {
                out_arr.slice_mut(s![.., .., ch]).assign(&process_channel(arr.slice(s![.., .., ch])));
            }
            Ok(out_arr.into_dyn().into_pyarray_bound(py).unbind())
        } else if ndim == 2{
            let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
            Ok(process_channel(channel.view()).into_dyn().into_pyarray_bound(py).unbind())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Image must be a 2d or 3d"))
        }
    }
}
