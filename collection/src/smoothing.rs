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

#[pyfunction]
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
                        let iy = (y as isize + ky as isize - pad as isize).clamp(0, h as isize - 1) as usize;
                        let ix = (x as isize + kx as isize - pad as isize).clamp(0, w as isize - 1) as usize;
                        window.push(channel[[iy,ix]])
                    }
                }
                window.sort_unstable();
                out[[y,x]] = window[window.len()/2];

            }
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


#[pyfunction]
pub fn apply_bilateral_filter<'py>(py:Python<'py>, img: PyReadonlyArrayDyn<'py, u8>, diameter: usize, sigma_color:f64, sigma_space: f64) -> PyResult<Py<PyArrayDyn<u8>>>{
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    let pad = diameter/2;

    let mut space_weights = numpy::ndarray::Array2::<f64>::zeros((diameter, diameter));
    for y in 0..diameter {
        for x in 0..diameter {
            let dx = x as f64 - pad as f64;
            let dy = y as f64 - pad as f64;
            space_weights[[y,x]] = (-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space)).exp();
        }
    }
    let process_channel = |channel: numpy::ndarray::ArrayView2<u8>|-> numpy::ndarray::Array2<u8> {
        let(h, w) = (channel.shape()[0], channel.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        
        for y in 0..h {
            for x in 0..w {
                let center_val = channel[[y,x]] as f64;
                let mut sum_weight = 0.0;
                let mut sum_val = 0.0;

                for ky in 0..diameter{
                    for kx in 0..diameter{
                        let iy = (y as isize + ky as isize - pad as isize).clamp(0, h as isize - 1) as usize;
                        let ix = (x as isize + kx as isize - pad as isize).clamp(0, w as isize - 1) as usize;
                        
                        let neighbor_val = channel[[iy,ix]] as f64;
                        let color_diff = neighbor_val - center_val;
                        let color_weight = (-(color_diff * color_diff) / (2.0 * sigma_color * sigma_color)).exp();

                        let total_weight = space_weights[[ky,kx]]*color_weight;
                        sum_val += neighbor_val * total_weight;
                        sum_weight += total_weight;
                    }
                }
                out[[y,x]] = (sum_val / sum_weight).round().clamp(0.0, 255.0) as u8;
                
            }
        }
        out
    };
    if ndim ==3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
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


