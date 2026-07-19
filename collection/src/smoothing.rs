use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use numpy::ndarray::s;
use crate::helpers;

/// apply_filter2d() - Convolve an image with a 2D float kernel.
/// @py: Python interpreter token.
/// @img: Grayscale (2D) or Color (3D) u8 input image.
/// @kernel: 2D float convolution kernel.
/// @border_type: Border padding mode ("reflect", "replicate", "wrap", "constant").
/// @border_value: Constant border value (defaults to 0).
///
/// Convolves the input image with the specified kernel.
///
/// Return: Filtered image array.
#[pyfunction]
#[pyo3(signature = (img, kernel, border_type = "reflect", border_value = 0))]
pub fn apply_filter2d<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    border_type: &str,
    border_value: u8,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = img.as_array();
    let k_arr = kernel.as_array();
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2d"))?;
    
    let ndim = arr.ndim();
    let shape = arr.shape();

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((h, w, c));

        for ch in 0..c {
            let filtered = helpers::convolve_2d_channel(arr.slice(s![.., .., ch]), k_2d.view(), border_type, border_value);
            out.slice_mut(s![.., .., ch]).assign(&filtered);
        }
        Ok(out.into_dyn().into_pyarray_bound(py).unbind())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let filtered = helpers::convolve_2d_channel(channel.view(), k_2d.view(), border_type, border_value);
        return Ok(filtered.into_dyn().into_pyarray_bound(py).unbind());
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be a 2d or 3d"))
    }
}

/// apply_blur() - Blurs an image using a normalized box filter.
/// @py: Python interpreter token.
/// @img: Grayscale (2D) or Color (3D) u8 input image.
/// @ksize_w: Width of the blur kernel.
/// @ksize_h: Height of the blur kernel.
/// @border_type: Border padding mode ("reflect", "replicate", "wrap", "constant").
/// @border_value: Constant border value (defaults to 0).
///
/// Blurs the input image by convolving it with a normalized box filter kernel.
///
/// Return: Blurred image array.
#[pyfunction]
#[pyo3(signature = (img, ksize_w, ksize_h, border_type = "reflect", border_value = 0))]
pub fn apply_blur<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    ksize_w: usize,
    ksize_h: usize,
    border_type: &str,
    border_value: u8,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let area = (ksize_w * ksize_h) as f64;
    let kernel = numpy::ndarray::Array2::<f64>::from_elem((ksize_h, ksize_w), 1.0 / area);
    apply_filter2d(py, img, kernel.into_dyn().into_pyarray_bound(py).readonly(), border_type, border_value)
}

/// apply_gaussian_blur() - Blurs an image using a Gaussian filter.
/// @py: Python interpreter token.
/// @img: Grayscale (2D) or Color (3D) u8 input image.
/// @ksize: Gaussian kernel size (ksize x ksize).
/// @sigma: Gaussian kernel standard deviation.
/// @border_type: Border padding mode ("reflect", "replicate", "wrap", "constant").
/// @border_value: Constant border value (defaults to 0).
///
/// Blurs the input image by convolving it with a Gaussian kernel.
///
/// Return: Blurred image array.
#[pyfunction]
#[pyo3(signature = (img, ksize, sigma, border_type = "reflect", border_value = 0))]
pub fn apply_gaussian_blur<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    ksize: usize,
    sigma: f64,
    border_type: &str,
    border_value: u8,
) -> PyResult<Py<PyArrayDyn<u8>>> {
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
    apply_filter2d(py, img, kernel.into_dyn().into_pyarray_bound(py).readonly(), border_type, border_value)
}

/// apply_median_blur() - Blurs an image using a median filter.
/// @py: Python interpreter token.
/// @img: Grayscale (2D) or Color (3D) u8 input image.
/// @ksize: Aperture linear size; must be odd and greater than 1.
/// @border_type: Border padding mode ("reflect", "replicate", "wrap", "constant").
/// @border_value: Constant border value (defaults to 0).
///
/// Smooths the image by replacing each pixel with the median of its neighborhood.
///
/// Return: Blurred image array.
#[pyfunction]
#[pyo3(signature = (img, ksize, border_type = "reflect", border_value = 0))]
pub fn apply_median_blur<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    ksize: usize,
    border_type: &str,
    border_value: u8,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    let pad = ksize / 2;

    let process_channel = |channel: numpy::ndarray::ArrayView2<u8>| -> numpy::ndarray::Array2<u8> {
        let (h, w) = (channel.shape()[0], channel.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        let mut window = Vec::with_capacity(ksize * ksize);

        for y in 0..h {
            for x in 0..w {
                window.clear();
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        let iy = y as isize + ky as isize - pad as isize;
                        let ix = x as isize + kx as isize - pad as isize;
                        let val = helpers::get_border_pixel(&channel, iy, ix, border_type, border_value);
                        window.push(val);
                    }
                }
                window.sort_unstable();
                out[[y, x]] = window[window.len() / 2];
            }
        }
        out
    };

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out_arr = numpy::ndarray::Array3::<u8>::zeros((h, w, c));
        for ch in 0..c {
            out_arr.slice_mut(s![.., .., ch]).assign(&process_channel(arr.slice(s![.., .., ch])));
        }
        Ok(out_arr.into_dyn().into_pyarray_bound(py).unbind())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        Ok(process_channel(channel.view()).into_dyn().into_pyarray_bound(py).unbind())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be a 2d or 3d"))
    }
}

/// apply_bilateral_filter() - Applies a bilateral filter to an image.
/// @py: Python interpreter token.
/// @img: Grayscale (2D) or Color (3D) u8 input image.
/// @diameter: Diameter of each pixel neighborhood.
/// @sigma_color: Filter sigma in the color space.
/// @sigma_space: Filter sigma in the coordinate space.
/// @border_type: Border padding mode ("reflect", "replicate", "wrap", "constant").
/// @border_value: Constant border value (defaults to 0).
///
/// Smooths the image while preserving edges by using a combination of domain and range filters.
///
/// Return: Filtered image array.
#[pyfunction]
#[pyo3(signature = (img, diameter, sigma_color, sigma_space, border_type = "reflect", border_value = 0))]
pub fn apply_bilateral_filter<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    diameter: usize,
    sigma_color: f64,
    sigma_space: f64,
    border_type: &str,
    border_value: u8,
) -> PyResult<Py<PyArrayDyn<u8>>>{
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    let pad = diameter / 2;

    let mut space_weights = numpy::ndarray::Array2::<f64>::zeros((diameter, diameter));
    for y in 0..diameter {
        for x in 0..diameter {
            let dx = x as f64 - pad as f64;
            let dy = y as f64 - pad as f64;
            space_weights[[y, x]] = (-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space)).exp();
        }
    }

    let process_channel = |channel: numpy::ndarray::ArrayView2<u8>| -> numpy::ndarray::Array2<u8> {
        let (h, w) = (channel.shape()[0], channel.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        
        for y in 0..h {
            for x in 0..w {
                let center_val = channel[[y, x]] as f64;
                let mut sum_weight = 0.0;
                let mut sum_val = 0.0;

                for ky in 0..diameter {
                    for kx in 0..diameter {
                        let iy = y as isize + ky as isize - pad as isize;
                        let ix = x as isize + kx as isize - pad as isize;
                        let neighbor_val = helpers::get_border_pixel(&channel, iy, ix, border_type, border_value) as f64;
                        let color_diff = neighbor_val - center_val;
                        let color_weight = (-(color_diff * color_diff) / (2.0 * sigma_color * sigma_color)).exp();

                        let total_weight = space_weights[[ky, kx]] * color_weight;
                        sum_val += neighbor_val * total_weight;
                        sum_weight += total_weight;
                    }
                }
                out[[y, x]] = (sum_val / sum_weight).round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    };

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out_arr = numpy::ndarray::Array3::<u8>::zeros((h, w, c));
        for ch in 0..c {
            out_arr.slice_mut(s![.., .., ch]).assign(&process_channel(arr.slice(s![.., .., ch])));
        }
        Ok(out_arr.into_dyn().into_pyarray_bound(py).unbind())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        Ok(process_channel(channel.view()).into_dyn().into_pyarray_bound(py).unbind())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be a 2d or 3d"))
    }
}


