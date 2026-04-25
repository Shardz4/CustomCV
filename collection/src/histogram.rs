use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3, ArrayView1, ArrayView2},
    IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
};

use crate::helpers::calculate_otsu_threshold;

// ==========================================
// INTERNAL HISTOGRAM HELPERS
// ==========================================

fn get_equalized_map(channel_data: ArrayView2<u8>) -> Vec<u8> {
    let mut hist = [0u32; 256];
    let mut count = 0;

    for &pixel in channel_data {
        hist[pixel as usize] += 1;
        count += 1;
    }

    let mut cdf = [0u32; 256];
    let mut sum = 0;
    for i in 0..256 {
        sum += hist[i];
        cdf[i] = sum;
    }

    let mut lut = vec![0u8; 256];
    let min_cdf = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0);
    
    for i in 0..256 {
        let num = (cdf[i] - min_cdf) as f32;
        let den = (count as u32 - min_cdf) as f32;
        
        if den > 0.0 {
            lut[i] = ((num / den) * 255.0).round() as u8;
        } else {
            lut[i] = i as u8; 
        }
    }
    lut
}

fn get_cdf(channel_data: ArrayView2<u8>) -> Vec<f32> {
    let mut hist = [0u32; 256];
    let mut count = 0;

    for &pixel in channel_data{
        hist[pixel as usize] += 1;
        count += 1;
    }

    let mut cdf = vec![0.0; 256];
    let mut sum = 0;
    for i in 0..256 {
        sum += hist[i];
        cdf[i] = sum as f32 / count as f32;
    }
    cdf
}

fn match_histograms(src_cdf: &[f32], tgt_cdf: &[f32]) -> Vec<u8> {
    let mut lut = vec![0u8; 256];
    let mut j: usize = 0;

    for i in 0..256 {
        let src_val = src_cdf[i];
        while j < 256 && tgt_cdf[j] < src_val {
            j += 1;
        }
        lut[i] = if j < 256 { j as u8 } else { 255 };
    }
    lut
}

fn get_target_cdf(target_pdf: ArrayView1<f32>) -> PyResult<Vec<f32>> {
    let mut cleaned = vec![0.0f32; 256];
    for i in 0..256 {
        let v = target_pdf[i];
        cleaned[i] = if v.is_finite() && v > 0.0 { v } else { 0.0 };
    }

    let total: f32 = cleaned.iter().sum();
    if total <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_hist must contain positive finite values",
        ));
    }

    let mut cdf = vec![0.0; 256];
    let mut sum = 0.0;
    for i in 0..256 {
        sum += cleaned[i] / total;
        cdf[i] = sum;
    }
    cdf[255] = 1.0;
    Ok(cdf)
}

// ==========================================
// EXPORTED HISTOGRAM FUNCTIONS
// ==========================================

#[pyfunction]
pub fn hist_equalize_rgb<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let shape = arr.shape();
    let h = shape[0];
    let w = shape[1];
    let mut out_arr = Array3::<u8>::zeros((h, w, 3));

    for c in 0..3 {
        let channel_data = arr.slice(s![.., .., c]);
        let lut = get_equalized_map(channel_data);
        let mut out_channel = out_arr.slice_mut(s![.., .., c]);
        for ((i, j), &pixel_val) in channel_data.indexed_iter() {
            out_channel[[i, j]] = lut[pixel_val as usize];
        }
    }
    Ok(out_arr.into_pyarray(py).to_dyn())
}

#[pyfunction]
pub fn hist_equalize_gray<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let r = arr.slice(s![.., .., 0]);
    let g = arr.slice(s![.., .., 1]);
    let b = arr.slice(s![.., .., 2]);
    let gray_arr = (&r.mapv(|v| v as u16) + &g.mapv(|v| v as u16) + &b.mapv(|v| v as u16)) / 3;
    let gray_arr = gray_arr.mapv(|v| v as u8);
    let lut = get_equalized_map(gray_arr.view());
    let result = gray_arr.mapv(|pixel| lut[pixel as usize]);
    Ok(result.into_pyarray(py).to_dyn())
}


#[pyfunction]
pub fn hist_spec_rgb<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>, target_hist: PyReadonlyArrayDyn<'py, f32>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = x.as_array();
    let tgt = target_hist.as_array();
    let tgt_1d = tgt.view().into_dimensionality().map_err(|_| pyo3::exceptions::PyValueError::new_err("target_hist must be 1D with 256 bins"))?;
    if tgt_1d.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("target_hist length must be 256"));
    }
    let tgt_cdf = get_target_cdf(tgt_1d)?;
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have shape (H, W, 3)"));
    }

    let mut out_arr = Array3::<u8>::zeros((shape[0], shape[1], 3));
    for c in 0..3 {
        let channel_data = arr.slice(s![.., .., c]);
        let src_cdf = get_cdf(channel_data);
        let lut = match_histograms(&src_cdf, &tgt_cdf);
        let mut out_channel = out_arr.slice_mut(s![.., .., c]);
        for ((i, j), &pixel_val) in channel_data.indexed_iter() {
            out_channel[[i, j]] = lut[pixel_val as usize];
        }
    }
    Ok(out_arr.into_pyarray(py).to_dyn().to_owned().into())
}

#[pyfunction]
pub fn hist_spec_gray<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>, target_hist: PyReadonlyArrayDyn<'py, f32>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = x.as_array();
    let tgt = target_hist.as_array();
    let tgt_1d = tgt.view().into_dimensionality().map_err(|_| pyo3::exceptions::PyValueError::new_err("target_hist must be 1D with 256 bins"))?;
    if tgt_1d.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("target_hist length must be 256"));
    }
    let tgt_cdf = get_target_cdf(tgt_1d)?;
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have shape (H, W, 3)"));
    }

    let r = arr.slice(s![.., .., 0]);
    let g = arr.slice(s![.., .., 1]);
    let b = arr.slice(s![.., .., 2]);
    let gray_arr = (&r.mapv(|v| 77u16 * v as u16) + &g.mapv(|v| 150u16 * v as u16) + &b.mapv(|v| 29u16 * v as u16)) / 256;
    let gray_u8 = gray_arr.mapv(|v| v as u8);

    let src_cdf = get_cdf(gray_u8.view().into_dimensionality().unwrap());
    let lut = match_histograms(&src_cdf, &tgt_cdf);
    let result = gray_u8.mapv(|pixel| lut[pixel as usize]);
    Ok(result.into_pyarray(py).to_dyn().to_owned().into())
}

#[pyfunction]
pub fn apply_otsu_threshold<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<(u8, &'py PyArrayDyn<u8>)> {
    let arr = x.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("GrayScale Required"))?;

    let threshold_value = calculate_otsu_threshold(channel.view());
    let result = channel.mapv(|pixel| {
        if pixel > threshold_value { 255 } else { 0 }
    });

    // Return both the value and the array
    Ok((threshold_value, result.into_pyarray(py).to_dyn()))
}
