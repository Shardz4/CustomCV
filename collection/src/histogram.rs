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

/// Compute a 1D histogram of a single channel from an image.
///
/// - `image`: Grayscale (2D) or Color (3D) u8 image.
/// - `channel_idx`: index of the channel (0 for grayscale).
/// - `hist_size`: number of bins (e.g. 256).
/// - `ranges`: (low, high) float bounds (e.g. (0.0, 256.0)).
/// Returns a float32 array of shape (hist_size, 1) containing bin counts.
#[pyfunction]
#[pyo3(signature = (image, channel_idx, hist_size, ranges))]
pub fn calc_hist<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    channel_idx: usize,
    hist_size: usize,
    ranges: (f32, f32),
) -> PyResult<&'py PyArrayDyn<f32>> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    let channel = if ndim == 2 {
        if channel_idx != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Grayscale image only has channel 0"));
        }
        arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap()
    } else if ndim == 3 {
        if channel_idx >= shape[2] {
            return Err(pyo3::exceptions::PyValueError::new_err("Channel index out of bounds"));
        }
        arr.slice(s![.., .., channel_idx]).into_dimensionality::<numpy::ndarray::Ix2>().unwrap()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"));
    };
    
    let mut hist = ndarray::Array2::<f32>::zeros((hist_size, 1));
    let (low, high) = ranges;
    let range_width = high - low;
    if range_width <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid range"));
    }
    
    for &val in &channel {
        let val_f = val as f32;
        if val_f >= low && val_f < high {
            let bin = (((val_f - low) / range_width) * hist_size as f32).floor() as usize;
            let bin = bin.min(hist_size - 1);
            hist[[bin, 0]] += 1.0;
        }
    }
    
    Ok(hist.into_pyarray(py).to_dyn())
}

/// Compare two histograms.
///
/// - `h1`: First histogram array.
/// - `h2`: Second histogram array (must have the same shape).
/// - `method`: comparison method:
///            0 = Correlation (HISTCMP_CORREL)
///            1 = Chi-Square (HISTCMP_CHISQR)
///            2 = Intersection (HISTCMP_INTERSECT)
///            3 = Bhattacharyya (HISTCMP_BHATTACHARYYA)
#[pyfunction]
pub fn compare_hist(
    h1: PyReadonlyArrayDyn<f32>,
    h2: PyReadonlyArrayDyn<f32>,
    method: i32,
) -> PyResult<f64> {
    let arr1 = h1.as_array();
    let arr2 = h2.as_array();
    
    if arr1.shape() != arr2.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("Histograms must have the same shape"));
    }
    
    let n = arr1.len();
    if n == 0 {
        return Ok(0.0);
    }
    
    let h1_flat: Vec<f64> = arr1.iter().map(|&v| v as f64).collect();
    let h2_flat: Vec<f64> = arr2.iter().map(|&v| v as f64).collect();
    
    match method {
        0 => {
            let mean1 = h1_flat.iter().sum::<f64>() / n as f64;
            let mean2 = h2_flat.iter().sum::<f64>() / n as f64;
            
            let mut num = 0.0;
            let mut den1 = 0.0;
            let mut den2 = 0.0;
            
            for i in 0..n {
                let diff1 = h1_flat[i] - mean1;
                let diff2 = h2_flat[i] - mean2;
                num += diff1 * diff2;
                den1 += diff1 * diff1;
                den2 += diff2 * diff2;
            }
            
            if den1 > 0.0 && den2 > 0.0 {
                Ok(num / (den1 * den2).sqrt())
            } else {
                Ok(0.0)
            }
        }
        1 => {
            let mut sum = 0.0;
            for i in 0..n {
                if h1_flat[i] > 1e-9 {
                    sum += (h1_flat[i] - h2_flat[i]).powi(2) / h1_flat[i];
                }
            }
            Ok(sum)
        }
        2 => {
            let mut sum = 0.0;
            for i in 0..n {
                sum += h1_flat[i].min(h2_flat[i]);
            }
            Ok(sum)
        }
        3 => {
            let sum1: f64 = h1_flat.iter().sum();
            let sum2: f64 = h2_flat.iter().sum();
            
            if sum1 <= 0.0 || sum2 <= 0.0 {
                return Ok(1.0);
            }
            
            let mut num = 0.0;
            for i in 0..n {
                num += (h1_flat[i] * h2_flat[i]).sqrt();
            }
            
            let term = num / (sum1 * sum2).sqrt();
            let val = 1.0 - term;
            Ok(val.max(0.0).sqrt())
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err("Unknown comparison method. Must be 0, 1, 2, or 3.")),
    }
}

/// Slide a template across a 2D grayscale image and score matches.
///
/// Returns a float32 scoring map of shape (H - th + 1, W - tw + 1).
/// - `method`: template matching method:
///            0 = SQDIFF
///            1 = SQDIFF_NORMED
///            2 = CCORR
///            3 = CCORR_NORMED
///            4 = CCOEFF
///            5 = CCOEFF_NORMED
#[pyfunction]
#[pyo3(signature = (image, templ, method = 0))]
pub fn match_template<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    templ: PyReadonlyArrayDyn<'py, u8>,
    method: i32,
) -> PyResult<&'py PyArrayDyn<f32>> {
    let img_arr = image.as_array();
    let t_arr = templ.as_array();
    
    let img = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("image must be 2D grayscale"))?;
    let t = t_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("templ must be 2D grayscale"))?;
        
    let (ih, iw) = (img.shape()[0], img.shape()[1]);
    let (th, tw) = (t.shape()[0], t.shape()[1]);
    
    if th > ih || tw > iw {
        return Err(pyo3::exceptions::PyValueError::new_err("Template size must be smaller than or equal to image size"));
    }
    
    let rh = ih - th + 1;
    let rw = iw - tw + 1;
    let mut result = ndarray::Array2::<f32>::zeros((rh, rw));
    
    let t_sum: f64 = t.iter().map(|&v| v as f64).sum();
    let t_len = (th * tw) as f64;
    let t_mean = t_sum / t_len;
    
    let mut t_var = 0.0;
    for &val in t.iter() {
        let diff = val as f64 - t_mean;
        t_var += diff * diff;
    }
    
    for y in 0..rh {
        for x in 0..rw {
            let mut score = 0.0;
            
            match method {
                0 | 1 => {
                    let mut sum_sq = 0.0;
                    let mut sum_i_sq = 0.0;
                    for ty in 0..th {
                        for tx in 0..tw {
                            let iv = img[[y + ty, x + tx]] as f64;
                            let tv = t[[ty, tx]] as f64;
                            let diff = tv - iv;
                            sum_sq += diff * diff;
                            sum_i_sq += iv * iv;
                        }
                    }
                    if method == 0 {
                        score = sum_sq;
                    } else {
                        let sum_t_sq: f64 = t.iter().map(|&v| (v as f64).powi(2)).sum();
                        let den = (sum_t_sq * sum_i_sq).sqrt();
                        if den > 0.0 {
                            score = sum_sq / den;
                        } else {
                            score = 1.0;
                        }
                    }
                }
                2 | 3 => {
                    let mut sum_mul = 0.0;
                    let mut sum_i_sq = 0.0;
                    for ty in 0..th {
                        for tx in 0..tw {
                            let iv = img[[y + ty, x + tx]] as f64;
                            let tv = t[[ty, tx]] as f64;
                            sum_mul += iv * tv;
                            sum_i_sq += iv * iv;
                        }
                    }
                    if method == 2 {
                        score = sum_mul;
                    } else {
                        let sum_t_sq: f64 = t.iter().map(|&v| (v as f64).powi(2)).sum();
                        let den = (sum_t_sq * sum_i_sq).sqrt();
                        if den > 0.0 {
                            score = sum_mul / den;
                        } else {
                            score = 0.0;
                        }
                    }
                }
                4 | 5 => {
                    let mut sum_i = 0.0;
                    for ty in 0..th {
                        for tx in 0..tw {
                            sum_i += img[[y + ty, x + tx]] as f64;
                        }
                    }
                    let i_mean = sum_i / t_len;
                    
                    let mut num = 0.0;
                    let mut i_var = 0.0;
                    for ty in 0..th {
                        for tx in 0..tw {
                            let iv = img[[y + ty, x + tx]] as f64;
                            let tv = t[[ty, tx]] as f64;
                            let diff_i = iv - i_mean;
                            let diff_t = tv - t_mean;
                            num += diff_i * diff_t;
                            i_var += diff_i * diff_i;
                        }
                    }
                    
                    if method == 4 {
                        score = num;
                    } else {
                        let den = (i_var * t_var).sqrt();
                        if den > 0.0 {
                            score = num / den;
                        } else {
                            score = 0.0;
                        }
                    }
                }
                _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown template matching method. Must be 0 to 5.")),
            }
            
            result[[y, x]] = score as f32;
        }
    }
    
    Ok(result.into_pyarray(py).to_dyn())
}
